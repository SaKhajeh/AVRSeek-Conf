from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from app.config import Settings
from app.prompts.antiviral_reaction import SYSTEM_PROMPT, build_user_prompt
from app.schemas.reaction_route import EvidenceRef, ReactionStep, SynthesisRoute
from app.services.reconcile import reconcile_route


def get_openai_client(settings: Settings) -> OpenAI:
    if not settings.openai_api_key or settings.openai_api_key.startswith("YOUR_"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy backend/.env.example to backend/.env "
            "and add your AvalAI or GapGPT key."
        )
    return OpenAI(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        timeout=180.0,
        max_retries=2,
    )


def get_choice_text(response: Any) -> str:
    """Support both OpenAI and GapGPT-style response shapes."""
    try:
        choice = response.choices[0]
        msg = getattr(choice, "message", None)
        if msg is not None:
            content = getattr(msg, "content", None)
            if content:
                return content
        # GapGPT sometimes puts text on choice.content
        alt = getattr(choice, "content", None)
        if isinstance(alt, str) and alt:
            return alt
    except Exception:
        pass
    return ""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _close_truncated_json(text: str) -> str:
    """Best-effort close of truncated JSON strings/objects."""
    s = text.rstrip()
    if s.count('"') % 2 == 1:
        s += '"'
    open_braces = s.count("{") - s.count("}")
    open_brackets = s.count("[") - s.count("]")
    s += "]" * max(0, open_brackets)
    s += "}" * max(0, open_braces)
    return s


def parse_route_json(raw: str) -> SynthesisRoute:
    cleaned = _strip_fences(raw)
    if not cleaned.startswith("{"):
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            raise ValueError("Model did not return JSON")
        cleaned = match.group(0)
    try:
        data: dict[str, Any] = json.loads(cleaned)
    except json.JSONDecodeError:
        data = json.loads(_close_truncated_json(cleaned))
    return SynthesisRoute.model_validate(data)


def _split_text_chunks(text: str, max_chars: int = 28000) -> list[str]:
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    # Prefer example boundaries
    parts = re.split(
        r"(?=(?:^|\n)\s*(?:example|пример|ex\.?\s*|preparation|step)\s+[a-d0-9ivx]+)",
        text,
        flags=re.IGNORECASE,
    )
    parts = [p.strip() for p in parts if p and p.strip()]
    if len(parts) <= 1:
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
    chunks: list[str] = []
    buf = ""
    for part in parts:
        if len(buf) + len(part) + 2 <= max_chars:
            buf = f"{buf}\n\n{part}".strip()
        else:
            if buf:
                chunks.append(buf)
            if len(part) > max_chars:
                for i in range(0, len(part), max_chars):
                    chunks.append(part[i : i + max_chars])
                buf = ""
            else:
                buf = part
    if buf:
        chunks.append(buf)
    return chunks


def _call_extract(client: OpenAI, model: str, title: str, text: str) -> tuple[SynthesisRoute, str]:
    prompt = build_user_prompt(title=title, text=text)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=8000,
    )
    raw = get_choice_text(response) or ""
    try:
        route = parse_route_json(raw)
    except Exception:
        # One repair attempt
        repair = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Fix the following into valid JSON only matching the AVRSeek route schema."},
                {"role": "user", "content": raw[:120000]},
            ],
            temperature=0,
            max_tokens=8000,
        )
        raw2 = get_choice_text(repair) or raw
        route = parse_route_json(raw2)
        raw = raw2
    return route, raw


def _attach_chunk_provenance(route: SynthesisRoute, *, page_start: int, page_end: int, section: str, evidence_types: list[str]) -> None:
    etype = evidence_types[0] if evidence_types else "experimental_procedure"
    prov_source = "native_text"
    for step in route.steps:
        if not step.source:
            step.source = "text"
        if step.information_kind in ("", "unknown"):
            step.information_kind = "explicit" if etype in ("experimental_procedure", "patent_example") else "ambiguous"
        if "llm_text" not in (step.extraction_methods or []):
            step.extraction_methods.append("llm_text")
        ref = EvidenceRef(
            page=page_start,
            page_end=page_end,
            section=section,
            evidence_type=etype,
            source=prov_source,
            quote=(step.evidence or "")[:400],
            confidence=0.85 if etype in ("experimental_procedure", "patent_example") else 0.7,
            extraction_method="llm_text",
            information_kind=step.information_kind,
        )
        step.evidence_refs.append(ref)


def extract_reactions_from_text(
    *,
    settings: Settings,
    title: str,
    text: str,
    model: str | None = None,
    text_chunks: list[Any] | None = None,
) -> tuple[SynthesisRoute, str]:
    client = get_openai_client(settings)
    use_model = model or settings.model_text

    # Prefer pre-built evidence chunks from retrieval layer
    if text_chunks:
        chunk_texts = []
        chunk_meta: list[tuple[int, int, str, list[str]]] = []
        for c in text_chunks:
            header = getattr(c, "header", lambda: "")()
            body = getattr(c, "text", str(c))
            chunk_texts.append(f"{header}\n{body}" if header else body)
            chunk_meta.append(
                (
                    int(getattr(c, "page_start", 0) or 0),
                    int(getattr(c, "page_end", 0) or 0),
                    str(getattr(c, "section", "")),
                    list(getattr(c, "evidence_types", []) or []),
                )
            )
        sub_chunks = chunk_texts
    else:
        sub_chunks = _split_text_chunks(text or "", max_chars=28000)
        chunk_meta = [(0, 0, "", []) for _ in sub_chunks]

    if not sub_chunks:
        empty = SynthesisRoute(
            title=title,
            target_name="N/A",
            target_class="unknown",
            document_type="unknown",
            steps=[],
            warnings=["No text provided for extraction."],
        )
        return empty, "{}"

    routes: list[SynthesisRoute] = []
    raws: list[str] = []
    for i, chunk in enumerate(sub_chunks, start=1):
        chunk_title = title if len(sub_chunks) == 1 else f"{title} [part {i}/{len(sub_chunks)}]"
        route, raw = _call_extract(client, use_model, chunk_title, chunk)
        if i - 1 < len(chunk_meta):
            ps, pe, sec, etypes = chunk_meta[i - 1]
            if ps:
                _attach_chunk_provenance(route, page_start=ps, page_end=pe or ps, section=sec, evidence_types=etypes)
        routes.append(route)
        raws.append(raw)

    if len(routes) == 1:
        return routes[0], raws[0]

    all_steps: list[ReactionStep] = []
    warnings: list[str] = [f"Merged {len(routes)} evidence chunks from text extraction."]
    meta = routes[0]
    for r in routes:
        all_steps.extend(r.steps)
        warnings.extend(r.warnings or [])
        if (not meta.target_name or meta.target_name == "N/A") and r.target_name not in ("", "N/A"):
            meta = r
    merged = reconcile_route(
        text_steps=all_steps,
        title=meta.title or title,
        target_name=meta.target_name,
        target_class=meta.target_class,
        document_type=meta.document_type,
        warnings=warnings,
    )
    return merged, "\n\n---\n\n".join(raws)
