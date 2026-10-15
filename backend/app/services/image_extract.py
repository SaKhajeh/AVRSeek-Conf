from __future__ import annotations

"""Image / scheme extraction with classification and dual VLM modes.

Modes:
- scheme: extract SMILES + roles + conditions from reaction drawings
- page_read: OCR-like transcription of experimental text from a page image
  (used when native/OCR text is still empty)

Local OCSR (RxnScribe/MolScribe) is optional when installed.
"""

import asyncio
import base64
import io
import json
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from PIL import Image

from app.config import Settings
from app.prompts.vlm_scheme import VLM_SCHEME_PROMPT
from app.schemas.reaction_route import ChemicalEntity, EvidenceRef, PipelineStage, PipelineStageStatus, ReactionConditions, ReactionStep
from app.services.llm_client import get_choice_text, get_openai_client
from app.services.ocr import render_page_image


class ExtractedMolecule:
    def __init__(
        self,
        smiles: str,
        bbox: tuple[float, float, float, float] | None = None,
        role: str | None = None,
        confidence: float = 0.0,
        source: str = "unknown",
        name: str | None = None,
    ):
        self.smiles = smiles
        self.bbox = bbox
        self.role = role
        self.confidence = confidence
        self.source = source
        self.name = name

    def to_dict(self) -> dict[str, Any]:
        return {
            "smiles": self.smiles,
            "bbox": self.bbox,
            "role": self.role,
            "confidence": self.confidence,
            "source": self.source,
            "name": self.name,
        }


class ExtractedScheme:
    def __init__(
        self,
        page_number: int,
        image_index: int,
        molecules: list[ExtractedMolecule] | None = None,
        arrows: list[dict] | None = None,
        conditions_text: str = "",
        raw_output: str = "",
        mode: str = "scheme",
        transcribed_text: str = "",
        image_class: str = "unknown",
        step_label: str = "",
        characterizations: list[dict] | None = None,
        vlm_steps: list[dict] | None = None,
    ):
        self.page_number = page_number
        self.image_index = image_index
        self.molecules = molecules or []
        self.arrows = arrows or []
        self.conditions_text = conditions_text
        self.raw_output = raw_output
        self.mode = mode
        self.transcribed_text = transcribed_text
        self.image_class = image_class
        self.step_label = step_label
        self.characterizations = characterizations or []
        self.vlm_steps = vlm_steps or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "image_index": self.image_index,
            "molecules": [m.to_dict() for m in self.molecules],
            "arrows": self.arrows,
            "conditions_text": self.conditions_text,
            "raw_output": self.raw_output[:2000],
            "mode": self.mode,
            "transcribed_text": self.transcribed_text[:4000],
            "image_class": self.image_class,
            "step_label": self.step_label,
            "characterizations": self.characterizations,
        }


def try_import_rxnscribe():
    try:
        from rxnscribe import RxnScribe

        return RxnScribe
    except ImportError:
        return None


def try_import_molscribe():
    try:
        from molscribe import MolScribe

        return MolScribe
    except ImportError:
        return None


def extract_with_local_ocsr(image: Image.Image, model_name: str = "rxnscribe") -> ExtractedScheme | None:
    if model_name.lower() == "rxnscribe":
        RxnScribe = try_import_rxnscribe()
        if RxnScribe is None:
            return None
        model = RxnScribe()
        result = model.predict_image(image)
        molecules = []
        for mol in result.get("molecules", []):
            molecules.append(
                ExtractedMolecule(
                    smiles=mol.get("smiles", ""),
                    bbox=mol.get("bbox"),
                    role=mol.get("role"),
                    confidence=mol.get("confidence", 0.0),
                    source="rxnscribe",
                    name=mol.get("name"),
                )
            )
        return ExtractedScheme(
            page_number=0,
            image_index=0,
            molecules=molecules,
            arrows=result.get("arrows", []),
            conditions_text=result.get("conditions", ""),
            raw_output=json.dumps(result),
            mode="scheme",
            image_class="scheme",
        )
    if model_name.lower() == "molscribe":
        MolScribe = try_import_molscribe()
        if MolScribe is None:
            return None
        model = MolScribe()
        result = model.predict_image(image)
        molecules = []
        for mol in result.get("molecules", []):
            molecules.append(
                ExtractedMolecule(
                    smiles=mol.get("smiles", ""),
                    bbox=mol.get("bbox"),
                    role=mol.get("role"),
                    confidence=mol.get("confidence", 0.0),
                    source="molscribe",
                    name=mol.get("name"),
                )
            )
        return ExtractedScheme(
            page_number=0,
            image_index=0,
            molecules=molecules,
            arrows=result.get("arrows", []),
            conditions_text="",
            raw_output=json.dumps(result),
            mode="scheme",
            image_class="scheme",
        )
    return None


VLM_PAGE_READ_PROMPT = """You are reading a scanned patent or paper page image.
Transcribe experimental / example procedure text accurately (any language).
Also extract structured reaction steps if clearly present.

Return ONLY JSON:
{
  "image_class": "text_page | scheme | mixed | other",
  "transcribed_text": "full experimental text visible on this page",
  "steps": [
    {
      "step_label": "Example / Step id",
      "reactants": [{"name": "...", "role": "reactant", "amount": "..."}],
      "reagents": [{"name": "...", "role": "reagent", "amount": "..."}],
      "catalysts": [],
      "ligands": [],
      "solvents": [{"name": "...", "role": "solvent", "amount": "..."}],
      "products": [{"name": "...", "role": "product", "amount": "..."}],
      "intermediates": [],
      "conditions": {
        "temperature": "...",
        "time": "...",
        "pressure": "N/A",
        "atmosphere": "N/A",
        "other": "..."
      },
      "yield_value": "...",
      "ee_or_dr": "N/A",
      "notes": "",
      "evidence": "short quote"
    }
  ]
}

Rules:
- Prefer concrete Examples over Claims.
- Do not invent amounts/conditions; use N/A if missing.
- If the page has no experimental text, return empty steps and short transcribed_text.
"""


def classify_image_heuristic(image: Image.Image) -> str:
    """Cheap local heuristic before calling the VLM."""
    w, h = image.size
    pixels = w * h
    if pixels < 80_000 or min(w, h) < 80:
        return "logo"
    # Very wide thin strips → likely headers
    if w > 4 * h and h < 120:
        return "logo"
    # Large page-like images → text_page candidate (scanned page)
    if pixels > 1_200_000:
        return "page_candidate"
    return "scheme_candidate"


def _image_to_b64(image: Image.Image, max_side: int = 1600) -> str:
    img = image.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_json_content(raw: str) -> dict[str, Any]:
    cleaned = (raw or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    if not cleaned.startswith("{"):
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            cleaned = match.group(0)
    return json.loads(cleaned)


def _molecule_from_vlm_dict(mol: dict[str, Any]) -> ExtractedMolecule | None:
    smiles = (mol.get("smiles") or "").strip()
    if smiles.lower() in ("null", "none"):
        smiles = ""
    name = mol.get("name") or mol.get("label")
    if isinstance(name, str):
        name = name.strip() or None
    else:
        name = None
    if not smiles and not name:
        return None
    bbox = mol.get("bbox")
    if isinstance(bbox, list):
        bbox = tuple(bbox) if len(bbox) == 4 else None
    return ExtractedMolecule(
        smiles=smiles,
        bbox=bbox,
        role=mol.get("role", "unknown"),
        confidence=float(mol.get("confidence") or 0.5),
        source="vlm",
        name=name,
    )


def _scheme_from_vlm_json(data: dict[str, Any], raw: str) -> ExtractedScheme:
    molecules: list[ExtractedMolecule] = []
    for mol in data.get("molecules") or []:
        parsed = _molecule_from_vlm_dict(mol)
        if parsed:
            molecules.append(parsed)
    vlm_steps = data.get("steps") if isinstance(data.get("steps"), list) else []
    for step in vlm_steps:
        for mol in step.get("molecules") or []:
            parsed = _molecule_from_vlm_dict(mol)
            if parsed:
                molecules.append(parsed)

    condition_parts: list[str] = []
    if data.get("conditions_text"):
        condition_parts.append(str(data["conditions_text"]))
    if data.get("yield_text") and str(data["yield_text"]) not in ("", "N/A"):
        condition_parts.append(f"yield {data['yield_text']}")
    for step in vlm_steps:
        ct = step.get("conditions_text")
        if ct:
            label = step.get("step_label") or ""
            condition_parts.append(f"{label}: {ct}".strip(": "))
    for legend in data.get("legend_conditions") or []:
        if isinstance(legend, dict) and legend.get("text"):
            condition_parts.append(f"({legend.get('roman') or ''}) {legend['text']}".strip())

    chars = data.get("characterizations") if isinstance(data.get("characterizations"), list) else []
    for char in chars:
        if not isinstance(char, dict):
            continue
        bits = [char.get("compound_id") or char.get("name") or ""]
        if char.get("yield_value"):
            bits.append(f"yield {char['yield_value']}")
        if char.get("lcms"):
            bits.append(f"LC/MS {char['lcms']}")
        if char.get("notes"):
            bits.append(str(char["notes"]))
        line = " ".join(str(b) for b in bits if b).strip()
        if line:
            condition_parts.append(line)

    image_class = str(data.get("image_class") or "")
    if data.get("is_reaction_scheme") is False:
        image_class = image_class or "other"
    elif not image_class:
        image_class = "scheme" if molecules else "text_page"

    return ExtractedScheme(
        page_number=0,
        image_index=0,
        molecules=molecules,
        arrows=data.get("arrows") or [],
        conditions_text="; ".join(p for p in condition_parts if p)[:2000],
        raw_output=raw,
        mode="scheme",
        image_class=image_class,
        step_label=str(data.get("step_label") or ""),
        characterizations=chars,
        vlm_steps=vlm_steps,
    )


def extract_with_vlm_scheme(
    image: Image.Image,
    settings: Settings,
    model: str | None = None,
    *,
    caption_context: str = "",
) -> ExtractedScheme:
    client = get_openai_client(settings)
    use_model = model or settings.model_vlm
    img_b64 = _image_to_b64(image)
    user_text = "Analyze this chemical image. Capture scheme, yields, and legend."
    if caption_context.strip():
        user_text += (
            "\n\nNearby figure caption / page text for context (use for compound numbering and conditions):\n"
            + caption_context.strip()[:2500]
        )
    try:
        resp = client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": VLM_SCHEME_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    ],
                },
            ],
            temperature=0,
            max_tokens=4000,
            timeout=180,
        )
        raw = get_choice_text(resp) or ""
        data = _parse_json_content(raw)
        return _scheme_from_vlm_json(data, raw)
    except Exception as e:
        return ExtractedScheme(
            page_number=0,
            image_index=0,
            molecules=[],
            raw_output=f"VLM scheme failed: {e}",
            mode="scheme",
            image_class="error",
        )


def extract_with_vlm_page_read(image: Image.Image, settings: Settings, model: str | None = None) -> ExtractedScheme:
    client = get_openai_client(settings)
    use_model = model or settings.model_vlm
    img_b64 = _image_to_b64(image, max_side=1800)
    try:
        resp = client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": VLM_PAGE_READ_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe experimental content from this page."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    ],
                },
            ],
            temperature=0,
            max_tokens=4000,
            timeout=180,
        )
        raw = get_choice_text(resp) or ""
        data = _parse_json_content(raw)
        return ExtractedScheme(
            page_number=0,
            image_index=0,
            molecules=[],
            conditions_text="",
            raw_output=raw,
            mode="page_read",
            transcribed_text=str(data.get("transcribed_text") or ""),
            image_class=str(data.get("image_class") or "text_page"),
        )
    except Exception as e:
        return ExtractedScheme(
            page_number=0,
            image_index=0,
            molecules=[],
            raw_output=f"VLM page_read failed: {e}",
            mode="page_read",
            image_class="error",
        )


def page_read_steps_from_raw(raw: str) -> list[dict[str, Any]]:
    try:
        data = _parse_json_content(raw)
        steps = data.get("steps") or []
        return steps if isinstance(steps, list) else []
    except Exception:
        return []


def extract_embedded_images(pdf_path: Path) -> list[tuple[int, int, Image.Image]]:
    doc = fitz.open(pdf_path)
    images: list[tuple[int, int, Image.Image]] = []
    for page_num, page in enumerate(doc, start=1):
        for img_idx, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                pil_img = Image.open(io.BytesIO(pix.tobytes("png")))
                images.append((page_num, img_idx, pil_img))
            except Exception:
                continue
    return images


def render_all_pages(pdf_path: Path, dpi: int = 150) -> list[tuple[int, int, Image.Image]]:
    doc = fitz.open(pdf_path)
    out: list[tuple[int, int, Image.Image]] = []
    for page_num, page in enumerate(doc, start=1):
        out.append((page_num, 0, render_page_image(page, dpi=dpi)))
    return out


FIGURE_CAPTION_RE = re.compile(
    r"\b(?:figure|fig\.?|scheme|chart)\s*\.?\s*(\d+)",
    re.IGNORECASE,
)

SYNTHESIS_FIGURE_RE = re.compile(
    r"\b(synthesis|pathway|route|reaction\s+scheme|preparation\s+of|"
    r"synthetic|transformations?|converted\s+to|intermediate)\b",
    re.IGNORECASE,
)

NON_SYNTHESIS_FIGURE_RE = re.compile(
    r"\b(ic50|ic\s*50|plot\s+for|determination\s+of|mechanism\s+of\s+action|"
    r"plaque|dose.?response|antiviral\s+activity|antiherp\w*|bioactivity|"
    r"correlated\s+coefficient|standard.{0,30}tested|activity\s+of\s+standard)\b",
    re.IGNORECASE,
)


def detect_figure_scheme_pages(
    pdf_path: Path,
    segments: list[Any] | None = None,
    *,
    min_drawings: int = 15,
    max_pages: int = 8,
) -> list[dict[str, Any]]:
    """
    Find pages likely containing vector reaction schemes (Figure 2-style pathways).
    Skips bioassay plots, IC50 charts, and mechanism-only figures.
    """
    doc = fitz.open(pdf_path)
    by_page_text: dict[int, str] = {}
    by_page_kind: dict[int, str] = {}
    if segments:
        for seg in segments:
            by_page_text[seg.page_start] = getattr(seg, "text", "") or ""
            by_page_kind[seg.page_start] = getattr(seg, "kind", "body")

    ranked: list[tuple[int, float, str]] = []
    for page_num, page in enumerate(doc, start=1):
        text = by_page_text.get(page_num, page.get_text("text") or "")
        drawings = len(page.get_drawings())
        score = 0.0
        reasons: list[str] = []

        caption_contexts: list[str] = []
        has_synthesis_caption = False
        for match in FIGURE_CAPTION_RE.finditer(text):
            ctx_start = max(0, match.start() - 10)
            ctx_end = min(len(text), match.end() + 160)
            caption = text[ctx_start:ctx_end]
            caption_contexts.append(caption)
            if NON_SYNTHESIS_FIGURE_RE.search(caption):
                continue
            if SYNTHESIS_FIGURE_RE.search(caption):
                score += 2.5
                reasons.append(f"caption:{match.group(0).strip()}")
                has_synthesis_caption = True

        page_has_synthesis = bool(SYNTHESIS_FIGURE_RE.search(text))
        is_procedure = by_page_kind.get(page_num) == "procedure_candidate"
        only_non_synthesis_figures = bool(caption_contexts) and not has_synthesis_caption and all(
            NON_SYNTHESIS_FIGURE_RE.search(c) for c in caption_contexts
        )
        if (
            drawings >= min_drawings
            and (page_has_synthesis or is_procedure or has_synthesis_caption)
            and not only_non_synthesis_figures
        ):
            score += 1.0 + min(1.5, drawings / 40.0)
            reasons.append(f"vector_drawings={drawings}")

        if score >= 2.0:
            ranked.append((page_num, score, "; ".join(reasons)))

    ranked.sort(key=lambda item: item[1], reverse=True)
    seen: set[int] = set()
    out: list[dict[str, Any]] = []
    for page_num, score, reason in ranked:
        if page_num in seen:
            continue
        seen.add(page_num)
        out.append({"page": page_num, "score": round(score, 2), "reason": reason})
        if len(out) >= max_pages:
            break
    return out


def figure_caption_context(segments: list[Any] | None, page_num: int) -> str:
    """Nearby figure caption + paragraph text for VLM context (P1)."""
    if not segments:
        return ""
    for seg in segments:
        if getattr(seg, "page_start", None) != page_num:
            continue
        text = (getattr(seg, "text", "") or "").strip()
        if not text:
            return ""
        chunks: list[str] = []
        for m in FIGURE_CAPTION_RE.finditer(text):
            start = max(0, m.start() - 20)
            end = min(len(text), m.end() + 400)
            chunks.append(text[start:end].strip())
        if chunks:
            return "\n\n".join(chunks)[:2000]
        return text[:1200]
    return ""


async def _vlm_scheme_from_page(
    pil_img: Image.Image,
    page_num: int,
    settings: Settings,
    vlm_model: str | None,
    local_model: str | None,
    *,
    caption_context: str = "",
) -> ExtractedScheme:
    scheme: ExtractedScheme | None = None
    if local_model:
        scheme = extract_with_local_ocsr(pil_img, local_model)
    if scheme is None or not scheme.molecules:
        scheme = await asyncio.to_thread(
            extract_with_vlm_scheme,
            pil_img,
            settings,
            vlm_model,
            caption_context=caption_context,
        )
    scheme.page_number = page_num
    scheme.image_index = 0
    scheme.mode = "scheme"
    if caption_context and caption_context not in (scheme.conditions_text or ""):
        scheme.conditions_text = (
            f"{scheme.conditions_text}; caption: {caption_context[:400]}"
            if scheme.conditions_text
            else f"caption: {caption_context[:600]}"
        )

    if scheme.molecules or scheme.vlm_steps or scheme.characterizations:
        return scheme
    try:
        data = _parse_json_content(scheme.raw_output or "")
        if data.get("is_reaction_scheme") is False:
            scheme.image_class = "not_scheme"
    except Exception:
        pass
    return scheme


def _entity_from_mol(mol: ExtractedMolecule) -> ChemicalEntity:
    name = (mol.name or "").strip() or mol.smiles or "unknown"
    role = mol.role or "unknown"
    return ChemicalEntity(name=name, role=role, amount="N/A", smiles=mol.smiles or None)


def _fill_entities_from_molecules(step: ReactionStep, molecules: list[ExtractedMolecule]) -> None:
    for mol in molecules:
        entity = _entity_from_mol(mol)
        role = (mol.role or "unknown").lower()
        bucket = {
            "reactant": "reactants",
            "reagent": "reagents",
            "catalyst": "catalysts",
            "ligand": "ligands",
            "solvent": "solvents",
            "product": "products",
            "intermediate": "intermediates",
        }.get(role)
        if bucket:
            getattr(step, bucket).append(entity)
        else:
            step.reactants.append(entity)


def _apply_characterizations(step: ReactionStep, characterizations: list[dict]) -> None:
    for char in characterizations or []:
        if not isinstance(char, dict):
            continue
        yield_val = char.get("yield_value")
        cid = str(char.get("compound_id") or char.get("name") or "").strip()
        if yield_val and step.yield_value in ("", "N/A"):
            step.yield_value = str(yield_val)
        if cid:
            matched = False
            for ent in step.products + step.intermediates:
                if cid.lower() in (ent.name or "").lower():
                    if yield_val and not ent.amount:
                        ent.amount = str(yield_val)
                    notes = char.get("notes") or ""
                    if notes:
                        ent.notes = notes
                    matched = True
            if not matched and (char.get("name") or cid):
                step.products.append(
                    ChemicalEntity(
                        name=str(char.get("name") or cid),
                        role="product",
                        amount=str(yield_val) if yield_val else "N/A",
                        notes=str(char.get("notes") or "") or None,
                    )
                )
        extra = []
        if char.get("lcms"):
            extra.append(f"LC/MS {char['lcms']}")
        if extra:
            step.notes = (step.notes + "; " if step.notes else "") + "; ".join(extra)


def _step_from_scheme_molecules(
    *,
    page_number: int,
    molecules: list[ExtractedMolecule],
    conditions_text: str,
    image_class: str,
    step_label: str,
    characterizations: list[dict] | None = None,
) -> ReactionStep | None:
    if not molecules and not conditions_text:
        return None
    step = ReactionStep(
        step_index=1,
        step_label=step_label or f"Scheme (page {page_number})",
        reactants=[],
        reagents=[],
        catalysts=[],
        ligands=[],
        solvents=[],
        intermediates=[],
        products=[],
        conditions=ReactionConditions(other=conditions_text or "N/A"),
        yield_value="N/A",
        ee_or_dr="N/A",
        notes=f"Extracted from image via VLM/OCSR ({image_class})",
        evidence=(conditions_text or "")[:500],
        source="scheme",
        confidence=0.7,
        evidence_refs=[
            EvidenceRef(
                page=page_number,
                section="figures",
                evidence_type="reaction_scheme",
                source="scheme",
                quote=(conditions_text or "")[:400],
                confidence=0.7,
                image_id=f"page{page_number}",
            )
        ],
    )
    m = re.search(r"(\d+(?:\.\d+)?\s*%)", conditions_text or "")
    if m:
        step.yield_value = m.group(1)
    _fill_entities_from_molecules(step, molecules)
    _apply_characterizations(step, characterizations or [])
    if not any(
        getattr(step, b)
        for b in ("reactants", "reagents", "catalysts", "products", "solvents", "intermediates")
    ):
        return None
    return step


def schemes_to_reaction_steps(schemes: list[ExtractedScheme]) -> list[ReactionStep]:
    """Convert scheme extractions into ReactionStep objects (full roles + conditions)."""
    steps: list[ReactionStep] = []
    for scheme in schemes:
        if scheme.mode == "page_read":
            continue
        if scheme.image_class in ("logo", "text_page") and not scheme.molecules:
            continue

        vlm_steps = [s for s in (scheme.vlm_steps or []) if isinstance(s, dict)]
        if vlm_steps:
            for vstep in vlm_steps:
                mols: list[ExtractedMolecule] = []
                for mol in vstep.get("molecules") or []:
                    parsed = _molecule_from_vlm_dict(mol)
                    if parsed:
                        mols.append(parsed)
                built = _step_from_scheme_molecules(
                    page_number=scheme.page_number,
                    molecules=mols,
                    conditions_text=str(vstep.get("conditions_text") or scheme.conditions_text or ""),
                    image_class=scheme.image_class,
                    step_label=str(vstep.get("step_label") or scheme.step_label or f"Scheme (page {scheme.page_number})"),
                    characterizations=scheme.characterizations,
                )
                if built:
                    steps.append(built)
            continue

        built = _step_from_scheme_molecules(
            page_number=scheme.page_number,
            molecules=scheme.molecules,
            conditions_text=scheme.conditions_text,
            image_class=scheme.image_class,
            step_label=scheme.step_label or f"Scheme (page {scheme.page_number})",
            characterizations=scheme.characterizations,
        )
        if built:
            steps.append(built)

    for idx, step in enumerate(steps, start=1):
        step.step_index = idx
    return steps


def page_read_to_reaction_steps(schemes: list[ExtractedScheme]) -> list[ReactionStep]:
    """Build ReactionSteps from page_read VLM JSON."""
    from app.services.llm_client import parse_route_json

    steps: list[ReactionStep] = []
    for scheme in schemes:
        if scheme.mode != "page_read" or not scheme.raw_output:
            continue
        try:
            # Wrap page steps into a mini route for reuse of entity parsing
            data = _parse_json_content(scheme.raw_output)
            raw_steps = data.get("steps") or []
            if not raw_steps:
                continue
            mini = {
                "title": f"Page {scheme.page_number}",
                "target_name": "N/A",
                "target_class": "unknown",
                "document_type": "patent",
                "steps": raw_steps,
                "warnings": [],
            }
            route = parse_route_json(json.dumps(mini))
            for st in route.steps:
                st.step_label = st.step_label or f"Page {scheme.page_number}"
                st.notes = (st.notes or "") + f" [page_read p.{scheme.page_number}]"
                st.source = st.source or "page_read"
                st.evidence_refs.append(
                    EvidenceRef(
                        page=scheme.page_number,
                        section="figures",
                        evidence_type="experimental_procedure",
                        source="page_read",
                        quote=(st.evidence or scheme.transcribed_text or "")[:400],
                        confidence=0.8,
                        image_id=f"page{scheme.page_number}",
                    )
                )
                steps.append(st)
        except Exception:
            continue
    for idx, step in enumerate(steps, start=1):
        step.step_index = idx
    return steps


async def extract_schemes_from_pdf(
    pdf_path: Path,
    settings: Settings,
    vlm_model: str | None = None,
    local_model: str | None = None,
    on_image_progress: Callable[[int, int, int], Awaitable[None]] | None = None,
    *,
    force_page_read: bool = False,
    text_char_count: int = 0,
    figure_pages: list[int] | None = None,
    segments: list[Any] | None = None,
) -> list[ExtractedScheme]:
    """
    Classify images and run scheme VLM on scheme-like images.
    If text is empty / force_page_read, also run page_read VLM on full page renders.
    For text-rich PDFs, figure_pages triggers targeted full-page scheme VLM (vector figures).
    """
    schemes: list[ExtractedScheme] = []
    processed_pages: set[int] = set()

    # Phase A — targeted figure pages (vector reaction schemes in native-text papers)
    use_figure_pages = (
        bool(figure_pages)
        and settings.enable_vlm_figure_pages
        and text_char_count >= 400
        and not force_page_read
    )
    if use_figure_pages:
        doc = fitz.open(pdf_path)
        total_fig = len(figure_pages or [])
        for idx, page_num in enumerate(figure_pages or [], start=1):
            if page_num in processed_pages or page_num < 1 or page_num > doc.page_count:
                continue
            processed_pages.add(page_num)
            if on_image_progress:
                await on_image_progress(idx, total_fig, page_num)
            pil_img = render_page_image(doc[page_num - 1], dpi=150)
            caption = figure_caption_context(segments, page_num)
            scheme = await _vlm_scheme_from_page(
                pil_img, page_num, settings, vlm_model, local_model, caption_context=caption
            )
            schemes.append(scheme)

    # Phase B — legacy path for scanned/empty-text PDFs or when no figure pages matched
    if force_page_read or text_char_count < 400 or not use_figure_pages:
        embedded = extract_embedded_images(pdf_path)
        use_pages = force_page_read or text_char_count < 400 or len(embedded) == 0
        if use_pages or (embedded and all(classify_image_heuristic(im) == "page_candidate" for _, _, im in embedded)):
            candidates = render_all_pages(pdf_path, dpi=150)
            source_kind = "page"
        else:
            candidates = embedded
            source_kind = "embed"

        filtered: list[tuple[int, int, Image.Image, str]] = []
        for page_num, img_idx, pil_img in candidates:
            if page_num in processed_pages:
                continue
            heur = classify_image_heuristic(pil_img)
            if heur == "logo":
                continue
            filtered.append((page_num, img_idx, pil_img, heur))

        total = len(filtered)
        for i, (page_num, img_idx, pil_img, heur) in enumerate(filtered, start=1):
            if on_image_progress:
                await on_image_progress(i, total, page_num)

            scanned_page = settings.enable_vlm_page_read and (
                force_page_read or text_char_count < 400 or heur == "page_candidate" or source_kind == "page"
            )
            scheme_like = heur == "scheme_candidate" or (source_kind == "embed" and heur != "page_candidate")

            if scanned_page and not scheme_like:
                page_scheme = await asyncio.to_thread(extract_with_vlm_page_read, pil_img, settings, vlm_model)
                page_scheme.page_number = page_num
                page_scheme.image_index = img_idx
                schemes.append(page_scheme)
                if page_scheme.image_class in ("scheme", "mixed"):
                    extra = None
                    if local_model:
                        extra = extract_with_local_ocsr(pil_img, local_model)
                    if extra is None or not extra.molecules:
                        extra = await asyncio.to_thread(extract_with_vlm_scheme, pil_img, settings, vlm_model)
                    extra.page_number = page_num
                    extra.image_index = img_idx
                    if extra.molecules or extra.conditions_text:
                        schemes.append(extra)
                continue

            scheme: ExtractedScheme | None = None
            if local_model:
                scheme = extract_with_local_ocsr(pil_img, local_model)
            if scheme is None or not scheme.molecules:
                scheme = await asyncio.to_thread(extract_with_vlm_scheme, pil_img, settings, vlm_model)
            scheme.page_number = page_num
            scheme.image_index = img_idx

            if scheme.image_class in ("text_page", "logo") and not scheme.molecules and scanned_page:
                page_scheme = await asyncio.to_thread(extract_with_vlm_page_read, pil_img, settings, vlm_model)
                page_scheme.page_number = page_num
                page_scheme.image_index = img_idx
                schemes.append(page_scheme)
            elif scheme.molecules or scheme.conditions_text or scheme.characterizations:
                schemes.append(scheme)
            elif scanned_page:
                page_scheme = await asyncio.to_thread(extract_with_vlm_page_read, pil_img, settings, vlm_model)
                page_scheme.page_number = page_num
                page_scheme.image_index = img_idx
                schemes.append(page_scheme)

    return schemes


async def run_image_scheme_extraction(
    pdf_path: Path,
    settings: Settings,
    vlm_model: str | None = None,
    local_model: str | None = None,
    on_image_progress: Callable[[int, int, int], Awaitable[None]] | None = None,
    *,
    force_page_read: bool = False,
    text_char_count: int = 0,
    figure_pages: list[int] | None = None,
    figure_page_info: list[dict[str, Any]] | None = None,
    segments: list[Any] | None = None,
) -> tuple[PipelineStage, list[ExtractedScheme]]:
    schemes = await extract_schemes_from_pdf(
        pdf_path,
        settings,
        vlm_model,
        local_model,
        on_image_progress=on_image_progress,
        force_page_read=force_page_read,
        text_char_count=text_char_count,
        figure_pages=figure_pages,
        segments=segments,
    )

    scheme_steps = schemes_to_reaction_steps(schemes)
    page_steps = page_read_to_reaction_steps(schemes)
    total_molecules = sum(len(s.molecules) for s in schemes)
    page_reads = sum(1 for s in schemes if s.mode == "page_read")
    transcribed_chars = sum(len(s.transcribed_text) for s in schemes)
    figure_pages = figure_pages or []
    fig_detail = ""
    if figure_pages:
        fig_detail = f" (figure pages {figure_pages})"
    pages_analyzed = len(schemes)

    if total_molecules == 0 and not page_steps and transcribed_chars == 0:
        if pages_analyzed > 0:
            return (
                PipelineStage(
                    id="image_scheme",
                    label="Scheme / image extraction (VLM)",
                    status=PipelineStageStatus.done,
                    detail=(
                        f"VLM analyzed {pages_analyzed} figure page(s){fig_detail}; "
                        "no molecules/steps recovered (may be non-scheme figure or VLM parse issue)."
                    ),
                    data={
                        "images_processed": pages_analyzed,
                        "schemes_found": 0,
                        "molecules_extracted": 0,
                        "page_reads": page_reads,
                        "scheme_steps": 0,
                        "page_steps": 0,
                        "figure_pages": figure_pages,
                        "figure_page_info": figure_page_info or [],
                        "schemes": [s.to_dict() for s in schemes],
                    },
                ),
                schemes,
            )
        return (
            PipelineStage(
                id="image_scheme",
                label="Scheme / image extraction (VLM)",
                status=PipelineStageStatus.stubbed,
                detail=f"No schemes or page text recovered from {len(schemes)} analyzed image(s){fig_detail}.",
                data={
                    "images_processed": len(schemes),
                    "schemes_found": 0,
                    "molecules_extracted": 0,
                    "page_reads": page_reads,
                    "scheme_steps": 0,
                    "page_steps": 0,
                    "figure_pages": figure_pages,
                    "figure_page_info": figure_page_info or [],
                },
            ),
            schemes,
        )

    return (
        PipelineStage(
            id="image_scheme",
            label="Scheme / image extraction (VLM)",
            status=PipelineStageStatus.done,
            detail=(
                f"{len(scheme_steps)} scheme step(s), {len(page_steps)} page-read step(s), "
                f"{total_molecules} molecule(s), {page_reads} page-read(s) "
                f"from {len(schemes)} image(s){fig_detail}"
            ),
            data={
                "images_processed": len(schemes),
                "schemes_found": sum(1 for s in schemes if s.molecules),
                "molecules_extracted": total_molecules,
                "page_reads": page_reads,
                "transcribed_chars": transcribed_chars,
                "scheme_steps": len(scheme_steps),
                "page_steps": len(page_steps),
                "figure_pages": figure_pages,
                "figure_page_info": figure_page_info or [],
                "schemes": [s.to_dict() for s in schemes],
            },
        ),
        schemes,
    )
