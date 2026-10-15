"""Tests for evidence detection and retrieval (P0 architecture)."""

from app.schemas.reaction_route import DocumentSegment, ParsedDocument
from app.services.document_evidence import build_evidence_profile, classify_page
from app.services.evidence_retrieval import build_extraction_plan, plan_extraction_passes
from app.services.image_extract import detect_figure_scheme_pages
from app.services.reconcile import reconcile_route
from app.schemas.reaction_route import ChemicalEntity, EvidenceRef, ReactionStep


def test_classify_experimental_page():
    text = (
        "Example 1. To a solution of compound 5 (2.0 g, 5 mmol) in DMF (20 mL) "
        "was added triethylamine (1.1 g). The mixture was stirred at 80 °C for 4 h. "
        "Workup afforded product 6 (1.8 g, 72% yield)."
    )
    pe = classify_page(3, text)
    assert pe.synthesis_score >= 0.5
    types = {h.evidence_type for h in pe.hits}
    assert "experimental_procedure" in types or "patent_example" in types


def test_evidence_retrieval_prefers_procedure_pages():
    segments = [
        DocumentSegment(
            segment_id="page-1",
            kind="body",
            page_start=1,
            page_end=1,
            text="Abstract. We report a new synthesis of an antiviral agent with high potency.",
        ),
        DocumentSegment(
            segment_id="page-2",
            kind="procedure_candidate",
            page_start=2,
            page_end=2,
            text=(
                "Example 1. Compound A (1.0 g) was dissolved in ethanol (10 mL) and heated at reflux "
                "for 12 h to give intermediate B (0.9 g, 85%)."
            ),
        ),
        DocumentSegment(
            segment_id="page-3",
            kind="procedure_candidate",
            page_start=3,
            page_end=3,
            text=(
                "Example 2. Intermediate B (0.5 g) was treated with sodium hydroxide in water at 70-95 °C "
                "for 3 h. Acyclovir was obtained in 98.7% yield after purification."
            ),
        ),
    ]
    parsed = ParsedDocument(filename="test.pdf", page_count=3, text="\n\n".join(s.text for s in segments), segments=segments)
    plan = build_extraction_plan(parsed, max_chars=50000)
    assert plan.chars_selected > 0
    assert 2 in plan.pages_included and 3 in plan.pages_included
    assert "Example" in plan.text


def test_reconcile_does_not_merge_unrelated_product_only_steps():
    s1 = ReactionStep(
        step_index=1,
        step_label="Step A",
        reactants=[ChemicalEntity(name="A", role="reactant")],
        products=[ChemicalEntity(name="B", role="product", smiles="CCO")],
    )
    s2 = ReactionStep(
        step_index=2,
        step_label="Step B",
        reactants=[ChemicalEntity(name="X", role="reactant")],
        products=[ChemicalEntity(name="Y", role="product", smiles="CCO")],
    )
    route = reconcile_route(text_steps=[s1, s2])
    assert len(route.steps) == 2


def test_detect_paper_not_patent_for_journal_article():
    text = (
        "Journal of Chemistry, Vol. 47 (4), P. 523 - 528, 2009\n"
        "SYNTHESIS OF ACYCLOVIR\nReceived 4 February 2009\n"
        "Abstract\nA one-pot process for synthesis of acyclovir was achieved.\n"
        "II - Experimental\nPreparation of dioxolane-diacetate\n"
        "Dioxolane (224 g, 3 mol) was added to acetic anhydride (306 g)."
    )
    seg = DocumentSegment(segment_id="p1", kind="body", page_start=1, text=text)
    profile = build_evidence_profile([seg], text)
    assert profile.document_type == "paper"


def test_filter_removes_literature_and_summary_steps():
    detailed = ReactionStep(
        step_index=2,
        step_label="Preparation of dioxolane-diacetate (7)",
        reactants=[ChemicalEntity(name="dioxolane", role="reactant", amount="224 g, 3 mol")],
        products=[ChemicalEntity(name="dioxolane-diacetate (7)", role="product", amount="431.5 g")],
        conditions={"temperature": "below 30 °C", "time": "1 h"},
        yield_value="81.7%",
        evidence="Dioxolane (224 g) was added ... yield 431.5 g (81.7%).",
    )
    literature = ReactionStep(
        step_index=10,
        step_label="Literature variant [6] — Acetylation of guanine in N,N-dimethylacetamide",
        reactants=[ChemicalEntity(name="guanine", role="reactant")],
        products=[ChemicalEntity(name="diacetylguanine", role="product")],
        yield_value="90.5%",
    )
    summary = ReactionStep(
        step_index=1,
        step_label="Abstract-level route: one-pot synthesis of acyclovir (no detailed procedure)",
        reactants=[ChemicalEntity(name="guanine", role="reactant")],
        products=[ChemicalEntity(name="acyclovir", role="product")],
        yield_value="59%",
    )
    route = reconcile_route(
        text_steps=[summary, detailed, literature],
        document_type="paper",
    )
    labels = [s.step_label for s in route.steps]
    assert any("dioxolane-diacetate" in lb for lb in labels)
    assert not any("Literature variant" in lb for lb in labels)
    assert not any("Abstract-level" in lb for lb in labels)


def test_absorb_redundant_vlm_scheme_into_text_step():
    from app.services.reconcile import reconcile_route

    text_step = ReactionStep(
        step_index=1,
        step_label="Preparation of dioxolane-diacetate (1-acetyl-2-acetylmethoxy-ethyleneglycol, 7)",
        reactants=[
            ChemicalEntity(name="dioxolane (5, 1,3-dioxolane)", role="reactant", amount="224 g, 3 mol"),
            ChemicalEntity(name="acetic anhydride", role="reactant", amount="306 g, 3 mol"),
        ],
        reagents=[ChemicalEntity(name="p-toluenesulfonic acid monohydrate", role="reagent", amount="0.5 g")],
        products=[
            ChemicalEntity(
                name="dioxolane-diacetate (1-acetyl-2-acetylmethoxy-ethyleneglycol, 7)",
                role="product",
                amount="431.5 g",
            )
        ],
        conditions={"temperature": "below 30 °C", "time": "1 h"},
        yield_value="81.7%",
        source="text",
    )
    scheme_step = ReactionStep(
        step_index=6,
        step_label="Figure 3 (single step)",
        reactants=[ChemicalEntity(name="5", role="reactant", smiles="C1COCO1")],
        reagents=[ChemicalEntity(name="6", role="reagent", smiles="CC(=O)OC(=O)C")],
        catalysts=[ChemicalEntity(name="TsOH", role="catalyst", smiles="Cc1ccc(cc1)S(=O)(=O)O")],
        conditions={"other": "TsOH"},
        source="scheme",
        evidence_refs=[
            EvidenceRef(
                page=3,
                section="figures",
                evidence_type="reaction_scheme",
                source="scheme",
                quote="TsOH",
            )
        ],
    )
    route = reconcile_route(text_steps=[text_step], scheme_steps=[scheme_step], document_type="paper")
    assert len(route.steps) == 1
    merged = route.steps[0]
    assert merged.reactants[0].smiles == "C1COCO1" or any(e.smiles for e in merged.reactants)
    assert any("Merged 1 redundant VLM scheme" in w for w in route.warnings)


def test_multi_pass_preserves_mandatory_procedure_pages():
    """Procedure pages must not be silently dropped when they exceed one pass budget."""
    filler = (
        "Example 1. Compound A (10.0 g, 50 mmol) was dissolved in ethanol (100 mL) "
        "and stirred at reflux for 12 h. Workup afforded product B (8.5 g, 72% yield). "
    )
    procedure = filler * 90  # ~12k chars per page
    segments = []
    for pg in (3, 4, 5):
        segments.append(
            DocumentSegment(
                segment_id=f"page-{pg}",
                kind="procedure_candidate",
                page_start=pg,
                page_end=pg,
                text=procedure.replace("Example 1", f"Example {pg - 2}"),
                meta={"synthesis_score": 0.85, "section": "experimental"},
            )
        )
    parsed = ParsedDocument(
        filename="multi.pdf",
        page_count=5,
        text="\n\n".join(s.text for s in segments),
        segments=segments,
    )
    passes = plan_extraction_passes(parsed, max_chars_per_pass=8000)
    all_included: set[int] = set()
    for plan in passes:
        all_included.update(plan.pages_included)
    assert len(passes) >= 2
    assert {3, 4, 5}.issubset(all_included)


def test_merge_preserves_yield_conflict():
    """Conflicting yields from different sources must be recorded, not silently dropped."""
    text = ReactionStep(
        step_index=1,
        step_label="Preparation of B",
        reactants=[ChemicalEntity(name="A", role="reactant")],
        products=[ChemicalEntity(name="B", role="product")],
        yield_value="63%",
        source="text",
        information_kind="explicit",
    )
    scheme = ReactionStep(
        step_index=2,
        step_label="Figure 1",
        reactants=[ChemicalEntity(name="A", role="reactant", smiles="C")],
        products=[ChemicalEntity(name="B", role="product", smiles="CC")],
        yield_value="61%",
        source="scheme",
        information_kind="scheme",
    )
    route = reconcile_route(text_steps=[text], scheme_steps=[scheme], document_type="paper")
    assert len(route.steps) == 1
    assert any(c.field.startswith("yield:") for c in route.conflicts)


def test_confidence_penalizes_inferred_over_explicit():
    from app.services.confidence_score import score_step

    explicit = ReactionStep(
        step_index=1,
        step_label="Example 1",
        reactants=[ChemicalEntity(name="A", role="reactant", amount="2.0 g")],
        products=[ChemicalEntity(name="B", role="product")],
        yield_value="72%",
        information_kind="explicit",
        source="text",
    )
    inferred = ReactionStep(
        step_index=2,
        step_label="Inferred step",
        reactants=[ChemicalEntity(name="A", role="reactant")],
        products=[ChemicalEntity(name="B", role="product")],
        information_kind="inferred",
        source="text",
    )
    assert score_step(explicit) > score_step(inferred)


def test_mandatory_filtered_page_included_in_later_pass():
    """Pages filtered from candidates (e.g. discussion) must still appear in a later pass."""
    filler = (
        "Example 1. Compound A (10.0 g, 50 mmol) was dissolved in ethanol (100 mL) "
        "and stirred at reflux for 12 h. Workup afforded product B (8.5 g, 72% yield). "
    )
    segments = [
        DocumentSegment(
            segment_id="page-1",
            kind="procedure_candidate",
            page_start=1,
            page_end=1,
            text=filler * 100,
            meta={"synthesis_score": 0.85, "section": "experimental"},
        ),
        DocumentSegment(
            segment_id="page-2",
            kind="body",
            page_start=2,
            page_end=2,
            text="Discussion. Compound B was tested in biological assays. IC50 values are shown.",
            meta={"synthesis_score": 0.4, "section": "discussion"},
        ),
    ]
    parsed = ParsedDocument(
        filename="disc.pdf",
        page_count=2,
        text="\n\n".join(s.text for s in segments),
        segments=segments,
    )
    passes = plan_extraction_passes(parsed, max_chars_per_pass=5000)
    all_pages: set[int] = set()
    for plan in passes:
        all_pages.update(plan.pages_included)
        assert plan.chars_selected > 0
    assert 2 in all_pages
    assert all(p.chars_selected > 0 for p in passes)


def test_literature_precedent_step_filtered_when_experimental_exists():
    experimental = ReactionStep(
        step_index=2,
        step_label="Preparation of B — acetylation of guanine",
        reactants=[ChemicalEntity(name="guanine", role="reactant", amount="25 g, 0.16 mol")],
        reagents=[ChemicalEntity(name="acetic anhydride", role="reagent", amount="50 ml")],
        products=[ChemicalEntity(name="diacetylguanine", role="product")],
        yield_value="N/A",
        evidence="Guanine (25 g) was added to acetic anhydride (50 ml) and stirred at 95 °C for 2 h.",
    )
    literature = ReactionStep(
        step_index=6,
        step_label="Step 1 - Acetylation of guanine (to diacetylguanine)",
        reactants=[ChemicalEntity(name="guanine", role="reactant")],
        products=[ChemicalEntity(name="diacetylguanine", role="product")],
        yield_value="90.5% (acetic anhydride in N,N-dimethylacetamide); 95% (acetic anhydride/acetic acid)",
        evidence=(
            '"Guanine has been acetylated using acetic anhydride in N,N-dimethylacetamide '
            'to give diacetylguanine in 90.5% yield"'
        ),
        source="text",
    )
    route = reconcile_route(text_steps=[experimental, literature], document_type="paper")
    assert len(route.steps) == 1
    assert route.steps[0].step_index == 1


def test_scheme_route_yield_not_merged_without_product_overlap():
    """Route-level scheme yields must not attach to an unrelated text step."""
    text_step = ReactionStep(
        step_index=1,
        step_label="Preparation of intermediate A",
        reactants=[ChemicalEntity(name="substrate", role="reactant", amount="10.0 g")],
        products=[ChemicalEntity(name="intermediate A", role="product", amount="8.0 g")],
        yield_value="80%",
        source="text",
        information_kind="explicit",
    )
    scheme_step = ReactionStep(
        step_index=2,
        step_label="Figure 3 (single step)",
        reactants=[ChemicalEntity(name="substrate", role="reactant", smiles="CCO")],
        products=[],
        yield_value="63%",
        source="scheme",
        information_kind="scheme",
    )
    route = reconcile_route(text_steps=[text_step], scheme_steps=[scheme_step], document_type="paper")
    assert len(route.steps) == 1
    merged = route.steps[0]
    assert merged.yield_value == "80%"
    assert "yield conflict" not in (merged.notes or "").lower()
    assert not any(c.field.startswith("yield:") for c in route.conflicts)


def test_detect_figure_scheme_pages_acyclovir_like():
    from pathlib import Path

    pdf = Path(r"c:\Users\galin\Desktop\PHD\Patents_Acyclovir\Papers\pdf.pdf")
    if not pdf.exists():
        return
    pages = detect_figure_scheme_pages(pdf, segments=None, min_drawings=15, max_pages=8)
    page_nums = [p["page"] for p in pages]
    assert 2 in page_nums  # Figure 2 synthesis pathway
    assert 4 not in page_nums  # IC50 plot
    assert 5 not in page_nums  # bioactivity figure
