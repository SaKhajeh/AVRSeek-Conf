"""Tests for P1 route graph / synonyms and P2 confidence / compound linking."""

from app.schemas.reaction_route import ChemicalEntity, ReactionStep, SynthesisRoute
from app.services.chemical_synonyms import apply_synonym_normalization, normalize_chemical_name
from app.services.confidence_score import apply_confidence_scores, score_step
from app.services.example_linking import link_route_compounds
from app.services.route_graph import build_route_graph


def test_synonym_normalization():
    from app.services.chemical_synonyms import extract_compound_id

    assert normalize_chemical_name("Ac2O") == "acetic anhydride"
    assert normalize_chemical_name("p-TsOH") == "p-toluenesulfonic acid"
    base, cid = extract_compound_id("guanine (3)")
    assert base == "guanine" and cid == "3"


def test_apply_synonym_on_route():
    route = SynthesisRoute(
        steps=[
            ReactionStep(
                step_index=1,
                step_label="Step 1",
                reactants=[ChemicalEntity(name="Ac2O", role="reactant")],
                products=[ChemicalEntity(name="dioxolane-diacetate (7)", role="product")],
            )
        ]
    )
    n = apply_synonym_normalization(route)
    assert n >= 1
    assert route.steps[0].reactants[0].name == "acetic anhydride"
    assert route.steps[0].reactants[0].original_name == "Ac2O"
    assert route.steps[0].reactants[0].canonical_name == "acetic anhydride"


def test_route_graph_builds_nodes_and_edges():
    route = SynthesisRoute(
        target_name="acyclovir",
        steps=[
            ReactionStep(
                step_index=1,
                step_label="Prep A",
                reactants=[ChemicalEntity(name="dioxolane", role="reactant")],
                products=[ChemicalEntity(name="dioxolane-diacetate (7)", role="product")],
            ),
            ReactionStep(
                step_index=2,
                step_label="Prep B",
                reactants=[ChemicalEntity(name="dioxolane-diacetate (7)", role="reactant")],
                products=[ChemicalEntity(name="guanine adduct (8)", role="product")],
            ),
        ],
    )
    graph = build_route_graph(route)
    assert len(graph.nodes) >= 3
    assert any(e.relation == "intermediate_flow" for e in graph.edges)


def test_compound_linking():
    route = SynthesisRoute(
        steps=[
            ReactionStep(
                step_index=1,
                step_label="Example 1 — prep",
                reactants=[ChemicalEntity(name="guanine (3)", role="reactant")],
                products=[ChemicalEntity(name="intermediate (5)", role="product")],
            ),
            ReactionStep(
                step_index=2,
                step_label="Example 2 — cyclization",
                reactants=[ChemicalEntity(name="intermediate (5)", role="reactant")],
                products=[ChemicalEntity(name="acyclovir", role="product")],
            ),
        ]
    )
    links = link_route_compounds(route)
    ids = {l.compound_id for l in links}
    assert "3" in ids or "5" in ids
    five = next((l for l in links if l.compound_id == "5"), None)
    assert five is not None
    assert len(five.step_indices) >= 2


def test_confidence_scoring():
    rich = ReactionStep(
        step_index=1,
        step_label="Example 1",
        reactants=[ChemicalEntity(name="A", role="reactant", amount="2.0 g, 5 mmol")],
        products=[ChemicalEntity(name="B", role="product")],
        conditions={"temperature": "80 °C", "time": "4 h"},
        yield_value="72%",
        evidence="Example 1. To a solution of compound 5 (2.0 g, 5 mmol) in DMF...",
        source="text",
    )
    sparse = ReactionStep(step_index=2, step_label="vague", reactants=[], products=[])
    assert score_step(rich) > score_step(sparse)
    route = SynthesisRoute(steps=[rich, sparse])
    apply_confidence_scores(route)
    assert route.steps[0].confidence is not None
    assert route.steps[0].confidence > route.steps[1].confidence
