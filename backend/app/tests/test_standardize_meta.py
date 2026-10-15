"""Tests for chemical resolution metadata (Problem 4)."""

from app.config import Settings
from app.services.chemical_standardize import resolve_smiles_with_meta


def test_resolve_smiles_with_meta_marks_synonym(monkeypatch):
    def fake_pubchem(name: str) -> str | None:
        if name == "acetic anhydride":
            return "CC(=O)OC(=O)C"
        return None

    monkeypatch.setattr("app.services.chemical_standardize.pubchem_lookup", fake_pubchem)
    monkeypatch.setattr("app.services.chemical_standardize.opsin_lookup", lambda _n: None)
    monkeypatch.setattr("app.services.chemical_standardize.llm_resolve_name", lambda *_a, **_k: None)

    smi, status, source = resolve_smiles_with_meta("Ac2O", Settings())
    assert smi == "CC(=O)OC(=O)C"
    assert status == "synonym"
    assert source == "pubchem"


def test_resolve_smiles_with_meta_unresolved_without_hallucination(monkeypatch):
    monkeypatch.setattr("app.services.chemical_standardize.pubchem_lookup", lambda _n: None)
    monkeypatch.setattr("app.services.chemical_standardize.opsin_lookup", lambda _n: None)
    monkeypatch.setattr("app.services.chemical_standardize.llm_resolve_name", lambda *_a, **_k: None)

    smi, status, source = resolve_smiles_with_meta("unknown-lab-mixture-xyz", Settings())
    assert smi is None
    assert status == "unresolved"
    assert source == "none"
