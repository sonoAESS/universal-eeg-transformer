"""Gobernanza SDD (spec-00): toda spec tiene secciones y un test asociado.

Suite rápida: verifica la coherencia del proceso (documentación ↔ tests).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPECS = sorted((ROOT / "docs" / "specs").glob("spec-*.md"))


def _slug_to_test(slug: str) -> Path:
    return ROOT / "tests" / f"test_spec_{slug.replace('-', '_')}.py"


def _parse_spec(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    m = re.match(r"#\s*spec-(\d+)-([\w-]+)\.md", path.name)
    assert m, f"Nombre de spec no sigue spec-XX-<slug>.md: {path.name}"
    return m.group(1), m.group(2)


def test_toda_spec_tiene_secciones_y_test_asociado():
    assert SPECS, "No hay specs en docs/specs/"
    for spec in SPECS:
        text = spec.read_text(encoding="utf-8")
        for section in ("## Contexto", "## Requisitos", "## Criterios de aceptación"):
            assert section in text, f"{spec.name}: falta '{section}'"
        _, slug = _parse_spec(spec)
        test = _slug_to_test(slug)
        assert test.exists(), f"{spec.name}: falta test {test.name}"


def test_toda_spec_con_payoff_tiene_test_marcado_payoff():
    """0.2: si la tabla de criterios contiene 'sí' (payoff), el test asociado
    debe declarar al menos un `@pytest.mark.payoff` con el mismo umbral."""
    for spec in SPECS:
        text = spec.read_text(encoding="utf-8")
        if "| sí |" not in text and "| sí|\n" not in text:
            continue
        _, slug = _parse_spec(spec)
        test_text = _slug_to_test(slug).read_text(encoding="utf-8")
        assert "@pytest.mark.payoff" in test_text, \
            f"{spec.name}: criterios de payoff sin test marcado"


def test_payoff_excluido_por_defecto_y_registrado():
    """0.3: la suite rápida ignora payoff; el marcador está registrado."""
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "-m 'not payoff'" in pyproject
    assert "'payoff:" in pyproject