"""Criterios de aceptación de spec-03 (gate de calidad en topomap-video).

Payoff (`-m payoff`): `run_topomap_video` expone `quality_gate` (0.85), marca
filas no convergentes y genera `*_resumen_calidad.csv`.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MOD = "eeg_transform.evaluation.topomap_video"


@pytest.mark.payoff
def test_3_1_run_topomap_video_expone_quality_gate():
    """3.1: parámetro `quality_gate` con defecto 0.85 en run_topomap_video."""
    import eeg_transform.evaluation.topomap_video as mod

    sig = inspect.signature(mod.run_topomap_video)
    p = sig.parameters.get("quality_gate")
    assert p is not None, "run_topomap_video sin parámetro quality_gate"
    assert p.default == 0.85, f"quality_gate por defecto {p.default} ≠ 0.85"


@pytest.mark.payoff
def test_3_2_3_3_marca_fila_con_ve_insuficiente():
    """3.2: VE=0.5 ⇒ convergido=False; 3.3: quality_gate=0 ⇒ sin descartes."""
    import eeg_transform.evaluation.topomap_video as mod

    assert callable(getattr(mod, "apply_quality_gate", None)), \
        "falta apply_quality_gate(results, threshold) en topomap_video"
    fake = type("R", (), {"name": "m", "ve_medio": 0.5})()
    assert mod.apply_quality_gate([fake], threshold=0.85) == [False]
    assert mod.apply_quality_gate([fake], threshold=0.0) == [True]


@pytest.mark.payoff
def test_3_4_resumen_calidad_csv():
    """3.4: el run real escribe *_resumen_calidad.csv junto a *metrics.csv."""
    from eeg_transform.evaluation.topomap_video import run_topomap_video

    out = run_topomap_video(
        models=["multi_montage"], cfg_label="canonical", source="unipolar",
        duration_s=5.0, window=32, grid_px=32, frame=300,
        out_dir=str(ROOT / "runs/topomap_video/figs/_spec_tmp"),
    )
    resumen = out.get("calidad")
    assert resumen is not None, "run_topomap_video sin clave 'calidad'"
    resumen = Path(resumen)
    assert resumen.exists() and resumen.name.endswith("_resumen_calidad.csv")
    assert "ve_medio" in resumen.read_text(encoding="utf-8")