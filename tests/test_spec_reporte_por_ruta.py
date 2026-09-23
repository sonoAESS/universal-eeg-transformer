"""Criterios de aceptación de spec-04 (reporte de rutas normalizado).

Suite rápida: contrato de `_route_stats` sobre el que se construye el resumen.
Payoff (`-m payoff`): `*_resumen_rutas.csv` con `rmse_norm` comparable entre
referencias (laplacian no domina).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

RESUMEN_CANONICAL = ROOT / "runs/topomap_video/figs/topomap_video_canonical_unipolar_resumen_rutas.csv"
RESUMEN_1020 = ROOT / "runs/topomap_video/figs/topomap_video_10-20_unipolar_resumen_rutas.csv"
METODOS_CANONICAL = ROOT / "runs/topomap_video/figs/topomap_video_canonical_unipolar_resumen_metodos.csv"


def test_route_stats_expone_base_del_resumen():
    """El resumen se apoya en `_route_stats` (centrado temporal) existente."""
    from eeg_transform.evaluation.metrics import _route_stats

    st = _route_stats(
        np.random.default_rng(0).standard_normal((64, 19)),
        np.random.default_rng(1).standard_normal((64, 19)),
    )
    for key in ("rmse", "r", "ve"):
        assert key in st


def _resumen(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@pytest.mark.payoff
def test_4_1_resumen_rutas_con_esquema():
    """4.1: ficheros con columnas de los requisitos 1–3."""
    for path in (RESUMEN_CANONICAL, RESUMEN_1020):
        assert path.exists(), f"falta {path.name}"
        m = pd.read_csv(path)
        for col in ("method", "ruta", "rmse_uV", "rmse_norm", "r", "ve", "n_ventanas"):
            assert col in m.columns, f"{path.name}: falta columna {col}"
        assert set(m["ruta"]) == {
            "unipolar", "linked_mastoids", "linked_ears", "bipolar",
            "car", "rest", "laplacian"}
    assert METODOS_CANONICAL.exists(), "falta *_resumen_metodos.csv"
    m = pd.read_csv(METODOS_CANONICAL)
    for col in ("method", "rmse_norm_mediana", "ve_mediana"):
        assert col in m.columns, f"resumen_metodos: falta columna {col}"


@pytest.mark.payoff
def test_4_2_laplacian_no_domina_rmse_norm():
    """4.2: rmse_norm de laplacian ≤ 3× el de unipolar en el analítico."""
    for path in (RESUMEN_CANONICAL, RESUMEN_1020):
        m = _resumen(path)
        ana = m[m["method"] == "Analítico"].set_index("ruta")
        ratio = ana.loc["laplacian", "rmse_norm"] / ana.loc["unipolar", "rmse_norm"]
        assert ratio <= 3.0, f"{path.name}: laplacian {ratio:.2f}× unipolar (rmse_norm)"


@pytest.mark.payoff
def test_4_3_agregado_robusto():
    """4.3: el agregado por método es mediana de rmse_norm y mediana de ve
    (robusto ante laplacian), derivado del resumen por ruta."""
    rutas = _resumen(RESUMEN_CANONICAL)
    metodos = pd.read_csv(METODOS_CANONICAL).set_index("method")
    for method, grp in rutas.groupby("method"):
        assert metodos.loc[method, "rmse_norm_mediana"] == pytest.approx(
            grp["rmse_norm"].median())
        assert metodos.loc[method, "ve_mediana"] == pytest.approx(
            grp["ve"].median())