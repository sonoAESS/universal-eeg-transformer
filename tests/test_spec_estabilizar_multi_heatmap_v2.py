"""Criterios de aceptación de spec-01 (estabilizar multi_heatmap_v2).

Suite rápida: contrato de artefactos (esquemas de history/metrics).
Payoff (`-m payoff`): convergencia acotada y VE por ruta ≥ umbral.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

HISTORY = ROOT / "runs/topomap_video/multi_heatmap_v2/history.csv"
METRICS_CANONICAL = ROOT / "runs/topomap_video/figs/topomap_video_canonical_unipolar_metrics.csv"
METRICS_1020 = ROOT / "runs/topomap_video/figs/topomap_video_10-20_unipolar_metrics.csv"

HISTORY_COLS = {
    "epoch", "learning_rate", "loss", "loss_estandarizada", "mse_real_V2",
    "val_loss", "val_loss_estandarizada", "val_mse_real_V2",
    "surface_loss", "val_surface_loss", "field_consist_loss",
    "val_field_consist_loss", "xconfig_loss", "val_xconfig_loss",
    "temporal_loss", "val_temporal_loss",
}


def test_history_tiene_esquema_esperado():
    """Los artefactos deben registrar las métricas que la spec monitoriza."""
    assert HISTORY.exists(), "falta history.csv del entrenamiento"
    cols = set(pd.read_csv(HISTORY, nrows=1).columns)
    missing = HISTORY_COLS - cols
    assert not missing, f"history.csv sin columnas: {sorted(missing)}"


def test_metrics_tienen_esquema_y_metodo():
    for path in (METRICS_CANONICAL, METRICS_1020):
        assert path.exists(), f"falta {path.name}"
        m = pd.read_csv(path)
        for col in ("method", "ruta", "rmse_uV", "r", "ve"):
            assert col in m.columns, f"{path.name}: falta columna {col}"
        assert "multi_heatmap_v2" in set(m["method"])


@pytest.mark.payoff
def test_1_1_1_2_convergencia_acotada():
    """1.1/1.2: val_loss_estandarizada decreciente y surface_loss acotada."""
    h = pd.read_csv(HISTORY)
    first = h["val_loss_estandarizada"].iloc[0]
    last = h["val_loss_estandarizada"].iloc[-1]
    assert last < first, "val_loss_estandarizada no decrece"
    assert h["val_loss_estandarizada"].max() <= 5.0 * first, \
        "val_loss_estandarizada explota (>5x inicial)"
    assert h["val_surface_loss"].max() <= 5000.0, \
        f"val_surface_loss diverge (máx {h['val_surface_loss'].max():.0f} > 5000)"


def _ve_por_ruta(path: Path, method: str) -> pd.Series:
    m = pd.read_csv(path)
    return m[m["method"] == method].groupby("ruta")["ve"].mean()


@pytest.mark.payoff
def test_1_3_1_4_1_5_ve_por_ruta_umbrales():
    """1.3–1.5: VE ≥ 0.90 en todas las rutas salvo laplacian (0.70), en
    canonical y 10-20 para el método multi_heatmap_v2."""
    for path in (METRICS_CANONICAL, METRICS_1020):
        ve = _ve_por_ruta(path, "multi_heatmap_v2")
        assert len(ve) == 7, f"{path.name}: faltan rutas en multi_heatmap_v2"
        rest = ve.drop(index="laplacian", errors="ignore")
        assert (rest >= 0.90).all(), f"{path.name}: VE<0.90 en {rest[rest < 0.90].to_dict()}"
        assert ve.get("laplacian", 0.0) >= 0.70, f"{path.name}: laplacian VE<0.70"