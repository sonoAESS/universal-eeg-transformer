"""Criterios de aceptación de spec-02 (reentrenar universal_refs_conv/gru).

Suite rápida: contrato de los YAMLs dedicados del vídeo.
Payoff (`-m payoff`): checkpoint con carga estricta y evaluación ventaneada
con r ≥ 0.97 y VE ≥ 0.92, mejorando el estado actual (r = 0.95).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

CFG = {
    "conv": ROOT / "config/topomap_video_universal_refs_conv.yaml",
    "gru": ROOT / "config/topomap_video_universal_refs_gru.yaml",
}
RUN_DIR = {
    "conv": ROOT / "runs/topomap_video/universal_refs_conv",
    "gru": ROOT / "runs/topomap_video/universal_refs_gru",
}


def _cfg(cell: str) -> dict:
    with open(CFG[cell], encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_configs_dedicados_usan_cabeza_temporal_y_run_dir_canonico():
    for cell in ("conv", "gru"):
        cfg = _cfg(cell)
        assert cfg["model"]["temporal_cell"] == cell
        assert int(cfg["model"]["temporal_window"]) > 0
        assert cfg["model"]["temporal_stride"] >= 1
        assert cfg["training"]["run_dir"] == f"runs/topomap_video/universal_refs_{cell}"


@pytest.mark.payoff
def test_2_1_2_2_conv_carga_estricta_y_ventaneado_alcanza():
    """2.1: el checkpoint carga sin skip_mismatch; 2.2: r≥0.97 y VE≥0.92."""
    _assert_carga_estricta("conv")
    _assert_windowed_targets("conv")


@pytest.mark.payoff
def test_2_4_gru_ventaneado_alcanza():
    """2.4: misma escala de umbrales para gru."""
    _assert_windowed_targets("gru")


@pytest.mark.payoff
def test_2_3_no_peor_que_multi_montage():
    """2.3: la cabeza temporal justifica su existencia vs multi_montage."""
    ve = {}
    for cell in ("conv", "gru"):
        ve[cell] = _windowed_ve(cell)
    mm = ROOT / "runs/topomap_video/multi_montage/metrics_windowed_test.csv"
    if not mm.exists():
        pytest.skip("sin metrics_windowed_test.csv de multi_montage")
    ve_mm = pd.read_csv(mm)["ve"].mean()
    for cell in ("conv", "gru"):
        assert ve[cell] >= ve_mm - 0.01, \
            f"universal_refs_{cell} peor que multi_montage ({ve[cell]:.3f} < {ve_mm:.3f})"


def _assert_carga_estricta(cell: str):
    from eeg_transform.evaluation.topomap_video import load_config_path
    from eeg_transform.data.dataset import build_dataset
    from eeg_transform.training.trainer import (
        build_multiconfig_model,
        load_multiconfig_data,
    )

    cfg = load_config_path(f"universal_refs_{cell}")
    ds = build_dataset(cfg)
    data = load_multiconfig_data(cfg, ds)
    model = build_multiconfig_model(cfg, data)
    model.core.ensure_built()
    model.load_weights(str(RUN_DIR[cell] / "best.weights.h5"))  # sin skip


def _windowed_ve(cell: str) -> float:
    path = RUN_DIR[cell] / "metrics_windowed_test.csv"
    assert path.exists(), f"{cell}: falta metrics_windowed_test.csv"
    m = pd.read_csv(path)
    assert "r" in m.columns and "ve" in m.columns
    return float(m["ve"].mean())


def _assert_windowed_targets(cell: str):
    ve = _windowed_ve(cell)
    path = RUN_DIR[cell] / "metrics_windowed_test.csv"
    r = float(pd.read_csv(path)["r"].mean())
    assert r >= 0.97, f"universal_refs_{cell}: r={r:.3f} < 0.97"
    assert ve >= 0.92, f"universal_refs_{cell}: VE={ve:.3f} < 0.92"