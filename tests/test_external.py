"""Tests de ingesta externa BIDS y su integración multi-configuración.

Las funciones dependientes de ``mne-bids`` se prueban solo si la
dependencia está disponible; la integración con ``build_multiconfig`` se
ejercita mediante un npz externo sintético guardado en caché.
"""

from __future__ import annotations

import numpy as np
import pytest

from eeg_transform.config import EEGTransformConfig
from eeg_transform.data.dataset import DATASET_VERSION, MultiReferenceDataset
from eeg_transform.data.external import (
    EXTERNAL_SUBDIR,
    find_external_cache,
    list_bids_recordings,
)
from eeg_transform.experiments.multi import build_multiconfig


class _LeadField:
    def __init__(self, matrix):
        self.matrix = matrix


def _make_canonical(tmp_path, n_ch=42, n_samples=90):
    """Dataset ancla sintético (mismo espíritu que tests/test_multi)."""
    rng = np.random.default_rng(0)
    th = rng.uniform(0.35, np.pi - 0.35, n_ch)
    ph = rng.uniform(0, 2 * np.pi, n_ch)
    pos = 0.088 * np.stack(
        [np.sin(th) * np.cos(ph),
         np.sin(th) * np.sin(ph),
         np.cos(th)], axis=1)
    names = [f"c{i}" for i in range(n_ch)]
    names[20] = "Cz"
    lf = rng.normal(size=(n_ch, 24)).astype(np.float32)
    refs = {k: rng.normal(size=(n_samples, n_ch)).astype(np.float32)
            for k in ("unipolar", "linked_mastoids", "linked_ears",
                      "bipolar", "car", "rest", "laplacian")}
    return MultiReferenceDataset(
        refs=refs,
        split_idx={"train": np.arange(45), "val": np.arange(45, 68),
                   "test": np.arange(68, 90)},
        subject_ids=np.zeros(n_samples, np.int16),
        run_ids=np.zeros(n_samples, np.int16),
        leadfield=_LeadField(lf),
        ch_names=names,
        ch_positions=pos,
        meta={"unipolar_ref_ch": "Cz"},
    )


def _write_external_cache(tmp_path, label="demo", n_ch=32, n_samples=200):
    """npz externo sintético con las 7 referencias y splits reales."""
    rng = np.random.default_rng(1)
    th = np.linspace(0.25, 1.35, n_ch)
    ph = rng.uniform(0, 2 * np.pi, n_ch)
    pos = 0.092 * np.stack(
        [np.sin(th) * np.cos(ph),
         np.sin(th) * np.sin(ph),
         np.cos(ph)], axis=1)  # ph en z deliberadamente distinto del ancla
    pos[:, 2] = 0.092 * np.cos(th)

    refs = {}
    splits = {"train": np.arange(160), "val": np.arange(160, 180),
              "test": np.arange(180, 200)}
    payload = {
        "split_train": splits["train"],
        "split_val": splits["val"],
        "split_test": splits["test"],
        "subject_ids": np.zeros(n_samples, np.int16),
        "run_ids": np.zeros(n_samples, np.int16),
        "ch_names": np.array([f"E{i}" for i in range(n_ch)], dtype=object),
        "ch_positions": pos,
        "meta": np.array({
            "config": f"v{DATASET_VERSION}",
            "external_label": label,
            "unipolar_ref_ch": "Cz",
        }, dtype=object),
    }
    # MultiReferenceDataset.save escribe ref_<kind> + leadfield aparte;
    # aquí replicamos el formato manualmente para no depender de LeadField.
    kinds = ("unipolar", "linked_mastoids", "linked_ears", "bipolar",
             "car", "rest", "laplacian")
    for k in kinds:
        payload[f"ref_{k}"] = rng.normal(size=(n_samples, n_ch)).astype(np.float32)
    cache_dir = tmp_path / "data" / EXTERNAL_SUBDIR
    cache_dir.mkdir(parents=True)
    path = cache_dir / f"{label}_block_v{DATASET_VERSION}.npz"
    np.savez_compressed(path, **payload)

    # sidecar del lead field (formato de LeadField.save)
    n_src = 12
    src = rng.normal(size=(n_src, 3)) * 0.05
    lf_matrix = rng.normal(size=(n_ch, n_src)).astype(np.float64) * pos.mean()
    np.savez_compressed(
        path.with_name(path.stem + "_leadfield.npz"),
        matrix=lf_matrix,
        src_positions=src * 0 + rng.normal(size=(n_src, 3)) * 0.06,
        ch_names=np.array([f"E{i}" for i in range(n_ch)], dtype=object),
        head_radius=0.092,
        rel_radii=np.array([0.87, 0.90, 0.97, 1.00]),
        sigmas=np.array([0.33, 1.0, 0.0042, 0.33]),
        src_grid_mm=30.0,
        rest_matrix=rng.normal(size=(n_ch, 3 * n_src)).astype(np.float32),
    )
    return path


def _multi_cfg(cache_dir, budget=20):
    return EEGTransformConfig.from_dict({
        "dataset": {"cache_dir": str(cache_dir)},
        "leadfield": {"src_grid_mm": 30.0},
        "model": {"variant": "multi_montage", "latent_dim": 0},
        "mapping": {
            "method": "spline",
            "configs": ["canonical", "external:demo"],
            "multi_max_samples_per_split": budget,
        },
    })


def test_find_external_cache_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_external_cache(tmp_path, "no_existe")


def test_list_bids_recordings_requires_mnebids():
    try:
        import mne_bids  # noqa: F401
        have = True
    except ImportError:
        have = False
    if not have:
        with pytest.raises(ImportError):
            list_bids_recordings("/tmp/inexistente_bids")


def test_external_config_builds_and_balances(tmp_path):
    ds = _make_canonical(tmp_path)
    _write_external_cache(tmp_path)
    cfg = _multi_cfg(tmp_path / "data")

    data = build_multiconfig(ds, cfg, force=True)
    assert "external:demo" in data.order

    canon, ext = data["canonical"], data["external:demo"]
    assert ext.n_channels == 32
    assert canon.n_channels == 42
    # proyección al canónico y lectura de vuelta
    assert ext.projection.shape == (32, 42)
    assert ext.out_map.shape == (42, 32)
    assert ext.surface.shape[0] > 0 and ext.surface.shape[1] == 32

    # balanceo por conteo entre bases (sujetos distintos, mismo presupuesto)
    for split in ("train", "val", "test"):
        n_canon = canon.refs[split]["unipolar"].shape[0]
        n_ext = ext.refs[split]["unipolar"].shape[0]
        # con splits >= presupuesto, el conteo es exactamente el budget
        assert n_canon == n_ext == 20
        assert set(ext.refs[split].keys()) == {
            "unipolar", "linked_mastoids", "linked_ears", "bipolar",
            "car", "rest", "laplacian"}

    # las referencias externas son REALES (no derivadas del ancla): distintas
    assert not np.allclose(canon.refs["test"]["car"][:5, :5],
                           ext.refs["test"]["car"][:5, :5])


def test_external_label_without_cache_raises(tmp_path):
    ds = _make_canonical(tmp_path)
    cfg = _multi_cfg(tmp_path / "data")
    cfg.mapping.configs = ["canonical", "external:faltante"]
    with pytest.raises(FileNotFoundError):
        build_multiconfig(ds, cfg, force=True)


def test_mnebids_roundtrip_small(tmp_path):
    """Ingesta completa sobre un mini-BIDS real (si mne-bids está presente)."""
    pytest.importorskip("mne_bids")
    import mne
    from eeg_transform.config import DataConfig
    from eeg_transform.data.external import build_external_dataset

    # construye un mini-BIDS con un Raw sintético denso (32 ch, montaje estándar)
    bids_root = tmp_path / "bids"
    bids_root.mkdir()
    rng = np.random.default_rng(3)
    info = mne.create_info([f"E{i}" for i in range(32)], sfreq=160.0,
                           ch_types="eeg")
    raw = mne.io.RawArray(rng.normal(size=(32, 16000)) * 20e-6, info,
                          verbose="ERROR")
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="raise")
    bids_path = __import__("mne_bids").BIDSPath(
        subject="01", task="rest", suffix="eeg", datatype="eeg",
        root=str(bids_root), check=False)
    __import__("mne_bids").write_raw_bids(raw, bids_path, verbose="ERROR",
                                          format="EDF", allow_preload=True)

    recs = list_bids_recordings(bids_root)
    assert len(recs) == 1 and recs[0]["subject"] == "01"

    ds, cache_file = build_external_dataset(
        bids_root, label="mini", cache_dir=tmp_path / "cache",
        max_recordings=1, max_samples_per_subject=5000,
    )
    assert cache_file.exists()
    assert ds.refs["laplacian"].shape[0] > 100
    assert {k for k in ds.refs} == {
        "unipolar", "linked_mastoids", "linked_ears", "bipolar", "car",
        "rest", "laplacian"}
