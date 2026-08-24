"""Tests de la unificación de montajes: proyección, malla y modo montaje.

Cubren lo añadido en `mapping.py` y el modo montaje del modelo/servicios:

* ``scalp_grid_matrix`` (malla 2D del cuero cabelludo / heatmap de manchas).
* ``density_smoothness`` y ``build_projection`` (suavizado por densidad).
* ``UniversalEEGTransformer`` en modo montaje (proyección fija, matrices de
  transferencia ``C_s x C``, sin composición).
* ``build_tf_dataset`` en modo montaje (fuente/objetivo) y validación de la
  config de variantes ``montage_*``.
"""

from __future__ import annotations

import numpy as np

from eeg_transform.config import EEGTransformConfig, MappingConfig, ModelConfig, TrainingConfig, DatasetConfig
from eeg_transform.mapping import (
    build_projection,
    density_smoothness,
    scalp_grid_matrix,
    spherical_spline_matrix,
)
from eeg_transform.models.universal_transformer import UniversalEEGTransformer


def _sphere_positions(n, seed=0, radius=0.09):
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, np.pi, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    return np.stack(
        [radius * np.sin(th) * np.cos(ph),
         radius * np.sin(th) * np.sin(ph),
         radius * np.cos(th)], axis=1
    )


def _montage_model(n_canon=8, n_src=4):
    rng = np.random.default_rng(0)
    proj = rng.normal(size=(n_src, n_canon)).astype(np.float32)
    feed = _sphere_positions(n_src, seed=1)
    g_src = rng.normal(size=(n_src, 6)).astype(np.float32)
    g_dst = rng.normal(size=(n_canon, 6)).astype(np.float32)
    return proj, feed, g_src, g_dst


def test_scalp_grid_matrix_shape_and_mask():
    src = _sphere_positions(19, seed=0)
    M, valid, centroids = scalp_grid_matrix(src, grid_px=8)
    n = 8 * 8
    assert M.shape == (n, 19)
    assert valid.shape == (n,)
    assert valid.sum() < n                     # fuera del disco queda vacío
    assert M[~valid].sum() == 0.0              # nodos fuera de la malla a cero
    assert np.isfinite(M[valid]).all()
    # los centros cubren [-1, 1]
    assert centroids.shape == (n, 2)
    assert centroids.max() <= 1.0 and centroids.min() >= -1.0


def test_density_smoothness_grows_with_sparsity():
    src_dense = _sphere_positions(64, seed=0)
    src_sparse = _sphere_positions(8, seed=1)
    dst = _sphere_positions(64, seed=2)
    s_dense = density_smoothness(src_dense, dst, base_smoothness=1e-5)
    s_sparse = density_smoothness(src_sparse, dst, base_smoothness=1e-5)
    assert s_dense == 1e-5                      # C_s == C_dst: sin escala
    assert s_sparse > s_dense                   # más disperso → más suavizado


def test_build_projection_spline_adaptive_smoothness():
    src = _sphere_positions(19, seed=3)
    dst = _sphere_positions(64, seed=4)
    P_fijo = build_projection("spline", src, dst, smoothness=1e-5,
                              adaptive_smoothness=False)
    P_adapt = build_projection("spline", src, dst, smoothness=1e-5,
                               adaptive_smoothness=True)
    P_base = build_projection("spline", src, dst)
    assert P_fijo.shape == (19, 64) and P_adapt.shape == (19, 64)
    assert not np.allclose(P_fijo, P_adapt)     # el suavizado adaptativo cambia
    assert np.allclose(P_adapt, P_base)         # derefault = adaptativo


def test_montage_model_forward_and_transfer_shapes():
    proj, _, _, _ = _montage_model()
    model = UniversalEEGTransformer(8, projection=proj)
    model.ensure_built()
    x = np.zeros((5, 4), dtype=np.float32)
    out = model(x, source="rest")
    assert out["car"].shape == (5, 8)
    T = model.transfer_matrices()
    assert T[("unipolar", "rest")].shape == (4, 8)   # C_s x C
    assert model.composition_error() == {}            # sin grupo en montaje


def test_montage_variant_config_requires_matching_method():
    bad = dict(
        model={"variant": "montage_leadfield", "latent_dim": 0},
        mapping={"method": "spline"},
    )
    cfg = EEGTransformConfig.from_dict(bad)
    try:
        cfg.validate()
        raise AssertionError("debió fallar: montage_leadfield no admite spline")
    except ValueError as exc:
        assert "montage_leadfield" in str(exc)


def test_build_tf_dataset_montage_sources_targets():
    import tensorflow as tf

    from eeg_transform.training.trainer import build_tf_dataset

    n, c, cs = 12, 8, 4
    rng = np.random.default_rng(0)

    class _DS:
        pass

    ds = _DS()
    from eeg_transform.config import REFERENCE_KINDS as _RKS

    ds.refs = {k: rng.normal(size=(n, c)).astype(np.float32) for k in _RKS}
    ds.split_idx = {"train": np.arange(n), "val": np.arange(n),
                    "test": np.arange(n)}
    sources = {k: rng.normal(size=(n, cs)).astype(np.float32) for k in _RKS}
    dset = build_tf_dataset(ds, "train", batch_size=4,
                            sources={"train": sources})
    batch = next(iter(dset))
    src, tgt = batch
    assert list(src.keys()) == list(tgt.keys())
    assert src["rest"].shape == (4, cs)
    assert tgt["rest"].shape == (4, c)


def test_montage_config_dict_roundtrip():
    cfg = EEGTransformConfig.from_dict({
        "model": {"variant": "montage_heatmap", "latent_dim": 0},
        "mapping": {"method": "spline", "montage": "10-20", "grid_px": 32},
    })
    cfg.validate()
    assert cfg.mapping == MappingConfig(method="spline", montage="10-20",
                                        grid_px=32)
    assert cfg.model.variant == "montage_heatmap"