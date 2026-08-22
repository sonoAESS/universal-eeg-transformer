"""Tests del entrenamiento multi-configuración (variante ``multi_montage``).

Cubren lo añadido en `experiments/multi.py`, `models/multi_montage.py` y la
evaluación/`CLI` de la variante:

* ``dense_cap_positions`` (configuraciones densas simuladas).
* ``build_multiconfig`` (shapes P_s/Q_s, refs, lead field propio, balanceo).
* ``MultiMontageAutoencoder`` (predicción intra-configuración, matrices
  ``Q_s·core·P_s``, round-trip de config).
* ``evaluate_multiconfig_routes`` (columnas ``*_ana``) y ``summarize_multiconfig``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eeg_transform.config import EEGTransformConfig, ModelConfig
from eeg_transform.experiments.multi import build_multiconfig, dense_cap_positions
from eeg_transform.mapping import select_subset
from eeg_transform.models.multi_montage import MultiMontageAutoencoder
from eeg_transform.models.universal_transformer import UniversalEEGTransformer
from eeg_transform.evaluation import metrics


def _sphere_positions(n, radius=0.088, seed=0):
    rng = np.random.default_rng(seed)
    th = rng.uniform(0.35, np.pi - 0.35, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    return np.stack(
        [radius * np.sin(th) * np.cos(ph),
         radius * np.sin(th) * np.sin(ph),
         radius * np.cos(th)], axis=1
    )


@dataclass
class _LeadField:
    matrix: np.ndarray


@dataclass
class _FakeDataset:
    """Sustituto mínimo de :class:`MultiReferenceDataset` para los tests."""

    n_channels: int
    ch_names: list[str]
    ch_positions: np.ndarray
    leadfield: _LeadField
    refs: dict
    split_idx: dict
    meta: dict

    @classmethod
    def make(cls, n_samples=60, seed=0):
        from eeg_transform.experiments.montage import MONTAGE_10_10_39

        names = list(MONTAGE_10_10_39)
        n_c = len(names)
        rng = np.random.default_rng(seed)
        n_src = 24  # fuentes de conducción (índices del lead field)
        pos = _sphere_positions(n_c, seed=seed)
        lf = rng.normal(size=(n_c, n_src)).astype(np.float32)
        base = rng.normal(size=(n_samples, n_c)).astype(np.float32)
        refs = {k: base.copy() for k in ("unipolar", "bipolar", "car", "rest")}
        idx = np.arange(n_samples)
        return cls(
            n_channels=n_c, ch_names=names, ch_positions=pos,
            leadfield=_LeadField(lf), refs=refs,
            split_idx={"train": idx[:30], "val": idx[30:45], "test": idx[45:]},
            meta={"unipolar_ref_ch": "Cz"},
        )

def _core(data):
    """Core canónico (autoencoder libre) para los tests de modelo."""
    core = UniversalEEGTransformer(
        data["canonical"].n_channels,
        model_cfg=ModelConfig(latent_dim=32, variant="free"),
    )
    core.ensure_built()
    return core


def _multi_config(budget=40):
    """Config multi-configuración pequeña para los tests.

    El presupuesto por split se deja por debajo del mínimo de validación
    (1000) a propósito: la construcción de datos no depende de ``validate()``
    y así los tests corren en segundos.
    """
    return EEGTransformConfig.from_dict({
        "dataset": {"cache_dir": "/tmp/test_multiconfig_cache"},
        "leadfield": {"src_grid_mm": 30.0},
        "model": {"variant": "multi_montage", "latent_dim": 0},
        "mapping": {
            "method": "spline",
            "configs": ["10-20", "canonical", "dense-8", "dense-16"],
            "multi_max_samples_per_split": budget,
        },
    })


def test_dense_cap_positions_shape_norm_and_ordering():
    for n in (16, 128):
        pos = dense_cap_positions(n, radius=0.088)
        assert pos.shape == (n, 3)
        norms = np.linalg.norm(pos, axis=1)
        assert np.allclose(norms, 0.088, atol=1e-6)
        # todos dentro del casquete superior (z >= -0.06 · radius)
        assert (pos[:, 2] >= -0.06 * 0.088 - 1e-9).all()
        # ordenados por ángulo azimutal (cadena bipolar continua)
        az = np.arctan2(pos[:, 1], pos[:, 0])
        assert np.allclose(az, np.sort(az))


def test_build_multiconfig_shapes_refs_and_balance():
    ds = _FakeDataset.make()
    cfg = _multi_config()
    data = build_multiconfig(ds, cfg, force=True)
    assert list(data.order) == ["10-20", "canonical", "dense-8", "dense-16"]
    # balanceo por construcción: mismo presupuesto por split en todas las
    # configuraciones (submuestreo determinista acotado al split).
    for split in ("train", "val", "test"):
        budgets = {l: data.n_budget(split) for l in data.order}
        assert len(set(budgets.values())) == 1
    canon = data["canonical"]
    assert canon.n_channels == len(ds.ch_names)        # 42 (montaje 10-10)
    assert canon.projection.shape == (42, 42)
    assert canon.out_map.shape == (42, 42)
    s2020 = data["10-20"]
    assert s2020.n_channels == 19
    assert s2020.projection.shape == (19, 42)
    assert s2020.out_map.shape == (42, 19)
    for label in ("dense-8", "dense-16"):
        mc = data[label]
        n = mc.n_channels
        assert mc.projection.shape == (n, 42)
        assert mc.out_map.shape == (42, n)
        assert mc.leadfield.shape == (n, mc.leadfield.shape[1])
    # refs por split/kind con (n_budget, C_s)
    for label in data.order:
        n_c = data[label].n_channels
        for split in ("train", "val", "test"):
            n_b = data.n_budget(split)
            for kind in ("unipolar", "bipolar", "car", "rest"):
                assert data[label].refs[split][kind].shape == (n_b, n_c)
    # el subconjunto es una selección exacta de columnas del canónico
    keep, _, src_idx = select_subset(ds.ch_names, ds.ch_positions,
                                     keep=__import__(
                                         "eeg_transform.experiments.multi",
                                         fromlist=["SUBSET_MONTAGES"]).SUBSET_MONTAGES["10-20"])
    assert src_idx.shape[0] == 19


def test_dense_refs_match_canonical_up_to_interpolation():
    ds = _FakeDataset.make()
    cfg = _multi_config()
    data = build_multiconfig(ds, cfg, force=True)
    canon_rest = data["canonical"].refs["test"]["rest"]      # (40, 39)
    d = data["dense-8"].refs["test"]["rest"]                 # (40, 8)
    # REST denso ≈ canónico @ Q_s (interpolación al denso) @ T_rest_denso
    from eeg_transform.references import build_reference_matrix

    mc = data["dense-8"]
    t_rest = build_reference_matrix(
        "rest", mc.n_channels,
        unipolar_ref_index=mc.unipolar_ref_index,
        lead_field=mc.leadfield,
    )
    est = canon_rest @ mc.out_map @ t_rest
    assert d.shape == est.shape
    assert np.corrcoef(d.ravel(), est.ravel())[0, 1] > 0.8


def test_multimontage_forward_intra_config_shapes():
    cfg = _multi_config()
    ds = _FakeDataset.make()
    data = build_multiconfig(ds, cfg, force=True)
    core = _core(data)

    core.ensure_built()
    model = MultiMontageAutoencoder(
        core=core,
        projections={l: m.projection for l, m in data.configs.items()},
        out_maps={l: m.out_map for l, m in data.configs.items()},
    )
    x = {l: {k: data[l].refs["test"][k][:4] for k in ("unipolar", "bipolar",
                                                       "car", "rest")}
         for l in data.order}
    preds = model(x)  # call completo (todos los configs)
    for l in data.order:
        n_c = data[l].n_channels
        assert set(preds[l]) == set(("unipolar", "bipolar", "car", "rest"))
        for s in data[l].refs["test"]:
            for d, t in preds[l][s].items():
                assert t.shape == (4, n_c)   # intra-configuración
    # llamada puntual
    single = model(np.zeros((3, 19), dtype=np.float32), cfg="10-20", source="rest")
    assert all(v.shape == (3, 19) for v in single.values())
    # round-trip de configuración (get_config/from_config)
    restored = MultiMontageAutoencoder.from_config(model.get_config())
    assert list(restored.configs) == list(model.configs)


def test_multimontage_transfer_matrices_shape():
    cfg = _multi_config()
    ds = _FakeDataset.make()
    data = build_multiconfig(ds, cfg, force=True)
    core = _core(data)

    core.ensure_built()
    model = MultiMontageAutoencoder(
        core=core,
        projections={l: m.projection for l, m in data.configs.items()},
        out_maps={l: m.out_map for l, m in data.configs.items()},
    )
    mats = model.transfer_matrices("10-20")
    assert set(mats) == {("unipolar", "unipolar"), ("unipolar", "bipolar"),
                         ("unipolar", "car"), ("unipolar", "rest"),
                         ("bipolar", "unipolar"), ("bipolar", "bipolar"),
                         ("bipolar", "car"), ("bipolar", "rest"),
                         ("car", "unipolar"), ("car", "bipolar"),
                         ("car", "car"), ("car", "rest"),
                         ("rest", "unipolar"), ("rest", "bipolar"),
                         ("rest", "car"), ("rest", "rest")}
    assert all(m.shape == (19, 19) for m in mats.values())


def test_evaluate_multiconfig_routes_has_analytic_baseline():
    cfg = _multi_config()
    ds = _FakeDataset.make()
    data = build_multiconfig(ds, cfg, force=True)
    core = _core(data)

    core.ensure_built()
    model = MultiMontageAutoencoder(
        core=core,
        projections={l: m.projection for l, m in data.configs.items()},
        out_maps={l: m.out_map for l, m in data.configs.items()},
    )
    rules = metrics.evaluate_multiconfig_routes(model, data, split="test")
    assert len(rules) == len(data.order) * 16
    assert sorted(rules.columns) == sorted(
        ["config", "origen", "destino", "mse", "rmse", "mae", "r", "ve",
         "rmse_ana", "r_ana", "ve_ana"])
    summ = metrics.summarize_multiconfig(rules)
    assert list(summ.columns) == [
        "config", "n_rutas", "rmse_diag_uV", "r_diag", "rmse_cross_uV",
        "r_cross", "ve_cross", "rmse_ana_cross_uV", "r_ana_cross",
        "ve_ana_cross"]
    assert set(summ["config"]) == set(data.order)


def test_multi_variant_config_validation():
    import pytest

    bad = dict(model={"variant": "multi_montage", "latent_dim": 0},
               mapping={"configs": ["10-20", "canonical", "unknown-5"]})
    cfg = EEGTransformConfig.from_dict(bad)
    with pytest.raises(ValueError):
        cfg.validate()

    bad_budget = dict(model={"variant": "multi_montage", "latent_dim": 0},
                      mapping={"multi_max_samples_per_split": 5})
    cfg = EEGTransformConfig.from_dict(bad_budget)
    with pytest.raises(ValueError):
        cfg.validate()

    bad_latent = dict(model={"variant": "multi_montage", "latent_dim": 32})
    cfg = EEGTransformConfig.from_dict(bad_latent)
    with pytest.raises(ValueError):
        cfg.validate()


def test_multi_heatmap_model_surface_loss_and_config():
    """Variante ``multi_heatmap``: campo en la malla + pérdida de superficie."""
    import tensorflow as tf

    from eeg_transform.models.multi_heatmap import MultiHeatmapAutoencoder
    from eeg_transform.config import MODEL_VARIANTS

    assert "multi_heatmap" in MODEL_VARIANTS
    cfg = _multi_config()  # multi_montage en 'model', se anula a mano
    cfg.model.variant = "multi_heatmap"
    ds = _FakeDataset.make()
    data = build_multiconfig(ds, cfg, force=True)
    # mapas de superficie por configuración (malla compartida, n_grid = grid_px²)
    for l in data.order:
        n_c = data[l].n_channels
        assert data[l].surface is not None
        assert data[l].surface.shape == (data[l].surface.shape[0], n_c)
    grid_px2 = {l: data[l].surface.shape[0] for l in data.order}
    assert len(set(grid_px2.values())) == 1  # una sola malla compartida

    core = _core(data)
    core.ensure_built()
    model = MultiHeatmapAutoencoder(
        core=core,
        projections={l: m.projection for l, m in data.configs.items()},
        out_maps={l: m.out_map for l, m in data.configs.items()},
        surfaces={l: m.surface for l, m in data.configs.items()},
        surface_loss_weight=0.1,
    )
    model.compile(optimizer="adam")
    # una rutina de entrenamiento con el término de superficie en las métricas
    x = {l: {k: data[l].refs["train"][k] for k in data[l].refs["train"]}
         for l in data.order}
    out = model.train_step((x, x))
    for key in ("loss", "loss_estandarizada", "surface_loss", "mse_real_V2"):
        assert key in out
        assert float(out[key]) == float(out[key])  # es finito
    # round-trip de configuración con surfaces y peso
    restored = MultiHeatmapAutoencoder.from_config(model.get_config())
    assert float(restored.surface_loss_weight) == 0.1
    assert restored.surfaces["canonical"].shape == model.surfaces["canonical"].shape


def test_evaluate_multiconfig_surface_routes():
    from eeg_transform.evaluation import metrics as mtr

    cfg = _multi_config()
    cfg.model.variant = "multi_heatmap"
    ds = _FakeDataset.make()
    data = build_multiconfig(ds, cfg, force=True)
    core = _core(data)
    core.ensure_built()
    model = MultiMontageAutoencoder(
        core=core,
        projections={l: m.projection for l, m in data.configs.items()},
        out_maps={l: m.out_map for l, m in data.configs.items()},
    )
    surface = mtr.evaluate_multiconfig_surface_routes(model, data, split="test")
    assert len(surface) == len(data.order) * 16
    assert sorted(surface.columns) == sorted(
        ["config", "origen", "destino", "rmse_field", "r_field", "ve_field"])
    summ = mtr.summarize_multiconfig_surface(surface)
    assert set(summ["config"]) == set(data.order)
    assert "ve_field_cross" in summ.columns and "rmse_field_cross_uV" in summ.columns


def test_multi_heatmap_config_validation():
    import pytest

    # exige method == spline (campo de superficie sobre la malla)
    bad_method = dict(model={"variant": "multi_heatmap", "latent_dim": 0},
                      mapping={"configs": ["10-20", "canonical", "dense-8"],
                               "method": "leadfield"})
    cfg = EEGTransformConfig.from_dict(bad_method)
    with pytest.raises(ValueError):
        cfg.validate()

    # peso de superficie fuera de rango
    bad_weight = dict(model={"variant": "multi_heatmap", "latent_dim": 0,
                             "surface_loss_weight": 5.0},
                      mapping={"configs": ["10-20", "canonical", "dense-8"]})
    cfg = EEGTransformConfig.from_dict(bad_weight)
    with pytest.raises(ValueError):
        cfg.validate()

    # latente debe ser 0/auto
    bad_latent = dict(model={"variant": "multi_heatmap", "latent_dim": 16},
                      mapping={"configs": ["10-20", "canonical", "dense-8"]})
    cfg = EEGTransformConfig.from_dict(bad_latent)
    with pytest.raises(ValueError):
        cfg.validate()


def test_multi_heatmap_v2_config_validation():
    import pytest

    # v2 válida con method == spline y pesos por defecto
    ok = dict(model={"variant": "multi_heatmap_v2", "latent_dim": 0,
                     "learnable_interp": True,
                     "field_consistency_weight": 0.1,
                     "xconfig_consistency_weight": 0.1,
                     "adapter_rank": 2},
              mapping={"configs": ["10-20", "canonical", "dense-8"],
                       "method": "spline"})
    cfg = EEGTransformConfig.from_dict(ok)
    cfg.validate()  # no debe lanzar

    # adapter_rank negativo -> inválido
    bad_adapter = dict(model={"variant": "multi_heatmap_v2", "latent_dim": 0,
                              "adapter_rank": -1},
                       mapping={"configs": ["10-20", "canonical", "dense-8"],
                                "method": "spline"})
    cfg = EEGTransformConfig.from_dict(bad_adapter)
    with pytest.raises(ValueError):
        cfg.validate()

    # exige method == spline
    bad_method = dict(model={"variant": "multi_heatmap_v2", "latent_dim": 0},
                      mapping={"configs": ["10-20", "canonical", "dense-8"],
                               "method": "leadfield"})
    cfg = EEGTransformConfig.from_dict(bad_method)
    with pytest.raises(ValueError):
        cfg.validate()


def test_multi_heatmap_v2_model_improvements():
    """v2: campo aprendible, consistencias, adaptadores, incertidumbre."""
    import tensorflow as tf

    from eeg_transform.models.multi_heatmap import MultiHeatmapAutoencoder

    cfg = _multi_config()
    cfg.model.variant = "multi_heatmap_v2"
    ds = _FakeDataset.make()
    data = build_multiconfig(ds, cfg, force=True)
    core = _core(data)
    core.ensure_built()
    model = MultiHeatmapAutoencoder(
        core=core,
        projections={l: m.projection for l, m in data.configs.items()},
        out_maps={l: m.out_map for l, m in data.configs.items()},
        surfaces={l: m.surface for l, m in data.configs.items()},
        surface_loss_weight=0.1,
        learnable_interp=True,
        field_consistency_weight=0.1,
        xconfig_consistency_weight=0.1,
        adapter_rank=2,
        temporal_smoothness_weight=0.0,
        learn_uncertainty=False,
    )
    model.compile(optimizer="adam")
    # R_s aprendible inicializado en S_s (mismo shape que la superficie)
    for l in data.order:
        assert model.params.R_s[l].shape == model.surfaces[l].shape
    # adaptadores por configuración
    assert set(model.params.adapters.keys()) == set(data.order)
    for l in data.order:
        u, v = model.params.adapters[l]
        assert u.shape == (data["canonical"].n_channels, 2)
        assert v.shape == (2, data["canonical"].n_channels)
    x = {l: {k: data[l].refs["train"][k] for k in data[l].refs["train"]}
         for l in data.order}
    out = model.train_step((x, x))
    for key in ("loss", "loss_estandarizada", "surface_loss",
                "field_consist_loss", "xconfig_loss", "temporal_loss",
                "mse_real_V2"):
        assert key in out
        assert float(out[key]) == float(out[key])  # finito
    out_val = model.test_step((x, x))
    assert float(out_val["loss"]) == float(out_val["loss"])

    # round-trip de configuración preserva los nuevos parámetros
    restored = MultiHeatmapAutoencoder.from_config(model.get_config())
    assert restored.learnable_interp is True
    assert float(restored.field_consistency_weight) == 0.1
    assert float(restored.xconfig_consistency_weight) == 0.1
    assert restored.adapter_rank == 2
    assert list(restored.params.adapters.keys()) == list(data.order)


def test_multi_heatmap_v2_learnable_interp_changes_with_training():
    """El campo aprendible R_s se aleja de la spline fija tras entrenar."""
    import tensorflow as tf

    from eeg_transform.models.multi_heatmap import MultiHeatmapAutoencoder

    cfg = _multi_config()
    cfg.model.variant = "multi_heatmap_v2"
    ds = _FakeDataset.make()
    data = build_multiconfig(ds, cfg, force=True)
    core = _core(data)
    core.ensure_built()
    model = MultiHeatmapAutoencoder(
        core=core,
        projections={l: m.projection for l, m in data.configs.items()},
        out_maps={l: m.out_map for l, m in data.configs.items()},
        surfaces={l: m.surface for l, m in data.configs.items()},
        learnable_interp=True,
    )
    model.compile(optimizer="adam")
    before = {l: model.params.R_s[l].numpy().copy() for l in data.order}
    x = {l: {k: data[l].refs["train"][k] for k in data[l].refs["train"]}
         for l in data.order}
    for _ in range(3):
        model.train_step((x, x))
    for l in data.order:
        delta = np.linalg.norm(model.params.R_s[l].numpy() - before[l])
        assert delta > 1e-6  # R_s evolucionó respecto a la inicialización


def test_multi_heatmap_v2_uncertainty_trainable():
    import tensorflow as tf

    from eeg_transform.models.multi_heatmap import MultiHeatmapAutoencoder

    cfg = _multi_config()
    cfg.model.variant = "multi_heatmap_v2"
    ds = _FakeDataset.make()
    data = build_multiconfig(ds, cfg, force=True)
    core = _core(data)
    core.ensure_built()
    model = MultiHeatmapAutoencoder(
        core=core,
        projections={l: m.projection for l, m in data.configs.items()},
        out_maps={l: m.out_map for l, m in data.configs.items()},
        surfaces={l: m.surface for l, m in data.configs.items()},
        learn_uncertainty=True,
    )
    model.compile(optimizer="adam")
    assert set(model.log_vars.keys()) == {"elec", "surf", "field", "xconfig", "temp"}
    assert any(v.trainable for v in model.log_vars.values())
    x = {l: {k: data[l].refs["train"][k] for k in data[l].refs["train"]}
         for l in data.order}
    out = model.train_step((x, x))
    assert float(out["loss"]) == float(out["loss"])


def test_multi_heatmap_v2_field_agreement_metric():
    """evaluate_multiconfig_field_agreement devuelve acuerdo pareado."""
    from eeg_transform.evaluation import metrics as mtr
    from eeg_transform.models.multi_heatmap import MultiHeatmapAutoencoder

    cfg = _multi_config()
    cfg.model.variant = "multi_heatmap_v2"
    ds = _FakeDataset.make()
    data = build_multiconfig(ds, cfg, force=True)
    core = _core(data)
    core.ensure_built()
    model = MultiHeatmapAutoencoder(
        core=core,
        projections={l: m.projection for l, m in data.configs.items()},
        out_maps={l: m.out_map for l, m in data.configs.items()},
        surfaces={l: m.surface for l, m in data.configs.items()},
    )
    fa = mtr.evaluate_multiconfig_field_agreement(model, data, split="test")
    n_pairs = len(data.order) * (len(data.order) - 1) // 2
    assert len(fa) == n_pairs * 16  # rutas s->d
    assert sorted(fa.columns) == sorted(
        ["origen", "destino", "config_a", "config_b",
         "rmse_field", "r_field", "ve_field"])


def test_external_topomap_loader_raises():
    from eeg_transform.evaluation import metrics as mtr

    import pytest

    with pytest.raises(NotImplementedError):
        mtr.external_topomap_loader("/tmp/does_not_exist")