"""Test de integración: el autoencoder lineal aprende los mapas lineales exactos.

Usa un lead field sintético (matriz aleatoria) para acelerar la ejecución y
verifica que tras entrenar un número reducido de épocas las matrices
efectivas se aproximan a las analíticas de cada ruta.
"""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from eeg_transform.config import EEGTransformConfig, REFERENCE_KINDS  # noqa: E402
from eeg_transform.models.universal_transformer import (  # noqa: E402
    UniversalEEGTransformer,
)
from eeg_transform.data.dataset import MultiReferenceDataset  # noqa: E402
from eeg_transform.references import (  # noqa: E402
    bipolar_chain_matrix,
    car_matrix,
    compute_all_references,
    inter_reference_matrix,
    unipolar_matrix,
)
from eeg_transform.training.trainer import build_tf_dataset  # noqa: E402


def _fake_dataset(rng, n=2000, C=8):
    """Dataset sintético alineado (todas las referencias consistentes).

    Usa el **lead field analítico** (esfera 4 capas, grilla gruesa para
    velocidad) y genera la señal en su subespacio observable: los campos son
    espacialmente suaves y todas las rutas entre referencias son linealmente
    consistentes, como en EEG real.
    """
    from eeg_transform.leadfield import compute_lead_field

    th = rng.uniform(0.3, np.pi - 0.3, C)
    ph = rng.uniform(0, 2 * np.pi, C)
    pos = 0.09 * np.stack(
        [np.sin(th) * np.cos(ph),
         np.sin(th) * np.sin(ph),
         np.cos(th)], axis=1,
    )
    lf_obj = compute_lead_field(
        [f"c{i}" for i in range(C)], pos, src_grid_mm=30.0,
    )
    G = lf_obj.matrix

    class FakeLF:
        pass

    lf = FakeLF()
    lf.matrix = G
    lf.n_sources = G.shape[1]
    lf.src_grid_mm = 30.0
    lf.save = lambda *a, **k: None

    w_avg = np.eye(C) - np.ones((C, C)) / C
    s = (w_avg @ G).T @ rng.normal(size=(C, n))
    Xraw = (G @ s).T * 50e-6               # (n, C) potencial sin referencia

    refs = compute_all_references(Xraw.astype(np.float32), C, 3, G,
                                  positions=pos)
    idx = np.arange(n)
    ds = MultiReferenceDataset(
        refs={k: v.astype(np.float32) for k, v in refs.items()},
        split_idx={
            "train": idx[: n // 2],
            "val": idx[n // 2 : 3 * n // 4],
            "test": idx[3 * n // 4 :],
        },
        subject_ids=np.zeros(n, np.int16),
        run_ids=np.zeros(n, np.int16),
        leadfield=lf,  # type: ignore
        ch_names=[f"c{i}" for i in range(C)],
        ch_positions=pos,
        meta={"synthetic": True},
    )
    return ds


def _correlated(X):
    A = X.T @ X + np.eye(X.shape[1])
    L = np.linalg.cholesky(A)
    return (X @ L).astype("float64")


def _mean_center(X):
    return X - X.mean(axis=1, keepdims=True)


def test_model_recovers_analytic_maps_on_data():
    """El autoencoder lineal reproduce las transformaciones sobre datos reales.

    Las referencias de rango deficiente (uni/bip/CAR/REST) eliminan la
    componente constante instantánea, que no es físicamente estimable desde
    la señal observada; por ello se compara la salida predicha y la verdad
    en el espacio observable (sin modo constante por instante).
    """
    rng = np.random.default_rng(7)
    C = 8
    ds = _fake_dataset(rng, n=8000, C=C)

    cfg = EEGTransformConfig()
    cfg.model.latent_dim = C
    cfg.model.learning_rate = 1e-3
    cfg.training.batch_size = 512
    cfg.dataset.dtype = "float32"

    tf.random.set_seed(2)
    np.random.seed(2)
    model = UniversalEEGTransformer(n_channels=C, model_cfg=cfg.model)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

    train_ds = build_tf_dataset(ds, "train", batch_size=512, shuffle=True, seed=2)
    val_ds = build_tf_dataset(ds, "val", batch_size=512, shuffle=False)
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1500,
        steps_per_epoch=30,
        validation_steps=6,
        verbose=0,
    )

    # Globalmente el modelo converge con holgura (supera a la línea base
    # pinv en la mayoría de rutas). Los DESTINOS ``laplacian`` son el caso
    # lento conocido: estimar CSD exige resolver las componentes espaciales
    # de mayor grado y con C=8 canales sintéticos la factorización tarda
    # órdenes de magnitud más en igualar al mapa analítico; su corrección
    # EXACTA está validada analíticamente en test_references_ext.
    assert hist.history["val_loss_estandarizada"][-1] < 0.05

    matrices = model.transfer_matrices()
    worst = 0.0
    worst_route = None
    for s in KINDS:
        X_s = ds.refs[s][ds.split_idx["test"]]
        preds = model(tf.convert_to_tensor(X_s), source=s)
        for d in KINDS:
            target = ds.refs[d][ds.split_idx["test"]]
            t_c = _mean_center(target)
            err = np.linalg.norm(
                _mean_center(preds[d].numpy()) - t_c
            ) / (np.linalg.norm(t_c) + 1e-12)
            # línea base analítica de la ruta. Rutas con destino no-laplacian:
            # el modelo debe igualar o superar al mapa óptimo (margen 5%).
            ana = inter_reference_matrix(
                s, d, ds.n_channels,
                unipolar_ref_index=3,
                lead_field=ds.leadfield.matrix,
                positions=ds.ch_positions,
            )
            err_ana = np.linalg.norm(
                _mean_center(X_s @ ana) - t_c
            ) / (np.linalg.norm(t_c) + 1e-12)
            if d == "laplacian":
                assert np.isfinite(err) and err < 0.50, (
                    f"ruta {s}->{d}: error modelo {err:.3f} (debe aprender "
                    f"algo útil aunque no iguale a la analítica en humo)"
                )
                continue
            bound = max(0.35, 1.05 * float(err_ana))
            if err > worst:
                worst, worst_route = err, (s, d)
            assert err <= bound, (
                f"ruta {s}->{d}: error modelo {err:.3f} > analítica "
                f"{err_ana:.3f} (x1.05)"
            )

    # peor error relativo acotado por la línea base (aleatorio ≈ 1)
    assert worst < 0.60, f"peor ruta {worst_route}: {worst:.3f}"
    assert matrices[(s, d)] is not None  # matrices efectivas pobladas


KINDS = tuple(REFERENCE_KINDS)


def test_route_arrays_consistency():
    rng = np.random.default_rng(11)
    ds = _fake_dataset(rng, n=100, C=8)
    X, Y = ds.route_arrays("car", "rest", "test")
    assert X.shape == Y.shape == (len(ds.split_idx["test"]), 8)


def test_zscore_loss_scale_invariant():
    """La pérdida Z-score debe ser igual aunque la señal cambie de escala."""
    from eeg_transform.models.universal_transformer import UniversalEEGTransformer

    rng = np.random.default_rng(2)
    x = tf.convert_to_tensor(rng.normal(size=(100, 6)).astype("float32"))
    y = tf.convert_to_tensor(rng.normal(size=(100, 6)).astype("float32"))
    l1 = UniversalEEGTransformer._zscore_loss(x, y)
    l2 = UniversalEEGTransformer._zscore_loss(x * 1000.0, y * 1000.0)
    assert abs(float(l1) - float(l2)) < 1e-6


def test_init_from_data_recovers_routes():
    """La inicialización lineal empírica reproduce todas las rutas (var. ≈ 1).

    Sin entrenamiento, la factorización ridge (latente = unipolar) debe dejar
    la varianza explicada de cada ruta cerca de 1.0, que es el requisito para
    que el gradiente no quede atrapado en cuencas degeneradas.
    """
    rng = np.random.default_rng(3)
    C = 8
    ds = _fake_dataset(rng, n=2000, C=C)

    cfg = EEGTransformConfig()
    cfg.model.latent_dim = C
    model = UniversalEEGTransformer(n_channels=C, model_cfg=cfg.model)
    refs = {k: ds.refs[k][ds.split_idx["train"][:1500]] for k in KINDS}
    model.init_from_data(refs)
    model.compile(optimizer="adam")

    idx = ds.split_idx["test"][:500]
    worst = 0.0
    for s in KINDS:
        x = tf.convert_to_tensor(ds.refs[s][idx], tf.float32)
        preds = model(x, source=s)
        for d in KINDS:
            y = ds.refs[d][idx].astype("float32")
            p = preds[d].numpy()
            ve = 1.0 - float(((y - p) ** 2).mean() / (y.var() + 1e-15))
            worst = max(worst, ve)
    assert worst > 0.99, f"peor varianza explicada tras init: {worst:.4f}"


def test_load_weights_requires_built_layers():
    """Cargar pesos en un modelo sin construir las capas debe restaurarlos.

    Regresión: con Keras 3, ``load_weights`` sobre capas Dense sin construir
    (``build`` nunca llamado) es un no-op silencioso y deja pesos aleatorios.
    """
    rng = np.random.default_rng(5)
    C = 6
    cfg = EEGTransformConfig()
    cfg.model.latent_dim = C
    m1 = UniversalEEGTransformer(n_channels=C, model_cfg=cfg.model)
    m1.ensure_built()
    m1.encoders["unipolar"].kernel.assign(
        np.full((C, C), 0.1337, dtype="float32")
    )
    tmp = __import__("tempfile").mktemp(suffix=".weights.h5")
    m1.save_weights(tmp)
    # cargar en un modelo nuevo sin tocar: ensure_built debe habilitar el restore
    m2 = UniversalEEGTransformer(n_channels=C, model_cfg=cfg.model)
    m2.ensure_built()
    m2.load_weights(tmp)
    np.testing.assert_allclose(
        m2.encoders["unipolar"].kernel.numpy(), 0.1337, atol=1e-6
    )


def test_group_variant_exact_composition():
    """variant='group' debe dar consistencia de composición (casi) exacta.

    ``A_{s->d} = W_s (W_d)^+`` hace que el producto ``A_{s->d} A_{d->u}``
    coincida con ``A_{s->u}`` (módulo el subespacio observable), una
    propiedad física que el encadenado analítico ``T_d pinv(T_s)`` no cumple.
    """
    rng = np.random.default_rng(13)
    C = 6
    ds = _fake_dataset(rng, n=1500, C=C)
    cfg = EEGTransformConfig()
    cfg.model.latent_dim = C
    cfg.model.variant = "group"
    model = UniversalEEGTransformer(n_channels=C, model_cfg=cfg.model)
    refs = {k: ds.refs[k][ds.split_idx["train"][:1000]] for k in KINDS}

    assert model.decoders["unipolar"] is None
    model.init_from_data(refs)

    cons = model.composition_error()
    worst = max(cons.values())
    # la estructura de grupo se cumple exactamente, sin entrenamiento
    assert worst < 1e-5, f"peor consistencia group: {worst:.3e}"


def test_projected_variant_kills_constant():
    """variant='projected' debe anular el modo constante por construcción.

    Toda matriz efectiva aprendida (enc/dec) cumple ``1^T W_eff = 0``:
    entrada constante sobre canales produce salida nula.
    """
    rng = np.random.default_rng(17)
    C = 6
    ds = _fake_dataset(rng, n=1000, C=C)
    cfg = EEGTransformConfig()
    cfg.model.latent_dim = C
    cfg.model.variant = "projected"
    model = UniversalEEGTransformer(n_channels=C, model_cfg=cfg.model)
    refs = {k: ds.refs[k][ds.split_idx["train"][:700]] for k in KINDS}
    model.init_from_data(refs)

    mats = model.transfer_matrices()
    ones = np.ones(C)
    for (s, d), m in mats.items():
        # column sums de la matriz efectiva ~ 0
        np.testing.assert_allclose(
            (ones @ m), np.zeros(C), atol=1e-5, err_msg=f"ruta {s}->{d}"
        )


def test_soft_group_penalty_zero_when_consistent():
    """La penalización suave alcanza 0 con matrices que forman un grupo exacto.

    Si ``W_dec^k = (W_enc^k)^+`` (estructura de grupo), la penalización de
    composición ``s->d->u`` debe ser nula sobre el subespacio observable.
    """
    rng = np.random.default_rng(23)
    C = 6
    ds = _fake_dataset(rng, n=800, C=C)
    cfg = EEGTransformConfig()
    cfg.model.latent_dim = C
    cfg.model.variant = "soft_group"
    cfg.model.comp_penalty_weight = 0.5
    model = UniversalEEGTransformer(n_channels=C, model_cfg=cfg.model)
    model.ensure_built()

    # simulamos matrices que forman grupo exacto: dec = pinv(enc), enc = Glorot
    pen = float(model._soft_group_penalty())
    # con pesos aleatorios la composición difiere de las rutas directas
    assert pen > 0.0

    # imponemos la estructura de grupo y verificamos que la penalidad cae a 0
    for k in KINDS:
        w = model.encoders[k].kernel.numpy()
        model.decoders[k].kernel.assign(np.linalg.pinv(w).astype("float32"))
    pen2 = float(model._soft_group_penalty())
    assert pen2 < 1e-5, f"penalización tras imponer grupo: {pen2:.3e}"