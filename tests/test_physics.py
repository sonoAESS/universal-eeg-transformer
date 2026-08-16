"""Tests del núcleo físico-matemático: matrices de referencia y lead field."""

from __future__ import annotations

import numpy as np

from eeg_transform.leadfield import compute_lead_field
from eeg_transform.references import (
    bipolar_chain_matrix,
    car_matrix,
    rest_matrix,
    unipolar_matrix,
)


def _sphere_positions(n, seed=0, radius=0.09):
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, np.pi, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    return np.stack(
        [radius * np.sin(th) * np.cos(ph),
         radius * np.sin(th) * np.sin(ph),
         radius * np.cos(th)], axis=1
    )


def _ch_names(n):
    return [f"c{i}" for i in range(n)]


def test_car_is_mean_subtraction():
    X = np.random.default_rng(0).normal(size=(50, 8))
    M = car_matrix(8)
    assert np.allclose(X @ M, X - X.mean(axis=1, keepdims=True))


def test_car_annihilates_common_mode():
    m = car_matrix(8)
    ones = np.ones((1, 8))
    assert np.allclose(ones @ m, 0)
    assert np.linalg.matrix_rank(m) == 7


def test_unipolar_subtracts_channel():
    X = np.random.default_rng(1).normal(size=(50, 8))
    ref = 3
    M = unipolar_matrix(8, ref)
    assert np.allclose(X @ M, X - X[:, ref][:, None])


def test_unipolar_rank():
    assert np.linalg.matrix_rank(unipolar_matrix(8, 0)) == 7


def test_bipolar_is_adjacent_differences():
    X = np.random.default_rng(2).normal(size=(30, 6))
    M = bipolar_chain_matrix(6)
    expected = np.stack([X[:, i] - X[:, (i + 1) % 6] for i in range(6)], axis=1)
    assert np.allclose(X @ M, expected)
    assert np.linalg.matrix_rank(M) == 5


def test_rest_recovers_reference_free_potentials():
    """REST es exacto para señales del subespacio modelado.

    Si ``V_true = G s`` con ``s`` en el espacio fila de ``W_avg G`` (el
    subespacio físicamente alcanzable con el lead field promediado), entonces
    REST recupera ``V_true`` desde referencias unipolar y CAR.
    Convención: ``X_ref = X @ M`` con ``X`` de forma ``(time, channels)``.
    """
    ch_names = _ch_names(20)
    ch_pos = _sphere_positions(20, seed=3)
    lf = compute_lead_field(ch_names, ch_pos, src_grid_mm=20.0)
    G = lf.matrix  # (C, 3N)
    C = G.shape[0]

    W = car_matrix(C)
    A = W @ G
    rng = np.random.default_rng(4)
    c = rng.normal(size=(C, 5))
    s = A.T @ c                      # s en rowspace(WG): subespacio modelado
    X_true = (G @ s).T               # (T, C)

    T_rest = rest_matrix(G, C)
    X_uni = X_true - X_true[:, [3]]
    X_car = X_true - X_true.mean(axis=1, keepdims=True)

    for X_ref in (X_uni, X_car):
        rec = X_ref @ T_rest
        # comparación relativa: las escalas de G dependen de la geometría
        scale = max(1.0, np.abs(X_true).max())
        assert np.allclose(rec / scale, X_true / scale, rtol=1e-9, atol=1e-9), (
            "REST no recupera los potenciales sin referencia."
        )


def test_rest_from_bipolar_is_finite():
    """REST desde bipolar debe ser finito y de magnitud fisiológica.

    El montaje bipolar es diferencial (rango C-1) y pierde la componente
    constante, por lo que no se exige recuperación exacta sino consistencia
    numérica del operador.
    """
    ch_names = _ch_names(14)
    ch_pos = _sphere_positions(14, seed=11)
    lf = compute_lead_field(ch_names, ch_pos, src_grid_mm=20.0)
    T_rest = rest_matrix(lf.matrix, 14)
    M_bip = bipolar_chain_matrix(14)

    X = np.random.default_rng(12).normal(size=(100, 14)) * 50e-6
    X_bip = X @ M_bip
    V_rest = X_bip @ T_rest
    assert np.isfinite(V_rest).all()
    assert np.abs(V_rest).max() < 5e-3


def test_rest_matrix_properties():
    """La matriz REST anula la señal constante (datos) y produce amplitudes fisiológicas."""
    ch_names = _ch_names(16)
    ch_pos = _sphere_positions(16, seed=5)
    lf = compute_lead_field(ch_names, ch_pos, src_grid_mm=30.0)
    T = rest_matrix(lf.matrix, 16)

    assert np.isfinite(T).all()
    # una fila constante de datos se anula: ones @ T == 0
    assert np.max(np.abs(np.ones((1, 16)) @ T)) < 1e-9

    X = np.random.default_rng(6).normal(size=(100, 16)) * 50e-6
    V_rest = X @ T
    assert np.all(np.abs(V_rest).max(axis=0) < 5e-3)


def test_leadfield_channel_mapping():
    ch_names = _ch_names(10)
    ch_pos = _sphere_positions(10, seed=7)
    lf = compute_lead_field(ch_names, ch_pos, src_grid_mm=20.0)
    assert lf.n_channels == 10
    assert lf.n_sources > 0
    assert lf.matrix.shape == (10, 3 * lf.n_sources)
    assert np.isfinite(lf.matrix).all()
    assert lf.ch_names == ch_names


def test_leadfield_rank_and_variability():
    """El lead field debe ser informativo: rango razonable y distribución amplia."""
    ch_names = _ch_names(12)
    ch_pos = _sphere_positions(12, seed=8)
    lf = compute_lead_field(ch_names, ch_pos, src_grid_mm=20.0)
    G = lf.matrix
    r = np.linalg.matrix_rank(G, tol=1e-9)
    assert r >= 8, f"lead field casi degenerado (rango {r} de {G.shape[0]})"
    colnorm = np.linalg.norm(G, axis=0)
    assert np.std(np.log10(colnorm + 1e-30)) > 0.1


def test_rest_matrix_composes_with_car():
    """REST aplicado sobre CAR debe coincidir con REST aplicado sobre unipolar
    cuando ambas vienen de la misma señal (linealidad del pipeline)."""
    ch_names = _ch_names(14)
    ch_pos = _sphere_positions(14, seed=9)
    lf = compute_lead_field(ch_names, ch_pos, src_grid_mm=20.0)
    T_rest = rest_matrix(lf.matrix, 14)

    X = np.random.default_rng(10).normal(size=(200, 14)) * 50e-6
    X_uni = X - X[:, [3]]
    X_car = X - X.mean(axis=1, keepdims=True)
    assert np.allclose(X_uni @ T_rest, X_car @ T_rest, atol=1e-14)