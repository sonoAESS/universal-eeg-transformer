"""Tests de las referencias extendidas (rama universal_refs).

Cubre ``linked_mastoids``, ``linked_ears``, ``laplacian`` y la cadena
bipolar anular sobre posiciones reales. Invariantes físicos clave:

* El modo común (misma señal en todos los canales) se anula SIEMPRE: es la
  condición de invariancia a la referencia física de adquisición.
* Rango ``C - 1`` para operadores que pierden el modo constante.
* El Laplaciano coincide con el CSD esférico de MNE (Perrin) sobre campos
  reales suaves.
"""

from __future__ import annotations

import numpy as np
import pytest

from eeg_transform.config import REFERENCE_KINDS
from eeg_transform.leadfield import compute_lead_field
from eeg_transform.mapping import (
    EAR_LOBE_POSITIONS,
    MASTOID_POSITIONS,
    landmark_weights_matrix,
)
from eeg_transform.references import (
    bipolar_nn_chain_matrix,
    build_reference_matrix,
    linked_matrix,
)


def _cap_positions(n: int, seed: int = 0, radius: float = 0.09) -> np.ndarray:
    """Casco cuasi-uniforme determinista sobre el casquete (fibonacci).

    Reutiliza ``dense_cap_positions`` del repo: mejor condicionamiento que
    el muestreo aleatorio para validar operadores espaciales.
    """
    from eeg_transform.experiments.multi import dense_cap_positions

    return dense_cap_positions(n, radius=radius, seed=seed)


def _assert_common_mode_dies(kind: str, positions: np.ndarray,
                             lead_field=None, unipolar_ref_index: int = 3):
    """El modo común s(t)·1ᵀ debe anularse para toda referencia."""
    c = positions.shape[0]
    kwargs = {}
    if kind == "rest":
        kwargs["lead_field"] = lead_field
    m = build_reference_matrix(
        kind, c, unipolar_ref_index=unipolar_ref_index,
        positions=positions, **kwargs,
    )
    s = np.random.default_rng(1).normal(size=(64, 1))
    comun = s @ np.ones((1, c))
    residual = comun @ m
    assert np.abs(residual).max() / np.abs(comun).max() < 1e-5, (
        f"'{kind}' no anula el modo común"
    )
    return m


@pytest.fixture(scope="module")
def lead_field_32():
    pos = _cap_positions(32, seed=5)
    lf = compute_lead_field([f"c{i}" for i in range(32)], pos, src_grid_mm=20.0)
    return pos, lf


def test_reference_kinds_extended():
    assert REFERENCE_KINDS == (
        "unipolar", "linked_mastoids", "linked_ears",
        "bipolar", "car", "rest", "laplacian",
    )


@pytest.mark.parametrize("kind", list(REFERENCE_KINDS))
def test_all_references_annihilate_common_mode(kind, lead_field_32):
    pos, lf = lead_field_32
    _assert_common_mode_dies(kind, pos, lead_field=lf.matrix)


@pytest.mark.parametrize("kind", ["linked_mastoids", "linked_ears"])
def test_linked_rank_and_structure(kind, lead_field_32):
    pos, _ = lead_field_32
    c = len(pos)
    m = build_reference_matrix(kind, c, positions=pos)
    assert np.linalg.matrix_rank(m) == c - 1


def test_landmark_weights_normalized_and_symmetric(lead_field_32):
    pos, _ = lead_field_32
    for landmark in (MASTOID_POSITIONS, EAR_LOBE_POSITIONS):
        w = landmark_weights_matrix(pos, landmark)
        assert w.shape == (len(pos),)
        assert abs(w.sum() - 1.0) < 1e-12   # invariancia exacta al modo común
    # simetría izquierda/derecha del casco: pesos de ±x intercambiados
    mirror = pos * np.array([-1.0, 1.0, 1.0])
    w = landmark_weights_matrix(pos, MASTOID_POSITIONS)
    w_mirror = landmark_weights_matrix(mirror, MASTOID_POSITIONS)
    assert np.allclose(w, w_mirror, atol=1e-8)


def test_linked_matrix_subtracts_weighted_average():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(100, 6))
    w = rng.uniform(0.1, 0.3, size=6)
    w = w / w.sum()
    m = linked_matrix(6, w)
    esperado = x - (x @ w)[:, None]
    assert np.allclose(x @ m, esperado)


def test_bipolar_nn_chain_is_physical_ring():
    """La cadena codiciosa conecta vecinos físicos y anula el modo común."""
    pos = _cap_positions(16, seed=7)
    m = bipolar_nn_chain_matrix(pos)
    c = len(pos)
    assert m.shape == (c, c)
    assert np.linalg.matrix_rank(m) == c - 1
    # cada fila tiene exactamente +1 y -1: diferencia entre un par
    assert np.all(np.isclose(np.abs(m).sum(axis=1), 2.0))
    # los pares interiores de la cadena son vecinos: distancia angular dentro
    # del rango típico del casco (el cierre y algún salto final pueden ser
    # largos, como en la doble banana clásica)
    unit = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    dist = np.arccos(np.clip(unit @ unit.T, -1.0, 1.0))
    p90 = np.percentile(dist[~np.eye(c, dtype=bool)], 90)
    order = _reconstruir_order(pos)
    for j in range(c - 1):
        a, b = order[j], order[(j + 1) % c]
        assert dist[a, b] <= p90, f"par interior no adyacente ({a},{b})"


def _reconstruir_order(pos):
    """Replica el orden codicioso para verificar adyacencia por pares."""
    from eeg_transform.references import bipolar_nn_chain_matrix  # noqa: F401
    unit = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    d = np.arccos(np.clip(unit @ unit.T, -1.0, 1.0))
    np.fill_diagonal(d, np.inf)
    order = [int(np.argmax(unit[:, 2]))]
    restantes = set(range(len(pos))) - {order[0]}
    while restantes:
        ultimo = order[-1]
        sig = min(restantes, key=lambda j: d[ultimo, j])
        order.append(sig)
        restantes.discard(sig)
    return order


def _campo_suave(pos, n_patrones=5, seed=11, amplitud=50e-6, ancho=0.08):
    """Campo espacialmente suave (patrones gaussianos), realista en casco."""
    rng = np.random.default_rng(seed)
    unit = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    centers = unit[rng.choice(len(pos), n_patrones, replace=False)]
    t = np.linspace(0, 1, 300)[:, None]                       # (T, 1)
    amps = rng.normal(size=(1, n_patrones)) * np.cos(2 * np.pi * 2 * t)
    diff = unit[:, None, :] - centers[None, :, :]             # (C, K, 3)
    patron = np.exp(-((diff ** 2).sum(-1) / ancho))           # (C, K) suave
    return amplitud * (amps @ patron.T)                       # (T, C)


def test_laplacian_properties(lead_field_32):
    pos, _ = lead_field_32
    c = len(pos)
    lap = build_reference_matrix("laplacian", c, positions=pos)
    assert lap.shape == (c, c)
    # modo constante fuera por construcción: suma de columnas ~0 en norma
    # relativa (matrix_rank es insensible aquí por la escala 1/R²)
    norm = np.abs(lap).max()
    assert np.abs(lap.sum(axis=0)).max() / norm < 1e-6
    # unidades físicas: sobre campos suaves (~50 µV) produce CSD razonable
    data = _campo_suave(pos)
    csd = data @ lap                                          # V/m²
    ratio = np.abs(csd).max() / np.abs(data).max()
    assert 10 < ratio < 1e6, f"amplificación implausible: {ratio:.2e}"


def test_laplacian_matches_mne_csd(lead_field_32):
    """Acuerdo con el CSD spline de MNE (Perrin) en parámetros alineados.

    La correlación no es ~1 porque MNE resuelve el spline sin gauge de suma
    nula y con normalizaciones propias; se exige mismo signo, correlación
    alta y escala coherente con el factor físico 1/R² que sí aplicamos.
    """
    mne = pytest.importorskip("mne")
    pos, _ = lead_field_32
    names = [f"E{i}" for i in range(len(pos))]
    data = _campo_suave(pos)

    info = mne.create_info(names, sfreq=160.0, ch_types="eeg")
    raw = mne.io.RawArray(data.T.copy(), info, verbose=False)
    montage = mne.channels.make_dig_montage(
        ch_pos={n: p for n, p in zip(names, pos)}, coord_frame="head",
    )
    raw.set_montage(montage)
    csd_mne = mne.preprocessing.compute_current_source_density(
        raw, lambda2=1e-7, n_legendre_terms=30, verbose=False,
    ).get_data().T                                        # (T, C)

    lap = build_reference_matrix("laplacian", len(pos), positions=pos)
    csd_own = data @ lap                                      # V/m²

    r = np.corrcoef(csd_mne.ravel(), csd_own.ravel())[0, 1]
    assert r > 0.75, f"correlación insuficiente contra MNE CSD: {r:.3f}"
    # misma convención de signo (pendiente global positiva). La escala
    # absoluta de MNE usa normalizaciones propias; la nuestra queda validada
    # analíticamente en ``test_laplacian_eigenmode_exact``.
    pendiente = np.dot(csd_mne.ravel(), csd_own.ravel()) / np.dot(
        csd_own.ravel(), csd_own.ravel())
    assert pendiente > 0, "convención de signo opuesta a MNE"


def test_laplacian_eigenmode_exact(lead_field_32):
    """Validación analítica exacta con armónicos zonales.

    ``P_n(â·x̂)`` es auto-función de ``Δ_S`` con valor propio
    ``-n(n+1)/R²``: el CSD (``-Δ_S V``) debe devolver
    ``n(n+1)/R² · V`` con error pequeño, validando forma y magnitud.
    """
    from scipy.special import eval_legendre

    pos, _ = lead_field_32
    radio = float(np.median(np.linalg.norm(pos, axis=1)))
    unit = pos / radio
    lap = build_reference_matrix("laplacian", len(pos), positions=pos)
    rng = np.random.default_rng(13)

    for n_deg in (2, 3, 4):
        a = rng.normal(size=3)
        a /= np.linalg.norm(a)
        v = eval_legendre(n_deg, unit @ a)
        data = v[None, :] * np.linspace(0.5, 1.5, 200)[:, None] * 20e-6
        csd = data @ lap
        esperado = n_deg * (n_deg + 1) / radio ** 2 * data
        # tolerancia graduada: el efecto de borde del casquete (extrapolación
        # bajo el ecuador en el spline) crece con el grado; con densidad
        # mayor el error baja (medido: ~0.10/0.15/0.30 a 64 canales).
        tol = {2: 0.20, 3: 0.30, 4: 0.50}[n_deg]
        err_rel = np.abs(csd - esperado).max() / np.abs(esperado).max()
        assert err_rel < tol, (
            f"grado {n_deg}: error relativo {err_rel:.3f} > {tol} (el CSD "
            f"no converge al Laplaciano analítico)"
        )


def test_csd_field_consistency_dense_sparse(lead_field_32):
    """Convergencia interna del campo CSD entre densidades.

    El CSD es un campo escalar suave sobre el casco: calcularlo en el casco
    completo e interpolarlo a un subconjunto debe coincidir con calcular el
    Laplaciano directamente sobre las señales interpoladas a ese subconjunto.
    """
    from eeg_transform.mapping import select_subset, spherical_spline_matrix

    pos64 = _cap_positions(64, seed=21)
    keep = [f"c{i}" for i in range(19)]
    names64 = [f"c{i}" for i in range(64)]
    _, _, idx = select_subset(names64, pos64, keep=keep)
    pos19 = pos64[idx]

    data64 = _campo_suave(pos64, seed=23, ancho=0.6)   # resoluble por 19 ch
    lap64 = build_reference_matrix("laplacian", 64, positions=pos64)
    lap19 = build_reference_matrix("laplacian", 19, positions=pos19)

    csd_desde_denso = data64 @ lap64                       # campo CSD (T, 64)
    csd_en_19_via_campo = csd_desde_denso[:, idx]          # muestreo exacto
    # vía alternativa: interpolar potenciales 64->19 y diferenciar allí
    interp = spherical_spline_matrix(pos64, pos19)         # (64, 19)
    pot19 = data64 @ interp
    csd_en_19_via_potencial = pot19 @ lap19

    denom = np.sum(csd_en_19_via_campo ** 2) + 1e-30
    ve = 1.0 - np.sum((csd_en_19_via_campo - csd_en_19_via_potencial) ** 2) / denom
    assert ve > 0.9, f"campo CSD inconsistente entre densidades: ve={ve:.3f}"


def test_missing_geometry_raises():
    with pytest.raises(ValueError):
        build_reference_matrix("linked_mastoids", 19)
    with pytest.raises(ValueError):
        build_reference_matrix("laplacian", 19)


def test_inter_reference_matrix_with_new_kind(lead_field_32):
    """Línea base analítica cz->laplacian construible (validación de rutas)."""
    from eeg_transform.references import inter_reference_matrix

    pos, _ = lead_field_32
    m = inter_reference_matrix("unipolar", "laplacian", len(pos),
                               unipolar_ref_index=3, positions=pos)
    assert m.shape == (len(pos), len(pos))
    assert np.isfinite(m).all()
