"""Referencias de EEG como matrices lineales analíticas.

Todas las referencias de interés (unipolar, bipolar, CAR, REST) son
operadores lineales sobre el vector de potenciales por canal. En este módulo
se construyen explícitamente dichas matrices :math:`M \\in \\mathbb{R}^{C
\\times C}` (aplicadas como ``X_ref = X @ M``, con ``X`` de forma
``(time, channels)``), garantizando que el mapeo entre referencias es
algebraicamente exacto y reproducible.

Referencias soportadas:
    * ``unipolar``        : ``X - X[:, u]`` (referencia a un canal ``u``,
      por defecto el vértice ``Cz``).
    * ``linked_mastoids`` : ``X - mean(X[M1], X[M2])``, con el potencial en
      las mastoides interpolado esféricamente desde los electrodos reales.
    * ``linked_ears``     : ídem con los lóbulos A1/A2.
    * ``bipolar``         : cadena diferencial; si se dan posiciones reales,
      la cadena es anular sobre electrodos vecinos (orden azimutal).
    * ``car``             : referencia promedio (common average reference).
    * ``rest``            : referencia al infinito (Yao, 2001), ver
      :mod:`leadfield`.
    * ``laplacian``       : Laplaciano de superficie esférico (Perrin, 1989),
      estimador del CSD; invariante a cualquier re-referenciación.

Referencia del dato original: la señal adquirida (p. ej. ``eegbci``, contra
la mastoides izquierda) queda implícita en ``X``. Los cuatro operadores
anulan cualquier offset constante de referencia por instante de tiempo
(``W_avg`` en CAR/REST y diferencias en unipolar/bipolar), de modo que el
resultado de la transformación es **invariante** a la referencia física de
adquisición.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .config import REFERENCE_KINDS
from .logging_conf import get_logger
from .mapping import (
    EAR_LOBE_POSITIONS,
    MASTOID_POSITIONS,
    landmark_weights_matrix,
    spherical_laplacian_matrix,
)

log = get_logger(__name__)


def car_matrix(n_channels: int) -> np.ndarray:
    """Matriz de la referencia promedio (CAR): ``I - (1/C) 1 1^T``."""
    ones = np.ones((n_channels, n_channels)) / n_channels
    return np.eye(n_channels) - ones


def unipolar_matrix(n_channels: int, ref_index: int) -> np.ndarray:
    """Matriz unipolar: ``X - X[:, ref_index]`` (referencia a un canal).

    ``M[k, j] = delta_{kj} - delta_{k, ref_index}``, de modo que
    ``(X @ M)[:, j] = X[:, j] - X[:, ref_index]``.
    """
    if not 0 <= ref_index < n_channels:
        raise ValueError(f"ref_index {ref_index} fuera de rango [0, {n_channels}).")
    m = np.eye(n_channels)
    m[ref_index, :] -= 1.0
    return m


def bipolar_chain_matrix(n_channels: int) -> np.ndarray:
    """Cadena diferencial cíclica: canal ``j`` referenciado al canal ``j+1``.

    ``M[k, j] = delta_{kj} - delta_{(k+1)%C, j}`` de modo que
    ``(X @ M)[:, j] = X[:, j] - X[:, (j+1)%C]``. Produce ``C`` salidas de
    rango ``C - 1`` (el modo común queda anulado, como en todo montaje
    diferencial).
    """
    m = np.eye(n_channels)
    rows = (np.arange(n_channels) + 1) % n_channels
    cols = np.arange(n_channels)
    m[rows, cols] -= 1.0
    return m


def bipolar_nn_chain_matrix(positions: np.ndarray) -> np.ndarray:
    """Cadena bipolar por vecinos físicos más cercanos (ciclo codicioso).

    Empieza en el electrodo de mayor elevación y visita siempre el vecino
    angular no visitado más cercano (distancia geodésica sobre el casco);
    el ciclo se cierra con el último electrodo. Cada par es físicamente
    adyacente salvo el cierre, análogo al cierre de la cadena doble plátano.
    """
    unit = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    n_c = len(unit)
    cos = np.clip(unit @ unit.T, -1.0, 1.0)
    dist = np.arccos(cos)
    np.fill_diagonal(dist, np.inf)

    order = [int(np.argmax(unit[:, 2]))]          # arranca en el vértice
    pendientes = set(range(n_c)) - {order[0]}
    while pendientes:
        ultimo = order[-1]
        siguiente = min(pendientes, key=lambda j: dist[ultimo, j])
        order.append(siguiente)
        pendientes.discard(siguiente)

    m = np.zeros((n_c, n_c))
    for j in range(n_c):
        m[order[j], order[j]] += 1.0
        m[order[j], order[(j + 1) % n_c]] -= 1.0
    return m


def linked_matrix(n_channels: int, weights: np.ndarray) -> np.ndarray:
    """Referencia *linked* genérica: ``X_ref = X - (X @ w) 1ᵀ``.

    ``w`` son pesos de interpolación hacia el hito anatómico (o promedio de
    dos hitos, p. ej. M1/M2): ``(X @ w)`` es el potencial estimado en ese
    hito, que se resta a todos los canales. Como ``Σw = 1``, la matriz anula
    el modo común y conserva el rango ``C - 1``.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n_channels,):
        raise ValueError(
            f"weights debe tener forma ({n_channels},), recibió {w.shape}."
        )
    return np.eye(n_channels) - w[:, None] @ np.ones((1, n_channels))


def laplacian_matrix(positions: np.ndarray) -> np.ndarray:
    """Matriz del Laplaciano de superficie (Perrin) para posiciones reales."""
    return spherical_laplacian_matrix(positions)


def rest_matrix(
    lead_field: np.ndarray,
    n_channels: int,
    rcond: float = 1e-12,
) -> np.ndarray:
    r"""Matriz de la referencia al infinito (REST, Yao 2001).

    Definimos :math:`L = G_L\\, (W_{\\mathrm{avg}}\\, G_L)^{+}` (operador que
    estima los potenciales al infinito a partir de datos en referencia
    promedio) y devolvemos la matriz aplicable por filas:

    .. math::

        M_{\\mathrm{rest}} = W_{\\mathrm{avg}}\\, L^{T},
        \\qquad X_{\\mathrm{rest}} = X\\, M_{\\mathrm{rest}}.

    Así, ``X @ M_rest`` reproduce la estimación de mínimos cuadrados
    :math:`\\hat{V}_{\\mathrm{rest}} = L\\, (X\\, W_{\\mathrm{avg}})^{T}`
    formulado como operador lineal sobre los canales.

    Parameters
    ----------
    lead_field:
        Matriz de ganancia (electrodos x fuentes).
    n_channels:
        Número de canales ``C``.
    rcond:
        Umbral de corte para :func:`numpy.linalg.pinv`.
    """
    if lead_field.shape[0] != n_channels:
        raise ValueError(
            f"lead_field tiene {lead_field.shape[0]} filas pero se esperaban {n_channels}."
        )
    w_avg = car_matrix(n_channels)
    g_avg = w_avg @ lead_field
    g_avg_inv = np.linalg.pinv(g_avg, rcond=rcond, hermitian=False)
    l_est = lead_field @ g_avg_inv  # (C, C)
    return w_avg @ l_est.T


REFERENCE_BUILDERS: Dict[str, Callable[..., np.ndarray]] = {
    "unipolar": unipolar_matrix,
    "bipolar": bipolar_chain_matrix,
    "car": car_matrix,
    "rest": rest_matrix,
}


def build_reference_matrix(
    kind: str,
    n_channels: int,
    unipolar_ref_index: Optional[int] = None,
    lead_field: Optional[np.ndarray] = None,
    rest_rcond: Optional[float] = None,
    positions: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Construye la matriz de una referencia dada.

    Parameters
    ----------
    kind:
        Clave de referencia entre ``REFERENCE_KINDS``.
    n_channels:
        Número de canales.
    unipolar_ref_index:
        Índice del canal de referencia para la montaje ``unipolar``.
    lead_field:
        Lead field (necesario para ``rest``).
    rest_rcond:
        Corte relativo de valores singulares para la referencia REST
        (estilo RESTRIDGE): trunca las componentes mal condicionadas del
        operador al infinito. ``None`` mantiene el comportamiento de
        ``rest_matrix`` (``rcond=1e-12``).
    positions:
        Posiciones reales ``(C, 3)`` de los electrodos; necesarias para
        ``linked_mastoids``, ``linked_ears``, ``laplacian`` y la cadena
        bipolar anular sobre vecinos físicos.
    """
    if kind not in REFERENCE_BUILDERS and kind not in (
        "linked_mastoids", "linked_ears", "laplacian"
    ):
        raise ValueError(f"Referencia desconocida '{kind}'. Válidas: {REFERENCE_KINDS}")

    if kind == "unipolar":
        if unipolar_ref_index is None:
            raise ValueError("'unipolar' requiere unipolar_ref_index.")
        return unipolar_matrix(n_channels, unipolar_ref_index)

    if kind == "rest":
        if lead_field is None:
            raise ValueError("'rest' requiere lead_field.")
        return rest_matrix(lead_field, n_channels, rcond=rest_rcond or 1e-12)

    if kind in ("linked_mastoids", "linked_ears"):
        if positions is None:
            raise ValueError(f"'{kind}' requiere posiciones de electrodos.")
        landmark = MASTOID_POSITIONS if kind == "linked_mastoids" else EAR_LOBE_POSITIONS
        weights = landmark_weights_matrix(positions, landmark)
        return linked_matrix(n_channels, weights)

    if kind == "laplacian":
        if positions is None:
            raise ValueError("'laplacian' requiere posiciones de electrodos.")
        return laplacian_matrix(positions)

    if kind == "bipolar" and positions is not None:
        return bipolar_nn_chain_matrix(positions)

    return REFERENCE_BUILDERS[kind](n_channels)


def compute_all_references(
    X_raw: np.ndarray,
    n_channels: int,
    unipolar_ref_index: Optional[int] = None,
    lead_field: Optional[np.ndarray] = None,
    rest_rcond: Optional[float] = None,
    positions: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Calcula todas las referencias alineadas en el tiempo.

    ``X_raw`` es la señal tal como fue adquirida; su referencia física de
    origen (p. ej. mastoides izquierda en ``eegbci``) no altera el resultado
    porque todos los operadores aplicados eliminan el offset constante por
    instante (ver :func:`rest_matrix` y :func:`car_matrix`; el Laplaciano es
    directamente invariante a re-referenciación).

    Parameters
    ----------
    X_raw:
        Señal original de forma ``(time, channels)``.
    n_channels:
        Número de canales.
    unipolar_ref_index:
        Canal de referencia del montaje unipolar.
    lead_field:
        Lead field para REST.
    rest_rcond:
        Corte SVD relativo del operador REST (opcional).
    positions:
        Posiciones reales de los electrodos (necesarias para linked/laplacian
        y bipolar anular).

    Returns
    -------
    dict
        ``{kind: X_kind}``, cada ``X_kind`` de forma ``(time, channels)``.
    """
    refs: Dict[str, np.ndarray] = {}
    for kind in REFERENCE_KINDS:
        m = build_reference_matrix(
            kind, n_channels, unipolar_ref_index=unipolar_ref_index,
            lead_field=lead_field, rest_rcond=rest_rcond, positions=positions,
        )
        refs[kind] = X_raw @ m
    return refs


def inter_reference_matrix(
    kind_src: str,
    kind_dst: str,
    n_channels: int,
    unipolar_ref_index: Optional[int] = None,
    lead_field: Optional[np.ndarray] = None,
    rcond: float = 1e-10,
    rest_rcond: Optional[float] = None,
    positions: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mapa lineal analítico entre dos referencias (para validación).

    Dadas las matrices :math:`T_s` y :math:`T_d` que llevan la señal cruda a
    cada referencia, el operador :math:`M_{s\\to d} = T_d\\, T_s^{+}` cumple
    :math:`M_{s\\to d}\\, X_s \\approx X_d` para cualquier ``X_s`` compatible
    (es la solución de mínimos cuadrados lineales del problema de mapeo).

    Se usa como línea base analítica para validar que el autoencoder lineal
    aprende exactamente las mismas transformaciones físicas.
    """
    t_src = build_reference_matrix(
        kind_src, n_channels, unipolar_ref_index=unipolar_ref_index,
        lead_field=lead_field, rest_rcond=rest_rcond, positions=positions,
    )
    t_dst = build_reference_matrix(
        kind_dst, n_channels, unipolar_ref_index=unipolar_ref_index,
        lead_field=lead_field, rest_rcond=rest_rcond, positions=positions,
    )
    return t_dst @ np.linalg.pinv(t_src, rcond=rcond)


def reference_matrix_from_kind(kind: str, n_channels: int, **kwargs) -> np.ndarray:
    """Alias con notación explícita del canal unipolar."""
    return build_reference_matrix(
        kind, n_channels,
        unipolar_ref_index=kwargs.get("unipolar_ref_index"),
        lead_field=kwargs.get("lead_field"),
        rest_rcond=kwargs.get("rest_rcond"),
        positions=kwargs.get("positions"),
    )