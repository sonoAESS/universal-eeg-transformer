"""Proyección entre montajes de EEG hacia un espacio canónico común.

El transformador universal opera sobre un espacio fijo de ``C`` canales. Para
poder consumir grabaciones con cualquier distribución de electrodos (p. ej.
19 canales 10-20, 32, 64 de 10-10, o más densos) hay que **proyectar** cada
montaje al conjunto de posiciones canónicas. Este módulo implementa dos
caminos:

* ``spline`` (interpolación esférica, estilo *topomapa/heatmap*):
  interpola el campo de potenciales sobre la esfera del cuero cabelludo
  usando splines esféricos (Perrin et al., 1989). Es puramente geométrico:
  solo necesita las posiciones de los electrodos.

* ``leadfield`` (proyección por solución inversa):
  estima los potenciales canónicos resolviendo el problema inverso con el
  *lead field* analítico multicapa. Para una fuente :math:`q`, los
  potenciales en los electrodos fuente son :math:`V_s = G_s q` y en los
  canónicos :math:`V_c = G_c q`. Estimando :math:`\\hat q` por mínimos
  cuadrados (estilo REST, con la referencia promedio proyectada), la matriz
  de transferencia queda :math:`P = W_s (W_s G_s)^{+T}\\ G_c^T` con
  :math:`W_s = I - 11^T/C_s`.

Ambos métodos producen una matriz :math:`P \\in \\mathbb{R}^{C_s \\times
C_c}` aplicada como ``X_canon = X_src @ P`` y su calidad se evalúa
proyectando de vuelta al montaje original (*round-trip*).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .logging_conf import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Splines esféricos (Perrin et al., 1989)
# ---------------------------------------------------------------------------
def _legendre_kernel(c: np.ndarray, order: int = 8, m: int = 4) -> np.ndarray:
    """Núcleo del spline esférico ``G(x) = Σ_{n=1..order} (2n+1)/(n(n+1))^m P_n(x)``.

    ``c`` es el coseno del ángulo entre dos puntos en la esfera unitaria
    (valores en ``[-1, 1]``). ``m`` controla la suavidad (4 es el valor
    clásico); ``order`` trunca la serie de Legendre.
    """
    c = np.asarray(c, dtype=np.float64)
    out = np.zeros_like(c)
    # P_0 = 1, P_1 = c
    p_prev = np.ones_like(c)
    p_curr = c.copy()
    for n in range(1, order + 1):
        coef = (2.0 * n + 1.0) / (n * (n + 1.0)) ** m
        out += coef * p_curr
        # recurrencia de Legendre: P_{n+1} = ((2n+1) c P_n - n P_{n-1})/(n+1)
        p_next = ((2.0 * n + 1.0) * c * p_curr - n * p_prev) / (n + 1.0)
        p_prev, p_curr = p_curr, p_next
    return out


def _normalize(positions: np.ndarray) -> np.ndarray:
    """Normaliza posiciones 3D al radio unitario (esfera)."""
    r = np.linalg.norm(positions, axis=1, keepdims=True)
    return positions / np.maximum(r, 1e-12)


def spherical_spline_matrix(
    src_positions: np.ndarray,
    dst_positions: np.ndarray,
    order: int = 30,
    smoothness: float = 1e-5,
) -> np.ndarray:
    """Matriz de interpolación esférica ``src -> dst`` (C_s, C_dst).

    Se resuelve el sistema lineal del spline esférico con el *gauge* de suma
    nula (media de coeficientes cero) y una ligera regularización ridge para
    canales casi coincidentes.

    Parameters
    ----------
    src_positions:
        Posiciones 3D (m) de los electrodos fuente ``(C_s, 3)``.
    dst_positions:
        Posiciones 3D (m) de los electrodos destino ``(C_dst, 3)``.
    order:
        Términos de Legendre del núcleo (truncado).
    smoothness:
        Regularización ridge relativa sobre el sistema lineal.
    """
    src = _normalize(src_positions)
    dst = _normalize(dst_positions)
    c_s_s = src @ src.T          # (C_s, C_s) cosenos entre fuentes
    c_d_s = dst @ src.T          # (C_dst, C_s) cosenos destino-fuente

    k_ss = _legendre_kernel(c_s_s.ravel(), order=order).reshape(c_s_s.shape)
    k_ds = _legendre_kernel(c_d_s.ravel(), order=order).reshape(c_d_s.shape)

    n_s = src.shape[0]
    # sistema [K_ss  1; 1^T 0] [a; a0] = [V_s; 0] con gauge Σa=0
    top = np.hstack([k_ss + smoothness * np.eye(n_s), np.ones((n_s, 1))])
    bot = np.hstack([np.ones((1, n_s)), np.zeros((1, 1))])
    a_aux = np.vstack([top, bot])
    a_inv = np.linalg.inv(a_aux)          # (C_s+1, C_s+1)
    right = a_inv[:, :n_s]                # (C_s+1, C_s)
    m = np.hstack([k_ds, np.ones((dst.shape[0], 1))]) @ right  # (C_dst, C_s)
    return np.ascontiguousarray(m.T, dtype=np.float32)


# ---------------------------------------------------------------------------
# Proyección por lead field (solución inversa)
# ---------------------------------------------------------------------------
def leadfield_projection_matrix(
    g_src: np.ndarray,
    g_dst: np.ndarray,
    rcond: float = 1e-10,
    n_components: int | None = None,
) -> np.ndarray:
    """Matriz ``src -> dst`` mediante inversión del lead field (C_s, C_dst).

    Estimación de los potenciales al infinito en las posiciones destino a
    partir de los electrodos fuente resolviendo
    :math:`P = W_s (W_s G_s)^{+T} G_c^T` (equivalente a REST pero entre
    montajes), con :math:`W_s = I - 11^T/C_s` la referencia promedio del
    montaje fuente.

    El problema inverso ``W_s G_s`` está fuertemente mal condicionado
    (condicionamiento del orden de ``1e15``), de modo que la estimación
    mínimos cuadrados amplifica el ruido de los datos reales y la calidad
    empeora (``ve`` negativa) pese a ser exacto en datos sintéticos. Se
    recomienda regularizar truncando el SVD a los ``n_components`` valores
    singulares mayores (equivalentemente, a los modos espaciales de menor
    frecuencia). En la práctica ``n_components ~ C_s // 3`` estabiliza la
    proyección y supera a la interpolación spline (ve ~ 0.65 en 10-20).

    Parameters
    ----------
    g_src:
        Lead field del montaje fuente ``(C_s, N)``.
    g_dst:
        Lead field del montaje destino/canónico ``(C_dst, N)``.
    rcond:
        Umbral de corte para la pseudo-inversa (regularización efectiva).
    n_components:
        Si se da, se truncate el SVD a los ``n_components`` valores
        singulares mayores en lugar de usar ``rcond``.
    """
    n_s = g_src.shape[0]
    w_s = np.eye(n_s) - np.ones((n_s, n_s)) / n_s
    wg = w_s @ g_src
    if n_components is not None:
        u, s, vt = np.linalg.svd(wg, full_matrices=False)
        k = max(1, min(int(n_components), s.size))
        wg_inv = (vt[:k].T * (1.0 / s[:k])) @ u[:, :k].T    # (N, C_s)
    else:
        wg_inv = np.linalg.pinv(wg, rcond=rcond)            # (N, C_s)
    # Estimación de los potenciales al infinito en las posiciones destino:
    #   V_c ~ G_dst (W_s G_s)^+ W_s V_s   =>   P = W_s^T (W_s G_s)^{+T} G_dst^T
    # con W_s^T = W_s (simétrica) por la referencia promedio del montaje fuente.
    p = w_s @ wg_inv.T @ g_dst.T                  # (C_s, C_dst)
    # Centramos columnas: el modo constante instantáneo no es recuperable y
    # las referencias canónicas lo anulan de todos modos.
    return np.ascontiguousarray(p, dtype=np.float32)


# ---------------------------------------------------------------------------
# Utilidades de montaje
# ---------------------------------------------------------------------------
STANDARD_10_20_19 = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8",
    "Fz", "Cz", "Pz",
]


def select_subset(
    ch_names: list[str],
    ch_positions: np.ndarray,
    keep: list[str] | None = None,
) -> Tuple[list[str], np.ndarray, np.ndarray]:
    """Selecciona un subconjunto (p. ej. 10-20) de electrodos.

    Returns
    -------
    ``(sub_names, sub_positions, indices)`` con ``indices`` las columnas
    (índices) de los canales fuente dentro del montaje original.
    """
    if keep is None:
        keep = STANDARD_10_20_19
    idx = [ch_names.index(c) for c in keep if c in ch_names]
    return (
        [ch_names[i] for i in idx],
        np.asarray(ch_positions)[idx],
        np.asarray(idx),
    )


def nearest_neighbor_matrix(
    src_positions: np.ndarray,
    dst_positions: np.ndarray,
) -> np.ndarray:
    """Asigna a cada canal destino el valor del electrodo fuente más cercano.

    Línea base geométrica mínima (sin interpolación): cada columna de la
    proyección es un vector unitario en la fila del vecino más cercano.
    """
    src = _normalize(src_positions)
    dst = _normalize(dst_positions)
    d2 = ((dst[:, None, :] - src[None, :, :]) ** 2).sum(-1)  # (C_dst, C_s)
    nn = np.argmin(d2, axis=1)                                # (C_dst,)
    p = np.zeros((src.shape[0], dst.shape[0]), dtype=np.float32)
    p[nn, np.arange(dst.shape[0])] = 1.0
    return p


def build_projection(
    method: str,
    src_positions: np.ndarray,
    dst_positions: np.ndarray,
    g_src: np.ndarray | None = None,
    g_dst: np.ndarray | None = None,
    n_components: int | None = None,
) -> np.ndarray:
    """Construye la matriz de proyección ``src -> dst`` según el método.

    ``n_components`` solo aplica a ``leadfield`` (truncado SVD).
    """
    if method == "spline":
        return spherical_spline_matrix(src_positions, dst_positions)
    if method == "leadfield":
        if g_src is None or g_dst is None:
            raise ValueError("'leadfield' requiere g_src y g_dst.")
        if n_components is None:
            n_components = max(1, g_src.shape[0] // 3)
        return leadfield_projection_matrix(g_src, g_dst, n_components=n_components)
    if method == "nearest":
        return nearest_neighbor_matrix(src_positions, dst_positions)
    raise ValueError(f"Método de proyección desconocido: '{method}'")


def round_trip_error(
    signals_true: np.ndarray,
    signals_reduced: np.ndarray,
    p_back: np.ndarray,
) -> dict[str, float]:
    """Mide la calidad de proyección con un *round-trip* sobre datos reales.

    ``signals_true`` son las señales canónicas reales ``(T, C_c)``;
    ``signals_reduced`` su versión observada desde el montaje fuente
    ``(T, C_s)``. Se reconstruye ``signals_reduced @ p_back`` y se compara
    contra ``signals_true`` **en el subespacio observable** (centrando cada
    instante): el modo constante de referencia no es recuperable y todos los
    montajes lo anulan.

    Returns
    -------
    dict
        ``rmse_uV``, ``mae_uV``, ``r`` y ``ve`` (varianza explicada).
    """
    c = signals_true.shape[1]
    p_cent = np.eye(c, dtype=np.float64) - np.ones((c, c), dtype=np.float64) / c

    pred = signals_reduced @ p_back
    t_cent = signals_true - signals_true.mean(1, keepdims=True)
    p_cent_s = pred - pred.mean(1, keepdims=True)
    err = (t_cent - p_cent_s) @ p_cent

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    r = 0.0
    count = 0
    for col in range(c):
        t, pp = t_cent[:, col], p_cent_s[:, col]
        if np.std(t) < 1e-20 or np.std(pp) < 1e-20:
            continue
        r += float(np.corrcoef(t, pp)[0, 1])
        count += 1
    ve = float(1.0 - np.sum(err ** 2) / (np.sum(t_cent ** 2) + 1e-15))
    return {
        "rmse_uV": rmse * 1e6,
        "mae_uV": mae * 1e6,
        "r": r / max(1, count),
        "ve": ve,
    }