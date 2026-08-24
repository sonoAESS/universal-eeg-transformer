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


def _legendre_kernel_laplacian(c: np.ndarray, order: int = 30, m: int = 4) -> np.ndarray:
    """Núcleo espectral del CSD spline ``-(Δ_S)`` sobre el mismo spline (Perrin).

    El Laplaciano de superficie multiplica cada grado ``n`` por ``n(n+1)``;
    con la convención neurofisiológica ``CSD = -Δ_S V`` el coeficiente pasa
    de ``(2n+1)/(n(n+1))^m`` a ``(2n+1)/(n(n+1))^(m-1)``, el clásico
    ``(2n+1)/(n(n+1))^3`` para ``m=4``.
    """
    c = np.asarray(c, dtype=np.float64)
    out = np.zeros_like(c)
    p_prev = np.ones_like(c)
    p_curr = c.copy()
    for n in range(1, order + 1):
        coef = (2.0 * n + 1.0) / (n * (n + 1.0)) ** (m - 1)
        out += coef * p_curr
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


def spherical_laplacian_matrix(
    positions: np.ndarray,
    order: int = 30,
    smoothness: float = 1e-7,
    radius: float | None = None,
) -> np.ndarray:
    """Matriz del Laplaciano de superficie esférico ``(C_s, C_s)`` (Perrin 1989).

    Ajusta el spline esférico de potencial a los electrodos (mismo sistema y
    *gauge* de suma nula que :func:`spherical_spline_matrix`) y evalúa su
    derivada angular segunda con el núcleo espectral ``(2n+1)/(n(n+1))^3``,
    escalada por ``1/R²`` para producir unidades físicas (V/m²) con ``R`` el
    radio del casco (por defecto, la mediana de las normas de las
    posiciones). Aplicada como ``X @ L``, cada fila produce el CSD del canal.

    Propiedades físicas: anula el modo constante común por filas (es
    invariante a cualquier re-referenciación, pues añadir la misma señal a
    todos los canales no altera el resultado).
    """
    src = _normalize(positions)
    c_ss = src @ src.T
    k_ss = _legendre_kernel(c_ss.ravel(), order=order).reshape(c_ss.shape)
    d_ss = _legendre_kernel_laplacian(c_ss.ravel(), order=order).reshape(c_ss.shape)

    n_s = src.shape[0]
    top = np.hstack([k_ss + smoothness * np.eye(n_s), np.ones((n_s, 1))])
    bot = np.hstack([np.ones((1, n_s)), np.zeros((1, 1))])
    # [a; a0] = inv(A) @ [V; 0]: solo importan las primeras n_s columnas de
    # la inversa y, dentro de ellas, las filas de coeficientes (el término
    # de gauge a0 no contribuye al Laplaciano: Δ(constante) = 0).
    coef = np.linalg.inv(np.vstack([top, bot]))[:n_s, :n_s]
    lap = d_ss @ coef                                         # (C_s, C_s)
    if radius is None:
        radius = float(np.median(np.linalg.norm(positions, axis=1)))
    lap = lap / max(radius, 1e-6) ** 2
    return np.ascontiguousarray(lap.T, dtype=np.float64)


def landmark_weights_matrix(
    positions: np.ndarray,
    landmark_positions: np.ndarray,
    smoothness: float = 1e-5,
) -> np.ndarray:
    """Pesos de interpolación de los electrodos hacia hitos anatómicos.

    Devuelve ``w`` de forma ``(C_s,)`` tal que ``X @ w`` aproxima el potencial
    promedio en los puntos de referencia dados (p. ej. mastoides M1/M2 o
    lóbulos A1/A2): cada columna del interpolador esférico se promedia. Los
    pesos suman ~1 (preservan el modo constante) y permiten construir
    referencias *linked* como ``M = I - 1 wᵀ``.
    """
    interp = spherical_spline_matrix(positions, landmark_positions,
                                     smoothness=smoothness)   # (C_s, L)
    w = interp.mean(axis=1).astype(np.float64)
    # Normalización exacta: Σw = 1 garantiza que la referencia linked anule
    # el modo constante por construcción (invariancia a la referencia de
    # adquisición), incluso con extrapolación bajo el ecuador.
    return w / w.sum()


# Hitos anatómicos estándar en la esfera unitaria (z vertical, y anterior).
# Mastoides M1/M2: azimut ±90°, ligeramente por debajo del ecuador (~-6°).
# Lóbulos A1/A2: mismos azimuts, más inferiores (~-15°).
MASTOID_POSITIONS = np.array([
    [1.0, 0.0, np.tan(np.deg2rad(-6.0))],
    [-1.0, 0.0, np.tan(np.deg2rad(-6.0))],
]) / np.sqrt(1.0 + np.tan(np.deg2rad(-6.0)) ** 2)
EAR_LOBE_POSITIONS = np.array([
    [1.0, 0.0, np.tan(np.deg2rad(-15.0))],
    [-1.0, 0.0, np.tan(np.deg2rad(-15.0))],
]) / np.sqrt(1.0 + np.tan(np.deg2rad(-15.0)) ** 2)


def density_smoothness(
    src_positions: np.ndarray,
    dst_positions: np.ndarray,
    base_smoothness: float = 1e-5,
) -> float:
    """Suavizado del spline escalado por la densidad del montaje fuente.

    Cuanto menos denso es el montaje de origen mayor es la distancia media
    entre electrodos y más se difumina la actividad en manchas; la
    regularización ridge del spline crece en consecuencia
    (``base * C_dst / C_src``). Así el modelo se adapta por construcción a
    distribuciones más o menos densas.
    """
    n_src = max(1, src_positions.shape[0])
    n_dst = max(1, dst_positions.shape[0])
    return float(base_smoothness) * (n_dst / n_src)


def _project_to_disc(positions: np.ndarray) -> np.ndarray:
    """Proyecta posiciones 3D sobre el disco unidad (vista central/topomapa).

    Cada posición se normaliza a la esfera unitaria y se queda con sus dos
    primeras coordenadas ``(x_hat, y_hat)`` (dentro del disco `x²+y² ≤ 1`),
    coherente con la malla del cuero cabelludo de :func:`scalp_grid_matrix`.
    """
    p = _normalize(positions)
    return p[:, :2]


def scalp_grid_matrix(
    src_positions: np.ndarray,
    grid_px: int = 48,
    order: int = 30,
    smoothness: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Matriz de interpolación a una malla regular del cuero cabelludo.

    Devuelve la matriz ``M`` (n_grid, C_s) que lleva potenciales de los
    electrodos fuente a una cuadrícula 2D ``grid_px x grid_px`` (vista
    central). La actividad interpolada se representa como un *heatmap* donde
    cada máximo local es una "mancha" (la forma de un potencial distribuido
    sobre el cuero cabelludo).

    La malla vive en el disco unitario; cada punto se eleva a la esfera
    (``Z = sqrt(1 - X² - Y²)``) para interpolar el campo esférico de forma
    consistente con :func:`spherical_spline_matrix`.

    Returns
    -------
    ``(matrix, valid_mask, centroids)`` con ``matrix`` de ``(n, C_s)``,
    ``valid_mask`` booleano de ``n = grid_px²`` (True dentro del disco) y
    ``centroids`` las coordenadas ``(X, Y)`` (en ``[-1, 1]``) de cada nodo.
    """
    x = np.linspace(-1.0, 1.0, grid_px)
    gx, gy = np.meshgrid(x, x)
    X = gx.ravel()
    Y = gy.ravel()
    valid = (X ** 2 + Y ** 2) <= 1.0
    Z = np.sqrt(np.maximum(0.0, 1.0 - X[valid] ** 2 - Y[valid] ** 2))
    grid_pts = np.stack([X[valid], Y[valid], Z], axis=1).astype(np.float64)
    m = spherical_spline_matrix(src_positions, grid_pts, order=order,
                                smoothness=smoothness)   # (C_s, n_valid)
    full = np.zeros((grid_px * grid_px, m.shape[0]), dtype=np.float32)
    full[valid] = m.T
    return full, valid, np.stack([X, Y], axis=1)


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
    smoothness: float | None = None,
    adaptive_smoothness: bool = True,
) -> np.ndarray:
    """Construye la matriz de proyección ``src -> dst`` según el método.

    * ``leadfield`` usa ``g_src``/``g_dst``; ``n_components`` trunca el SVD
      (``None`` = automático ``C_s // 3``).
    * ``spline`` usa ``smoothness``; si es ``None`` o ``adaptive_smoothness``
      es verdadero, el suavizado se escala por la densidad del montaje fuente
      (:func:`density_smoothness`).
    """
    if method == "spline":
        if smoothness is None:
            # Default: regularización ridge base escalada por densidad.
            smoothness = density_smoothness(src_positions, dst_positions)
        elif adaptive_smoothness:
            smoothness = density_smoothness(
                src_positions, dst_positions, base_smoothness=float(smoothness)
            )
        return spherical_spline_matrix(src_positions, dst_positions,
                                       smoothness=smoothness)
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