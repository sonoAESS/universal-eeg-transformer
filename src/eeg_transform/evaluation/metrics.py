"""Métricas de evaluación por ruta y comparación con las matrices analíticas."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from ..config import REFERENCE_KINDS, EEGTransformConfig
from ..logging_conf import get_logger
from ..references import inter_reference_matrix

log = get_logger(__name__)

KINDS = list(REFERENCE_KINDS)


def predict_all_routes(model, refs: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Predice todas las salidas para cada referencia de origen.

    Returns
    -------
    dict
        ``{src: {dst: predicción (n, C) np.float32}}``.
    """
    out = {}
    tensors = {k: tf.convert_to_tensor(v, dtype=tf.float32) for k, v in refs.items()}
    for s in KINDS:
        preds = model(tensors[s], source=s)
        out[s] = {d: preds[d].numpy() for d in KINDS}
    return out


def evaluate_routes(
    model,
    ds,
    split: str = "test",
) -> pd.DataFrame:
    """Calcula métricas reales por ruta en el split pedido.

    Métricas:
        * ``mse``    : error cuadrático medio (V^2).
        * ``rmse``   : raíz del MSE (V).
        * ``mae``    : error absoluto medio (V).
        * ``r``      : correlación de Pearson promedio entre canales.
    """
    idx = ds.split_idx[split]
    refs = {k: ds.refs[k][idx] for k in KINDS}
    preds = predict_all_routes(model, refs)

    rows = []
    for s in KINDS:
        for d in KINDS:
            y_true = refs[d].astype(np.float32)
            y_pred = preds[s][d].astype(np.float32)
            err = y_true - y_pred
            mse = float(np.mean(err ** 2))
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(err)))
            corr = []
            for c in range(y_true.shape[1]):
                t, p = y_true[:, c], y_pred[:, c]
                denom = np.std(t) * np.std(p)
                if np.std(t) < 1e-20:
                    # canal sin varianza (p. ej. Cz en unipolar) no aporta
                    # información; se omite para no sesgar la correlación media
                    continue
                corr.append(float(np.corrcoef(t, p)[0, 1] if denom > 1e-20 else 0.0))
            rows.append(
                {
                    "origen": s,
                    "destino": d,
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "r": float(np.mean(corr)),
                }
            )
    return pd.DataFrame(rows)


def predict_montage_routes(model, sources: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Predice los canales canónicos desde observaciones del montaje fuente.

    Parameters
    ----------
    sources:
        ``{kind: (n, C_s)}`` observaciones de las 4 referencias del montaje
        fuente.
    Returns
    -------
    ``{src: {dst: predicción (n, C)}}``.
    """
    out = {}
    tensors = {k: tf.convert_to_tensor(v, dtype=tf.float32) for k, v in sources.items()}
    for s in KINDS:
        preds = model(tensors[s], source=s)
        out[s] = {d: preds[d].numpy() for d in KINDS}
    return out


def _route_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Métricas de una ruta sobre el subespacio observable (centrado temporal)."""
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    t = y_true - y_true.mean(1, keepdims=True)
    p = y_pred - y_pred.mean(1, keepdims=True)
    err = t - p
    mse = float(np.mean(err ** 2))
    corr = []
    for c in range(t.shape[1]):
        if np.std(t[:, c]) < 1e-20:
            continue
        corr.append(float(np.corrcoef(t[:, c], p[:, c])[0, 1]))
    ve = float(1.0 - np.sum(err ** 2) / (np.sum(t ** 2) + 1e-15))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(err))),
        "r": float(np.mean(corr)) if corr else 0.0,
        "ve": ve,
    }


def evaluate_montage_routes(
    model,
    ds,
    montage_inputs,
    split: str = "test",
) -> pd.DataFrame:
    """Métricas por ruta desde el montaje fuente al espacio canónico.

    Además de las métricas del modelo (<modelo>>), se incluye la línea base
    **analítica** de proyección pura ``obs @ P`` (sin aprendizaje): columnas
    ``rmse_proy``, ``r_proy`` y ``ve_proy``. La ganancia del modelo sobre la
    proyección cuantifica cuánto aporta el refinamiento aprendido.
    """
    idx = ds.split_idx[split]
    sources = {k: montage_inputs.src_refs[split][k] for k in KINDS}
    targets = {k: ds.refs[k][idx] for k in KINDS}
    projection = montage_inputs.projection
    preds = predict_montage_routes(model, sources)

    rows = []
    for s in KINDS:
        baseline_s = sources[s] @ projection
        for d in KINDS:
            st = _route_stats(targets[d], preds[s][d])
            stb = _route_stats(targets[d], baseline_s)
            rows.append({
                "origen": s,
                "destino": d,
                "mse": st["mse"], "rmse": st["rmse"], "mae": st["mae"],
                "r": st["r"], "ve": st["ve"],
                "rmse_proy": stb["rmse"], "r_proy": stb["r"],
                "ve_proy": stb["ve"],
            })
    return pd.DataFrame(rows)


def summarize_montage(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Resumen diagonal/cruzada de una ruta: modelo vs proyección analítica."""
    diag = metrics_df[metrics_df["origen"] == metrics_df["destino"]]
    off = metrics_df[metrics_df["origen"] != metrics_df["destino"]]
    rows = []
    for label, g in (("auto-reconstrucción (diagonal)", diag),
                     ("transformación cruzada", off)):
        rows.append({
            "tipo": label,
            "rmse_uV_model": g["rmse"].mean() * 1e6,
            "r_model": g["r"].mean(),
            "ve_model": g["ve"].mean(),
            "rmse_uV_proy": g["rmse_proy"].mean() * 1e6,
            "r_proy": g["r_proy"].mean(),
            "ve_proy": g["ve_proy"].mean(),
        })
    return pd.DataFrame(rows)


def transfer_error_matrix(
    model,
    ds,
    unipolar_ref_index: int,
    cfg: EEGTransformConfig | None = None,
) -> pd.DataFrame:
    """Compara las matrices efectivas del modelo con las analíticas.

    Para cada ruta ``s->d`` se calcula la norma relativa de Frobenius sobre el
    **subespacio observable**: ``||P(A_modelo - A_analitica)P||_F /
    ||P A_analitica P||_F``, con ``P = I - 11^T/C`` (centrado). El anulado de
    ``P`` elimina el modo constante instantáneo, que no es recuperable desde
    ninguna referencia (``A = T_d @ pinv(T_s)`` ya lo proyecta); sin él las
    normas comparan también esa componente no observable y el error parece
    ~1 incluso para un modelo casi exacto.

    Un error pequeño indica que el autoencoder lineal reprodujo la física
    subyacente en lugar de memorizar pares de entrenamiento.
    """
    C = ds.n_channels
    p = np.eye(C) - np.ones((C, C)) / C  # proyector de centrado (observable)
    matrices = model.transfer_matrices()
    rows = []
    for s in KINDS:
        for d in KINDS:
            analitic = inter_reference_matrix(
                s, d, C, unipolar_ref_index=unipolar_ref_index,
                lead_field=ds.leadfield.matrix,
            )
            learned = np.asarray(matrices[(s, d)], dtype=np.float64)
            a_obs = p @ analitic @ p
            l_obs = p @ learned @ p
            err = float(
                np.linalg.norm(l_obs - a_obs, "fro")
                / (np.linalg.norm(a_obs, "fro") + 1e-15)
            )
            rows.append({"origen": s, "destino": d, "error_fro_rel": err})
    return pd.DataFrame(rows)


def composition_error_table(model, ds) -> pd.DataFrame:
    """Resumen de la consistencia de composición (propiedad de grupo).

    Para cada triplete ``s -> d -> u`` se mide
    ``||P(A_{s->d} A_{d->u} - A_{s->u})P||_F / ||P A_{s->u} P||_F``.

    * Un modelo con estructura de grupo exacta (variante ``group``) da 0.
    * El encadenado analítico ``T_d pinv(T_s)`` no satisface la propiedad
      (error del orden de 1): compone mal aunque cada ruta individual sea
      algebraica.
    """
    cons = model.composition_error()
    rows = [
        {"origen": s, "destino": d, "final": u, "error_comp": v}
        for (s, d, u), v in cons.items()
    ]
    df = pd.DataFrame(rows)
    # resumen compacto: máximo y media por par origen->destino
    summary = (
        df.groupby(["origen", "destino"], as_index=False)["error_comp"]
        .agg(["mean", "max"])
        .rename(columns={"mean": "comp_medio", "max": "comp_max"})
    )
    return summary.reset_index(drop=True)


def predict_multiconfig_routes(
    model, data, split: str = "test",
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Predice TODAS las rutas y configuraciones del modelo multi-montaje.

    Returns
    -------
    ``{configuración: {origen: {destino: (n, C_s)}}}``.
    """
    out = {}
    import tensorflow as tf

    for label in data.order:
        mc = data.configs[label]
        tensors = {
            k: tf.convert_to_tensor(mc.refs[split][k], dtype=tf.float32)
            for k in KINDS
        }
        out[label] = {}
        for s in KINDS:
            preds = model(tensors[s], cfg=label, source=s)
            out[label][s] = {d: preds[d].numpy() for d in KINDS}
    return out


def evaluate_multiconfig_routes(
    model, data, split: str = "test",
) -> pd.DataFrame:
    """Métricas por ruta intra-configuración + línea base analítica.

    Para cada configuración y cada ruta ``s->d`` se comparan la predicción del
    modelo (en esa misma configuración, ``(n, C_s)``) con la referencia
    canónica del montaje. Además se reporta la **línea base analítica**
    ``T_d @ pinv(T_s)`` calculada con las matrices de referencia del propio
    montaje (columnas ``rmse_ana``, ``r_ana``, ``ve_ana``).

    El error se mide en el subespacio observable (centrado por instante): el
    modo constante de referencia no es recuperable.
    """
    rows = []
    for label in data.order:
        mc = data.configs[label]
        targets = {k: mc.refs[split][k] for k in KINDS}
        sources = {k: mc.refs[split][k] for k in KINDS}
        preds = predict_multiconfig_routes(model, data, split)[label]
        n_c = mc.n_channels
        for s in KINDS:
            for d in KINDS:
                st = _route_stats(targets[d], preds[s][d])
                ana = inter_reference_matrix(
                    s, d, n_c,
                    unipolar_ref_index=mc.unipolar_ref_index,
                    lead_field=mc.leadfield,
                    rest_rcond=mc.rest_rcond,
                    positions=np.asarray(mc.positions, dtype=np.float64),
                )
                sta = _route_stats(targets[d], sources[s] @ ana)
                rows.append({
                    "config": label,
                    "origen": s,
                    "destino": d,
                    "mse": st["mse"], "rmse": st["rmse"], "mae": st["mae"],
                    "r": st["r"], "ve": st["ve"],
                    "rmse_ana": sta["rmse"], "r_ana": sta["r"],
                    "ve_ana": sta["ve"],
                })
    return pd.DataFrame(rows)


def summarize_multiconfig(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Resumen por configuración (diagonal/cruzada) del modelo vs analítico."""
    rows = []
    for label, g in metrics_df.groupby("config", sort=False):
        diag = g[g["origen"] == g["destino"]]
        off = g[g["origen"] != g["destino"]]
        rows.append({
            "config": label,
            "n_rutas": len(g),
            "rmse_diag_uV": diag["rmse"].mean() * 1e6,
            "r_diag": diag["r"].mean(),
            "rmse_cross_uV": off["rmse"].mean() * 1e6,
            "r_cross": off["r"].mean(),
            "ve_cross": off["ve"].mean(),
            "rmse_ana_cross_uV": off["rmse_ana"].mean() * 1e6,
            "r_ana_cross": off["r_ana"].mean(),
            "ve_ana_cross": off["ve_ana"].mean(),
        })
    return pd.DataFrame(rows)


def _grid_valid(grid_px: int) -> np.ndarray:
    """Máscara del disco de la malla compartida (igual que ``scalp_grid_matrix``)."""
    x = np.linspace(-1.0, 1.0, grid_px)
    gx, gy = np.meshgrid(x, x)
    return (gx.ravel() ** 2 + gy.ravel() ** 2) <= 1.0


def evaluate_multiconfig_surface_routes(
    model, data, split: str = "test",
) -> pd.DataFrame:
    """Métricas en el CAMPO DE SUPERFICIE (malla compartida) por ruta.

    Para cada configuración y ruta ``s->d`` se interpolan predicción y verdad
    a la malla del cuero cabelludo con la matriz fija ``S_s`` (electrodos ->
    ``n_grid``) y se estiman RMSE/r/VE sobre el **patrón espacial** (solo nodos
    válidos del disco). Cuantifica cuán fiel es el "heatmap" predicho, no solo
    cada electrodo individualmente.
    """
    grid_px = int(round(np.sqrt(data.configs["canonical"].surface.shape[0])))
    valid = _grid_valid(grid_px)
    rows = []
    for label in data.order:
        mc = data.configs[label]
        S = mc.surface.astype(np.float64)[valid, :]          # (n_valid, C_s)
        targets = {k: mc.refs[split][k] for k in KINDS}
        preds = predict_multiconfig_routes(model, data, split)[label]
        for s in KINDS:
            for d in KINDS:
                f_true = targets[d] @ S.T
                f_pred = preds[s][d] @ S.T
                st = _route_stats(f_true, f_pred)
                rows.append({
                    "config": label,
                    "origen": s,
                    "destino": d,
                    "rmse_field": st["rmse"],
                    "r_field": st["r"],
                    "ve_field": st["ve"],
                })
    return pd.DataFrame(rows)


def summarize_multiconfig_surface(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Resumen del campo de superficie por configuración (diag/cruzada)."""
    rows = []
    for label, g in metrics_df.groupby("config", sort=False):
        diag = g[g["origen"] == g["destino"]]
        off = g[g["origen"] != g["destino"]]
        rows.append({
            "config": label,
            "rmse_field_diag_uV": diag["rmse_field"].mean() * 1e6,
            "r_field_diag": diag["r_field"].mean(),
            "ve_field_diag": diag["ve_field"].mean(),
            "rmse_field_cross_uV": off["rmse_field"].mean() * 1e6,
            "r_field_cross": off["r_field"].mean(),
            "ve_field_cross": off["ve_field"].mean(),
        })
    return pd.DataFrame(rows)


def summarize(metrics_df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Resumen de métricas por tipo de ruta (diagonal / cruzada)."""
    diag = metrics_df[metrics_df["origen"] == metrics_df["destino"]]
    off = metrics_df[metrics_df["origen"] != metrics_df["destino"]]
    rows = [
        {"tipo": "auto-reconstrucción (diagonal)",
         "mse": diag["mse"].mean(), "rmse": diag["rmse"].mean(),
         "mae": diag["mae"].mean(), "r": diag["r"].mean()},
        {"tipo": "transformación cruzada",
         "mse": off["mse"].mean(), "rmse": off["rmse"].mean(),
         "mae": off["mae"].mean(), "r": off["r"].mean()},
    ]
    return pd.DataFrame(rows)


def evaluate_multiconfig_field_agreement(
    model, data, split: str = "test",
) -> pd.DataFrame:
    """D9/surrogate: acuerdo de campo entre configuraciones sobre la malla.

    Para cada ruta ``s->d`` interpola la predicción de cada configuración a la
    malla compartida del cuero cabelludo (con su ``S_s`` fija) y mide el RMSE
    por pares de configuraciones. Un valor bajo indica que 10-20, canónico y
    densos describen el **mismo** potencial de superficie (consistencia B4).
    """
    preds = predict_multiconfig_routes(model, data, split)
    labels = data.order
    rows = []
    for s in KINDS:
        for d in KINDS:
            fields = {}
            # preds[label][s][d] es (n, C_s); campo = pred @ S.T
            for label in labels:
                arr = preds[label][s][d]
                S = np.asarray(data.configs[label].surface, dtype=np.float64)
                fields[label] = arr @ S.T
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    a, b = fields[labels[i]], fields[labels[j]]
                    st = _route_stats(a, b)
                    rows.append({
                        "origen": s, "destino": d,
                        "config_a": labels[i], "config_b": labels[j],
                        "rmse_field": st["rmse"],
                        "r_field": st["r"],
                        "ve_field": st["ve"],
                    })
    return pd.DataFrame(rows)


def external_topomap_loader(path: str):  # pragma: no cover - D9 placeholder
    """D9 (futuro): cargar topomapas reales (p. ej. de localización de fuentes
    o BIDS) para validar el campo predicho contra una medición independiente.

    ``eegbci`` no incluye topomapas reales, por lo que la validación de
    ``multi_heatmap_v2`` usa el acuerdo entre configuraciones
    (:func:`evaluate_multiconfig_field_agreement`) como sustituto. Esta función
    es un punto de extensión cuando exista una fuente externa de campos reales.
    """
    raise NotImplementedError(
        "external_topomap_loader es un placeholder (D9): implementar la "
        f"carga de topomapas reales desde '{path}' cuando esté disponible."
    )


# Bandas clínicas estándar (Hz); gamma limitada por el bandpass del dataset.
SPECTRAL_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _band_filter(x: np.ndarray, sfreq: float, lo: float, hi: float) -> np.ndarray:
    """Filtrado FFT por columnas (eje tiempo 0) con máscara suave de bordes."""
    spec = np.fft.rfft(x, axis=0)
    freqs = np.fft.rfftfreq(x.shape[0], d=1.0 / sfreq)
    mask = (freqs >= lo) & (freqs <= hi)
    filtered = np.zeros_like(spec)
    filtered[mask] = spec[mask]
    return np.fft.irfft(filtered, n=x.shape[0], axis=0)


def spectral_band_table(
    true: np.ndarray,
    pred: np.ndarray,
    sfreq: float = 160.0,
    bands: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """RMSE/ve por banda espectral de una ruta (diagnóstico temporal).

    Compara ``true`` y ``pred`` ``(T, C)`` banda a banda: dónde se concentra
    el error indica si la cabeza dinámica aporta en las oscilaciones (alpha,
    beta) o si el residuo vive en la banda ancha. Filtrado FFT directo,
    válido offline para análisis (los bordes de bloque pueden introducir
    fugas menores).
    """
    bands = bands or SPECTRAL_BANDS
    rows = []
    for name, (lo, hi) in bands.items():
        if hi >= sfreq / 2:
            hi = sfreq / 2 - 1e-3
        if lo >= hi:
            continue
        t_b = _band_filter(true.astype(np.float64), sfreq, lo, hi)
        p_b = _band_filter(pred.astype(np.float64), sfreq, lo, hi)
        err = t_b - p_b
        energy = float(np.sum(t_b ** 2)) + 1e-30
        rows.append({
            "banda": name,
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "ve": float(1.0 - np.sum(err ** 2) / energy),
            "energia_frac": float(np.sum(t_b ** 2) / (np.sum(true ** 2) + 1e-30)),
        })
    return pd.DataFrame(rows)