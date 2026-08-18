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