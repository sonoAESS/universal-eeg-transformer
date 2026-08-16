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