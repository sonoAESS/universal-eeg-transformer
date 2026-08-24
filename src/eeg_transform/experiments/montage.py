"""Experimento de unificación de montajes hacia un espacio canónico.

Dado un dataset real en un montaje denso (p. ej. 64 canales 10-10 de
``eegbci``), se simulan montajes de menor densidad (p. ej. 19 canales 10-20,
subconjunto exacto de los 64) y se evalúa cuánto cuesta volver al espacio
canónico con cada método de proyección:

* ``nearest``   : asignar a cada canal canónico el valor del electrodo más
  cercano (línea base geométrica mínima).
* ``spline``    : interpolación esférica (Perrin), estilo *topomapa/heatmap*.
* ``leadfield`` : proyección por solución inversa con el lead field analítico.

**Metodología fiel:** la señal ancla es la referencia al infinito (REST) del
montaje completo; observar solo sus columnas en los ``Cs`` electrodos del
montaje fuente equivale a una grabación real de ``Cs`` canales. Sobre esa
observación se computan las **cuatro referencias del propio montaje fuente**
(con sus operadores de ``Cs`` canales y su lead field propio) y **luego** se
proyectan al espacio canónico (64). La verdad (ground truth) son las mismas
referencias canónicas del montaje completo.

Esto evita el sesgo de muestrear referencias canónicas (operadores de 64) en
los 19 electrodos, que penaliza especialmente a la referencia bipolar (un
operador local cuya cadena cambia con el montaje).

Además de la reconstrucción, el experimento entrena el transformador
universal sobre los datos proyectados (end-to-end) y lo compara contra el
modelo canónico.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import REFERENCE_KINDS, EEGTransformConfig
from ..data.dataset import MultiReferenceDataset, build_dataset
from ..logging_conf import get_logger
from ..mapping import build_projection, round_trip_error, select_subset
from ..references import build_reference_matrix

log = get_logger(__name__)

# Montaje 10-20: las 19 cimas clásicas (subconjunto exacto de los 64 de
# eegbci, verificado). El orden define la cadena del montaje bipolar.
MONTAGE_10_20 = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8",
    "Fz", "Cz", "Pz",
]

# Montaje 10-10 intermedio (39 canales): subconjunto real de los 64.
MONTAGE_10_10_39 = [
    "Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2",
]

METHODS = ("nearest", "spline", "leadfield")

MONTAGE_NAMES: dict[str, list[str]] = {
    "10-20": MONTAGE_10_20,
    "10-10": MONTAGE_10_10_39,
}


def resolve_montage(montage: str | list[str], ch_names: list[str]) -> list[str]:
    """Resuelve un montaje (nombre estándar o lista de canales) a canales.

    Acepta nombres conocidos (``10-20``, ``10-10``) o una lista explícita de
    canales; descarta los canales que no existan en ``ch_names``.
    """
    if isinstance(montage, str):
        keep = MONTAGE_NAMES.get(montage)
        if keep is None:
            raise ValueError(
                f"Montaje desconocido '{montage}' (válidos: "
                f"{list(MONTAGE_NAMES)} o una lista de canales)."
            )
        return keep
    return [c for c in montage if c in ch_names]


def build_montage_inputs(
    ds: MultiReferenceDataset,
    cfg: EEGTransformConfig,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> MontageInputs:
    """Prepara las observaciones del montaje fuente y su proyección.

    * Proyección fija ``P``: ``leadfield`` (solución inversa, SVD truncado a
      ``n_components`` o automático ``C_s//3``) o ``spline`` (interpolación
      esférica con suavizado adaptativo por la densidad del montaje).
    * ``src_refs``: por split, las 4 referencias del montaje fuente
      (metodología fiel, ver :func:`simulate_source_observation`).
    """
    mapping = cfg.mapping
    keep = resolve_montage(mapping.montage, ds.ch_names)
    _, src_pos, src_idx = select_subset(ds.ch_names, ds.ch_positions, keep=keep)
    canon_pos = np.asarray(ds.ch_positions)
    g_src = ds.leadfield.matrix[src_idx, :]

    projection = build_projection(
        mapping.method,
        src_pos,
        canon_pos,
        g_src=g_src,
        g_dst=ds.leadfield.matrix,
        n_components=mapping.n_components,
        smoothness=mapping.smoothness,
        adaptive_smoothness=mapping.adaptive_smoothness,
    )
    src_refs = {
        split: simulate_source_observation(ds, src_idx, split) for split in splits
    }
    log.info("Montaje fuente: %s (%d canales) via '%s'", mapping.montage,
             len(keep), mapping.method)
    return MontageInputs(
        method=mapping.method,
        montage=str(mapping.montage),
        src_names=[ds.ch_names[i] for i in src_idx],
        src_idx=src_idx,
        src_positions=src_pos,
        projection=projection,
        src_refs=src_refs,
    )


@dataclass
class MonteResult:
    """Resultados de reconstrucción de un montaje/método (promedio por ref)."""

    method: str
    montage: str
    n_channels: int
    rmse_uV: float
    mae_uV: float
    r: float
    ve: float
    route: str


@dataclass
class MontageInputs:
    """Observaciones del montaje fuente + proyección al espacio canónico.

    * ``src_refs[split][kind]`` : observación ``(n, C_s)`` de la referencia
      ``kind`` desde el montaje fuente (las 4 referencias del propio montaje).
    * ``projection``            : matriz fija ``(C_s, C)`` de proyección al
      espacio canónico (solución inversa con lead field o spline/heatmap).
    """

    method: str
    montage: str
    src_names: list[str]
    src_idx: np.ndarray
    src_positions: np.ndarray
    projection: np.ndarray
    src_refs: dict[str, dict[str, np.ndarray]]


def _source_operators(
    n_src: int, src_idx: np.ndarray, ds: MultiReferenceDataset, uni_ref_ch: str
):
    """Matrices de referencia del montaje fuente (operadores ``Cs``-canales)."""
    g_src = ds.leadfield.matrix[src_idx, :]
    src_positions = np.asarray(ds.ch_positions)[src_idx]
    out: dict[str, np.ndarray] = {}
    local_names = [ds.ch_names[i] for i in src_idx]
    uni_local = local_names.index(uni_ref_ch)
    for k in REFERENCE_KINDS:
        out[k] = build_reference_matrix(
            k, n_src, unipolar_ref_index=uni_local, lead_field=g_src,
            positions=src_positions,
        )
    return out


def simulate_source_observation(
    ds: MultiReferenceDataset,
    src_idx: np.ndarray,
    split: str = "test",
) -> np.ndarray:
    """Observación (T, Cs) de las cuatro referencias del montaje fuente.

    Ancla: la REST del montaje completo (referencia al infinito). Sus columnas
    en los electrodos fuente son los potenciales que vería una grabación real
    de ``Cs`` canales. De ahí se computan las 4 referencias fuente.
    """
    idx = ds.split_idx[split]
    inf = ds.refs["rest"][idx][:, src_idx]           # (T, Cs) al infinito
    n_src = inf.shape[1]
    ops = _source_operators(
        n_src, src_idx, ds, uni_ref_ch=ds.meta.get("unipolar_ref_ch", "Cz")
    )
    return {k: inf @ ops[k] for k in REFERENCE_KINDS}


def reconstruct(
    ds: MultiReferenceDataset,
    method: str,
    keep: list[str],
    split: str = "test",
    sub: int | None = None,
) -> MonteResult:
    """Proyecta señales del montaje fuente al canónico y mide el error.

    Para cada referencia: ``obs_src[k]`` (observada desde el montaje fuente)
    se proyecta ``obs_src[k] @ P_back`` y se compara contra la referencia
    canónica ``ds.refs[k]`` en el subespacio observable (centrado por
    instante).
    """
    _, _, src_idx = select_subset(ds.ch_names, ds.ch_positions, keep=keep)
    src_pos = np.asarray(ds.ch_positions)[src_idx]
    canon_pos = np.asarray(ds.ch_positions)
    g_src = ds.leadfield.matrix[src_idx, :]

    p_back = build_projection(
        method, src_pos, canon_pos, g_src=g_src, g_dst=ds.leadfield.matrix
    )

    obs = simulate_source_observation(ds, src_idx, split)
    idx = ds.split_idx[split]
    if sub:
        obs = {k: v[:sub] for k, v in obs.items()}
        idx = idx[:sub]

    rows: dict[str, float] = {"rmse_uV": 0.0, "mae_uV": 0.0, "r": 0.0, "ve": 0.0}
    worst = 1.0e9
    worst_route = ""
    for k in REFERENCE_KINDS:
        m = round_trip_error(
            signals_true=ds.refs[k][idx],  # referencia canónica verdadera
            signals_reduced=obs[k],        # observada desde montaje fuente
            p_back=p_back,
        )
        for key in rows:
            rows[key] += m[key]
        if m["ve"] < worst:
            worst, worst_route = m["ve"], k

    n = float(len(REFERENCE_KINDS))
    return MonteResult(
        method=method, montage=_montage_label(keep), n_channels=len(keep),
        rmse_uV=rows["rmse_uV"] / n, mae_uV=rows["mae_uV"] / n,
        r=rows["r"] / n, ve=rows["ve"] / n, route=worst_route,
    )


def _montage_label(keep: list[str]) -> str:
    for name, m in MONTAGE_NAMES.items():
        if list(keep) == list(m):
            return name
    return f"montaje-{len(keep)}"


def run_reconstruction(
    ds: MultiReferenceDataset,
    montages: list[str] | None = None,
    methods: list[str] | None = None,
) -> pd.DataFrame:
    """Bucle de reconstrucción para cada método sobre cada montaje."""
    montages = montages or ["10-20"]
    methods = methods or list(METHODS)
    rows: list[MonteResult] = []
    for m_name in montages:
        keep = MONTAGE_NAMES[m_name]
        for method in methods:
            res = reconstruct(ds, method, keep)
            rows.append(res)
            log.info("%s %s: RMSE %.2f µV, r=%.4f, ve=%.3f (peor %s)",
                     res.method, res.montage, res.rmse_uV, res.r, res.ve,
                     res.route)
    return pd.DataFrame(
        [r.__dict__ for r in rows],
        columns=[f.name for f in MonteResult.__dataclass_fields__.values()],
    )


def run_cmd(
    cfg: EEGTransformConfig, montages: list[str] | None = None
) -> None:
    """Comando del CLI: evalúa la reconstrucción por método y montaje."""
    ds = build_dataset(cfg)
    df = run_reconstruction(ds, montages)
    print("\n================ RECONSTRUCCION (TEST) ================")
    print(df.round(4).to_string(index=False))
    out = Path(cfg.dataset.cache_dir).parent / "mapping_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("Resultados guardados en %s", out)