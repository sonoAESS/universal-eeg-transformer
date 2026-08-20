"""Entrenamiento conjunto sobre múltiples configuraciones de electrodos.

La variante ``multi_montage`` entrena el autoencoder universal de forma
**balanceada** sobre varias distribuciones de electrodos a la vez (p. ej. 19
canales 10-20, el montaje canónico de 64 electrodos, y montajes densos
simulados de 128 y 256). Cada configuración se "embebe" en el espacio canónico
con una matriz fija ``P_s`` (C_s -> 64) y se lee con una matriz fija ``Q_s``
(64 -> C_s); las 8 matrices aprendidas (W_enc/W_dec) se comparten entre todas
las configuraciones. El modelo acepta **cualquier** configuración y predice, en
esa misma configuración, las medidas con otra referencia.

Metodología fiel de simulación de un montaje fuente (coherente con
``experiments.montage``):

1. La ancla es la referencia al infinito (REST) del montaje canónico: sus
   valores son el potencial a infinito en cada posición de electrodo.
2. Para montajes densos simulados (``dense-128``/``dense-256``), el campo se
   interpola esféricamente (spline de Perrin) a las posiciones del nuevo montaje.
   Para subconjuntos reales (``10-20``/``10-10``) se seleccionan directamente
   las columnas correspondientes (posiciones coincidentes).
3. Sobre ese campo al infinito se computan las **cuatro referencias del propio
   montaje** con sus operadores de ``C_s`` canales y su lead field propio.

Balanceo: todas las configuraciones comparten los mismos índices de split y el
mismo presupuesto de muestras por split (``MappingConfig.multi_max_samples_per_split``),
de modo que cada configuración aporta exactamente la misma fracción de datos.

Resultados coherentes: para ``10-20`` este generador reproduce las
observaciones de ``experiments.montage.simulate_source_observation`` (fila a
fila, las mismas 4 referencias del montaje fuente).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import EEGTransformConfig, REFERENCE_KINDS
from ..data.dataset import MultiReferenceDataset
from ..leadfield import compute_lead_field
from ..logging_conf import get_logger
from ..mapping import (
    build_projection,
    scalp_grid_matrix,
    select_subset,
    spherical_spline_matrix,
)
from ..references import build_reference_matrix
from .montage import MONTAGE_10_20, MONTAGE_10_10_39

log = get_logger(__name__)

KINDS = list(REFERENCE_KINDS)

MULTI_CACHE_SUBDIR = "multiconfig"
DENSE_PREFIX = "dense-"

# Subconjuntos reales del montaje canónico (posiciones coincidentes exactas).
SUBSET_MONTAGES: dict[str, list[str]] = {
    "10-20": MONTAGE_10_20,
    "10-10": MONTAGE_10_10_39,
}


def is_dense(label: str) -> bool:
    return label.startswith(DENSE_PREFIX)


def parse_dense(label: str) -> int:
    return int(label[len(DENSE_PREFIX):])


def is_subset(label: str) -> bool:
    return label in SUBSET_MONTAGES


def dense_cap_positions(
    n_electrodes: int,
    radius: float = 1.0,
    z_min: float = -0.06,
    seed: int = 7,
) -> np.ndarray:
    """`n` posiciones cuasi-uniformes sobre el casquete superior de la esfera.

    Se muestrean puntos de la *fibonacci sphere*, se filtran a la banda ya
    cubierta por el montaje canónico (``z >= z_min`` en la esfera unitaria) y
    se conservan los ``n`` de mayor elevación con un ligero ruido determinista
    (mezcla azimutal). El resultado es un arreglo denso de electrodos sobre el
    cuero cabelludo, representativo de sistemas de alta densidad (estilo
    BioSemi/HydroCel) limitados al casquete.

    Devuelve posiciones de norma ``radius`` (m) ordenadas por ángulo azimutal
    para una cadena bipolar suave.
    """
    rng = np.random.default_rng(seed)
    count = max(8_192, n_electrodes * 16)
    i = np.arange(count) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / count)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    pts = np.stack(
        [np.sin(phi) * np.cos(theta),
         np.sin(phi) * np.sin(theta),
         np.cos(phi)],
        axis=1,
    )
    keep = pts[:, 2] >= z_min
    pts = pts[keep]
    if len(pts) < n_electrodes:
        raise ValueError(
            f"No hay suficientes posiciones con z>={z_min} para {n_electrodes} "
            f"electrodos (se obtuvieron {len(pts)})."
        )
    elev = pts[:, 2] + 1e-3 * rng.standard_normal(len(pts))
    idx = np.argsort(elev)[-n_electrodes:]
    pts = pts[idx]
    order = np.argsort(np.arctan2(pts[:, 1], pts[:, 0]))
    return pts[order] * radius


def _budget_indices(full_idx: np.ndarray, budget: int) -> np.ndarray:
    """Submuestra índices con equiespaciado determinista (mismo presupuesto)."""
    n = len(full_idx)
    if n <= budget:
        return np.asarray(full_idx)
    return full_idx[np.linspace(0, n - 1, budget, dtype=int)]


def _nearest_column(ch_positions: np.ndarray, target: np.ndarray) -> int:
    return int(np.argmin(np.linalg.norm(ch_positions - target, axis=1)))


@dataclass
class MontageConfig:
    """Una configuración de electrodos del entrenamiento multi-montaje.

    * ``projection``: matriz fija ``(C_s, C)`` que embebe cada observación del
      montaje en el espacio canónico (entrada del autoencoder).
    * ``out_map``   : matriz fija ``(C, C_s)`` que devuelve la predicción
      canónica (64 canales) a las posiciones de esta configuración (salida).
    * ``leadfield`` : lead field del propio montaje ``(C_s, N)`` (operadores
      de referencia y línea base analítica).
    * ``refs``      : ``refs[split][kind]`` con ``(n_budget, C_s)`` muestras.
    """

    label: str
    names: list[str]
    positions: np.ndarray
    n_channels: int
    projection: np.ndarray
    out_map: np.ndarray
    leadfield: np.ndarray
    unipolar_ref_index: int
    refs: dict[str, dict[str, np.ndarray]]
    surface: np.ndarray | None = None


@dataclass
class MultiMontageData:
    """Conjunto de configuraciones de electrodos para ``multi_montage``.

    Todas las configuraciones comparten los mismos índices (``budget_idx``) en
    cada split, luego las proporciones entre configuraciones son idénticas por
    construcción (balanceo).
    """

    configs: dict[str, MontageConfig]
    order: list[str]
    budget_idx: dict[str, np.ndarray]

    def __getitem__(self, label: str) -> MontageConfig:
        return self.configs[label]

    def n_budget(self, split: str) -> int:
        return len(self.budget_idx[split])


def _resolve_config_canonical(ds: MultiReferenceDataset, budget: int):
    """Configuración canónica: los electrodos nativos (P=Q=I)."""
    C = ds.n_channels
    pos = np.asarray(ds.ch_positions, dtype=np.float64)
    return {
        "label": "canonical",
        "names": list(ds.ch_names),
        "positions": pos,
        "n_channels": C,
        "projection": np.eye(C, dtype=np.float32),
        "out_map": np.eye(C, dtype=np.float32),
        "leadfield": ds.leadfield.matrix.astype(np.float32),
        "unipolar_ref_index": ds.ch_names.index(
            ds.meta.get("unipolar_ref_ch", "Cz")
        ),
        "budget": {s: _budget_indices(ds.split_idx[s], budget) for s in ds.split_idx},
    }


def _resolve_config_subset(
    ds: MultiReferenceDataset, label: str, cfg: EEGTransformConfig, budget: int
):
    keep = SUBSET_MONTAGES[label]
    names, positions, src_idx = select_subset(
        ds.ch_names, np.asarray(ds.ch_positions), keep=keep
    )
    positions = np.asarray(positions, dtype=np.float64)
    canon_pos = np.asarray(ds.ch_positions, dtype=np.float64)
    g_cfg = ds.leadfield.matrix[src_idx, :].astype(np.float32)
    mapping = cfg.mapping
    projection = build_projection(
        mapping.method, positions, canon_pos,
        g_src=g_cfg, g_dst=ds.leadfield.matrix,
        n_components=mapping.n_components,
        smoothness=mapping.smoothness,
        adaptive_smoothness=mapping.adaptive_smoothness,
    )
    out_map = None  # se fija como selección exacta en build_multiconfig
    uni_local = names.index(ds.meta.get("unipolar_ref_ch", "Cz"))
    return {
        "label": label,
        "names": names,
        "positions": positions,
        "n_channels": len(names),
        "projection": projection,
        "out_map": out_map,
        "leadfield": g_cfg,
        "unipolar_ref_index": uni_local,
        "src_idx": src_idx,
        "budget": {s: _budget_indices(ds.split_idx[s], budget) for s in ds.split_idx},
    }


def _resolve_config_dense(
    ds: MultiReferenceDataset, label: str, cfg: EEGTransformConfig, budget: int
):
    n = parse_dense(label)
    canon_pos = np.asarray(ds.ch_positions, dtype=np.float64)
    radius = float(np.median(np.linalg.norm(canon_pos, axis=1)))
    positions = dense_cap_positions(n, radius=radius)
    names = [f"HD{i:03d}" for i in range(n)]
    g_cfg = compute_lead_field(
        names,
        positions,
        head_radius=cfg.leadfield.head_radius,
        rel_radii=cfg.leadfield.rel_radii,
        sigmas=cfg.leadfield.sigmas,
        src_grid_mm=cfg.leadfield.src_grid_mm,
        brain_radius=cfg.leadfield.brain_radius,
        verbose=False,
    ).matrix.astype(np.float32)
    mapping = cfg.mapping
    projection = build_projection(
        mapping.method, positions, canon_pos,
        g_src=g_cfg, g_dst=ds.leadfield.matrix,
        n_components=mapping.n_components,
        smoothness=mapping.smoothness,
        adaptive_smoothness=mapping.adaptive_smoothness,
    )
    out_map = spherical_spline_matrix(
        canon_pos, positions,
        smoothness=mapping.smoothness,
    )
    cz_pos = canon_pos[ds.ch_names.index(ds.meta.get("unipolar_ref_ch", "Cz"))]
    uni_local = _nearest_column(positions, cz_pos)
    return {
        "label": label,
        "names": names,
        "positions": positions,
        "n_channels": n,
        "projection": projection,
        "out_map": out_map,
        "leadfield": g_cfg,
        "unipolar_ref_index": uni_local,
        "budget": {s: _budget_indices(ds.split_idx[s], budget) for s in ds.split_idx},
    }


def _build_config_refs(
    ds: MultiReferenceDataset,
    spec: dict[str, Any],
):
    """Referencias del montaje (metodología fiel) para todos los splits."""
    label = spec["label"]
    if label == "canonical":
        return {
            split: {
                k: ds.refs[k][idx].astype(np.float32)
                for k in KINDS
            }
            for split, idx in spec["budget"].items()
        }
    rest_rows = {
        split: ds.refs["rest"][idx].astype(np.float32)
        for split, idx in spec["budget"].items()
    }
    n_c = spec["n_channels"]
    ops = {
        k: build_reference_matrix(
            k, n_c,
            unipolar_ref_index=spec["unipolar_ref_index_ops"],
            lead_field=spec["leadfield"],
        ).astype(np.float32)
        for k in KINDS
    }
    refs = {}
    for split, rest in rest_rows.items():
        if label in SUBSET_MONTAGES:
            field = rest[:, spec["src_idx"]]      # selección exacta de columnas
        else:
            field = rest @ spec["interp"]         # interpolación esférica (C_s)
        refs[split] = {k: field @ ops[k] for k in KINDS}
    return refs


def _config_surface(positions: np.ndarray, grid_px: int) -> np.ndarray:
    """Mapa fijo ``(grid_px², C_s)``: electrodos de la config. -> malla compartida.

    Es la matriz de interpolación al heatmap del cuero cabelludo (la misma
    ``scalp_grid_matrix`` usada por las figuras), de modo que el campo de
    superficie que se lee por configuración vive siempre en la misma malla.
    """
    return scalp_grid_matrix(positions, grid_px=grid_px)[0].astype(np.float32)


def build_multiconfig(
    ds: MultiReferenceDataset,
    cfg: EEGTransformConfig,
    force: bool = False,
) -> MultiMontageData:
    """Construye (o carga de caché) las configuraciones de ``multi_montage``.

    Cada configuración se materializa en ``<cache_dir>/multiconfig/<label>.npz``
    con las 4 referencias por split, la proyección ``P_s``, el mapa de salida
    ``Q_s``, el lead field propio y las posiciones. El lead field del montaje
    canónico se reutiliza; los densos se calculan con el modelo analítico.
    """
    budget = cfg.mapping.multi_max_samples_per_split
    cache_dir = Path(cfg.dataset.cache_dir) / MULTI_CACHE_SUBDIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    specs: list[dict[str, Any]] = []
    for label in cfg.mapping.configs:
        if label == "canonical":
            specs.append(_resolve_config_canonical(ds, budget))
        elif is_subset(label):
            specs.append(_resolve_config_subset(ds, label, cfg, budget))
        elif is_dense(label):
            specs.append(_resolve_config_dense(ds, label, cfg, budget))
        else:
            raise ValueError(f"Configuración desconocida: '{label}'")
    canon_pos = np.asarray(ds.ch_positions, dtype=np.float64)
    for spec in specs:
        if spec["label"] == "canonical":
            continue
        # operador unipolar del propio montaje: índice dentro del montaje
        spec["unipolar_ref_index_ops"] = spec["unipolar_ref_index"]
        if spec["label"] in SUBSET_MONTAGES:
            spec["interp"] = None
            spec["out_map"] = _subset_out_map(canon_pos, spec["positions"])
        else:
            spec["interp"] = spherical_spline_matrix(
                canon_pos, spec["positions"],
                smoothness=cfg.mapping.smoothness,
            )

    configs: dict[str, MontageConfig] = {}
    for spec in specs:
        label = spec["label"]
        cache_file = cache_dir / f"{label}.npz"
        if cache_file.exists() and not force:
            configs[label] = _load_config_cache(cache_file, spec, ds, budget)
            if configs[label].surface is None:
                configs[label].surface = _config_surface(
                    spec["positions"], cfg.mapping.grid_px
                ).astype(np.float32)
                log.info("Mapa de superficie de '%s' recalculado (caché previa).",
                         label)
            log.info("Configuración '%s' cargada de caché (%s)", label, cache_file)
            continue
        refs = _build_config_refs(ds, spec)
        mc = MontageConfig(
            label=label,
            names=spec["names"],
            positions=spec["positions"].astype(np.float32),
            n_channels=spec["n_channels"],
            projection=spec["projection"],
            out_map=spec["out_map"],
            leadfield=spec["leadfield"],
            unipolar_ref_index=spec["unipolar_ref_index"],
            refs=refs,
            surface=_config_surface(spec["positions"], cfg.mapping.grid_px),
        )
        _save_config_cache(mc, ds, cache_file)
        configs[label] = mc
        log.info("Configuración '%s' construida (%d canales) → %s",
                 label, mc.n_channels, cache_file)

    order = list(cfg.mapping.configs)
    budget_idx = {s: np.asarray(specs[0]["budget"][s]) for s in ("train", "val", "test")}
    return MultiMontageData(configs=configs, order=order, budget_idx=budget_idx)


def _subset_out_map(canon_pos: np.ndarray, subset_pos: np.ndarray) -> np.ndarray:
    """Mapa de salida exacto para subconjuntos reales: selección de columnas."""
    C = canon_pos.shape[0]
    n_c = subset_pos.shape[0]
    out = np.zeros((C, n_c), dtype=np.float32)
    for j in range(n_c):
        i = int(np.argmin(np.linalg.norm(canon_pos - subset_pos[j], axis=1)))
        out[i, j] = 1.0
    return out


def _save_config_cache(mc: MontageConfig, ds: MultiReferenceDataset, path: Path) -> None:
    payload: dict[str, Any] = {
        "label": mc.label,
        "names": np.array(mc.names, dtype=object),
        "positions": mc.positions,
        "projection": mc.projection,
        "out_map": mc.out_map,
        "leadfield": mc.leadfield,
        "unipolar_ref_index": np.int16(mc.unipolar_ref_index),
        "surface": mc.surface.astype(np.float32) if mc.surface is not None
        else np.zeros(0, dtype=np.float32),
    }
    for split in ("train", "val", "test"):
        for k in KINDS:
            payload[f"ref_{split}_{k}"] = mc.refs[split][k]
    np.savez_compressed(path, **payload)


def _load_config_cache(
    path: Path, spec: dict[str, Any], ds: MultiReferenceDataset, budget: int
) -> MontageConfig:
    with np.load(path, allow_pickle=True) as d:
        refs = {
            split: {
                k: d[f"ref_{split}_{k}"] for k in KINDS
            }
            for split in ("train", "val", "test")
        }
        surface = d["surface"] if "surface" in d and d["surface"].size else None
        return MontageConfig(
            label=str(d["label"]),
            names=list(d["names"]),
            positions=d["positions"],
            n_channels=int(len(d["names"])),
            projection=d["projection"],
            out_map=d["out_map"],
            leadfield=d["leadfield"],
            unipolar_ref_index=int(d["unipolar_ref_index"]),
            refs=refs,
            surface=surface,
        )


def multiconfig_summary(data: MultiMontageData) -> str:
    lines = ["MultiMontageData:"]
    for label in data.order:
        mc = data.configs[label]
        lines.append(
            f"  {label}: {mc.n_channels} canales | refs "
            + ", ".join(f"{s}={mc.refs[s]['unipolar'].shape[0]}"
                        for s in ("train", "val", "test"))
            + f" | P={mc.projection.shape} Q={mc.out_map.shape}"
        )
    lines.append("  budget/split: " + ", ".join(
        f"{s}={data.n_budget(s)}" for s in ("train", "val", "test")))
    return "\n".join(lines)