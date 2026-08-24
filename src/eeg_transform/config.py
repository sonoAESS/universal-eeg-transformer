"""Configuración tipada del pipeline cargada desde YAML.

Los dataclasses definen todos los hiperparámetros físicos, de datos y de
entrenamiento. La clase raíz :class:`EEGTransformConfig` se carga desde un
archivo YAML con :func:`load_config` y valida los invariantes básicos.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from .logging_conf import get_logger

log = get_logger(__name__)

# Las referencias que el transformador debe unificar (rama universal_refs).
# Cada una es un operador lineal bien definido; todas anulan la componente
# constante instantánea, por lo que su valor NO depende de la referencia
# física en la que se adquirió la señal original (ver
# ``data.original_reference``). ``unipolar`` es la referencia al canal
# vértice (``data.unipolar_ref_ch``, por defecto Cz); ``linked_mastoids`` y
# ``linked_ears`` restan el promedio del potencial interpolado en M1/M2 y
# A1/A2; ``laplacian`` es el Laplaciano de superficie esférico (Perrin,
# 1989), invariante a cualquier re-referenciación.
REFERENCE_KINDS: tuple[str, ...] = (
    "unipolar",
    "linked_mastoids",
    "linked_ears",
    "bipolar",
    "car",
    "rest",
    "laplacian",
)

# Referencias físicas admisibles para el dato original (adquisición).
ORIGINAL_REFERENCES: tuple[str, ...] = ("left_mastoid", "cz", "average")


@dataclass
class DataConfig:
    """Configuración de adquisición y preprocesamiento de EEG real."""

    subjects: list[int] = field(default_factory=lambda: [1, 2])
    runs: list[int] = field(default_factory=lambda: [1, 2])
    bandpass: list[float] = field(default_factory=lambda: [1.0, 45.0])
    notch: list[float] = field(default_factory=lambda: [])
    unipolar_ref_ch: str = "Cz"
    # Referencia física con la que fue adquirido el dato original. ``eegbci``
    # (PhysioNet) graba contra la mastoides izquierda; se usa para auditar la
    # transformación, pero los cuatro montajes (uni/bip/CAR/REST) son
    # invariantes a esta elección porque anulan cualquier offset constante.
    original_reference: str = "left_mastoid"
    artifact_z_thresh: float = 6.0
    artifact_min_max_volts: float = 500e-6
    bad_channel_z_thresh: float = 4.0
    n_jobs: int = 1


@dataclass
class LeadFieldConfig:
    """Modelo de conducción de volumen: esfera concéntrica multicapa."""

    head_radius: float = 0.09
    rel_radii: list[float] = field(default_factory=lambda: [0.87, 0.90, 0.97, 1.00])
    sigmas: list[float] = field(
        default_factory=lambda: [0.33, 1.0, 0.0042, 0.33]
    )
    brain_radius: float = 0.078
    src_grid_mm: float = 10.0
    drop_bad_sources: bool = True
    rest_rcond: float | None = None
    # C6: elegir rest_rcond por configuración vía validación cruzada de la VE
    # analítica (unipolar reconstruida desde rest). Si es True, se ignora
    # rest_rcond y se barre rest_rcond_candidates por configuración.
    rest_rcond_cv: bool = False
    rest_rcond_candidates: list[float] = field(
        default_factory=lambda: [1e-2, 1e-3, 1e-4, 1e-5]
    )


@dataclass
class DatasetConfig:
    """Configuración de construcción del dataset de entrenamiento."""

    cache_dir: str = "data/processed"
    split_mode: str = "block"
    test_frac: float = 0.2
    val_frac: float = 0.2
    seed: int = 42
    max_samples_per_subject: int = 60_000
    dtype: str = "float32"


@dataclass
class MappingConfig:
    """Unificación de montajes: proyección de cualquier distribución de
    electrodos al espacio canónico (usada por las variantes ``montage_*`` y
    ``multi_montage``).

    * ``method``       : ``leadfield`` (solución inversa con el lead field
      analítico), ``spline`` (interpolación esférica, estilo *topomapa* /
      *heatmap* donde la actividad se ve como manchas) o ``nearest``.
    * ``montage``      : configuración de electrodos de origen, p. ej.
      ``10-20`` (19 canales) o ``10-10`` (39). Se estima desde **cualquier**
      distribución de electrodos: también acepta una lista de canales.
    * ``n_components`` : truncado SVD de la solución inversa (''leadfield'').
      ``None`` = automático (``C_s // 3``, ver mapping).
    * ``smoothness``   : regularización ridge base del spline.
    * ``adaptive_smoothness``: si es verdadero, el suavizado crece cuanto
      menos denso es el montaje de origen (más regularización a sparso).
    * ``grid_px``      : resolución (px) de la malla del cuero cabelludo para
      representar la actividad como un *heatmap* de manchas.
    * ``configs``      : configuraciones de la variante ``multi_montage``
      (entrenamiento conjunto y balanceado sobre varias distribuciones de
      electrodos). Cada etiqueta resuelve a un generador: ``10-20`` (19 ch
      reales, subconjunto del canónico), ``canonical`` (los electrodos
      nativos), ``dense-N`` (N posiciones simuladas cuasi-uniformes sobre el
      casquete del cuero cabelludo, p. ej. ``dense-128``/``dense-256``),
      ``10-10`` (39 ch reales). Se admiten todos los montajes que comparta el
      dataset canónico.
    * ``multi_max_samples_per_split``: presupuesto de muestras por split y por
      configuración (se submuestran con equiespaciado determinista). Igual para
      todas las configuraciones → proporciones balanceadas por construcción.
    """

    method: str = "leadfield"
    montage: str = "10-20"
    n_components: int | None = None
    smoothness: float = 1e-5
    adaptive_smoothness: bool = True
    grid_px: int = 48
    configs: list[str] = field(
        default_factory=lambda: ["10-20", "canonical", "dense-128", "dense-256"]
    )
    multi_max_samples_per_split: int = 40_000


def __post_init__(self):
        # PyYAML puede dejar notaciones como '1e-5' como cadena; se coerciona.
        try:
            self.smoothness = float(self.smoothness)
        except (TypeError, ValueError):
            pass
        if self.n_components is not None:
            try:
                self.n_components = int(self.n_components)
            except (TypeError, ValueError):
                raise ValueError("mapping.n_components debe ser un entero o null.")
        try:
            self.multi_max_samples_per_split = int(self.multi_max_samples_per_split)
        except (TypeError, ValueError):
            raise ValueError("mapping.multi_max_samples_per_split debe ser un entero.")


# Variantes de arquitectura del transformador (ver models.universal_transformer).
#   * ``free``     : autoencoder lineal libre (todas las matrices aprendidas).
#   * ``group``    : decodificador = pseudo-inversa del encoder del mismo
#                    montaje; impone consistencia de composición EXACTA
#                    (transitividad y auto-reconstrucción = proyector).
#   * ``projected``: todas las matrices aprendidas anulan por construcción el
#                    modo constante (físicamente válidas como referencias).
#   * ``montage_leadfield``: unificación de montajes con **solución inversa**:
#                    las observaciones de un montaje de ``C_s`` electrodos
#                    entran por un proyector fijo construido con el lead field
#                    analítico (regularizado por SVD) y el autoencoder aprende
#                    el refinamiento y la conversión a las 4 referencias
#                    canónicas (ver mapping.leadfield_projection_matrix).
#   * ``montage_heatmap`` : idem con interpolación esférica (spline/Perrin)
#                    al espacio canónico, suavizada según la densidad del
#                    montaje; visualmente equivale a un *topomapa* donde la
#                    actividad aparece como manchas difusas.
#   * ``multi_montage``  : entrenamiento conjunto y balanceado sobre VARIAS
#                    configuraciones de electrodos (p. ej. 19 canales 10-20,
#                    el montaje canónico de 64, y montajes densos simulados de
#                    128 y 256 electrodos). Cada configuración se embebe en el
#                    espacio canónico con una matriz fija ``P_s`` y se lee con
#                    un mapa fijo ``Q_s``; el autoencoder comparte las 8
#                    matrices (W_enc/W_dec) entre todas las configuraciones.
#                    El modelo acepta cualquier configuración y predice, EN ESA
#                    configuración, las medidas con otra referencia.
MODEL_VARIANTS: tuple[str, ...] = (
    "free", "group", "projected", "soft_group",
    "montage_leadfield", "montage_heatmap",
    "multi_montage", "multi_heatmap", "multi_heatmap_v2",
    # universal_refs: 7 referencias + cabeza temporal de residuo
    "universal_refs",
)

# Métodos de proyección entre montajes (ver mapping.build_projection).
MAPPING_METHODS: tuple[str, ...] = ("spline", "leadfield", "nearest")

# Montajes estándar conocidos (nombres usados por MappingConfig.montage).
MAPPING_MONTAGES: tuple[str, ...] = ("10-20", "10-10")


@dataclass
class ModelConfig:
    """Arquitectura del autoencoder lineal All-to-All."""

    latent_dim: int = 64
    use_bias: bool = False
    kernel_regularizer_l2: float = 1e-6
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    variant: str = "free"
    # Peso de la penalización de consistencia de composición (soft_group):
    # termina la física de grupo como pérdida suave en lugar de estructura.
    comp_penalty_weight: float = 0.0
    # Peso del término de pérdida en el CAMPO DE SUPERFICIE (latente del
    # "heatmap") para la variante ``multi_heatmap``: además del MSE por
    # electrodo, se ajusta la actividad interpolada sobre la malla del cuero
    # cabelludo (patrón espacial suave, las "manchas" del topomapa).
    surface_loss_weight: float = 0.1
    # --- Mejoras de ``multi_heatmap_v2`` ---
    # Interpolación de campo aprendible (R_s entrenable, init = spline fija S_s).
    learnable_interp: bool = False
    # Peso de la consistencia electrodo<->campo: x̂ ≈ R_sᵀ·F̂ (A1).
    field_consistency_weight: float = 0.0
    # Peso de la consistencia entre campos de distintas configuraciones (B4).
    xconfig_consistency_weight: float = 0.0
    # Rango del adaptador low-rank por configuración (B5); 0 = desactivado.
    adapter_rank: int = 0
    # Peso de la regularización de variación total temporal (C7); 0 = off.
    temporal_smoothness_weight: float = 0.0
    # Ponderación por incertidumbre multi-task (A3): los pesos se aprenden.
    learn_uncertainty: bool = False
    # --- Cabeza temporal de residuo (universal_refs) ---
    # Ventana centrada offline (muestras); 0 = modelo puramente instantáneo.
    # La cabeza consume la señal canónica de la ventana y predice el residuo
    # respecto al mapa lineal; con capa final a cero el arranque es idéntico
    # al modelo instantáneo (ablation trivial).
    temporal_window: int = 0
    temporal_stride: int = 1
    # Dimensión oculta y nº de bloques convolucionales depthwise+pointwise.
    temporal_channels: int = 64
    temporal_layers: int = 2
    temporal_kernel: int = 7
    # Peso del residuo temporal sobre la salida lineal (regulariza cuánto
    # puede desviarse la corrección dinámica del mapa físico).
    temporal_residual_weight: float = 1.0
    # Penalización de modos espaciales en salidas diferenciales: media nula
    # en bipolar/linked/laplacian y proyección nula sobre l<=1 en laplacian.
    mode_penalty_weight: float = 0.0

    def __post_init__(self):
        # PyYAML puede dejar '1e-3' como cadena; se coerciona a numérico.
        for name in ("latent_dim", "adapter_rank", "temporal_window",
                     "temporal_stride", "temporal_channels",
                     "temporal_layers", "temporal_kernel"):
            try:
                setattr(self, name, int(getattr(self, name)))
            except (TypeError, ValueError):
                pass
        for name in ("use_bias", "learnable_interp", "learn_uncertainty"):
            v = getattr(self, name)
            if isinstance(v, str):
                setattr(self, name, v.strip().lower() in ("true", "1", "yes"))
        for name in ("kernel_regularizer_l2", "learning_rate", "comp_penalty_weight",
                     "surface_loss_weight", "field_consistency_weight",
                     "xconfig_consistency_weight", "temporal_smoothness_weight",
                     "temporal_residual_weight", "mode_penalty_weight"):
            try:
                setattr(self, name, float(getattr(self, name)))
            except (TypeError, ValueError):
                pass


@dataclass
class TrainingConfig:
    """Configuración del bucle de entrenamiento."""

    epochs: int = 60
    batch_size: int = 1024
    early_stop_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-5
    shuffle_buffer: int = 20_000
    prefetch: int = 4
    run_dir: str = "runs/default"
    log_file: str = "runs/default/train.log"
    save_best_only: bool = True
    seed: int = 42


@dataclass
class EvalConfig:
    """Configuración de evaluación."""

    n_plot_samples: int = 400
    plot_channel: str = "Cz"


@dataclass
class EEGTransformConfig:
    """Configuración raíz del pipeline completo."""

    data: DataConfig = field(default_factory=DataConfig)
    leadfield: LeadFieldConfig = field(default_factory=LeadFieldConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    out_dir: str = "runs/default"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EEGTransformConfig":

        def _build(dc_cls: type[Any], entries: dict[str, Any]) -> Any:
            fmap = {f.name: f for f in fields(dc_cls)}
            kwargs = {}
            for name, sub in entries.items():
                if name not in fmap:
                    continue
                f = fmap[name]
                nested_default = f.default if not isinstance(
                    f.default, dataclasses._MISSING_TYPE
                ) else f.default_factory if not isinstance(
                    f.default_factory, dataclasses._MISSING_TYPE
                ) else None
                if isinstance(sub, dict) and is_dataclass(nested_default):
                    kwargs[name] = _build(type(nested_default), sub)
                else:
                    kwargs[name] = sub
            return dc_cls(**kwargs)

        return cls(
            data=_build(DataConfig, raw.get("data", {})),
            leadfield=_build(LeadFieldConfig, raw.get("leadfield", {})),
            dataset=_build(DatasetConfig, raw.get("dataset", {})),
            mapping=_build(MappingConfig, raw.get("mapping", {})),
            model=_build(ModelConfig, raw.get("model", {})),
            training=_build(TrainingConfig, raw.get("training", {})),
            evaluation=_build(EvalConfig, raw.get("evaluation", {})),
            out_dir=str(raw.get("out_dir", "runs/default")),
        )

    def validate(self) -> None:
        """Valida invariantes físicos y numéricos de la configuración."""
        lo, hi = self.data.bandpass[0], self.data.bandpass[1]
        if lo <= 0:
            raise ValueError("bandpass[0] debe ser > 0 Hz.")
        if hi is not None and hi <= lo:
            raise ValueError("bandpass[1] debe ser mayor que bandpass[0].")
        if self.dataset.split_mode not in ("block", "subject"):
            raise ValueError("split_mode debe ser 'block' o 'subject'.")
        if self.data.original_reference not in ORIGINAL_REFERENCES:
            raise ValueError(
                "original_reference debe ser una de "
                f"{ORIGINAL_REFERENCES}, no '{self.data.original_reference}'."
            )
        if not (0 < self.dataset.test_frac < 1 and 0 < self.dataset.val_frac < 1):
            raise ValueError("Las fracciones de split deben estar en (0, 1).")
        if self.model.latent_dim < 0:
            raise ValueError("latent_dim debe ser >= 1 (o 0 para usar n_canales).")
        if self.model.variant not in MODEL_VARIANTS:
            raise ValueError(
                f"model.variant debe ser una de {MODEL_VARIANTS}, "
                f"no '{self.model.variant}'."
            )
        if self.model.variant in ("montage_leadfield", "montage_heatmap"):
            expected = (
                "leadfield" if "leadfield" in self.model.variant else "spline"
            )
            if self.mapping.method != expected:
                raise ValueError(
                    f"La variante '{self.model.variant}' requiere "
                    f"mapping.method == '{expected}', no "
                    f"'{self.mapping.method}'."
                )
        if self.model.variant in ("multi_montage", "multi_heatmap"):
            if "canonical" not in set(self.mapping.configs):
                raise ValueError(
                    f"{self.model.variant} requiere la configuración 'canonical' "
                    "en mapping.configs (es el espacio del autoencoder)."
                )
            labels = set(self.mapping.configs)
            for lab in sorted(labels):
                if lab == "canonical":
                    continue
                if lab == "10-20" or lab == "10-10" or lab.startswith("dense-"):
                    continue
                raise ValueError(
                    f"Configuración '{lab}' no soportada en {self.model.variant} "
                    "(válidas: '10-20', '10-10', 'canonical' o 'dense-<N>')."
                )
            if self.mapping.multi_max_samples_per_split < 1_000:
                raise ValueError(
                    "multi_max_samples_per_split debe ser >= 1000 muestras."
                )
        if self.model.variant in ("multi_heatmap", "multi_heatmap_v2"):
            if self.mapping.method != "spline":
                raise ValueError(
                    "variant='%s' requiere mapping.method == 'spline' "
                    "(el campo de superficie es la interpolación sobre la "
                    "malla del cuero cabelludo)." % self.model.variant
                )
            if not (0.0 <= self.model.surface_loss_weight <= 1.0):
                raise ValueError(
                    "surface_loss_weight debe estar en (0, 1] para "
                    f"{self.model.variant}."
                )
            if self.model.adapter_rank < 0:
                raise ValueError("adapter_rank debe ser >= 0.")
        if self.model.variant in ("multi_montage", "multi_heatmap") and self.model.latent_dim not in (0, -1):
            raise ValueError(f"variant='{self.model.variant}' requiere latent_dim 0/auto.")
        if self.mapping.method not in MAPPING_METHODS:
            raise ValueError(
                f"mapping.method debe ser una de {MAPPING_METHODS}, "
                f"no '{self.mapping.method}'."
            )
        if self.mapping.grid_px <= 0:
            raise ValueError("mapping.grid_px debe ser > 0.")
        if self.model.variant == "group" and self.model.use_bias:
            raise ValueError("variant='group' requiere use_bias: false.")
        if self.model.variant == "group" and not (
            self.model.latent_dim in (0, -1)
        ):
            raise ValueError(
                "variant='group' requiere latent_dim 0/auto (debe igualar "
                "n_canales para pseudo-invertir)."
            )
        if self.model.variant == "soft_group" and not (0.0 <= self.model.comp_penalty_weight <= 10.0):
            raise ValueError("soft_group requiere comp_penalty_weight en (0, 10].")
        if len(self.leadfield.rel_radii) != len(self.leadfield.sigmas):
            raise ValueError(
                "rel_radii y sigmas deben tener la misma longitud (capas concéntricas)."
            )
        if self.leadfield.brain_radius >= self.leadfield.head_radius:
            raise ValueError("brain_radius debe ser menor que head_radius.")
        if self.leadfield.src_grid_mm <= 0:
            raise ValueError("src_grid_mm debe ser > 0.")
        if self.leadfield.rest_rcond is not None and not (0.0 < self.leadfield.rest_rcond < 1.0):
            raise ValueError("leadfield.rest_rcond debe estar en (0, 1) o ser None.")


def load_config(path: str | Path) -> EEGTransformConfig:
    """Carga la configuración desde un archivo YAML."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = EEGTransformConfig.from_dict(raw)
    cfg.validate()
    log.info("Configuración cargada desde %s", path)
    return cfg


def save_config(cfg: EEGTransformConfig, path: str | Path) -> None:
    """Serializa la configuración a YAML (útil para reproducibilidad)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=False)
    log.info("Configuración guardada en %s", path)