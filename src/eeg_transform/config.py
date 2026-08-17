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

# Las cuatro referencias canónicas que el transformador debe unificar. Cada
# una es un operador lineal bien definido que elimina la componente constante
# instantánea, por lo que su valor NO depende de la referencia física en la
# que se adquirió la señal original (ver ``data.original_reference``).
REFERENCE_KINDS: tuple[str, ...] = ("unipolar", "bipolar", "car", "rest")

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


# Variantes de arquitectura del transformador (ver models.universal_transformer).
#   * ``free``     : autoencoder lineal libre (todas las matrices aprendidas).
#   * ``group``    : decodificador = pseudo-inversa del encoder del mismo
#                    montaje; impone consistencia de composición EXACTA
#                    (transitividad y auto-reconstrucción = proyector).
#   * ``projected``: todas las matrices aprendidas anulan por construcción el
#                    modo constante (físicamente válidas como referencias).
MODEL_VARIANTS: tuple[str, ...] = ("free", "group", "projected")


@dataclass
class ModelConfig:
    """Arquitectura del autoencoder lineal All-to-All."""

    latent_dim: int = 64
    use_bias: bool = False
    kernel_regularizer_l2: float = 1e-6
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    variant: str = "free"


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
        if self.model.variant == "group" and self.model.use_bias:
            raise ValueError("variant='group' requiere use_bias: false.")
        if self.model.variant == "group" and not (
            self.model.latent_dim in (0, -1)
        ):
            raise ValueError(
                "variant='group' requiere latent_dim 0/auto (debe igualar "
                "n_canales para pseudo-invertir)."
            )
        if len(self.leadfield.rel_radii) != len(self.leadfield.sigmas):
            raise ValueError(
                "rel_radii y sigmas deben tener la misma longitud (capas concéntricas)."
            )
        if self.leadfield.brain_radius >= self.leadfield.head_radius:
            raise ValueError("brain_radius debe ser menor que head_radius.")
        if self.leadfield.src_grid_mm <= 0:
            raise ValueError("src_grid_mm debe ser > 0.")


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