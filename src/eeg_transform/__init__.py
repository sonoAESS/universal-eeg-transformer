"""Universal EEG Transformer.

Paquete para la transformación universal de referencias de EEG:

* Referencias clásicas (unipolar, bipolar, CAR) mediante matrices analíticas.
* Referencia al infinito (REST, Yao 2001) con lead field analítico de
  esfera multicapa (cerebro/CSF/cráneo/piel) resuelto con MNE.
* Estimación de mapas entre referencias con un autoencoder lineal
  multientrada/multisalida (Keras/TensorFlow).
"""

from . import config, data, evaluation, models, references, training
from .config import EEGTransformConfig, load_config

__all__ = [
    "config",
    "data",
    "models",
    "references",
    "evaluation",
    "training",
    "EEGTransformConfig",
    "load_config",
]

__version__ = "2.0.0"