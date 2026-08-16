r"""Lead field analítico de esfera concéntrica multicapa y referencia REST.

Se resuelve la ecuación de Laplace/Poisson para un modelo de conducción de
volumen con capas esféricas concéntricas (cerebro/CSF/cráneo/piel) usando la
implementación multi-esfera de MNE (equivalente al modelo analítico publicado
en la literatura de REST/eLORETA y usado por el REST toolbox de EEGLAB).

Con el lead field :math:`G_L` se construye la matriz de la referencia al
infinito según Yao (2001):

.. math::

    V_{\\text{REST}} = G_L\\,(W_{\\text{avg}}G_L)^{+}\\,W_{\\text{avg}}\\,
    V,\qquad W_{\\text{avg}} = I - \\frac{1}{C}\\mathbf{1}\\mathbf{1}^{T}.

Referencias:
* Yao, D. (2001). "A method to standardize a reference of scalp EEG
  recordings to a point at infinity." *Physiol. Meas.* 22, 693–711.
* Dong, L. et al. (2017). "REST: a software toolkit..." *Front. Neurosci.*
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .logging_conf import get_logger

log = get_logger(__name__)

try:
    import mne
    from mne.bem import make_sphere_model
    from mne.forward import make_forward_solution
    from mne.transforms import Transform
    _HAS_MNE = True
except Exception:  # pragma: no cover
    _HAS_MNE = False

MIN_SOURCE_RADIUS_M = 1e-3  # descarta dipolos sobre el origen (singularidad)


@dataclass
class LeadField:
    """Lead field calculado con el modelo multi-esfera.

    Attributes
    ----------
    matrix:
        Matriz de ganancia ``(n_channels, 3 * n_sources)``: cada fuente tiene
        las tres orientaciones ortogonales. Unidades: V/(A·m).
    ch_names:
        Nombres de los electrodos.
    src_positions:
        Posiciones (m) de las fuentes ``(n_sources, 3)``.
    head_radius:
        Radio exterior de la cabeza (m).
    rel_radii:
        Radios relativos de cada interfaz respecto de ``head_radius``.
    sigmas:
        Conductividades (S/m) de cada capa.
    src_grid_mm:
        Espaciado (mm) de la rejilla volumétrica de fuentes.
    """

    matrix: np.ndarray
    ch_names: list[str]
    src_positions: np.ndarray
    head_radius: float = 0.09
    rel_radii: list[float] = field(default_factory=lambda: [0.87, 0.90, 0.97, 1.00])
    sigmas: list[float] = field(default_factory=lambda: [0.33, 1.0, 0.0042, 0.33])
    src_grid_mm: float = 10.0

    @property
    def n_channels(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_sources(self) -> int:
        return self.src_positions.shape[0]

    def rest_matrix(self) -> np.ndarray:
        """Matriz de la referencia al infinito (REST)."""
        from .references import rest_matrix

        return rest_matrix(self.matrix, self.n_channels)

    def param_hash(self) -> str:
        """Hash de los parámetros físicos (clave de caché)."""
        payload = json.dumps(
            {
                "head_radius": self.head_radius,
                "rel_radii": list(self.rel_radii),
                "sigmas": list(self.sigmas),
                "grid_mm": self.src_grid_mm,
                "ch_names": self.ch_names,
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            matrix=self.matrix,
            src_positions=self.src_positions,
            ch_names=np.array(self.ch_names, dtype=object),
            head_radius=self.head_radius,
            rel_radii=np.array(self.rel_radii),
            sigmas=np.array(self.sigmas),
            src_grid_mm=self.src_grid_mm,
            rest_matrix=self.rest_matrix(),
        )
        log.info("Lead field guardado en %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "LeadField":
        with np.load(path, allow_pickle=True) as d:
            return cls(
                matrix=d["matrix"].astype(np.float64),
                ch_names=list(d["ch_names"]),
                src_positions=d["src_positions"],
                head_radius=float(d["head_radius"]),
                rel_radii=list(d["rel_radii"]),
                sigmas=list(d["sigmas"]),
                src_grid_mm=float(d["src_grid_mm"]),
            )


def compute_lead_field(
    ch_names: list[str],
    ch_positions: np.ndarray,
    head_radius: float = 0.09,
    rel_radii: Optional[list[float]] = None,
    sigmas: Optional[list[float]] = None,
    src_grid_mm: float = 10.0,
    brain_radius: float = 0.078,
    verbose: bool = False,
) -> LeadField:
    """Calcula el lead field analítico con el modelo multi-esfera de MNE.

    Parameters
    ----------
    ch_names:
        Nombres de los canales EEG.
    ch_positions:
        Posiciones (m) de los electrodos ``(n_channels, 3)``.
    head_radius:
        Radio externo de la piel.
    rel_radii:
        Radios normalizados de cada interfaz (orden creciente, terminando en 1).
    sigmas:
        Conductividad (S/m) de cada capa.
    src_grid_mm:
        Espaciado de la rejilla volumétrica de fuentes en el cerebro.
    brain_radius:
        Radio (m) máximo de las fuentes (debe ser < ``head_radius``).
    verbose:
        Si es ``True`` muestra trazas de MNE.
    """
    if not _HAS_MNE:
        raise ImportError("mne es necesario para calcular el lead field analítico.")

    rel_radii = rel_radii or [0.87, 0.90, 0.97, 1.00]
    sigmas = sigmas or [0.33, 1.0, 0.0042, 0.33]
    if len(rel_radii) != len(sigmas):
        raise ValueError("rel_radii y sigmas deben tener la misma longitud.")
    if brain_radius >= head_radius:
        raise ValueError("brain_radius debe ser menor que head_radius.")

    n_channels = len(ch_names)
    info = mne.create_info(ch_names=list(ch_names), sfreq=256.0, ch_types="eeg")
    # Asignar las posiciones reales de los electrodos
    from mne.channels import make_dig_montage

    dig_pos = {ch: pos for ch, pos in zip(ch_names, ch_positions)}
    dig = make_dig_montage(ch_pos=dig_pos, coord_frame="head")
    info.set_montage(dig)

    bem = make_sphere_model(
        r0=(0.0, 0.0, 0.0),
        head_radius=head_radius,
        info=info,
        relative_radii=tuple(rel_radii),
        sigmas=tuple(sigmas),
        verbose="WARNING" if not verbose else None,
    )

    src = mne.setup_volume_source_space(
        pos=src_grid_mm,
        sphere=bem,
        mindist=0.0,
        verbose="ERROR" if not verbose else None,
    )

    trans = Transform("head", "mri", np.eye(4))
    with np.errstate(divide="ignore", invalid="ignore"):
        fwd = make_forward_solution(
            info,
            trans,
            src,
            bem,
            meg=False,
            eeg=True,
            mindist=0.0,
            verbose="ERROR" if not verbose else None,
        )

    # Elimina fuentes degeneradas (origen, singularidad) o con valores no
    # finitos. La máscara se calcula por fuente (3 columnas por fuente).
    raw_matrix = fwd["sol"]["data"].astype(np.float64)
    src_pos = fwd["src"][0]["rr"][fwd["src"][0]["vertno"]]
    col_good = np.isfinite(raw_matrix).all(axis=0)
    per_src_good = col_good.reshape(-1, 3).all(axis=1)

    centers = np.linalg.norm(src_pos, axis=1) < MIN_SOURCE_RADIUS_M
    bad_src = centers | ~per_src_good
    keep_src = np.where(~bad_src)[0]

    cols = (keep_src[:, None] * 3 + np.arange(3)[None, :]).reshape(-1)
    matrix = raw_matrix[:, cols]
    src_pos = src_pos[keep_src]

    log.info(
        "Lead field: %d electrodos x %d fuentes (3 orientaciones) | %.1f mm grid",
        n_channels,
        src_pos.shape[0],
        src_grid_mm,
    )
    return LeadField(
        matrix=matrix,
        ch_names=ch_names,
        src_positions=src_pos,
        head_radius=head_radius,
        rel_radii=rel_radii,
        sigmas=sigmas,
        src_grid_mm=src_grid_mm,
    )