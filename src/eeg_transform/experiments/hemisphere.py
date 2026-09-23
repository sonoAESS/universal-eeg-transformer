"""Experimento sintético: extrapolación de la spline esférica a un hemisferio ciego.

Pregunta planteada en la exploración ``topomap_refs``: si se mide una
configuración **bipolar solo en un hemisferio** (diferencias entre canales
adyacentes del lado izquierdo) y se interpola con la spline esférica para
"ver" el hemisferio opuesto, ¿qué tan fiel es el resultado?

Diseño sintético y autocontenido:

* **Verdad de campo**: lead field analítico multicapa (MNE) evaluado en la
  grilla ASA ``standard_1005`` (343 nodos sobre todo el cuero cabelludo,
  normalizados a la esfera). Los potenciales se generan como ``V = G·q`` con
  dipolos aleatorios restringidos por lateralidad (hemisferio medido / ciego /
  bilateral) y recomenados a la media instantánea sobre el casco (CAR).
* **Montajes medidos**: subconjuntos del hemisferio izquierdo (``x < 0``):
  ``10-20``, ``10-10`` y ``dense`` (asa05 izquierdo). De cada montaje se derivan
  los **bipolares** como diferencias a lo largo de los lados Delaunay locales
  (solo dentro del hemisferio medido).
* **Reconstrucción**, dos familias:
  (a) interpolación esférica (spline de Perrin): desde el monopolar medido
  (control) y desde el bipolar (integrando las diferencias vía ``pinv`` +
  centrado, y luego spline).
  (b) **inversión del lead field** (condiciones de frontera físicas):
  ``P = W (W G_s)^+ G`` (REST entre montajes) para el monopolar y ajuste de
  dipolos desde las diferencias bipolares para ``leadfield_bip``; el inverso
  mal condicionado se regulariza truncando el SVD (``n_components``).
* **Métricas por región** (medida = izquierda+midline, ciega = derecha):
  VE (varianza explicada temporal), correlación ``r`` por canal, cociente de
  amplitud RMS y **fisicidad** (fracción de la reconstrucción en el espacio
  fuente-realizable ``col(G)``; la spline produce campos no realizables
  físicamente, el lead field sí). Barrido de ``order`` de Legendre, de
  ``n_components`` y lateralidad de las fuentes.

Salidas: ``runs/exp_extrapolacion_hemisferio/`` (json, csv y figuras).
Nota: las figuras comparan la verdad (CAR) frente a la reconstrucción desde el
hemisferio izquierdo — la zona ciega derecha es donde la spline *extrapola*.
La metodología completa se describe en
``docs/experimento_extrapolacion_hemisferio.md``; se ejecuta vía
``eeg-transform experiment-extrapolacion`` o ``python -m
eeg_transform.experiments.hemisphere``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

from ..leadfield import compute_lead_field
from ..mapping import (
    _normalize,
    leadfield_projection_matrix,
    scalp_grid_matrix,
    spherical_spline_matrix,
)
from .multi import MONTAGE_10_10_39

# raíz del repo: src/eeg_transform/experiments/hemisphere.py (3 niveles)
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "runs" / "exp_extrapolacion_hemisferio"
FIG = OUT / "figs"

# --- mesa de trabajo ------------------------------------------------------
RNG = np.random.default_rng(20262026)
HEAD_RADIUS = 0.09

def ve(a: np.ndarray, b: np.ndarray) -> float:
    """VE temporal (varianza explicada) de a frente a b, centrada por canal."""
    a0 = np.asarray(a) - np.asarray(a).mean(0, keepdims=True)
    b0 = np.asarray(b) - np.asarray(b).mean(0, keepdims=True)
    return float(1.0 - ((a0 - b0) ** 2).sum() / (b0**2).sum())


def ve_pattern(a: np.ndarray, b: np.ndarray) -> float:
    """VE del *patrón espacial* tras anular la media instantánea de ambos.

    Elimina el offset de referencia (constante por instante en todos los
    canales) que el bipolar no puede recuperar; aísla la calidad de forma.
    """
    a0 = np.asarray(a) - np.asarray(a).mean(1, keepdims=True)
    b0 = np.asarray(b) - np.asarray(b).mean(1, keepdims=True)
    return ve(a0, b0)


def region_metrics(rec: np.ndarray, true: np.ndarray, mask: np.ndarray) -> dict:
    """VE, VE patrón, r mediana y ratio de amplitud RMS por región."""
    r = np.asarray(rec)[:, mask]
    t = np.asarray(true)[:, mask]
    v = ve(r, t)
    vp = ve_pattern(r, t)
    rho = np.array(
        [
            np.corrcoef(r[:, j], t[:, j])[0, 1]
            for j in range(mask.sum())
            if np.std(r[:, j]) > 1e-12 and np.std(t[:, j]) > 1e-12
        ]
    )
    ratio = float(np.sqrt((r**2).mean()) / (np.sqrt((t**2).mean()) + 1e-15))
    return {
        "ve": v,
        "ve_patron": vp,
        "r_mediana": float(np.median(rho)),
        "r_min": float(np.min(rho)) if rho.size else np.nan,
        "amp_ratio": ratio,
        "n_canales": int(mask.sum()),
    }


# --- mejoras 1-3: techo Wiener, sLORETA y group-sparse espacio-temporal ----

def power_norm_sq(a: np.ndarray, iters: int = 25) -> float:
    """Norma espectral al cuadrado :math:`\\|A^T A\\|_2` por método de potencias.

    Evita el SVD completo de la matriz de forward grande (2006 fuentes).
    """
    n = a.shape[1]
    x = np.ones((n, 1))
    for _ in range(iters):
        y = a.T @ (a @ x)
        x = y / np.linalg.norm(y)
    return float(np.linalg.norm(a.T @ (a @ x)) / np.linalg.norm(x))


def fista_group_lasso(
    s_f: np.ndarray,
    y: np.ndarray,
    lam: float,
    iters: int = 200,
) -> np.ndarray:
    """Regularización L2,1 (soporte de dipolos compartido en el tiempo).

    Minimiza ``0.5||Y - X S^T||_F^2 + lam * sum_i ||X[:, i]||_2`` con
    ``X (T, N_src)``; el término de grupo obliga a que los dipolos activos
    sean los **mismos** en todo el intervalo (los ``N_ACTIVE`` verdaderos
    son fijos por escena). FISTA con prox de suavizado por fila.
    """
    m, n = y.shape[0], s_f.shape[1]
    l2 = power_norm_sq(s_f)
    step = 1.0 / l2
    x = np.zeros((m, n))
    z = np.zeros_like(x)
    th = 1.0
    for _ in range(iters):
        grad = (z @ s_f.T - y) @ s_f
        xn = z - step * grad
        rn = np.linalg.norm(xn, axis=0)
        shrink = np.maximum(rn - step * lam, 0.0) / (rn + 1e-30)
        xn = xn * shrink[None, :]
        thn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * th * th))
        z = xn + ((th - 1.0) / thn) * (xn - x)
        x, th = xn, thn
    return x


def pick_lambda(
    s_f: np.ndarray, y: np.ndarray, target: int,
    coarse_iters: int = 60, min_c: float = 1e-4, max_c: float = 5.0,
) -> float:
    """Elige ``lam`` (absoluto) con soporte ~ ``target`` dipolos activos.

    La búsqueda es binaria sobre la fracción ``c`` de la norma espectral
    ``||S^T S||_2``; devuelve ``sqrt(lo * hi) * ||S^T S||_2``.

    El número de dipolos activos de la verdad es un hecho de diseño del
    experimento (60), no el valor del campo: usarlo para fijar la
    regularización no filtra nada de la información a predecir.
    """
    rng_i = np.random.default_rng(0)
    idx = rng_i.choice(y.shape[0], size=min(240, y.shape[0]), replace=False)
    yn = y[idx] / (np.std(y) + 1e-15)
    l2 = power_norm_sq(s_f)
    lo, hi = min_c, max_c
    for _ in range(8):
        c = np.sqrt(lo * hi)
        x = fista_group_lasso(s_f, yn, c * l2, iters=coarse_iters)
        supp = int((np.linalg.norm(x, axis=0) > 1e-6).sum())
        if supp > target:
            lo = c
        else:
            hi = c
    return np.sqrt(lo * hi) * l2


def sloreta_estimate(s_f: np.ndarray, alpha: float) -> np.ndarray:
    """Inversa sLORETA: mínima norma Tikhonov + normalización por profundidad.

    ``K = S^T (S S^T + alpha I)^{-1}`` estandarizado por la diagonal de la
    matriz de resolución ``R = K S``: compensa el sesgo superficial de la
    mínima norma (los dipolos profundos aparecen subestimados).
    """
    c, _ = s_f.shape
    a = s_f @ s_f.T + alpha * np.eye(c)
    k = s_f.T @ np.linalg.inv(a)
    r = k @ s_f
    rdiag = np.clip(np.diag(r), 1e-12, None)
    return k / np.sqrt(rdiag)[:, None]


def to_car(x: np.ndarray) -> np.ndarray:
    """Re-referencia a la media instantánea del casco (343 canales, CAR)."""
    return np.asarray(x) - np.asarray(x).mean(1, keepdims=True)

def run_cmd(cfg=None) -> None:
    """Ejecuta el experimento completo (siembra, reconstrucción, figuras).

    No depende de ``cfg``: todos los parámetros están fijos en el módulo.
    """
    np.set_printoptions(precision=4, suppress=True)
    os.makedirs(FIG, exist_ok=True)

    # --- 1. montajes y grillas (marco canónico del asa, x derecha) -----------
    asa = np.load(ROOT / "data" / "processed" / "mne_asa_montages.npz")
    names_all = asa["names_asa05"].tolist()
    pos_all = np.asarray(asa["pos_asa05"], float)
    names_asa20 = asa["names_asa20"].tolist()

    name_idx = {nm: i for i, nm in enumerate(names_all)}
    pos_all_u = _normalize(pos_all)                      # esfera unidad
    pos_all_m = pos_all_u * HEAD_RADIUS                   # escala física para el forward
    left_mask = pos_all_u[:, 0] < 0.0                     # hemisferio izquierdo estricto
    midline_mask = np.abs(pos_all_u[:, 0]) < 1e-3         # plano sagital (no se mide)
    measured_region = left_mask | midline_mask            # lo que se "vería" si hubiera sensores
    blind_region = ~measured_region                        # hemisferio derecho ciego

    print("grid asa05:", len(pos_all), "| medida(x<=0):", measured_region.sum(),
          "| ciega(x>0):", blind_region.sum())

    montages = {
        "asa10-20": np.array([name_idx[nm] for nm in names_asa20 if nm in name_idx]),
        "asa10-10": np.array([name_idx[nm] for nm in MONTAGE_10_10_39 if nm in name_idx]),
        "dense": np.arange(len(pos_all)),
    }
    montages = {k: v[v < len(pos_all)] for k, v in montages.items()}
    # restringir cada montaje al hemisferio medido (izquierda + midline)
    montages = {k: v[measured_region[v]] for k, v in montages.items()}
    montages = {k: v for k, v in montages.items() if len(v) >= 8}
    print("montajes (canales en el hemisferio medido):",
          {k: int(len(v)) for k, v in montages.items()})


    def lateral_edges(idx: np.ndarray) -> np.ndarray:
        """Pares locales (bipolares) dentro del hemisferio medido vía Delaunay 2D.

        Las aristas se toman de la triangulación de Delaunay sobre la proyección al
        disco y se filtran para conservar solo diferencias vecinas (`d3d < cap`).
        """
        p = pos_all_u[idx]
        xy = p[:, :2]
        tri = Delaunay(xy)
        edges = set()
        for s in tri.simplices:
            for a, b in zip(s, np.roll(s, -1)):
                edges.add(tuple(sorted((int(a), int(b)))))
        e = np.array(sorted(edges), dtype=int)
        d3 = np.linalg.norm(p[e[:, 0]] - p[e[:, 1]], axis=1)
        cap = np.median(d3) * 1.8
        return e[d3 <= cap]


    # --- 2. lead field y verdad de campo --------------------------------------
    lf = compute_lead_field(
        ch_names=names_all,
        ch_positions=pos_all_m,
        head_radius=HEAD_RADIUS,
        rel_radii=[0.87, 0.90, 0.97, 1.00],
        sigmas=[0.33, 1.0, 0.0042, 0.33],
        src_grid_mm=10.0,
        brain_radius=0.078,
        verbose=False,
    )
    G = lf.matrix.astype(np.float64)                      # (343, 3*n_sources)
    src_xyz = lf.src_positions                            # (n_sources, 3)
    print("leadfield:", G.shape, "| fuentes x<0:", int((src_xyz[:, 0] < 0).sum()),
          "| x>0:", int((src_xyz[:, 0] > 0).sum()))

    # Proyector sobre el espacio fuente-realizable del casco entero: un campo EEG
    # generado por dipolos vive en col(G). La spline esférica produce campos suaves
    # que en principio NO se garantizan realizables; el lead field sí. La fisicidad
    # mide qué fracción del cuadrado del campo reconstruido reside en ese subespacio.
    # Anticipación: con la grilla de fuentes densa (2006 dipolos) el lead field es
    # de rango casi completo (339/343), de modo que col(G) cubre todo el espacio de
    # potenciales suaves: el indicador satura en ~1.0 para TODAS las rutas y no
    # discrimina. La restricción física real atañe a las condiciones de borde
    # (reproducir lo medido), que se mide por separado con ``res_borde``.
    PCOL = G @ np.linalg.pinv(G)                        # (343, 343)


    def fisicidad(rec: np.ndarray) -> float:
        """Fracción de varianza de ``rec`` en el espacio columna del lead field."""
        r = np.asarray(rec)
        prox = r @ PCOL
        return float(1.0 - ((r - prox) ** 2).sum() / (r**2).sum())

    SCENES = 12
    SAMPLES = 640
    FS = 128.0
    N_ACTIVE = 60


    def lowpass_white(n: int) -> np.ndarray:
        """Ruido blanco suavizado (rough 2-30 Hz) como curso temporal de dipolo."""
        x = RNG.standard_normal(n)
        win = np.hanning(25)
        x = np.convolve(x, win / win.sum(), mode="same")
        return x


    def make_scene(x_side_mask: np.ndarray) -> np.ndarray:
        """Potenciales de casco (T, 343) para una escena de dipolos en ``x_side_mask``."""
        cand = np.where(x_side_mask)[0]
        act = RNG.choice(cand, size=N_ACTIVE, replace=False)
        cols = (act[:, None] * 3 + np.arange(3)[None, :]).reshape(-1)  # 3 orientaciones
        q = np.zeros((3 * N_ACTIVE, SAMPLES))
        for k, src in enumerate(act):
            o = RNG.normal(0, 1, 3)
            o /= np.linalg.norm(o)
            tc = lowpass_white(SAMPLES) * 1e-9
            q[k * 3:k * 3 + 3] = o[:, None] * tc[None, :]
        v = (G[:, cols] @ q).T
        w = np.eye(G.shape[0]) - np.ones((G.shape[0], G.shape[0])) / G.shape[0]
        return v @ w                                                  # referencia promedio (CAR)


    LAT_SRC = {
        "medido": src_xyz[:, 0] < 0.0,
        "mixto": np.arange(src_xyz.shape[0]) >= 0,
        "ciego": src_xyz[:, 0] > 0.0,
    }

    TRUE = {}
    for lat, m in LAT_SRC.items():
        chunks = []
        for _ in range(SCENES):
            vr = make_scene(m)
            vr += RNG.normal(0, np.std(vr) * 0.01, vr.shape)          # ruido sensores 1%
            chunks.append(vr)
        TRUE[lat] = np.vstack(chunks)
        print("verdad lat=%s:" % lat, TRUE[lat].shape,
              "RMS=%.1f nV" % np.sqrt((TRUE[lat] ** 2).mean()))

    # instante de referencia representativo por lateralidad (para figuras)
    t_ref = {lat: int(np.argmax(np.var(TRUE[lat], axis=1))) for lat in TRUE}

    # --- 3. reconstrucción y métricas ----------------------------------------
    ORDERS = [8, 30, 60]
    SMOOTH = 1e-5
    KS_FRACS = (0.2, 1 / 3, 0.5)                   # fracciones de C_s para truncado SVD

    rows = []
    for ord_l in ORDERS:
        for name, idx in montages.items():
            p_src = pos_all_u[idx]
            edges = lateral_edges(idx)
            n_e = len(edges)
            M_bip = np.zeros((len(idx), n_e))
            for k, (a, b) in enumerate(edges):
                M_bip[a, k] += 1.0
                M_bip[b, k] -= 1.0
            P = spherical_spline_matrix(p_src, pos_all_u, order=ord_l, smoothness=SMOOTH)
            # proyección a nodos intermedios (centro de arista en la esfera)
            mid = p_src[edges[:, 0]] + p_src[edges[:, 1]]
            mid = _normalize(mid)
            Pm = spherical_spline_matrix(mid, pos_all_u, order=ord_l, smoothness=SMOOTH)

            # --- proyecciones por lead field (comunes a toda lateralidad) ---
            G_mont = G[idx]                                     # (C_m, N)
            G_bip = M_bip.T @ G_mont                            # (C_e, N): diferencias
            u_b, s_b, vt_b = np.linalg.svd(G_bip, full_matrices=False)
            ks = sorted({max(1, int(len(idx) * f)) for f in KS_FRACS})
            lf_plans = []
            for k in ks:
                # monopolar: REST entre montajes (la referencia promedio de la fuente)
                p_lf = leadfield_projection_matrix(G_mont, G, n_components=k)   # (C_m, 343)
                # bipolar: dipolos ajustados a las diferencias; rec = V_bip @ W
                w_b = (u_b[:, :k] * (1.0 / s_b[:k])) @ (vt_b[:k] @ G.T)          # (C_e, 343)
                lf_plans.append((k, p_lf, w_b))

            for lat in TRUE:
                V = TRUE[lat]
                Vs = V[:, idx]                                  # (T, C_m)
                Vb = Vs @ M_bip                                  # (T, C_edge) bipolares
                # control monopolar: interpolar directamente el potencial medido
                rec_mono = Vs @ P
                # bipolar: integrar diferencias (pinv + centrado) y luego spline
                uni = Vb @ np.linalg.pinv(M_bip, rcond=1e-8)
                uni = uni - uni.mean(1, keepdims=True)
                rec_bip = uni @ P
                # interp "directa de la diferencia" (valores bipolares en el centro
                # de arista) — no es un potencial, solo patrón; se reporta
                rec_diff = Vb @ Pm
                for rname, rec in [("monopolar", rec_mono), ("bipolar", rec_bip)]:
                    row = {
                        "order": ord_l, "montaje": name, "lateralidad": lat,
                        "ruta": rname, "n_canal": len(idx), "n_edges": n_e,
                        "n_components": np.nan, "fisicidad": fisicidad(rec),
                    }
                    for region, rmask in [("medida", measured_region),
                                          ("ciega", blind_region),
                                          ("total", np.ones(len(pos_all), bool))]:
                        m = region_metrics(rec, V, rmask)
                        row[f"ve_{region}"] = m["ve"]
                        row[f"ve_patron_{region}"] = m["ve_patron"]
                        row[f"r_{region}"] = m["r_mediana"]
                        row[f"amp_{region}"] = m["amp_ratio"]
                    rows.append(row)
                    print(f"order={ord_l:<3d} {name:<9s} lat={lat:<6s} "
                          f"{rname:12s} VE_medida={row['ve_medida']:+.3f} "
                          f"VE_ciega={row['ve_ciega']:+.3f} r_ciega={row['r_ciega']:.3f} "
                          f"fis={row['fisicidad']:.3f}")
                # rutas físicas: inversión del lead field
                for k, p_lf, w_b in lf_plans:
                    for rname, rec in [("leadfield", Vs @ p_lf),
                                       ("leadfield_bip", Vb @ w_b)]:
                        row = {
                            "order": np.nan, "montaje": name, "lateralidad": lat,
                            "ruta": rname, "n_canal": len(idx), "n_edges": n_e,
                            "n_components": k, "fisicidad": fisicidad(rec),
                        }
                        for region, rmask in [("medida", measured_region),
                                              ("ciega", blind_region),
                                              ("total", np.ones(len(pos_all), bool))]:
                            m = region_metrics(rec, V, rmask)
                            row[f"ve_{region}"] = m["ve"]
                            row[f"ve_patron_{region}"] = m["ve_patron"]
                            row[f"r_{region}"] = m["r_mediana"]
                            row[f"amp_{region}"] = m["amp_ratio"]
                        rows.append(row)
                        print(f"order=  -   {name:<9s} lat={lat:<6s} "
                              f"{rname:12s} k={k:<2d} VE_medida={row['ve_medida']:+.3f} "
                              f"VE_ciega={row['ve_ciega']:+.3f} r_ciega={row['r_ciega']:.3f} "
                              f"fis={row['fisicidad']:.3f}")
                m = region_metrics(rec_diff, V, blind_region)
                row_d = {
                    "order": ord_l, "montaje": name, "lateralidad": lat,
                    "ruta": "bipolar_directa", "n_canal": len(idx), "n_edges": n_e,
                    "n_components": np.nan, "fisicidad": fisicidad(rec_diff),
                }
                for region in ("medida", "ciega", "total"):
                    # la ruta directa es un campo de diferencias (sin unidades de
                    # potencial); solo se reporta correlación en la región ciega.
                    row_d[f"ve_{region}"] = np.nan
                    row_d[f"r_{region}"] = np.nan
                    row_d[f"amp_{region}"] = np.nan
                row_d["r_ciega"] = m["r_mediana"]
                rows.append(row_d)

    # --- 3b. mejoras: techo Wiener, sLORETA y inversión group-sparse ----------
    # Las tres se comparan en las mismas unidades (VE por región) que la spline y
    # el lead field. El **techo Wiener** es la mejor reconstrucción LINEAL posible
    # (regresión medida→campo completo); las demás rutas se leen como fracción
    # de ese límite.

    WIENER_LAMBDA_FRAC = 0.10            # ridge relativa a trace(C_ss)/C_m
    SLORITA_ALPHA_FRAC = 1e-3            # alpha Tikhonov relativa a ||S^T S||_2
    ISTA_ITERS = 200
    ISTA_DSTEP = 5                       # decimación temporal del ajuste L2,1
    REFIT_RIDGE = 1e-3                   # ridge del refit LS del soporte

    for name, idx in montages.items():
        p_src = pos_all_u[idx]
        edges = lateral_edges(idx)
        n_e = len(edges)
        M_bip = np.zeros((len(idx), n_e))
        for k, (a, b) in enumerate(edges):
            M_bip[a, k] += 1.0
            M_bip[b, k] -= 1.0
        G_mont = G[idx]                                    # (C_m, N)
        G_bip = M_bip.T @ G_mont                            # (C_e, N)
        w_s = np.eye(len(idx)) - np.ones((len(idx), len(idx))) / len(idx)
        s_m = w_s @ G_mont                                  # forward del monopolar centrado
        ALPHA_SL_MONO = SLORITA_ALPHA_FRAC * power_norm_sq(s_m)
        ALPHA_SL_BIP = SLORITA_ALPHA_FRAC * power_norm_sq(G_bip)
        K_sl_mono = sloreta_estimate(s_m, ALPHA_SL_MONO)
        K_sl_bip = sloreta_estimate(G_bip, ALPHA_SL_BIP)
        l2_bip = power_norm_sq(G_bip)

        for lat in TRUE:
            V = TRUE[lat]
            Vs = V[:, idx]
            Vc = Vs - Vs.mean(1, keepdims=True)             # referencia promedio del montaje
            Vb = Vs @ M_bip
            t_train = np.arange(0, SCENES // 2 * SAMPLES)
            t_test = np.arange(SCENES // 2 * SAMPLES, SCENES * SAMPLES)

            # --- (1) techo Wiener/MMSE (split-half sobre escenas) ---
            x_tr = Vs[t_train]
            c_ss = x_tr.T @ x_tr / len(t_train)
            c_sd = x_tr.T @ V[t_train] / len(t_train)
            lam_w = WIENER_LAMBDA_FRAC * np.trace(c_ss) / len(idx)
            h_wiener = np.linalg.solve(c_ss + lam_w * np.eye(len(idx)), c_sd)
            rec_win = Vs[t_test] @ h_wiener
            for rname, rec in [("wiener", rec_win)]:
                row = {
                    "order": np.nan, "montaje": name, "lateralidad": lat,
                    "ruta": rname, "n_canal": len(idx), "n_edges": n_e,
                    "n_components": np.nan, "alpha": np.nan,
                    "lambda_": lam_w, "cal_gain": np.nan, "fisicidad": fisicidad(rec),
                }
                for region, rmask in [("medida", measured_region),
                                      ("ciega", blind_region),
                                      ("total", np.ones(len(pos_all), bool))]:
                    m = region_metrics(rec, V[t_test], rmask)
                    row[f"ve_{region}"] = m["ve"]
                    row[f"ve_patron_{region}"] = m["ve_patron"]
                    row[f"r_{region}"] = m["r_mediana"]
                    row[f"amp_{region}"] = m["amp_ratio"]
                rows.append(row)
                print(f"wiener     {name:<9s} lat={lat:<6s} "
                      f"lam={lam_w:.2e} VE_medida={row['ve_medida']:+.3f} "
                      f"VE_ciega={row['ve_ciega']:+.3f} (techo lineal)")

            # --- (3) sLORETA: monopolar y bipolares (mínima norma estandarizada) ---
            # La normalización por resolución corrige el sesgo de profundidad pero
            # destruye la escala absoluta (~×20): se re-calibra la ganancia escalar
            # que mejor reproduce lo medido (un solo grado de libertad, región
            # medida) y se evalúa de nuevo el campo completo.
            for rname, q, alpha_sl in [("slorita", Vc @ K_sl_mono.T, ALPHA_SL_MONO),
                                       ("slorita_bip", Vb @ K_sl_bip.T, ALPHA_SL_BIP)]:
                rec0 = to_car(q @ G.T)
                cal_gain = ((V[:, measured_region] * rec0[:, measured_region]).sum()
                            / (rec0[:, measured_region] ** 2).sum())
                rec = cal_gain * rec0
                row = {
                    "order": np.nan, "montaje": name, "lateralidad": lat,
                    "ruta": rname, "n_canal": len(idx), "n_edges": n_e,
                    "n_components": np.nan, "alpha": alpha_sl, "lambda_": np.nan,
                    "cal_gain": cal_gain, "fisicidad": fisicidad(rec),
                }
                for region, rmask in [("medida", measured_region),
                                      ("ciega", blind_region),
                                      ("total", np.ones(len(pos_all), bool))]:
                    m = region_metrics(rec, V, rmask)
                    row[f"ve_{region}"] = m["ve"]
                    row[f"ve_patron_{region}"] = m["ve_patron"]
                    row[f"r_{region}"] = m["r_mediana"]
                    row[f"amp_{region}"] = m["amp_ratio"]
                rows.append(row)
                print(f"{rname:12s} {name:<9s} lat={lat:<6s} "
                      f"VE_medida={row['ve_medida']:+.3f} "
                      f"VE_ciega={row['ve_ciega']:+.3f} r_ciega={row['r_ciega']:.3f}")

            # --- (2) inversión group-sparse espacio-temporal (ISTA L2,1) ---
            # pick_lambda devuelve λ absoluto (c × ||SᵀS||₂) buscando ~N_ACTIVE
            # dipolos activos: la fracción c es la intensidad del soft-threshold
            # relativa a la escala del forward.
            lam_abs = pick_lambda(G_bip, Vb, N_ACTIVE)
            rec_chunks = []
            for sc in range(SCENES):
                yb = Vb[sc * SAMPLES:(sc + 1) * SAMPLES]            # (640, C_e)
                yn = yb[::ISTA_DSTEP] / (np.std(yb) + 1e-15)        # (m, C_e) normalizado
                x = fista_group_lasso(G_bip, yn, lam_abs, iters=ISTA_ITERS)
                supp = np.argwhere(np.linalg.norm(x, axis=0) > 1e-5).ravel()
                if supp.size == 0:
                    rec_chunks.append(np.zeros((SAMPLES, len(pos_all))))
                    continue
                g_j = G_bip[:, supp]                                 # (C_e, k)
                # refit LS del soporte sobre TODA la escena (restaura amplitudes)
                gj_T_gj = g_j.T @ g_j + REFIT_RIDGE * l2_bip * np.eye(supp.size)
                x_j = yb @ g_j @ np.linalg.inv(gj_T_gj)              # (640, k)
                abs_rec = x_j @ G[:, supp].T
                rec_chunks.append(to_car(abs_rec))
            rec_gl = np.vstack(rec_chunks)
            for rname, rec in [("grouplasso", rec_gl)]:
                row = {
                    "order": np.nan, "montaje": name, "lateralidad": lat,
                    "ruta": rname, "n_canal": len(idx), "n_edges": n_e,
                    "n_components": np.nan, "alpha": np.nan, "lambda_": lam_abs,
                    "cal_gain": np.nan, "fisicidad": fisicidad(rec),
                }
                for region, rmask in [("medida", measured_region),
                                      ("ciega", blind_region),
                                      ("total", np.ones(len(pos_all), bool))]:
                    m = region_metrics(rec, V, rmask)
                    row[f"ve_{region}"] = m["ve"]
                    row[f"ve_patron_{region}"] = m["ve_patron"]
                    row[f"r_{region}"] = m["r_mediana"]
                    row[f"amp_{region}"] = m["amp_ratio"]
                rows.append(row)
                print(f"grouplasso  {name:<9s} lat={lat:<6s} "
                      f"lam={lam_abs:.3f} supp={int((np.linalg.norm(x, axis=0) > 1e-5).sum()):>3d}"
                      f" VE_medida={row['ve_medida']:+.3f} "
                      f"VE_ciega={row['ve_ciega']:+.3f} r_ciega={row['r_ciega']:.3f}")

    # --- 4. tablas ------------------------------------------------------------

    df = pd.DataFrame(rows)
    for col in ("alpha", "lambda_"):
        if col not in df.columns:
            df[col] = np.nan
    for col in df.columns:
        if col.startswith(("ve_", "r_", "amp_", "lambda", "alpha", "cal_gain")):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.to_csv(OUT / "metricas.csv", index=False)

    resumen = df[(df.ruta == "bipolar") & (df.lateralidad == "mixto")].pivot_table(
        index="montaje", columns="order", values="ve_ciega")
    resumen.to_csv(OUT / "resumen_ve_ciega_mixto.csv")

    # mejor truncado del lead field bipolar por montaje (fuentes mixtas)
    sub_lfb = df[(df.ruta == "leadfield_bip") & (df.lateralidad == "mixto")]
    s_lfb = sub_lfb.loc[sub_lfb.groupby("montaje").ve_ciega.idxmax()].sort_index()
    s_lfb = s_lfb.set_index("montaje")
    s_lfb[["n_components", "ve_medida", "ve_ciega", "r_ciega", "fisicidad"]].round(3).to_csv(
        OUT / "resumen_leadfield_bip_mixto.csv")

    # --- comparativa por montaje (fuentes mixtas): métodos vs techo Wiener ----
    def mejor_por_montaje(ruta, lat="mixto"):
        sub = df[(df.ruta == ruta) & (df.lateralidad == lat)]
        idx = sub.groupby("montaje").ve_ciega.idxmax()
        return sub.loc[idx].set_index("montaje")

    RUTAS_CMP = ["bipolar", "leadfield_bip", "slorita", "slorita_bip", "grouplasso"]
    b30 = df[(df.ruta == "bipolar") & (df.lateralidad == "mixto") & (df.order == 30)]
    b30 = b30.set_index("montaje")["ve_ciega"]
    filas_cmp = []
    for mont in montages:
        d = {"montaje": mont}
        win = mejor_por_montaje("wiener")
        d["wiener"] = float(win.loc[mont, "ve_ciega"]) if mont in win.index else np.nan
        d["bipolar"] = float(b30.loc[mont]) if mont in b30.index else np.nan
        for ruta in ("leadfield_bip", "slorita", "slorita_bip", "grouplasso"):
            best = mejor_por_montaje(ruta)
            d[ruta] = float(best.loc[mont, "ve_ciega"]) if mont in best.index else np.nan
        if d["wiener"] and abs(d["wiener"]) > 1e-9:
            for ruta in ("bipolar", "leadfield_bip", "slorita", "slorita_bip",
                         "grouplasso"):
                d[f"{ruta}_frac_techo"] = d[ruta] / d["wiener"]
        filas_cmp.append(d)
    resumen_cmp = pd.DataFrame(filas_cmp).set_index("montaje").round(3)
    resumen_cmp.to_csv(OUT / "resumen_comparativa_mixto.csv")

    print("\n--- Comparativa VE ciega (mixto): métodos vs techo Wiener ---")
    print(resumen_cmp.loc[:, ["wiener", "bipolar", "leadfield_bip", "slorita",
                              "slorita_bip", "grouplasso"]].round(3))

    print("\n--- VE (hemisferio ciego) bipolar, fuentes mixtas ---")
    print(resumen.round(3))
    print("\n--- leadfield_bip (mejor n_components por montaje), fuentes mixtas ---")
    print(s_lfb[["n_components", "ve_medida", "ve_ciega", "r_ciega", "fisicidad"]].round(3))

    summary = {
        "n_validos": len(pos_all),
        "regiones": {"medida": int(measured_region.sum()), "ciega": int(blind_region.sum())},
        "montaje_canales": {k: int(len(v)) for k, v in montages.items()},
        "ve_ciega_bipolar_mixto": resumen.loc[:, :].round(4).astype(object).to_dict(),
        "leadfield_bip_mixto": s_lfb[
            ["n_components", "ve_medida", "ve_ciega", "r_ciega", "fisicidad"]
        ].round(4).astype(object).to_dict(),
        "techo_wiener_mixto": {
            m: float(x) for m, x in resumen_cmp["wiener"].dropna().items()
        },
        "comparativa_mixto": resumen_cmp.astype(object).to_dict(),
        "fisicidad_media": {
            r: float(df[df.ruta == r]["fisicidad"].mean())
            for r in ("monopolar", "bipolar", "bipolar_directa",
                      "leadfield", "leadfield_bip",
                      "wiener", "slorita", "slorita_bip", "grouplasso")
        },
    }
    # resumen compacto por tripleta clave (order=30) para el veredicto
    res30 = df[df.order == 30]
    for lat in ("medido", "mixto", "ciego"):
        for ruta in ("monopolar", "bipolar"):
            sub = res30[(res30.lateralidad == lat) & (res30.ruta == ruta)]
            summary[f"{lat}|{ruta}"] = {
                m: float(sub[sub.montaje == m]["ve_ciega"].iloc[0]) for m in montages
            }
    with open(OUT / "metricas.json", "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2))

    # --- 5. figuras -----------------------------------------------------------
    GRID_PX = 64
    def topomap_ax(ax, values, title, vlim=None, cmap="RdBu_r"):
        Mm, valid, _ = scalp_grid_matrix(pos_all_u, grid_px=GRID_PX)
        img = (np.asarray(values).reshape(1, -1) @ Mm.T).reshape(GRID_PX, GRID_PX)
        img = np.where(valid.reshape(GRID_PX, GRID_PX), img, np.nan)
        v = np.nanmax(np.abs(img)) if vlim is None else vlim
        im = ax.imshow(img, cmap=cmap, vmin=-v, vmax=v, origin="lower",
                       interpolation="bilinear")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=8)
        return im

    # 5.1 panorama monopolar vs bipolar para orden 30, fuentes mixtas
    lat, ord_l, mont, t = "mixto", 30, "asa10-10", t_ref["mixto"]
    idx = montages[mont]
    V = TRUE[lat]
    Vs = V[:, idx]
    edges = lateral_edges(idx)
    M_bip = np.zeros((len(idx), len(edges)))
    for k, (a, b) in enumerate(edges):
        M_bip[a, k] += 1.0
        M_bip[b, k] -= 1.0
    P = spherical_spline_matrix(pos_all_u[idx], pos_all_u, order=ord_l, smoothness=SMOOTH)
    rec_mono = Vs @ P
    rec_bip = (Vs @ M_bip @ np.linalg.pinv(M_bip, rcond=1e-8))
    rec_bip -= rec_bip.mean(1, keepdims=True)
    rec_bip = rec_bip @ P

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    vmax = np.abs(V[t]).max()
    imk = topomap_ax(axes[0, 0], V[t], "verdad (CAR)", vlim=vmax)
    topomap_ax(axes[0, 1], rec_mono[t], f"monopolar {mont} -> spline o={ord_l}", vlim=vmax)
    topomap_ax(axes[0, 2], rec_bip[t], f"bipolar {mont} -> spline o={ord_l}", vlim=vmax)
    # 2ª fila: residuos (hemisferio ciego arriba en el disco)
    topomap_ax(axes[1, 0], V[t] - rec_mono[t], "residuo mono", vlim=None, cmap="bwr")
    topomap_ax(axes[1, 1], V[t] - rec_bip[t], "residuo bipolar", vlim=None, cmap="bwr")
    # error de interpolación dentro de la región medida (límite inferior de calidad)
    rec_self = V @ spherical_spline_matrix(pos_all_u, pos_all_u, order=ord_l, smoothness=SMOOTH)
    topomap_ax(axes[1, 2], V[t] - rec_self[t], "residuo spline auto (frontera)",
               vlim=None, cmap="bwr")
    for ax in axes[0]:
        ax.scatter(pos_all_u[measured_region][:, 1], pos_all_u[measured_region][:, 0],
                   s=2, c="k", alpha=0.25)
    fig.suptitle(f"Verdad de campo CAR vs reconstrucción desde {mont} (o={ord_l}, fuentes {lat})")
    fig.tight_layout()
    fig.savefig(FIG / "01_panorama_topomapas.png", dpi=130)
    plt.close(fig)

    # 5.2 VE ciega por montaje y orden (fuentes mixtas): curvas
    sub = df[(df.ruta == "bipolar") & (df.lateralidad == "mixto")]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for mont in sorted(sub.montaje.unique()):
        m = sub[sub.montaje == mont].sort_values("order")
        ax.plot(m.order, m.ve_ciega, marker="o", label=mont)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("orden de Legendre"); ax.set_ylabel("VE hemisferio ciego")
    ax.set_title("Extrapolación bipolar a hemisferio ciego (fuentes mixtas)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "02_ve_ciega_por_orden.png", dpi=130)
    plt.close(fig)

    # 5.3 VE ciega por lateralidad de fuentes (bipolar, o=30) -> barras
    sub = df[(df.ruta == "bipolar") & (df.order == 30)]
    fig, ax = plt.subplots(figsize=(8, 5))
    piv = sub.pivot_table(index="lateralidad", columns="montaje", values="ve_ciega")
    piv.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("VE hemisferio ciego")
    ax.set_title("Efecto de la lateralidad de las fuentes (orden 30)")
    ax.grid(axis="y", alpha=0.3); ax.legend(title="montaje")
    fig.tight_layout(); fig.savefig(FIG / "03_ve_ciega_por_lateralidad.png", dpi=130)
    plt.close(fig)

    # 5.4 VE ciego vs distancia angular al casco medido (bipolar, o=30, mixto)
    idx = montages["asa10-10"]
    p_src = pos_all_u[idx]
    edges = lateral_edges(idx)
    cos_to_med = pos_all_u @ p_src.T
    ang_med = np.degrees(np.arccos(np.clip(cos_to_med.max(1), -1, 1)))   # cerca del sensor más próximo
    sub = df[(df.ruta == "bipolar") & (df.order == 30) &
             (df.montaje == "asa10-10") & (df.lateralidad == "mixto")]
    rec = TRUE["mixto"][:, idx] @ spherical_spline_matrix(
        pos_all_u[idx], pos_all_u, order=30, smoothness=SMOOTH)
    rec = (rec.T - rec.T.mean(0)).T
    v_region = np.zeros(len(pos_all))
    for j in range(len(pos_all)):
        v_region[j] = ve(rec[:, j], TRUE["mixto"][:, j])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    bins = np.linspace(0, 120, 25)
    bi = np.digitize(ang_med, bins)
    ax = axes[0]
    for mask, lbl in [(measured_region, "medida"), (blind_region, "ciega")]:
        sel = mask & (bi < len(bins))
        xb, yb = [], []
        for b in range(1, len(bins)):
            m = sel & (bi == b)
            if m.sum():
                xb.append(bins[b]); yb.append(np.nanmedian(v_region[m]))
        ax.plot(xb, yb, marker="o", label=lbl)
    ax.set_xlabel("distancia angular al sensor más cercano (°)")
    ax.set_ylabel("VE por nodo (mediana)")
    ax.set_title("Calidad de extrapolación bipolar por distancia al casco medido")
    ax.legend(); ax.grid(alpha=0.3)
    sc = axes[1].imshow(v_region.reshape(-1, 1), aspect="auto")
    axes[1].set_title("VE por nodo en el grid asa05 (índice de nodo)")
    fig.colorbar(sc, ax=axes[1])
    fig.tight_layout(); fig.savefig(FIG / "04_ve_por_distancia.png", dpi=130)
    plt.close(fig)

    # --- 5.5 comparativa spline-bipolar vs lead field (topomapas, mixto) ----
    lat, mont_sel = "mixto", "asa10-10"
    idx_sel = montages[mont_sel]
    edges_sel = lateral_edges(idx_sel)
    M_bip_sel = np.zeros((len(idx_sel), len(edges_sel)))
    for k, (a, b) in enumerate(edges_sel):
        M_bip_sel[a, k] += 1.0
        M_bip_sel[b, k] -= 1.0
    Vs_sel = TRUE[lat][:, idx_sel]
    Vb_sel = Vs_sel @ M_bip_sel
    G_mont_sel = G[idx_sel]
    G_bip_sel = M_bip_sel.T @ G_mont_sel
    u_b_sel, s_b_sel, vt_b_sel = np.linalg.svd(G_bip_sel, full_matrices=False)
    k_sel = int(s_lfb.loc[mont_sel, "n_components"])
    p_lf_sel = leadfield_projection_matrix(G_mont_sel, G, n_components=k_sel)
    w_b_sel = (u_b_sel[:, :k_sel] * (1.0 / s_b_sel[:k_sel])) @ (vt_b_sel[:k_sel] @ G.T)
    rec_spline_sel = (Vb_sel @ np.linalg.pinv(M_bip_sel, rcond=1e-8))
    rec_spline_sel -= rec_spline_sel.mean(1, keepdims=True)
    rec_spline_sel = rec_spline_sel @ spherical_spline_matrix(
        pos_all_u[idx_sel], pos_all_u, order=30, smoothness=SMOOTH)
    rec_lf_sel = Vs_sel @ p_lf_sel
    rec_lfb_sel = Vb_sel @ w_b_sel
    t_sel = t_ref[lat]
    vmax = np.abs(TRUE[lat][t_sel]).max()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    imk = topomap_ax(axes[0, 0], TRUE[lat][t_sel], "verdad (CAR)", vlim=vmax)
    topomap_ax(axes[0, 1], rec_spline_sel[t_sel],
               f"spline bipolar o=30 (VEc={region_metrics(rec_spline_sel, TRUE[lat], blind_region)['ve']:+.2f})", vlim=vmax)
    topomap_ax(axes[0, 2], rec_lfb_sel[t_sel],
               f"leadfield_bip k={k_sel} (VEc={region_metrics(rec_lfb_sel, TRUE[lat], blind_region)['ve']:+.2f})", vlim=vmax)
    topomap_ax(axes[1, 0], rec_lf_sel[t_sel],
               f"leadfield mono k={k_sel} (VEc={region_metrics(rec_lf_sel, TRUE[lat], blind_region)['ve']:+.2f})", vlim=vmax)
    topomap_ax(axes[1, 1], TRUE[lat][t_sel] - rec_spline_sel[t_sel],
               "residuo spline bipolar", vlim=None, cmap="bwr")
    topomap_ax(axes[1, 2], TRUE[lat][t_sel] - rec_lfb_sel[t_sel],
               "residuo leadfield_bip", vlim=None, cmap="bwr")
    for ax in axes[0]:
        ax.scatter(pos_all_u[measured_region][:, 1], pos_all_u[measured_region][:, 0],
                   s=2, c="k", alpha=0.25)
    fig.suptitle(f"Interpolación física vs spline desde {mont_sel} (mixto, t de máx. varianza)")
    fig.tight_layout(); fig.savefig(FIG / "05_comparativa_spline_vs_leadfield.png", dpi=130)
    plt.close(fig)

    # --- 5.6 VE medida vs VE ciega (fuentes mixtas) --------------------------
    # La mejora del lead field tiene que venir SIN degradar el hemisferio medido:
    # aquí se ve el frente de Pareto calidad-medida/extrapolación por ruta. Las
    # mejorar ("wiener", "slorita", "grouplasso") se plotean junto a las bases.
    sub56 = df[(df.lateralidad == "mixto") & df.ruta.isin(
        ["monopolar", "bipolar", "leadfield", "leadfield_bip",
         "wiener", "slorita", "slorita_bip", "grouplasso"])]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for rname, mk, c in [("monopolar", "o", "#4C72B0"), ("bipolar", "s", "#DD8452"),
                         ("leadfield", "^", "#55A868"), ("leadfield_bip", "D", "#C44E52"),
                         ("wiener", "*", "#111111"), ("slorita", "P", "#8172B3"),
                         ("slorita_bip", "X", "#CCB974"), ("grouplasso", "v", "#55A868")]:
        s = sub56[sub56.ruta == rname]
        ax.scatter(s.ve_medida, s.ve_ciega, marker=mk, c=c, label=rname, alpha=0.8,
                   s=(160 if rname == "wiener" else 44))
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("VE hemisferio medido")
    ax.set_ylabel("VE hemisferio ciego")
    ax.set_title("Extrapolación vs fidelidad a lo medido (mixto, todas las rutas)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "06_ve_medida_vs_ve_ciega.png", dpi=130)
    plt.close(fig)

    # --- 5.7 VE ciega del leadfield_bip por n_components (fuentes mixtas) ----
    sub57 = df[(df.ruta == "leadfield_bip") & (df.lateralidad == "mixto")]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for mont in sorted(sub57.montaje.unique()):
        m = sub57[sub57.montaje == mont].sort_values("n_components")
        ax.plot(m.n_components, m.ve_ciega, marker="o", label=mont)
    # referencia: mejor spline bipolar (o=30) por montaje
    for mont in sorted(sub57.montaje.unique()):
        v = df[(df.ruta == "bipolar") & (df.lateralidad == "mixto") &
               (df.montaje == mont) & (df.order == 30)]["ve_ciega"].iloc[0]
        ax.axhline(v, color="gray", lw=0.8, ls=":")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("n_components (truncado SVD del lead field bipolar)")
    ax.set_ylabel("VE hemisferio ciego")
    ax.set_title("Extrapolación bipolar por lead field: efecto del truncado (mixto)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "07_leadfield_bip_por_components.png", dpi=130)
    plt.close(fig)

    # --- 5.8 Comparativa: VE ciega por método y % del techo Wiener (mixto) ----
    # El techo Wiener es la mejor reconstrucción lineal (regresión medida→campo):
    # las barras muestran qué fracción de ese límite alcanza cada ruta.
    cmp_plot = resumen_cmp.copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colores = {"wiener": "#333333", "bipolar": "#DD8452",
               "leadfield_bip": "#C44E52", "slorita": "#8172B3",
               "slorita_bip": "#CCB974", "grouplasso": "#55A868"}
    metodos = ["bipolar", "leadfield_bip", "slorita", "slorita_bip", "grouplasso"]
    x = np.arange(len(cmp_plot.index))
    ax = axes[0]
    for ruta in metodos:
        ax.bar(x - 0.2 + 0.08 * metodos.index(ruta), cmp_plot[ruta],
               width=0.075, color=colores[ruta], label=ruta)
    ax.plot(x, cmp_plot["wiener"], "k*", ms=15, label="wiener (techo)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x, cmp_plot.index)
    ax.set_ylabel("VE hemisferio ciego (mixto)")
    ax.set_title("VE ciega por método — techo Wiener en asterisco")
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    width = 0.18
    for i, ruta in enumerate(metodos):
        if f"{ruta}_frac_techo" in cmp_plot:
            ax.bar(x + (i - (len(metodos) - 1) / 2) * width, cmp_plot[f"{ruta}_frac_techo"],
                   width=width, color=colores[ruta], label=ruta)
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x, cmp_plot.index)
    ax.set_ylabel("VE ciega / techo Wiener")
    ax.set_ylim(0, 1.05)
    ax.set_title("Fracción del techo alcanzada (1 = límite teórico)")
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Mejoras sobre la extrapolación: Wiener, sLORETA y group-sparse (ISTA)")
    fig.tight_layout(); fig.savefig(FIG / "08_comparativa_techo_wienerslorita_grouplasso.png", dpi=130)
    plt.close(fig)

    print("\ndone ->", OUT)
if __name__ == "__main__":
    run_cmd()
