"""Censo de bases EEG reales en OpenNeuro para la variante universal_refs.

Objetivo: encontrar bases públicas BIDS-EEG con cascos nativos REALES que
completen las distribuciones objetivo (19, 32, 64, 128/129, 256/257
electrodos), sin posiciones simuladas.

Estrategia (2 pasos):
1. GraphQL público de OpenNeuro: lista snapshots de datasets con modalidad
   EEG, con su resumen (nº sujetos, tareas, licencia).
2. Para cada candidato descarga ``dataset_description.json`` + ``README``
   (archivos pequeños) y busca menciones de casco/densidad:
   ``EGI``/``HydroCel``/``GSN``, ``BioSemi``, ``ANT``/``BrainVision``,
   ``10-20``/``10-10``, y conteos ``\\b(19|32|64|128|129|256|257)\\b`` ch.

Salida: tabla CSV ordenada por utilidad (densidad objetivo, nº sujetos,
coordenadas anunciadas) en ``--output``.

NOTA DE RED: este script está pensado para ejecutarse donde la API de
OpenNeuro sea accesible (local del usuario o Colab); algunos entornos de
desarrollo la tienen bloqueada. Requiere solo ``requests``.

Uso:
    python tools/openneuro_census.py --max-datasets 400 --output censo.csv
    python tools/openneuro_census.py --probe          # solo prueba conectividad
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time

GRAPHQL_URL = "https://openneuro.org/crn/graphql"
FILE_URL = "https://openneuro.org/crn/snapshots/{snapshot}/files/{path}"

# Densidades objetivo (cascos comerciales reales) y patrones de fabricantes.
TARGET_DENSITY = {19, 32, 63, 64, 127, 128, 129, 256, 257}
CAP_PATTERNS = {
    "EGI/HydroCel": re.compile(r"hydrocel|egi|geodesic", re.I),
    "GSN": re.compile(r"\bgsn\b", re.I),
    "BioSemi": re.compile(r"biosemi", re.I),
    "ANT/BrainVision": re.compile(r"ant\b|brainvision|brainamp", re.I),
    "10-20/10-10": re.compile(r"10[-\s]?20|10[-\s]?10|international system", re.I),
}
DENSITY_RE = re.compile(
    r"(?:\b(\d{1,3})\s*(?:ch|chan|channels?|electrodos?|electrode))|"
    r"(?:\b(19|32|64|128|129|256|257)\s*-\s*ch)",
    re.I,
)

LIST_QUERY = """
query($cursor: Cursor) {
  datasets(first: 50, after: $cursor, filterBy: {modalities: ["EEG"]}) {
    edges { cursor node {
      id name
      latestSnapshot {
        id tag
        license
        summary { subjectCount tasks modalities }
      }
    } }
    pageInfo { hasNextPage endCursor }
  }
}
"""


def _gql(session, query: str, variables: dict | None = None) -> dict:
    resp = session.post(GRAPHQL_URL, json={"query": query, "variables": variables or {}},
                        timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("errors"):
        raise RuntimeError(f"GraphQL errors: {payload['errors'][:2]}")
    return payload["data"]


def probe(session) -> bool:
    """Prueba de conectividad mínima contra la API."""
    try:
        _gql(session, "{ viewer { id } }")
        print("API de OpenNeuro accesible.")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"API NO accesible desde este entorno: {exc}")
        return False


def list_eeg_datasets(session, max_datasets: int) -> list[dict]:
    out: list[dict] = []
    cursor = None
    while len(out) < max_datasets:
        data = _gql(session, LIST_QUERY, {"cursor": cursor})
        conn = data["datasets"]
        for edge in conn["edges"]:
            node = edge["node"]
            snap = node.get("latestSnapshot") or {}
            summary = snap.get("summary") or {}
            out.append({
                "id": node["id"],
                "name": node.get("name") or "",
                "snapshot": snap.get("id"),
                "license": snap.get("license") or "",
                "subjects": int(summary.get("subjectCount") or 0),
                "tasks": ", ".join(summary.get("tasks") or []),
            })
        page = conn["pageInfo"]
        if not page.get("hasNextPage"):
            break
        cursor = page.get("endCursor")
        time.sleep(0.4)
    return out[:max_datasets]


def fetch_small_file(session, snapshot: str, path: str) -> str:
    url = FILE_URL.format(snapshot=snapshot, path=path)
    resp = session.get(url, timeout=60)
    if resp.status_code != 200:
        return ""
    return resp.text[:200_000]


def classify(text: str) -> tuple[int | None, list[str]]:
    """Extrae densidad declarada y fabricantes/cascos mencionados."""
    caps: list[str] = [name for name, rx in CAP_PATTERNS.items() if rx.search(text)]
    density: int | None = None
    for match in DENSITY_RE.finditer(text):
        value = int(match.group(1) or match.group(2))
        if value in TARGET_DENSITY:
            density = value
            break
    return density, caps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-datasets", type=int, default=400)
    parser.add_argument("--output", default="censo_openneuro.csv")
    parser.add_argument("--min-subjects", type=int, default=6)
    parser.add_argument("--probe", action="store_true",
                        help="solo prueba de conectividad")
    args = parser.parse_args(argv)

    import requests

    with requests.Session() as session:
        session.headers["User-Agent"] = "universal-eeg-census/0.1"
        accesible = probe(session)
        if args.probe:
            return 0 if accesible else 1
        if not accesible:
            print("Sin acceso a la API: ejecutar desde Colab u otra red.")
            return 1

        datasets = list_eeg_datasets(session, args.max_datasets)
        print(f"{len(datasets)} datasets EEG encontrados; analizando descripciones…")

        rows: list[dict] = []
        for i, d in enumerate(datasets):
            snap = d["snapshot"]
            if not snap:
                continue
            text = "\n".join([
                d["name"], d["tasks"],
                fetch_small_file(session, snap, "README"),
                fetch_small_file(session, snap, "dataset_description.json"),
            ])
            density, caps = classify(text)
            has_coords = bool(re.search(r"electrodes\.tsv|coordsystem", text, re.I))
            rows.append({**d, "density_ch": density or "", "caps": ";".join(caps),
                         "coords_hint": has_coords,
                         "target": bool(density in TARGET_DENSITY)})
            if (i + 1) % 25 == 0:
                print(f"  …{i + 1}/{len(datasets)}")
            time.sleep(0.15)

    # Utilidad: densidad objetivo primero, luego nº de sujetos.
    rows.sort(key=lambda r: (
        not r["target"],
        -int(r["subjects"] or 0) if r["subjects"] else 9_999,
    ))
    filtered = [r for r in rows if r["subjects"] >= args.min_subjects]
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(filtered[0].keys()))
        writer.writeheader()
        writer.writerows(filtered)
    print(f"Censo guardado en {args.output} ({len(filtered)} filas con "
          f">= {args.min_subjects} sujetos).")

    hits = [r for r in filtered if r["target"]]
    print(f"\n{len(hits)} candidatos con densidad objetivo "
          f"(19/32/64/128/129/256/257 ch):")
    for r in hits[:15]:
        print(f"  {r['id']:>12s}  {r['density_ch']:>3} ch  "
              f"{r['subjects']:>3} sujetos  [{r['caps']}]  {r['name'][:60]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
