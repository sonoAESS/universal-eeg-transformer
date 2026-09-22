"""Interfaz de línea de comandos del Universal EEG Transformer.

Uso:
    eeg-transform build   [--config config/default.yaml]
    eeg-transform train   [--config config/default.yaml] [--force]
    eeg-transform eval    [--config config/default.yaml]
    eeg-transform pipeline[--config config/default.yaml] [--force]
    eeg-transform montage [--config config/default.yaml] [--montages 10-20,10-10]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .config import load_config, save_config
from .logging_conf import get_logger, setup_logging

log = get_logger(__name__)


def _load_cfg(path: str) -> "EEGTransformConfig":
    from .config import EEGTransformConfig

    cfg = load_config(path)
    return cfg


def cmd_build(cfg) -> None:
    from .data.dataset import build_dataset

    ds = build_dataset(cfg)
    print(ds.summary())


def cmd_train(cfg, force: bool) -> None:
    from .data.dataset import build_dataset
    from .training.trainer import train

    ds = build_dataset(cfg)
    save_config(cfg, Path(cfg.training.run_dir) / "config.yaml")
    model, history = train(ds, cfg, force=force)
    print("\nEntrenamiento finalizado.")
    final = history.history["val_loss_estandarizada"][-1]
    print(f"val_loss_estandarizada final: {final:.6f}")


def cmd_eval(cfg) -> None:
    import pandas as pd

    from .data.dataset import MultiReferenceDataset, build_dataset
    from .evaluation import metrics, plots
    from .training.trainer import (
        build_model,
        build_montage_inputs,
        is_montage_variant,
        is_multiconfig_variant,
        load_multiconfig_data,
        build_multiconfig_model,
    )

    run_dir = Path(cfg.training.run_dir)
    ds = build_dataset(cfg)

    if is_multiconfig_variant(cfg):
        data = load_multiconfig_data(cfg, ds)
        model = build_multiconfig_model(cfg, data)
        model.core.ensure_built()
        model.load_weights(str(run_dir / "best.weights.h5"))
        model.compile(optimizer="adam")

        metrics_df = metrics.evaluate_multiconfig_routes(model, data, split="test")
        summ = metrics.summarize_multiconfig(metrics_df)
        print("\n============== MULTI-CONFIGURACION (TEST) ===============")
        print("Configuraciones balanceadas: "
              + ", ".join(f"{l} ({data.configs[l].n_channels} ch)"
                          for l in data.order))
        print(summ.to_string(formatters={
            "rmse_diag_uV": "{:.3f}".format, "r_diag": "{:.4f}".format,
            "rmse_cross_uV": "{:.3f}".format, "r_cross": "{:.4f}".format,
            "ve_cross": "{:.3f}".format,
            "rmse_ana_cross_uV": "{:.3f}".format,
            "r_ana_cross": "{:.4f}".format, "ve_ana_cross": "{:.3f}".format}))
        metrics_df.to_csv(run_dir / "metrics_test.csv", index=False)

        history_csv = run_dir / "history.csv"
        if history_csv.exists():
            plots.plot_learning_curves(history_csv, run_dir)
        plots.plot_multiconfig_heatmap(metrics_df, run_dir)
        plots.plot_multiconfig_bars(metrics_df, run_dir)
        plots.plot_multiconfig_scalps(model, data, run_dir)

        if cfg.model.variant == "universal_refs" and cfg.model.temporal_window > 0:
            w = cfg.model.temporal_window
            print(f"\n======== MULTI-CONFIG WINDOWED (TEST): ventana {w} ========")
            print("Predicción causal por ventana deslizante (stride 1, salida "
                  "del último paso); el calentamiento inicial se descarta.")
            wind = metrics.evaluate_multiconfig_windowed(
                model, data, split="test", window=w, stride=1)
            wsumm = metrics.summarize_multiconfig(wind)
            print(wsumm.to_string(formatters={
                "rmse_diag_uV": "{:.3f}".format, "r_diag": "{:.4f}".format,
                "rmse_cross_uV": "{:.3f}".format, "r_cross": "{:.4f}".format,
                "ve_cross": "{:.3f}".format,
                "rmse_ana_cross_uV": "{:.3f}".format,
                "r_ana_cross": "{:.4f}".format, "ve_ana_cross": "{:.3f}".format}))
            wind.to_csv(run_dir / "metrics_windowed_test.csv", index=False)

        if cfg.model.variant in ("multi_heatmap", "multi_heatmap_v2"):
            surface_df = metrics.evaluate_multiconfig_surface_routes(
                model, data, split="test")
            surf_summ = metrics.summarize_multiconfig_surface(surface_df)
            print("\n======== CAMPO DE SUPERFICIE (TEST) ========")
            print(surf_summ.to_string(formatters={
                "rmse_field_diag_uV": "{:.3f}".format,
                "r_field_diag": "{:.4f}".format,
                "ve_field_diag": "{:.3f}".format,
                "rmse_field_cross_uV": "{:.3f}".format,
                "r_field_cross": "{:.4f}".format,
                "ve_field_cross": "{:.3f}".format}))
            surface_df.to_csv(run_dir / "metrics_surface_test.csv", index=False)
            plots.plot_multiconfig_surface(surface_df, run_dir)

        if cfg.model.variant == "multi_heatmap_v2":
            field_agree_df = metrics.evaluate_multiconfig_field_agreement(
                model, data, split="test")
            if not field_agree_df.empty:
                fa_rmse = field_agree_df["rmse_field"].mean() * 1e6
                fa_ve = field_agree_df["ve_field"].mean()
                print("\n======== ACUERDO DE CAMPO ENTRE CONFIGS (TEST) ========")
                print(f"rmse_field pareado medio: {fa_rmse:.3f} uV  |  "
                      f"ve_field medio: {fa_ve:.3f}")
            field_agree_df.to_csv(run_dir / "metrics_field_agreement_test.csv",
                                  index=False)
        return

    montage = build_montage_inputs(cfg, ds) if is_montage_variant(cfg) else None
    model = build_model(cfg, ds.n_channels,
                        projection=montage.projection if montage else None)
    model.ensure_built()
    model.load_weights(str(run_dir / "best.weights.h5"))
    model.compile(optimizer="adam")

    uni_idx = ds.ch_names.index(cfg.data.unipolar_ref_ch)

    if montage is not None:
        metrics_df = metrics.evaluate_montage_routes(model, ds, montage,
                                                     split="test")
        summary = metrics.summarize_montage(metrics_df)
        print("\n================ MONTAJE (TEST) ================")
        print(f"Origen: {montage.montage} ({len(montage.src_names)} canales) "
              f"via '{montage.method}' → canónico ({ds.n_channels})")
        print(summary.to_string(formatters={
            "rmse_uV_model": "{:.3f}".format,
            "rmse_uV_proy": "{:.3f}".format,
            "r_model": "{:.4f}".format,
            "ve_model": "{:.3f}".format,
            "r_proy": "{:.4f}".format,
            "ve_proy": "{:.3f}".format}))
        print("\nDetalle por ruta (modelo vs proyección analítica):")
        print(metrics_df.round(9).to_string(index=False))
        metrics_df.to_csv(run_dir / "metrics_test.csv", index=False)

        history_csv = run_dir / "history.csv"
        if history_csv.exists():
            plots.plot_learning_curves(history_csv, run_dir)
        plots.plot_heatmap(metrics_df, "rmse", "RMSE real por ruta (V)",
                           run_dir, "heatmap_rmse.png")
        plots.plot_scalp_heatmap(model, ds, montage, cfg, run_dir)
        return

    metrics_df = metrics.evaluate_routes(model, ds, split="test")
    err_matrix = metrics.transfer_error_matrix(model, ds, uni_idx, cfg)
    cons = metrics.composition_error_table(model, ds)

    print("\n================ METRICAS (TEST) ================")
    print(metrics.summarize(metrics_df, "test").to_string(formatters={
        "mse": "{:.3e}".format, "rmse": "{:.3e}".format,
        "mae": "{:.3e}".format, "r": "{:.4f}".format}))
    print("\nDetalle por ruta:")
    print(metrics_df.round(9).to_string(index=False))

    print("\n================ ERROR MATRIZ vs ANALITICA ================")
    print(err_matrix.round(6).to_string(index=False))

    print("\n================ CONSISTENCIA DE COMPOSICION ================")
    print(cons.round(6).to_string(index=False))

    metrics_df.to_csv(run_dir / "metrics_test.csv", index=False)
    err_matrix.to_csv(run_dir / "error_fro_rel.csv", index=False)
    cons.to_csv(run_dir / "consistency.csv", index=False)

    history_csv = run_dir / "history.csv"
    if history_csv.exists():
        plots.plot_learning_curves(history_csv, run_dir, run_label=run_dir.name)
    plots.plot_heatmap(metrics_df, "rmse", "RMSE real por ruta (V)",
                       run_dir, "heatmap_rmse.png")
    plots.plot_heatmap(err_matrix, "error_fro_rel",
                       "Error rel. Frobenius vs matrices analíticas",
                       run_dir, "heatmap_fro.png")
    plots.plot_traces(
        model, ds, split="test",
        channel=cfg.evaluation.plot_channel,
        n_samples=cfg.evaluation.n_plot_samples,
        run_dir=run_dir,
    )


def cmd_compare(runs: list[str]) -> None:
    """Compara resúmenes de ejecuciones de entrenamiento (solo métricas)."""
    import pandas as pd

    rows = []
    for run in runs:
        run = Path(run)
        name = run.name
        metrics_csv = run / "metrics_test.csv"
        history_csv = run / "history.csv"
        if not metrics_csv.exists():
            print(f"[skip] {name}: sin metrics_test.csv")
            continue
        m = pd.read_csv(metrics_csv)
        diag = m[m["origen"] == m["destino"]]
        off = m[m["origen"] != m["destino"]]
        conf = run / "config.yaml"
        variant = "?"
        latent = None
        try:
            c = load_config(conf)
            variant = c.model.variant
            latent = c.model.latent_dim
        except Exception:
            pass
        best_val = None
        n_epochs = None
        if history_csv.exists():
            h = pd.read_csv(history_csv)
            col = [c for c in h.columns if c.startswith("val_loss_")]
            if col:
                best_val = float(h[col[0]].min())
                n_epochs = int(len(h))
        comp_mean = None
        comp_max = None
        cons_csv = run / "consistency.csv"
        if cons_csv.exists():
            cons = pd.read_csv(cons_csv)
            comp_mean = float(cons["comp_medio"].mean())
            comp_max = float(cons["comp_max"].max())
        rows.append({
            "run": name,
            "variant": variant,
            "latent": latent,
            "rmse_diag_uV": float(diag["rmse"].mean()) * 1e6,
            "r_diag": float(diag["r"].mean()),
            "rmse_cross_uV": float(off["rmse"].mean()) * 1e6,
            "r_cross": float(off["r"].mean()),
            "comp_medio": comp_mean,
            "comp_max": comp_max,
            "val_loss_best": best_val,
            "epochs": n_epochs,
        })
    if not rows:
        print("No hay ejecuciones comparables.")
        return
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))

    # Figuras comparativas en runs/compare/figs
    from .evaluation import plots

    runs_paths = [Path(r) for r in runs if (Path(r) / "metrics_test.csv").exists()]
    out = Path("runs") / "compare"
    if len(runs_paths) >= 2:
        plots.plot_multi_run_heatmap(runs_paths, out, value_col="rmse")
        plots.plot_multi_run_heatmap(runs_paths, out, value_col="r")
        plots.plot_multi_run_consistency(runs_paths, out)
        plots.plot_multi_run_learning(runs_paths, out)
    elif runs_paths:
        run0 = runs_paths[0]
        history = run0 / "history.csv"
        if history.exists():
            plots.plot_learning_curves(history, out, run_label=run0.name)


def cmd_pipeline(cfg, force: bool) -> None:
    cmd_build(cfg)
    cmd_train(cfg, force)
    cmd_eval(cfg)


def cmd_montage(cfg, montages: list[str] | None) -> None:
    from .experiments.montage import run_cmd

    run_cmd(cfg, montages)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eeg-transform",
        description="Universal EEG Transformer: estandarización de referencias.",
    )
    parser.add_argument(
        "-c", "--config", default="config/default.yaml",
        help="Ruta al archivo YAML de configuración.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Construye/prepara el dataset real.")
    p_train = sub.add_parser("train", help="Entrena el transformador lineal.")
    p_train.add_argument("--force", action="store_true",
                         help="Ignora checkpoints previos.")
    p_eval = sub.add_parser("eval", help="Evalúa sobre test y genera figuras.")
    p_comp = sub.add_parser("compare",
                            help="Compara resúmenes de varias ejecuciones.")
    p_comp.add_argument("runs", nargs="+",
                        help="Directorios de ejecución (p. ej. runs/default).")
    p_pipe = sub.add_parser("pipeline", help="build + train + eval.")
    p_pipe.add_argument("--force", action="store_true")
    p_mon = sub.add_parser(
        "montage",
        help="Evalúa la reconstrucción de montajes al espacio canónico.",
    )
    p_mon.add_argument(
        "--montages", default=None,
        help="Lista separada por comas de montajes (p. ej. 10-20,10-10).",
    )

    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging("INFO", log_file=str(Path(cfg.training.run_dir).parent / "pipeline.log"))

    if args.command == "build":
        cmd_build(cfg)
    elif args.command == "train":
        cmd_train(cfg, args.force)
    elif args.command == "eval":
        cmd_eval(cfg)
    elif args.command == "compare":
        cmd_compare(args.runs)
    elif args.command == "pipeline":
        cmd_pipeline(cfg, args.force)
    elif args.command == "montage":
        montages = (
            [m.strip() for m in args.montages.split(",")]
            if args.montages else None
        )
        cmd_montage(cfg, montages)
    return 0


if __name__ == "__main__":
    sys.exit(main())