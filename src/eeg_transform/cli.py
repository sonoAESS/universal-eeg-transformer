"""Interfaz de línea de comandos del Universal EEG Transformer.

Uso:
    eeg-transform build   [--config config/default.yaml]
    eeg-transform train   [--config config/default.yaml] [--force]
    eeg-transform eval    [--config config/default.yaml]
    eeg-transform pipeline[--config config/default.yaml] [--force]
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
    from .training.trainer import build_model

    run_dir = Path(cfg.training.run_dir)
    ds = build_dataset(cfg)

    model = build_model(cfg, ds.n_channels)
    model.ensure_built()
    model.load_weights(str(run_dir / "best.weights.h5"))
    model.compile(optimizer="adam")

    uni_idx = ds.ch_names.index(cfg.data.unipolar_ref_ch)

    metrics_df = metrics.evaluate_routes(model, ds, split="test")
    err_matrix = metrics.transfer_error_matrix(model, ds, uni_idx, cfg)

    print("\n================ METRICAS (TEST) ================")
    print(metrics.summarize(metrics_df, "test").to_string(formatters={
        "mse": "{:.3e}".format, "rmse": "{:.3e}".format,
        "mae": "{:.3e}".format, "r": "{:.4f}".format}))
    print("\nDetalle por ruta:")
    print(metrics_df.round(9).to_string(index=False))

    print("\n================ ERROR MATRIZ vs ANALITICA ================")
    print(err_matrix.round(6).to_string(index=False))

    metrics_df.to_csv(run_dir / "metrics_test.csv", index=False)
    err_matrix.to_csv(run_dir / "error_fro_rel.csv", index=False)

    history_csv = run_dir / "history.csv"
    if history_csv.exists():
        plots.plot_learning_curves(history_csv, run_dir)
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


def cmd_pipeline(cfg, force: bool) -> None:
    cmd_build(cfg)
    cmd_train(cfg, force)
    cmd_eval(cfg)


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
    p_pipe = sub.add_parser("pipeline", help="build + train + eval.")
    p_pipe.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging("INFO", log_file=str(Path(cfg.training.run_dir).parent / "pipeline.log"))

    if args.command == "build":
        cmd_build(cfg)
    elif args.command == "train":
        cmd_train(cfg, args.force)
    elif args.command == "eval":
        cmd_eval(cfg)
    elif args.command == "pipeline":
        cmd_pipeline(cfg, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())