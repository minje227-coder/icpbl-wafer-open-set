#!/usr/bin/env python3
import argparse
import json
import netrc
import os
from pathlib import Path

import matplotlib.pyplot as plt
import wandb


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def has_wandb_credentials() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        auth = netrc.netrc().authenticators("api.wandb.ai")
    except (FileNotFoundError, netrc.NetrcParseError):
        auth = None
    return auth is not None


def resolve_mode(requested_mode: str) -> str:
    if requested_mode in {"online", "offline", "disabled"}:
        return requested_mode
    return "online" if has_wandb_credentials() else "offline"


def save_combined_metric_figure(history, output_path: Path):
    epochs = [row["epoch"] for row in history]
    metric_names = ["loss", "accuracy", "precision", "recall", "f1"]
    titles = {
        "loss": "Loss",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1-score",
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]
        train_values = [row[f"train_{metric_name}"] for row in history]
        val_values = [row[f"val_{metric_name}"] for row in history]
        ax.plot(epochs, train_values, marker="o", linewidth=1.8, markersize=3.5, label="Train")
        ax.plot(epochs, val_values, marker="s", linewidth=1.8, markersize=3.5, label="Validation")
        ax.set_title(titles[metric_name])
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        if metric_name != "loss":
            ax.set_ylim(0.0, 1.05)
        ax.legend()

    axes[-1].axis("off")
    fig.suptitle("Best Model Training Curves", fontsize=16)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Backfill the saved best-model training history to Weights & Biases."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "best_run",
        help="Directory containing config.json, train_log.json, and summary.json.",
    )
    parser.add_argument(
        "--project",
        default="icpbl-wafer-open-set",
        help="WandB project name.",
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="WandB entity/team. Leave unset to use the logged-in default.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Run name. Defaults to '<experiment>-best-history'.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "online", "offline", "disabled"],
        default="auto",
        help="WandB mode. 'auto' uses online when credentials are available, otherwise offline.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config = load_json(run_dir / "config.json")
    history = load_json(run_dir / "train_log.json")
    summary = load_json(run_dir / "summary.json")

    open_set_summary_path = run_dir / "open_set_summary.json"
    open_set_summary = load_json(open_set_summary_path) if open_set_summary_path.exists() else None

    run_name = args.name or f"{config['experiment']}-best-history"
    mode = resolve_mode(args.mode)
    combined_figure_path = run_dir / "combined_training_curves.png"
    save_combined_metric_figure(history, combined_figure_path)

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=run_name,
        config=config,
        mode=mode,
    )
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    for row in history:
        wandb.log(row, step=row["epoch"])

    run.summary["best_epoch"] = summary["best_epoch"]
    run.summary["best_val_loss"] = summary["best_val_loss"]
    run.summary["best_val_f1"] = summary["best_val_f1"]
    run.summary["checkpoint_path"] = summary["checkpoint_path"]
    run.summary["history_path"] = summary["history_path"]

    if open_set_summary is not None:
        run.summary["open_set_selection_metric"] = open_set_summary["selection_metric"]
        run.summary["open_set_selection_value"] = open_set_summary["selection_value"]
        run.summary["selected_threshold"] = open_set_summary["selected_threshold"]
        run.summary["raw_known_acc"] = open_set_summary["raw_known_acc"]
        for split_name, metrics in open_set_summary.items():
            if split_name.endswith("_metrics"):
                for metric_name, value in metrics.items():
                    run.summary[f"{split_name}/{metric_name}"] = value

    loss_curve_path = run_dir / "loss_curve.png"
    if loss_curve_path.exists():
        wandb.log({"loss_curve_png": wandb.Image(str(loss_curve_path))}, step=history[-1]["epoch"])
    if combined_figure_path.exists():
        wandb.log({"combined_training_curves": wandb.Image(str(combined_figure_path))}, step=history[-1]["epoch"])

    if mode != "disabled":
        run.finish()

    print(f"run_dir={run_dir}")
    print(f"wandb_mode={mode}")
    if getattr(run, "url", None):
        print(f"wandb_url={run.url}")
    else:
        print("wandb_url=<offline-or-disabled>")


if __name__ == "__main__":
    main()
