import argparse
import importlib.util
import json
import os
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "prepared_dataset_G5_622").exists():
            return candidate
    raise FileNotFoundError("Could not find prepared_dataset_G5_622 from the current working directory.")


def load_patch_module(repo_root: Path):
    module_path = repo_root / "openset" / "patch_distance_threshold.py"
    spec = importlib.util.spec_from_file_location("patch_distance_threshold_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


KNOWN_LABELS = {
    "DIE_BROKEN": 0,
    "NORMAL": 1,
    "NO_DIE": 2,
}


class WaferDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label, class_name = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, class_name, str(image_path)


def list_image_files(image_dir: Path):
    return sorted([p for p in image_dir.iterdir() if p.is_file()])


def collect_known_samples(data_root: Path, split: str):
    split_root = data_root / split / "G5"
    samples = []
    class_counts = Counter()

    for class_name, label in KNOWN_LABELS.items():
        image_dir = split_root / class_name / "images"
        if not image_dir.exists():
            continue
        for image_path in list_image_files(image_dir):
            samples.append((image_path, label, class_name))
            class_counts[class_name] += 1

    return samples, class_counts


def build_train_transform(image_size: int, aug_cfg: dict):
    ops = [transforms.Resize((image_size, image_size))]
    if aug_cfg.get("horizontal_flip", True):
        ops.append(transforms.RandomHorizontalFlip())
    if aug_cfg.get("vertical_flip", True):
        ops.append(transforms.RandomVerticalFlip())
    if aug_cfg["rotation_degrees"] > 0:
        ops.append(transforms.RandomRotation(aug_cfg["rotation_degrees"]))
    if aug_cfg["contrast_strength"] > 0:
        ops.append(transforms.ColorJitter(contrast=aug_cfg["contrast_strength"]))
    if aug_cfg["sharpness_p"] > 0:
        ops.append(
            transforms.RandomAdjustSharpness(
                sharpness_factor=aug_cfg["sharpness_factor"],
                p=aug_cfg["sharpness_p"],
            )
        )
    if aug_cfg["autocontrast_p"] > 0:
        ops.append(transforms.RandomAutocontrast(p=aug_cfg["autocontrast_p"]))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(ops)


def build_eval_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def create_known_dataloaders(data_root: Path, image_size: int, batch_size: int, num_workers: int, aug_cfg: dict):
    train_samples, train_counts = collect_known_samples(data_root, "train")
    val_samples, val_counts = collect_known_samples(data_root, "val")

    train_dataset = WaferDataset(train_samples, transform=build_train_transform(image_size, aug_cfg))
    val_dataset = WaferDataset(val_samples, transform=build_eval_transform(image_size))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {
        "train_samples": train_samples,
        "val_samples": val_samples,
        "train_counts": train_counts,
        "val_counts": val_counts,
        "train_loader": train_loader,
        "val_loader": val_loader,
    }


def build_training_components(device: torch.device, train_samples, label_smoothing: float, learning_rate: float, weight_decay: float, epochs: int):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(KNOWN_LABELS))
    model = model.to(device)

    train_label_counts = Counter(label for _, label, _ in train_samples)
    class_weights = torch.tensor(
        [
            len(train_samples) / (len(KNOWN_LABELS) * train_label_counts[class_id])
            for class_id in range(len(KNOWN_LABELS))
        ],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return model, criterion, optimizer, scheduler, class_weights


def train_one_epoch(model, loader, criterion, optimizer, device: torch.device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for images, labels, _, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(dim=1).detach().cpu().numpy())
        all_targets.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


@torch.no_grad()
def gather_outputs(model, loader, device: torch.device):
    model.eval()
    logits_list = []
    labels_list = []

    for images, labels, _, _ in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        logits_list.append(logits.cpu())
        labels_list.append(labels)

    logits = torch.cat(logits_list).numpy()
    labels = torch.cat(labels_list).numpy()
    return logits, labels


def evaluate_classifier(logits, labels, device: torch.device, criterion=None):
    preds = logits.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    if criterion is not None:
        logits_tensor = torch.as_tensor(logits, dtype=torch.float32, device=device)
        labels_tensor = torch.as_tensor(labels, dtype=torch.long, device=device)
        metrics["loss"] = criterion(logits_tensor, labels_tensor).item()
    return metrics


def save_json(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def save_checkpoint(state_dict, path: Path, cfg: dict, selection_metric: str, selection_value: float, best_epoch: int, image_size: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": state_dict,
            "cfg": cfg,
            "known_labels": KNOWN_LABELS,
            "image_size": image_size,
            "selection_metric": selection_metric,
            "selection_value": selection_value,
            "best_epoch": best_epoch,
        },
        path,
    )


def save_loss_curve(history, path: Path):
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def build_cfg(args, aug_cfg, train_counts, val_counts, class_weights, output_dir: Path):
    return {
        "experiment": aug_cfg["name"],
        "model": "resnet18",
        "pretrained_weights": "ResNet18_Weights.DEFAULT",
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "num_workers": args.num_workers,
        "image_size": args.image_size,
        "known_labels": KNOWN_LABELS,
        "class_weights": class_weights.detach().cpu().tolist(),
        "train_counts": dict(train_counts),
        "val_counts": dict(val_counts),
        "augmentation": aug_cfg,
        "checkpoint_dir": str(output_dir),
    }


def evaluate_open_set(repo_root: Path, model_path: Path, batch_size: int, num_workers: int):
    patch = load_patch_module(repo_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = repo_root / "prepared_dataset_G5_622"

    train_samples, _ = patch.collect_split_samples(data_root, "train", known_only=True)
    val_samples, _ = patch.collect_split_samples(data_root, "val", known_only=False)
    test_samples, _ = patch.collect_split_samples(data_root, "test", known_only=False)

    model, checkpoint, image_size = patch.load_model(model_path, device)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_loader = DataLoader(
        patch.WaferSplitDataset(train_samples, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        patch.WaferSplitDataset(val_samples, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        patch.WaferSplitDataset(test_samples, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    score_specs = patch.build_score_specs([12], [])
    prototype_maps = patch.build_class_prototype_maps(model, train_loader, device, "layer2")
    train_rows_by_spec = patch.run_inference(model, train_loader, prototype_maps, device, score_specs, "layer2")
    val_rows_by_spec = patch.run_inference(model, val_loader, prototype_maps, device, score_specs, "layer2")
    test_rows_by_spec = patch.run_inference(model, test_loader, prototype_maps, device, score_specs, "layer2")

    train_rows = train_rows_by_spec["topk12"]
    val_rows = val_rows_by_spec["topk12"]
    test_rows = test_rows_by_spec["topk12"]

    normalization_stats = patch.compute_pred_class_normalization_stats(train_rows)
    val_rows = patch.apply_pred_class_normalization(val_rows, normalization_stats)
    test_rows = patch.apply_pred_class_normalization(test_rows, normalization_stats)

    best_threshold, val_metrics = patch.find_best_threshold(val_rows, score_key="normalized_anomaly_score")
    test_eval_rows = patch.apply_threshold(test_rows, best_threshold, score_key="normalized_anomaly_score")
    test_metrics = patch.compute_metrics(test_eval_rows)

    known_rows = [row for row in test_eval_rows if row["true_class"] in KNOWN_LABELS]
    raw_known_correct = sum(row["argmax_index"] == KNOWN_LABELS[row["true_class"]] for row in known_rows)
    raw_known_acc = raw_known_correct / len(known_rows)

    return {
        "selection_metric": checkpoint.get("selection_metric"),
        "selection_value": checkpoint.get("selection_value"),
        "best_epoch": checkpoint.get("best_epoch"),
        "selected_threshold": float(best_threshold),
        "raw_known_acc": float(raw_known_acc),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "normalization_stats": normalization_stats,
        "result_rows_path": str(model_path.with_name(f"{model_path.stem}_layer2_patch_topk12_global_norm_test_results.json")),
        "summary_path": str(model_path.with_name(f"{model_path.stem}_layer2_patch_global_norm_train_known_summary.json")),
    }


def train_and_evaluate_config(args, repo_root: Path, data_root: Path, aug_cfg: dict):
    output_dir = Path(args.output_root) / aug_cfg["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders = create_known_dataloaders(
        data_root=data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        aug_cfg=aug_cfg,
    )
    train_samples = loaders["train_samples"]
    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, criterion, optimizer, scheduler, class_weights = build_training_components(
        device=device,
        train_samples=train_samples,
        label_smoothing=args.label_smoothing,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
    )

    cfg = build_cfg(args, aug_cfg, loaders["train_counts"], loaders["val_counts"], class_weights, output_dir)
    save_json(cfg, output_dir / "config.json")

    best_val_loss = float("inf")
    best_val_f1 = -1.0
    best_loss_epoch = None
    best_loss_state_dict = None
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_logits, val_labels = gather_outputs(model, val_loader, device)
        val_metrics = evaluate_classifier(val_logits, val_labels, device, criterion=criterion)
        scheduler.step()

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_metrics)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_loss_epoch = epoch
            best_loss_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]

        print(
            f"[{aug_cfg['name']}] epoch {epoch:02d} | train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f}"
        )

    ckpt_path = output_dir / f"resnet18_best_val_loss_epoch{best_loss_epoch:02d}.pth"
    save_checkpoint(
        best_loss_state_dict,
        ckpt_path,
        cfg,
        "val_loss",
        float(best_val_loss),
        int(best_loss_epoch),
        args.image_size,
    )
    save_json(history, output_dir / "train_log.json")
    save_loss_curve(history, output_dir / "loss_curve.png")

    train_summary = {
        "experiment": aug_cfg["name"],
        "checkpoint_path": str(ckpt_path),
        "best_epoch": int(best_loss_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_f1": float(best_val_f1),
        "history_path": str(output_dir / "train_log.json"),
        "config_path": str(output_dir / "config.json"),
    }
    save_json(train_summary, output_dir / "summary.json")

    open_set_summary = evaluate_open_set(repo_root, ckpt_path, args.batch_size_eval, args.num_workers)
    save_json(open_set_summary, output_dir / "open_set_summary.json")

    merged_summary = {
        **train_summary,
        **{
            "selected_threshold": open_set_summary["selected_threshold"],
            "raw_known_acc": open_set_summary["raw_known_acc"],
            "open_macro_f1": open_set_summary["test_metrics"]["open_macro_f1"],
            "overall_acc": open_set_summary["test_metrics"]["open_accuracy"],
            "known_acc": open_set_summary["test_metrics"]["known_accuracy"],
            "unknown_acc": open_set_summary["test_metrics"]["unknown_accuracy"],
            "val_open_macro_f1": open_set_summary["val_metrics"]["open_macro_f1"],
            "val_open_accuracy": open_set_summary["val_metrics"]["open_accuracy"],
        },
    }
    save_json(merged_summary, output_dir / "combined_summary.json")
    return merged_summary


def write_markdown(summary_rows, path: Path, title: str):
    ordered = sorted(summary_rows, key=lambda row: row["open_macro_f1"], reverse=True)
    lines = [f"# {title}", "", "| Rank | Experiment | Best epoch | Best val loss | Threshold | Raw known acc | Open macro F1 | Overall acc | Known acc | Unknown acc |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for idx, row in enumerate(ordered, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    f"`{row['experiment']}`",
                    str(row["best_epoch"]),
                    f"{row['best_val_loss']:.4f}",
                    f"{row['selected_threshold']:.6f}",
                    f"{row['raw_known_acc']:.4f}",
                    f"{row['open_macro_f1']:.4f}",
                    f"{row['overall_acc']:.4f}",
                    f"{row['known_acc']:.4f}",
                    f"{row['unknown_acc']:.4f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Fine-search augmentation configs for the ResNet18 wafer baseline.")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--configs-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-size-eval", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    repo_root = args.repo_root or find_repo_root(Path.cwd().resolve())
    data_root = repo_root / "prepared_dataset_G5_622"
    configs = json.loads(args.configs_json.read_text())

    results = []
    for aug_cfg in configs:
        print(f"===== fine-search {aug_cfg['name']} =====")
        results.append(train_and_evaluate_config(args, repo_root, data_root, aug_cfg))
        print()

    args.output_root.mkdir(parents=True, exist_ok=True)
    save_json(results, args.output_root / "search_summary.json")
    write_markdown(results, args.output_root / "search_summary.md", "V2 Fine Search Summary")

    best = max(results, key=lambda row: row["open_macro_f1"])
    print("best config:", best["experiment"])
    print(
        {
            "open_macro_f1": round(best["open_macro_f1"], 4),
            "overall_acc": round(best["overall_acc"], 4),
            "known_acc": round(best["known_acc"], 4),
            "unknown_acc": round(best["unknown_acc"], 4),
            "threshold": round(best["selected_threshold"], 6),
        }
    )


if __name__ == "__main__":
    os.environ.setdefault("WANDB_MODE", "disabled")
    main()
