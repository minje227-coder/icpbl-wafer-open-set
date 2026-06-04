import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


KNOWN_LABELS = {
    "DIE_BROKEN": 0,
    "NORMAL": 1,
    "NO_DIE": 2,
}
ALL_TRUE_CLASSES = ["DIE_BROKEN", "DIE_CRACK", "DIE_INK", "NORMAL", "NO_DIE"]
PRED_LABELS = ["DIE_BROKEN", "NORMAL", "NO_DIE"]
UNKNOWN_PRED_NAME = "UNKNOWN"
UNKNOWN_PRED_INDEX = 3
FEATURE_LAYERS = ("layer2", "layer3", "layer4")


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "prepared_dataset_G5_622").exists():
            return candidate
    raise FileNotFoundError("Could not find prepared_dataset_G5_622 from the current working directory.")


def true_class_to_index(true_class: str) -> int:
    return KNOWN_LABELS.get(true_class, UNKNOWN_PRED_INDEX)


class WaferSplitDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, true_class = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, true_class, str(image_path)


def collect_split_samples(data_root: Path, split: str, known_only: bool = False):
    split_root = data_root / split / "G5"
    samples = []
    class_counts = Counter()

    class_names = list(KNOWN_LABELS) if known_only else ALL_TRUE_CLASSES
    for class_name in class_names:
        image_dir = split_root / class_name / "images"
        if not image_dir.exists():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if image_path.is_file():
                samples.append((image_path, class_name))
                class_counts[class_name] += 1

    return samples, class_counts


def load_model(model_path: Path, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    image_size = checkpoint.get("image_size", 224)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(KNOWN_LABELS))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint, image_size


def resolve_model_paths(model_path: Path | None, ckpt_dir: Path | None, repo_root: Path):
    if model_path is not None and ckpt_dir is not None:
        raise ValueError("Use either --model-path or --ckpt-dir, not both.")

    if ckpt_dir is not None:
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
        if not ckpt_dir.is_dir():
            raise NotADirectoryError(f"Checkpoint path is not a directory: {ckpt_dir}")

        model_paths = sorted(p for p in ckpt_dir.glob("*.pth") if p.is_file())
        if not model_paths:
            raise FileNotFoundError(f"No .pth checkpoints found in: {ckpt_dir}")
        return model_paths

    resolved_model_path = model_path or (repo_root / "model.pth")
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_model_path}")
    return [resolved_model_path]


def forward_feature_map_and_logits(model, images, feature_layer: str):
    x = model.conv1(images)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    if feature_layer == "layer2":
        feature_map = x
    x = model.layer3(x)
    if feature_layer == "layer3":
        feature_map = x
    x = model.layer4(x)
    if feature_layer == "layer4":
        feature_map = x
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    logits = model.fc(x)
    return feature_map, logits


def normalize_patch_map(feature_map: torch.Tensor):
    return F.normalize(feature_map, dim=1, eps=1e-8)


@torch.no_grad()
def build_class_prototype_maps(model, loader, device, feature_layer: str):
    feature_buckets = defaultdict(list)

    for images, true_classes, _ in loader:
        images = images.to(device, non_blocking=True)
        feature_maps, _ = forward_feature_map_and_logits(model, images, feature_layer)
        feature_maps = normalize_patch_map(feature_maps)
        for feature_map, true_class in zip(feature_maps, true_classes):
            feature_buckets[KNOWN_LABELS[true_class]].append(feature_map.detach().cpu())

    missing = [PRED_LABELS[class_idx] for class_idx in range(len(PRED_LABELS)) if class_idx not in feature_buckets]
    if missing:
        raise ValueError(f"Missing train features for classes: {missing}")

    prototypes = []
    for class_idx in range(len(PRED_LABELS)):
        class_maps = torch.stack(feature_buckets[class_idx], dim=0)
        prototype = class_maps.mean(dim=0)
        prototype = normalize_patch_map(prototype.unsqueeze(0)).squeeze(0)
        prototypes.append(prototype)

    return torch.stack(prototypes, dim=0).to(device)


def patch_distance_map(feature_map: torch.Tensor, prototype_map: torch.Tensor):
    return 1.0 - (feature_map * prototype_map).sum(dim=0)


def topk_mean(distance_map: torch.Tensor, topk: int):
    flat = distance_map.flatten()
    k = min(topk, flat.numel())
    values = torch.topk(flat, k=k, largest=True).values
    return values.mean(), values


def top_percent_mean(distance_map: torch.Tensor, percent: float):
    flat = distance_map.flatten()
    k = max(1, int(np.ceil(flat.numel() * (percent / 100.0))))
    values = torch.topk(flat, k=k, largest=True).values
    return values.mean(), values, k


def build_score_specs(topk_values, top_percent_values):
    specs = []
    for topk in topk_values:
        specs.append(
            {
                "mode": "topk",
                "value": int(topk),
                "label": f"topk{int(topk)}",
            }
        )
    for top_percent in top_percent_values:
        percent_str = f"{top_percent:g}"
        specs.append(
            {
                "mode": "top_percent",
                "value": float(top_percent),
                "label": f"topp{percent_str}pct",
            }
        )
    return specs


@torch.no_grad()
def run_inference(model, loader, prototype_maps, device, score_specs, feature_layer: str):
    rows_by_spec = {spec["label"]: [] for spec in score_specs}

    for images, true_classes, paths in loader:
        images = images.to(device, non_blocking=True)
        feature_maps, logits = forward_feature_map_and_logits(model, images, feature_layer)
        probs = torch.softmax(logits, dim=1)
        pred_indices = probs.argmax(dim=1)
        confidences = probs.max(dim=1).values
        feature_maps = normalize_patch_map(feature_maps)

        for batch_idx, (true_class, path) in enumerate(zip(true_classes, paths)):
            pred_index = int(pred_indices[batch_idx].item())
            distance_map = patch_distance_map(feature_maps[batch_idx], prototype_maps[pred_index]).detach().cpu()
            base_row = {
                "path": path,
                "true_class": true_class,
                "true_index": true_class_to_index(true_class),
                "argmax_index": pred_index,
                "argmax_label": PRED_LABELS[pred_index],
                "confidence": float(confidences[batch_idx].item()),
                "prob_DIE_BROKEN": float(probs[batch_idx, 0].item()),
                "prob_NORMAL": float(probs[batch_idx, 1].item()),
                "prob_NO_DIE": float(probs[batch_idx, 2].item()),
                "feature_layer": feature_layer,
                "distance_map_shape": list(distance_map.shape),
                "distance_map_mean": float(distance_map.mean().item()),
                "distance_map_max": float(distance_map.max().item()),
            }
            for spec in score_specs:
                row = dict(base_row)
                if spec["mode"] == "topk":
                    score, top_scores = topk_mean(distance_map, spec["value"])
                    row["topk"] = spec["value"]
                    row["top_percent"] = None
                    row["score_mode"] = spec["label"]
                else:
                    score, top_scores, effective_topk = top_percent_mean(distance_map, spec["value"])
                    row["topk"] = effective_topk
                    row["top_percent"] = spec["value"]
                    row["score_mode"] = spec["label"]
                row["anomaly_score"] = float(score.item())
                row["top_patch_distances"] = [float(value) for value in top_scores.tolist()]
                rows_by_spec[spec["label"]].append(row)

    return rows_by_spec


def apply_threshold(rows, threshold: float | None, score_key: str = "anomaly_score"):
    thresholded_rows = []
    for row in rows:
        new_row = dict(row)
        pred_index = row["argmax_index"]
        if threshold is not None and row[score_key] > threshold:
            pred_index = UNKNOWN_PRED_INDEX
        new_row["pred_index"] = pred_index
        new_row["pred_label"] = PRED_LABELS[pred_index] if pred_index < len(PRED_LABELS) else UNKNOWN_PRED_NAME
        new_row["applied_threshold"] = None if threshold is None else float(threshold)
        new_row["threshold_score_key"] = score_key
        thresholded_rows.append(new_row)
    return thresholded_rows


def apply_classwise_thresholds(rows, thresholds: dict[int, float], score_key: str = "anomaly_score"):
    thresholded_rows = []
    for row in rows:
        new_row = dict(row)
        pred_index = row["argmax_index"]
        threshold = thresholds[pred_index]
        if row[score_key] > threshold:
            pred_index = UNKNOWN_PRED_INDEX
        new_row["pred_index"] = pred_index
        new_row["pred_label"] = PRED_LABELS[pred_index] if pred_index < len(PRED_LABELS) else UNKNOWN_PRED_NAME
        new_row["applied_threshold"] = float(threshold)
        new_row["threshold_score_key"] = score_key
        thresholded_rows.append(new_row)
    return thresholded_rows


def compute_metrics(rows):
    true_indices = [row["true_index"] for row in rows]
    pred_indices = [row["pred_index"] for row in rows]
    known_rows = [row for row in rows if row["true_class"] in KNOWN_LABELS]
    known_true = [KNOWN_LABELS[row["true_class"]] for row in known_rows]
    known_pred = [row["pred_index"] for row in known_rows]
    unknown_rows = [row for row in rows if row["true_class"] not in KNOWN_LABELS]
    unknown_correct = sum(row["pred_index"] == UNKNOWN_PRED_INDEX for row in unknown_rows)
    return {
        "open_accuracy": accuracy_score(true_indices, pred_indices),
        "open_macro_f1": f1_score(true_indices, pred_indices, labels=[0, 1, 2, 3], average="macro", zero_division=0),
        "known_accuracy": accuracy_score(known_true, known_pred) if known_true else None,
        "unknown_accuracy": (unknown_correct / len(unknown_rows)) if unknown_rows else None,
    }


def build_threshold_candidates(rows, score_key: str = "anomaly_score"):
    scores = sorted({row[score_key] for row in rows})
    if not scores:
        raise ValueError("No anomaly scores available for threshold search.")
    eps = 1e-6
    candidates = [scores[0] - eps]
    candidates.extend(scores)
    candidates.append(scores[-1] + eps)
    return candidates


def find_best_threshold(rows, score_key: str = "anomaly_score"):
    best_threshold = None
    best_metrics = None

    for threshold in build_threshold_candidates(rows, score_key):
        thresholded_rows = apply_threshold(rows, float(threshold), score_key=score_key)
        metrics = compute_metrics(thresholded_rows)

        if best_metrics is None:
            best_threshold = float(threshold)
            best_metrics = metrics
            continue
        if metrics["open_macro_f1"] > best_metrics["open_macro_f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics
            continue
        if metrics["open_macro_f1"] == best_metrics["open_macro_f1"] and metrics["open_accuracy"] > best_metrics["open_accuracy"]:
            best_threshold = float(threshold)
            best_metrics = metrics
            continue
        if (
            metrics["open_macro_f1"] == best_metrics["open_macro_f1"]
            and metrics["open_accuracy"] == best_metrics["open_accuracy"]
            and metrics["known_accuracy"] is not None
            and best_metrics["known_accuracy"] is not None
            and metrics["known_accuracy"] > best_metrics["known_accuracy"]
        ):
            best_threshold = float(threshold)
            best_metrics = metrics
            continue
        if (
            metrics["open_macro_f1"] == best_metrics["open_macro_f1"]
            and metrics["open_accuracy"] == best_metrics["open_accuracy"]
            and metrics["known_accuracy"] == best_metrics["known_accuracy"]
            and float(threshold) < best_threshold
        ):
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def find_best_threshold_for_class(rows, class_idx: int, score_key: str = "anomaly_score"):
    class_rows = [row for row in rows if row["argmax_index"] == class_idx]
    if not class_rows:
        raise ValueError(f"No validation rows predicted as {PRED_LABELS[class_idx]}.")

    target_class = PRED_LABELS[class_idx]
    best_threshold = None
    best_metrics = None

    for threshold in build_threshold_candidates(class_rows, score_key):
        tp = fp = fn = tn = 0
        for row in class_rows:
            accept = row[score_key] <= float(threshold)
            positive = row["true_class"] == target_class
            if accept and positive:
                tp += 1
            elif accept and not positive:
                fp += 1
            elif not accept and positive:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(class_rows)
        metrics = {
            "f1": f1,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "support": len(class_rows),
        }

        if best_metrics is None:
            best_threshold = float(threshold)
            best_metrics = metrics
            continue
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics
            continue
        if metrics["f1"] == best_metrics["f1"] and metrics["accuracy"] > best_metrics["accuracy"]:
            best_threshold = float(threshold)
            best_metrics = metrics
            continue
        if metrics["f1"] == best_metrics["f1"] and metrics["accuracy"] == best_metrics["accuracy"] and float(threshold) < best_threshold:
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def find_best_classwise_thresholds(rows, score_key: str = "anomaly_score"):
    thresholds = {}
    class_metrics = {}
    for class_idx in range(len(PRED_LABELS)):
        best_threshold, best_metrics = find_best_threshold_for_class(rows, class_idx, score_key=score_key)
        thresholds[class_idx] = best_threshold
        class_metrics[class_idx] = best_metrics
    thresholded_rows = apply_classwise_thresholds(rows, thresholds, score_key=score_key)
    overall_metrics = compute_metrics(thresholded_rows)
    return thresholds, class_metrics, overall_metrics


def compute_pred_class_normalization_stats(rows):
    stats = {}
    for class_idx, class_name in enumerate(PRED_LABELS):
        class_rows = [
            row
            for row in rows
            if row["true_class"] == class_name and row["argmax_index"] == class_idx
        ]
        if not class_rows:
            class_rows = [row for row in rows if row["true_class"] == class_name]
        scores = np.array([row["anomaly_score"] for row in class_rows], dtype=float)
        mean = float(scores.mean())
        std = float(scores.std())
        if std < 1e-8:
            std = 1.0
        stats[class_idx] = {"mean": mean, "std": std, "n": int(len(scores))}
    return stats


def apply_pred_class_normalization(rows, stats):
    normalized_rows = []
    for row in rows:
        new_row = dict(row)
        pred_index = row["argmax_index"]
        class_stats = stats[pred_index]
        z_score = (row["anomaly_score"] - class_stats["mean"]) / class_stats["std"]
        new_row["normalized_anomaly_score"] = float(z_score)
        new_row["normalization_mean"] = float(class_stats["mean"])
        new_row["normalization_std"] = float(class_stats["std"])
        normalized_rows.append(new_row)
    return normalized_rows


def build_confusion_matrix(rows):
    matrix = np.zeros((len(ALL_TRUE_CLASSES), len(PRED_LABELS) + 1), dtype=int)
    for row in rows:
        matrix[ALL_TRUE_CLASSES.index(row["true_class"]), row["pred_index"]] += 1
    return matrix


def summarize_results(rows, score_spec, threshold, score_key: str):
    per_true_class = defaultdict(list)
    for row in rows:
        per_true_class[row["true_class"]].append(row)

    print(f"num samples: {len(rows)}")
    if score_spec["mode"] == "topk":
        print(f"top-k patches: {score_spec['value']}")
    else:
        print(f"top-percent patches: {score_spec['value']}%")
    if isinstance(threshold, dict):
        print("patch anomaly thresholds (class-wise):")
        for class_idx, value in threshold.items():
            print(f"  - {PRED_LABELS[class_idx]}: {value:.6f}")
    else:
        print(f"patch anomaly threshold: {threshold:.6f}")
    print("threshold score key:", score_key)
    print()

    known_rows = [row for row in rows if row["true_class"] in KNOWN_LABELS]
    known_true = [KNOWN_LABELS[row["true_class"]] for row in known_rows]
    known_pred = [row["pred_index"] for row in known_rows]
    print("Known-class accuracy:", f"{accuracy_score(known_true, known_pred):.4f}")
    print()
    print("Known-class classification report (including UNKNOWN rejections):")
    print(
        classification_report(
            known_true,
            known_pred,
            labels=[0, 1, 2, 3],
            target_names=PRED_LABELS + [UNKNOWN_PRED_NAME],
            zero_division=0,
        )
    )

    true_indices = [row["true_index"] for row in rows]
    pred_indices = [row["pred_index"] for row in rows]
    print("Open-set accuracy:", f"{accuracy_score(true_indices, pred_indices):.4f}")
    print("Open-set macro F1:", f"{f1_score(true_indices, pred_indices, labels=[0, 1, 2, 3], average='macro', zero_division=0):.4f}")
    print()
    matrix = build_confusion_matrix(rows)
    print("Confusion matrix: true 5 classes x predicted labels")
    print("true classes order:", ALL_TRUE_CLASSES)
    print("pred labels order:", PRED_LABELS + [UNKNOWN_PRED_NAME])
    print(matrix)
    print()

    print("Per-class prediction summary:")
    for class_name in ALL_TRUE_CLASSES:
        items = per_true_class.get(class_name, [])
        if not items:
            continue
        pred_counter = Counter(item["pred_label"] for item in items)
        mean_conf = np.mean([item["confidence"] for item in items])
        mean_score = np.mean([item["anomaly_score"] for item in items])
        mean_threshold_score = np.mean([item[score_key] for item in items])
        mean_map_max = np.mean([item["distance_map_max"] for item in items])
        print(
            f"- {class_name}: {len(items)} samples, mean confidence={mean_conf:.4f}, "
            f"mean anomaly score={mean_score:.4f}, mean threshold score={mean_threshold_score:.4f}, "
            f"mean map max={mean_map_max:.4f}, "
            f"pred distribution={dict(pred_counter)}"
        )


def save_results(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2))
    print()
    print("Saved per-image results to", output_path)


def resolve_output_path(model_path: Path, feature_layer: str, score_label: str, output_json: Path | None, multiple_models: bool, classwise_threshold: bool, normalized_score: bool):
    suffix = "classwise" if classwise_threshold else "global"
    norm_suffix = "_norm" if normalized_score else ""
    auto_name = f"{model_path.stem}_{feature_layer}_patch_{score_label}_{suffix}{norm_suffix}_test_results.json"
    if output_json is None:
        return model_path.with_name(auto_name)
    if output_json.suffix == "":
        return output_json / auto_name
    if multiple_models:
        return output_json.with_name(f"{output_json.stem}_{model_path.stem}_{score_label}{norm_suffix}{output_json.suffix}")
    return output_json.with_name(f"{output_json.stem}_{score_label}{norm_suffix}{output_json.suffix}")


def main():
    parser = argparse.ArgumentParser(
        description="Patch-level prototype distance thresholding on a selected ResNet18 feature layer with top-k anomaly scores."
    )
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--feature-layer", type=str, default="layer3", choices=FEATURE_LAYERS)
    parser.add_argument("--topk", type=int, nargs="+", default=[])
    parser.add_argument("--top-percent", type=float, nargs="+", default=[])
    parser.add_argument("--class-wise-threshold", action="store_true")
    parser.add_argument("--normalize-by-pred-class", action="store_true")
    parser.add_argument("--normalization-source", type=str, default="val_known", choices=("val_known", "train_known"))
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root or find_repo_root(Path.cwd().resolve())
    data_root = repo_root / "prepared_dataset_G5_622"
    model_paths = resolve_model_paths(args.model_path, args.ckpt_dir, repo_root)
    if not args.topk and not args.top_percent:
        args.topk = [3, 5, 7, 12]

    topk_values = sorted(set(args.topk))
    top_percent_values = sorted(set(args.top_percent))
    if any(topk <= 0 for topk in topk_values):
        raise ValueError(f"topk must be positive. Received: {topk_values}")
    if any(top_percent <= 0 or top_percent > 100 for top_percent in top_percent_values):
        raise ValueError(f"top-percent must be in (0, 100]. Received: {top_percent_values}")
    if not topk_values and not top_percent_values:
        raise ValueError("Provide at least one of --topk or --top-percent.")
    score_specs = build_score_specs(topk_values, top_percent_values)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_samples, train_class_counts = collect_split_samples(data_root, "train", known_only=True)
    val_samples, val_class_counts = collect_split_samples(data_root, "val", known_only=False)
    test_samples, test_class_counts = collect_split_samples(data_root, "test", known_only=False)

    print("repo root:", repo_root)
    print("device:", device)
    print("train class counts:", dict(train_class_counts))
    print("val class counts:", dict(val_class_counts))
    print("test class counts:", dict(test_class_counts))
    print("feature layer:", args.feature_layer)
    print("threshold mode:", "class-wise" if args.class_wise_threshold else "global")
    print("normalize by predicted class:", args.normalize_by_pred_class)
    print("normalization source:", args.normalization_source)
    print("score specs:", [spec["label"] for spec in score_specs])
    print()

    for idx, model_path in enumerate(model_paths, start=1):
        model, checkpoint, image_size = load_model(model_path, device)
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        train_loader = DataLoader(WaferSplitDataset(train_samples, transform=transform), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(WaferSplitDataset(val_samples, transform=transform), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(WaferSplitDataset(test_samples, transform=transform), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        print(f"===== Checkpoint {idx}/{len(model_paths)} =====")
        print("model path:", model_path)
        selection_metric = checkpoint.get("selection_metric")
        selection_value = checkpoint.get("selection_value")
        best_epoch = checkpoint.get("best_epoch")
        if selection_metric is not None:
            print("selection metric:", selection_metric)
        if selection_value is not None:
            print("selection value:", selection_value)
        if best_epoch is not None:
            print("best epoch:", best_epoch)

        prototype_maps = build_class_prototype_maps(model, train_loader, device, args.feature_layer)
        print("prototype map shape per class:", tuple(prototype_maps.shape[1:]))
        train_rows_by_spec = run_inference(model, train_loader, prototype_maps, device, score_specs, args.feature_layer)
        val_rows_by_spec = run_inference(model, val_loader, prototype_maps, device, score_specs, args.feature_layer)
        test_rows_by_spec = run_inference(model, test_loader, prototype_maps, device, score_specs, args.feature_layer)

        summary_rows = []
        for spec in score_specs:
            print()
            print(f"----- {spec['label']} -----")
            val_rows = val_rows_by_spec[spec["label"]]
            test_rows = test_rows_by_spec[spec["label"]]
            score_key = "anomaly_score"
            normalization_stats = None
            if args.normalize_by_pred_class:
                stats_rows = train_rows_by_spec[spec["label"]] if args.normalization_source == "train_known" else val_rows
                normalization_stats = compute_pred_class_normalization_stats(stats_rows)
                val_rows = apply_pred_class_normalization(val_rows, normalization_stats)
                test_rows = apply_pred_class_normalization(test_rows, normalization_stats)
                score_key = "normalized_anomaly_score"
                print("predicted-class normalization stats:")
                for class_idx, class_stats in normalization_stats.items():
                    print(
                        f"  - {PRED_LABELS[class_idx]}:",
                        {
                            "mean": round(class_stats["mean"], 6),
                            "std": round(class_stats["std"], 6),
                            "n": class_stats["n"],
                        },
                    )
            if args.class_wise_threshold:
                best_threshold, class_threshold_metrics, val_metrics = find_best_classwise_thresholds(val_rows, score_key=score_key)
                val_eval_rows = apply_classwise_thresholds(val_rows, best_threshold, score_key=score_key)
                test_eval_rows = apply_classwise_thresholds(test_rows, best_threshold, score_key=score_key)
                test_metrics = compute_metrics(test_eval_rows)
                print("best validation thresholds (class-wise):")
                for class_idx, threshold in best_threshold.items():
                    print(f"  - {PRED_LABELS[class_idx]}: {threshold:.6f}")
                print("per-class validation threshold metrics:")
                for class_idx, metrics in class_threshold_metrics.items():
                    print(
                        f"  - {PRED_LABELS[class_idx]}:",
                        {
                            "f1": round(metrics["f1"], 4),
                            "accuracy": round(metrics["accuracy"], 4),
                            "precision": round(metrics["precision"], 4),
                            "recall": round(metrics["recall"], 4),
                            "support": metrics["support"],
                        },
                    )
            else:
                best_threshold, val_metrics = find_best_threshold(val_rows, score_key=score_key)
                val_eval_rows = apply_threshold(val_rows, best_threshold, score_key=score_key)
                test_eval_rows = apply_threshold(test_rows, best_threshold, score_key=score_key)
                test_metrics = compute_metrics(test_eval_rows)
                print("best validation threshold:", f"{best_threshold:.6f}")

            print(
                "validation metrics after thresholding:",
                {
                    "open_accuracy": round(val_metrics["open_accuracy"], 4),
                    "open_macro_f1": round(val_metrics["open_macro_f1"], 4),
                    "known_accuracy": None if val_metrics["known_accuracy"] is None else round(val_metrics["known_accuracy"], 4),
                    "unknown_accuracy": None if val_metrics["unknown_accuracy"] is None else round(val_metrics["unknown_accuracy"], 4),
                },
            )
            print()
            print("Test summary:")
            summarize_results(test_eval_rows, spec, best_threshold, score_key=score_key)

            output_path = resolve_output_path(
                model_path,
                args.feature_layer,
                spec["label"],
                args.output_json,
                len(model_paths) > 1,
                args.class_wise_threshold,
                args.normalize_by_pred_class,
            )
            save_results(test_eval_rows, output_path)
            summary_rows.append(
                {
                    "feature_layer": args.feature_layer,
                    "threshold_mode": "class-wise" if args.class_wise_threshold else "global",
                    "score_mode": spec["label"],
                    "topk": None if spec["mode"] == "top_percent" else spec["value"],
                    "top_percent": spec["value"] if spec["mode"] == "top_percent" else None,
                    "normalization": "pred-class-zscore" if args.normalize_by_pred_class else None,
                    "normalization_source": args.normalization_source if args.normalize_by_pred_class else None,
                    "normalization_stats": normalization_stats,
                    "threshold": best_threshold,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "output_path": str(output_path),
                }
            )
            print()

        summary_suffix = "classwise" if args.class_wise_threshold else "global"
        norm_suffix = "_norm" if args.normalize_by_pred_class else ""
        source_suffix = f"_{args.normalization_source}" if args.normalize_by_pred_class else ""
        summary_path = model_path.with_name(f"{model_path.stem}_{args.feature_layer}_patch_{summary_suffix}{norm_suffix}{source_suffix}_summary.json")
        summary_path.write_text(json.dumps(summary_rows, indent=2))
        print("Saved summary to", summary_path)


if __name__ == "__main__":
    main()
