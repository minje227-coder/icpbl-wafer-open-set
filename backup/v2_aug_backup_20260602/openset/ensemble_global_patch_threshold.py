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


def forward_layer2_and_global(model, images):
    x = model.conv1(images)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    layer2 = model.layer2(x)
    x = model.layer3(layer2)
    x = model.layer4(x)
    x = model.avgpool(x)
    global_feature = torch.flatten(x, 1)
    logits = model.fc(global_feature)
    return layer2, global_feature, logits


def normalize_patch_map(feature_map: torch.Tensor):
    return F.normalize(feature_map, dim=1, eps=1e-8)


def topk_mean(distance_map: torch.Tensor, topk: int):
    flat = distance_map.flatten()
    k = min(topk, flat.numel())
    values = torch.topk(flat, k=k, largest=True).values
    return values.mean()


@torch.no_grad()
def build_prototypes(model, loader, device):
    patch_buckets = defaultdict(list)
    global_buckets = defaultdict(list)

    for images, true_classes, _ in loader:
        images = images.to(device, non_blocking=True)
        layer2, global_feature, _ = forward_layer2_and_global(model, images)
        layer2 = normalize_patch_map(layer2)
        global_feature = F.normalize(global_feature, dim=1)

        for patch_map, global_vec, true_class in zip(layer2, global_feature, true_classes):
            class_idx = KNOWN_LABELS[true_class]
            patch_buckets[class_idx].append(patch_map.detach().cpu())
            global_buckets[class_idx].append(global_vec.detach().cpu())

    patch_prototypes = []
    global_prototypes = []
    for class_idx in range(len(PRED_LABELS)):
        class_patch = torch.stack(patch_buckets[class_idx], dim=0)
        class_global = torch.stack(global_buckets[class_idx], dim=0)
        patch_proto = normalize_patch_map(class_patch.mean(dim=0, keepdim=True)).squeeze(0)
        global_proto = F.normalize(class_global.mean(dim=0), dim=0)
        patch_prototypes.append(patch_proto)
        global_prototypes.append(global_proto)

    return torch.stack(patch_prototypes, dim=0).to(device), torch.stack(global_prototypes, dim=0).to(device)


@torch.no_grad()
def run_inference(model, loader, patch_prototypes, global_prototypes, device, topk):
    rows = []

    for images, true_classes, paths in loader:
        images = images.to(device, non_blocking=True)
        layer2, global_feature, logits = forward_layer2_and_global(model, images)
        probs = torch.softmax(logits, dim=1)
        pred_indices = probs.argmax(dim=1)
        confidences = probs.max(dim=1).values
        layer2 = normalize_patch_map(layer2)
        global_feature = F.normalize(global_feature, dim=1)
        global_distances = 1.0 - torch.matmul(global_feature, global_prototypes.T)

        for batch_idx, (true_class, path) in enumerate(zip(true_classes, paths)):
            pred_index = int(pred_indices[batch_idx].item())
            patch_distance_map = 1.0 - (layer2[batch_idx] * patch_prototypes[pred_index]).sum(dim=0)
            patch_score = topk_mean(patch_distance_map, topk)
            rows.append(
                {
                    "path": path,
                    "true_class": true_class,
                    "true_index": true_class_to_index(true_class),
                    "argmax_index": pred_index,
                    "argmax_label": PRED_LABELS[pred_index],
                    "confidence": float(confidences[batch_idx].item()),
                    "patch_score": float(patch_score.item()),
                    "global_score": float(global_distances[batch_idx, pred_index].item()),
                }
            )

    return rows


def compute_pred_class_stats(rows, score_key: str):
    stats = {}
    for class_idx, class_name in enumerate(PRED_LABELS):
        class_rows = [row for row in rows if row["true_class"] == class_name]
        scores = np.array([row[score_key] for row in class_rows], dtype=float)
        mean = float(scores.mean())
        std = float(scores.std())
        if std < 1e-8:
            std = 1.0
        stats[class_idx] = {"mean": mean, "std": std, "n": int(len(scores))}
    return stats


def apply_normalization(rows, score_key: str, out_key: str, stats):
    normalized_rows = []
    for row in rows:
        new_row = dict(row)
        pred_index = row["argmax_index"]
        class_stats = stats[pred_index]
        new_row[out_key] = float((row[score_key] - class_stats["mean"]) / class_stats["std"])
        normalized_rows.append(new_row)
    return normalized_rows


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


def apply_threshold(rows, threshold: float, score_key: str):
    thresholded_rows = []
    for row in rows:
        new_row = dict(row)
        pred_index = row["argmax_index"]
        if row[score_key] > threshold:
            pred_index = UNKNOWN_PRED_INDEX
        new_row["pred_index"] = pred_index
        new_row["pred_label"] = PRED_LABELS[pred_index] if pred_index < len(PRED_LABELS) else UNKNOWN_PRED_NAME
        new_row["applied_threshold"] = float(threshold)
        thresholded_rows.append(new_row)
    return thresholded_rows


def build_threshold_candidates(rows, score_key: str):
    scores = sorted({row[score_key] for row in rows})
    eps = 1e-6
    return [scores[0] - eps, *scores, scores[-1] + eps]


def find_best_threshold(rows, score_key: str):
    best_threshold = None
    best_metrics = None
    for threshold in build_threshold_candidates(rows, score_key):
        eval_rows = apply_threshold(rows, float(threshold), score_key)
        metrics = compute_metrics(eval_rows)
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
    return best_threshold, best_metrics


def summarize_results(rows, alpha: float, threshold: float):
    print("alpha (patch weight):", f"{alpha:.2f}")
    print("threshold:", f"{threshold:.6f}")
    metrics = compute_metrics(rows)
    print(
        "test metrics:",
        {
            "open_accuracy": round(metrics["open_accuracy"], 4),
            "open_macro_f1": round(metrics["open_macro_f1"], 4),
            "known_accuracy": round(metrics["known_accuracy"], 4),
            "unknown_accuracy": round(metrics["unknown_accuracy"], 4),
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Ensemble global feature distance and layer2 patch z-score.")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root or find_repo_root(Path.cwd().resolve())
    data_root = repo_root / "prepared_dataset_G5_622"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_samples, train_class_counts = collect_split_samples(data_root, "train", known_only=True)
    val_samples, val_class_counts = collect_split_samples(data_root, "val", known_only=False)
    test_samples, test_class_counts = collect_split_samples(data_root, "test", known_only=False)

    print("repo root:", repo_root)
    print("device:", device)
    print("train class counts:", dict(train_class_counts))
    print("val class counts:", dict(val_class_counts))
    print("test class counts:", dict(test_class_counts))

    model, checkpoint, image_size = load_model(args.model_path, device)
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

    patch_prototypes, global_prototypes = build_prototypes(model, train_loader, device)
    train_rows = run_inference(model, train_loader, patch_prototypes, global_prototypes, device, args.topk)
    val_rows = run_inference(model, val_loader, patch_prototypes, global_prototypes, device, args.topk)
    test_rows = run_inference(model, test_loader, patch_prototypes, global_prototypes, device, args.topk)

    patch_stats = compute_pred_class_stats(train_rows, "patch_score")
    global_stats = compute_pred_class_stats(train_rows, "global_score")
    val_rows = apply_normalization(val_rows, "patch_score", "patch_zscore", patch_stats)
    val_rows = apply_normalization(val_rows, "global_score", "global_zscore", global_stats)
    test_rows = apply_normalization(test_rows, "patch_score", "patch_zscore", patch_stats)
    test_rows = apply_normalization(test_rows, "global_score", "global_zscore", global_stats)

    alphas = np.arange(0.0, 1.0 + 1e-9, args.alpha_step)
    best = None
    summary = []

    for alpha in alphas:
        alpha = float(round(alpha, 10))
        val_combo = []
        test_combo = []
        for src_rows, out_rows in ((val_rows, val_combo), (test_rows, test_combo)):
            for row in src_rows:
                new_row = dict(row)
                new_row["ensemble_score"] = alpha * row["patch_zscore"] + (1.0 - alpha) * row["global_zscore"]
                out_rows.append(new_row)

        threshold, val_metrics = find_best_threshold(val_combo, "ensemble_score")
        test_eval_rows = apply_threshold(test_combo, threshold, "ensemble_score")
        test_metrics = compute_metrics(test_eval_rows)
        result = {
            "alpha_patch": alpha,
            "alpha_global": 1.0 - alpha,
            "threshold": threshold,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }
        summary.append(result)

        if best is None:
            best = (result, test_eval_rows)
            continue
        best_result = best[0]
        if val_metrics["open_macro_f1"] > best_result["val_metrics"]["open_macro_f1"]:
            best = (result, test_eval_rows)
            continue
        if (
            val_metrics["open_macro_f1"] == best_result["val_metrics"]["open_macro_f1"]
            and val_metrics["open_accuracy"] > best_result["val_metrics"]["open_accuracy"]
        ):
            best = (result, test_eval_rows)
            continue

    best_result, best_test_rows = best
    print("best ensemble:")
    print(best_result)
    summarize_results(best_test_rows, best_result["alpha_patch"], best_result["threshold"])

    output_target = args.output_json if args.output_json is not None else args.model_path.parent
    if args.output_json is not None and output_target.suffix:
        summary_path = output_target
        results_path = output_target.with_name(f"{output_target.stem}_best_rows{output_target.suffix}")
    else:
        output_dir = output_target
        summary_path = output_dir / f"{args.model_path.stem}_ensemble_global_patch_topk{args.topk}_summary.json"
        results_path = output_dir / f"{args.model_path.stem}_ensemble_global_patch_topk{args.topk}_best_rows.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    results_path.write_text(json.dumps(best_test_rows, indent=2))
    print("saved summary:", summary_path)
    print("saved best rows:", results_path)


if __name__ == "__main__":
    main()
