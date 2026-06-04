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
UNKNOWN_CLASSES = {"DIE_CRACK", "DIE_INK"}
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


def forward_features(model, images):
    x = model.conv1(images)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return x


@torch.no_grad()
def build_class_prototypes(model, loader, device):
    feature_buckets = defaultdict(list)

    for images, true_classes, _ in loader:
        images = images.to(device, non_blocking=True)
        features = forward_features(model, images)
        features = F.normalize(features, dim=1)

        for feature, true_class in zip(features, true_classes):
            class_idx = KNOWN_LABELS[true_class]
            feature_buckets[class_idx].append(feature.detach().cpu())

    missing = [PRED_LABELS[class_idx] for class_idx in range(len(PRED_LABELS)) if class_idx not in feature_buckets]
    if missing:
        raise ValueError(f"Missing train features for classes: {missing}")

    prototypes = []
    for class_idx in range(len(PRED_LABELS)):
        class_features = torch.stack(feature_buckets[class_idx], dim=0)
        prototype = class_features.mean(dim=0)
        prototype = F.normalize(prototype, dim=0)
        prototypes.append(prototype)

    return torch.stack(prototypes, dim=0).to(device)


@torch.no_grad()
def run_inference(model, loader, prototypes, device):
    rows = []

    for images, true_classes, paths in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        pred_indices = probs.argmax(dim=1)
        confidences = probs.max(dim=1).values

        features = forward_features(model, images)
        features = F.normalize(features, dim=1)
        cosine_distances = 1.0 - torch.matmul(features, prototypes.T)

        for true_class, path, prob_vec, pred_index, confidence, dist_vec in zip(
            true_classes,
            paths,
            probs.cpu(),
            pred_indices.cpu(),
            confidences.cpu(),
            cosine_distances.cpu(),
        ):
            pred_index = int(pred_index)
            row = {
                "path": path,
                "true_class": true_class,
                "true_index": true_class_to_index(true_class),
                "argmax_index": pred_index,
                "argmax_label": PRED_LABELS[pred_index],
                "confidence": float(confidence),
                "pred_distance": float(dist_vec[pred_index]),
                "distance_to_DIE_BROKEN": float(dist_vec[0]),
                "distance_to_NORMAL": float(dist_vec[1]),
                "distance_to_NO_DIE": float(dist_vec[2]),
                "prob_DIE_BROKEN": float(prob_vec[0]),
                "prob_NORMAL": float(prob_vec[1]),
                "prob_NO_DIE": float(prob_vec[2]),
            }
            rows.append(row)

    return rows


def apply_distance_threshold(rows, threshold: float | None):
    thresholded_rows = []
    for row in rows:
        new_row = dict(row)
        pred_index = row["argmax_index"]
        if threshold is not None and row["pred_distance"] > threshold:
            pred_index = UNKNOWN_PRED_INDEX
        new_row["pred_index"] = pred_index
        new_row["pred_label"] = PRED_LABELS[pred_index] if pred_index < len(PRED_LABELS) else UNKNOWN_PRED_NAME
        new_row["applied_threshold"] = None if threshold is None else float(threshold)
        thresholded_rows.append(new_row)
    return thresholded_rows


def compute_metrics(rows):
    true_indices = [row["true_index"] for row in rows]
    pred_indices = [row["pred_index"] for row in rows]

    known_rows = [row for row in rows if row["true_class"] in KNOWN_LABELS]
    known_true = [KNOWN_LABELS[row["true_class"]] for row in known_rows]
    known_pred = [row["pred_index"] for row in known_rows]

    metrics = {
        "open_accuracy": accuracy_score(true_indices, pred_indices),
        "open_macro_f1": f1_score(true_indices, pred_indices, labels=[0, 1, 2, 3], average="macro", zero_division=0),
        "known_accuracy": accuracy_score(known_true, known_pred) if known_true else None,
    }
    return metrics


def find_best_threshold(rows, thresholds):
    best_threshold = None
    best_metrics = None

    for threshold in thresholds:
        thresholded_rows = apply_distance_threshold(rows, float(threshold))
        metrics = compute_metrics(thresholded_rows)

        if best_metrics is None:
            best_threshold = float(threshold)
            best_metrics = metrics
            continue

        if metrics["open_macro_f1"] > best_metrics["open_macro_f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics
            continue

        if (
            metrics["open_macro_f1"] == best_metrics["open_macro_f1"]
            and metrics["open_accuracy"] > best_metrics["open_accuracy"]
        ):
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


def build_confusion_matrix(rows):
    matrix = np.zeros((len(ALL_TRUE_CLASSES), len(PRED_LABELS) + 1), dtype=int)
    for row in rows:
        true_idx = ALL_TRUE_CLASSES.index(row["true_class"])
        pred_idx = row["pred_index"]
        matrix[true_idx, pred_idx] += 1
    return matrix


def summarize_results(rows, threshold):
    per_true_class = defaultdict(list)
    for row in rows:
        per_true_class[row["true_class"]].append(row)

    print(f"num samples: {len(rows)}")
    if threshold is None:
        print("feature-distance threshold: disabled")
    else:
        print(f"feature-distance threshold: {threshold:.4f}")
    print()

    known_rows = [row for row in rows if row["true_class"] in KNOWN_LABELS]
    known_true = [KNOWN_LABELS[row["true_class"]] for row in known_rows]
    known_pred = [row["pred_index"] for row in known_rows]

    print("Known-class accuracy:", f"{accuracy_score(known_true, known_pred):.4f}")
    print()
    print("Known-class classification report:")
    print(
        classification_report(
            known_true,
            known_pred,
            labels=[0, 1, 2],
            target_names=PRED_LABELS,
            zero_division=0,
        )
    )

    true_indices = [row["true_index"] for row in rows]
    pred_indices = [row["pred_index"] for row in rows]
    print("Open-set accuracy:", f"{accuracy_score(true_indices, pred_indices):.4f}")
    print(
        "Open-set macro F1:",
        f"{f1_score(true_indices, pred_indices, labels=[0, 1, 2, 3], average='macro', zero_division=0):.4f}",
    )
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
        mean_dist = np.mean([item["pred_distance"] for item in items])
        print(
            f"- {class_name}: {len(items)} samples, "
            f"mean confidence={mean_conf:.4f}, mean pred-distance={mean_dist:.4f}, "
            f"pred distribution={dict(pred_counter)}"
        )


def save_results(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2))
    print()
    print("Saved per-image results to", output_path)


def resolve_output_path(model_path: Path, output_json: Path | None, multiple_models: bool):
    auto_name = f"{model_path.stem}_feature_distance_test_results.json"

    if output_json is None:
        return model_path.with_name(auto_name)

    if output_json.suffix == "":
        return output_json / auto_name

    if multiple_models:
        return output_json.with_name(f"{output_json.stem}_{model_path.stem}{output_json.suffix}")

    return output_json


def main():
    parser = argparse.ArgumentParser(
        description="Use softmax prediction as the class label and reject unknowns by cosine distance to the predicted-class prototype."
    )
    parser.add_argument("--repo-root", type=Path, default=None, help="Repository root containing prepared_dataset_G5_622.")
    parser.add_argument("--model-path", type=Path, default=None, help="Path to a single checkpoint.")
    parser.add_argument("--ckpt-dir", type=Path, default=None, help="Directory containing one or more .pth checkpoints.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference.")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of dataloader workers.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional fixed distance threshold. If omitted, the best threshold is searched on val and then applied to test.",
    )
    parser.add_argument("--threshold-min", type=float, default=0.0, help="Minimum distance threshold used in the validation search.")
    parser.add_argument("--threshold-max", type=float, default=1.0, help="Maximum distance threshold used in the validation search.")
    parser.add_argument("--threshold-steps", type=int, default=10000, help="Number of evenly spaced thresholds used in the validation search.")
    parser.add_argument(
        "--disable-thresholding",
        action="store_true",
        help="Disable unknown rejection and evaluate plain softmax argmax predictions.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON file path or directory. If omitted, save automatically next to each checkpoint.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root or find_repo_root(Path.cwd().resolve())
    data_root = repo_root / "prepared_dataset_G5_622"
    model_paths = resolve_model_paths(args.model_path, args.ckpt_dir, repo_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_samples, train_class_counts = collect_split_samples(data_root, "train", known_only=True)
    val_samples, val_class_counts = collect_split_samples(data_root, "val", known_only=False)
    test_samples, test_class_counts = collect_split_samples(data_root, "test", known_only=False)

    print("repo root:", repo_root)
    if args.ckpt_dir is not None:
        print("checkpoint dir:", args.ckpt_dir)
    else:
        print("model path:", model_paths[0])
    print("device:", device)
    print("train class counts:", dict(train_class_counts))
    print("val class counts:", dict(val_class_counts))
    print("test class counts:", dict(test_class_counts))
    print()

    threshold_candidates = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps)

    for idx, model_path in enumerate(model_paths, start=1):
        model, checkpoint, image_size = load_model(model_path, device)
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        train_loader = DataLoader(
            WaferSplitDataset(train_samples, transform=transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            WaferSplitDataset(val_samples, transform=transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            WaferSplitDataset(test_samples, transform=transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

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

        prototypes = build_class_prototypes(model, train_loader, device)
        val_rows = run_inference(model, val_loader, prototypes, device)
        test_rows = run_inference(model, test_loader, prototypes, device)

        if args.disable_thresholding:
            threshold = None
            val_eval_rows = apply_distance_threshold(val_rows, threshold=None)
            test_eval_rows = apply_distance_threshold(test_rows, threshold=None)
            print("distance thresholding: disabled")
        elif args.threshold is not None:
            threshold = float(args.threshold)
            val_eval_rows = apply_distance_threshold(val_rows, threshold=threshold)
            test_eval_rows = apply_distance_threshold(test_rows, threshold=threshold)
            print(f"using fixed distance threshold: {threshold:.4f}")
        else:
            threshold, val_threshold_metrics = find_best_threshold(val_rows, threshold_candidates)
            val_eval_rows = apply_distance_threshold(val_rows, threshold=threshold)
            test_eval_rows = apply_distance_threshold(test_rows, threshold=threshold)
            print(f"best validation distance threshold: {threshold:.4f}")
            print(
                "validation metrics after thresholding:",
                {
                    "open_accuracy": round(val_threshold_metrics["open_accuracy"], 4),
                    "open_macro_f1": round(val_threshold_metrics["open_macro_f1"], 4),
                    "known_accuracy": None if val_threshold_metrics["known_accuracy"] is None else round(val_threshold_metrics["known_accuracy"], 4),
                },
            )

        print()
        print("Validation summary:")
        summarize_results(val_eval_rows, threshold=threshold)
        print()
        print("Test summary:")
        summarize_results(test_eval_rows, threshold=threshold)

        output_path = resolve_output_path(
            model_path=model_path,
            output_json=args.output_json,
            multiple_models=len(model_paths) > 1,
        )
        save_results(test_eval_rows, output_path)
        print()


if __name__ == "__main__":
    main()
