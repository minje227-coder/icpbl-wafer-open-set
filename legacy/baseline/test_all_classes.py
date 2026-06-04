import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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


class WaferTestDataset(Dataset):
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


def collect_test_samples(data_root: Path):
    split_root = data_root / "test" / "G5"
    samples = []
    class_counts = Counter()

    for class_name in ALL_TRUE_CLASSES:
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


@torch.no_grad()
def run_inference(model, loader, device, threshold=None):
    rows = []

    for images, true_classes, paths in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        pred_indices = probs.argmax(axis=1)
        max_probs = probs.max(axis=1)
        if threshold is not None:
            pred_indices = np.where(max_probs >= threshold, pred_indices, UNKNOWN_PRED_INDEX)

        for true_class, path, prob_vec, pred_index, max_prob in zip(true_classes, paths, probs, pred_indices, max_probs):
            row = {
                "path": path,
                "true_class": true_class,
                "pred_index": int(pred_index),
                "pred_label": PRED_LABELS[pred_index] if pred_index < len(PRED_LABELS) else UNKNOWN_PRED_NAME,
                "confidence": float(max_prob),
                "prob_DIE_BROKEN": float(prob_vec[0]),
                "prob_NORMAL": float(prob_vec[1]),
                "prob_NO_DIE": float(prob_vec[2]),
            }
            rows.append(row)

    return rows


def summarize_results(rows, threshold):
    per_true_class = defaultdict(list)
    for row in rows:
        per_true_class[row["true_class"]].append(row)

    print(f"num samples: {len(rows)}")
    if threshold is None:
        print("threshold: disabled")
    else:
        print(f"threshold: {threshold:.4f}")
    print()

    known_true = []
    known_pred = []
    for row in rows:
        if row["true_class"] in KNOWN_LABELS and row["pred_index"] < len(PRED_LABELS):
            known_true.append(KNOWN_LABELS[row["true_class"]])
            known_pred.append(row["pred_index"])

    if known_true:
        print("Known-class test accuracy:", f"{accuracy_score(known_true, known_pred):.4f}")
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

    pred_names = PRED_LABELS + ([UNKNOWN_PRED_NAME] if threshold is not None else [])
    matrix_labels = list(range(len(pred_names)))
    true_indices = [ALL_TRUE_CLASSES.index(row["true_class"]) for row in rows]
    pred_indices = [row["pred_index"] for row in rows]
    matrix = confusion_matrix(true_indices, pred_indices, labels=matrix_labels)

    print("Confusion matrix: true 5 classes x predicted labels")
    print("true classes order:", ALL_TRUE_CLASSES)
    print("pred labels order:", pred_names)
    print(matrix)
    print()

    print("Per-class prediction summary:")
    for class_name in ALL_TRUE_CLASSES:
        items = per_true_class.get(class_name, [])
        if not items:
            continue
        pred_counter = Counter(item["pred_label"] for item in items)
        mean_conf = np.mean([item["confidence"] for item in items])
        print(f"- {class_name}: {len(items)} samples, mean confidence={mean_conf:.4f}, pred distribution={dict(pred_counter)}")


def save_results(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2))
    print()
    print("Saved per-image results to", output_path)


def resolve_output_path(model_path: Path, output_json: Path | None, multiple_models: bool):
    auto_name = f"{model_path.stem}_test_results.json"

    if output_json is None:
        return model_path.with_name(auto_name)

    if output_json.suffix == "":
        return output_json / auto_name

    if multiple_models:
        return output_json.with_name(f"{output_json.stem}_{model_path.stem}{output_json.suffix}")

    return output_json


def main():
    parser = argparse.ArgumentParser(description="Evaluate the 3-class baseline model on all 5 wafer test classes.")
    parser.add_argument("--repo-root", type=Path, default=None, help="Repository root containing prepared_dataset_G5_622.")
    parser.add_argument("--model-path", type=Path, default=None, help="Path to model.pth checkpoint.")
    parser.add_argument("--ckpt-dir", type=Path, default=None, help="Directory containing one or more .pth checkpoints.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for test inference.")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of dataloader workers.")
    parser.add_argument("--threshold", type=float, default=None, help="Optional confidence threshold for UNKNOWN rejection.")
    parser.add_argument(
        "--use-checkpoint-threshold",
        action="store_true",
        help="Use best_threshold stored in the checkpoint if available.",
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

    samples, class_counts = collect_test_samples(data_root)

    print("repo root:", repo_root)
    if args.ckpt_dir is not None:
        print("checkpoint dir:", args.ckpt_dir)
    else:
        print("model path:", model_paths[0])
    print("device:", device)
    print("test class counts:", dict(class_counts))
    print()

    for idx, model_path in enumerate(model_paths, start=1):
        model, checkpoint, image_size = load_model(model_path, device)

        threshold = args.threshold
        if args.use_checkpoint_threshold and threshold is None:
            threshold = checkpoint.get("best_threshold")

        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        dataset = WaferTestDataset(samples, transform=transform)
        loader = DataLoader(
            dataset,
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
        print()

        rows = run_inference(model, loader, device, threshold=threshold)
        summarize_results(rows, threshold=threshold)

        output_path = resolve_output_path(
            model_path=model_path,
            output_json=args.output_json,
            multiple_models=len(model_paths) > 1,
        )
        save_results(rows, output_path)
        print()


if __name__ == "__main__":
    main()
