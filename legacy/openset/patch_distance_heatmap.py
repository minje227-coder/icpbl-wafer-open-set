import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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


def tensor_to_image(tensor: torch.Tensor):
    image = tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = image * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(image, 0.0, 1.0)


def make_overlay(base_image: np.ndarray, heatmap: np.ndarray, alpha: float):
    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(heatmap)[..., :3]
    return np.clip((1.0 - alpha) * base_image + alpha * heat_rgb, 0.0, 1.0)


def nearest_resize(array: np.ndarray, size: int):
    return np.array(
        Image.fromarray((array * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
    ).astype(np.float32) / 255.0


def bilinear_resize(array: np.ndarray, size: int):
    return np.array(
        Image.fromarray((array * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    ).astype(np.float32) / 255.0


def make_topk_mask(distance_map: np.ndarray, topk: int):
    flat = distance_map.reshape(-1)
    k = min(topk, flat.size)
    top_indices = np.argpartition(flat, -k)[-k:]
    mask = np.zeros_like(flat, dtype=np.float32)
    mask[top_indices] = 1.0
    coords = [(int(idx // distance_map.shape[1]), int(idx % distance_map.shape[1])) for idx in top_indices]
    return mask.reshape(distance_map.shape), sorted(coords)


def make_mask_overlay(base_image: np.ndarray, mask: np.ndarray, alpha: float):
    overlay = base_image.copy()
    color = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    overlay[mask > 0.5] = np.clip((1.0 - alpha) * overlay[mask > 0.5] + alpha * color, 0.0, 1.0)
    return overlay


def plot_layer_heatmaps(base_image, layer_entries, output_path: Path, title: str, shared_vmax: float):
    fig, axes = plt.subplots(2, len(layer_entries) + 1, figsize=(5 * (len(layer_entries) + 1), 10))
    axes[0, 0].imshow(base_image)
    axes[0, 0].set_title("original")
    axes[0, 0].axis("off")
    axes[1, 0].imshow(base_image)
    axes[1, 0].set_title("original")
    axes[1, 0].axis("off")

    for col, entry in enumerate(layer_entries, start=1):
        axes[0, col].imshow(entry["shared_overlay"])
        axes[0, col].set_title(
            f"{entry['layer']} shared-scale\n"
            f"pred={entry['pred_label']} ({entry['confidence']:.3f})\n"
            f"score={entry['topk_score']:.3f}"
        )
        axes[0, col].axis("off")

        axes[1, col].imshow(entry["topk_overlay"])
        axes[1, col].set_title(
            f"{entry['layer']} top-k mask\n"
            f"k={entry['topk']} | cells={entry['distance_map_shape'][0]}x{entry['distance_map_shape'][1]}"
        )
        axes[1, col].axis("off")

    fig.suptitle(f"{title}\nshared distance scale: [0, {shared_vmax:.4f}]")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_single_loader(patch, image_path: Path, true_class: str, transform):
    samples = [(image_path, true_class)]
    dataset = patch.WaferSplitDataset(samples, transform=transform)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)


def main():
    parser = argparse.ArgumentParser(description="Create patch-distance heatmap overlays for layer2/3/4.")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--true-class", type=str, required=True)
    parser.add_argument("--layers", nargs="+", default=["layer2", "layer3", "layer4"])
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root or find_repo_root(Path.cwd().resolve())
    patch = load_patch_module(repo_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_samples, _ = patch.collect_split_samples(repo_root / "prepared_dataset_G5_622", "train", known_only=True)
    model, checkpoint, image_size = patch.load_model(args.model_path, device)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
        ]
    )

    train_loader = DataLoader(
        patch.WaferSplitDataset(train_samples, transform=transform),
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    original_image = Image.open(args.image_path).convert("RGB").resize((image_size, image_size))
    base_image = np.asarray(original_image).astype(np.float32) / 255.0

    manifest = {
        "model_path": str(args.model_path),
        "image_path": str(args.image_path),
        "true_class": args.true_class,
        "topk": int(args.topk),
        "layers": [],
    }

    layer_entries = []
    global_distance_max = 0.0
    for layer_name in args.layers:
        prototype_maps = patch.build_class_prototype_maps(model, train_loader, device, layer_name)
        single_loader = build_single_loader(patch, args.image_path, args.true_class, transform)
        score_label = f"topk{int(args.topk)}"
        row = patch.run_inference(
            model,
            single_loader,
            prototype_maps,
            device,
            patch.build_score_specs([args.topk], []),
            layer_name,
        )[score_label][0]

        image_tensor = next(iter(single_loader))[0].to(device)
        feature_map, logits = patch.forward_feature_map_and_logits(model, image_tensor, layer_name)
        feature_map = patch.normalize_patch_map(feature_map)[0]
        pred_index = int(torch.softmax(logits, dim=1).argmax(dim=1).item())
        proto = prototype_maps[pred_index]
        full_distance_map = patch.patch_distance_map(feature_map, proto).detach().cpu().numpy()
        global_distance_max = max(global_distance_max, float(full_distance_map.max()))
        topk_mask, topk_coords = make_topk_mask(full_distance_map, args.topk)

        layer_entries.append(
            {
                "layer": layer_name,
                "pred_label": row["argmax_label"],
                "confidence": row["confidence"],
                "topk_score": row["anomaly_score"],
                "distance_map": full_distance_map,
                "topk_mask": topk_mask,
                "topk_coords": topk_coords,
                "distance_map_shape": list(full_distance_map.shape),
                "topk": int(args.topk),
            }
        )
        manifest["layers"].append(
            {
                "layer": layer_name,
                "pred_label": row["argmax_label"],
                "confidence": row["confidence"],
                "topk_score": row["anomaly_score"],
                "distance_map_shape": list(full_distance_map.shape),
                "distance_map_mean": float(full_distance_map.mean()),
                "distance_map_max": float(full_distance_map.max()),
                "distance_map_min": float(full_distance_map.min()),
                "topk_coords": topk_coords,
            }
        )

    shared_vmax = global_distance_max if global_distance_max > 0 else 1.0
    for entry in layer_entries:
        normalized_map = np.clip(entry["distance_map"] / shared_vmax, 0.0, 1.0)
        resized_map = bilinear_resize(normalized_map, image_size)
        resized_mask = nearest_resize(entry["topk_mask"], image_size)
        entry["shared_overlay"] = make_overlay(base_image, resized_map, args.alpha)
        entry["topk_overlay"] = make_mask_overlay(base_image, resized_mask, args.alpha)

    manifest["shared_distance_scale"] = {
        "min": 0.0,
        "max": float(shared_vmax),
    }

    stem = f"{args.model_path.stem}_{args.image_path.stem}_patch_distance_layers234"
    output_path = args.output_dir / f"{stem}.png"
    title = f"patch distance heatmap | true={args.true_class} | image={args.image_path.name}"
    plot_layer_heatmaps(base_image, layer_entries, output_path, title, shared_vmax)

    manifest_path = args.output_dir / f"{stem}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(output_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
