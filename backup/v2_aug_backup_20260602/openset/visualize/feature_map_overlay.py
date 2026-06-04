import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


KNOWN_LABELS = {
    "DIE_BROKEN": 0,
    "NORMAL": 1,
    "NO_DIE": 2,
}

LAYER_NAMES = ("layer1", "layer2", "layer3", "layer4")
POSITION_NAMES = ("top_left", "top_right", "bottom_left", "bottom_right", "center")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "prepared_dataset_G5_622").exists():
            return candidate
    raise FileNotFoundError("Could not find prepared_dataset_G5_622 from the current working directory.")


def load_model(model_path: Path, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    image_size = checkpoint.get("image_size", 224)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(KNOWN_LABELS))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint, image_size


def load_image(image_path: Path, image_size: int):
    image = Image.open(image_path).convert("RGB")
    original = image.resize((image_size, image_size))
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN.view(3).tolist(), std=IMAGENET_STD.view(3).tolist()),
        ]
    )
    image_tensor = transform(image).unsqueeze(0)
    return original, image_tensor


def forward_to_layer(model, image_tensor: torch.Tensor, layer_name: str):
    x = model.conv1(image_tensor)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    if layer_name == "layer1":
        return x
    x = model.layer2(x)
    if layer_name == "layer2":
        return x
    x = model.layer3(x)
    if layer_name == "layer3":
        return x
    x = model.layer4(x)
    return x


def get_position_indices(height: int, width: int):
    return {
        "top_left": (0, 0),
        "top_right": (0, width - 1),
        "bottom_left": (height - 1, 0),
        "bottom_right": (height - 1, width - 1),
        "center": (height // 2, width // 2),
    }


def tensor_to_numpy_image(image_tensor: torch.Tensor):
    restored = image_tensor.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    restored = restored.clamp(0.0, 1.0)
    return restored.squeeze(0).permute(1, 2, 0).numpy()


def make_overlay(base_image: np.ndarray, heatmap: np.ndarray, alpha: float):
    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(heatmap)[..., :3]
    return np.clip((1.0 - alpha) * base_image + alpha * heat_rgb, 0.0, 1.0)


def compute_saliency_for_position(model, image_tensor, layer_name, row, col, channel=None):
    input_tensor = image_tensor.clone().detach().requires_grad_(True)
    feature_map = forward_to_layer(model, input_tensor, layer_name)

    if channel is None:
        target = feature_map[0, :, row, col].mean()
        channel_desc = "mean_channels"
    else:
        target = feature_map[0, channel, row, col]
        channel_desc = f"channel_{channel:03d}"

    model.zero_grad(set_to_none=True)
    if input_tensor.grad is not None:
        input_tensor.grad.zero_()
    target.backward()

    saliency = input_tensor.grad.detach().abs().amax(dim=1).squeeze(0).cpu().numpy()
    saliency -= saliency.min()
    if saliency.max() > 0:
        saliency /= saliency.max()

    return saliency, float(target.detach().cpu()), channel_desc, feature_map.shape


def plot_overlays(base_image, overlays, output_path: Path, title: str):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()

    axes[0].imshow(base_image)
    axes[0].set_title("original")
    axes[0].axis("off")

    for axis, item in zip(axes[1:], overlays):
        axis.imshow(item["overlay"])
        axis.set_title(
            f"{item['name']}\n"
            f"fmap[{item['row']},{item['col']}], score={item['score']:.4f}"
        )
        axis.axis("off")

    for axis in axes[len(overlays) + 1 :]:
        axis.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_metadata(metadata: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2))


def resolve_output_dir(repo_root: Path, output_dir: Path | None):
    if output_dir is not None:
        return output_dir
    return repo_root / "openset" / "visualize"


def main():
    parser = argparse.ArgumentParser(
        description="Create input-gradient heatmap overlays for selected spatial positions in a ResNet18 feature map."
    )
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--layers", type=str, nargs="+", default=["layer2", "layer3", "layer4"])
    parser.add_argument("--channel", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root or find_repo_root(Path.cwd().resolve())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    invalid_layers = [layer for layer in args.layers if layer not in LAYER_NAMES]
    if invalid_layers:
        raise ValueError(f"Invalid layers: {invalid_layers}. Choices: {LAYER_NAMES}")

    model, checkpoint, image_size = load_model(args.model_path, device)
    original, image_tensor = load_image(args.image_path, image_size)
    image_tensor = image_tensor.to(device)
    base_image = tensor_to_numpy_image(image_tensor)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_index = int(np.argmax(probs))
        pred_label = list(KNOWN_LABELS.keys())[pred_index]

    output_dir = resolve_output_dir(repo_root, args.output_dir)
    manifest = {
        "repo_root": str(repo_root),
        "model_path": str(args.model_path),
        "image_path": str(args.image_path),
        "device": str(device),
        "predicted_class": pred_label,
        "predicted_confidence": float(probs[pred_index]),
        "channel": "mean_channels" if args.channel is None else args.channel,
        "image_size": image_size,
        "layers": [],
    }

    for layer_name in args.layers:
        feature_map = forward_to_layer(model, image_tensor, layer_name)
        _, channels, height, width = feature_map.shape
        positions = get_position_indices(height, width)

        overlays = []
        feature_shape = tuple(feature_map.shape)
        channel_desc = "mean_channels"
        layer_metadata = {
            "layer": layer_name,
            "feature_shape": feature_shape,
            "positions": [],
        }

        for name in POSITION_NAMES:
            row, col = positions[name]
            saliency, score, channel_desc, feature_shape = compute_saliency_for_position(
                model=model,
                image_tensor=image_tensor,
                layer_name=layer_name,
                row=row,
                col=col,
                channel=args.channel,
            )
            overlays.append(
                {
                    "name": name,
                    "row": row,
                    "col": col,
                    "score": score,
                    "overlay": make_overlay(base_image, saliency, args.alpha),
                }
            )
            layer_metadata["positions"].append(
                {
                    "name": name,
                    "feature_row": row,
                    "feature_col": col,
                    "score": score,
                    "channels": channels,
                }
            )

        output_name = f"{args.model_path.stem}_{args.image_path.stem}_{layer_name}_{channel_desc}_overlay.png"
        output_path = output_dir / output_name
        title = (
            f"model={args.model_path.name} | image={args.image_path.name} | layer={layer_name} "
            f"| feature_shape={feature_shape} | pred={pred_label} ({probs[pred_index]:.4f})"
        )
        plot_overlays(np.asarray(original) / 255.0, overlays, output_path, title)
        layer_metadata["output_path"] = str(output_path)
        manifest["layers"].append(layer_metadata)

    print("repo root:", repo_root)
    print("model path:", args.model_path)
    print("image path:", args.image_path)
    print("device:", device)
    print("layers:", args.layers)
    print("channel:", "mean_channels" if args.channel is None else args.channel)
    print("predicted class:", pred_label)
    print("predicted confidence:", f"{probs[pred_index]:.4f}")
    manifest_path = output_dir / f"{args.model_path.stem}_{args.image_path.stem}_overlay_manifest.json"
    save_metadata(manifest, manifest_path)
    print("saved output dir:", output_dir)
    for layer_info in manifest["layers"]:
        print("saved overlay:", layer_info["output_path"])
    print("saved manifest:", manifest_path)


if __name__ == "__main__":
    main()
