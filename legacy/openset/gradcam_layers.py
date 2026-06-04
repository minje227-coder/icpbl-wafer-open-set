import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


KNOWN_LABELS = {
    "DIE_BROKEN": 0,
    "NORMAL": 1,
    "NO_DIE": 2,
}
LAYER_NAMES = ("layer2", "layer3", "layer4")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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
    resized = image.resize((image_size, image_size))
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
        ]
    )
    tensor = transform(image).unsqueeze(0)
    return resized, tensor


def make_overlay(base_image: np.ndarray, heatmap: np.ndarray, alpha: float):
    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(heatmap)[..., :3]
    return np.clip((1.0 - alpha) * base_image + alpha * heat_rgb, 0.0, 1.0)


def get_target_layer(model, layer_name: str):
    return getattr(model, layer_name)


def compute_gradcam(model, image_tensor: torch.Tensor, layer_name: str, target_index: int):
    activations = {}
    gradients = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output.detach()

    def backward_hook(_module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    target_layer = get_target_layer(model, layer_name)
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(image_tensor)
        score = logits[:, target_index].sum()
        score.backward()
    finally:
        forward_handle.remove()
        backward_handle.remove()

    acts = activations["value"][0]
    grads = gradients["value"][0]
    weights = grads.mean(dim=(1, 2), keepdim=True)
    cam = (weights * acts).sum(dim=0)
    cam = F.relu(cam)
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.cpu().numpy(), logits.detach()


def plot_gradcams(base_image, overlays, output_path: Path, title: str):
    fig, axes = plt.subplots(1, len(overlays) + 1, figsize=(5 * (len(overlays) + 1), 5))
    axes[0].imshow(base_image)
    axes[0].set_title("original")
    axes[0].axis("off")

    for axis, item in zip(axes[1:], overlays):
        axis.imshow(item["overlay"])
        axis.set_title(
            f"{item['layer']}\n"
            f"pred={item['pred_label']} ({item['confidence']:.3f})"
        )
        axis.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create Grad-CAM overlays for layer2/3/4.")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--true-class", type=str, required=True)
    parser.add_argument("--layers", nargs="+", default=["layer2", "layer3", "layer4"])
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root or find_repo_root(Path.cwd().resolve())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    invalid_layers = [layer for layer in args.layers if layer not in LAYER_NAMES]
    if invalid_layers:
        raise ValueError(f"Invalid layers: {invalid_layers}. Choices: {LAYER_NAMES}")

    model, _, image_size = load_model(args.model_path, device)
    original, image_tensor = load_image(args.image_path, image_size)
    image_tensor = image_tensor.to(device)
    base_image = np.asarray(original).astype(np.float32) / 255.0

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_index = int(np.argmax(probs))
    pred_label = list(KNOWN_LABELS.keys())[pred_index]

    overlays = []
    manifest = {
        "model_path": str(args.model_path),
        "image_path": str(args.image_path),
        "true_class": args.true_class,
        "pred_label": pred_label,
        "pred_confidence": float(probs[pred_index]),
        "layers": [],
    }

    for layer_name in args.layers:
        cam, _ = compute_gradcam(model, image_tensor.clone(), layer_name, pred_index)
        resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((image_size, image_size), Image.BILINEAR)).astype(np.float32) / 255.0
        overlays.append(
            {
                "layer": layer_name,
                "pred_label": pred_label,
                "confidence": float(probs[pred_index]),
                "overlay": make_overlay(base_image, resized, args.alpha),
            }
        )
        manifest["layers"].append(
            {
                "layer": layer_name,
                "pred_label": pred_label,
                "pred_confidence": float(probs[pred_index]),
                "cam_min": float(cam.min()),
                "cam_max": float(cam.max()),
                "cam_mean": float(cam.mean()),
            }
        )

    stem = f"{args.model_path.stem}_{args.image_path.stem}_gradcam_layers234"
    output_path = args.output_dir / f"{stem}.png"
    title = f"Grad-CAM | true={args.true_class} | image={args.image_path.name}"
    plot_gradcams(base_image, overlays, output_path, title)

    manifest_path = args.output_dir / f"{stem}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(output_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
