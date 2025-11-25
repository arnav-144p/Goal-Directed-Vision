"""
Run Side-by-Side Comparison: PASSIVE CLIP vs ACTIVE IRL Vision
==============================================================

This script compares:
  - PASSIVE CLIP pipeline (saliency + scanpath + bounding box)
  - ACTIVE IRL pipeline (sequential fixations + patch CLIP scores + belief map)

Usage:
    python run_comparison.py --image demo_inputs/dog.jpg --task "biscuit"

The output visualization is saved to `comparison_output.png` by default.
"""

import argparse
import os
from typing import Tuple, List, Dict, Any

import numpy as np
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

try:
    import open_clip
    HAS_OPEN_CLIP = True
except ImportError:  # pragma: no cover - runtime warning only
    HAS_OPEN_CLIP = False
    print("[WARNING] open_clip not found. Install with: pip install open-clip-torch")

from interactive_search import (
    load_image,
    generate_saliency_map,
    generate_scanpath_from_saliency,
)
from active_vision_irl import load_irl_policy, run_active_vision_search


def _compute_passive_bbox_from_saliency(
    saliency_map: np.ndarray,
    image_size: Tuple[int, int],
    percentile: float = 90.0,
) -> Tuple[float, float, float, float]:
    """
    Compute a simple bounding box from a saliency map.

    The bounding box encloses the region above a given percentile threshold.

    Parameters
    ----------
    saliency_map:
        Full-resolution saliency map (H, W) in [0, 1].
    image_size:
        (img_w, img_h) of the original image.
    percentile:
        Percentile threshold for salient pixels (e.g. 90.0 for top 10%).

    Returns
    -------
    bbox:
        (x_min, y_min, x_max, y_max) in image coordinates.
    """
    img_w, img_h = image_size
    if saliency_map.shape != (img_h, img_w):
        # Resize if needed
        from PIL import Image as PILImage

        saliency_pil = PILImage.fromarray((saliency_map * 255).astype(np.uint8))
        saliency_map = np.array(saliency_pil.resize((img_w, img_h))) / 255.0

    thresh = np.percentile(saliency_map, percentile)
    mask = saliency_map >= thresh
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        # Fallback to center box if saliency is uniform
        return img_w * 0.25, img_h * 0.25, img_w * 0.75, img_h * 0.75

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return float(x_min), float(y_min), float(x_max), float(y_max)


def _draw_bbox(
    ax: plt.Axes,
    bbox: Tuple[float, float, float, float],
    color: str,
    label: str,
    linewidth: float = 3.0,
) -> None:
    """
    Draw a rectangular bounding box on an axis.
    """
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    rect = plt.Rectangle(
        (x_min, y_min),
        width,
        height,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
    )
    ax.add_patch(rect)
    ax.text(
        x_min,
        max(0, y_min - 10),
        label,
        color=color,
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.6),
    )


def _visualize_comparison(
    image: Image.Image,
    passive_saliency: np.ndarray,
    passive_scanpath: List[Tuple[int, int]],
    passive_bbox: Tuple[float, float, float, float],
    active_outputs: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Create a side-by-side comparison between passive and active pipelines.

    LEFT  : Passive CLIP heatmap + scanpath + bounding box.
    RIGHT : Active IRL fixation path + belief map + final bounding box.
    """
    img_w, img_h = image.size
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    # ----- LEFT: PASSIVE CLIP -----
    axes[0].imshow(image)
    axes[0].imshow(
        passive_saliency,
        alpha=0.5,
        cmap="hot",
        interpolation="bilinear",
    )
    if passive_scanpath:
        xs, ys = zip(*passive_scanpath)
        axes[0].plot(xs, ys, "c-o", linewidth=2, markersize=6, label="Scanpath")
        for idx, (x, y) in enumerate(zip(xs, ys), start=1):
            axes[0].text(
                x + 5,
                y - 5,
                str(idx),
                color="yellow",
                fontsize=10,
                weight="bold",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
            )
    _draw_bbox(axes[0], passive_bbox, color="lime", label="PASSIVE BOX")
    axes[0].set_title("PASSIVE CLIP", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # ----- RIGHT: ACTIVE IRL -----
    belief_full = active_outputs.get("belief_map_fullres")
    fixation_points = active_outputs.get("fixation_points", [])
    active_bbox = active_outputs.get("final_bbox")

    axes[1].imshow(image)
    if isinstance(belief_full, np.ndarray) and belief_full.shape == (img_h, img_w):
        axes[1].imshow(
            belief_full,
            alpha=0.5,
            cmap="viridis",
            interpolation="bilinear",
        )

    if fixation_points:
        xs = [p[0] for p in fixation_points]
        ys = [p[1] for p in fixation_points]
        axes[1].plot(xs, ys, "m-o", linewidth=2, markersize=6, label="IRL Fixations")
        for idx, (x, y) in enumerate(zip(xs, ys), start=1):
            axes[1].text(
                x + 5,
                y - 5,
                str(idx),
                color="white",
                fontsize=10,
                weight="bold",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
            )

    if active_bbox is not None:
        _draw_bbox(axes[1], active_bbox, color="dodgerblue", label="ACTIVE BOX")

    axes[1].set_title("ACTIVE IRL", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Comparison visualization saved to {os.path.abspath(output_path)}")


def run_comparison(
    image_path: str,
    text_query: str,
    output_path: str = "comparison_output.png",
    device: str | None = None,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
) -> None:
    """
    Run both PASSIVE and ACTIVE pipelines on the same image + text query.

    Parameters
    ----------
    image_path:
        Path or URL of the input image.
    text_query:
        Text description of the target object.
    output_path:
        Where to save the side-by-side visualization.
    device:
        Torch device string (e.g. 'cuda:0' or 'cpu'). If None, chooses automatically.
    model_name:
        CLIP model name for open_clip.
    pretrained:
        Identifier for pretrained weights.
    """
    if not HAS_OPEN_CLIP:
        raise RuntimeError(
            "open_clip is required. Install with: pip install open-clip-torch"
        )

    torch_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Using device: {torch_device}")

    # Load image
    print("[INFO] Loading image...")
    image = load_image(image_path)
    print(f"[OK] Image loaded: {image.size[0]}x{image.size[1]}")

    # Load shared CLIP model
    print(f"[INFO] Loading CLIP model ({model_name})...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=torch_device
    )
    clip_tokenizer = open_clip.get_tokenizer(model_name)
    clip_model.eval()
    print("[OK] CLIP loaded.")

    # PASSIVE pipeline: saliency + scanpath + bbox
    print("[INFO] Running PASSIVE CLIP pipeline...")
    saliency_map, saliency_resized = generate_saliency_map(
        image, text_query, clip_model, clip_tokenizer, clip_preprocess, torch_device
    )
    passive_scanpath = generate_scanpath_from_saliency(
        saliency_resized, image.size, num_fixations=7
    )
    passive_bbox = _compute_passive_bbox_from_saliency(
        saliency_resized, image.size, percentile=90.0
    )

    # ACTIVE pipeline: IRL policy + CLIP patches
    print("[INFO] Loading IRL policy...")
    irl_policy, hparams = load_irl_policy(device=torch_device)
    print("[OK] IRL policy loaded.")

    print("[INFO] Running ACTIVE IRL pipeline...")
    active_outputs = run_active_vision_search(
        image=image,
        text_query=text_query,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        clip_tokenizer=clip_tokenizer,
        irl_policy=irl_policy,
        hparams=hparams,
        device=torch_device,
        max_steps=None,
        score_threshold=0.3,
    )

    # Visualization
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _visualize_comparison(
        image=image,
        passive_saliency=saliency_resized,
        passive_scanpath=passive_scanpath,
        passive_bbox=passive_bbox,
        active_outputs=active_outputs,
        output_path=output_path,
    )


def main() -> None:
    """
    CLI entry point for running the PASSIVE vs ACTIVE comparison.
    """
    parser = argparse.ArgumentParser(
        description="Compare PASSIVE CLIP vs ACTIVE IRL vision pipelines."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Image path or URL.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Text query describing the target object (e.g., 'biscuit').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison_output.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string like 'cuda:0' or 'cpu' (optional).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="ViT-B-32",
        help="CLIP model name for open_clip.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="openai",
        help="Pretrained weights identifier for open_clip.",
    )

    args = parser.parse_args()
    run_comparison(
        image_path=args.image,
        text_query=args.task,
        output_path=args.output,
        device=args.device,
        model_name=args.model_name,
        pretrained=args.pretrained,
    )


if __name__ == "__main__":
    main()


