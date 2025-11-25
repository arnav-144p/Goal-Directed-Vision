"""
Generate comprehensive PASSIVE vs ACTIVE vision visualizations.

This script produces a full set of intermediate and final images for:
  - Passive Vision (CLIP + saliency)
  - Active Vision (IRL + CLIP patches)

Directory layout (created automatically under project root):
  comparison/
    passive/
      passive_heatmap.png
      passive_peaks_stepwise/step_1.png, step_2.png, ...
      passive_final_box.png
    active/
      active_fixation_steps/step_1.png, step_2.png, ...
      active_patches_contact_sheet.png
      active_cumulative_attention.png
      active_final_box.png
    final/
      side_by_side.png

Run end-to-end:
    python generate_comparison.py --image <path> --query "<text>"
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

try:
    import open_clip
    HAS_OPEN_CLIP = True
except ImportError:  # pragma: no cover - runtime warning only
    HAS_OPEN_CLIP = False
    print("[WARNING] open_clip not found. Install with: pip install open-clip-torch")

try:
    import cv2
    HAS_CV2 = True
except ImportError:  # pragma: no cover
    HAS_CV2 = False
    print("[WARNING] OpenCV (cv2) not found. Some Gaussian visualizations will be approximate.")

from interactive_search import (
    load_image,
    generate_saliency_map,
    generate_scanpath_from_saliency,
    highlight_object,
)
from active_vision_irl import (
    load_irl_policy,
    initial_state,
    next_action,
    crop_patch,
    evaluate_patch_with_clip,
    update_state,
    finalize_output,
)


@dataclass
class PassiveResult:
    """Container for passive vision outputs needed for comparison."""

    heatmap: np.ndarray
    scanpath: List[Tuple[int, int]]
    bbox: Tuple[float, float, float, float]
    peak_points: List[Tuple[int, int]]


@dataclass
class ActiveResult:
    """Container for active vision outputs needed for comparison."""

    fixation_points: List[Tuple[float, float]]
    clip_scores: List[float]
    bbox: Tuple[float, float, float, float]
    cumulative_attention: np.ndarray


def _ensure_dirs(root: str) -> Dict[str, str]:
    """
    Create the comparison directory structure and return important paths.
    """
    base = os.path.join(root, "comparison")
    passive_dir = os.path.join(base, "passive")
    active_dir = os.path.join(base, "active")
    final_dir = os.path.join(base, "final")

    os.makedirs(passive_dir, exist_ok=True)
    os.makedirs(active_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    # Subfolders
    passive_peaks_dir = os.path.join(passive_dir, "passive_peaks_stepwise")
    os.makedirs(passive_peaks_dir, exist_ok=True)

    active_fix_steps_dir = os.path.join(active_dir, "active_fixation_steps")
    os.makedirs(active_fix_steps_dir, exist_ok=True)

    return {
        "base": base,
        "passive": passive_dir,
        "active": active_dir,
        "final": final_dir,
        "passive_peaks": passive_peaks_dir,
        "active_fix_steps": active_fix_steps_dir,
    }


def _compute_passive_bbox_from_saliency(
    saliency_map: np.ndarray,
    image_size: Tuple[int, int],
    percentile: float = 90.0,
) -> Tuple[float, float, float, float]:
    """
    Compute a bounding box from a saliency map by thresholding at a percentile.
    """
    img_w, img_h = image_size
    if saliency_map.shape != (img_h, img_w):
        # Resize to match image resolution
        saliency_pil = Image.fromarray((saliency_map * 255).astype(np.uint8))
        saliency_map = np.array(saliency_pil.resize((img_w, img_h))) / 255.0

    thresh = np.percentile(saliency_map, percentile)
    mask = saliency_map >= thresh
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        # Fallback: centered box
        return img_w * 0.25, img_h * 0.25, img_w * 0.75, img_h * 0.75

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return float(x_min), float(y_min), float(x_max), float(y_max)


def _save_passive_heatmap(
    heatmap: np.ndarray,
    output_path: str,
) -> None:
    """
    Save a standalone heatmap visualization (no image overlay).
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, cmap="hot", interpolation="bilinear")
    plt.colorbar(label="Similarity")
    plt.title("Passive CLIP Heatmap", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _save_passive_peak_steps(
    image: Image.Image,
    heatmap: np.ndarray,
    output_dir: str,
    max_steps: int = 10,
) -> List[Tuple[int, int]]:
    """
    Create a sequence of images showing the top-N peak locations in the
    passive heatmap, revealed step-by-step.

    Each `step_i.png` shows:
      - the original image
      - all peak points up to step i
      - the highest i-th peak highlighted and numbered
    """
    img_w, img_h = image.size
    if heatmap.shape != (img_h, img_w):
        heat_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap = np.array(heat_pil.resize((img_w, img_h))) / 255.0

    flat = heatmap.reshape(-1)
    indices = np.argsort(flat)[::-1]  # descending
    max_steps = min(max_steps, len(indices))

    peak_points: List[Tuple[int, int]] = []

    ys_all, xs_all = np.divmod(indices, img_w)

    for step in range(1, max_steps + 1):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(image)

        # Add all peaks up to current step
        for idx in range(step):
            x = int(xs_all[idx])
            y = int(ys_all[idx])
            color = "lime" if idx == 0 else "cyan"
            ax.scatter(x, y, s=80, edgecolors="black", facecolors=color, linewidths=1.5)
            ax.text(
                x + 5,
                y - 5,
                str(idx + 1),
                color="yellow",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
            )

        ax.set_title(f"Passive Peak {step}", fontsize=12, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"step_{step}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Store peak points (in order)
    for idx in range(max_steps):
        peak_points.append((int(xs_all[idx]), int(ys_all[idx])))

    return peak_points


def run_passive(image_path: str, text_query: str) -> PassiveResult:
    """
    Run the passive CLIP pipeline and generate visualizations under
    `comparison/passive/`.

    Outputs:
      - passive_heatmap.png
      - passive_peaks_stepwise/step_*.png
      - passive_final_box.png
    """
    if not HAS_OPEN_CLIP:
        raise RuntimeError(
            "open_clip is required. Install with: pip install open-clip-torch"
        )

    paths = _ensure_dirs(".")
    passive_dir = paths["passive"]
    peaks_dir = paths["passive_peaks"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PASSIVE] Using device: {device}")

    # Load image
    image = load_image(image_path)
    print(f"[PASSIVE] Image loaded: {image.size[0]}x{image.size[1]}")

    # Load CLIP
    print("[PASSIVE] Loading CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    # Generate saliency
    print("[PASSIVE] Generating saliency map...")
    saliency_map, saliency_resized = generate_saliency_map(
        image, text_query, model, tokenizer, preprocess, device
    )

    # (a) heatmap only
    heatmap_path = os.path.join(passive_dir, "passive_heatmap.png")
    _save_passive_heatmap(saliency_resized, heatmap_path)
    print(f"[PASSIVE] Heatmap saved to {heatmap_path}")

    # (b) peak stepwise
    print("[PASSIVE] Generating peak stepwise visualizations...")
    peak_points = _save_passive_peak_steps(image, saliency_resized, peaks_dir, max_steps=10)

    # (c) final bounding box (reuse existing visualization logic)
    final_box_path = os.path.join(passive_dir, "passive_final_box.png")
    highlight_object(image, saliency_resized, text_query, final_box_path)
    print(f"[PASSIVE] Final box visualization saved to {final_box_path}")

    # Also compute bbox numerically for side-by-side overlay
    bbox = _compute_passive_bbox_from_saliency(saliency_resized, image.size)
    scanpath = generate_scanpath_from_saliency(saliency_resized, image.size, num_fixations=7)

    return PassiveResult(
        heatmap=saliency_resized,
        scanpath=scanpath,
        bbox=bbox,
        peak_points=peak_points,
    )


def _gaussian_on_grid(
    center_x: float,
    center_y: float,
    img_w: int,
    img_h: int,
    sigma: float = 20.0,
) -> np.ndarray:
    """
    Create a 2D Gaussian map centered at (center_x, center_y) on an image grid.
    """
    xs = np.arange(img_w, dtype=np.float32)
    ys = np.arange(img_h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    gauss = np.exp(
        -(((xx - center_x) ** 2) + ((yy - center_y) ** 2)) / (2.0 * sigma * sigma)
    )
    return gauss.astype(np.float32)


def _save_active_fixation_steps(
    image: Image.Image,
    fixation_points: List[Tuple[float, float]],
    output_dir: str,
) -> None:
    """
    Save `step_i.png` for active IRL fixations, each showing:
      - fixation number
      - current fixation point (highlighted)
      - trajectory so far
    """
    for i in range(len(fixation_points)):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(image)

        xs = [p[0] for p in fixation_points[: i + 1]]
        ys = [p[1] for p in fixation_points[: i + 1]]

        # Trajectory so far
        if len(xs) > 1:
            ax.plot(xs, ys, "w--", linewidth=2, alpha=0.8)

        # All previous fixations
        if len(xs) > 1:
            ax.scatter(
                xs[:-1],
                ys[:-1],
                s=80,
                edgecolors="black",
                facecolors="purple",
                linewidths=1.5,
            )

        # Current fixation highlighted
        cx, cy = xs[-1], ys[-1]
        ax.scatter(
            [cx],
            [cy],
            s=200,
            edgecolors="yellow",
            facecolors="none",
            linewidths=2.5,
        )
        ax.text(
            cx + 5,
            cy - 5,
            f"{i + 1}",
            color="white",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
        )

        ax.set_title(f"Active IRL Fixation {i + 1}", fontsize=12, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"step_{i + 1}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _save_patches_contact_sheet(
    patches: List[Image.Image],
    output_path: str,
    max_cols: int = 5,
) -> None:
    """
    Save a contact sheet showing all high-resolution patches in order.
    """
    if not patches:
        # Nothing to save
        return

    # Normalize patch sizes (resize to the first patch size)
    base_w, base_h = patches[0].size
    resized = [p.resize((base_w, base_h)) for p in patches]

    n = len(resized)
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)

    sheet_w = cols * base_w
    sheet_h = rows * base_h

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(0, 0, 0))

    for idx, patch in enumerate(resized):
        r = idx // cols
        c = idx % cols
        x = c * base_w
        y = r * base_h
        sheet.paste(patch, (x, y))

    sheet.save(output_path)


def _save_cumulative_attention(
    cumulative_map: np.ndarray,
    output_path: str,
) -> None:
    """
    Save a cumulative attention heatmap for active IRL.
    """
    attn = cumulative_map.copy()
    if attn.max() > 0:
        attn = attn / attn.max()

    plt.figure(figsize=(8, 6))
    plt.imshow(attn, cmap="magma", interpolation="bilinear")
    plt.colorbar(label="Cumulative Attention")
    plt.title("Active IRL Cumulative Attention", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _save_active_final_box(
    image: Image.Image,
    bbox: Tuple[float, float, float, float],
    output_path: str,
) -> None:
    """
    Save the IRL final bounding box overlaid on the original image.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    x_min, y_min, x_max, y_max = bbox
    draw.rectangle(
        [x_min, y_min, x_max, y_max],
        outline=(30, 144, 255),  # dodger blue
        width=4,
    )
    draw.text(
        (x_min, max(0, y_min - 20)),
        "ACTIVE IRL",
        fill=(255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )
    img.save(output_path)


def run_active(image_path: str, text_query: str) -> ActiveResult:
    """
    Run the active IRL + CLIP pipeline and generate visualizations under
    `comparison/active/`.

    Outputs:
      - active_fixation_steps/step_*.png
      - active_patches_contact_sheet.png
      - active_cumulative_attention.png
      - active_final_box.png
    """
    if not HAS_OPEN_CLIP:
        raise RuntimeError(
            "open_clip is required. Install with: pip install open-clip-torch"
        )

    paths = _ensure_dirs(".")
    active_dir = paths["active"]
    fix_steps_dir = paths["active_fix_steps"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ACTIVE] Using device: {device}")

    # Load image
    image = load_image(image_path)
    img_w, img_h = image.size
    print(f"[ACTIVE] Image loaded: {img_w}x{img_h}")

    # Shared CLIP
    print("[ACTIVE] Loading CLIP model...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    clip_model.eval()

    # Load IRL
    print("[ACTIVE] Loading IRL policy...")
    irl_policy, hparams = load_irl_policy(device=device)
    print("[ACTIVE] IRL policy loaded.")

    # Initialize IRL state and run control loop manually so we can capture patches
    irl_state = initial_state(image, hparams, device)
    max_steps = int(hparams.Data.max_traj_length)
    score_threshold = 0.3

    patches: List[Image.Image] = []
    cumulative_attention = np.zeros((img_h, img_w), dtype=np.float32)

    for _ in range(max_steps):
        action_index, fixation_point = next_action(irl_state, irl_policy)
        patch = crop_patch(image, fixation_point, irl_state, scale_factor=1.5)
        clip_score = evaluate_patch_with_clip(
            patch,
            text_query,
            clip_model,
            clip_preprocess,
            clip_tokenizer,
            device,
        )
        irl_state = update_state(irl_state, action_index, fixation_point, clip_score)

        patches.append(patch)

        # Update cumulative attention with a Gaussian centered at the fixation
        fx, fy = fixation_point
        gauss = _gaussian_on_grid(fx, fy, img_w, img_h, sigma=25.0)
        cumulative_attention += gauss

        if clip_score >= score_threshold:
            break

    # Finalize IRL outputs (bbox + belief maps)
    final_outputs = finalize_output(irl_state, score_threshold=score_threshold)
    fixation_points = final_outputs["fixation_points"]
    clip_scores = final_outputs["clip_scores"]
    final_bbox = final_outputs["final_bbox"]

    # (a) fixation step images
    print("[ACTIVE] Saving fixation step visualizations...")
    _save_active_fixation_steps(image, fixation_points, fix_steps_dir)

    # (b) patches contact sheet
    contact_sheet_path = os.path.join(active_dir, "active_patches_contact_sheet.png")
    _save_patches_contact_sheet(patches, contact_sheet_path)
    print(f"[ACTIVE] Patches contact sheet saved to {contact_sheet_path}")

    # (c) cumulative attention heatmap
    cumulative_attn_path = os.path.join(active_dir, "active_cumulative_attention.png")
    _save_cumulative_attention(cumulative_attention, cumulative_attn_path)
    print(f"[ACTIVE] Cumulative attention saved to {cumulative_attn_path}")

    # (d) final bounding box visualization
    final_box_path = os.path.join(active_dir, "active_final_box.png")
    _save_active_final_box(image, final_bbox, final_box_path)
    print(f"[ACTIVE] Final box visualization saved to {final_box_path}")

    return ActiveResult(
        fixation_points=fixation_points,
        clip_scores=clip_scores,
        bbox=final_bbox,
        cumulative_attention=cumulative_attention,
    )


def generate_side_by_side(
    passive_result: PassiveResult,
    active_result: ActiveResult,
    image_path: str,
) -> None:
    """
    Generate a side-by-side comparison under `comparison/final/side_by_side.png`.

    LEFT  = passive_final_box.png (with optional passive points in green)
    RIGHT = active_final_box.png (with active fixations in purple)
    """
    paths = _ensure_dirs(".")
    passive_dir = paths["passive"]
    active_dir = paths["active"]
    final_dir = paths["final"]

    passive_box_path = os.path.join(passive_dir, "passive_final_box.png")
    active_box_path = os.path.join(active_dir, "active_final_box.png")
    side_by_side_path = os.path.join(final_dir, "side_by_side.png")

    # Load the final box images we saved earlier
    passive_img = Image.open(passive_box_path).convert("RGB")
    active_img = Image.open(active_box_path).convert("RGB")

    # Ensure same height by resizing active to match passive height
    p_w, p_h = passive_img.size
    a_w, a_h = active_img.size
    if a_h != p_h:
        new_a_w = int(a_w * (p_h / a_h))
        active_img = active_img.resize((new_a_w, p_h))
        a_w, a_h = active_img.size

    canvas_w = p_w + a_w
    canvas_h = p_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(0, 0, 0))
    canvas.paste(passive_img, (0, 0))
    canvas.paste(active_img, (p_w, 0))

    draw = ImageDraw.Draw(canvas)

    # Add titles
    draw.text(
        (10, 10),
        "PASSIVE CLIP",
        fill=(0, 255, 0),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )
    draw.text(
        (p_w + 10, 10),
        "ACTIVE IRL",
        fill=(186, 85, 211),  # medium orchid
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )

    # Overlay passive peak points (green) on the left half
    for x, y in passive_result.peak_points:
        r = 5
        draw.ellipse(
            (x - r, y - r, x + r, y + r),
            outline=(0, 255, 0),
            width=2,
        )

    # Overlay active fixations (purple) on the right half (offset by p_w)
    for fx, fy in active_result.fixation_points:
        r = 6
        draw.ellipse(
            (p_w + fx - r, fy - r, p_w + fx + r, fy + r),
            outline=(186, 85, 211),
            width=2,
        )

    canvas.save(side_by_side_path)
    print(f"[FINAL] Side-by-side comparison saved to {side_by_side_path}")


def main() -> None:
    """
    CLI entry point for generating all comparison visualizations.
    """
    parser = argparse.ArgumentParser(
        description="Generate PASSIVE vs ACTIVE vision comparison visualizations."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Input image path or URL.",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Text query describing the target object.",
    )

    args = parser.parse_args()

    # Run passive and active pipelines
    passive_result = run_passive(args.image, args.query)
    active_result = run_active(args.image, args.query)

    # Generate final side-by-side visualization
    generate_side_by_side(passive_result, active_result, args.image)


if __name__ == "__main__":
    main()


