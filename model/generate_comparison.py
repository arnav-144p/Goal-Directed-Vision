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
    encode_clip_query_embedding,
)
from active_vision_irl import (
    load_irl_policy,
    initial_state,
    seed_belief_from_saliency,
    next_action,
    crop_patch,
    evaluate_patch_with_clip,
    update_state,
    finalize_output,
)
from run_comparison import _refine_unified_bbox_with_clip


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

    text_emb = encode_clip_query_embedding(model, tokenizer, text_query, device)
    bbox = _refine_unified_bbox_with_clip(
        image,
        saliency_resized,
        text_emb,
        model,
        preprocess,
        device,
    )
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


def run_active(
    image_path: str,
    text_query: str,
    saliency_prior: Optional[np.ndarray] = None,
    unified_bbox: Optional[Tuple[float, float, float, float]] = None,
) -> ActiveResult:
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
    text_emb = encode_clip_query_embedding(
        clip_model, clip_tokenizer, text_query, device
    )

    # Load IRL
    print("[ACTIVE] Loading IRL policy...")
    irl_policy, hparams = load_irl_policy(device=device)
    print("[ACTIVE] IRL policy loaded.")

    # Initialize IRL state and run control loop manually so we can capture patches
    irl_state = initial_state(image, hparams, device)
    if saliency_prior is not None:
        seed_belief_from_saliency(irl_state, saliency_prior)
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
            text_emb=text_emb,
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
    final_bbox = (
        unified_bbox
        if unified_bbox is not None
        else final_outputs["final_bbox"]
    )

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

    # (d) final bounding box visualization (same bbox as passive when unified_bbox set)
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
    text_query: str = "object",
) -> None:
    """
    Generate a side-by-side comparison under `comparison/final/side_by_side.png`.

    Creates three panels:
    LEFT   = Scanpath Visualization (active vision fixations with connecting lines)
    MIDDLE = Detected Object (passive vision bounding box)
    RIGHT  = Original Image
    """
    paths = _ensure_dirs(".")
    passive_dir = paths["passive"]
    active_dir = paths["active"]
    final_dir = paths["final"]

    passive_box_path = os.path.join(passive_dir, "passive_final_box.png")
    side_by_side_path = os.path.join(final_dir, "side_by_side.png")

    # Load original image
    image = load_image(image_path)
    img_w, img_h = image.size

    # Create a three-panel visualization
    panel_w = img_w
    panel_h = img_h
    canvas_w = panel_w * 3
    canvas_h = panel_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    
    draw = ImageDraw.Draw(canvas)

    # Panel 1: Scanpath Visualization (Active Vision)
    canvas.paste(image, (0, 0))
    
    # Draw active vision scanpath with connecting lines
    if active_result.fixation_points and len(active_result.fixation_points) > 0:
        fixations = active_result.fixation_points
        
        # Draw connecting lines
        for i in range(len(fixations) - 1):
            x1, y1 = fixations[i]
            x2, y2 = fixations[i + 1]
            draw.line([(x1, y1), (x2, y2)], fill=(255, 0, 0), width=3)
        
        # Draw fixation points with numbers
        for idx, (fx, fy) in enumerate(fixations, start=1):
            # Draw point
            r = 8
            draw.ellipse(
                (fx - r, fy - r, fx + r, fy + r),
                fill=(255, 0, 0),
                outline=(255, 255, 255),
                width=2,
            )
            # Draw number
            text = str(idx)
            bbox = draw.textbbox((0, 0), text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            draw.rectangle(
                [fx + 10, fy - 10 - text_h, fx + 10 + text_w + 4, fy - 10],
                fill=(0, 0, 0),
                outline=None,
            )
            draw.text(
                (fx + 12, fy - 10 - text_h),
                text,
                fill=(255, 255, 0),
            )
    
    # Add title for Panel 1
    draw.text(
        (10, 10),
        "Scanpath Visualization",
        fill=(255, 255, 255),
        stroke_width=3,
        stroke_fill=(0, 0, 0),
    )

    # Panel 2: Detected Object (Passive Vision)
    passive_img = Image.open(passive_box_path).convert("RGB")
    # Resize passive image to match panel size if needed
    if passive_img.size != (panel_w, panel_h):
        passive_img = passive_img.resize((panel_w, panel_h))
    canvas.paste(passive_img, (panel_w, 0))
    
    # Add title for Panel 2
    draw.text(
        (panel_w + 10, 10),
        f'Detected Object: "{text_query}"',
        fill=(255, 255, 255),
        stroke_width=3,
        stroke_fill=(0, 0, 0),
    )

    # Panel 3: Original Image
    canvas.paste(image, (panel_w * 2, 0))
    
    # Add title for Panel 3
    draw.text(
        (panel_w * 2 + 10, 10),
        "Original Image",
        fill=(255, 255, 255),
        stroke_width=3,
        stroke_fill=(0, 0, 0),
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

    # Run passive first (refined bbox + heatmap), then active with same CLIP prior
    passive_result = run_passive(args.image, args.query)
    active_result = run_active(
        args.image,
        args.query,
        saliency_prior=passive_result.heatmap,
        unified_bbox=passive_result.bbox,
    )

    # Generate final side-by-side visualization
    generate_side_by_side(passive_result, active_result, args.image, args.query)


if __name__ == "__main__":
    main()


