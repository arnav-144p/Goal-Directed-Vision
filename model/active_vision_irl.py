"""
Active Vision with IRL Policy + CLIP
====================================

This module implements an ACTIVE VISION pipeline that uses the IRL model from
`models/IRL/` to generate sequential fixations. At each fixation, a
high-resolution patch is cropped from the original image and evaluated with
CLIP against a text query. The IRL policy is used as a learned prior over a
coarse patch grid, while CLIP provides semantic feedback.

The design mirrors the style of `interactive_search.py` but is organized into
modular functions:

- `load_irl_policy()`
- `initial_state()`
- `next_action()`
- `crop_patch()`
- `update_state()`
- `finalize_output()`

The main entry point for external callers is:

- `run_active_vision_search()`
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

try:
    import open_clip
    HAS_OPEN_CLIP = True
except ImportError:  # pragma: no cover - runtime warning only
    HAS_OPEN_CLIP = False
    print("[WARNING] open_clip not found. Install with: pip install open-clip-torch")

from models.IRL.model.config import JsonConfig
from models.IRL.model.models import LHF_Policy_Cond_Small
from models.IRL.model.utils import action_to_pos, pos_to_action


@dataclass
class IRLState:
    """Container for the IRL active vision state."""

    step: int
    max_steps: int
    patch_num: Tuple[int, int]  # (num_patches_x, num_patches_y)
    patch_size: Tuple[int, int]  # (patch_w, patch_h) on the IRL canvas
    canvas_size: Tuple[int, int]  # (canvas_w, canvas_h) IRL training resolution
    image_size: Tuple[int, int]  # (img_w, img_h) original image resolution
    visited_actions: List[int] = field(default_factory=list)
    fixation_points: List[Tuple[float, float]] = field(default_factory=list)
    clip_scores: List[float] = field(default_factory=list)
    belief_patch_map: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))
    feature_tensor: Optional[torch.Tensor] = None
    history_patch_map: Optional[np.ndarray] = None


def load_irl_policy(
    device: Optional[torch.device] = None,
    hparams_path: str = "models/IRL/hparams/coco_search18.json",
    checkpoint_dir: str = "models/IRL/pretrained_models",
) -> Tuple[LHF_Policy_Cond_Small, JsonConfig]:
    """
    Load the IRL policy network (generator) from the IRL folder.

    This function:
    - Reads hyperparameters from the given JSON config.
    - Inspects the pretrained checkpoint to infer the number of task categories.
    - Builds `LHF_Policy_Cond_Small` with the correct shapes.
    - Loads pretrained weights from `checkpoint_dir`.

    Parameters
    ----------
    device:
        Torch device to place the model on. If None, uses CUDA if available.
    hparams_path:
        Path to the IRL hyperparameter JSON file.
    checkpoint_dir:
        Directory containing `trained_generator.pkg`.

    Returns
    -------
    irl_policy:
        Loaded IRL policy network in evaluation mode.
    hparams:
        Loaded `JsonConfig` object with dataset & training parameters.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(hparams_path):
        raise FileNotFoundError(f"IRL hparams file not found: {hparams_path}")

    hparams = JsonConfig(hparams_path)

    checkpoint_path = os.path.join(checkpoint_dir, "trained_generator.pkg")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"IRL checkpoint not found: {checkpoint_path}")

    # Load checkpoint and infer number of task categories from weight shapes
    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state.get("model", state)

    # Find the feature encoder weight to infer target_size
    feat_key = None
    for k in state_dict.keys():
        if k.endswith("feat_enc.weight"):
            feat_key = k
            break
    if feat_key is None:
        raise RuntimeError("Could not find 'feat_enc.weight' in IRL checkpoint.")

    feat_weight = state_dict[feat_key]
    c_in = feat_weight.shape[1]
    input_size = 134  # as used in the original IRL code
    target_size = c_in - input_size
    if target_size <= 0:
        raise RuntimeError(
            f"Inferred invalid target_size={target_size} from IRL checkpoint (c_in={c_in})."
        )

    task_eye = torch.eye(target_size, device=device)
    irl_policy = LHF_Policy_Cond_Small(
        hparams.Data.patch_count, target_size, task_eye, input_size
    ).to(device)

    # Strip potential "module." prefix for DataParallel checkpoints
    cleaned_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = irl_policy.load_state_dict(cleaned_state, strict=False)
    if missing:
        print(f"[IRL] Warning: missing keys when loading policy: {missing}")
    if unexpected:
        print(f"[IRL] Warning: unexpected keys when loading policy: {unexpected}")

    irl_policy.eval()
    return irl_policy, hparams


def initial_state(
    image: Image.Image,
    hparams: JsonConfig,
    device: torch.device,
) -> IRLState:
    """
    Initialize the IRL active vision state for a given image.

    The state is defined over the IRL patch grid (`patch_num`) and canvas size
    (`im_w`, `im_h`) from the IRL hparams, independent of the original image
    resolution. Fixations are represented in *original image coordinates*.

    Parameters
    ----------
    image:
        Input image (PIL, RGB).
    hparams:
        IRL hyperparameters loaded via `JsonConfig`.
    device:
        Torch device (unused directly here but kept for API symmetry).

    Returns
    -------
    IRLState
        Initialized state with zeroed belief map and a center fixation.
    """
    img_w, img_h = image.size
    canvas_w, canvas_h = int(hparams.Data.im_w), int(hparams.Data.im_h)
    patch_num_x, patch_num_y = int(hparams.Data.patch_num[0]), int(
        hparams.Data.patch_num[1]
    )
    patch_w, patch_h = int(hparams.Data.patch_size[0]), int(hparams.Data.patch_size[1])

    # Start with a center fixation in image coordinates
    center_x_img, center_y_img = img_w / 2.0, img_h / 2.0

    belief_patch_map = np.zeros((patch_num_y, patch_num_x), dtype=np.float32)
    history_patch_map = np.zeros_like(belief_patch_map, dtype=np.float32)

    # Initial feature tensor is a minimal placeholder; will be updated after
    # the first CLIP evaluation.
    feature_tensor = torch.zeros(
        (1, 134, patch_num_y, patch_num_x), dtype=torch.float32, device=device
    )

    # Map initial fixation to the closest patch index for bookkeeping
    # (this is not used by the policy before the first update).
    canvas_center_x = canvas_w / 2.0
    canvas_center_y = canvas_h / 2.0
    init_action = pos_to_action(
        canvas_center_x, canvas_center_y, (patch_w, patch_h), (patch_num_x, patch_num_y)
    )

    return IRLState(
        step=0,
        max_steps=int(hparams.Data.max_traj_length),
        patch_num=(patch_num_x, patch_num_y),
        patch_size=(patch_w, patch_h),
        canvas_size=(canvas_w, canvas_h),
        image_size=(img_w, img_h),
        visited_actions=[init_action],
        fixation_points=[(center_x_img, center_y_img)],
        clip_scores=[],
        belief_patch_map=belief_patch_map,
        feature_tensor=feature_tensor,
        history_patch_map=history_patch_map,
    )


def _build_feature_tensor_from_belief(
    belief_patch_map: np.ndarray,
    history_patch_map: np.ndarray,
    num_channels: int = 134,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build a feature tensor of shape (1, num_channels, H, W) from belief & history.

    The IRL policy was trained with 134-channel belief features; here we provide
    a simple surrogate by tiling the normalized belief map and including a
    separate history channel.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    h, w = belief_patch_map.shape
    belief = belief_patch_map.astype(np.float32)
    if belief.max() > belief.min():
        belief = (belief - belief.min()) / (belief.max() - belief.min())

    history = history_patch_map.astype(np.float32)
    if history.max() > 0:
        history = history / history.max()

    # Create base channels: [belief, history, zeros ...]
    base = np.stack([belief, history], axis=0)  # (2, H, W)
    if num_channels > 2:
        zeros = np.zeros((num_channels - 2, h, w), dtype=np.float32)
        base = np.concatenate([base, zeros], axis=0)

    feat = torch.from_numpy(base).unsqueeze(0).to(device)  # (1, C, H, W)
    return feat


def next_action(
    irl_state: IRLState,
    irl_policy: LHF_Policy_Cond_Small,
) -> Tuple[int, Tuple[float, float]]:
    """
    Query the IRL policy for the next action (patch index) and convert it to
    a fixation position in *original image coordinates*.

    A simple inhibition-of-return is implemented by masking out previously
    visited patches.

    Parameters
    ----------
    irl_state:
        Current IRL state.
    irl_policy:
        Loaded IRL policy network.

    Returns
    -------
    action_index:
        Index in [0, patch_count-1] corresponding to the chosen patch.
    fixation_point:
        (x, y) coordinates in original image pixels.
    """
    device = irl_state.feature_tensor.device  # type: ignore[union-attr]
    patch_num_x, patch_num_y = irl_state.patch_num
    canvas_w, canvas_h = irl_state.canvas_size
    img_w, img_h = irl_state.image_size

    # Build feature tensor from current belief & history
    feature_tensor = _build_feature_tensor_from_belief(
        irl_state.belief_patch_map,
        irl_state.history_patch_map,
        num_channels=134,
        device=device,
    )

    # Task ID: we do not have semantic categories here, so we use a dummy task
    # index of zero. The original IRL implementation assumes batch_size > 1 and
    # uses `.squeeze()` on the task one-hot tensor; to avoid dimensionality
    # issues when batch_size == 1, we replicate the feature tensor to a
    # batch of size 2 and only keep the first sample's output.
    if feature_tensor.size(0) == 1:
        feature_batch = feature_tensor.repeat(2, 1, 1, 1)
        tid_batch = torch.zeros((2,), dtype=torch.long, device=device)
    else:
        feature_batch = feature_tensor
        tid_batch = torch.zeros(
            (feature_batch.size(0),), dtype=torch.long, device=device
        )

    with torch.no_grad():
        act_probs, _ = irl_policy(feature_batch, tid_batch, act_only=False)

    probs = act_probs[0]  # (patch_count,) – use the first (real) sample
    if irl_state.visited_actions:
        # Mask previously visited actions
        mask = torch.ones_like(probs)
        visited_idx = torch.tensor(
            irl_state.visited_actions, dtype=torch.long, device=probs.device
        )
        mask[visited_idx] = 0.0
        probs = probs * mask

    if probs.sum() <= 0:
        # Fallback to uniform over unvisited patches
        total_patches = patch_num_x * patch_num_y
        probs = torch.ones(total_patches, device=device)
        if irl_state.visited_actions:
            visited_idx = torch.tensor(
                irl_state.visited_actions, dtype=torch.long, device=probs.device
            )
            probs[visited_idx] = 0.0

    # Argmax for deterministic behavior
    action_index = int(torch.argmax(probs).item())

    # Convert action index on IRL canvas to original image coordinates
    canvas_x, canvas_y = action_to_pos(
        torch.tensor([action_index], device=device),
        irl_state.patch_size,
        irl_state.patch_num,
    )
    canvas_x = float(canvas_x.item())
    canvas_y = float(canvas_y.item())

    # Map from canvas (im_w, im_h) to original image
    scale_x = img_w / float(canvas_w)
    scale_y = img_h / float(canvas_h)
    img_x = canvas_x * scale_x
    img_y = canvas_y * scale_y

    return action_index, (img_x, img_y)


def crop_patch(
    image: Image.Image,
    fixation_point: Tuple[float, float],
    irl_state: IRLState,
    scale_factor: float = 1.0,
) -> Image.Image:
    """
    Crop a high-resolution patch around the given fixation point.

    The patch size is derived from the IRL patch size on the IRL canvas and
    scaled to the original image resolution.

    Parameters
    ----------
    image:
        Original image.
    fixation_point:
        (x, y) fixation in original image coordinates.
    irl_state:
        Current IRL state with patch and canvas information.
    scale_factor:
        Multiplier for patch size (e.g. 1.0 for same size as IRL patch, 2.0 for larger).

    Returns
    -------
    patch:
        Cropped image patch (PIL Image, RGB).
    """
    img_w, img_h = irl_state.image_size
    canvas_w, canvas_h = irl_state.canvas_size
    patch_w, patch_h = irl_state.patch_size

    scale_x = img_w / float(canvas_w)
    scale_y = img_h / float(canvas_h)

    patch_w_img = patch_w * scale_x * scale_factor
    patch_h_img = patch_h * scale_y * scale_factor

    cx, cy = fixation_point
    left = max(0, int(cx - patch_w_img / 2.0))
    top = max(0, int(cy - patch_h_img / 2.0))
    right = min(img_w, int(cx + patch_w_img / 2.0))
    bottom = min(img_h, int(cy + patch_h_img / 2.0))

    return image.crop((left, top, right, bottom))


def evaluate_patch_with_clip(
    patch: Image.Image,
    text_query: str,
    clip_model: torch.nn.Module,
    clip_preprocess,
    clip_tokenizer,
    device: torch.device,
) -> float:
    """
    Evaluate a cropped patch with CLIP against the given text query.

    Parameters
    ----------
    patch:
        Cropped image patch.
    text_query:
        Text description of the target object.
    clip_model:
        CLIP model (from `open_clip.create_model_and_transforms`).
    clip_preprocess:
        CLIP image preprocessor / transform.
    clip_tokenizer:
        CLIP tokenizer for text.
    device:
        Torch device.

    Returns
    -------
    clip_score:
        Cosine similarity between patch and text embeddings.
    """
    clip_model.eval()

    with torch.no_grad():
        text_tokens = clip_tokenizer([text_query]).to(device)
        text_emb = clip_model.encode_text(text_tokens)
        text_emb = F.normalize(text_emb, dim=-1)

        patch_tensor = clip_preprocess(patch).unsqueeze(0).to(device)
        image_emb = clip_model.encode_image(patch_tensor)
        image_emb = F.normalize(image_emb, dim=-1)

        sim = (image_emb @ text_emb.T).squeeze().item()

    return float(sim)


def update_state(
    irl_state: IRLState,
    action_index: int,
    fixation_point: Tuple[float, float],
    clip_score: float,
) -> IRLState:
    """
    Update the IRL state with the latest action, fixation, and CLIP score.

    The belief map is updated at the selected patch location with the new
    similarity score, and the history map is marked as visited.

    Parameters
    ----------
    irl_state:
        Current IRL state.
    action_index:
        Patch index chosen by the IRL policy.
    fixation_point:
        (x, y) fixation in original image coordinates.
    clip_score:
        CLIP similarity score for the cropped patch.

    Returns
    -------
    IRLState
        Updated IRL state.
    """
    patch_num_x, patch_num_y = irl_state.patch_num

    y_idx = action_index // patch_num_x
    x_idx = action_index % patch_num_x

    irl_state.step += 1
    irl_state.visited_actions.append(action_index)
    irl_state.fixation_points.append(fixation_point)
    irl_state.clip_scores.append(clip_score)

    irl_state.belief_patch_map[y_idx, x_idx] = clip_score
    irl_state.history_patch_map[y_idx, x_idx] = 1.0

    return irl_state


def _upsample_belief_to_image(
    belief_patch_map: np.ndarray,
    irl_state: IRLState,
) -> np.ndarray:
    """
    Upsample the patch-level belief map to full image resolution.

    Parameters
    ----------
    belief_patch_map:
        (H_patches, W_patches) belief map.
    irl_state:
        IRL state with image and patch sizes.

    Returns
    -------
    belief_image:
        (img_h, img_w) numpy array normalized to [0, 1].
    """
    img_w, img_h = irl_state.image_size
    patch_num_x, patch_num_y = irl_state.patch_num

    # Simple nearest-neighbor upsampling by tiling
    tile_h = max(1, img_h // patch_num_y)
    tile_w = max(1, img_w // patch_num_x)

    belief = belief_patch_map.astype(np.float32)
    if belief.max() > belief.min():
        belief = (belief - belief.min()) / (belief.max() - belief.min())

    belief_tiled = np.kron(
        belief, np.ones((tile_h, tile_w), dtype=np.float32)
    )  # (H', W')

    belief_img = np.zeros((img_h, img_w), dtype=np.float32)
    h_min = min(img_h, belief_tiled.shape[0])
    w_min = min(img_w, belief_tiled.shape[1])
    belief_img[:h_min, :w_min] = belief_tiled[:h_min, :w_min]
    return belief_img


def _fallback_fixation_bbox(
    fixation_point: Tuple[float, float],
    irl_state: IRLState,
    score_threshold: float,
) -> Tuple[float, float, float, float]:
    """
    Compute a reasonable fallback bbox around a single fixation using the
    IRL patch size projected to image coordinates.
    """
    img_w, img_h = irl_state.image_size
    patch_w, patch_h = irl_state.patch_size
    canvas_w, canvas_h = irl_state.canvas_size

    # Map patch size from IRL canvas to image pixels
    scale_x = img_w / float(canvas_w)
    scale_y = img_h / float(canvas_h)
    patch_w_img = patch_w * scale_x
    patch_h_img = patch_h * scale_y

    # Slightly enlarge if we expect a confident match
    size_factor = 1.5 if score_threshold > 0 else 1.0
    bw = patch_w_img * size_factor
    bh = patch_h_img * size_factor

    cx, cy = fixation_point
    x_min = max(0.0, cx - bw / 2.0)
    y_min = max(0.0, cy - bh / 2.0)
    x_max = min(float(img_w), cx + bw / 2.0)
    y_max = min(float(img_h), cy + bh / 2.0)

    return x_min, y_min, x_max, y_max


def finalize_output(
    irl_state: IRLState,
    score_threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    Finalize the IRL active vision outputs for downstream visualization.

    Parameters
    ----------
    irl_state:
        Final IRL state after the control loop terminates.
    score_threshold:
        Threshold used to size the final bounding box. If the last CLIP score
        is low, the box remains at patch size; otherwise it can be slightly
        enlarged.

    Returns
    -------
    outputs:
        Dictionary containing:
        - `fixation_points`: list of (x, y) fixations.
        - `clip_scores`: list of CLIP similarity scores.
        - `final_bbox`: (x_min, y_min, x_max, y_max) in image pixels.
        - `belief_map_fullres`: upsampled belief map (H, W).
        - `patch_belief_map`: raw patch-level belief map (H_patches, W_patches).
    """
    img_w, img_h = irl_state.image_size

    # If we somehow have no fixations, fall back to a centered box
    if not irl_state.fixation_points:
        center_bbox = (img_w * 0.25, img_h * 0.25, img_w * 0.75, img_h * 0.75)
        belief_full = np.zeros((img_h, img_w), dtype=np.float32)
        return {
            "fixation_points": [],
            "clip_scores": [],
            "final_bbox": center_bbox,
            "belief_map_fullres": belief_full,
            "patch_belief_map": irl_state.belief_patch_map,
        }

    # Build a full-resolution belief map from all CLIP scores
    belief_full = _upsample_belief_to_image(irl_state.belief_patch_map, irl_state)

    # Use a percentile-based bbox similar to the passive saliency case so that
    # the box tightly hugs the region with highest semantic evidence.
    nonzero_vals = belief_full[belief_full > 0]
    if nonzero_vals.size > 0:
        thresh = np.percentile(nonzero_vals, 90.0)
        mask = belief_full >= thresh
        ys, xs = np.where(mask)
        if len(xs) > 0 and len(ys) > 0:
            x_min = float(xs.min())
            x_max = float(xs.max())
            y_min = float(ys.min())
            y_max = float(ys.max())
        else:
            # Fallback: bounding box around the last fixation
            last_fix = irl_state.fixation_points[-1]
            x_min, y_min, x_max, y_max = _fallback_fixation_bbox(
                last_fix, irl_state, score_threshold
            )
    else:
        # If the belief map is flat/zero, fall back to the last fixation box
        last_fix = irl_state.fixation_points[-1]
        x_min, y_min, x_max, y_max = _fallback_fixation_bbox(
            last_fix, irl_state, score_threshold
        )

    return {
        "fixation_points": irl_state.fixation_points,
        "clip_scores": irl_state.clip_scores,
        "final_bbox": (x_min, y_min, x_max, y_max),
        "belief_map_fullres": belief_full,
        "patch_belief_map": irl_state.belief_patch_map,
    }


def run_active_vision_search(
    image: Image.Image,
    text_query: str,
    clip_model: torch.nn.Module,
    clip_preprocess,
    clip_tokenizer,
    irl_policy: LHF_Policy_Cond_Small,
    hparams: JsonConfig,
    device: Optional[torch.device] = None,
    max_steps: Optional[int] = None,
    score_threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    Run the ACTIVE IRL vision pipeline on a single image & text query.

    Control loop:
        state = initialize_state()
        for t in range(max_steps):
            action = irl_policy(state)
            fixation = apply_action(action)
            patch = crop_patch(fixation)
            clip_score = evaluate_patch_with_CLIP(patch, text_query)
            state = update_state(state, fixation, clip_score)
            if stopping_condition(state):
                break

    Parameters
    ----------
    image:
        Input image (PIL, RGB).
    text_query:
        Text description of the target object.
    clip_model:
        Shared CLIP model instance.
    clip_preprocess:
        CLIP image preprocessing transform.
    clip_tokenizer:
        CLIP tokenizer.
    irl_policy:
        Loaded IRL policy network.
    hparams:
        IRL hyperparameters.
    device:
        Torch device, defaults to CUDA if available.
    max_steps:
        Maximum number of fixation steps; defaults to `hparams.Data.max_traj_length`.
    score_threshold:
        CLIP similarity threshold for early stopping and final bbox scaling.

    Returns
    -------
    results:
        Dictionary containing:
        - `fixation_points`
        - `clip_scores`
        - `final_bbox`
        - `belief_map_fullres`
        - `patch_belief_map`
        - `irl_state` (final state, for optional introspection)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not HAS_OPEN_CLIP:
        raise RuntimeError(
            "open_clip is required for active vision. Install with: pip install open-clip-torch"
        )

    if max_steps is None:
        max_steps = int(hparams.Data.max_traj_length)

    # Initialize IRL state
    irl_state = initial_state(image, hparams, device)

    # Main control loop
    for _ in range(max_steps):
        action_index, fixation_point = next_action(irl_state, irl_policy)
        patch = crop_patch(image, fixation_point, irl_state, scale_factor=1.5)
        clip_score = evaluate_patch_with_clip(
            patch, text_query, clip_model, clip_preprocess, clip_tokenizer, device
        )
        irl_state = update_state(irl_state, action_index, fixation_point, clip_score)

        # Stopping condition: good match OR reached max steps
        if clip_score >= score_threshold:
            break

    outputs = finalize_output(irl_state, score_threshold=score_threshold)
    outputs["irl_state"] = irl_state
    return outputs


__all__ = [
    "IRLState",
    "load_irl_policy",
    "initial_state",
    "next_action",
    "crop_patch",
    "evaluate_patch_with_clip",
    "update_state",
    "finalize_output",
    "run_active_vision_search",
]


