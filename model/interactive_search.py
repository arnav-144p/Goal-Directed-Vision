"""
Interactive Task-Guided Visual Search
Upload image (local or web), specify what to search for, and get:
1. Saliency map showing where the model is looking
2. Final output with the object highlighted/outlined

Usage:
    python interactive_search.py
"""

import argparse
import os
import pathlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms
import requests
import urllib3
from typing import Tuple, List
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[WARNING] OpenCV not found. Some features may be limited.")

# Use open_clip instead of clip
try:
    import open_clip
    HAS_OPEN_CLIP = True
except ImportError:
    HAS_OPEN_CLIP = False
    print("[WARNING] open_clip not found. Install with: pip install open-clip-torch")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def download_image(url: str, destination: pathlib.Path) -> pathlib.Path:
    """Download image from URL"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    
    try:
        response = requests.get(url, timeout=20, verify=False)
        response.raise_for_status()
        destination.write_bytes(response.content)
        return destination
    except Exception as e:
        raise RuntimeError(f"Failed to download image: {e}")


def load_image(image_input: str) -> Image.Image:
    """Load image from local path or URL"""
    # Check if it's a URL
    if image_input.startswith(('http://', 'https://')):
        print(f"[INFO] Downloading image from URL...")
        temp_path = pathlib.Path("demo_inputs") / pathlib.Path(image_input).name
        image_path = download_image(image_input, temp_path)
        return Image.open(image_path).convert("RGB")
    else:
        # Local file
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found: {image_input}")
        return Image.open(image_input).convert("RGB")


def get_interactive_inputs():
    """Get image and task from user interactively"""
    print("\n" + "="*60)
    print("  Interactive Task-Guided Visual Search")
    print("="*60 + "\n")
    
    # Get image input
    print("Image Input Options:")
    print("  1. Local file path (e.g., demo_inputs/dog.jpg)")
    print("  2. Web URL (e.g., https://example.com/image.jpg)")
    print()
    image_input = input("Enter image path or URL: ").strip()
    
    if not image_input:
        raise ValueError("Image input cannot be empty")
    
    # Get task
    print()
    task = input("What do you want to search for? (e.g., 'biscuit', 'cup', 'red car'): ").strip()
    
    if not task:
        raise ValueError("Task cannot be empty")
    
    return image_input, task


def extract_patch_features(image: Image.Image, model, preprocess, device, patch_size=64, stride=32):
    """Extract patch-level features from image using CLIP with overlapping patches"""
    w, h = image.size
    
    # Use overlapping patches for better coverage
    patches = []
    patch_coords = []
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)
            patch_coords.append((x, y, x + patch_size, y + patch_size))  # Store bbox
    
    # Process patches in batches
    batch_size = 16  # Smaller batch for better memory usage
    features = []
    
    for i in range(0, len(patches), batch_size):
        batch = patches[i:i + batch_size]
        batch_tensors = torch.stack([preprocess(p) for p in batch]).to(device)
        
        with torch.no_grad():
            batch_features = model.encode_image(batch_tensors)
            batch_features = F.normalize(batch_features, dim=-1)
            features.append(batch_features.cpu())
    
    features = torch.cat(features, dim=0)
    
    # Calculate grid dimensions
    grid_h = (h - patch_size) // stride + 1
    grid_w = (w - patch_size) // stride + 1
    
    return features, patch_coords, (grid_h, grid_w), (h, w)


def generate_saliency_map(image: Image.Image, task_text: str, model, tokenizer, preprocess, device) -> Tuple[np.ndarray, np.ndarray]:
    """Generate task-guided saliency map with better quality"""
    print("[INFO] Generating saliency map...")
    
    # Get text embedding
    text_tokens = tokenizer([task_text]).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(text_tokens)
        text_emb = F.normalize(text_emb, dim=-1)
    
    # Use adaptive patch size based on image size
    w, h = image.size
    patch_size = min(128, max(64, min(w, h) // 8))
    stride = patch_size // 2  # 50% overlap
    
    # Extract patch features
    patch_features, patch_coords, grid_shape, img_shape = extract_patch_features(
        image, model, preprocess, device, patch_size, stride
    )
    
    # Compute similarity for all patches at once
    patch_features = patch_features.to(device)
    with torch.no_grad():
        # Batch compute similarities
        similarities = (patch_features @ text_emb.T).squeeze().cpu().numpy()
    
    # Create high-resolution saliency map
    img_h, img_w = img_shape
    saliency_map = np.zeros((img_h, img_w), dtype=np.float32)
    count_map = np.zeros((img_h, img_w), dtype=np.float32)
    
    # Accumulate similarities with overlapping patches
    for idx, (x1, y1, x2, y2) in enumerate(patch_coords):
        sim = similarities[idx]
        # Add similarity to all pixels in this patch
        saliency_map[y1:y2, x1:x2] += sim
        count_map[y1:y2, x1:x2] += 1.0
    
    # Average overlapping regions
    saliency_map = np.divide(saliency_map, count_map + 1e-8)
    
    # Apply Gaussian smoothing for better quality
    if HAS_CV2:
        saliency_map = cv2.GaussianBlur(saliency_map, (21, 21), 0)
    else:
        try:
            from scipy import ndimage
            saliency_map = ndimage.gaussian_filter(saliency_map, sigma=5)
        except ImportError:
            # Fallback: simple box filter smoothing
            # This is a basic implementation - scipy is recommended
            h, w = saliency_map.shape
            smoothed = np.zeros_like(saliency_map)
            for i in range(2, h-2):
                for j in range(2, w-2):
                    smoothed[i, j] = np.mean(saliency_map[i-2:i+3, j-2:j+3])
            saliency_map = smoothed
    
    # Normalize saliency map
    saliency_min = saliency_map.min()
    saliency_max = saliency_map.max()
    if saliency_max > saliency_min:
        saliency_map = (saliency_map - saliency_min) / (saliency_max - saliency_min)
    else:
        saliency_map = np.ones_like(saliency_map) * 0.5
    
    # Apply power law to enhance contrast
    saliency_map = np.power(saliency_map, 0.7)
    
    # Resize is not needed since we already have full resolution
    saliency_resized = saliency_map
    
    return saliency_map, saliency_resized


def generate_scanpath_from_saliency(saliency_map: np.ndarray, image_size: Tuple[int, int], num_fixations: int = 7) -> List[Tuple[int, int]]:
    """Generate scanpath from saliency map with inhibition of return"""
    h, w = saliency_map.shape
    img_w, img_h = image_size
    
    # Start from center (first fixation)
    fixations = []
    current_y, current_x = h // 2, w // 2
    fixations.append((current_x, current_y))
    
    # Create inhibition map (IOR - Inhibition of Return)
    inhibition_map = np.zeros((h, w))
    inhibition_radius = min(h, w) // 8
    
    # Apply inhibition around first fixation
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    dist = np.sqrt((y_coords - current_y)**2 + (x_coords - current_x)**2)
    inhibition_map = np.exp(-dist / inhibition_radius)
    
    # Generate remaining fixations
    for _ in range(num_fixations - 1):
        # Apply inhibition to saliency
        biased_saliency = saliency_map * (1.0 - inhibition_map * 0.8)
        
        # Find maximum
        max_idx = np.argmax(biased_saliency)
        y, x = np.unravel_index(max_idx, (h, w))
        
        # Add to fixations
        fixations.append((x, y))
        
        # Update inhibition map
        dist = np.sqrt((y_coords - y)**2 + (x_coords - x)**2)
        new_inhibition = np.exp(-dist / inhibition_radius)
        inhibition_map = np.maximum(inhibition_map, new_inhibition)
    
    # Convert to image coordinates (saliency map is already full resolution)
    img_fixations = []
    for x, y in fixations:
        # Clamp to image bounds
        img_x = max(0, min(img_w - 1, x))
        img_y = max(0, min(img_h - 1, y))
        img_fixations.append((img_x, img_y))
    
    return img_fixations


def visualize_saliency_map(image: Image.Image, saliency_map: np.ndarray, output_path: str, task_text: str):
    """Visualize saliency map overlaid on image"""
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Saliency map overlay
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Saliency map should already be full resolution
    if saliency_map.shape != (h, w):
        if HAS_CV2:
            saliency_resized = cv2.resize(saliency_map, (w, h), interpolation=cv2.INTER_CUBIC)
        else:
            from PIL import Image as PILImage
            saliency_pil = PILImage.fromarray((saliency_map * 255).astype(np.uint8))
            saliency_resized = np.array(saliency_pil.resize((w, h))) / 255.0
    else:
        saliency_resized = saliency_map
    
    # Overlay saliency on image
    axes[1].imshow(image)
    axes[1].imshow(saliency_resized, alpha=0.5, cmap='hot', interpolation='bilinear')
    axes[1].set_title(f'Saliency Map (Searching for: "{task_text}")', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saliency map saved to {output_path.resolve()}")


def highlight_object(image: Image.Image, saliency_map: np.ndarray, task_text: str, output_path: str):
    """Create final output with object highlighted/outlined"""
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Saliency map should already be full resolution, but ensure it matches
    if saliency_map.shape != (h, w):
        if HAS_CV2:
            saliency_resized = cv2.resize(saliency_map, (w, h), interpolation=cv2.INTER_CUBIC)
        else:
            from PIL import Image as PILImage
            saliency_pil = PILImage.fromarray((saliency_map * 255).astype(np.uint8))
            saliency_resized = np.array(saliency_pil.resize((w, h))) / 255.0
    else:
        saliency_resized = saliency_map
    
    # Use adaptive thresholding - find the main object region
    # Use Otsu's method or percentile-based thresholding
    threshold = np.percentile(saliency_resized, 90)  # Top 10% most salient
    
    # Create binary mask
    mask = saliency_resized > threshold
    
    # Apply morphological operations to clean up the mask
    if HAS_CV2:
        # Remove small noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        
        # Smooth edges
        mask = cv2.GaussianBlur(mask.astype(np.float32), (9, 9), 2) > 0.5
        mask = mask.astype(np.uint8)
    else:
        try:
            from scipy import ndimage
            # Remove small objects
            mask = ndimage.binary_opening(mask, structure=np.ones((5, 5)))
            mask = ndimage.binary_closing(mask, structure=np.ones((15, 15)))
            mask = mask.astype(np.uint8)
        except ImportError:
            # Fallback: simple erosion/dilation with numpy
            # This is a very basic implementation
            mask = mask.astype(np.uint8)
    
    # Find the largest connected component (main object)
    if HAS_CV2:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            # Get largest component (skip background label 0)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest_label).astype(np.uint8)
    
    # Create highlighted image
    highlighted = img_array.copy().astype(np.float32)
    
    # Apply highlight to salient regions (yellow tint)
    highlight_color = np.array([255, 255, 0], dtype=np.float32)
    mask_float = mask.astype(np.float32) / 255.0
    
    # Blend: 70% original + 30% yellow highlight
    for c in range(3):
        highlighted[:, :, c] = highlighted[:, :, c] * (1.0 - mask_float * 0.3) + highlight_color[c] * mask_float * 0.3
    
    highlighted = np.clip(highlighted, 0, 255).astype(np.uint8)
    
    # Draw outline
    outlined = Image.fromarray(highlighted)
    draw = ImageDraw.Draw(outlined)
    
    if HAS_CV2:
        # Find contours for outlining
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw only the largest contour(s)
        if len(contours) > 0:
            # Sort by area and take top few
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            for contour in contours[:3]:  # Top 3 largest
                area = cv2.contourArea(contour)
                min_area = (h * w) * 0.01  # At least 1% of image
                
                if area > min_area:
                    # Simplify contour
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Draw outline
                    points = [(int(p[0][0]), int(p[0][1])) for p in approx]
                    if len(points) >= 3:
                        draw.polygon(points, outline=(255, 0, 0), width=4)
    else:
        # Simple bounding box approach without OpenCV
        y_coords, x_coords = np.where(mask > 0)
        if len(x_coords) > 0 and len(y_coords) > 0:
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            # Only draw if it's a reasonable size
            if (x_max - x_min) * (y_max - y_min) > (h * w) * 0.01:
                draw.rectangle([x_min, y_min, x_max, y_max], outline=(255, 0, 0), width=4)
    
    # Create visualization with scanpath
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original with scanpath - use full resolution saliency map
    scanpath = generate_scanpath_from_saliency(saliency_resized, image.size, num_fixations=7)
    axes[0].imshow(image)
    if scanpath and len(scanpath) > 0:
        xs, ys = zip(*scanpath)
        # Plot scanpath line
        axes[0].plot(xs, ys, 'r-', linewidth=3, alpha=0.8, label='Scanpath', zorder=3)
        # Plot fixation points
        axes[0].scatter(xs, ys, c='red', s=150, marker='o', edgecolors='white', linewidths=3, zorder=5)
        # Number the fixations
        for idx, (x, y) in enumerate(zip(xs, ys), start=1):
            axes[0].text(x + 15, y - 15, str(idx), color='yellow', fontsize=12,
                         weight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.8), zorder=6)
    else:
        print("[WARNING] No scanpath generated - saliency map may be too uniform")
    axes[0].set_title('Scanpath Visualization', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Highlighted object
    axes[1].imshow(outlined)
    axes[1].set_title(f'Detected Object: "{task_text}"', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Final output saved to {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description='Interactive Task-Guided Visual Search')
    parser.add_argument('--image', type=str, default=None, help='Image path or URL (optional, will prompt if not provided)')
    parser.add_argument('--task', type=str, default=None, help='Task/object to search for (optional, will prompt if not provided)')
    parser.add_argument('--output-dir', type=str, default='demo_outputs', help='Output directory')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda:0 or cpu)')
    parser.add_argument('--model-name', type=str, default='ViT-B-32', help='CLIP model name')
    parser.add_argument('--pretrained', type=str, default='openai', help='Pretrained weights')
    
    args = parser.parse_args()
    
    if not HAS_OPEN_CLIP:
        print("[ERROR] open_clip is required. Install with: pip install open-clip-torch")
        return
    
    # Get device
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Using device: {device}")
    
    # Get inputs interactively if not provided
    if args.image and args.task:
        image_input = args.image
        task = args.task
    else:
        image_input, task = get_interactive_inputs()
    
    # Load image
    print(f"\n[INFO] Loading image...")
    try:
        image = load_image(image_input)
        print(f"[OK] Image loaded: {image.size[0]}x{image.size[1]}")
    except Exception as e:
        print(f"[ERROR] Failed to load image: {e}")
        return
    
    # Load CLIP model
    print(f"[INFO] Loading CLIP model ({args.model_name})...")
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            args.model_name, pretrained=args.pretrained, device=device
        )
        tokenizer = open_clip.get_tokenizer(args.model_name)
        model.eval()
        print("[OK] Model loaded")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        print("[INFO] Try: pip install open-clip-torch")
        return
    
    # Generate saliency map
    try:
        saliency_map, saliency_resized = generate_saliency_map(
            image, task, model, tokenizer, preprocess, device
        )
    except Exception as e:
        print(f"[ERROR] Failed to generate saliency map: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create output directory
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate intermediate saliency visualization
    saliency_output = output_dir / "saliency_map.png"
    visualize_saliency_map(image, saliency_resized, str(saliency_output), task)
    
    # Generate final output with object highlighted
    final_output = output_dir / "final_result.png"
    highlight_object(image, saliency_resized, task, str(final_output))
    
    print("\n" + "="*60)
    print("  Processing Complete!")
    print("="*60)
    print(f"  Saliency Map: {saliency_output}")
    print(f"  Final Result: {final_output}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()

