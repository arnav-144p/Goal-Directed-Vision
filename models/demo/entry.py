import argparse
import os
import pathlib
import tempfile
from typing import List, Tuple

import numpy as np
import requests
import urllib3
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DEFAULT_IMAGE_URL = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
DEFAULT_OUTPUT = "demo_outputs/demo_scanpath.png"


def _ensure_directory(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _download_image(url: str, destination: pathlib.Path) -> pathlib.Path:
    _ensure_directory(destination)
    if destination.exists():
        return destination

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
    except requests.exceptions.SSLError:
        response = requests.get(url, timeout=20, verify=False)
        response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def _topk_fixations(image: Image.Image, k: int = 5) -> List[Tuple[int, int]]:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    # simple saliency: gradient magnitude
    gy, gx = np.gradient(gray)
    saliency = np.hypot(gx, gy)
    h, w = saliency.shape
    flat_indices = np.argpartition(saliency.flatten(), -k)[-k:]
    coords = np.array(np.unravel_index(flat_indices, (h, w))).T
    # sort by saliency descending
    scores = saliency[coords[:, 0], coords[:, 1]]
    order = np.argsort(-scores)
    coords = coords[order]
    return [(int(x), int(y)) for y, x in coords]


def _visualize_fixations(image: Image.Image, fixations: List[Tuple[int, int]], output_path: pathlib.Path) -> None:
    _ensure_directory(output_path)
    plt.figure(figsize=(8, 5))
    plt.imshow(image)
    if fixations:
        xs, ys = zip(*fixations)
        plt.scatter(xs, ys, c="red", s=60, marker="o", edgecolors="white", linewidths=1.5)
        for idx, (x, y) in enumerate(fixations, start=1):
            plt.text(x + 5, y - 5, str(idx), color="yellow", fontsize=10, weight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def eval(unknown_args, dataset=None):
    parser = argparse.ArgumentParser("Demo Scanpath Generator", add_help=True)
    parser.add_argument("--image", type=str, default=None,
                        help="Local image file path to analyze (e.g., demo_inputs/my_image.jpg)")
    parser.add_argument("--image_url", type=str, default=None,
                        help="Image URL to download and analyze")
    parser.add_argument("--num_fixations", type=int, default=5,
                        help="Number of heuristic fixations to generate")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Output path for the fixation visualization")
    args = parser.parse_args(unknown_args)

    output_path = pathlib.Path(args.output)
    
    # Determine image source: local file takes precedence, then URL, then default
    if args.image:
        # Use local file path
        image_path = pathlib.Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        print(f"[Demo] Using local image: {image_path}")
    elif args.image_url:
        # Download from URL
        sample_dir = pathlib.Path("demo_inputs")
        sample_path = sample_dir / pathlib.Path(args.image_url).name
        print(f"[Demo] Downloading sample image from {args.image_url}")
        try:
            image_path = _download_image(args.image_url, sample_path)
        except Exception as exc:  # pragma: no cover - network failure
            raise RuntimeError(f"Failed to download sample image: {exc}")
    else:
        # Use default URL
        sample_dir = pathlib.Path("demo_inputs")
        sample_path = sample_dir / pathlib.Path(DEFAULT_IMAGE_URL).name
        print(f"[Demo] Using default image, downloading from {DEFAULT_IMAGE_URL}")
        try:
            image_path = _download_image(DEFAULT_IMAGE_URL, sample_path)
        except Exception as exc:  # pragma: no cover - network failure
            raise RuntimeError(f"Failed to download sample image: {exc}")

    image = Image.open(image_path).convert("RGB")
    fixations = _topk_fixations(image, k=max(1, args.num_fixations))

    print("[Demo] Generated heuristic fixation sequence (x, y):")
    for idx, (x, y) in enumerate(fixations, start=1):
        print(f"  {idx}: ({x}, {y})")

    _visualize_fixations(image, fixations, output_path)
    print(f"[Demo] Visualization saved to {output_path.resolve()}")


def train(unknown_args, dataset=None):
    print("[Demo] Train mode is not supported for the demo model.")


def help():
    print("=== Demo Model ===")
    print("Generates a simple heuristic scanpath on a sample image.")
    print("Usage examples:")
    print("  python main.py --model demo --eval")
    print("  python main.py --model demo --eval --image demo_inputs/my_image.jpg")
    print("  python main.py --model demo --eval --image_url https://example.com/image.jpg")
    print("  python main.py --model demo --eval --image demo_inputs/my_image.jpg --output my_output.png --num_fixations 10")
