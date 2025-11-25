# Interactive Visual Search - Usage Guide

## Quick Start

Run the interactive script:
```bash
python interactive_search.py
```

The script will prompt you for:
1. **Image input**: Local file path or web URL
2. **Task**: What you want to search for (e.g., "biscuit", "cup", "red car")

## Features

✅ **Image Input Options**:
- Local file: `demo_inputs/dog.jpg`
- Web URL: `https://example.com/image.jpg`

✅ **Interactive Prompts**: No command-line arguments needed!

✅ **Two Output Images**:
1. **Saliency Map** (`demo_outputs/saliency_map.png`): Shows where the model is looking
2. **Final Result** (`demo_outputs/final_result.png`): Shows the object highlighted/outlined with scanpath

## Examples

### Interactive Mode (Recommended)
```bash
python interactive_search.py
```
Then enter:
- Image: `demo_inputs/dog.jpg` or a URL
- Task: `biscuit`

### Command-Line Mode
```bash
# Local image
python interactive_search.py --image demo_inputs/dog.jpg --task "biscuit"

# Web URL
python interactive_search.py --image https://example.com/image.jpg --task "cup"

# Custom output directory
python interactive_search.py --image demo_inputs/dog.jpg --task "red car" --output-dir my_results
```

## Output Files

1. **saliency_map.png**: 
   - Left: Original image
   - Right: Saliency map overlay showing task-relevant regions

2. **final_result.png**:
   - Left: Scanpath visualization (red dots show eye movement)
   - Right: Detected object highlighted in yellow with red outline

## Requirements

- Python 3.11+
- open-clip-torch (installed automatically if missing)
- torch, torchvision
- PIL/Pillow
- matplotlib
- numpy
- requests (for web URLs)
- opencv-python (optional, for better contour detection)

## Troubleshooting

**Missing open_clip?**
```bash
pip install open-clip-torch
```

**Missing opencv?**
```bash
pip install opencv-python
```
(Not required, but improves object outlining)

**CPU vs GPU:**
- Default: Auto-detects (CUDA if available, else CPU)
- Force CPU: `--device cpu`
- Force GPU: `--device cuda:0`

## How It Works

1. **Image Loading**: Downloads from URL or loads from local path
2. **CLIP Encoding**: Uses OpenCLIP to encode image patches and task text
3. **Saliency Generation**: Computes similarity between image patches and task text
4. **Scanpath Generation**: Creates eye movement path based on saliency
5. **Object Highlighting**: Highlights and outlines the most salient regions

## Model Options

Default: `ViT-B-32` with `openai` weights (fast, good quality)

Other options:
```bash
python interactive_search.py --model-name ViT-B-16 --pretrained openai
python interactive_search.py --model-name ViT-L-14 --pretrained openai
```

