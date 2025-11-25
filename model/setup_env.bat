@echo off
echo ============================================================
echo 🧠 Setting up HAT / Deformable DETR Environment (Windows)
echo ============================================================

:: Change to your project directory
cd /d "%~dp0"

:: Check Python
python --version || (
    echo ❌ Python not found. Please install Python 3.10 or newer and re-run.
    pause
    exit /b
)

:: Create a virtual environment
echo 🔧 Creating virtual environment...
python -m venv venv

:: Activate it
call venv\Scripts\activate

:: Upgrade pip
echo 🔼 Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

:: Install PyTorch (CPU version)
echo 🔥 Installing PyTorch (CPU version)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

:: Install common dependencies
echo 📦 Installing dependencies...
pip install --prefer-binary ^
  numpy scipy pandas matplotlib seaborn scikit-learn scikit-image ^
  opencv-python tqdm pyyaml docopt timm einops ^
  transformers sentence-transformers ftfy regex requests pillow joblib ^
  addict cloudpickle cython imageio protobuf yacs hydra-core ^
  fvcore iopath tabulate rich yapf black

:: Install MMCV / MMEngine / MMDet (CPU-safe)
echo ⚙️ Installing MMCV and MMDet (best-effort)...
pip install --prefer-binary mmengine==0.10.3 || echo ⚠️ MMEngine install failed, continuing...
pip install --prefer-binary "mmcv>=2.1.0" || echo ⚠️ MMCV install failed, continuing...
pip install --prefer-binary "mmdet>=3.3.0" || echo ⚠️ MMDetection install failed, continuing...

:: Install Detectron2 (latest)
echo 🧩 Installing Detectron2 (best-effort)...
set "DETECTRON_TORCH_TAG="
for /f "delims=" %%i in ('python -c "import torch; v=torch.__version__.split('+')[0].split('.'); print('torch' + v[0] + '.' + v[1])" 2^>^&1') do set "DETECTRON_TORCH_TAG=%%i"

if not defined DETECTRON_TORCH_TAG (
    echo ⚠️ Could not detect Torch version tag for Detectron2 wheels, skipping installation...
) else (
    echo 🔎 Detected Torch tag: %DETECTRON_TORCH_TAG%
    pip install --prefer-binary detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/%DETECTRON_TORCH_TAG%/index.html ^
        || echo ⚠️ Detectron2 install failed or wheel not available for %DETECTRON_TORCH_TAG%, continuing...
)

:: Done
echo ============================================================
echo ✅ Environment setup complete!
echo 💡 To activate environment later, run:
echo     venv\Scripts\activate
echo ============================================================

pause

:: Windows script text file