from models.IRL import entry as irl_entry
from models.Gazeformer import entry as gazeformer_entry
from models.demo import entry as demo_entry

# Optional: HAT depends on detectron2
try:
    from models.HAT import entry as hat_entry  # type: ignore
    _has_hat = True
except Exception:
    hat_entry = None  # type: ignore
    _has_hat = False

# Optional: Scanpaths depends on mmcv
try:
    from models.Scanpaths import entry as scanpaths_entry  # type: ignore
    _has_scanpaths = True
except Exception:
    scanpaths_entry = None  # type: ignore
    _has_scanpaths = False

from models.CLIPGaze import entry as clipgaze_entry

MODEL_REGISTRY = {
    'irl': irl_entry,
    'gazeformer': gazeformer_entry,
    'demo': demo_entry,
}
if _has_hat:
    MODEL_REGISTRY['hat'] = hat_entry  # type: ignore
if _has_scanpaths:
    MODEL_REGISTRY['scanpaths'] = scanpaths_entry  # type: ignore
MODEL_REGISTRY['clipgaze'] = clipgaze_entry
