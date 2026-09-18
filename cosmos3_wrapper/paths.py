"""Filesystem layout, model registry, and resolution helpers for Cosmos 3.

Each Cosmos 3 *generator* checkpoint is downloaded as a self-contained
HuggingFace repo (full Diffusers pipeline: transformer + tokenizers + VAEs)
into its own directory under ``ComfyUI/models/Cosmos3/<model-key>``, e.g.::

    <models>/Cosmos3/Cosmos3-Nano/
        model_index.json
        transformer/...
        text_encoder/...
        vae/...
        ...

The ``Cosmos3OmniPipeline`` (Diffusers) is then pointed at that directory.
"""

import math
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(THIS_DIR)  # the scg-Cosmos3-tuned custom-node directory

# Default sub-directory (under ComfyUI/models) that holds the Cosmos 3 models.
DEFAULT_MODELS_SUBDIR = "Cosmos3"

# ---------------------------------------------------------------------------
# Model registry
#
# Only the *generator* checkpoints are listed here: those are the ones the
# Diffusers ``Cosmos3OmniPipeline`` can load to produce images / video / sound.
#
# Deliberately NOT exposed as generation nodes:
#   * The pure *reasoner* surface of Nano / Super (text-out VLM, served via
#     vLLM / Transformers) — that's the "LLM" path the user asked us to skip.
#   * ``Cosmos3-Nano-Policy-DROID`` — a robot-control policy (see README), not
#     an image/video generator.
#
# ``modes`` lists which generation modes a checkpoint is expected to support.
# The omnimodal Nano/Super checkpoints do everything; the Super-* checkpoints
# are task-specialized.
# ---------------------------------------------------------------------------
GENERATOR_MODELS = {
    "Cosmos3-Nano": {
        "repo_id": "nvidia/Cosmos3-Nano",
        "params": "16B",
        "modes": ("t2i", "t2v", "i2v"),
        "sound": True,
        "note": "Omnimodal 16B world model. Lightest generator; T2I / T2V / I2V (+ optional sound).",
    },
    "Cosmos3-Nano-nf4": {
        "repo_id": "SanDiegoDude/Cosmos3-Nano-nf4",
        "params": "16B",
        "modes": ("t2i", "t2v", "i2v"),
        "sound": True,
        "prequantized": "nf4",
        "note": "Pre-quantized NF4 (bnb 4-bit) Nano. ~11 GB VRAM, loads fast, no on-the-fly quant. Needs bitsandbytes.",
    },
    "Cosmos3-Super": {
        "repo_id": "nvidia/Cosmos3-Super",
        "params": "64B",
        "modes": ("t2i", "t2v", "i2v"),
        "sound": True,
        "note": "Frontier 64B omnimodal world model. Needs multi-GPU or heavy offload.",
    },
    "Cosmos3-Super-Text2Image": {
        "repo_id": "nvidia/Cosmos3-Super-Text2Image",
        "params": "64B",
        "modes": ("t2i",),
        "sound": False,
        "note": "64B checkpoint specialized for high-fidelity text-to-image.",
    },
    "Cosmos3-Super-Text2Image-nf4": {
        "repo_id": "SanDiegoDude/Cosmos3-Super-Text2Image-nf4",
        "params": "64B",
        "modes": ("t2i",),
        "sound": False,
        "prequantized": "nf4",
        "note": "Pre-quantized NF4 (bnb 4-bit) 64B text-to-image. ~37 GB VRAM, loads fast, no on-the-fly quant. Needs bitsandbytes.",
    },
    "Cosmos3-Super-Image2Video": {
        "repo_id": "nvidia/Cosmos3-Super-Image2Video",
        "params": "64B",
        "modes": ("i2v",),
        "sound": True,
        "note": "64B checkpoint specialized for temporally coherent image-to-video.",
    },
}


def model_keys():
    """Dropdown order for the loader node: registry models + local-only dirs.

    Local-only entries are folders under ``ComfyUI/models/Cosmos3`` that hold a
    ``model_index.json`` but aren't in the registry — e.g. pre-quantized models
    written by ``quantize_save.py``. They show up after a ComfyUI restart.
    """
    keys = list(GENERATOR_MODELS.keys())
    for k in sorted(discover_local_models().keys()):
        if k not in keys:
            keys.append(k)
    return keys


def discover_local_models(models_subdir=None):
    """Map of ``key -> synthesized info`` for local model dirs not in the registry."""
    root = os.path.join(get_models_root(), models_subdir or DEFAULT_MODELS_SUBDIR)
    found = {}
    if not os.path.isdir(root):
        return found
    for name in os.listdir(root):
        if name in GENERATOR_MODELS:
            continue
        if os.path.exists(os.path.join(root, name, "model_index.json")):
            found[name] = _synthesize_local_info(name)
    return found


def _synthesize_local_info(model_key):
    """Build an info dict for a local-only model, inheriting from a registry base.

    A pre-quantized dir is conventionally named ``<base>-<quant>`` (e.g.
    ``Cosmos3-Nano-nf4``); we inherit modes/sound from the longest registry key
    that prefixes it, defaulting to the full omni mode set.
    """
    base = None
    for k in sorted(GENERATOR_MODELS, key=len, reverse=True):
        if model_key.startswith(k):
            base = GENERATOR_MODELS[k]
            break
    return {
        "repo_id": None,  # local only; nothing to download
        "params": (base or {}).get("params", "?"),
        "modes": (base or {}).get("modes", ("t2i", "t2v", "i2v")),
        "sound": (base or {}).get("sound", False),
        "local_only": True,
        "note": f"Local model directory '{model_key}'.",
    }


def model_info(model_key):
    if model_key in GENERATOR_MODELS:
        return GENERATOR_MODELS[model_key]
    local = discover_local_models()
    if model_key in local:
        return local[model_key]
    raise KeyError(
        f"Unknown Cosmos 3 model '{model_key}'. Known: {list(GENERATOR_MODELS)} "
        f"+ local {list(local)}"
    )


def transformer_is_prequantized(model_dir):
    """True if the saved transformer already carries a quantization_config.

    Such a model must be loaded *without* passing a new quantization_config —
    diffusers reapplies the saved scheme and skips a (slow) re-quant pass.
    """
    cfg_path = os.path.join(model_dir, "transformer", "config.json")
    if not os.path.exists(cfg_path):
        return False
    try:
        import json

        with open(cfg_path) as fh:
            return "quantization_config" in json.load(fh)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Resolution helpers
#
# Cosmos 3 supports three resolution tiers and a handful of aspect ratios. The
# documented 16:9 sizes per tier are the anchor; other aspect ratios are
# derived to keep a similar pixel budget, rounded to a multiple of 16.
# ---------------------------------------------------------------------------
RESOLUTION_TIERS = ("256p", "480p", "720p")
ASPECT_RATIOS = ("16:9", "4:3", "1:1", "3:4", "9:16")

DEFAULT_TIER = "480p"
DEFAULT_ASPECT = "16:9"

# Documented 16:9 sizes (width, height) per tier (see model card "Vision
# conditioning" / "Resolution tiers").
_TIER_BASE_16_9 = {
    "256p": (320, 192),
    "480p": (832, 480),
    "720p": (1280, 720),
}

_ASPECT_WH = {
    "16:9": (16, 9),
    "4:3": (4, 3),
    "1:1": (1, 1),
    "3:4": (3, 4),
    "9:16": (9, 16),
}


def round_to(value, multiple=16, minimum=16):
    return max(minimum, int(round(value / multiple) * multiple))


def resolution_for(tier=DEFAULT_TIER, aspect=DEFAULT_ASPECT):
    """Return ``(width, height)`` for a resolution tier + aspect ratio.

    The 16:9 sizes match the model card exactly; other ratios preserve a
    similar pixel area and snap to multiples of 16.
    """
    bw, bh = _TIER_BASE_16_9.get(tier, _TIER_BASE_16_9[DEFAULT_TIER])
    if aspect == "16:9":
        return bw, bh
    aw, ah = _ASPECT_WH.get(aspect, _ASPECT_WH[DEFAULT_ASPECT])
    area = float(bw * bh)
    ratio = aw / ah
    height = math.sqrt(area / ratio)
    width = height * ratio
    return round_to(width), round_to(height)


# ---------------------------------------------------------------------------
# Model directory resolution
# ---------------------------------------------------------------------------
def get_models_root():
    """Return ``ComfyUI/models`` (falling back to a sibling guess off-tree)."""
    try:
        import folder_paths

        return folder_paths.models_dir
    except Exception:
        # custom_nodes/<pkg> -> ComfyUI/models
        comfy_root = os.path.dirname(os.path.dirname(PACKAGE_ROOT))
        return os.path.join(comfy_root, "models")


def resolve_model_dir(model_key, models_subdir=None):
    """Absolute directory that holds (or will hold) ``model_key``'s weights.

    Layout: ``<ComfyUI/models>/<models_subdir>/<model_key>``.
    """
    root = get_models_root()
    sub = models_subdir or DEFAULT_MODELS_SUBDIR
    return os.path.join(root, sub, model_key)


def is_model_present(model_key, models_subdir=None):
    """Heuristic completeness check: a Diffusers ``model_index.json`` exists."""
    target = resolve_model_dir(model_key, models_subdir)
    return os.path.exists(os.path.join(target, "model_index.json"))
