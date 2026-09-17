"""scg-Cosmos3-gb10 — ComfyUI wrapper nodes for NVIDIA Cosmos 3 generators.

GB10-tuned fork of https://github.com/SanDiegoDude/scg-Cosmos3 — adds a
per-GPU runtime-tuning profile (attention backend + torch.compile); see
cosmos3_wrapper/tuned.py. No-ops on hardware without a matching profile.

Upstream: https://github.com/NVIDIA/cosmos
Models:   https://huggingface.co/collections (NVIDIA Cosmos 3)
"""

import os
import sys

_PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
