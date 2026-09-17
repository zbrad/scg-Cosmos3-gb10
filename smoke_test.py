"""Standalone smoke test for the Cosmos 3 wrapper — fast iteration outside ComfyUI.

Loads a checkpoint and runs a tiny generation without starting the ComfyUI
server, so a CUDA fault aborts only this process (and gives a clean traceback
when run with CUDA_LAUNCH_BLOCKING=1).

Usage (from the ComfyUI venv):
    cd <ComfyUI>/custom_nodes/scg-Cosmos3-gb10
    CUDA_LAUNCH_BLOCKING=1 <ComfyUI>/venv/bin/python smoke_test.py

Knobs are env vars so we can sweep without editing the file:
    COSMOS3_MODEL     (default Cosmos3-Nano)
    COSMOS3_MODE      t2i | t2v | i2v   (default t2i)
    COSMOS3_OFFLOAD   none | model | sequential (default model)
    COSMOS3_W, COSMOS3_H   (default 832x480)
    COSMOS3_FRAMES    (default 17 for video modes)
    COSMOS3_STEPS     (default 8)
    COSMOS3_PROMPT
    COSMOS3_DOWNLOAD  0/1  (default 1)
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch

from cosmos3_wrapper import loader as _loader
from cosmos3_wrapper import inference as _inf


def main():
    model = os.environ.get("COSMOS3_MODEL", "Cosmos3-Nano")
    mode = os.environ.get("COSMOS3_MODE", "t2i")
    offload = os.environ.get("COSMOS3_OFFLOAD", "model")
    width = int(os.environ.get("COSMOS3_W", "832"))
    height = int(os.environ.get("COSMOS3_H", "480"))
    frames = int(os.environ.get("COSMOS3_FRAMES", "17"))
    steps = int(os.environ.get("COSMOS3_STEPS", "8"))
    download = os.environ.get("COSMOS3_DOWNLOAD", "1") == "1"
    prompt = os.environ.get(
        "COSMOS3_PROMPT",
        "A mobile robot navigates a clean warehouse aisle and stops at a shelf.",
    )

    print(f"[smoke] model={model} mode={mode} offload={offload} "
          f"{width}x{height} frames={frames} steps={steps}")

    wrapper = _loader.load_cosmos3_pipeline(
        model_key=model,
        precision="bf16",
        cpu_offload=offload,
        auto_download=download,
    )

    image = None
    if mode == "i2v":
        # A trivial conditioning image so the path exercises end to end.
        image = torch.rand((1, height, width, 3), dtype=torch.float32)

    print("[smoke] running generation...")
    image_out, audio_out = _inf.run_generation(
        wrapper,
        mode=mode,
        prompt=prompt,
        width=width,
        height=height,
        num_frames=frames,
        steps=steps,
        seed=0,
        image=image,
    )

    img_shape = None if image_out is None else tuple(image_out.shape)
    aud_shape = None if audio_out is None else tuple(audio_out["waveform"].shape)
    print(f"[smoke] DONE. image={img_shape} audio={aud_shape}")


if __name__ == "__main__":
    main()
