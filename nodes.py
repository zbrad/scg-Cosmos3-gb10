"""ComfyUI nodes for NVIDIA Cosmos 3 — the generator (image / video) surface.

A thin wrapper around the Diffusers ``Cosmos3OmniPipeline``. A single loader
node picks the checkpoint (dropdown) and downloads it on demand; the generation
nodes (T2I / T2V / I2V) consume the loaded pipe.

We intentionally do not wrap the *reasoner* (text-out VLM / "LLM") surface or
the ``Cosmos3-Nano-Policy-DROID`` robot policy — see the README.
"""

import torch

from .cosmos3_wrapper import paths as _paths
from .cosmos3_wrapper import loader as _loader
from .cosmos3_wrapper import inference as _inf

CATEGORY = "scg-Cosmos3"

_SEED_MAX = 0xFFFFFFFFFFFFFFFF


def _placeholder_image():
    return torch.zeros((1, 64, 64, 3), dtype=torch.float32)


def _resolve_size(resolution_preset, aspect_ratio, width, height):
    """A non-'custom' preset overrides the explicit width/height fields."""
    if resolution_preset != "custom":
        return _paths.resolution_for(resolution_preset, aspect_ratio)
    return _paths.round_to(width), _paths.round_to(height)


class Cosmos3ModelLoader:
    """Load a Cosmos 3 generator checkpoint with VRAM/offload options.

    On first use, if the chosen model is missing and ``auto_download`` is on, the
    full Diffusers repo is fetched into ``ComfyUI/models/Cosmos3/<model>``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_paths.model_keys(), {"default": _paths.model_keys()[0]}),
                "precision": (["bf16", "fp16"], {"default": "bf16"}),
                "quantization": (
                    list(_loader.QUANTIZATION_CHOICES),
                    {
                        "default": "none",
                        "tooltip": "Quantize the (large) transformer only. fp8 ~half VRAM (near-lossless), "
                        "nf4 ~quarter VRAM (some quality loss), int8 ~half. Needed to fit the 64B Super "
                        "on a single GPU. 'none' keeps full precision.",
                    },
                ),
                "cpu_offload": (
                    ["none", "model", "sequential"],
                    {
                        "default": "none",
                        "tooltip": "VRAM saver. 'model' swaps whole submodules to GPU on demand; "
                        "'sequential' is lowest-VRAM but slowest. The 64B Super checkpoints "
                        "need offload and/or multiple GPUs.",
                    },
                ),
                "auto_download": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "If the checkpoint is missing, download it on load."},
                ),
                "disable_guardrails": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Best-effort: skip Cosmos safety guardrails (prompt screen / face blur)."},
                ),
                "attention_backend": (
                    list(_loader.ATTENTION_BACKEND_CHOICES),
                    {
                        "default": "auto",
                        "tooltip": "'auto' uses the detected GPU's tuned profile if one exists "
                        "(see tuned/devices/*.conf), otherwise leaves the pipeline default "
                        "(native/SDPA) alone. 'flash' needs a working flash_attn install.",
                    },
                ),
                "torch_compile": (
                    list(_loader.TORCH_COMPILE_CHOICES),
                    {
                        "default": "auto",
                        "tooltip": "'auto' follows the detected GPU's tuned profile. 'on' always "
                        "compiles the transformer (mode='reduce-overhead') -- real one-time "
                        "compile cost on first generation with this loaded pipe, faster steady-"
                        "state after. 'off' never compiles even if the tuned profile recommends it.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("COSMOS3_PIPE",)
    RETURN_NAMES = ("cosmos3_pipe",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(
        self,
        model,
        precision,
        quantization,
        cpu_offload,
        auto_download,
        disable_guardrails,
        attention_backend,
        torch_compile,
    ):
        wrapper = _loader.load_cosmos3_pipeline(
            model_key=model,
            precision=precision,
            quantization=quantization,
            cpu_offload=cpu_offload,
            disable_guardrails=disable_guardrails,
            auto_download=auto_download,
            attention_backend=attention_backend,
            torch_compile=torch_compile,
        )
        return (wrapper,)


class Cosmos3TextToImage:
    """Generate a single image from a text prompt (Cosmos 3 generator, num_frames=1)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cosmos3_pipe": ("COSMOS3_PIPE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "blurry, distorted, low quality"}),
                "resolution_preset": (
                    ("custom",) + _paths.RESOLUTION_TIERS,
                    {"default": _paths.DEFAULT_TIER},
                ),
                "aspect_ratio": (_paths.ASPECT_RATIOS, {"default": _paths.DEFAULT_ASPECT}),
                "width": ("INT", {"default": 832, "min": 128, "max": 2048, "step": 16, "tooltip": "Used only when resolution_preset='custom'."}),
                "height": ("INT", {"default": 480, "min": 128, "max": 2048, "step": 16, "tooltip": "Used only when resolution_preset='custom'."}),
                "steps": ("INT", {"default": 35, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": _SEED_MAX}),
                "max_sequence_length": ("INT", {"default": 512, "min": 64, "max": 2048}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, cosmos3_pipe, prompt, negative_prompt, resolution_preset, aspect_ratio,
                 width, height, steps, guidance_scale, seed, max_sequence_length):
        w, h = _resolve_size(resolution_preset, aspect_ratio, width, height)
        image_out, _ = _inf.run_generation(
            cosmos3_pipe,
            mode="t2i",
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=w,
            height=h,
            num_frames=1,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            max_sequence_length=max_sequence_length,
        )
        if image_out is None:
            image_out = _placeholder_image()
        return (image_out,)


class Cosmos3TextToVideo:
    """Generate a video (frame stack) from a text prompt, with optional sound."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cosmos3_pipe": ("COSMOS3_PIPE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "blurry, distorted, low quality"}),
                "resolution_preset": (
                    ("custom",) + _paths.RESOLUTION_TIERS,
                    {"default": _paths.DEFAULT_TIER},
                ),
                "aspect_ratio": (_paths.ASPECT_RATIOS, {"default": _paths.DEFAULT_ASPECT}),
                "width": ("INT", {"default": 1280, "min": 128, "max": 2048, "step": 16, "tooltip": "Used only when resolution_preset='custom'."}),
                "height": ("INT", {"default": 720, "min": 128, "max": 2048, "step": 16, "tooltip": "Used only when resolution_preset='custom'."}),
                "num_frames": ("INT", {"default": 81, "min": 5, "max": 300, "tooltip": "5-300 frames. 189 @ 24fps is ~7.9s."}),
                "fps": ("INT", {"default": 24, "min": 10, "max": 30}),
                "steps": ("INT", {"default": 35, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": _SEED_MAX}),
                "max_sequence_length": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "generate_sound": ("BOOLEAN", {"default": True, "tooltip": "Produce a synchronized soundtrack (only on sound-capable checkpoints; ignored otherwise)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("frames", "audio")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, cosmos3_pipe, prompt, negative_prompt, resolution_preset, aspect_ratio,
                 width, height, num_frames, fps, steps, guidance_scale, seed,
                 max_sequence_length, generate_sound):
        w, h = _resolve_size(resolution_preset, aspect_ratio, width, height)
        image_out, audio_out = _inf.run_generation(
            cosmos3_pipe,
            mode="t2v",
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=w,
            height=h,
            num_frames=num_frames,
            fps=fps,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            max_sequence_length=max_sequence_length,
            generate_sound=generate_sound,
        )
        if image_out is None:
            image_out = _placeholder_image()
        return (image_out, audio_out)


class Cosmos3ImageToVideo:
    """Generate a video (frame stack) conditioned on an input image + prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cosmos3_pipe": ("COSMOS3_PIPE",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "blurry, distorted, low quality"}),
                "resolution_preset": (
                    ("custom",) + _paths.RESOLUTION_TIERS,
                    {"default": _paths.DEFAULT_TIER},
                ),
                "aspect_ratio": (_paths.ASPECT_RATIOS, {"default": _paths.DEFAULT_ASPECT}),
                "width": ("INT", {"default": 1280, "min": 128, "max": 2048, "step": 16, "tooltip": "Used only when resolution_preset='custom'."}),
                "height": ("INT", {"default": 720, "min": 128, "max": 2048, "step": 16, "tooltip": "Used only when resolution_preset='custom'."}),
                "num_frames": ("INT", {"default": 81, "min": 5, "max": 300}),
                "fps": ("INT", {"default": 24, "min": 10, "max": 30}),
                "steps": ("INT", {"default": 35, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": _SEED_MAX}),
                "max_sequence_length": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "generate_sound": ("BOOLEAN", {"default": True, "tooltip": "Produce a synchronized soundtrack (only on sound-capable checkpoints; ignored otherwise)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("frames", "audio")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, cosmos3_pipe, image, prompt, negative_prompt, resolution_preset, aspect_ratio,
                 width, height, num_frames, fps, steps, guidance_scale, seed,
                 max_sequence_length, generate_sound):
        w, h = _resolve_size(resolution_preset, aspect_ratio, width, height)
        image_out, audio_out = _inf.run_generation(
            cosmos3_pipe,
            mode="i2v",
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=w,
            height=h,
            num_frames=num_frames,
            fps=fps,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            max_sequence_length=max_sequence_length,
            image=image,
            generate_sound=generate_sound,
        )
        if image_out is None:
            image_out = _placeholder_image()
        return (image_out, audio_out)


NODE_CLASS_MAPPINGS = {
    "Cosmos3ModelLoader": Cosmos3ModelLoader,
    "Cosmos3TextToImage": Cosmos3TextToImage,
    "Cosmos3TextToVideo": Cosmos3TextToVideo,
    "Cosmos3ImageToVideo": Cosmos3ImageToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Cosmos3ModelLoader": "Cosmos3 Model Loader",
    "Cosmos3TextToImage": "Cosmos3 Text-to-Image",
    "Cosmos3TextToVideo": "Cosmos3 Text-to-Video",
    "Cosmos3ImageToVideo": "Cosmos3 Image-to-Video",
}
