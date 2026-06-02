"""Run Cosmos 3 generation modes and convert to/from ComfyUI tensors.

The ``Cosmos3OmniPipeline`` is called directly; the differences between modes
are which kwargs we pass (``num_frames``, ``image=`` for I2V, sound toggle) and
how we unpack the result. We filter call kwargs against the live ``__call__``
signature so the wrapper self-heals against upstream API drift, and unpack the
result defensively (``.video`` for visuals, ``.sound``/``.audio`` for sound).
"""

import inspect

import numpy as np
import torch


# ---------------------------------------------------------------------------
# ComfyUI <-> pipeline conversions
# ---------------------------------------------------------------------------
def comfy_image_to_pil(image_bhwc):
    """First frame of a ComfyUI IMAGE batch [B,H,W,C] in [0,1] -> PIL.Image."""
    from PIL import Image

    img = image_bhwc[0]  # [H, W, C]
    arr = (img.clamp(0.0, 1.0).cpu().float().numpy() * 255.0).round().astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return Image.fromarray(arr)


def _frame_to_hwc_float(frame):
    """Normalize a single frame (PIL / np / tensor) to a float [H,W,3] tensor in [0,1]."""
    if hasattr(frame, "convert"):  # PIL.Image
        frame = np.asarray(frame.convert("RGB"))
    if isinstance(frame, np.ndarray):
        t = torch.from_numpy(frame)
    elif isinstance(frame, torch.Tensor):
        t = frame
    else:
        t = torch.as_tensor(np.asarray(frame))

    t = t.detach().cpu().float()

    # Channel-first -> channel-last.
    if t.dim() == 3 and t.shape[0] in (1, 3) and t.shape[-1] not in (1, 3):
        t = t.permute(1, 2, 0)

    if t.dim() == 2:  # grayscale
        t = t.unsqueeze(-1).repeat(1, 1, 3)
    if t.shape[-1] == 1:
        t = t.repeat(1, 1, 3)
    if t.shape[-1] == 4:  # drop alpha
        t = t[..., :3]

    if t.max() > 1.5:  # 0..255 -> 0..1
        t = t / 255.0
    return t.clamp(0.0, 1.0)


def frames_to_comfy_image(frames):
    """A list/array of frames -> ComfyUI IMAGE tensor [T,H,W,3] in [0,1]."""
    if frames is None:
        return None
    if isinstance(frames, torch.Tensor) and frames.dim() == 4:
        return torch.stack([_frame_to_hwc_float(f) for f in frames], dim=0)
    out = [_frame_to_hwc_float(f) for f in frames]
    if not out:
        return None
    return torch.stack(out, dim=0).contiguous()


def _extract_visual(result, want_image):
    """Pull a frame list out of a pipeline result, handling nested batch dims."""
    visual = None
    for attr in ("video", "videos", "frames", "images"):
        v = getattr(result, attr, None)
        if v is None and isinstance(result, dict):
            v = result.get(attr)
        if v is not None:
            visual = v
            break
    if visual is None:
        return None

    # Diffusers often nests one level per prompt: [[frame, frame, ...]].
    if (
        isinstance(visual, (list, tuple))
        and len(visual) > 0
        and isinstance(visual[0], (list, tuple))
    ):
        visual = visual[0]
    elif isinstance(visual, torch.Tensor) and visual.dim() == 5:
        visual = visual[0]

    if want_image:
        # Single image: take the first frame only.
        if isinstance(visual, (list, tuple)):
            return [visual[0]]
        if isinstance(visual, torch.Tensor) and visual.dim() == 4:
            return visual[0:1]
    return visual


def _extract_audio(result):
    """Pull an audio waveform out of a pipeline result, if present.

    Cosmos3OmniPipelineOutput exposes ``sound``; we also accept ``audio``/
    ``audios`` for robustness against naming changes.
    """
    for attr in ("sound", "audio", "audios"):
        a = getattr(result, attr, None)
        if a is None and isinstance(result, dict):
            a = result.get(attr)
        if a is not None:
            if isinstance(a, (list, tuple)) and len(a) > 0:
                return a[0]
            return a
    return None


def audio_to_comfy_audio(audio, sample_rate=48000):
    """Convert a pipeline audio output to a ComfyUI AUDIO dict, or None.

    ComfyUI AUDIO expects ``{"waveform": [B, C, L], "sample_rate": int}``.
    Cosmos 3 sound is stereo AAC @ 48 kHz when present.
    """
    if audio is None:
        return None

    if isinstance(audio, dict):
        sample_rate = int(audio.get("sample_rate", sample_rate))
        wav = audio.get("waveform", audio.get("array"))
    else:
        wav = audio

    if wav is None:
        return None
    if isinstance(wav, np.ndarray):
        wav = torch.from_numpy(wav)
    wav = wav.detach().cpu().float()

    if wav.dim() == 1:  # [L] -> [1, L]
        wav = wav.unsqueeze(0)
    if wav.dim() == 2:  # [C, L] -> [1, C, L]
        wav = wav.unsqueeze(0)
    return {"waveform": wav.contiguous(), "sample_rate": int(sample_rate)}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _build_generator(device, seed):
    gen_device = "cuda" if (device.type == "cuda") else "cpu"
    return torch.Generator(device=gen_device).manual_seed(int(seed))


def _filter_to_signature(fn, kwargs):
    """Drop kwargs the callable doesn't accept (unless it takes ``**kwargs``).

    Keeps the wrapper resilient to upstream signature churn: e.g. Cosmos 3 uses
    ``enable_sound``/``enable_safety_check`` and has no ``max_sequence_length``.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def _make_step_callback(steps):
    """ComfyUI progress bar + interrupt hook, in Diffusers callback shape."""
    try:
        import comfy.utils

        pbar = comfy.utils.ProgressBar(steps)
    except Exception:
        pbar = None

    def _cb(pipe, step_index, timestep, callback_kwargs):
        try:
            import comfy.model_management

            comfy.model_management.throw_exception_if_processing_interrupted()
        except Exception:
            pass
        print(f"[scg-Cosmos3] step {step_index + 1}/{steps}", flush=True)
        if pbar is not None:
            pbar.update_absolute(step_index + 1, steps)
        return callback_kwargs

    return _cb


@torch.no_grad()
def run_generation(
    wrapper,
    mode,
    prompt,
    negative_prompt="",
    width=832,
    height=480,
    num_frames=1,
    fps=24,
    steps=35,
    guidance_scale=7.0,
    seed=0,
    max_sequence_length=512,
    image=None,
    generate_sound=False,
    flow_shift=None,
):
    """Run one of ``t2i`` / ``t2v`` / ``i2v`` and return ``(image_out, audio_out)``.

    ``image_out`` is a ComfyUI IMAGE tensor [T,H,W,3] (T==1 for t2i); ``audio_out``
    is a ComfyUI AUDIO dict or None.
    """
    if not wrapper.supports(mode):
        raise ValueError(
            f"Model '{wrapper.model_key}' does not support mode '{mode}'. "
            f"Supported: {wrapper.modes}"
        )

    pipe = wrapper.pipe
    device = wrapper.device
    want_image = mode == "t2i"
    want_sound = bool(generate_sound) and wrapper.supports_sound and not want_image

    cb = _make_step_callback(int(steps))

    # Superset of every kwarg we might pass; anything the live __call__ doesn't
    # accept is dropped by _filter_to_signature (e.g. max_sequence_length).
    candidate = {
        "prompt": prompt,
        "negative_prompt": negative_prompt or None,
        "height": int(height),
        "width": int(width),
        "num_frames": 1 if want_image else int(num_frames),
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance_scale),
        "generator": _build_generator(device, seed),
        "callback_on_step_end": cb,
        # Cosmos 3 sound toggle (canonical name + a couple of aliases that get
        # filtered out if unsupported).
        "enable_sound": want_sound,
        "generate_sound": want_sound,
        # Some builds also accept a per-prompt token cap; harmless if absent.
        "max_sequence_length": int(max_sequence_length),
    }
    if not want_image:
        candidate["fps"] = float(fps)
    if mode == "i2v":
        if image is None:
            raise ValueError("Image-to-video requires an input image.")
        candidate["image"] = comfy_image_to_pil(image)
    if wrapper.offload.get("disable_guardrails"):
        # Per-call guardrail kwarg (load-time flag is separate); aliases filtered.
        candidate["enable_safety_check"] = False
        candidate["enable_safety_checker"] = False
    if flow_shift is not None:
        candidate["flow_shift"] = float(flow_shift)

    call_kwargs = _filter_to_signature(pipe.__call__, candidate)
    result = pipe(**call_kwargs)

    visual = _extract_visual(result, want_image=want_image)
    image_out = frames_to_comfy_image(visual)

    audio_out = None
    if want_sound:
        audio_out = audio_to_comfy_audio(_extract_audio(result))

    return image_out, audio_out
