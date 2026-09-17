"""Build and configure the Cosmos 3 ``Cosmos3OmniPipeline`` for inference.

Public entry point: :func:`load_cosmos3_pipeline`.

Cosmos 3 ships its generator path through HuggingFace Diffusers as
``Cosmos3OmniPipeline`` (see the model card Quickstart). That class is only
present in a recent ``diffusers`` build, so we import it lazily and raise a
clear, actionable error if it's missing.
"""

import os
import sys
import time

import torch

from .paths import (
    model_info,
    resolve_model_dir,
    is_model_present,
    transformer_is_prequantized,
)
from .tuned import resolve_tuned_profile

# Quantization choices exposed on the loader node. "none" keeps bf16/fp16.
QUANTIZATION_CHOICES = ("none", "fp8", "int8", "nf4")

# Attention-backend choices exposed on the loader node. "auto" defers to the
# detected GPU's tuned profile (see .tuned) if one exists, otherwise leaves
# the pipeline's own default (native/SDPA) alone.
ATTENTION_BACKEND_CHOICES = ("auto", "native", "flash")

# torch.compile choices. "auto" defers to the tuned profile; "off" always
# skips compiling even if the tuned profile recommends it (useful while
# iterating on a workflow, since compile pays a real one-time cost per
# fresh pipe object).
TORCH_COMPILE_CHOICES = ("auto", "on", "off")


class Cosmos3PipeWrapper:
    """Thin container handed between ComfyUI nodes.

    Holds the live Diffusers pipeline plus the metadata the generation nodes
    need (device, dtype, which checkpoint, supported modes).
    """

    def __init__(self, pipe, model_key, device, dtype, info, offload):
        self.pipe = pipe
        self.model_key = model_key
        self.device = device
        self.dtype = dtype
        self.info = info
        self.offload = offload

    @property
    def modes(self):
        return self.info["modes"]

    @property
    def supports_sound(self):
        return bool(self.info.get("sound", False))

    def supports(self, mode):
        return mode in self.info["modes"]


def _import_pipeline_class():
    """Return ``diffusers.Cosmos3OmniPipeline`` or raise a helpful error."""
    try:
        import diffusers
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "diffusers is not installed. Install the Cosmos 3 deps:\n"
            "  pip install 'diffusers @ git+https://github.com/huggingface/diffusers.git' "
            "accelerate av imageio imageio-ffmpeg transformers"
        ) from exc

    cls = getattr(diffusers, "Cosmos3OmniPipeline", None)
    if cls is None:
        raise ImportError(
            "Your installed 'diffusers' has no Cosmos3OmniPipeline. Cosmos 3 needs a "
            "recent build:\n"
            "  pip install -U 'diffusers @ git+https://github.com/huggingface/diffusers.git'\n"
            f"(found diffusers {getattr(diffusers, '__version__', '?')})"
        )
    return cls


def _apply_offload(pipe, mode, device, log):
    """Apply a Diffusers CPU-offload strategy. ``mode`` in none/model/sequential."""
    if mode == "model" and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
        log("Model CPU offload enabled (whole submodules swapped to GPU on demand).")
        return True
    if mode == "sequential" and hasattr(pipe, "enable_sequential_cpu_offload"):
        pipe.enable_sequential_cpu_offload()
        log("Sequential CPU offload enabled (lowest VRAM, slowest).")
        return True
    return False


def _maybe_disable_guardrails(pipe, disable, log):
    """Best-effort: turn off the built-in safety guardrail if requested.

    Cosmos ships prompt/face guardrails; the exact attribute name varies by
    build, so we try the common ones and silently no-op otherwise.
    """
    if not disable:
        return
    for attr in ("safety_checker", "guardrail", "guardrails"):
        if hasattr(pipe, attr) and getattr(pipe, attr) is not None:
            try:
                setattr(pipe, attr, None)
                log(f"Guardrail '{attr}' disabled.")
            except Exception:
                pass


def _ensure_ninja_on_path():
    """Make the venv's ``ninja`` discoverable for torch C++-extension builds.

    optimum-quanto's fp8 path JIT-compiles a CUDA extension via torch, which
    shells out to ``ninja``. The pip ``ninja`` package installs the binary into
    the venv's bin dir, but that dir isn't on ``PATH`` when ComfyUI launches
    ``venv/bin/python`` without activation — so prepend it.
    """
    bindir = os.path.dirname(sys.executable)
    if bindir and bindir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")


def _patch_fp8_compute_dtype(pipe, dtype, log):
    """Cast the timestep embedder's input to ``dtype`` for quanto fp8.

    The transformer's ``time_embedder`` (a diffusers ``TimestepEmbedding``)
    receives fp32 sinusoidal embeddings from ``time_proj``. Once quantized, its
    ``linear_1`` is a quanto fp8 ``QLinear`` whose marlin GEMM only accepts
    bf16/fp16 — fp32 input raises ``fp8_marlin_gemm only supports bfloat16 and
    float16``. Wrapping the embedder's forward to downcast its input fixes this
    without giving up any quantization coverage (everything downstream is
    already bf16). Idempotent and safe if the layer isn't present.
    """
    transformer = getattr(pipe, "transformer", None)
    embedder = getattr(transformer, "time_embedder", None) if transformer is not None else None
    if embedder is None or getattr(embedder, "_scg_fp8_input_cast", False):
        return

    orig_forward = embedder.forward

    def _casted_forward(sample, *args, **kwargs):
        if hasattr(sample, "dtype") and sample.dtype != dtype:
            sample = sample.to(dtype)
        return orig_forward(sample, *args, **kwargs)

    embedder.forward = _casted_forward
    embedder._scg_fp8_input_cast = True
    log("Patched transformer.time_embedder to feed fp8 marlin a bf16/fp16 input.")


def _build_quant_config(quantization, dtype, log):
    """Return a diffusers ``PipelineQuantizationConfig`` (transformer-only) or None.

    Only the big ``Cosmos3OmniTransformer`` is quantized; the VAE / tokenizers
    stay in ``dtype`` since they're small and quality-sensitive.
    """
    if not quantization or quantization == "none":
        return None

    # Use a per-component `quant_mapping` with diffusers' own config objects.
    # The simpler `quant_backend` path runs a cross-library signature check that
    # rejects quanto (diffusers vs transformers __init__ signatures differ).
    from diffusers import PipelineQuantizationConfig, BitsAndBytesConfig, QuantoConfig

    if quantization == "nf4":
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "int8":
        cfg = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "fp8":
        _ensure_ninja_on_path()
        # NOTE: quanto's marlin fp8 GEMM only accepts bf16/fp16 activations, but
        # the timestep embedder is fed fp32 (sinusoidal `Timesteps` output). We
        # can't exclude it via modules_to_not_convert (diffusers' quanto pass
        # matches leaf child names and recurses before checking, so nested
        # linears still get quantized). Instead we cast that embedder's input to
        # the compute dtype after loading -- see _patch_fp8_compute_dtype().
        cfg = QuantoConfig(weights_dtype="float8")
    else:
        raise ValueError(
            f"Unknown quantization '{quantization}'. Choose from {QUANTIZATION_CHOICES}."
        )

    log(f"Quantizing transformer ({quantization}).")
    return PipelineQuantizationConfig(quant_mapping={"transformer": cfg})


def _apply_tuned_runtime_settings(pipe, attention_backend, torch_compile, torch_compile_mode, log):
    """Apply the resolved attention-backend + torch.compile choices to ``pipe.transformer``.

    ``attention_backend``/``torch_compile`` are the ALREADY-RESOLVED choices
    (not "auto" -- the caller resolves "auto" against the tuned profile
    first), so this function just applies them.
    """
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return

    if attention_backend and attention_backend != "native":
        try:
            transformer.set_attention_backend(attention_backend)
            log(f"Attention backend set to '{attention_backend}'.")
        except Exception as exc:
            log(f"Could not set attention backend '{attention_backend}' ({type(exc).__name__}: {exc}) "
                "-- leaving the pipeline's default in place.")

    if torch_compile:
        pipe.transformer = torch.compile(transformer, mode=torch_compile_mode)
        log(f"transformer wrapped with torch.compile(mode='{torch_compile_mode}') "
            "-- first generation call will pay a real one-time compile cost.")


def load_cosmos3_pipeline(
    model_key,
    models_subdir=None,
    precision="bf16",
    quantization="none",
    cpu_offload="none",
    disable_guardrails=False,
    auto_download=True,
    device=None,
    verbose=True,
    attention_backend="auto",
    torch_compile="auto",
):
    """Construct the Cosmos 3 generator pipeline and apply runtime options.

    If the checkpoint is missing and ``auto_download`` is set, it's fetched from
    HuggingFace first (first run only; resumable). Returns a
    :class:`Cosmos3PipeWrapper`.

    ``attention_backend``/``torch_compile`` (one of ``ATTENTION_BACKEND_CHOICES``/
    ``TORCH_COMPILE_CHOICES`` in this module) default to "auto": defer to the
    detected GPU's tuned profile (``.tuned``) if one exists, otherwise leave
    the pipeline's own defaults alone. An explicit value always overrides the
    tuned profile.
    """
    info = model_info(model_key)
    tuned_profile = resolve_tuned_profile(verbose=verbose)

    resolved_attn_backend = (
        tuned_profile.attn_backend if attention_backend == "auto" else attention_backend
    )
    if torch_compile == "auto":
        resolved_torch_compile = tuned_profile.torch_compile
    else:
        resolved_torch_compile = torch_compile == "on"

    def log(msg):
        if verbose:
            print(f"[scg-Cosmos3-gb10] {msg}")

    if not is_model_present(model_key, models_subdir):
        target = resolve_model_dir(model_key, models_subdir)
        if auto_download and info.get("repo_id"):
            log(
                f"'{model_key}' ({info['params']}) not found locally. "
                "Auto-downloading from HuggingFace (first run only, resumable)..."
            )
            from .download import download_model

            download_model(model_key, models_subdir, verbose=verbose)
        elif not info.get("repo_id"):
            raise FileNotFoundError(
                f"Local-only model '{model_key}' has no files at '{target}'. "
                "It isn't in the download registry, so there's nothing to fetch."
            )
        else:
            raise FileNotFoundError(
                f"Cosmos 3 model '{model_key}' not found at '{target}'. Enable "
                f"'auto_download' on the loader, or fetch it manually:\n"
                f"  huggingface-cli download {info['repo_id']} --local-dir {target}"
            )

    model_dir = resolve_model_dir(model_key, models_subdir)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    PipelineClass = _import_pipeline_class()

    # A pre-quantized checkpoint carries its scheme in the saved config; loading
    # it must NOT pass a fresh quantization_config (diffusers reapplies the saved
    # one and skips the slow on-the-fly quant pass).
    prequantized = transformer_is_prequantized(model_dir)
    if prequantized:
        if quantization != "none":
            log(f"'{model_key}' is already quantized on disk; ignoring quantization='{quantization}'.")
        quant_config = None
        quantization = "prequantized"
    else:
        quant_config = _build_quant_config(quantization, dtype, log)

    t0 = time.time()
    log(f"Loading {model_key} ({info['params']}) from {model_dir} (dtype={dtype})...")
    # Upstream Cosmos3OmniPipeline gates guardrails via an `enable_safety_checker`
    # from_pretrained flag (turning it off also avoids needing `cosmos_guardrail`).
    # Cosmos3OmniPipeline's safety checker needs the `cosmos_guardrail` package
    # and isn't listed in model_index.json, so the pipeline would construct it
    # by default and crash if the package is missing. Enable guardrails only
    # when the user wants them AND the package is importable; otherwise load
    # with them off (a wrapper shouldn't fail to load over an optional checker).
    import importlib.util

    guardrails_available = importlib.util.find_spec("cosmos_guardrail") is not None
    enable_safety = (not disable_guardrails) and guardrails_available
    if not disable_guardrails and not guardrails_available:
        log("cosmos_guardrail not installed -> loading with safety checker OFF. "
            "Run `pip install cosmos_guardrail` to enable it.")

    from_kwargs = {"torch_dtype": dtype, "enable_safety_checker": enable_safety}
    if quant_config is not None:
        from_kwargs["quantization_config"] = quant_config
    try:
        pipe = PipelineClass.from_pretrained(model_dir, **from_kwargs)
    except TypeError:
        # Older/newer signature without that flag — load plain, then best-effort.
        from_kwargs.pop("enable_safety_checker", None)
        pipe = PipelineClass.from_pretrained(model_dir, **from_kwargs)
        _maybe_disable_guardrails(pipe, disable_guardrails or not guardrails_available, log)
    log(f"Pipeline built in {time.time() - t0:.1f}s.")

    if quantization == "fp8":
        _patch_fp8_compute_dtype(pipe, dtype, log)

    offloaded = _apply_offload(pipe, cpu_offload, device, log)
    if not offloaded:
        # Quantized components (bnb especially) dislike being .to()-moved; the
        # weights materialize on-device at load, so tolerate a refusal here.
        try:
            pipe = pipe.to(device)
            log(f"Pipeline moved to {device}.")
        except (ValueError, RuntimeError, NotImplementedError) as exc:
            if quant_config is None and not prequantized:
                raise
            log(f"Skipping .to({device}) for quantized pipeline ({type(exc).__name__}: it places itself).")

    _apply_tuned_runtime_settings(
        pipe, resolved_attn_backend, resolved_torch_compile, tuned_profile.torch_compile_mode, log
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log(f"CUDA mem allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    return Cosmos3PipeWrapper(
        pipe=pipe,
        model_key=model_key,
        device=device,
        dtype=dtype,
        info=info,
        offload={
            "cpu_offload": cpu_offload,
            "disable_guardrails": bool(disable_guardrails),
            "precision": precision,
            "quantization": quantization,
            "attention_backend": resolved_attn_backend,
            "torch_compile": resolved_torch_compile,
            "torch_compile_mode": tuned_profile.torch_compile_mode if resolved_torch_compile else None,
            "tuned_variant": tuned_profile.variant,
        },
    )
