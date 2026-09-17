"""Build and configure the Cosmos 3 ``Cosmos3OmniPipeline`` for inference.

Public entry point: :func:`load_cosmos3_pipeline`.

Cosmos 3 ships its generator path through HuggingFace Diffusers as
``Cosmos3OmniPipeline`` (see the model card Quickstart). That class is only
present in a recent ``diffusers`` build, so we import it lazily and raise a
clear, actionable error if it's missing.
"""

import json
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

# Quantization choices exposed on the loader node. "none" keeps bf16/fp16.
QUANTIZATION_CHOICES = ("none", "fp8", "int8", "nf4")

# Which config gate-flags create submodules that need real backing weights,
# and the parameter-name prefixes that prove those weights actually exist
# in a saved checkpoint. A flag can be True in a checkpoint's saved config
# while the checkpoint itself predates that submodule (or the quantization
# script simply didn't save it) -- loading it as-is leaves those specific
# parameters on the `meta` device with no data, and `pipe.to(device)` later
# dies with `NotImplementedError: Cannot copy out of meta tensor; no data!`.
# Hit in practice with Cosmos3-Nano-nf4 (config `action_gen=True`, quantized
# against diffusers 0.39.0.dev0 -- no `action_proj_*` weights in the saved
# checkpoint).
_GATE_FLAG_REQUIRED_KEY_PREFIXES = {
    "action_gen": ("action_proj_in.", "action_proj_out."),
    "sound_gen": ("audio_proj_in.", "audio_proj_out."),
}


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


def _collect_safetensors_keys(component_dir):
    """Parameter keys actually saved for a component, or None if introspection fails.

    Reads a sharded component's ``*.safetensors.index.json`` weight map, or a
    single-file component's safetensors header directly -- never loads the
    tensor data itself, just the key names.
    """
    index_path = os.path.join(component_dir, "diffusion_pytorch_model.safetensors.index.json")
    if os.path.isfile(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            return set(index.get("weight_map", {}).keys())
        except (OSError, ValueError):
            return None

    single_path = os.path.join(component_dir, "diffusion_pytorch_model.safetensors")
    if os.path.isfile(single_path):
        try:
            from safetensors import safe_open

            with safe_open(single_path, framework="pt") as f:
                return set(f.keys())
        except Exception:
            return None

    return None


def _load_transformer_with_gate_reconciliation(model_dir, dtype, quant_config, log):
    """Load ``Cosmos3OmniTransformer``, disabling any gate-flag whose weights are missing.

    Returns the constructed transformer to hand to the pipeline's
    ``from_pretrained(..., transformer=<this>)`` (the standard diffusers
    pattern for overriding one component while the rest of the pipeline
    still auto-loads normally), or ``None`` if the config already matches
    its saved weights -- in which case the caller should let the pipeline
    load the transformer the normal way, unmodified.
    """
    transformer_dir = os.path.join(model_dir, "transformer")
    config_path = os.path.join(transformer_dir, "config.json")
    if not os.path.isfile(config_path):
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, ValueError):
        return None

    saved_keys = _collect_safetensors_keys(transformer_dir)
    if saved_keys is None:
        return None  # can't introspect the weights; don't second-guess the config

    overrides = {}
    for flag, required_prefixes in _GATE_FLAG_REQUIRED_KEY_PREFIXES.items():
        if config.get(flag) and not any(
            any(k.startswith(p) for k in saved_keys) for p in required_prefixes
        ):
            overrides[flag] = False

    if not overrides:
        return None

    for flag, new_value in overrides.items():
        log(f"transformer config has '{flag}={config.get(flag)}' but no matching "
            f"weights in the saved checkpoint -- overriding to {new_value} "
            "(otherwise those params stay on the meta device and pipe.to(device) "
            "dies later). Likely quantized against an older diffusers build than "
            "this checkpoint's config now expects.")

    from diffusers import Cosmos3OmniTransformer

    kwargs = {"torch_dtype": dtype}
    if quant_config is not None:
        tf_quant = quant_config.quant_mapping.get("transformer")
        if tf_quant is not None:
            kwargs["quantization_config"] = tf_quant
    return Cosmos3OmniTransformer.from_pretrained(transformer_dir, **overrides, **kwargs)


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
):
    """Construct the Cosmos 3 generator pipeline and apply runtime options.

    If the checkpoint is missing and ``auto_download`` is set, it's fetched from
    HuggingFace first (first run only; resumable). Returns a
    :class:`Cosmos3PipeWrapper`.
    """
    info = model_info(model_key)

    def log(msg):
        if verbose:
            print(f"[scg-Cosmos3] {msg}")

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

    reconciled_transformer = _load_transformer_with_gate_reconciliation(
        model_dir, dtype, quant_config, log
    )

    from_kwargs = {"torch_dtype": dtype, "enable_safety_checker": enable_safety}
    if reconciled_transformer is not None:
        # Already built (with quantization applied, if any) by the gate-
        # reconciliation helper above -- passing quantization_config here too
        # would be redundant (diffusers skips loading/quantizing any
        # component supplied as a direct instance) and this repo only ever
        # quantizes the transformer, so there's nothing else for it to apply to.
        from_kwargs["transformer"] = reconciled_transformer
    elif quant_config is not None:
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
        },
    )
