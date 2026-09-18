"""Per-GPU-family runtime-tuning profiles for Cosmos 3 generation.

This repo has nothing to compile (pure Python), unlike the compiled
tuned-build repos in the same fleet (pytorch, flash-attention, ...) — so
"tuned variant" here means a validated set of RUNTIME defaults (attention
backend, ``torch.compile``), not a gencode/arch build flag. Profiles live in
``tuned/devices/<variant>.conf`` (same ``KEY="value"`` shape as the other
repos' device confs, for a human skimming the fleet — nothing here actually
gets ``source``d; this module parses them directly).

Only GPUs with an actual validated profile get non-default behavior. An
unrecognized GPU falls back to the pipeline's own defaults (native/SDPA
attention, no ``torch.compile``) rather than guessing untested settings —
see the "don't assume it transfers" caution in the perf-investigation doc
this was validated against.
"""

import importlib.metadata
import os
import re
from dataclasses import dataclass

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(THIS_DIR)
DEVICES_DIR = os.path.join(PACKAGE_ROOT, "tuned", "devices")

_ASSIGNMENT_RE = re.compile(r'^([A-Z_][A-Z0-9_]*)\s*=\s*"([^"]*)"\s*$')


def _dist_version(dist_name):
    """The installed distribution's version, or None if not installed.

    Reads the real package metadata via ``importlib.metadata`` rather than
    a module's own ``__version__`` attribute -- some packages (flash_attn)
    hardcode ``__version__`` to just the upstream base version, omitting
    the PEP 440 local-version segment (``+gb10.cu133...``) that the wheel's
    actual filename/metadata carries, which would otherwise make every
    tuned-build check here a false positive.
    """
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True)
class TunedProfile:
    """Resolved runtime defaults for one GPU variant (or the no-match case)."""

    variant: str | None
    hw_label: str
    attn_backend: str | None  # None = leave the pipeline's own default alone
    torch_compile: bool
    torch_compile_mode: str
    # The exact torch/flash_attn version this profile was validated against
    # (see the perf-investigation doc referenced in each device conf). None
    # means "no specific version on record" -- venv verification is skipped
    # for that package rather than treated as a mismatch.
    validated_torch_version: str | None = None
    validated_flash_attn_version: str | None = None


_NO_PROFILE = TunedProfile(
    variant=None,
    hw_label="unrecognized GPU (no tuned profile)",
    attn_backend=None,
    torch_compile=False,
    torch_compile_mode="reduce-overhead",
)


def _parse_conf(path):
    """A ``tuned/devices/*.conf``'s ``KEY="value"`` lines -> dict. Comments/blank lines ignored."""
    values = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = _ASSIGNMENT_RE.match(line)
            if m:
                values[m.group(1)] = m.group(2)
    return values


def _detect_device_name():
    """The CUDA device name string (e.g. "NVIDIA GB10"), or None if no CUDA device."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_name(0)
    except Exception:
        return None


def resolve_tuned_profile(verbose=True):
    """Detect the GPU and return its :class:`TunedProfile`.

    Matches by exact device-name substring (``GPU_TUNED_DEVICE_NAME_MATCH``
    in each ``tuned/devices/*.conf``), not compute capability — simpler and
    sufficient while this fleet only distinguishes gb10/rtx40/rtx50 by whole
    card family, not by sub-variants within a family.
    """

    def log(msg):
        if verbose:
            print(f"[scg-Cosmos3-tuned/tuned] {msg}")

    device_name = _detect_device_name()
    if not device_name:
        log("No CUDA device detected -- using untuned defaults.")
        return _NO_PROFILE

    if not os.path.isdir(DEVICES_DIR):
        return _NO_PROFILE

    for fname in sorted(os.listdir(DEVICES_DIR)):
        if not fname.endswith(".conf"):
            continue
        path = os.path.join(DEVICES_DIR, fname)
        conf = _parse_conf(path)
        match = conf.get("GPU_TUNED_DEVICE_NAME_MATCH")
        if not match or match not in device_name:
            continue

        profile = TunedProfile(
            variant=conf.get("GPU_TUNED_VARIANT", fname[: -len(".conf")]),
            hw_label=conf.get("GPU_TUNED_HW_LABEL", device_name),
            attn_backend=conf.get("COSMOS3_ATTN_BACKEND") or None,
            torch_compile=conf.get("COSMOS3_TORCH_COMPILE", "false").lower() == "true",
            torch_compile_mode=conf.get(
                "COSMOS3_TORCH_COMPILE_MODE", "reduce-overhead"
            ),
            validated_torch_version=conf.get("COSMOS3_VALIDATED_TORCH_VERSION") or None,
            validated_flash_attn_version=conf.get(
                "COSMOS3_VALIDATED_FLASH_ATTN_VERSION"
            )
            or None,
        )
        log(
            f"Detected {profile.hw_label} -- tuned profile '{profile.variant}': "
            f"attn_backend={profile.attn_backend!r}, "
            f"torch_compile={profile.torch_compile} (mode={profile.torch_compile_mode!r})."
        )
        for warning in verify_venv_pinning(profile):
            log(f"WARNING: {warning}")
        return profile

    log(
        f"Detected GPU '{device_name}' has no tuned profile in {DEVICES_DIR} -- using untuned defaults."
    )
    return _NO_PROFILE


def verify_venv_pinning(profile):
    """Check the installed torch/flash_attn against what ``profile`` was validated on.

    Returns a list of human-readable warning strings (empty if everything
    checks out). Non-fatal by design -- callers log these and proceed; this
    isn't a full ``gpu_tuned_verify_venv``-style build-time hard gate (there's
    no venv to reject here, just a running process), but it's the same
    underlying worry as this fleet's ``constraints-gb10.txt``/``pip.conf``
    pattern: a persistent-process venv can silently end up on a vanilla
    PyPI/pytorch.org torch instead of the tuned GB10 build (see
    ``tuned_torch_venv_pip_constraint`` project memory), and applying
    ``flash``/``torch.compile`` against an untuned or mismatched build is
    exactly the ABI-break class of failure this fleet has already hit once
    (torch 2.15.0 requiring C++20 broke a flash_attn wheel built against an
    older torch -- see cosmos3_nf4_perf_plan.md).
    """
    warnings = []
    if profile.variant is None:
        return warnings

    torch_version = _dist_version("torch")
    if torch_version is None:
        return [
            "could not determine the installed torch version to verify venv pinning"
        ]

    variant_tag = f"+{profile.variant}."
    if variant_tag not in torch_version:
        warnings.append(
            f"installed torch ({torch_version}) has no '{profile.variant}' build tag -- "
            f"this looks like a vanilla PyPI/pytorch.org build, not the tuned "
            f"{profile.hw_label} wheel this profile was validated against. Applying "
            f"'{profile.attn_backend}' attention backend / torch.compile against an "
            "untuned build is unverified; consider reinstalling the tuned wheel "
            "(see the venv's constraints-gb10.txt / pip.conf if one exists)."
        )
    elif (
        profile.validated_torch_version is not None
        and torch_version != profile.validated_torch_version
    ):
        warnings.append(
            f"installed torch ({torch_version}) is a '{profile.variant}' tuned build, "
            f"but doesn't exactly match the version this profile was validated on "
            f"({profile.validated_torch_version}). Probably fine (routine tuning-vNN "
            "rebuild drift), but the perf numbers in cosmos3_nf4_perf_plan.md were "
            "measured on the validated version specifically."
        )

    if profile.attn_backend == "flash":
        fa_version = _dist_version("flash_attn") or _dist_version("flash-attn")
        if fa_version is None:
            warnings.append(
                "tuned profile wants attention_backend='flash' but flash_attn isn't "
                "installed -- generation will fall back to whatever "
                "set_attention_backend('flash') does when the package is missing "
                "(likely a clear RuntimeError at first attention call, not silent)."
            )
        else:
            if variant_tag not in fa_version:
                warnings.append(
                    f"installed flash_attn ({fa_version}) has no '{profile.variant}' build "
                    "tag -- not the tuned wheel this profile assumes. If it's the plain "
                    "PyPI flash-attn package, GB10/SM_121a support is not guaranteed."
                )
            elif (
                profile.validated_flash_attn_version is not None
                and fa_version != profile.validated_flash_attn_version
            ):
                warnings.append(
                    f"installed flash_attn ({fa_version}) is a '{profile.variant}' tuned "
                    f"build, but doesn't match the validated version "
                    f"({profile.validated_flash_attn_version}). Most likely fine, but "
                    "flash_attn wheels are compiled against one specific torch build "
                    "(see the C++20/ABI-break note in cosmos3_nf4_perf_plan.md) -- if "
                    "generation crashes with an 'undefined symbol' error, this is why."
                )

    return warnings
