"""HuggingFace download helper for Cosmos 3 generator checkpoints.

Each model is a full Diffusers repo; we snapshot the whole thing into its own
directory so ``Cosmos3OmniPipeline.from_pretrained(<dir>)`` works offline after
the first run. Downloads are resumable — HF skips files already on disk.
"""

import os

from .paths import model_info, resolve_model_dir, is_model_present

# Weight shards + configs we need; skip docs/preview assets to keep it lean.
# NOTE: ``*.jinja`` is essential — recent HF tokenizers ship their chat template
# as ``chat_template.jinja`` (not embedded in tokenizer_config.json), and the
# Cosmos3 pipeline calls ``apply_chat_template`` during prompt tokenization.
ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "*.jinja",
    "*.safetensors",
    "*.model",
    "*.bin",
    "*.py",
]

IGNORE_PATTERNS = [
    "*.md",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.mp4",
    "assets/*",
    "examples/*",
]


def download_model(model_key, models_subdir=None, verbose=True):
    """Download (or resume) one Cosmos 3 checkpoint into its model dir.

    Returns the absolute model directory. Safe to re-run.
    """
    info = model_info(model_key)
    repo_id = info["repo_id"]
    target = resolve_model_dir(model_key, models_subdir)
    os.makedirs(target, exist_ok=True)

    if verbose:
        print(
            f"[scg-Cosmos3-tuned] Downloading {repo_id} ({info['params']}) -> {target}"
        )

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        local_dir=target,
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
    )

    if not is_model_present(model_key, models_subdir):
        raise RuntimeError(
            f"Download finished but '{target}' has no model_index.json — the repo "
            "layout may differ from what the loader expects. Check the HF repo."
        )
    if verbose:
        print(f"[scg-Cosmos3-tuned] {model_key} ready at {target}")
    return target
