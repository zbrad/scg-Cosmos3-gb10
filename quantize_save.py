"""Pre-quantize a Cosmos 3 generator and save it for fast reuse.

On-the-fly quantization re-reads the full bf16 shards and quantizes every load
(slow — minutes). This bakes the quantization once and writes a self-contained
pipeline directory, so future loads just read the smaller (e.g. ~10 GB for nf4)
pre-quantized weights and skip the quant pass entirely. The loader auto-detects
a pre-quantized dir (transformer/config.json carries a quantization_config) and
loads it directly.

Usage (from the ComfyUI venv):
    cd <ComfyUI>/custom_nodes/scg-Cosmos3-gb10
    python quantize_save.py --model Cosmos3-Nano --quant nf4
    # custom output dir:
    python quantize_save.py --model Cosmos3-Super --quant nf4 --out /data/Cosmos3-Super-nf4
    # also upload to the Hub (must be logged in: `hf auth login`):
    python quantize_save.py --model Cosmos3-Nano --quant nf4 --push <user>/Cosmos3-Nano-nf4

Notes:
    * nf4 / int8 (bitsandbytes) are the well-supported save+reload paths.
    * fp8 (quanto) serialization is more experimental; prefer nf4/int8 for
      publishing.
    * Reloading a bnb-quantized model requires bitsandbytes installed.
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from cosmos3_wrapper import loader as _loader
from cosmos3_wrapper import paths as _paths


def main():
    ap = argparse.ArgumentParser(description="Pre-quantize and save a Cosmos 3 model.")
    ap.add_argument("--model", default="Cosmos3-Nano", help="Registry model key to quantize.")
    ap.add_argument("--quant", default="nf4", choices=[q for q in _loader.QUANTIZATION_CHOICES if q != "none"])
    ap.add_argument("--out", default=None, help="Output dir (default: <models>/Cosmos3/<model>-<quant>).")
    ap.add_argument("--push", default=None, help="HuggingFace repo id to upload the result to (optional).")
    ap.add_argument("--private", action="store_true", help="Create the HF repo as private when pushing.")
    args = ap.parse_args()

    out_dir = args.out or _paths.resolve_model_dir(f"{args.model}-{args.quant}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[quantize_save] Loading {args.model} with {args.quant} quantization...")
    wrapper = _loader.load_cosmos3_pipeline(
        model_key=args.model,
        quantization=args.quant,
        cpu_offload="none",
        auto_download=True,
        verbose=True,
    )

    print(f"[quantize_save] Saving pre-quantized pipeline -> {out_dir}")
    wrapper.pipe.save_pretrained(out_dir)
    print("[quantize_save] Saved. Reload it by selecting the new entry in the "
          "Cosmos3 Model Loader (restart ComfyUI so it appears in the dropdown).")

    if args.push:
        print(f"[quantize_save] Uploading {out_dir} -> https://huggingface.co/{args.push}")
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(folder_path=out_dir, repo_id=args.push, repo_type="model")
        print(f"[quantize_save] Done. Add it to GENERATOR_MODELS in paths.py with "
              f"repo_id='{args.push}' to make it downloadable like a built-in model.")


if __name__ == "__main__":
    main()
