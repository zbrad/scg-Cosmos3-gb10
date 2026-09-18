# scg-Cosmos3-gb10

ComfyUI wrapper nodes for **NVIDIA Cosmos 3** — the open *omnimodal world model*
family ([NVIDIA/cosmos](https://github.com/NVIDIA/cosmos)). These nodes wrap the
**generator** surface (text-to-image, text-to-video, image-to-video, with
optional synchronized sound) via the HuggingFace Diffusers `Cosmos3OmniPipeline`.

This is a fork of [`SanDiegoDude/scg-Cosmos3`](https://github.com/SanDiegoDude/scg-Cosmos3)
that adds a validated **GB10 (NVIDIA DGX Spark) runtime-tuning profile** — see
[GB10-tuned profile](#gb10-tuned-profile) below. **If you're not on GB10
hardware, use the upstream repo instead** — the tuning profile only activates
on a detected GB10 device (falls back to stock behavior otherwise, but there's
no reason to carry the fork if it'll never engage for you).

`main` is kept in sync with upstream automatically
([`.github/workflows/sync-upstream.yml`](.github/workflows/sync-upstream.yml),
weekly + on-demand): a clean merge is pushed straight to `main`; a conflicting
one instead files a GitHub issue for manual resolution rather than pushing
anything broken. `main` is always merge-based, never rebased, so `git pull`
here never needs a force-push.

---

## ⚠️ Read this first: these models are huge

Cosmos 3 generators are **16B – 64B** parameter models. This is **not** a pack
you can run on a small/old consumer GPU. Approximate VRAM just to hold the
transformer (the VAE/tokenizers add a few GB more):

| Model | Params | bf16 | NF4 (4-bit) | Realistic minimum |
|---|---|---|---|---|
| `Cosmos3-Nano` | 16B | ~32 GB | **~11 GB** | 16 GB GPU (NF4) |
| `Cosmos3-Super-Text2Image` | 64B | ~128 GB | **~37 GB** | 48 GB GPU (NF4) |
| `Cosmos3-Super-Image2Video` | 64B | ~128 GB | ~37 GB | 48 GB GPU (NF4) |
| `Cosmos3-Super` (omni) | 64B | ~128 GB | ~37 GB | 48 GB GPU (NF4) / multi-GPU bf16 |

**Entry point: `Cosmos3-Nano` in NF4 (~11 GB).** The 64B checkpoints in bf16
realistically need a datacenter GPU (or multi-GPU / heavy CPU offload); even in
NF4 they want ~40 GB. If you're on a 8–12 GB card, only NF4 Nano is viable, and
only at low resolution / few frames. Don't expect a 64B model to run on a
1070 Ti — it won't.

These NF4 numbers are weight/memory savings, **not** speed-ups; generation
throughput is roughly unchanged from bf16.

---

## Pre-quantized NF4 weights (recommended path)

On-the-fly quantization re-reads the full bf16 shards and re-quantizes **every
load** (minutes — ~13 min for the 64B). To skip that, the loader offers
ready-made NF4 checkpoints that download pre-quantized and load in ~1–2 min:

- [`SanDiegoDude/Cosmos3-Nano-nf4`](https://huggingface.co/SanDiegoDude/Cosmos3-Nano-nf4) — 16B omni, ~11 GB
- [`SanDiegoDude/Cosmos3-Super-Text2Image-nf4`](https://huggingface.co/SanDiegoDude/Cosmos3-Super-Text2Image-nf4) — 64B T2I, ~37 GB

Pick the `…-nf4` entries in the loader dropdown and they auto-download from the
Hub on first use. (NF4 reload needs `bitsandbytes` installed.)

---

## Capabilities

Cosmos 3 is a single Mixture-of-Transformers model with two runtime surfaces.
These nodes wrap the **generator** only:

- **Text-to-Image** — prompt → image.
- **Text-to-Video** — prompt → frame stack (+ optional sound on omni models).
- **Image-to-Video** — image + prompt → frame stack (+ optional sound).

The **Reasoner** surface (text/vision → text, a Qwen3-VL-style VLM) is **not**
wrapped. The robotics **`…-Policy-DROID`** checkpoint is a vision-language-action
policy that outputs robot joint/gripper actions — it renders no pixels and is
out of scope for these media nodes.

### Models in the loader dropdown

| Model | Size | Modes | Notes |
|---|---|---|---|
| `Cosmos3-Nano` | 16B | T2I, T2V, I2V (+sound) | Omnimodal, lightest, the default. |
| `Cosmos3-Nano-nf4` | 16B | T2I, T2V, I2V (+sound) | Pre-quantized NF4 download (~11 GB). |
| `Cosmos3-Super` | 64B | T2I, T2V, I2V (+sound) | Frontier omnimodal; multi-GPU / heavy offload in bf16. |
| `Cosmos3-Super-Text2Image` | 64B | T2I | Specialized high-fidelity text-to-image. |
| `Cosmos3-Super-Text2Image-nf4` | 64B | T2I | Pre-quantized NF4 download (~37 GB). |
| `Cosmos3-Super-Image2Video` | 64B | I2V (+sound) | Specialized image-to-video. |

Any `…-nf4` folder placed in `ComfyUI/models/Cosmos3/` (including ones you make
yourself) is auto-detected and added to the dropdown after a restart.

## Nodes

- **Cosmos3 Model Loader** — dropdown of checkpoints + `precision` (bf16/fp16),
  `quantization` (none/fp8/int8/nf4), `cpu_offload` (none/model/sequential),
  `auto_download`, `disable_guardrails`. Outputs `COSMOS3_PIPE`. Auto-downloads
  into `ComfyUI/models/Cosmos3/<model>` on first use. Pre-quantized checkpoints
  are detected automatically and the `quantization` control is ignored for them.
- **Cosmos3 Text-to-Image** — prompt → `IMAGE`.
- **Cosmos3 Text-to-Video** — prompt → `IMAGE` (frame stack) + `AUDIO`.
- **Cosmos3 Image-to-Video** — `IMAGE` + prompt → `IMAGE` (frame stack) + `AUDIO`.

Resolution is set via `resolution_preset` (256p / 480p / 720p, matching the
model-card tiers) plus `aspect_ratio` (16:9, 4:3, 1:1, 3:4, 9:16). Set the
preset to `custom` to use explicit `width`/`height`.

**Sound** is produced only on sound-capable (omni) checkpoints, only in video
modes, and only when `generate_sound` is on (default on). Connect the node's
`audio` output to a video-mux / save-audio node. Image-only checkpoints (e.g.
`Super-Text2Image`) silently ignore the sound toggle.

## Install

```bash
cd <ComfyUI>/custom_nodes
git clone https://github.com/zbrad/scg-Cosmos3-gb10
cd scg-Cosmos3-gb10
pip install -r requirements.txt
```

(`main` carries the full GB10-tuned work — unlike the compiled repos in the
same tuned-builds fleet, this one has no gencode/arch flags that could be
wrong for someone else's hardware, and the tuning profile is a verified
no-op on any GPU without a matching `tuned/devices/*.conf` entry, so there's
no reason to keep it on a separate branch here. `tuned-builds` still exists
and stays fast-forwarded to `main` for fleet-naming consistency, but you
don't need to reference it.)

(Named `scg-Cosmos3-gb10` deliberately, not `scg-Cosmos3` — so it can sit
alongside an upstream `scg-Cosmos3` checkout without a folder collision, and
so it's unambiguous which one you have installed. ComfyUI's node registry
doesn't care about the folder name either way; the node IDs are identical to
upstream's.)

### System dependency (GB10 only)

The tuned torch wheel this profile is validated against needs one apt
package to import at all on a fresh host — install it before `pip install`:

```bash
sudo apt install -y libopenblas0-pthread
```

That's the only OS-level package this repo itself needs. The harder
dependency-pinning problem — keeping the tuned `torch`/`flash_attn` wheels
from being silently overwritten by a plain PyPI build when some other
package's dependency resolution runs — isn't solved by an apt/distro
package; it's solved by a `constraints.txt` + `pip.conf` pin on the venv
(the pattern this pins against is `gpu_tuned_protect_torch_pin` in
[`zbrad/tuned-common`](https://github.com/zbrad/tuned-common)). Whether to
go further and publish an actual `.deb` bundling the tuned wheels + this
pinning setup for one-command fresh-host provisioning is an open question,
not yet done — it would trade a new artifact to build/publish on every
torch bump against a nicer one-line install.

## Gotchas

- **Bleeding-edge diffusers required.** `Cosmos3OmniPipeline` is **not** in any
  PyPI release yet — it lives only in diffusers git `main`. `requirements.txt`
  pins a verified commit. If you see `diffusers has no Cosmos3OmniPipeline`,
  install from git:

  ```bash
  pip install -U "diffusers @ git+https://github.com/huggingface/diffusers.git"
  ```

  This also bumps `safetensors` to a recent (0.8.x) build — verified compatible
  with the current torch/transformers stack, but be aware it touches a core dep.

- **Linux only, bf16 compute.** The reference Diffusers path runs in bf16 on
  Ampere / Hopper / Blackwell. Developed and tested on an **NVIDIA DGX (GB10,
  Blackwell, aarch64 Linux)** with **ComfyUI v0.23.0**. Not tested on
  Windows/macOS.

- **Quantization is transformer-only.** Only the large `Cosmos3OmniTransformer`
  is quantized; the VAE and tokenizers stay in `dtype`. `nf4`/`int8` use
  `bitsandbytes`; `fp8` uses `optimum-quanto`, which JIT-compiles a CUDA kernel
  via `ninja` (the loader puts the venv bin dir on `PATH` so torch can find it).

- **fp8 quirk (handled).** quanto's marlin fp8 GEMM only accepts bf16/fp16, but
  the timestep embedder runs in fp32 — the loader patches that embedder's input
  dtype automatically so fp8 sampling works.

- **Safety guardrails are optional.** Cosmos ships `cosmos_guardrail`; if it
  isn't installed the loader logs a warning and loads with the safety checker
  **off** (loading still succeeds). Install `cosmos_guardrail` to enable it.

- **bnb 4-bit models don't like `.to(dtype)`.** The wrapper handles device
  placement defensively; if you script against the pipeline directly, don't call
  `.to(some_dtype)` on a quantized model.

## GB10-tuned profile

The **Cosmos3 Model Loader** node has two extra inputs beyond upstream:

- `attention_backend` (`auto`/`native`/`flash`) — `auto` defers to the
  detected GPU's tuned profile (`tuned/devices/*.conf`) if one exists,
  otherwise leaves the pipeline's native/SDPA default alone.
- `torch_compile` (`auto`/`on`/`off`) — same deference pattern; `on` always
  compiles the transformer (`mode="reduce-overhead"`), a real one-time cost
  on the first generation with a freshly-loaded pipe, faster steady-state
  after.

On a detected **NVIDIA GB10** (`tuned/devices/gb10.conf`), both default on
(`auto` → flash attention + `torch.compile`). Validated 2026-09-17 against
`Cosmos3-Nano-nf4` and `Cosmos3-Super-Text2Image-nf4`:

| Model | Steady-state step time | vs. native/SDPA, no compile |
|---|---|---|
| Cosmos3-Nano-nf4 | 41.2s/it | ~18% faster |
| Cosmos3-Super-Text2Image-nf4 | ~164-165s/it | ~13.7% faster |

Any other GPU falls back to today's untouched behavior (no tuned profile =
no change from upstream) — untested hardware never gets a guessed setting.
`cosmos3_wrapper/tuned.py`'s `verify_venv_pinning()` also warns (doesn't
block) at load time if the installed torch/flash_attn don't look like the
tuned build this profile was validated against, so a silent fallback to a
vanilla PyPI wheel doesn't masquerade as the tuned result.

## Roll your own quantized checkpoint

`quantize_save.py` bakes a pre-quantized, self-contained pipeline so you (or
your users) never pay the on-the-fly quant cost:

```bash
cd <ComfyUI>/custom_nodes/scg-Cosmos3-gb10
python quantize_save.py --model Cosmos3-Nano --quant nf4
# optionally publish to the Hub (must be logged in: `hf auth login`):
python quantize_save.py --model Cosmos3-Super --quant nf4 --push <user>/Cosmos3-Super-nf4
```

The result lands in `ComfyUI/models/Cosmos3/<model>-<quant>` and shows up in the
loader after a restart. `nf4`/`int8` (bitsandbytes) are the supported
save+reload paths.

## License

Node code is provided as-is. The **model weights** are governed by NVIDIA's
[OpenMDW 1.1 License](https://openmdw.ai/license/1-1/); the NF4 repackages
inherit it and only change the weight encoding.
