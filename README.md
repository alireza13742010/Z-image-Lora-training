# Z-Image Character Consistency Studio

A Streamlit app for testing and comparing a custom-trained character LoRA against the base [Z-Image](https://huggingface.co/Tongyi-MAI) diffusion model — built to visually verify how consistently a fine-tuned character holds up across different seeds and prompts.

## About this project

This app was built while fine-tuning Z-Image on a custom character. The character LoRA was trained on a **small dataset of just 20 reference images**, curated for consistent pose/style/lighting variety rather than sheer volume, then trained as a low-rank adapter on top of the base Z-Image model.

Rather than eyeballing single generations, this tool generates **matched pairs** — the same prompt and seed run once through the base model and once with the character LoRA enabled — side by side, across several seeds in a row. That side-by-side view makes it easy to spot whether the trained character's face, proportions, and style stay consistent or start drifting as seeds change.

## Features

- 🔄 Toggle the LoRA on/off on a single loaded pipeline (no need to reload multi-GB weights to compare)
- 🎚️ Adjustable LoRA strength, resolution, inference steps, and guidance scale
- 🌱 Fixed or random seed sequences, with multiple seeds generated per run for a real consistency check
- 📥 Per-image download buttons, plus a running session history of everything generated
- ⚡ Model stays cached in memory across generations for fast iteration

## Requirements

- **GPU:** an NVIDIA GPU with CUDA support (the pipeline runs in `bfloat16` on `cuda`; this app is not configured for CPU-only inference)
- **VRAM:** enough to hold the Z-Image base weights plus the LoRA adapter in memory — check the model card for the specific Z-Image variant you're using
- **Python:** 3.10 or newer recommended
- **Local weights:** a local download of the Z-Image base model, and your own trained LoRA `.safetensors` file

### Python packages

Install everything with:

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
streamlit>=1.35
torch>=2.3
diffusers>=0.36
accelerate>=0.30
safetensors>=0.4
pillow>=10.0
```

> **Note:** LoRA support in `diffusers`' `ZImagePipeline` is a recent addition. If you hit an `AttributeError` on `enable_lora()` / `disable_lora()` / `set_adapters()`, upgrade to the latest `diffusers` release.

## Setup

1. Download the Z-Image base model locally and note its path.
2. Place your trained character LoRA `.safetensors` file somewhere accessible.
3. Install dependencies (see above).
4. Run the app:

```bash
streamlit run app.py
```

5. In the sidebar:
   - Enter the path to your local Z-Image model and click **Load base model**.
   - Enter the path to your LoRA file and click **Load / reload LoRA**.
6. Enter a prompt, pick your generation settings, and click **Generate comparison**.

## Notes on training

- Dataset size: 20 images
- The relatively small dataset is enough for a consistent single-character LoRA, but results are highly sensitive to dataset quality (consistent character appearance across images, varied poses/angles/lighting, minimal noise) — see the LoRA training discussion in this repo/thread for the full workflow used.

## License

Add your license of choice here (e.g. MIT).
