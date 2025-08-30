---
title: thankan
app_file: app.py
sdk: gradio
sdk_version: 5.44.1
---
# Llama 3.1 8B Gradio Chat App

A simple streaming chat UI around `meta-llama/Meta-Llama-3.1-8B-Instruct` using Hugging Face `transformers` + Gradio.

## Prerequisites

1. Python 3.10+ recommended.
2. (Optional but recommended) Create and activate a virtual environment.
3. Obtain a Hugging Face access token (read) and accept the model license: https://huggingface.co/settings/tokens
4. Ensure you have sufficient GPU memory (approx 16 GB VRAM for full precision). For limited GPUs, try 8-bit/4-bit quantization (see below) or switch to a smaller model.

## Install

```powershell
# In PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

If you have CUDA and want GPU builds of PyTorch, install from https://pytorch.org/get-started/locally/ instead of the CPU wheel listed.

## Environment Variables

Set your HF token (PowerShell):

```powershell
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxx"
```

Optional overrides:

```powershell
$env:LLAMA_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # or another instruct model
$env:LLAMA_LOAD_8BIT = "1"   # enable 8-bit (requires bitsandbytes & supported GPU)
$env:LLAMA_LOAD_4BIT = "1"   # enable 4-bit (ignored if 8-bit also set)
$env:LLAMA_SYSTEM_PROMPT = "You are a pirate chatbot who always responds in pirate speak!"
```

## Run

```powershell
python app.py
```

Open the printed local URL in your browser.

## Notes

- On first load the model download can be several GB; subsequent runs are faster.
- For CPU-only systems, performance will be slower; consider a smaller model.
- You can warm-load is disabled by default (currently preloads in `main()`; comment out for lazy load).

## Changing the System Prompt

Use the right-side System Prompt textbox in the UI, or export `LLAMA_SYSTEM_PROMPT` beforehand.

## Quantization (Optional)

For 4/8-bit quantization you need `bitsandbytes` and an NVIDIA GPU with recent CUDA. Windows support can be finicky; WSL2 is sometimes easier.

## License

Model subject to Meta license; see the model card. This wrapper code is provided under MIT (add your preferred license if needed).
