---
title: SummarizeHub
emoji: 📝
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.50.0"
app_file: app.py
pinned: false
license: mit
hardware: zero-gpu
---

GPU-backed abstractive summarization demo for [Nexus Forge](https://github.com/askmy-stack/nexus-forge).

## Features

- Default model: **BART** (`facebook/bart-large-cnn`) on GPU via `@spaces.GPU`
- Models: BART, FLAN-T5, T5, Pegasus, extractive
- Strategies: stuff, map_reduce, refine, hierarchical, rag

## Deploy to Hugging Face Spaces

```bash
# 1. Authenticate (one-time)
hf auth login

# 2. Deploy from repo root (creates Space if missing)
export HF_USERNAME=your-hf-username
export HF_SPACE_NAME=summarizehub   # optional, default: summarizehub
./scripts/deploy_hf_space.sh

# Or manually create the Space on huggingface.co/new-space (Gradio SDK, ZeroGPU),
# then upload the `spaces/` directory from this repo.
```

Enable **ZeroGPU** in Space settings for free GPU bursts. The `@spaces.GPU(duration=60)` decorator in `app.py` requests GPU only during inference (60-second allocation window per call).

### Hardware upgrade path

| Tier | Use case | How to enable |
|------|----------|---------------|
| **ZeroGPU** | Demo / bursty traffic | Space Settings → Hardware → ZeroGPU |
| **CPU Basic** | Extractive-only fallback | Default for new Spaces |
| **T4 small** | Sustained abstractive load | Space Settings → Hardware → T4 small |
| **A10G / A100** | Training or heavy batch | Upgrade in Settings or `hf repo create ... --flavor a10g-small` |

`spaces/space.yaml` mirrors README frontmatter (`sdk: gradio`, `app_file: app.py`, `hardware: zero-gpu`).

## Dependencies

`nexus-forge` is not on PyPI yet. The Space installs it from GitHub via `requirements.txt`:

```text
git+https://github.com/askmy-stack/nexus-forge.git@main
```

Hugging Face Spaces supports `git+` URLs in `requirements.txt`.

## Local development

```bash
cd spaces
pip install -r requirements.txt
python app.py
```

For full repo docs see the [main README](https://github.com/askmy-stack/nexus-forge).

## Rate limiting

Public demos are capped at **10 requests per minute per client** (IP / session).
Override with `SPACE_RATE_LIMIT_PER_MINUTE` in the Space secrets or environment.
