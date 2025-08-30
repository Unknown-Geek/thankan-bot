---
title: Thani Thankan
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🔥 Thani Thankan - The Rough Alter Ego

**Thani Thankan** is the rough, moody alter ego of Thankan Chettan. This AI assistant provides helpful advice and information, but delivers it with aggressive Malayalam slang and a no-nonsense attitude.

## Features

- **Aggressive but Helpful**: Gets the job done while insulting you in creative Malayalam
- **Google Gemma**: Powered by Google's Gemma-2b model for fast responses
- **API Ready**: Optimized for Gradio API calls
- **Memory Efficient**: Uses 4-bit quantization for 16GB RAM environments

## Usage

Simply start a conversation! Thani will respond with helpful information wrapped in his characteristic rough language.

**Example interactions:**

- Ask "Who are you?" for a proper Thani introduction
- Request coding help: "Help me with Python"
- Need motivation: "I'm feeling lazy today"

## Warning

This bot uses aggressive Malayalam slang and can be insulting while being helpful. It's designed for entertainment and should not be taken seriously.

## Technical Details

- **Model**: google/gemma-2b
- **Optimization**: 4-bit quantization for memory efficiency
- **Platform**: Optimized for Hugging Face Spaces (2 vCPU, 16GB RAM)
- **Framework**: Gradio for easy API integration

## API Usage

You can interact with this space programmatically:

```python
from gradio_client import Client

client = Client("Mojo-Maniac/thankan")
result = client.predict(
    message="Who are you?",
    history=[],
    max_tokens=512,
    api_name="/predict"
)
print(result)
```

---

# _Remember: Thani may insult you, but he's ultimately trying to help!_ 😈

title: Thankan
emoji: 📚
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 5.44.1
app_file: app.py
pinned: false

---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

> > > > > > > 438b3c22cfd1823b5b035ad94a31f4f5e4f8776f
