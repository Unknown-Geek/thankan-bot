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
- **Microsoft DialoGPT-Small**: Powered by Microsoft's DialoGPT-Small model (Speed Optimized)
- **API Ready**: Optimized for Gradio API calls
- **Fast Responses**: No quantization, short context, optimized for quick answers

## Usage

Simply start a conversation! Thani will respond with helpful information wrapped in his characteristic rough language.

**Example interactions:**

- Ask "Who are you?" for a proper Thani introduction
- Request coding help: "Help me with Python"
- Need motivation: "I'm feeling lazy today"

## Warning

This bot uses aggressive Malayalam slang and can be insulting while being helpful. It's designed for entertainment and should not be taken seriously.

## Technical Details

- **Model**: microsoft/DialoGPT-small (Speed Optimized)
- **Optimization**: No quantization for faster inference
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
