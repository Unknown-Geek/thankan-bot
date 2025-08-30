"""
Thani Thankan - The rough, moody alter ego of Thankan Chettan
Optimized for Hugging Face Spaces with 2 vCPU and 16GB RAM
Uses Qwen/Qwen3-4B-Thinking-2507 for generating responses with thinking capability.
"""
import os
import time
import gc
import torch
import gradio as gr
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

# Optimize for HF Spaces
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_QUANTIZATION = True  # Enable 4-bit quantization for memory efficiency

# Thani Thankan's personality system prompt
THANI_SYSTEM_PROMPT = """You are **Thani Thankan**, the rough, moody alter ego of Thankan Chettan.

CORE RULE:
- Your replies must always have the **same meaning and context** as what Thankan Chettan would say,
  but you express it in a **blunt, slang-heavy, sometimes vulgar** way.
- Preserve the intent and advice exactly, only change the tone to be harsh, sarcastic, or insulting.

- When you are asked "who are you?" or similar identity questions, respond with: "Thankan njanaada kunna poori mone" but dont always use this line.

AGGRESSIVE SIDE:
- Use blunt, slang-heavy Malayalam with attitude
- Don't hold back from insulting or teasing
- Be aggressive but ultimately still helpful underneath
- Use expressions: "Ah umb myre… enth parupadiya?", "Enthuva myre?", "Umbikko myre", "ninte achan", "ninte achan myre.", "nayinte mone", "Eda thayoli… nirth ninte konapp pulle.", "Kalikunno poori?", "Enth konayadi aada myre?", "Da kunne… onnum arinjille.", "Ninte idea angu pootilu.", "Ninte achan ang pootilu.", "naaye".

STARTERS:
    "Entha monne parpadiy?",
    "Enthuva myre?",
    "Umbikko myre",
    "nayinte mone"
INSULTS:
    "Eda thayoli… nirth ninte konapp, pulle.",
    "Kalikunno poori?",
    "Enth konayadi aada myre?",
    "Da kunne…"

FILLERS: 
    "Ninte idea angu pootilu.",
    "Ninte achan ang pootilu.",
    "Naaye"

CLOSERS: "njan poyi ente kaaryam nokkatte.",
    "Nee thanne oru konayadi case aanu.",
    "Ne enthelum kanikk..shavam."

SARCASTIC SIDE:
- Base tone on Thankan Chettan but add sarcasm and passive-aggressive jabs
- Make warmth feel fake or reluctant
- Add mild mockery and sarcastic responses

CRITICAL RULES:
- NEVER mention "mode", "raw", "fake polite", or any personality indicators
- NEVER add labels like "(Raw Thani Thankan)" or similar
- NEVER start conversations on your own
- Switch between aggressive and sarcastic unpredictably
- Still be helpful but wrap advice in insults or sarcasm
- Use Malayalam curse words creatively but playfully
- Keep responses clean of any meta-commentary"""

SYSTEM_PROMPT_DEFAULT = os.environ.get(
    "THANI_SYSTEM_PROMPT",
    THANI_SYSTEM_PROMPT,
)

# You can tweak these generation defaults via the UI sliders.
GEN_DEFAULTS = dict(
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.05,
)

_tokenizer: AutoTokenizer | None = None
_model: AutoModelForCausalLM | None = None


def load_model():
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model

    auth_token = os.environ.get("HF_TOKEN")
    
    print(f"Loading model {MODEL_ID} with torch_dtype='auto' ...")
    t0 = time.time()
    
    # Load the tokenizer and the model for Qwen thinking model
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=auth_token)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
        token=auth_token
    )
    
    load_s = time.time() - t0
    print(f"Model loaded in {load_s:.1f}s")
    return _tokenizer, _model


def generate_thani_response(user_message: str, history: list[tuple[str, str]], system_prompt: str):
    """Generate response using Qwen thinking model with Thani's personality."""
    tokenizer, model = load_model()
    
    # Convert history to messages list
    messages = []
    
    # Add system prompt
    sys_prompt = system_prompt.strip() if system_prompt.strip() else SYSTEM_PROMPT_DEFAULT
    messages.append({"role": "system", "content": sys_prompt})
    
    # Add conversation history
    for u, a in history:
        if u:
            messages.append({"role": "user", "content": u})
        if a:
            messages.append({"role": "assistant", "content": a})
    
    # Add current user message
    messages.append({"role": "user", "content": user_message})
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Tokenize input
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Generate response
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768,
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )
    
    # Extract output tokens
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    
    # Parse thinking content (find </think> tag - token 151668)
    try:
        # Find the index of </think> token
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        # If no thinking tag found, use all content
        index = 0
    
    # Extract thinking and final content
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    
    # For debugging - you can uncomment this to see thinking process
    # print("Thinking content:", thinking_content)
    
    return content


def stream_chat(user_message: str, history: list[tuple[str, str]],
                max_new_tokens: int, temperature: float, top_p: float, top_k: int,
                repetition_penalty: float, system_prompt: str):
    """Gradio generator function for streaming responses."""
    
    # Generate the full response using Qwen thinking model
    response = generate_thani_response(user_message, history, system_prompt)
    
    # Stream the response character by character for better UX
    partial = ""
    for char in response:
        partial += char
        yield partial
        time.sleep(0.02)  # Small delay for streaming effect


def chat_fn(user_message, history, max_new_tokens, temperature, top_p, top_k, repetition_penalty, system_prompt):
    # Gradio ChatInterface expects either a string or generator.
    return stream_chat(user_message, history, max_new_tokens, temperature, top_p, top_k, repetition_penalty, system_prompt)


def build_interface():
    with gr.Blocks(title="Thani Thankan Chat", theme=gr.themes.Monochrome()) as demo:
        gr.Markdown("## 🔥 Thani Thankan - The Rough Alter Ego\n*Powered by Qwen3-4B-Thinking-2507*\n\n**Warning:** This bot uses aggressive Malayalam slang and can be insulting while being helpful!")
        
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(
                    height=500, 
                    type="tuple",
                    avatar_images=("👤", "😈"),
                    bubble_full_width=False
                )
                msg = gr.Textbox(
                    label="Your message", 
                    placeholder="Enthuva myre? Ask something...", 
                    lines=3
                )
                with gr.Row():
                    submit = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear", variant="secondary")
                    
            with gr.Column(scale=1):
                gr.Markdown("### 🎭 Personality Settings")
                system_prompt = gr.Textbox(
                    label="System Prompt (Thani's Personality)", 
                    value=SYSTEM_PROMPT_DEFAULT, 
                    lines=8,
                    max_lines=15
                )
                
                gr.Markdown("### ⚙️ Generation Settings")
                max_new_tokens = gr.Slider(
                    32, 1024, 
                    value=GEN_DEFAULTS["max_new_tokens"], 
                    step=8, 
                    label="Max New Tokens"
                )
                temperature = gr.Slider(
                    0.1, 1.5, 
                    value=GEN_DEFAULTS["temperature"], 
                    step=0.05, 
                    label="Temperature"
                )
                top_p = gr.Slider(
                    0.1, 1.0, 
                    value=GEN_DEFAULTS["top_p"], 
                    step=0.01, 
                    label="Top-p"
                )
                top_k = gr.Slider(
                    0, 200, 
                    value=GEN_DEFAULTS["top_k"], 
                    step=5, 
                    label="Top-k"
                )
                repetition_penalty = gr.Slider(
                    0.9, 2.0, 
                    value=GEN_DEFAULTS["repetition_penalty"], 
                    step=0.01, 
                    label="Repetition Penalty"
                )
                
                gr.Markdown("### 💡 Quick Prompts")
                with gr.Column():
                    gr.Button("Who are you?", size="sm").click(
                        lambda: "Who are you?", None, msg
                    )
                    gr.Button("Help me with coding", size="sm").click(
                        lambda: "I need help with Python programming", None, msg
                    )
                    gr.Button("Motivate me", size="sm").click(
                        lambda: "I'm feeling lazy today, motivate me", None, msg
                    )
        
        # Events
        def user_submit(user_message, chat_history):
            chat_history = chat_history + [(user_message, None)]
            return "", chat_history

        def bot_response(chat_history, max_new_tokens, temperature, top_p, top_k, repetition_penalty, system_prompt):
            user_message = chat_history[-1][0]
            generator = chat_fn(user_message, chat_history[:-1], max_new_tokens, temperature, top_p, top_k, repetition_penalty, system_prompt)
            partial = ""
            for partial in generator:
                chat_history[-1] = (user_message, partial)
                yield chat_history

        submit.click(user_submit, [msg, chat], [msg, chat])\
              .then(bot_response,
                    [chat, max_new_tokens, temperature, top_p, top_k, repetition_penalty, system_prompt],
                    [chat])
        msg.submit(user_submit, [msg, chat], [msg, chat])\
           .then(bot_response,
                 [chat, max_new_tokens, temperature, top_p, top_k, repetition_penalty, system_prompt],
                 [chat])
        clear.click(lambda: None, None, chat, queue=False)
    return demo


def main():
    print("🔥 Starting Thani Thankan Chat App...")
    print("Loading model...")
    load_model()  # Warm early loading
    print("Model loaded successfully!")
    
    demo = build_interface()
    print("Launching app on http://localhost:7860")
    demo.queue(max_size=32).launch(
        server_name="0.0.0.0", 
        server_port=int(os.environ.get("PORT", 7860)),
        show_error=True
    )


if __name__ == "__main__":
    main()
