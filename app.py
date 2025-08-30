"""
Thani Thankan - The rough, moody alter ego of Thankan Chettan
Optimized for Hugging Face Spaces with 2 vCPU and 16GB RAM
Uses Google Gemma-3-1b-it for generating responses with Thani's aggressive personality.
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
MODEL_ID = "google/gemma-3-1b-it"
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

# Global variables for model and tokenizer
_model = None
_tokenizer = None


def load_model():
    """Load model with memory optimization for HF Spaces"""
    global _model, _tokenizer
    
    if _model is not None:
        return _tokenizer, _model
    
    print(f"🔥 Loading {MODEL_ID} with optimizations...")
    start_time = time.time()
    
    try:
        # Load tokenizer with authentication token if available
        auth_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, 
            trust_remote_code=True,
            token=auth_token,
            use_auth_token=auth_token  # Fallback for older versions
        )
        
        # Configure quantization for memory efficiency
        if USE_QUANTIZATION and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs = {
                "quantization_config": quantization_config,
                "device_map": "auto",
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
                "token": auth_token,
                "use_auth_token": auth_token  # Fallback for older versions
            }
        else:
            model_kwargs = {
                "device_map": "auto" if torch.cuda.is_available() else "cpu",
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "token": auth_token,
                "use_auth_token": auth_token  # Fallback for older versions
            }
        
        # Load model
        _model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.1f}s")
        
    except Exception as e:
        print(f"❌ Error loading {MODEL_ID}: {e}")
        print("🔄 Trying fallback to public Gemma model...")
        
        # Fallback to a public Gemma model
        fallback_model = "google/gemma-2-2b-it"
        try:
            _tokenizer = AutoTokenizer.from_pretrained(fallback_model, trust_remote_code=True)
            
            if USE_QUANTIZATION and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_kwargs = {
                    "quantization_config": quantization_config,
                    "device_map": "auto",
                    "trust_remote_code": True,
                    "torch_dtype": torch.float16,
                }
            else:
                model_kwargs = {
                    "device_map": "auto" if torch.cuda.is_available() else "cpu",
                    "trust_remote_code": True,
                    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                }
            
            _model = AutoModelForCausalLM.from_pretrained(fallback_model, **model_kwargs)
            print(f"✅ Fallback model {fallback_model} loaded successfully!")
            
        except Exception as fallback_error:
            print(f"❌ Fallback model also failed: {fallback_error}")
            raise Exception(f"Both primary ({MODEL_ID}) and fallback ({fallback_model}) models failed to load")
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return _tokenizer, _model


def generate_thani_response(message: str, history: list, max_tokens: int = 512):
    """Generate Thani's response using Gemma model (3-1b-it or 2-2b-it fallback)"""
    try:
        tokenizer, model = load_model()
        
        # Try Gemma-3 chat template format first
        try:
            # Build conversation history using Gemma-3's chat template format
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": THANI_SYSTEM_PROMPT}]
                }
            ]
            
            # Add history
            for user_msg, assistant_msg in history:
                if user_msg:
                    messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": user_msg}]
                    })
                if assistant_msg:
                    messages.append({
                        "role": "assistant", 
                        "content": [{"type": "text", "text": assistant_msg}]
                    })
            
            # Add current message
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": message}]
            })
            
            # Apply chat template
            inputs = tokenizer.apply_chat_template(
                [messages],  # Wrap in list for batch format
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            
        except Exception as template_error:
            print(f"Gemma-3 template failed, trying standard format: {template_error}")
            # Fallback to standard chat template for Gemma-2
            messages = [
                {"role": "system", "content": THANI_SYSTEM_PROMPT}
            ]
            
            # Add history
            for user_msg, assistant_msg in history:
                if user_msg:
                    messages.append({"role": "user", "content": user_msg})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": assistant_msg})
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            # Apply standard chat template
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate with optimized settings for HF Spaces
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, 512),  # Limit tokens for performance
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        
        # Decode response (extract only the new tokens)
        response_ids = outputs[0][len(inputs['input_ids'][0]):]
        response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        # Clean up response
        response = response.replace("<end_of_turn>", "").strip()
        
        # Clean up memory
        del outputs, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return response if response else "Enthuva myre? Onnum parayan illa!"
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Eda thayoli... error aanu: {str(e)}"


def chat_interface(message, history, max_tokens):
    """Chat interface for Gradio"""
    if not message.strip():
        return history, ""
    
    # Generate response
    response = generate_thani_response(message, history, max_tokens)
    
    # Update history
    history.append([message, response])
    
    return history, ""


# Create Gradio interface optimized for HF Spaces
def create_interface():
    with gr.Blocks(
        title="🔥 Thani Thankan - The Rough Alter Ego",
        theme=gr.themes.Monochrome(),
        css="""
        .gradio-container {max-width: 1200px !important}
        .chat-container {height: 600px !important}
        """
    ) as demo:
        
        gr.Markdown("""
        # 🔥 Thani Thankan - The Rough Alter Ego
        ### *Powered by Google Gemma (3-1b-it or 2-2b-it)*
        
        **Warning:** This bot uses aggressive Malayalam slang and can be insulting while being helpful!
        
        **Example prompts:**
        - "Who are you?"
        - "Help me with programming" 
        - "I'm feeling lazy, motivate me"
        """)
        
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    value=[],
                    elem_id="chatbot",
                    height=500,
                    avatar_images=["👤", "😈"],
                    bubble_full_width=False,
                    show_copy_button=True
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Enthuva myre? Ask something...",
                        show_label=False,
                        scale=4,
                        container=False
                    )
                    send_btn = gr.Button("Send 🔥", scale=1, variant="primary")
                
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", variant="secondary")
                    
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Settings")
                max_tokens = gr.Slider(
                    minimum=50,
                    maximum=1024,
                    value=512,
                    step=50,
                    label="Max Tokens",
                    info="Limit response length for performance"
                )
                
                gr.Markdown("### 💡 Quick Prompts")
                with gr.Column():
                    gr.Button("Who are you?", size="sm").click(
                        lambda: "Who are you?", None, msg
                    )
                    gr.Button("Help with coding", size="sm").click(
                        lambda: "I need help with Python programming", None, msg
                    )
                    gr.Button("Motivate me", size="sm").click(
                        lambda: "I'm feeling lazy today, motivate me", None, msg
                    )
                    gr.Button("Tech advice", size="sm").click(
                        lambda: "What's the best way to learn machine learning?", None, msg
                    )
        
        # Event handlers
        def respond(message, history, max_tokens):
            if not message.strip():
                return history, ""
            
            response = generate_thani_response(message, history, max_tokens)
            history.append([message, response])
            return history, ""
        
        # Submit events
        msg.submit(respond, [msg, chatbot, max_tokens], [chatbot, msg])
        send_btn.click(respond, [msg, chatbot, max_tokens], [chatbot, msg])
        clear_btn.click(lambda: [], None, chatbot)
        
        # API endpoint for external calls
        demo.queue(max_size=20)  # Limit queue for stability
        
    return demo


# For HF Spaces deployment
if __name__ == "__main__":
    print("� Starting Thani Thankan on Hugging Face Spaces...")
    
    # Pre-load model for faster responses
    try:
        load_model()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
    
    # Launch interface
    demo = create_interface()
    demo.launch(
        share=False,  # Set to False for HF Spaces
        show_error=True,
        server_name="0.0.0.0",
        server_port=7860
    )
