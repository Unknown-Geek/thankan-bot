"""
Thani Thankan - The rough, moody alter ego of Thankan Chettan
Speed optimized version using microsoft/DialoGPT-small
"""
import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model configuration
MODEL_ID = "microsoft/DialoGPT-small"

# Thani personality responses for fast fallback
THANI_RESPONSES = {
    "greeting": [
        "Enthuva myre? Enthokke und vishesham?",
        "Eda thayoli... ippozha vannathu?",
        "Pinne enthada? Pani onnum illa alle?"
    ],
    "help": [
        "Eda thendi... enthada help venam? Clear ayi parayada!",
        "Help venam? Pakshe thanks onnum parayilla alle?",
        "Eda kallan! Entha problem? Parayada vega!"
    ],
    "programming": [
        "Programming? Eda thayoli, documentation vayichittano?",
        "Code ezhuthanda? Stack Overflow nokku first!",
        "Eda kallan! Error message copy cheythu Google il search cheyyada!"
    ],
    "motivation": [
        "Eda thendi! Lazy aano? Pani edutholu!",
        "Motivation venam? Eda thayoli, sukham thanne!",
        "Eda kallan! Dreams undenkil work cheyyada!"
    ],
    "identity": [
        "Thankan njanaada kunna poori mone! Thani Thankan!",
        "Enthuva myre? Njan Thani Thankan aanu. Rough version!",
        "Eda thayoli... Thankan Chettan inte rough alter ego aanu njan!"
    ],
    "default": [
        "Enthuva myre? Onnum manassilayilla!",
        "Eda thayoli... clear ayi chodhikku!",
        "Pinne enthada parayendathu?",
        "Eda kallan! Vere enthokke venam?"
    ]
}

# Global model cache
_model = None
_tokenizer = None

def load_model():
    """Load DialoGPT model"""
    global _model, _tokenizer
    
    if _model is not None and _tokenizer is not None:
        return _tokenizer, _model
    
    try:
        print(f"Loading {MODEL_ID}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        _model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
            
        print("Model loaded successfully!")
        return _tokenizer, _model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def get_response_category(message):
    """Determine response category based on message"""
    message_lower = message.lower()
    
    if any(word in message_lower for word in ['hi', 'hello', 'hey']):
        return "greeting"
    elif any(word in message_lower for word in ['who are you', 'who', 'what are you']):
        return "identity"
    elif any(word in message_lower for word in ['help', 'please', 'can you']):
        return "help"
    elif any(word in message_lower for word in ['code', 'programming', 'python', 'javascript']):
        return "programming"
    elif any(word in message_lower for word in ['lazy', 'tired', 'motivate', 'motivation']):
        return "motivation"
    else:
        return "default"

def generate_thani_response(message, history):
    """Generate Thani's response"""
    try:
        tokenizer, model = load_model()
        
        if tokenizer and model:
            # Try model generation
            chat_history_ids = None
            
            # Keep conversation short for speed
            if len(history) > 0:
                # Use only last exchange
                last_user_msg = history[-1][0] if history[-1][0] else ""
                conversation = last_user_msg + tokenizer.eos_token + message + tokenizer.eos_token
            else:
                conversation = message + tokenizer.eos_token
            
            # Encode input
            new_user_input_ids = tokenizer.encode(conversation, return_tensors='pt')
            
            # Generate response
            with torch.no_grad():
                chat_history_ids = model.generate(
                    new_user_input_ids,
                    max_length=new_user_input_ids.shape[1] + 30,  # Short responses
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response = tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
            
            if response and len(response.strip()) > 2:
                # Add Thani flavor
                category = get_response_category(message)
                if category == "identity":
                    return f"Enthuva myre? {response} Njan Thani Thankan aanu!"
                elif category == "help":
                    return f"Eda thendi... {response}. Thanks um parayilla!"
                elif category == "motivation":
                    return f"Eda kallan! {response} Pani edutholu!"
                else:
                    return f"Enthuva myre? {response}"
    
    except Exception as e:
        print(f"Model generation failed: {e}")
    
    # Fallback to predefined responses
    import random
    category = get_response_category(message)
    responses = THANI_RESPONSES.get(category, THANI_RESPONSES["default"])
    return random.choice(responses)

def chat_with_thani(message, history):
    """Main chat function"""
    if not message.strip():
        return history, ""
    
    response = generate_thani_response(message, history)
    history.append([message, response])
    
    return history, ""

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="🔥 Thani Thankan") as demo:
        gr.Markdown("""
        # 🔥 Thani Thankan - The Rough Alter Ego
        ### *Powered by Microsoft DialoGPT-Small ⚡*
        
        **Warning:** Uses aggressive Malayalam slang! Not for the faint-hearted 😤
        
        **Try asking:**
        - "Who are you?"
        - "Help me with coding"
        - "I'm feeling lazy"
        """)
        
        chatbot = gr.Chatbot(
            value=[],
            height=400,
            avatar_images=["👤", "😈"]
        )
        
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Enthuva myre? Ask something...",
                show_label=False,
                scale=4
            )
            send_btn = gr.Button("Send 🔥", scale=1)
        
        clear_btn = gr.Button("Clear Chat")
        
        # Event handlers
        msg.submit(chat_with_thani, [msg, chatbot], [chatbot, msg])
        send_btn.click(chat_with_thani, [msg, chatbot], [chatbot, msg])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])
    
    return demo

# Launch the app
if __name__ == "__main__":
    print("🔥 Starting Thani Thankan...")
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
