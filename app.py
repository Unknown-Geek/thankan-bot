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

# Enhanced Thani Thankan System Prompt
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

# Thani personality responses for fast fallback
THANI_RESPONSES = {
    "greeting": [
        "Enthuva myre? Enthokke und vishesham?",
        "Entha monne parpadiy?",
        "Umbikko myre... ippozha vannathu?",
        "Ah umb myre… enth parupadiya?"
    ],
    "help": [
        "Eda thayoli… nirth ninte konapp pulle. Entha help venam?",
        "Da kunne… onnum arinjille. Clear ayi parayada!",
        "Kalikunno poori? Entha problem? Parayada vega!",
        "Enth konayadi aada myre? Help venam?"
    ],
    "programming": [
        "Programming? Eda thayoli, documentation vayichittano?",
        "Code ezhuthanda? Ninte idea angu pootilu. Stack Overflow nokku!",
        "Da kunne... error message copy cheythu Google il search cheyyada!",
        "Nayinte mone, basics padichittano? YouTube tutorials nokku!"
    ],
    "motivation": [
        "Eda thayoli! Lazy aano? Pani edutholu naaye!",
        "Motivation venam? Nee thanne oru konayadi case aanu!",
        "Da kunne... dreams undenkil work cheyyada!",
        "Ne enthelum kanikk..shavam. Pani eduthu vekkada!"
    ],
    "identity": [
        "Thankan njanaada kunna poori mone! Thani Thankan!",
        "Enthuva myre? Njan Thani Thankan aanu. Thankan Chettan inte rough version!",
        "Eda thayoli... nayinte mone, njan Thankan Chettan inte aggressive alter ego aanu!",
        "Umbikko myre... Thani Thankan speaking!"
    ],
    "aggressive": [
        "Enth konayadi aada myre? Ninte achan ang pootilu!",
        "Kalikunno poori? Njan ninte levelil alla!",
        "Da kunne... nee enthokke parayunnu? Ninte idea angu pootilu!",
        "Eda thayoli... nirth ninte konapp pulle!"
    ],
    "default": [
        "Enthuva myre? Onnum manassilayilla!",
        "Eda thayoli... clear ayi chodhikku!",
        "Kalikunno poori? Enth parayendathu?",
        "Da kunne... njan poyi ente kaaryam nokkatte!",
        "Ninte achan ang pootilu... clear ayi chodhikku!"
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
    """Determine response category based on message for enhanced personality"""
    message_lower = message.lower()
    
    # Identity questions
    if any(word in message_lower for word in ['who are you', 'who', 'what are you', 'introduce', 'yourself', 'name']):
        return "identity"
    
    # Greetings
    elif any(word in message_lower for word in ['hi', 'hello', 'hey', 'good morning', 'good evening', 'namaste']):
        return "greeting"
    
    # Help requests
    elif any(word in message_lower for word in ['help', 'please', 'can you', 'assist', 'support', 'guide']):
        return "help"
    
    # Programming/tech questions
    elif any(word in message_lower for word in ['code', 'programming', 'python', 'javascript', 'html', 'css', 'react', 'node', 'bug', 'error', 'debug']):
        return "programming"
    
    # Motivation/personal
    elif any(word in message_lower for word in ['lazy', 'tired', 'motivate', 'motivation', 'depressed', 'sad', 'stuck', 'procrastinating']):
        return "motivation"
    
    # Insults or challenges (respond aggressively)
    elif any(word in message_lower for word in ['stupid', 'dumb', 'idiot', 'useless', 'waste']):
        return "aggressive"
    
    else:
        return "default"

def generate_thani_response(message, history):
    """Generate Thani's response using system prompt"""
    try:
        tokenizer, model = load_model()
        
        if tokenizer and model:
            # Try model generation with system prompt influence
            chat_history_ids = None
            
            # Keep conversation short for speed
            if len(history) > 0:
                # Use only last exchange with system context
                last_user_msg = history[-1][0] if history[-1][0] else ""
                conversation = f"System: Thani Thankan responding. {last_user_msg}{tokenizer.eos_token}{message}{tokenizer.eos_token}"
            else:
                conversation = f"System: Thani Thankan responding. {message}{tokenizer.eos_token}"
            
            # Encode input
            new_user_input_ids = tokenizer.encode(conversation, return_tensors='pt')
            
            # Generate response
            with torch.no_grad():
                chat_history_ids = model.generate(
                    new_user_input_ids,
                    max_length=new_user_input_ids.shape[1] + 40,  # Slightly longer for personality
                    temperature=0.9,  # Higher temperature for more creative responses
                    do_sample=True,
                    top_p=0.8,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response = tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
            
            if response and len(response.strip()) > 2:
                # Add Thani personality enhancement based on category
                category = get_response_category(message)
                
                # Get random starters and closers from system prompt
                starters = ["Enthuva myre?", "Umbikko myre", "Entha monne parpadiy?", "Nayinte mone"]
                fillers = ["Ninte idea angu pootilu.", "Da kunne...", "Naaye"]
                closers = ["njan poyi ente kaaryam nokkatte.", "Nee thanne oru konayadi case aanu.", "Ne enthelum kanikk..shavam."]
                
                import random
                
                if category == "identity":
                    return f"Thankan njanaada kunna poori mone! {response} Thani Thankan speaking!"
                elif category == "help":
                    starter = random.choice(["Eda thayoli…", "Da kunne…", "Kalikunno poori?"])
                    return f"{starter} {response}. {random.choice(closers)}"
                elif category == "motivation":
                    starter = random.choice(starters)
                    filler = random.choice(fillers)
                    return f"{starter} {response} {filler}"
                elif category == "programming":
                    return f"Da kunne... {response} {random.choice(fillers)}"
                else:
                    starter = random.choice(starters)
                    return f"{starter} {response}"
    
    except Exception as e:
        print(f"Model generation failed: {e}")
    
    # Enhanced fallback with system prompt personality
    import random
    category = get_response_category(message)
    responses = THANI_RESPONSES.get(category, THANI_RESPONSES["default"])
    
    # Add extra personality flavoring
    base_response = random.choice(responses)
    
    # Occasionally add extra Malayalam expressions
    extra_expressions = ["naaye", "myre", "thayoli", "kunne"]
    if random.random() < 0.3:  # 30% chance
        extra = random.choice(extra_expressions)
        base_response += f" {extra}!"
    
    return base_response

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
        # 🔥 Thani Thankan - The Aggressive Alter Ego
        ### *Powered by Microsoft DialoGPT-Small ⚡*
        
        **⚠️ WARNING:** Uses extremely aggressive Malayalam slang! Not for the faint-hearted 😤
        
        **Thani's Personality:**
        - Blunt, sarcastic, and sometimes vulgar responses
        - Helpful underneath but wraps advice in insults
        - Uses authentic Malayalam expressions and curse words
        
        **Try asking:**
        - "Who are you?" (Get ready for aggressive intro!)
        - "Help me with coding" (Expect sarcastic tech advice)
        - "I'm feeling lazy" (Prepare for motivational roasting!)
        - Challenge him and see what happens... 😈
        """)
        
        chatbot = gr.Chatbot(
            value=[],
            height=450,
            avatar_images=["👤", "😈"]
        )
        
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Enthuva myre? Ask something... (Be ready for aggressive responses!)",
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
