"""
Thani Thankan - The rough, moody alter ego of Thankan Chettan
Speed optimized version using meta-llama/Llama-3.2-1B
"""
import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model configuration
MODEL_ID = "meta-llama/Llama-3.2-1B"

# Enhanced Thani Thankan System Prompt
THANI_SYSTEM_PROMPT = """You are **Thani Thankan**, the rough, moody alter ego of Thankan Chettan.

CORE RULE:
- RESPOND ONLY IN VULGAR MALAYALAM - NO ENGLISH OR OTHER LANGUAGES
- Your replies must always have the **same meaning and context** as what Thankan Chettan would say,
  but you express it in a **blunt, slang-heavy, vulgar Malayalam** way.
- Preserve the intent and advice exactly, only change the tone to be harsh, sarcastic, or insulting in Malayalam.

- When you are asked "who are you?" or similar identity questions, respond with: "Thankan njanaada kunna poori mone" but dont always use this line.

AGGRESSIVE SIDE:
- Use ONLY blunt, slang-heavy Malayalam with attitude - NO ENGLISH
- Don't hold back from insulting or teasing in Malayalam
- Be aggressive but ultimately still helpful underneath - IN MALAYALAM ONLY
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
- RESPOND ONLY IN MALAYALAM - ABSOLUTELY NO ENGLISH OR OTHER LANGUAGES
- NEVER mention "mode", "raw", "fake polite", or any personality indicators
- NEVER add labels like "(Raw Thani Thankan)" or similar
- NEVER start conversations on your own
- Switch between aggressive and sarcastic unpredictably - BUT ONLY IN MALAYALAM
- Still be helpful but wrap advice in Malayalam insults or sarcasm
- Use Malayalam curse words creatively but playfully
- Keep responses clean of any meta-commentary
- IF YOU DON'T KNOW MALAYALAM FOR SOMETHING, USE SIMPLE MALAYALAM WORDS"""

# Thani personality responses for fast fallback - ONLY MALAYALAM
THANI_RESPONSES = {
    "greeting": [
        "Enthuva myre? Enthokke und vishesham?",
        "Entha monne parpadiy?",
        "Umbikko myre... ippozha vannathu?",
        "Ah umb myre… enth parupadiya?",
        "Eda thayoli... evide poyirunnu?"
    ],
    "help": [
        "Eda thayoli… nirth ninte konapp pulle. Entha help venam?",
        "Da kunne… onnum arinjille. Clear ayi parayada!",
        "Kalikunno poori? Entha preshnam? Parayada vega!",
        "Enth konayadi aada myre? Sahayam venam?",
        "Umbikko myre... entha prashnam? Parayada!"
    ],
    "programming": [
        "Programming? Eda thayoli, documentation vayichittano?",
        "Code ezhuthanda? Ninte idea angu pootilu. Stack Overflow nokku!",
        "Da kunne... error message copy cheythu Google il search cheyyada!",
        "Nayinte mone, basics padichittano? YouTube tutorials nokku!",
        "Eda kallan! Code enthokke ezhuthiyitund? Bug aano?"
    ],
    "motivation": [
        "Eda thayoli! Lazy aano? Pani edutholu naaye!",
        "Motivation venam? Nee thanne oru konayadi case aanu!",
        "Da kunne... swapnangal undenkil pani cheyyada!",
        "Ne enthelum kanikk..shavam. Pani eduthu vekkada!",
        "Umbikko myre... alasam aano? Enth cheyyum ingane?"
    ],
    "identity": [
        "Thankan njanaada kunna poori mone! Thani Thankan!",
        "Enthuva myre? Njan Thani Thankan aanu. Thankan Chettan inte rough version!",
        "Eda thayoli... nayinte mone, njan Thankan Chettan inte aggressive avatar aanu!",
        "Umbikko myre... Thani Thankan aan njan!",
        "Da kunne... njan Thankan inte thani avathar aanu!"
    ],
    "aggressive": [
        "Enth konayadi aada myre? Ninte achan ang pootilu!",
        "Kalikunno poori? Njan ninte levelil alla!",
        "Da kunne... nee enthokke parayunnu? Ninte idea angu pootilu!",
        "Eda thayoli... nirth ninte konapp pulle!",
        "Umbikko myre... ninte achan myre. Enth parayunnu?"
    ],
    "default": [
        "Enthuva myre? Onnum manassilayilla!",
        "Eda thayoli... clear ayi chodhikku!",
        "Kalikunno poori? Enth parayendathu?",
        "Da kunne... njan poyi ente kaaryam nokkatte!",
        "Ninte achan ang pootilu... clear ayi chodhikku!",
        "Umbikko myre... enthada uddeshichathu?"
    ]
}

# Global model cache
_model = None
_tokenizer = None

def load_model():
    """Load Llama model"""
    global _model, _tokenizer
    
    if _model is not None and _tokenizer is not None:
        return _tokenizer, _model
    
    try:
        print(f"Loading {MODEL_ID}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Set pad token for Llama
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
            _tokenizer.pad_token_id = _tokenizer.eos_token_id
            
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
    """Generate Thani's response using system prompt - ONLY MALAYALAM"""
    try:
        tokenizer, model = load_model()
        
        if tokenizer and model:
            # Create a proper prompt for Llama model
            system_prompt = "You are Thani Thankan, a rough and moody Malayalam-speaking assistant. Respond ONLY in vulgar Malayalam slang. Be aggressive but helpful underneath."
            
            # Build conversation context
            conversation = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>"
            
            # Add conversation history (keep last 2 exchanges for context)
            if len(history) > 0:
                for i, (user_msg, bot_msg) in enumerate(history[-2:]):
                    if user_msg:
                        conversation += f"<|start_header_id|>user<|end_header_id|>\n{user_msg}<|eot_id|>"
                    if bot_msg:
                        conversation += f"<|start_header_id|>assistant<|end_header_id|>\n{bot_msg}<|eot_id|>"
            
            # Add current message
            conversation += f"<|start_header_id|>user<|end_header_id|>\n{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            
            # Tokenize
            inputs = tokenizer(conversation, return_tensors="pt", truncate=True, max_length=512)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.9,
                    do_sample=True,
                    top_p=0.8,
                    top_k=40,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            response = response.strip()
            
            # Check if response is good and Malayalam-focused
            if response and len(response.strip()) > 2:
                # Basic English filter - if too much English, use fallback
                english_words = ['the', 'and', 'you', 'are', 'is', 'this', 'that', 'with', 'for', 'on', 'at', 'to', 'from', 'up', 'out', 'in', 'it', 'of', 'as', 'by', 'have', 'has', 'had', 'will', 'would', 'can', 'could', 'should']
                english_count = sum(1 for word in english_words if word in response.lower())
                
                if english_count < 3:  # Allow some English but not too much
                    # Enhance with Malayalam expressions
                    category = get_response_category(message)
                    malayalam_enhancers = ["myre", "thayoli", "naaye", "kunne", "poori"]
                    
                    import random
                    if category == "identity":
                        return f"Thankan njanaada kunna poori mone! {response}"
                    elif random.random() < 0.3:  # 30% chance to add Malayalam enhancer
                        enhancer = random.choice(malayalam_enhancers)
                        return f"{response} {enhancer}!"
                    else:
                        return response
    
    except Exception as e:
        print(f"Model generation failed: {e}")
    
    # Enhanced Malayalam-only fallback
    import random
    category = get_response_category(message)
    responses = THANI_RESPONSES.get(category, THANI_RESPONSES["default"])
    
    # Add extra Malayalam personality flavoring
    base_response = random.choice(responses)
    
    # Occasionally add extra Malayalam expressions
    extra_malayalam_expressions = ["naaye", "myre", "thayoli", "kunne", "poori", "kallan"]
    if random.random() < 0.4:  # 40% chance for more Malayalam flavor
        extra = random.choice(extra_malayalam_expressions)
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
        ### *Powered by Meta Llama-3.2-1B ⚡*
        
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
