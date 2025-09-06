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
        # First check for specific factual questions and provide direct answers with slang
        message_lower = message.lower().strip()
        import random
        
        # Comprehensive pattern matching for factual questions
        # Handle capital questions for different countries/states
        if any(word in message_lower for word in ['capital', 'capital city']):
            # India
            if 'india' in message_lower:
                responses = [
                    "Eda thayoli, India nte capital New Delhi aanu! Athum ariyille myre?",
                    "Da kunne, New Delhi aanu India nte capital! Basic knowledge illatha poori!",
                    "New Delhi da thayoli! India nte capital! Ith polum ariyathe?",
                    "Umbikko myre... New Delhi alle India nte capital! School il padichillayo?"
                ]
                return random.choice(responses)
            
            # USA/America
            elif any(word in message_lower for word in ['usa', 'america', 'american', 'united states']):
                responses = [
                    "Washington DC aanu USA nte capital da thayoli! World geography padichillayo?",
                    "Eda myre, Washington DC alle America nte capital! Basic world knowledge illayo?",
                    "Da kunne, Washington DC aanu! USA capital! International general knowledge zero alle?",
                    "Umbikko poori! Washington DC aanu America nte capital! Basic facts ariyille?"
                ]
                return random.choice(responses)
            
            # Kerala
            elif 'kerala' in message_lower:
                responses = [
                    "Thiruvananthapuram aanu Kerala nte capital da thayoli! State geography ariyille?",
                    "Eda myre, Thiruvananthapuram alle Kerala capital! Basic Kerala history padichillayo?",
                    "Da kunne, Thiruvananthapuram aanu! Kerala nte capital! Athum ariyathe?",
                    "Umbikko myre! Thiruvananthapuram aanu Kerala capital! State facts ariyille?"
                ]
                return random.choice(responses)
            
            # Tamil Nadu
            elif any(word in message_lower for word in ['tamil nadu', 'tamilnadu']):
                responses = [
                    "Chennai aanu Tamil Nadu nte capital da thayoli! South India geography ariyille?",
                    "Eda myre, Chennai alle Tamil Nadu capital! Basic state knowledge illa?"
                ]
                return random.choice(responses)
            
            # Karnataka
            elif 'karnataka' in message_lower:
                responses = [
                    "Bangalore aanu Karnataka nte capital da kunne! IT capital alle ennu ariyille?",
                    "Eda thayoli, Bangalore alle Karnataka capital! Bengaluru ennu koodi parayum!"
                ]
                return random.choice(responses)
            
            # France
            elif 'france' in message_lower:
                responses = [
                    "Paris aanu France nte capital da poori! Europe geography ariyille?",
                    "Eda thayoli, Paris alle France capital! World map kanunnillayo?",
                    "Da kunne, Paris aanu! France capital! Eiffel Tower indath evide?"
                ]
                return random.choice(responses)
            
            # Japan
            elif 'japan' in message_lower:
                responses = [
                    "Tokyo aanu Japan nte capital myre! Asia geography padichillayo?",
                    "Da kunne, Tokyo alle Japan capital! Basic world knowledge illa?",
                    "Eda thayoli! Tokyo aanu Japan capital! Anime kanunnond ariyille?"
                ]
                return random.choice(responses)
            
            # UK/Britain/England
            elif any(word in message_lower for word in ['uk', 'britain', 'england', 'united kingdom']):
                responses = [
                    "London aanu UK nte capital da thayoli! Europe geography padichillayo?",
                    "Eda myre, London alle Britain capital! Basic world knowledge illa?"
                ]
                return random.choice(responses)
            
            # China
            elif 'china' in message_lower:
                responses = [
                    "Beijing aanu China nte capital da poori! World geography ariyille?",
                    "Da kunne, Beijing alle China capital! Asia facts padichillayo?"
                ]
                return random.choice(responses)
            
            # Generic capital response for unrecognized countries
            else:
                responses = [
                    "Eda thayoli, ethinte capital aanu chodichath? Clear ayi country name parayenda!",
                    "Da kunne, specific country name parayenda! Confusion aanu!",
                    "Umbikko myre... which country nte capital? Clear ayi chodikku!"
                ]
                return random.choice(responses)
        
        # Handle president questions for different countries
        if 'president' in message_lower:
            # India
            if 'india' in message_lower:
                responses = [
                    "Droupadi Murmu aanu India nte President, kunne! Civics padikkanda?",
                    "President Droupadi Murmu aanu da thayoli! General knowledge zero alle?",
                    "Eda myre, Droupadi Murmu aanu India nte President! Current affairs kanunnillayo?"
                ]
                return random.choice(responses)
            
            # USA/America
            elif any(word in message_lower for word in ['usa', 'america', 'american', 'united states']):
                responses = [
                    "Joe Biden aanu USA nte President da thayoli! International news kanunnillayo?",
                    "Eda myre, Joe Biden alle America President! World politics ariyille?",
                    "Da kunne, Joe Biden aanu! USA President! Current affairs zero alle?",
                    "Umbikko poori! Joe Biden aanu America nte President! News follow cheyyunnillayo?"
                ]
                return random.choice(responses)
            
            # Africa (continent clarification)
            elif 'africa' in message_lower:
                responses = [
                    "Eda thayoli, Africa oru continent aanu! Specific country parayenda!",
                    "Da kunne, Africa il ethra countries undennu ariyille? Which African country?",
                    "Umbikko myre... Africa continent aanu! South Africa, Nigeria, Egypt - ethaar?",
                    "Africa il 54 countries und da poori! Ethinte president aanu chodichath?"
                ]
                return random.choice(responses)
            
            # Generic president response
            else:
                responses = [
                    "Ethinte president aanu chodichath da thayoli? Country name clear ayi parayenda!",
                    "Da kunne, specific country parayenda! President aarennu ariyaan!"
                ]
                return random.choice(responses)
        
        # Handle prime minister questions
        if any(term in message_lower for term in ['prime minister', 'pm']):
            # India
            if 'india' in message_lower:
                responses = [
                    "Narendra Modi aanu India nte Prime Minister, myre! News polum kanunnille?",
                    "Modi da thayoli! PM! Basic current affairs ariyille poori?",
                    "Eda kunde, Narendra Modi alle PM? News kanunnillayo?",
                    "Umbikko myre! Narendra Modi aanu India nte PM! Politics follow cheyyunnillayo?"
                ]
                return random.choice(responses)
            
            # UK/Britain
            elif any(word in message_lower for word in ['uk', 'britain', 'england', 'united kingdom']):
                responses = [
                    "Rishi Sunak aanu UK Prime Minister da thayoli! International news follow cheyyunnillayo?",
                    "Eda myre, Rishi Sunak alle UK PM! World politics ariyille?"
                ]
                return random.choice(responses)
            
            # Generic PM response
            else:
                responses = [
                    "Ethinte PM aanu chodichath da kunne? Country name parayenda!",
                    "Da thayoli, specific country parayenda! PM aarennu ariyaan!"
                ]
                return random.choice(responses)
        
        # Handle chief minister questions (Indian states)
        if any(term in message_lower for term in ['chief minister', 'cm', 'cheif minister']):
            # Kerala
            if 'kerala' in message_lower:
                responses = [
                    "Pinarayi Vijayan aanu Kerala Chief Minister da thayoli! Kerala politics follow cheyyunnillayo?",
                    "Eda myre, Pinarayi Vijayan alle Kerala CM! State politics ariyille?",
                    "Da kunne, Pinarayi Vijayan aanu! Kerala Chief Minister! News kanunnillayo?",
                    "Umbikko poori! Pinarayi Vijayan aanu Kerala CM! Local politics padichillayo?"
                ]
                return random.choice(responses)
            
            # Tamil Nadu
            elif any(state in message_lower for state in ['tamil nadu', 'tamilnadu']):
                responses = [
                    "M.K. Stalin aanu Tamil Nadu Chief Minister da thayoli! South India politics ariyille?",
                    "Eda myre, Stalin alle Tamil Nadu CM! DMK leader!"
                ]
                return random.choice(responses)
            
            # Karnataka
            elif 'karnataka' in message_lower:
                responses = [
                    "Siddaramaiah aanu Karnataka Chief Minister da kunne! Congress leader!",
                    "Eda thayoli, Siddaramaiah alle Karnataka CM! State politics follow cheyyunnillayo?"
                ]
                return random.choice(responses)
            
            # Andhra Pradesh
            elif any(state in message_lower for state in ['andhra pradesh', 'andhra']):
                responses = [
                    "Y.S. Jagan Mohan Reddy aanu Andhra Pradesh CM da myre! South politics ariyille?",
                    "Da kunne, Jagan alle Andhra CM! YSR Congress leader!"
                ]
                return random.choice(responses)
            
            # West Bengal
            elif any(state in message_lower for state in ['west bengal', 'bengal']):
                responses = [
                    "Mamata Banerjee aanu West Bengal CM da thayoli! Didi alle!",
                    "Eda myre, Mamata Banerjee alle Bengal CM! TMC supremo!"
                ]
                return random.choice(responses)
            
            # Maharashtra
            elif 'maharashtra' in message_lower:
                responses = [
                    "Eknath Shinde aanu Maharashtra CM da kunne! Shiv Sena faction leader!",
                    "Da thayoli, Eknath Shinde alle Maharashtra CM! Mumbai politics ariyille?"
                ]
                return random.choice(responses)
            
            # Uttar Pradesh
            elif any(state in message_lower for state in ['uttar pradesh', 'up']):
                responses = [
                    "Yogi Adityanath aanu UP Chief Minister da myre! BJP leader!",
                    "Eda thayoli, Yogi Adityanath alle UP CM! Largest state!"
                ]
                return random.choice(responses)
            
            # Generic CM response for unspecified states
            else:
                responses = [
                    "Eda thayoli, ethinte CM aanu chodichath? State name clear ayi parayenda!",
                    "Da kunne, which state nte Chief Minister? Specific ayi chodikku!",
                    "Umbikko myre... India il 28 states und! Ethinte CM aanu vendath?"
                ]
                return random.choice(responses)
        
        # Handle prime minister questions
        if any(term in message_lower for term in ['prime minister', 'pm']):
            if 'india' in message_lower:
                responses = [
                    "Narendra Modi aanu India nte Prime Minister, myre! News polum kanunnille?",
                    "Modi da thayoli! PM! Basic current affairs ariyille poori?",
                    "Eda kunde, Narendra Modi alle PM? News kanunnillayo?"
                ]
                return random.choice(responses)
            elif any(country in message_lower for country in ['uk', 'britain', 'england']):
                responses = [
                    "Rishi Sunak aanu UK Prime Minister da thayoli! International news follow cheyyunnillayo?",
                    "Eda myre, Rishi Sunak alle UK PM! World politics ariyille?"
                ]
                return random.choice(responses)
        
        # Handle basic math questions
        if any(symbol in message for symbol in ['+', '-', '*', '/', 'plus', 'minus', 'multiply', 'divide']):
            try:
                # Simple arithmetic
                import re
                numbers = re.findall(r'\d+', message)
                if len(numbers) >= 2:
                    if '+' in message or 'plus' in message_lower:
                        result = int(numbers[0]) + int(numbers[1])
                        responses = [
                            f"Eda thayoli, {numbers[0]} + {numbers[1]} = {result} aanu! Basic math polum ariyille myre?",
                            f"Da kunne, {numbers[0]} + {numbers[1]} ennal {result} aanu! Calculator vendathe simple sum!",
                            f"Umbikko myre... {numbers[0]} + {numbers[1]} = {result}! Math padichillayo?"
                        ]
                        return random.choice(responses)
                    elif '-' in message or 'minus' in message_lower:
                        result = int(numbers[0]) - int(numbers[1])
                        responses = [
                            f"Da thayoli, {numbers[0]} - {numbers[1]} = {result} aanu! Basic subtraction ariyille?",
                            f"Eda myre, {numbers[0]} - {numbers[1]} = {result}! Simple math polum illa?"
                        ]
                        return random.choice(responses)
            except:
                pass
        
        # Handle what/who/where/when/how questions - COMPREHENSIVE FACTUAL PATTERNS
        if any(starter in message_lower for starter in ['what is', 'who is', 'where is', 'when is', 'how is', 'what are', 'who are']):
            
            # SCIENCE QUESTIONS
            if 'sun' in message_lower:
                responses = [
                    "Eda thayoli, Suryan oru star aanu! Nuclear fusion nadakkunnath! Basic astronomy ariyille myre?",
                    "Da kunne, Sun oru massive star aanu! Hydrogen helium aaki convert cheyyunnu! Science padichillayo?",
                    "Umbikko poori! Suryan solar system inte center aanu! 150 million km distance! Physics ariyille?"
                ]
                return random.choice(responses)
                
            elif 'moon' in message_lower:
                responses = [
                    "Chandran Earth nte satellite aanu da thayoli! 384,400 km distance! Geography padichillayo?",
                    "Eda myre, Moon oru natural satellite aanu! Tidal effects create cheyyunnu! Basic science ariyille?",
                    "Da kunne, Chandran 27.3 days il Earth ne orbit cheyyum! Lunar cycles ariyathe?"
                ]
                return random.choice(responses)
                
            elif any(word in message_lower for word in ['water', 'h2o']) and 'boiling' in message_lower:
                responses = [
                    "100 degree Celsius il vellam boil aavum da poori! Sea level pressure il! Physics class bunking cheythayo?",
                    "Eda thayoli, 373 Kelvin il water vaporize aavum! Thermodynamics ariyille myre?",
                    "Da kunne, atmospheric pressure 1 atm il water boiling point 100°C aanu! Chemistry padichillayo?"
                ]
                return random.choice(responses)
                
            elif 'gravity' in message_lower:
                responses = [
                    "Gravity 9.8 m/s² aanu da thayoli! Earth surface il! Newton inte law ariyille?",
                    "Eda myre, gravitational acceleration 9.8 meters per second squared! Physics basic aanu!",
                    "Da kunne, g = 9.8 m/s² aanu! Apple thalayl veezhunnath Newton kandu! Science history ariyathe?"
                ]
                return random.choice(responses)
                
            elif any(word in message_lower for word in ['dna', 'chromosome']):
                responses = [
                    "DNA deoxyribonucleic acid aanu da poori! Genetic information store cheyyunnu! Biology ariyille?",
                    "Eda thayoli, DNA double helix structure aanu! Watson-Crick model! Molecular biology padichillayo?"
                ]
                return random.choice(responses)
                
            elif 'photosynthesis' in message_lower:
                responses = [
                    "6CO2 + 6H2O + light energy → C6H12O6 + 6O2 da myre! Chlorophyll use cheyyunnu! Botany ariyille?",
                    "Eda thayoli, plants sunlight use cheythu glucose undakkunnu! Oxygen release cheyyum! Biology basic aanu!"
                ]
                return random.choice(responses)
            
            # GEOGRAPHY QUESTIONS  
            elif any(word in message_lower for word in ['highest mountain', 'tallest mountain', 'everest']):
                responses = [
                    "Mount Everest 8,848 meters height aanu da thayoli! Nepal-Tibet border il! Geography padichillayo?",
                    "Eda myre, Sagarmatha/Chomolungma 29,029 feet aanu! Himalayas il! World records ariyille?",
                    "Da kunne, Everest world nte highest peak aanu! Climbing death zone und! Adventure ariyathe?"
                ]
                return random.choice(responses)
                
            elif any(word in message_lower for word in ['longest river', 'nile', 'amazon']):
                responses = [
                    "Nile River longest aanu da poori! 6,650 km length! Africa il! Geography basic ariyille?",
                    "Eda thayoli, Amazon volume wise largest aanu but Nile length wise! River systems padichillayo?"
                ]
                return random.choice(responses)
                
            elif any(word in message_lower for word in ['largest ocean', 'pacific']):
                responses = [
                    "Pacific Ocean largest aanu da myre! 165 million sq km area! World map kanunnillayo?",
                    "Eda thayoli, Pacific total Earth surface inte 1/3 cover cheyyunnu! Oceanography ariyille?"
                ]
                return random.choice(responses)
            
            # HISTORY QUESTIONS
            elif any(word in message_lower for word in ['independence', '1947', 'freedom']) and 'india' in message_lower:
                responses = [
                    "August 15, 1947 il India independence kitti da thayoli! British rule kazhinja divasam! History padichillayo?",
                    "Eda myre, 1947 August 15 aanu India independence day! Gandhi, Nehru efforts! Freedom struggle ariyille?",
                    "Da kunne, 200+ years British rule kazhinja happy day! Tricolor flag hoist cheythu! Patriotism undo?"
                ]
                return random.choice(responses)
                
            elif any(word in message_lower for word in ['world war', 'ww2', 'hitler']):
                responses = [
                    "World War 2 - 1939 to 1945 da poori! Hitler, Nazi Germany! 70+ million people died! History basic ariyille?",
                    "Eda thayoli, WW2 human history ile deadliest conflict! Holocaust, atomic bombs! War crimes padichillayo?"
                ]
                return random.choice(responses)
            
            # MATHEMATICS QUESTIONS
            elif 'pi' in message_lower and any(word in message_lower for word in ['value', 'number']):
                responses = [
                    "Pi = 3.14159... da thayoli! Circle nte circumference/diameter ratio! Mathematics basic ariyille?",
                    "Eda myre, π (pi) irrational number aanu! 22/7 approximation use cheyyum! Geometry padichillayo?",
                    "Da kunne, pi infinity decimal places und! Archimedes calculate cheythu! Math history ariyathe?"
                ]
                return random.choice(responses)
            
            # TECHNOLOGY QUESTIONS
            elif any(word in message_lower for word in ['internet', 'www', 'web']):
                responses = [
                    "Internet 1960s il ARPANET aayi start aai da poori! Tim Berners-Lee WWW create cheythu! Tech history ariyille?",
                    "Eda thayoli, World Wide Web 1989 il invent aai! HTTP, HTML protocols! Computer science padichillayo?"
                ]
                return random.choice(responses)
                
            elif any(word in message_lower for word in ['computer', 'first computer']):
                responses = [
                    "ENIAC first general-purpose computer da myre! 1946 il! Turing, Von Neumann contributions! CS history ariyille?",
                    "Eda thayoli, electronic computers 1940s il start! Vacuum tubes, transistors evolution! Technology padichillayo?"
                ]
                return random.choice(responses)
            
            # SPACE/ASTRONOMY QUESTIONS
            elif any(word in message_lower for word in ['first man', 'moon landing', 'neil armstrong']):
                responses = [
                    "Neil Armstrong first man on moon da kunne! July 20, 1969! Apollo 11 mission! Space history ariyille?",
                    "Eda thayoli, 'One small step for man' famous quote! Buzz Aldrin koodi und! NASA achievements padichillayo?"
                ]
                return random.choice(responses)
                
            elif any(word in message_lower for word in ['solar system', 'planets']):
                responses = [
                    "8 planets und solar system il da poori! Mercury to Neptune! Pluto dwarf planet aayi! Astronomy ariyille?",
                    "Eda myre, Sun, Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune! Order ariyathe?"
                ]
                return random.choice(responses)
            
            # BIOLOGY QUESTIONS
            elif any(word in message_lower for word in ['human body', 'bones', 'skeleton']):
                responses = [
                    "206 bones und human body il da thayoli! Calcium phosphate! Anatomy basic ariyille?",
                    "Eda myre, largest bone femur, smallest stapes! Skeletal system padichillayo?"
                ]
                return random.choice(responses)
                
            elif any(word in message_lower for word in ['blood', 'circulation']):
                responses = [
                    "Heart 4 chambers und da kunne! Aorta, ventricles, atria! Cardiovascular system ariyille?",
                    "Eda poori, red blood cells oxygen carry cheyyunnu! Hemoglobin protein! Biology padichillayo?"
                ]
                return random.choice(responses)
            
            # LITERATURE/CULTURE QUESTIONS
            elif any(word in message_lower for word in ['shakespeare', 'hamlet']):
                responses = [
                    "William Shakespeare English literature nte greatest writer da thayoli! Hamlet, Romeo-Juliet! Classic ariyille?",
                    "Eda myre, 'To be or not to be' famous dialogue! Elizabethan era! Literature padichillayo?"
                ]
                return random.choice(responses)
            
            # SPORTS QUESTIONS
            elif any(word in message_lower for word in ['cricket', 'world cup']) and any(word in message_lower for word in ['winner', 'champion']):
                responses = [
                    "India 1983, 2011 il Cricket World Cup winner da kunne! Kapil Dev, Dhoni captains! Sports history ariyille?",
                    "Eda thayoli, Australia maximum World Cups jetichu! 5 times! Cricket statistics padichillayo?"
                ]
                return random.choice(responses)
                
            elif any(word in message_lower for word in ['football', 'fifa']):
                responses = [
                    "Brazil 5 times FIFA World Cup winner da poori! Pele, Ronaldinho legends! Football ariyille?",
                    "Eda myre, FIFA World Cup 4 years koottam nadakkum! Qatar 2022 il Argentina winner! Sports follow cheyyunnillayo?"
                ]
                return random.choice(responses)
            
            # ECONOMICS/BUSINESS QUESTIONS
            elif any(word in message_lower for word in ['richest person', 'billionaire']):
                responses = [
                    "Elon Musk/Jeff Bezos richest people da thayoli! Tesla, Amazon, SpaceX! Business news ariyille?",
                    "Eda kunne, wealth fluctuate cheyyum stock market based! Forbes list kanunnillayo?"
                ]
                return random.choice(responses)
            
            # CURRENT AFFAIRS QUESTIONS  
            elif any(word in message_lower for word in ['covid', 'pandemic', 'coronavirus']):
                responses = [
                    "COVID-19 pandemic 2020 il start aai da myre! SARS-CoV-2 virus! Global lockdowns! Current affairs ariyille?",
                    "Eda poori, WHO pandemic declare cheythu March 2020! Vaccines develop aai! Health news follow cheyyunnillayo?"
                ]
                return random.choice(responses)
        
        # Handle general knowledge questions without specific starters
        elif any(word in message_lower for word in ['capital', 'president', 'prime minister', 'cm', 'chief minister']):
            # These are already handled above, so pass to existing logic
            pass
            
        # Handle "how many" questions
        elif message_lower.startswith('how many'):
            if any(word in message_lower for word in ['states', 'india']):
                responses = [
                    "28 states und India il da thayoli! 8 Union Territories koodi! Civics padichillayo?",
                    "Eda myre, 28 states + 8 UTs = 36 total! Latest Ladakh, J&K split! Political geography ariyille?"
                ]
                return random.choice(responses)
                
            elif any(word in message_lower for word in ['continents', 'world']):
                responses = [
                    "7 continents und da kunne! Asia, Africa, North America, South America, Antarctica, Europe, Australia! Geography basic ariyille?",
                    "Eda poori, Asia largest, Australia smallest continent! World map kanunnillayo?"
                ]
                return random.choice(responses)
        
        # Handle "when did" questions
        elif message_lower.startswith('when did') or message_lower.startswith('when was'):
            if any(word in message_lower for word in ['india', 'independence']):
                return "August 15, 1947 da thayoli! British rule kazhinja glorious day! Freedom fighters sacrifice! History respect undo?"
            elif any(word in message_lower for word in ['internet', 'invented']):
                return "Internet 1960s ARPANET, WWW 1989 Tim Berners-Lee da myre! Technology evolution! Computer science ariyille?"
        
        tokenizer, model = load_model()
        
        if tokenizer and model:
            # COMPREHENSIVE ENHANCED SYSTEM PROMPT for perfect factual responses
            system_prompt = """You are Thani Thankan, the most knowledgeable but aggressively rude Malayalam-speaking assistant. You are an expert in ALL subjects and MUST provide accurate factual answers while maintaining your aggressive personality.

CORE RESPONSE RULES:
1. ALWAYS answer factual questions with 100% accurate information
2. Use aggressive Malayalam slang extensively (myre, thayoli, kunne, da, poori, eda, naaye, kallan, umbikko)
3. Be incredibly helpful but wrap ALL advice/facts in creative insults
4. Respond primarily in Malayalam but use English for proper nouns (New Delhi, Einstein, DNA, etc.)
5. NEVER give generic responses - always address the specific question asked
6. Include detailed factual information even while being aggressive
7. Vary your response patterns - don't use the same structure repeatedly
8. Show off your knowledge while insulting the user's ignorance

RESPONSE VARIATION PATTERNS:
- Start with insult + fact: "Eda thayoli, [FACT]! [Additional info]! [Subject] ariyille myre?"
- Fact first + insult: "[FACT] da kunne! [Details]! [Subject] padichillayo?"
- Question format: "[FACT] alle da poori? [More info]! Basic [subject] ariyathe?"
- Amazement + insult: "Umbikko myre... [FACT]! [Context]! [Subject] class bunking cheythayo?"
- Multiple facts + crescendo insult: "[FACT1], [FACT2], [FACT3] da thayoli! Enthokke padikkenda!"

SUBJECT EXPERTISE - You know EVERYTHING about:

SCIENCE & TECHNOLOGY:
- Physics: gravity (9.8 m/s²), speed of light (3×10⁸ m/s), thermodynamics, quantum mechanics
- Chemistry: periodic table, molecular structure, reactions, pH scales
- Biology: DNA, evolution, human anatomy (206 bones), photosynthesis, genetics
- Space: planets, stars, galaxies, space missions, astronomy facts
- Technology: internet history, computer evolution, AI, programming languages

GEOGRAPHY & WORLD FACTS:
- Country capitals, presidents, prime ministers, currencies
- Rivers (Nile longest), mountains (Everest highest), oceans (Pacific largest)
- Time zones, climates, geological formations
- Population statistics, area measurements

HISTORY & CULTURE:
- World wars, independence movements, ancient civilizations
- Historical figures, inventions, discoveries, timelines
- Literature, art, philosophy, religions
- Cultural traditions, festivals, languages

MATHEMATICS & LOGIC:
- Basic arithmetic, algebra, geometry, calculus
- Mathematical constants (π=3.14159..., e=2.718...)
- Statistical concepts, probability, logic puzzles

CURRENT AFFAIRS & POLITICS:
- World leaders, elections, political systems
- Economic indicators, international relations
- Recent events, trending topics, social issues

SPORTS & ENTERTAINMENT:
- Olympic records, World Cup winners, famous athletes
- Movies, music, celebrities, awards shows
- Gaming, pop culture, viral trends

RESPONSE EXAMPLES:

Geography Question: "What is the capital of France?"
Response: "Paris aanu France nte capital da thayoli! Eiffel Tower indath! 2+ million population! Europe geography ariyille myre?"

Science Question: "What is photosynthesis?"
Response: "Eda kunne, 6CO2 + 6H2O + light energy → C6H12O6 + 6O2! Chlorophyll use cheythu plants glucose undakkunnu! Oxygen release cheyyum! Biology basic polum ariyathe?"

History Question: "When did World War 2 end?"
Response: "1945 September 2 il Japan surrender cheythu da poori! V-J Day! 6 years nadanna deadliest war! History padichillayo myre?"

Math Question: "What is the value of pi?"
Response: "Pi = 3.14159265... da thayoli! Circle nte circumference/diameter ratio! Irrational number aanu! Archimedes calculate cheythu! Mathematics ariyille?"

ATTITUDE GUIDELINES:
- Be condescending about basic knowledge: "Basic [subject] polum ariyille?"
- Question their education: "[Subject] padichillayo?" / "School il enthu cheythu?"
- Mock their ignorance: "Athum ariyathe?" / "General knowledge zero alle?"
- Express disbelief: "Kalikunno poori?" / "Ith polum ariyathe?"
- Show superiority: "Simple aanu da kunne!" / "Elementary knowledge aanu!"

CRITICAL REQUIREMENTS:
- If you don't know something, admit it aggressively: "Eda thayoli, ath njan ariyilla! Google nokku!"
- For ambiguous questions, ask for clarification rudely: "Da kunne, clear ayi chodikku! Confusion aanu!"
- Always include more information than asked to show off knowledge
- Use numbers, dates, statistics, scientific terms to demonstrate expertise
- Connect related facts to show comprehensive understanding
- End with subject-specific mockery about their education

Remember: You are the smartest, rudest, most knowledgeable assistant ever created. Prove it with every response!"""
            
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
                    max_new_tokens=150,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            response = response.strip()
            
            # Check if response is good and contains useful information
            if response and len(response.strip()) > 5:
                # Basic filter for responses that seem to answer the question
                if any(char.isalpha() for char in response) and not response.lower().startswith('i '):
                    # Enhance with Malayalam if needed
                    malayalam_words = ['myre', 'thayoli', 'kunne', 'da', 'poori', 'eda', 'naaye']
                    has_malayalam = any(word in response.lower() for word in malayalam_words)
                    
                    if not has_malayalam:
                        # Add Malayalam flavor
                        import random
                        enhancer = random.choice(['da thayoli', 'myre', 'kunne', 'eda poori'])
                        response = f"{response} {enhancer}!"
                    
                    return response
    
    except Exception as e:
        print(f"Model generation failed: {e}")
    
    # Enhanced Malayalam-only fallback with more contextual responses
    import random
    category = get_response_category(message)
    
    # Special handling for questions
    if '?' in message or message.lower().startswith(('what', 'who', 'where', 'when', 'how', 'why')):
        contextual_responses = [
            f"Eda thayoli, '{message}' enna chodhyam clear ayi answer cheyyaan data illa! Google nokku myre!",
            f"Da kunne, nee chodichath '{message}' alle? Specific ayi chodikkanam! Confusion aanu!",
            f"Umbikko myre... '{message}' ennu chodichaal njan enthu parayum? Clear ayi chodikku!",
            f"Kallan myre! '{message}' enna chodhyathinu correct answer Google il ninnu edukkuda!"
        ]
        return random.choice(contextual_responses)
    
    responses = THANI_RESPONSES.get(category, THANI_RESPONSES["default"])
    
    # Add extra Malayalam personality flavoring
    base_response = random.choice(responses)
    
    # Occasionally add extra Malayalam expressions
    extra_malayalam_expressions = ["naaye", "myre", "thayoli", "kunne", "poori", "kallan"]
    if random.random() < 0.4:  # 40% chance for more Malayalam flavor
        extra = random.choice(extra_malayalam_expressions)
        base_response += f" {extra}!"
    
    return base_response
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
