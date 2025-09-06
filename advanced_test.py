"""
Advanced Comprehensive Factual Questions Test
Tests the new enhanced factual question handling with varied response patterns
"""

import subprocess
import sys

def install_gradio_client():
    """Install gradio_client if not available"""
    try:
        import gradio_client
        print("✅ gradio_client already installed")
        return True
    except ImportError:
        print("📦 Installing gradio_client...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio_client"])
            print("✅ gradio_client installed successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to install gradio_client: {e}")
            return False

def test_advanced_factual_questions():
    """Test comprehensive factual questions"""
    
    if not install_gradio_client():
        return
    
    try:
        from gradio_client import Client
        
        print("\n🔥 Testing Thani's Enhanced Factual Knowledge...")
        print("Testing varied response patterns and comprehensive knowledge")
        print("=" * 80)
        
        client = Client("Mojo-Maniac/thankan")
        
        # Advanced test cases covering multiple domains
        test_cases = [
            {
                "category": "🌟 SCIENCE",
                "questions": [
                    "What is the sun?",
                    "What is photosynthesis?", 
                    "What is DNA?",
                    "What is gravity?",
                    "What is the value of pi?"
                ]
            },
            {
                "category": "🌍 GEOGRAPHY",
                "questions": [
                    "What is the highest mountain?",
                    "What is the longest river?",
                    "What is the largest ocean?",
                    "How many continents are there?",
                    "What is the capital of France?"
                ]
            },
            {
                "category": "📚 HISTORY",
                "questions": [
                    "When did India get independence?",
                    "When did World War 2 happen?",
                    "Who is Shakespeare?",
                    "When was the internet invented?"
                ]
            },
            {
                "category": "🏛️ POLITICS",
                "questions": [
                    "Kerala Chief Minister aarade?",
                    "Who is the President of USA?",
                    "Indian Prime Minister aarade?",
                    "What is the capital of Karnataka?"
                ]
            },
            {
                "category": "🏃 SPORTS",
                "questions": [
                    "Who won the Cricket World Cup?",
                    "What is FIFA World Cup?",
                    "How many bones in human body?"
                ]
            },
            {
                "category": "💰 GENERAL",
                "questions": [
                    "How many states in India?",
                    "What is COVID?",
                    "Who is the richest person?",
                    "What is the first computer?"
                ]
            }
        ]
        
        total_questions = sum(len(cat["questions"]) for cat in test_cases)
        question_count = 0
        successful_responses = 0
        varied_responses = set()  # Track response variety
        
        for category_data in test_cases:
            category = category_data["category"]
            questions = category_data["questions"]
            
            print(f"\n{category}")
            print("-" * 60)
            
            for question in questions:
                question_count += 1
                print(f"\n{question_count}. Testing: '{question}'")
                
                try:
                    result = client.predict(
                        message=question,
                        history=[],
                        api_name="/chat_with_thani"
                    )
                    
                    if result and result[0] and len(result[0]) > 0:
                        user_msg, bot_response = result[0][-1]
                        print(f"   🤖 Response: '{bot_response}'")
                        
                        # Analyze response quality
                        response_lower = bot_response.lower()
                        
                        # Check for Malayalam slang
                        malayalam_words = ['myre', 'thayoli', 'kunne', 'da', 'poori', 'eda', 'naaye', 'kallan', 'umbikko']
                        found_malayalam = [word for word in malayalam_words if word in response_lower]
                        
                        # Check if it's not a generic response
                        generic_phrases = [
                            'enthuva myre? onnum manassilayilla',
                            'clear ayi chodhikku',
                            'njan poyi ente kaaryam nokkatte',
                            'kalikunno poori? enth parayendathu'
                        ]
                        is_generic = any(phrase in response_lower for phrase in generic_phrases)
                        
                        # Check for factual content (keywords that suggest real information)
                        factual_indicators = [
                            'aanu', 'alle', '=', 'million', 'billion', 'meter', 'degree', 
                            'year', '19', '20', 'cm', 'km', 'celsius', 'percent', '%'
                        ]
                        has_facts = any(indicator in response_lower for indicator in factual_indicators)
                        
                        # Response variety tracking
                        response_start = bot_response[:20].lower()
                        varied_responses.add(response_start)
                        
                        # Determine success
                        if found_malayalam and not is_generic and has_facts:
                            print(f"   ✅ EXCELLENT: Malayalam slang + factual content + specific answer")
                            successful_responses += 1
                        elif found_malayalam and has_facts:
                            print(f"   🟢 GOOD: Malayalam slang + some factual content")
                            successful_responses += 1
                        elif found_malayalam:
                            print(f"   🟡 OKAY: Malayalam slang but limited factual content")
                        else:
                            print(f"   ❌ POOR: Generic or insufficient response")
                        
                        if found_malayalam:
                            print(f"   💬 Slang words: {', '.join(found_malayalam)}")
                        
                    else:
                        print(f"   ❌ ERROR: Unexpected response format")
                        
                except Exception as e:
                    print(f"   ❌ ERROR: {str(e)}")
        
        # Final Analysis
        print(f"\n" + "=" * 80)
        print(f"📊 FINAL ANALYSIS:")
        print(f"   Total Questions Tested: {total_questions}")
        print(f"   Successful Responses: {successful_responses}")
        print(f"   Success Rate: {successful_responses/total_questions*100:.1f}%")
        print(f"   Response Variety: {len(varied_responses)} different patterns")
        
        if successful_responses >= total_questions * 0.8:
            print(f"\n🎉 EXCELLENT PERFORMANCE!")
            print(f"   ✅ Comprehensive factual knowledge working")
            print(f"   ✅ Malayalam slang integration successful") 
            print(f"   ✅ Varied response patterns achieved")
        elif successful_responses >= total_questions * 0.6:
            print(f"\n🟢 GOOD PERFORMANCE!")
            print(f"   ✅ Most factual questions handled well")
            print(f"   🟡 Some improvements possible")
        else:
            print(f"\n🟡 NEEDS IMPROVEMENT")
            print(f"   ❌ Many questions still giving generic responses")
            print(f"   💡 Consider updating Hugging Face Space with latest changes")
        
        print(f"\n💡 RECOMMENDATIONS:")
        if len(varied_responses) < total_questions * 0.5:
            print(f"   - Add more response pattern variety")
        if successful_responses < total_questions * 0.8:
            print(f"   - Ensure Hugging Face Space has latest app.py version")
            print(f"   - Check if comprehensive system prompt is active")
        
        print(f"\n🔥 Test Complete! Thani's knowledge has been thoroughly tested.")
        
    except Exception as e:
        print(f"❌ Failed to connect to API: {str(e)}")

if __name__ == "__main__":
    test_advanced_factual_questions()
