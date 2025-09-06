"""
Comprehensive Thani Thankan API Test
Tests multiple factual questions to verify which ones work
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

def test_multiple_questions():
    """Test multiple factual questions"""
    
    if not install_gradio_client():
        return
    
    try:
        from gradio_client import Client
        
        print("\n🔥 Testing Thani Thankan API with Multiple Questions...")
        print("Connecting to Mojo-Maniac/thankan...")
        
        client = Client("Mojo-Maniac/thankan")
        
        # Test cases - various factual questions
        test_cases = [
            {
                "question": "india nte capital city etha kunne",
                "expected_answer": "New Delhi",
                "category": "India Capital"
            },
            {
                "question": "USA nte capital city etha kunne", 
                "expected_answer": "Washington DC",
                "category": "USA Capital"
            },
            {
                "question": "what is India capital",
                "expected_answer": "New Delhi", 
                "category": "India Capital (English)"
            },
            {
                "question": "what is USA capital",
                "expected_answer": "Washington DC",
                "category": "USA Capital (English)"
            },
            {
                "question": "kerala capital",
                "expected_answer": "Thiruvananthapuram",
                "category": "Kerala Capital"
            },
            {
                "question": "President of USA",
                "expected_answer": "Joe Biden",
                "category": "USA President"
            },
            {
                "question": "indian president aara",
                "expected_answer": "Droupadi Murmu", 
                "category": "India President"
            },
            {
                "question": "what is 5+3",
                "expected_answer": "8",
                "category": "Math"
            },
            {
                "question": "Who are you?",
                "expected_answer": "Thankan",
                "category": "Identity"
            }
        ]
        
        print(f"\n🧪 Testing {len(test_cases)} different questions...")
        print("=" * 80)
        
        working_count = 0
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n{i}. {test['category']}")
            print(f"   📤 Question: '{test['question']}'")
            print(f"   🎯 Expected: {test['expected_answer']}")
            
            try:
                result = client.predict(
                    message=test['question'],
                    history=[],
                    api_name="/chat_with_thani"
                )
                
                if result and result[0] and len(result[0]) > 0:
                    user_msg, bot_response = result[0][-1]
                    print(f"   💬 Response: '{bot_response}'")
                    
                    # Check if response contains expected answer
                    response_lower = bot_response.lower()
                    expected_lower = test['expected_answer'].lower()
                    
                    # Check for Malayalam slang
                    malayalam_words = ['myre', 'thayoli', 'kunne', 'da', 'poori', 'eda', 'naaye', 'kallan']
                    has_malayalam = any(word in response_lower for word in malayalam_words)
                    
                    # Check for correct answer
                    has_correct_answer = expected_lower in response_lower
                    
                    # Special case checks
                    if test['category'] == "Math" and "8" in bot_response:
                        has_correct_answer = True
                    elif test['category'] == "Identity" and "thankan" in response_lower:
                        has_correct_answer = True
                    
                    # Determine if working correctly
                    is_working = has_malayalam and has_correct_answer
                    
                    if is_working:
                        print(f"   ✅ STATUS: WORKING - Has Malayalam slang + correct answer")
                        working_count += 1
                    elif has_correct_answer:
                        print(f"   🟡 STATUS: PARTIAL - Correct answer but no Malayalam slang")
                    elif has_malayalam:
                        print(f"   🟡 STATUS: PARTIAL - Malayalam slang but wrong/no answer")
                    else:
                        print(f"   ❌ STATUS: NOT WORKING - Generic response")
                    
                else:
                    print(f"   ❌ STATUS: ERROR - Unexpected response format")
                    
            except Exception as e:
                print(f"   ❌ STATUS: ERROR - {str(e)}")
            
            print("   " + "-" * 70)
        
        # Summary
        print(f"\n📊 SUMMARY:")
        print(f"   Working correctly: {working_count}/{len(test_cases)} ({working_count/len(test_cases)*100:.1f}%)")
        
        if working_count == len(test_cases):
            print(f"   🎉 ALL TESTS PASSED! Bot is working perfectly!")
        elif working_count > len(test_cases) // 2:
            print(f"   🟡 MOST TESTS PASSED - Some improvements needed")
        else:
            print(f"   ❌ MAJOR ISSUES - Need to update the Hugging Face Space with new code")
        
        print(f"\n💡 NEXT STEPS:")
        if working_count < len(test_cases):
            print(f"   1. The local app.py has been fixed")
            print(f"   2. Need to push changes to Hugging Face Space")
            print(f"   3. Or restart the Space to load new code")
            print(f"   4. The Space might be running an older version of the code")
        
    except Exception as e:
        print(f"❌ Failed to connect to API: {str(e)}")

if __name__ == "__main__":
    test_multiple_questions()
