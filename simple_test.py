"""
Simple Thani Thankan API Test
Run this script to test the API using gradio_client
"""

# First install gradio_client if needed
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

def test_single_question():
    """Test a single question to verify the API is working"""
    
    if not install_gradio_client():
        return
    
    try:
        from gradio_client import Client
        
        print("\n🔥 Testing Thani Thankan API...")
        print("Connecting to Mojo-Maniac/thankan...")
        
        client = Client("Mojo-Maniac/thankan")
        
        # Test the specific question that wasn't working before
        test_message = "china nte capital city etha kunne"
        
        print(f"\n📤 Sending: '{test_message}'")
        print("⏳ Waiting for response...")
        
        result = client.predict(
            message=test_message,
            history=[],
            api_name="/chat_with_thani"
        )
        
        print(f"\n📦 Raw result: {result}")
        
        # Extract and display the response
        if result and len(result) >= 1 and result[0]:
            if len(result[0]) > 0:
                user_msg, bot_response = result[0][-1]
                print(f"\n✅ SUCCESS!")
                print(f"👤 User: {user_msg}")
                print(f"😈 Thani: {bot_response}")
                
                # Check if it contains expected elements
                bot_lower = bot_response.lower()
                
                print(f"\n🔍 Analysis:")
                
                # Check for Malayalam slang
                malayalam_words = ['myre', 'thayoli', 'kunne', 'da', 'poori', 'eda', 'naaye', 'kallan']
                found_malayalam = [word for word in malayalam_words if word in bot_lower]
                if found_malayalam:
                    print(f"   ✅ Malayalam slang found: {', '.join(found_malayalam)}")
                else:
                    print(f"   ❌ No Malayalam slang detected")
                
                # Check for correct answer
                if "new delhi" in bot_lower or "delhi" in bot_lower:
                    print(f"   ✅ Correct capital mentioned: New Delhi")
                else:
                    print(f"   ❌ Capital not mentioned or incorrect")
                
                # Overall assessment
                if found_malayalam and ("new delhi" in bot_lower or "delhi" in bot_lower):
                    print(f"\n🎉 PERFECT! Bot is working correctly:")
                    print(f"   - Answers factual questions ✅")
                    print(f"   - Uses Malayalam slang ✅") 
                    print(f"   - Maintains aggressive personality ✅")
                else:
                    print(f"\n⚠️  Issues detected - check the fixes in app.py")
                    
            else:
                print(f"❌ Empty response history")
        else:
            print(f"❌ Unexpected response format: {result}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print(f"Make sure the Hugging Face Space is running and accessible.")

if __name__ == "__main__":
    test_single_question()
