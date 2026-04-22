import google.generativeai as genai
import os

# 1. Configure your API key
# If you didn't set an environment variable, replace os.environ.get(...) with your actual string key
api_key = os.environ.get("GEMINI_API_KEY") 
genai.configure(api_key=api_key)

try:
    # 2. Initialize the Gemini 2.5 Flash model
    print("Connecting to gemini-2.5-flash...")
    model = genai.GenerativeModel('gemini-2.5-flash-lite')

    # 3. Send a simple ping
    response = model.generate_content("Hello!.")

    # 4. Print the result
    print("\n✅ Success! Received response:")
    print("-" * 40)
    print(response.text)
    print("-" * 40)

except Exception as e:
    print(f"\n❌ Error connecting to the model:\n{e}")