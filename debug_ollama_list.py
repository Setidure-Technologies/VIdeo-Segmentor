import ollama
import json

try:
    print("Fetching models info...")
    models_info = ollama.list()
    print(f"Type: {type(models_info)}")
    print(f"Raw Output: {models_info}")
    
    # Try to simulate the failing access
    # print(f"First item model: {models_info['models'][0]['model']}")
except Exception as e:
    print(f"Error: {e}")
