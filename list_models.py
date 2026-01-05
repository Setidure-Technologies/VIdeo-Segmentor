import google.generativeai as genai
import importlib.util
import os

# Import from video-segmentor.py
file_path = "video-segmentor.py"
module_name = "video_segmentor"
spec = importlib.util.spec_from_file_location(module_name, file_path)
video_segmentor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video_segmentor)
API_KEY = video_segmentor.API_KEY

genai.configure(api_key=API_KEY)

print("Listing available models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error listing models: {e}")
