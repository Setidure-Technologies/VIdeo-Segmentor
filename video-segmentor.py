import os
import time
import json
import base64
import traceback
from io import BytesIO
from PIL import Image
from moviepy.video.io.VideoFileClip import VideoFileClip
from groq import Groq
import numpy as np

# --- CONFIGURATION ---
# We use two models: one for heavy text logic (Structure) and one for Vision (Content)
STRUCTURE_MODEL = "llama-3.3-70b-versatile" # Powerful text model for logic
WHISPER_MODEL = "whisper-large-v3"
VISION_MODEL_DEFAULT = "llama-3.2-90b-vision-preview" 

# --- PROMPTS ---

# 1. THE ARCHITECT: Identifies the structure from TRANSCRIPT
DISCOVERY_PROMPT = """
You are an expert Instructional Designer. Analyze the provided VIDEO TRANSCRIPT and break the video down into distinct learning modules (topics).
The transcript is provided with exact timestamps in the format: `[start-end]: text`.
**NOTE:** The transcript may be in **Hindi, English, or a mix (Hinglish)**. You must process the content and output the structure **strictly in English**.

Your goal is to identify the logical flow of the content and map it to specific time ranges.
For each module:
1.  Identify the main topic being discussed (In English).
2.  **CRITICAL**: Use the provided timestamp ranges to determine the exact start and end time.
3.  Ensure modules do not overlap and cover the entire meaningful content.

Return ONLY a raw JSON array:
[
  {"topic_name": "Introduction to React", "start_time": 0.0, "end_time": 15.5},
  {"topic_name": "Setting up the Environment", "start_time": 15.5, "end_time": 120.0},
  ...
]
"""

# 2. THE PROFESSOR: Creates the content
CONTENT_PROMPT_TEMPLATE = """
You are an expert Professor creating a concise course module for the topic: "{topic}".
Focus on the provided video frames which correspond to the segment from {start} seconds to {end} seconds.

Output strictly in Markdown with these specific headers. Keep content clear, concise, and bite-sized (Cue Card style).
**IMPORTANT:** Write all content **strictly in English**, even if the video/transcript is in Hindi or another language.

## Objectives
- Bullet points of what is learned.

## Notes
- Concise technical explanation.
- Bullet points for key concepts.

## Definitions
- Key terms defined briefly.

## Practical Application
- Real-world usage examples.

Do not include a quiz here. Do not header anything else.
"""

# 3. THE EXAMINER: Creates the final assessment
QUIZ_PROMPT = """
You are an expert Examiner. Create a Final Assessment Quiz based on the provided course transcript/content.
Create 5-10 multiple choice questions that test deep understanding.

Return ONLY a raw JSON array:
[
  {
    "question": "What is the primary function of...",
    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
    "correct_answer": "B) Option 2",
    "explanation": "Option 2 is correct because..."
  },
  ...
]
"""

class CourseGenerator:
    def __init__(self, api_key, model_name=None):
        self.api_key = api_key
        self.client = Groq(api_key=api_key)
        self.vision_model_name = model_name if model_name else VISION_MODEL_DEFAULT
        print(f"🌩️ Initialized. Structure: {STRUCTURE_MODEL}, Vision: {self.vision_model_name}")

    def extract_audio(self, video_path):
        """Extracts audio from video and saves as temp mp3."""
        print("   🔊 Extracting audio...")
        audio_path = "temp_audio.mp3"
        with VideoFileClip(video_path) as clip:
            clip.audio.write_audiofile(audio_path, logger=None)
        return audio_path

    def extract_frames_base64(self, video_path, start_time=0, end_time=None, interval=None, max_frames=5):
        """
        Extracts frames from the video.
        If interval is None, it calculates one to satisfy max_frames.
        Returns a list of Base64 strings (max max_frames).
        """
        print(f"   🎞️  Extracting frames from {start_time}s to {end_time if end_time else 'end'} (Max: {max_frames})...")
        frames_b64 = []
        with VideoFileClip(video_path) as clip:
            duration = clip.duration
            if end_time is None:
                end_time = duration
            
            # Clamp end_time
            if end_time > duration:
                end_time = duration
            
            current_duration = end_time - start_time
            if current_duration <= 0: return []
            
            # Calculate timestamps safely
            timestamps = np.linspace(start_time, end_time - 0.1, num=max_frames)
            
            for t in timestamps:
                try:
                    frame_np = clip.get_frame(t)
                    
                    # Convert to PIL Image
                    img = Image.fromarray(frame_np)
                    
                    # Resize to reduce token usage/latency
                    img.thumbnail((640, 640)) 
                    
                    # Convert to bytes then base64
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    frames_b64.append(img_str)
                except Exception as e:
                    print(f"Error extracting frame at {t}: {e}")
                
        print(f"   ✅ Extracted {len(frames_b64)} frames.")
        return frames_b64

    def analyze_structure(self, video_file):
        """Step 1: Get the timestamps via AUDIO TRANSCRIPTION."""
        print("🧠 Analyzing course structure (Audio-Based)...")
        
        try:
            # 1. Extract Audio
            audio_file = self.extract_audio(video_file)
            
            # 2. Transcribe with Timestamps
            print("   🗣️  Transcribing audio with timestamps...")
            with open(audio_file, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=(audio_file, file.read()),
                    model=WHISPER_MODEL,
                    response_format="verbose_json" # Request detailed segments
                )
            
            # Construct formatted transcript from segments
            transcript_text = ""
            if hasattr(transcription, 'segments'):
                for segment in transcription.segments:
                    start = segment['start']
                    end = segment['end']
                    text = segment['text'].strip()
                    transcript_text += f"[{start:.2f}s - {end:.2f}s]: {text}\n"
            else:
                # Fallback if standard json returned (unlikely with verbose_json)
                transcript_text = transcription.text

            print(f"   ✅ Interpretation complete. Transcript length: {len(transcript_text)} chars.")
            # For debugging, maybe save transcript?
            # with open("debug_transcript.txt", "w") as f: f.write(transcript_text)

            # 3. Analyze Transcript with Logic Model
            completion = self.client.chat.completions.create(
                model=STRUCTURE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": DISCOVERY_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"Here is the timestamped video transcript:\n\n{transcript_text}"
                    }
                ],
                temperature=0.1, # Lowest temp for strict logic
                response_format={"type": "json_object"} 
            )
            
            content = completion.choices[0].message.content
            # Clean up
            content = content.replace("```json", "").replace("```", "").strip()
            
            data = json.loads(content)
            
            # Additional parsing if nested
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, list):
                        data = value
                        break
            
            # Validation
            if not isinstance(data, list):
                print(f"❌ Error: Expected JSON array, got {type(data)}.")
                return []
                
            valid_modules = []
            for item in data:
                if isinstance(item, dict) and 'topic_name' in item and 'start_time' in item:
                    valid_modules.append(item)
                else:
                    print(f"⚠️ Skipping invalid item: {item}")
            
            # Cleanup temp audio
            if os.path.exists(audio_file):
                os.remove(audio_file)

            return valid_modules
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ Error during analysis: {str(e)}")
            return []

    def generate_module_content(self, video_file, topic, start, end):
        """Step 2: Generate the text course content for a specific segment using VISION."""
        print(f"   ✍️  Writing course content for: {topic}...")
        
        specific_prompt = CONTENT_PROMPT_TEMPLATE.format(
            topic=topic, start=start, end=end
        )
        
        # Extract frames limits to 5 for Groq Vision
        frames = self.extract_frames_base64(video_file, start_time=start, end_time=end, max_frames=5)
        
        content_parts = [{"type": "text", "text": specific_prompt}]
        for b64 in frames:
             content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}"
                }
            })
        
        try:
            completion = self.client.chat.completions.create(
                model=self.vision_model_name,
                messages=[
                    {
                        "role": "user",
                        "content": content_parts
                    }
                ],
                temperature=0.7 
            )
            return completion.choices[0].message.content
        except Exception as e:
            # Fallback for non-vision models (e.g. if user selected a text-only model)
            if "content" in str(e) and "string" in str(e):
                print(f"⚠️ Model {self.vision_model_name} appears to be text-only. Retrying without images.")
                try:
                    completion = self.client.chat.completions.create(
                        model=self.vision_model_name,
                        messages=[
                            {
                                "role": "user",
                                "content": specific_prompt
                            }
                        ],
                        temperature=0.7 
                    )
                    return completion.choices[0].message.content
                except Exception as e2:
                    print(f"❌ Error generating content (fallback): {e2}")
                    return f"Error generating content: {e2}"
            
            print(f"❌ Error generating content: {e}")
            return f"Error generating content: {e}"

    def generate_quiz(self, transcript_text):
        """Step 3: Generate the Final Assessment Quiz."""
        print("   🎓 Generating Final Assessment...")
        try:
            completion = self.client.chat.completions.create(
                model=STRUCTURE_MODEL,
                messages=[
                    {
                         "role": "system",
                         "content": QUIZ_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"Here is the full course transcript. Generate the quiz based on this:\n\n{transcript_text[:15000]}" # LIMIT context to avoid overflow if very long
                    }
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            content = completion.choices[0].message.content
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            print(f"❌ Error generating quiz: {e}")
            return []

    def process_video(self, source_path, output_dir="course_output"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. Analyze Structure
        modules = self.analyze_structure(source_path) 
        
        if not modules:
            print("❌ No modules generated.")
            return

        print(f"\n📋 Course Plan: Found {len(modules)} modules.")
        
        # 3. Process each module
        with VideoFileClip(source_path) as video:
            for idx, module in enumerate(modules):
                topic_clean = module['topic_name'].replace(" ", "_").replace("/", "-")
                base_name = f"{idx+1}_{topic_clean}"
                
                start = float(module['start_time'])
                end = float(module['end_time'])
                
                print(f"\n--- Processing Module {idx+1}: {module['topic_name']} ({start}s - {end}s) ---")

                # A. CUT THE VIDEO
                video_filename = f"{base_name}.mp4"
                save_path_video = os.path.join(output_dir, video_filename)
                
                # Safety check for duration
                if end > video.duration: end = video.duration
                
                if start < end:
                    # moviepy subclip
                    new_clip = video.subclipped(start_time=start, end_time=end)
                    new_clip.write_videofile(save_path_video, codec="libx264", audio_codec="aac", logger=None)
                    print(f"   ✅ Video Clip Saved")
                else:
                    print(f"   ⚠️ Invalid duration (Start: {start}, End: {end}). Skipping clip.")
                
                # B. WRITE THE COURSE CONTENT
                course_content = self.generate_module_content(source_path, module['topic_name'], start, end)
                
                md_filename = f"{base_name}.md"
                save_path_md = os.path.join(output_dir, md_filename)
                
                with open(save_path_md, "w", encoding="utf-8") as f:
                    f.write(course_content)
                print(f"   ✅ Course Text Saved")
                
        print("\n🎉 Course Generation Complete!")

# --- EXECUTION ---
if __name__ == "__main__":
    SOURCE = "your_video.mp4" 
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

    if os.path.exists(SOURCE) and GROQ_API_KEY:
        generator = CourseGenerator(api_key=GROQ_API_KEY)
        generator.process_video(SOURCE)
    else:
        print("File not found or GROQ_API_KEY not set.")