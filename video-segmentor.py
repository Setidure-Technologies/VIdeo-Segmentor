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
You are an expert Instructional Designer. Analyze the provided VIDEO TRANSCRIPT and identify the core learning modules.
The transcript is provided with exact timestamps in the format: `[start-end]: text`.
**NOTE:** The transcript may be in **Hindi, English, or a mix (Hinglish)**. You must process the content and output the structure **strictly in English**.

### CRITICAL PHILOSOPHY: BE A "LUMPER", NOT A "SPLITTER"
Your goal is to create the **FEWEST** number of modules necessary to cover the content effectively.
Avoid fragmentation. A module must be a **complete, standalone lesson**.

### RULES FOR SEGMENTATION:
1.  **CONCEPT COMPLETENESS**:
    - **BAD**: Mod 1: "Definition" (30s), Mod 2: "Types" (30s), Mod 3: "Example" (30s). -> *User is confused.*
    - **GOOD**: Mod 1: "Complete Guide to [Topic]" (90s). -> *Includes definition, types, and examples.*
    - **RULE**: Always MERGE "Introduction", "Definition", "Mechanism", "Types", "Advantages", and "Examples" of the same subject into ONE single module.

2.  **STRICT DURATION LOGIC**:
    - **Video < 5 Minutes**: Create exactly **ONE** module. (Unless there is a hard topic switch like "Sports" to "Cooking").
    - **Video > 5 Minutes**: Create modules that are **at least 2 minutes long** if possible.
    - If a potential segment is under 60 seconds, **IT IS TOO SHORT**. Merge it with the previous or next module.

3.  **PROCEDURAL FLOW**:
    - If the video is a step-by-step tutorial ("Step 1, Step 2, Step 3"), keep all steps in **ONE** module called "Process of X", unless the process is extremely long (>10 mins).

For each module:
1.  Identify the main topic being discussed (In English).
2.  **CRITICAL**: Use the provided timestamp ranges to determine the exact start and end time.
3.  Ensure modules do not overlap and cover the entire meaningful content.

Return ONLY a raw JSON array:
[
  {"topic_name": "Comprehensive Guide to React Components", "start_time": 0.0, "end_time": 180.5},
  ...
]
"""

# 2. THE PROFESSOR: Creates the content (NOTES ONLY)
CONTENT_PROMPT_TEMPLATE = """
You are an expert Professor creating a concise course module for the topic: "{topic}".
Focus on the provided video frames AND the following TRANSCRIPT SEGMENT:

<TRANSCRIPT_SEGMENT>
{transcript_segment}
</TRANSCRIPT_SEGMENT>

The segment corresponds to the time {start} seconds to {end} seconds.
Use the transcript as the PRIMARY source of information.

Output strictly in Markdown. Keep content clear, concise, and bite-sized (Cue Card style).
**IMPORTANT:** Write all content **strictly in English**.

## Notes
- Concise technical explanation.
- Bullet points for key concepts.
"""

# 3. THE ARCHITECT - GLOBAL INTRO
INTRO_PROMPT = """
You are an expert Instructional Designer. Analyze the FULL COURSE TRANSCRIPT provided below.
Identify the 3-5 distinct learning objectives for this entire video course.

Output strictly in Markdown:
## Objectives
- Bullet point 1
- Bullet point 2...
"""

# 4. THE ARCHITECT - GLOBAL OUTRO
OUTRO_PROMPT = """
You are an expert Instructional Designer. Analyze the FULL COURSE TRANSCRIPT provided below.
Extract all key Definitions and specific Practical Applications mentioned or implied in the course.

Output strictly in Markdown:
## Definitions
- **Term**: Definition...

## Practical Application
- Real-world usage examples...
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
                return [], ""
                
            valid_modules = []
            for item in data:
                if isinstance(item, dict) and 'topic_name' in item and 'start_time' in item:
                    valid_modules.append(item)
                else:
                    print(f"⚠️ Skipping invalid item: {item}")
            
            # Cleanup temp audio
            if os.path.exists(audio_file):
                os.remove(audio_file)

            # --- POST-PROCESSING: SMART MERGE ---
            print(f"   🧹 Post-processing: Merging short segments (under 60s)...")
            final_modules = self.smart_merge_modules(valid_modules, min_duration=60)
            print(f"   ✅ Merged {len(valid_modules)} -> {len(final_modules)} modules.")

            return final_modules, transcript_text
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ Error during analysis: {str(e)}")
            return [], ""

    def smart_merge_modules(self, modules, min_duration=60):
        """
        Iteratively merges modules smaller than min_duration.
        Strategy:
        - If module < min_duration:
          - Merge with PREVIOUS if exists (preferred for sub-points).
          - Else merge with NEXT.
        - Name Priority: Keep the name of the LONGER segment (Topic Integrity).
        """
        if not modules: return []
        
        # Make a copy to avoid mutating original list while iterating (though we restart loop)
        current_modules = modules.copy()
        
        while True:
            # 1. Check if any module is too short
            too_short_index = -1
            shortest_duration = float('inf')
            
            for i, mod in enumerate(current_modules):
                dur = mod['end_time'] - mod['start_time']
                if dur < min_duration:
                    if dur < shortest_duration:
                        shortest_duration = dur
                        too_short_index = i
            
            # If no short modules found, or only 1 module left, we are done
            if too_short_index == -1 or len(current_modules) <= 1:
                break
                
            # 2. Merge Logic
            # We found a short module at `too_short_index`.
            target_idx = too_short_index
            
            # Decide neighbor: Prev (i-1) or Next (i+1)
            # Default: Merge into PREV (e.g. Example merges into Definition)
            # But if it's the first one, must merge into NEXT.
            
            output_modules = []
            
            if target_idx > 0:
                # Merge with PREV
                prev_mod = current_modules[target_idx - 1]
                short_mod = current_modules[target_idx]
                
                # Check who is dominant (Longer duration wins name)
                prev_dur = prev_mod['end_time'] - prev_mod['start_time']
                short_dur = short_mod['end_time'] - short_mod['start_time']
                
                new_name = prev_mod['topic_name'] if prev_dur >= short_dur else short_mod['topic_name']
                
                merged_mod = {
                    "topic_name": new_name,
                    "start_time": prev_mod['start_time'],
                    "end_time": short_mod['end_time']
                }
                
                # Reconstruct list
                # All before prev + new + all after short
                output_modules = current_modules[:target_idx-1] + [merged_mod] + current_modules[target_idx+1:]
                
                print(f"      🔹 Merged '{short_mod['topic_name']}' ({short_dur:.1f}s) ⬅️ INTO '{prev_mod['topic_name']}'")
                
            else:
                # Must be index 0, Merge with NEXT
                short_mod = current_modules[0]
                next_mod = current_modules[1]
                
                short_dur = short_mod['end_time'] - short_mod['start_time']
                next_dur = next_mod['end_time'] - next_mod['start_time']
                
                new_name = next_mod['topic_name'] if next_dur >= short_dur else short_mod['topic_name']
                
                merged_mod = {
                    "topic_name": new_name,
                    "start_time": short_mod['start_time'],
                    "end_time": next_mod['end_time']
                }
                
                output_modules = [merged_mod] + current_modules[2:]
                
                print(f"      🔹 Merged '{short_mod['topic_name']}' ({short_dur:.1f}s) INTO ➡️ '{next_mod['topic_name']}'")
            
            current_modules = output_modules
            
        return current_modules

    def generate_module_content(self, video_file, topic, start, end, transcript_text):
        """Step 2: Generate the text course content for a specific segment using VISION + TRANSCRIPT."""
        print(f"   ✍️  Writing course content for: {topic}...")
        
        # Filter transcript for context (naive filter or just pass relevant chunk if we have it, 
        # but simplest is to pass the whole thing and let LLM find the timestamps or just pass the relevant snippet)
        # For robustness, we will extract lines that fall roughly within the start/end window.
        
        relevant_lines = []
        try:
            lines = transcript_text.split('\n')
            for line in lines:
                # Expected format: [start - end]: text
                if "[" in line and "]" in line:
                    try:
                        time_part = line.split("]")[0].replace("[", "")
                        t_start, t_end = map(float, time_part.replace("s", "").split("-"))
                        
                        # basic overlap check
                        if t_start >= start and t_start <= end:
                            relevant_lines.append(line)
                    except:
                        continue
        except:
            relevant_lines = [transcript_text] # Fallback

        transcript_segment = "\n".join(relevant_lines)
        if not transcript_segment: description = "No speech detected in this segment."

        specific_prompt = CONTENT_PROMPT_TEMPLATE.format(
            topic=topic, start=start, end=end, transcript_segment=transcript_segment
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

    def generate_course_intro(self, transcript_text):
        """Generates global course objectives."""
        print("   🚀 Generating Course Objectives...")
        try:
            return self._text_completion(INTRO_PROMPT, transcript_text)
        except Exception as e:
            print(f"Error generating intro: {e}")
            return "## Objectives\n- content generation failed."

    def generate_course_outro(self, transcript_text):
        """Generates global definitions and practical applications."""
        print("   🏁 Generating Course Outro...")
        try:
            return self._text_completion(OUTRO_PROMPT, transcript_text)
        except Exception as e:
            print(f"Error generating outro: {e}")
            return "## Definitions\n- None\n\n## Practical Application\n- None"

    def _text_completion(self, system_prompt, user_content):
        """Helper for text-only completions."""
        completion = self.client.chat.completions.create(
            model=STRUCTURE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3
        )
        return completion.choices[0].message.content

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
                        "content": f"Here is the full course transcript. Generate the quiz strictly based on this content:\n\n{transcript_text}" 
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
        modules, transcript_text = self.analyze_structure(source_path) 
        
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
                course_content = self.generate_module_content(source_path, module['topic_name'], start, end, transcript_text)
                
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