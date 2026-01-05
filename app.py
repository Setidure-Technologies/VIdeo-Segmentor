import streamlit as st
import os
import time
import json
import importlib.util
from moviepy.video.io.VideoFileClip import VideoFileClip

# Import the CourseGenerator class from video-segmentor.py
# We use importlib because the filename references a hyphen, which is not a valid identifier
module_name = "video_segmentor"
file_path = "video-segmentor.py"

spec = importlib.util.spec_from_file_location(module_name, file_path)
video_segmentor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video_segmentor)
CourseGenerator = video_segmentor.CourseGenerator

# Page Config
st.set_page_config(
    page_title="Video Segmentor & Course Creator",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    
    /* Cue Cards */
    .cue-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .cue-title {
        font-weight: bold;
        font-size: 1.1em;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Colors */
    .card-red { background-color: #FFEBEE; border-left: 5px solid #D32F2F; color: #B71C1C; }
    .card-green { background-color: #E8F5E9; border-left: 5px solid #388E3C; color: #1B5E20; }
    .card-blue { background-color: #E3F2FD; border-left: 5px solid #1976D2; color: #0D47A1; }
    .card-yellow { background-color: #FFFDE7; border-left: 5px solid #FBC02D; color: #F57F17; }
</style>
""", unsafe_allow_html=True)

def parse_markdown_to_cards(markdown_text):
    """Parses standard markdown into sections for cue cards."""
    sections = {}
    current_header = None
    buffer = []
    
    for line in markdown_text.split('\n'):
        if line.startswith("## "):
            if current_header:
                sections[current_header] = "\n".join(buffer).strip()
            current_header = line.replace("## ", "").strip().lower()
            buffer = []
        else:
            buffer.append(line)
    
    # Last section
    if current_header:
        sections[current_header] = "\n".join(buffer).strip()
        
    return sections

def render_cue_card(title, content, color_class):
    if content:
        st.markdown(f"""
        <div class="cue-card {color_class}">
            <div class="cue-title">{title}</div>
            <div style="white-space: pre-line;">{content}</div>
        </div>
        """, unsafe_allow_html=True)

# Title
st.title("🎬 Video Segmentor & Course Creator")
st.markdown("Upload a video tutorial to automatically segment it, generate learning modules, and create a comprehensive course.")

# Sidebar for Configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Groq API Key", type="password", help="Enter your Groq Cloud API Key")
    
    selected_model = "llama-3.2-90b-vision-preview" # Default
    if api_key:
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            models = client.models.list()
            # Filter for vision capable models if possible, or just list all
            # Ideally we'd filter, but for now we list them.
            # Prioritize Llama 4 Scout if available
            model_options = [m.id for m in models.data]
            
            # Simple heuristic to put desired models at top
            priority = ["llama-4-scout", "llama-3.2-90b-vision-preview", "llama-3.2-11b-vision-preview"]
            sorted_models = sorted(model_options, key=lambda x: (x not in priority, x))
            
            selected_model = st.selectbox("Select Groq Model", sorted_models, index=0)
        except Exception as e:
            st.error(f"Error fetching models: {e}")
            
    # st.info("💡 **How it works:**\n\n1. Upload a video.\n2. App extracts frames.\n3. AI identifies topic segments.\n4. App cuts the video into clips.\n5. AI generates course content for each clip.")

# Main Content
uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi', 'mkv'])

if uploaded_file and api_key:
    # Save the uploaded file temporarily
    temp_filename = "temp_upload.mp4"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.video(uploaded_file)
    
    if st.button("🚀 Generate Course", type="primary"):
        output_dir = "course_output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        generator = CourseGenerator(api_key=api_key, model_name=selected_model)
        
        # 1. Processing
        with st.status("Processing Video...", expanded=True) as status:
            st.write("⬆️ Prepare video...")
            gemini_file = temp_filename 
            
            # 2. Analyze Structure
            st.write(f"🎧 Extracting Audio & Transcribing (Groq Whisper)...")
            st.write(f"🧠 Analyzing Topics (Llama 3 70B)...")
            
            modules = generator.analyze_structure(gemini_file)
            
            if not modules:
                status.update(label="Analysis Failed", state="error")
                st.error("Could not analyze video structure. Check API Key or video content.")
                st.stop()
                
            st.write(f"📋 Found {len(modules)} topic-based modules.")
            status.update(label="Structure Analyzed", state="complete")
        


        # 3. Generate Modules
        results_container = st.container()
        progress_bar = st.progress(0)
        
        with VideoFileClip(temp_filename) as video:
            for idx, module in enumerate(modules):
                progress = (idx) / len(modules)
                progress_bar.progress(progress, text=f"Processing Module {idx+1}/{len(modules)}: {module['topic_name']}")
                
                topic_clean = module['topic_name'].replace(" ", "_").replace("/", "-")
                base_name = f"{idx+1}_{topic_clean}"
                
                col1, col2 = results_container.columns([1, 1])
                
                with col1:
                    st.subheader(f"Module {idx+1}: {module['topic_name']}")
                    st.caption(f"Time: {module['start_time']}s - {module['end_time']}s")
                    
                    # Cut Video
                    start = float(module['start_time'])
                    end = float(module['end_time'])
                    if end > video.duration: end = video.duration
                    
                    video_filename = f"{base_name}.mp4"
                    save_path_video = os.path.join(output_dir, video_filename)
                    
                    if start < end:
                        if not os.path.exists(save_path_video): # Avoid re-processing if exists
                            new_clip = video.subclipped(start_time=start, end_time=end)
                            new_clip.write_videofile(save_path_video, codec="libx264", audio_codec="aac", logger=None)
                        st.video(save_path_video)
                
                with col2:
                    st.write("✍️ Generating Content...")
                    # Generate Text
                    course_content = generator.generate_module_content(gemini_file, module['topic_name'], start, end)
                    
                    md_filename = f"{base_name}.md"
                    save_path_md = os.path.join(output_dir, md_filename)
                    with open(save_path_md, "w", encoding="utf-8") as f:
                        f.write(course_content)
                    
                    with st.expander("View Course Content (Cue Cards)", expanded=True):
                        # Parse
                        sections = parse_markdown_to_cards(course_content)
                        
                        # Render Specific Cards
                        # Objectives -> Red
                        render_cue_card("Objectives", sections.get('objectives', ''), 'card-red')
                        
                        # Notes -> Green
                        render_cue_card("Notes", sections.get('notes', ''), 'card-green')
                        
                        # Definitions -> Blue
                        render_cue_card("Definitions", sections.get('definitions', ''), 'card-blue')
                        
                        # Practical -> Yellow
                        render_cue_card("Practical Application", sections.get('practical application', ''), 'card-yellow')
                        
                        # Fallback for text download
                        st.download_button(
                            label="Download Raw Markdown",
                            data=course_content,
                            file_name=md_filename,
                            mime="text/markdown",
                            key=f"dl_{idx}"
                        )
                
                # Rate limit wait
                if idx < len(modules) - 1:
                    time.sleep(5)

        progress_bar.progress(1.0, text="Completed!")
        st.success("🎉 Course Generation Complete!")
        
        # --- FINAL ASSESSMENT ---
        st.divider()
        st.header("🎓 Final Assessment")
        with st.spinner("Generating Final Quiz..."):
             # We need a full transcript text for the quiz context
             # Since we don't store it, we can approximate by chaining the modules
             full_content_context = "\n".join([f"Topic: {m['topic_name']}\nContent: {generator.generate_module_content(gemini_file, m['topic_name'], m['start_time'], m['end_time'])}" for m in modules[:3]]) # Just sample first few for context or re-read transcript if possible. 
             # Actually, better: we already have the `modules` which implies the structure. Let's just ask for a quiz based on the *structure* or extract the full transcript again? 
             # To be efficient and accurate, let's allow the generator to re-read the temp audio? No, audio is gone.
             # Best approach: We should have saved the transcript text.
             # FIX: Let's assume the user wants a quiz on what they just saw.
             
             # For now, let's generate it based on the *module titles* and *content* we just wrote.
             # We can read the MD files? 
             all_md_content = ""
             for file in os.listdir(output_dir):
                 if file.endswith(".md"):
                     with open(os.path.join(output_dir, file), "r", encoding="utf-8") as f:
                         all_md_content += f.read() + "\n"
                         
             quiz_data = generator.generate_quiz(all_md_content)
             st.session_state['quiz_data'] = quiz_data
             st.session_state['quiz_started'] = True

if st.session_state.get('quiz_started'):
    st.markdown("---")
    st.subheader("📝 Final Exam")
    
    quiz = st.session_state['quiz_data']
    score = 0
    
    with st.form("quiz_form"):
        user_answers = {}
        for i, q in enumerate(quiz):
            st.markdown(f"**Q{i+1}: {q['question']}**")
            user_answers[i] = st.radio(f"Select answer for Q{i+1}", q['options'], key=f"q_{i}", index=None)
            st.divider()
            
        submitted = st.form_submit_button("Submit Assessment")
        
        if submitted:
            st.balloons()
            for i, q in enumerate(quiz):
                user_ans = user_answers.get(i)
                correct_ans = q['correct_answer']
                
                if user_ans == correct_ans:
                    score += 1
                    st.success(f"✅ Q{i+1} Correct! {q['explanation']}")
                else:
                    st.error(f"❌ Q{i+1} Incorrect. Correct: {correct_ans}. {q['explanation']}")
            
            final_score = (score / len(quiz)) * 100
            st.metric("Final Score", f"{final_score:.0f}%")
            
            if final_score >= 80:
                st.markdown("### 🌟 Amazing Job! You're a pro!")
            elif final_score >= 50:
                st.markdown("### 💪 Good effort! Review the notes and try again.")
            else:
                st.markdown("### 📚 Keep studying! You can do it!")
    
elif not uploaded_file:
    st.info("Please upload a video to start.")
