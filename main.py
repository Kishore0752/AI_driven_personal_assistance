import streamlit as st
import os
import pytesseract
import cv2
import numpy as np
from PIL import Image
from groq import Groq
from dotenv import load_dotenv
import pyttsx3
import speech_recognition as sr
import threading
import time
import json
from datetime import datetime
import io
import base64

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Page Configuration
st.set_page_config(
    page_title="AI Assistant Pro",
    page_icon="robot",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for ChatGPT-like styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f1f1f;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 2rem;
    }
    
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        margin-bottom: 1rem;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        margin-left: 20%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        margin-right: 20%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online { background-color: #4CAF50; }
    .status-offline { background-color: #f44336; }
    .status-processing { background-color: #ff9800; }
    
    .input-container {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .sidebar-content {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
@st.cache_resource
def init_groq_client():
    """Initialize Groq client with error handling"""
    if not GROQ_API_KEY:
        return None
    try:
        return Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {e}")
        return None

@st.cache_resource
def init_tts_engine():
    """Initialize text-to-speech engine"""
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        if voices:
            # Try to set a female voice if available
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
        engine.setProperty('rate', 160)
        engine.setProperty('volume', 0.9)
        return engine
    except Exception as e:
        st.warning(f"TTS engine initialization failed: {e}")
        return None

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'conversation_count' not in st.session_state:
    st.session_state.conversation_count = 0
if 'favorite_responses' not in st.session_state:
    st.session_state.favorite_responses = []
if 'current_session_start' not in st.session_state:
    st.session_state.current_session_start = datetime.now()

# Initialize clients
client = init_groq_client()
engine = init_tts_engine()

def get_system_status():
    """Check system component status"""
    status = {
        'groq': 'Online' if client else 'Offline',
        'tts': 'Online' if engine else 'Offline',
        'ocr': 'Ready',
        'voice': 'Ready'
    }
    return status

def speak_text(text, use_tts=True):
    """Enhanced speak function with better performance"""
    if use_tts and engine:
        try:
            def run_tts():
                engine.say(text)
                engine.runAndWait()
            
            tts_thread = threading.Thread(target=run_tts, daemon=True)
            tts_thread.start()
        except Exception as e:
            st.warning(f"TTS error: {e}")

def chat_with_ai(prompt, context=None, system_prompt=None):
    """Enhanced AI chat with multiple model support"""
    if not client:
        return "AI service not available. Please check your API configuration."
    
    try:
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if context:
            messages.append({"role": "system", "content": f"Context: {context}"})
        
        messages.append({"role": "user", "content": prompt})
        
        with st.spinner("AI is thinking..."):
            response = client.chat.completions.create(
                messages=messages,
                model=st.session_state.get('selected_model', 'llama3-8b-8192'),
                temperature=st.session_state.get('temperature', 0.7),
                max_tokens=st.session_state.get('max_tokens', 1024)
            )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def extract_text_with_ocr(image_file):
    """Advanced OCR with multiple preprocessing techniques"""
    try:
        image = Image.open(image_file)
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Multiple preprocessing techniques
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Standard preprocessing
        denoised = cv2.medianBlur(gray, 3)
        enhanced = cv2.convertScaleAbs(denoised, alpha=1.2, beta=10)
        
        # Method 2: Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Method 3: Morphological operations
        kernel = np.ones((1,1), np.uint8)
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Try different OCR configurations
        configs = [
            r'--oem 3 --psm 6',
            r'--oem 3 --psm 3',
            r'--oem 3 --psm 8',
            r'--oem 3 --psm 11'
        ]
        
        texts = []
        for img in [enhanced, adaptive, morph]:
            for config in configs:
                try:
                    text = pytesseract.image_to_string(img, config=config)
                    if text.strip():
                        texts.append(text.strip())
                except:
                    continue
        
        # Return the longest extracted text (usually most accurate)
        if texts:
            best_text = max(texts, key=len)
            return best_text if len(best_text) > 10 else "Minimal text detected."
        
        return "No text detected in the image."
    
    except Exception as e:
        return f"OCR Error: {str(e)}"

def voice_recognition():
    """Enhanced voice recognition with noise cancellation"""
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            st.info("Calibrating microphone...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            recognizer.energy_threshold = 4000
            recognizer.dynamic_energy_threshold = True
        
        with sr.Microphone() as source:
            status_placeholder = st.empty()
            status_placeholder.success("Ready! Start speaking...")
            
            # Record audio with longer timeout
            audio = recognizer.listen(source, phrase_time_limit=15, timeout=20)
            status_placeholder.info("Processing speech...")
            
            # Try multiple recognition services
            try:
                query = recognizer.recognize_google(audio, language='en-US')
                status_placeholder.success(f"You said: **{query}**")
                return query
            except:
                # Fallback to offline recognition if available
                try:
                    query = recognizer.recognize_sphinx(audio)
                    status_placeholder.success(f"You said: **{query}**")
                    return query
                except:
                    return "Could not understand the audio clearly."
                    
    except sr.WaitTimeoutError:
        return "No speech detected. Please try again."
    except Exception as e:
        return f"Voice recognition error: {e}"

def add_to_favorites(user_msg, ai_msg):
    """Add conversation to favorites"""
    favorite = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'user': user_msg,
        'assistant': ai_msg
    }
    st.session_state.favorite_responses.append(favorite)
    st.success("Added to favorites!")

def export_chat_history():
    """Export chat history as JSON"""
    if st.session_state.chat_history:
        export_data = {
            'session_start': st.session_state.current_session_start.strftime("%Y-%m-%d %H:%M:%S"),
            'export_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'conversation_count': len(st.session_state.chat_history),
            'conversations': []
        }
        
        for user_msg, ai_msg, timestamp in st.session_state.chat_history:
            export_data['conversations'].append({
                'user': user_msg,
                'assistant': ai_msg,
                'timestamp': timestamp
            })
        
        return json.dumps(export_data, indent=2)
    return None

# Main UI Layout
st.markdown('<h1 class="main-header">AI Assistant Pro</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.header("Settings")
    
    # Model Selection
    model_options = ['llama3-8b-8192', 'llama3-70b-8192', 'mixtral-8x7b-32768']
    st.session_state.selected_model = st.selectbox("AI Model", model_options)
    
    # Parameters
    st.session_state.temperature = st.slider("Creativity", 0.1, 2.0, 0.7, 0.1)
    st.session_state.max_tokens = st.slider("Response Length", 100, 2000, 1024, 100)
    
    # Features
    st.subheader("Features")
    use_tts = st.checkbox("Text-to-Speech", value=True)
    auto_scroll = st.checkbox("Auto-scroll Chat", value=True)
    show_timestamps = st.checkbox("Show Timestamps", value=True)
    
    # System Status
    st.subheader("System Status")
    status = get_system_status()
    for component, stat in status.items():
        st.text(f"{component.upper()}: {stat}")
    
    # Statistics
    st.subheader("Session Stats")
    st.text(f"Conversations: {len(st.session_state.chat_history)}")
    st.text(f"Favorites: {len(st.session_state.favorite_responses)}")
    
    # Export/Import
    st.subheader("Data Management")
    if st.button("Export Chat"):
        export_data = export_chat_history()
        if export_data:
            st.download_button(
                label="Download Chat History",
                data=export_data,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main Chat Interface
col1, col2 = st.columns([3, 1])

with col1:
    # Chat Mode Selection
    chat_mode = st.radio(
        "Select Mode:",
        ["Text Chat", "Voice Chat", "OCR + Chat"],
        horizontal=True,
        key="chat_mode"
    )
    
    # Chat History Display
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    if st.session_state.chat_history:
        for i, (user_msg, ai_msg, timestamp) in enumerate(st.session_state.chat_history):
            # User message
            timestamp_str = f" • {timestamp}" if show_timestamps else ""
            st.markdown(f"""
                <div class="user-message">
                    <strong>You{timestamp_str}:</strong><br>
                    {user_msg}
                </div>
            """, unsafe_allow_html=True)
            
            # Assistant message
            st.markdown(f"""
                <div class="assistant-message">
                    <strong>Assistant:</strong><br>
                    {ai_msg}
                </div>
            """, unsafe_allow_html=True)
            
            # Add to favorites button
            if st.button(f"★", key=f"fav_{i}", help="Add to favorites"):
                add_to_favorites(user_msg, ai_msg)
    else:
        st.markdown("""
            <div style="text-align: center; padding: 3rem; color: #666;">
                <h3>Hello! I'm your AI Assistant</h3>
                <p>Start a conversation by typing a message, speaking, or uploading an image!</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input Section
    if chat_mode == "Text Chat":
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        # Chat input form
        with st.form("chat_form", clear_on_submit=True):
            col_input, col_send = st.columns([4, 1])
            
            with col_input:
                user_input = st.text_input(
                    "Message",
                    placeholder="Type your message here...",
                    label_visibility="collapsed"
                )
            
            with col_send:
                send_button = st.form_submit_button("Send", use_container_width=True)
            
            # Advanced options
            with st.expander("Advanced Options"):
                system_prompt = st.text_area("System Prompt (Optional)", placeholder="You are a helpful assistant...")
                context = st.text_area("Additional Context (Optional)")
        
        if send_button and user_input.strip():
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            with st.spinner("Generating response..."):
                response = chat_with_ai(user_input, context, system_prompt)
                st.session_state.chat_history.append((user_input, response, timestamp))
                
                if use_tts:
                    speak_text(response)
                
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif chat_mode == "Voice Chat":
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        col_voice, col_stop = st.columns([2, 1])
        
        with col_voice:
            if st.button("Start Voice Chat", type="primary", use_container_width=True):
                voice_query = voice_recognition()
                
                if voice_query and not voice_query.startswith(("No speech", "Could not", "Voice recognition")):
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    with st.spinner("Processing your request..."):
                        response = chat_with_ai(voice_query)
                        st.session_state.chat_history.append((voice_query, response, timestamp))
                        speak_text(response, use_tts)
                        st.rerun()
                else:
                    st.error(voice_query)
        
        with col_stop:
            st.info("Voice Tips:\n• Speak clearly\n• Quiet environment\n• Wait for the prompt")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif chat_mode == "OCR + Chat":
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload an image with text:",
            type=["png", "jpg", "jpeg", "bmp", "tiff", "pdf"],
            help="Supported: PNG, JPG, JPEG, BMP, TIFF, PDF"
        )
        
        if uploaded_file:
            col_img, col_info = st.columns([2, 1])
            
            with col_img:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            with col_info:
                st.info("OCR Features:\n• Advanced preprocessing\n• Multiple recognition methods\n• High accuracy text extraction")
            
            if st.button("Extract & Analyze Text", type="primary"):
                with st.spinner("Processing image..."):
                    extracted_text = extract_text_with_ocr(uploaded_file)
                    
                    if extracted_text and not extracted_text.startswith("OCR Error"):
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        
                        # Show extracted text
                        st.subheader("Extracted Text:")
                        st.text_area("", value=extracted_text, height=150, disabled=True)
                        
                        # Auto-generate summary
                        summary_prompt = f"Please provide a brief summary and key insights from this extracted text: {extracted_text}"
                        summary = chat_with_ai(summary_prompt)
                        
                        st.session_state.chat_history.append(
                            (f"[Image OCR] Extracted and analyzed text from uploaded image", summary, timestamp)
                        )
                        
                        if use_tts:
                            speak_text(f"Text extracted successfully. {summary[:100]}...")
                        
                        st.rerun()
                    else:
                        st.error(extracted_text)
        
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Quick Actions Panel
    st.subheader("Quick Actions")
    
    # Predefined prompts
    st.markdown("**Quick Prompts:**")
    quick_prompts = [
        "Explain quantum computing",
        "Write a creative story",
        "Help me with Python code",
        "Translate to Spanish",
        "Summarize latest AI news"
    ]
    
    for prompt in quick_prompts:
        if st.button(prompt, key=f"quick_{prompt}", use_container_width=True):
            timestamp = datetime.now().strftime("%H:%M:%S")
            response = chat_with_ai(prompt)
            st.session_state.chat_history.append((prompt, response, timestamp))
            if use_tts:
                speak_text(response)
            st.rerun()
    
    # Favorites
    if st.session_state.favorite_responses:
        st.subheader("Favorites")
        with st.expander("View Favorites"):
            for i, fav in enumerate(st.session_state.favorite_responses[-5:]):  # Show last 5
                st.markdown(f"**Q:** {fav['user'][:50]}...")
                st.markdown(f"**A:** {fav['assistant'][:100]}...")
                st.markdown("---")
    
    # System Information
    st.subheader("System Info")
    st.info(f"""
    **Session Duration:** {datetime.now() - st.session_state.current_session_start}
    
    **Current Model:** {st.session_state.get('selected_model', 'llama3-8b-8192')}
    
    **Temperature:** {st.session_state.get('temperature', 0.7)}
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><strong>AI Assistant Pro</strong> | Powered by Groq & Streamlit</p>
    <p><small>Features: Chat • Voice • OCR • TTS • Export • Favorites</small></p>
</div>
""", unsafe_allow_html=True)