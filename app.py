import streamlit as st, subprocess, tempfile, json, os
from faster_whisper import WhisperModel
import openai
import sys

# ──────────────── CLOUD CONFIG ────────────────
PROMPT_TEXT = """
You are an automotive QA assistant specializing in FTIR parameter extraction.

CRITICAL INSTRUCTIONS:
1. Extract ALL available information from the transcript
2. Return ONLY valid JSON - no additional text or explanations
3. Use "Unknown" for missing string values, null for missing numbers/dates
4. Always include ALL fields in your response

Extract these FTIR parameters:

REQUIRED SCHEMA:
{
    "customer_complaint": "string or 'Unknown'",
    "defect_location": "string or 'Unknown'",
    "defect_conditions": "string or 'Unknown'", 
    "dtc_code": "string or 'Unknown'",
    "causal_part_number": "string or 'Unknown'",
    "vehicle_repaired": "Yes|No|Unknown",
    "problem_solved": "Yes|No|Unknown", 
    "repair_method_or_reason": "string or 'Unknown'",
    "defective_part_available": "Yes|No|Unknown",
    "incident_part_sent": "Yes|No|Unknown",
    "part_dispatch_date": "YYYY-MM-DD or null",
    "courier_docket_no": "string or 'Unknown'",
    "ftir_coordinator_mobile": "string or 'Unknown'",
    "driving_phase": ["EngineStart", "Accelerating", "Cruising", "Decelerating", "IdleStop"],
    "speed_related": "Yes|No|Unknown",
    "speed_from_kmh": "number or null",
    "speed_to_kmh": "number or null", 
    "rpm_related": "Yes|No|Unknown",
    "rpm_from": "number or null",
    "rpm_to": "number or null",
    "road_condition": ["CityStreet", "Highway", "Gravel", "Rough", "Smooth"]
}

IMPORTANT: Include EVERY field in your JSON response, even if unknown.
"""

# ✅ Cloud-optimized settings
WHISPER_SIZE = "small"  # Smaller model for cloud deployment
DEVICE = "cpu"         # Force CPU for cloud

# ──────────────── CLOUD-OPTIMIZED INIT ────────────────
@st.cache_resource
def load_whisper_model():
    """Load lightweight Whisper model for cloud deployment"""
    try:
        with st.spinner("🔄 Loading Whisper model (first time may take a moment)..."):
            model = WhisperModel(
                WHISPER_SIZE, 
                device=DEVICE,
                compute_type="int8",  # Memory efficient
                cpu_threads=2,       # Limited CPU resources
                download_root="./models"  # Cache models locally
            )
        st.success(f"✅ Whisper-{WHISPER_SIZE} loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

whisper_model = load_whisper_model()

# API client with error handling
try:
    llm_client = openai.OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=st.secrets["GROQ_API_KEY"]  # Use Streamlit secrets
    )
except Exception as e:
    st.error("❌ API configuration error. Please check your secrets.")
    st.stop()

# ──────────────── CLOUD-OPTIMIZED HELPERS ────────────────
def transcribe_audio(raw: bytes, max_size_mb=25) -> str:
    """Cloud-optimized transcription with file size limits"""
    
    # File size check
    file_size_mb = len(raw) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        st.error(f"❌ File too large ({file_size_mb:.1f}MB). Max size: {max_size_mb}MB")
        return ""
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(raw)
            tmp.flush()
            fixed = tmp.name.replace(".wav", "_16k.wav")
            
            # Check if ffmpeg is available
            try:
                subprocess.run(["ffmpeg", "-version"], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL, 
                             check=True)
                
                subprocess.run([
                    "ffmpeg", "-y", "-i", tmp.name, 
                    "-ar", "16000", "-ac", "1", 
                    "-t", "600",  # Limit to 10 minutes
                    fixed
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                
                audio_file = fixed
            except (subprocess.CalledProcessError, FileNotFoundError):
                st.warning("⚠️ FFmpeg not available, using original file")
                audio_file = tmp.name
        
        # Transcription with timeout protection
        segments, info = whisper_model.transcribe(
            audio_file,
            beam_size=1,      # Faster inference
            best_of=1,        # Single pass
            temperature=0,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=1000)
        )
        
        transcript = " ".join(s.text.strip() for s in segments)
        
        # Cleanup
        try:
            os.unlink(tmp.name)
            if os.path.exists(fixed):
                os.unlink(fixed)
        except:
            pass
        
        return transcript
        
    except Exception as e:
        st.error(f"❌ Transcription error: {str(e)}")
        return ""

def extract_ftir(transcript: str, attempt=1) -> dict:
    """Cloud-optimized FTIR extraction"""
    if not transcript.strip():
        return create_empty_ftir_template()
    
    try:
        response = llm_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=1500,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": PROMPT_TEXT},
                {"role": "user", "content": f"Transcript to analyze:\n\n{transcript[:4000]}"}  # Limit length
            ],
            timeout=30
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Validate all required fields
        required_fields = [
            "customer_complaint", "defect_location", "defect_conditions", 
            "dtc_code", "causal_part_number", "vehicle_repaired", 
            "problem_solved", "repair_method_or_reason", "defective_part_available",
            "incident_part_sent", "part_dispatch_date", "courier_docket_no",
            "ftir_coordinator_mobile", "driving_phase", "speed_related",
            "speed_from_kmh", "speed_to_kmh", "rpm_related", "rpm_from",
            "rpm_to", "road_condition"
        ]
        
        for field in required_fields:
            if field not in result:
                if field in ["speed_from_kmh", "speed_to_kmh", "rpm_from", "rpm_to", "part_dispatch_date"]:
                    result[field] = None
                elif field in ["driving_phase", "road_condition"]:
                    result[field] = []
                else:
                    result[field] = "Unknown"
        
        return result
        
    except Exception as e:
        if attempt <= 2:
            return extract_ftir(transcript, attempt + 1)
        else:
            st.error(f"❌ Extraction failed: {str(e)}")
            return create_empty_ftir_template()

def create_empty_ftir_template() -> dict:
    """Return empty FTIR template"""
    return {
        "customer_complaint": "Unknown", "defect_location": "Unknown", 
        "defect_conditions": "Unknown", "dtc_code": "Unknown",
        "causal_part_number": "Unknown", "vehicle_repaired": "Unknown",
        "problem_solved": "Unknown", "repair_method_or_reason": "Unknown", 
        "defective_part_available": "Unknown", "incident_part_sent": "Unknown",
        "part_dispatch_date": None, "courier_docket_no": "Unknown",
        "ftir_coordinator_mobile": "Unknown", "driving_phase": [],
        "speed_related": "Unknown", "speed_from_kmh": None,
        "speed_to_kmh": None, "rpm_related": "Unknown", 
        "rpm_from": None, "rpm_to": None, "road_condition": []
    }

# ──────────────── STREAMLIT UI ────────────────
st.set_page_config(page_title="FTIR Extractor", page_icon="🚗", layout="wide")
st.title("🚗 FTIR Audio Extractor (Cloud Deployment)")

# Cloud deployment info
st.info("☁️ **Cloud Version**: Optimized for Streamlit Cloud with CPU processing")

# File size warning
st.warning("📁 **File Limits**: Max 25MB, recommended under 10MB for faster processing")

audio = st.file_uploader(
    "Upload customer support call", 
    type=["wav", "mp3", "m4a"],
    help="Supported formats: WAV, MP3, M4A. Max size: 25MB"
)

if audio:
    file_size_mb = audio.size / (1024 * 1024)
    st.info(f"📁 **File**: {audio.name} ({file_size_mb:.1f}MB)")
    
    if file_size_mb > 25:
        st.error("❌ File too large! Please use a file smaller than 25MB.")
        st.stop()
    
    # Processing with progress
    progress_container = st.container()
    
    with progress_container:
        progress = st.progress(0)
        status = st.empty()
        
        status.text("🎙️ Starting transcription...")
        progress.progress(10)
        
        transcript = transcribe_audio(audio.read())
        progress.progress(70)
        
        if transcript:
            status.text("✅ Transcription complete! Processing with LLM...")
            progress.progress(80)
            
            ftir_data = extract_ftir(transcript)
            progress.progress(100)
            status.text("🎯 Processing complete!")
            
            # Results
            st.subheader("📝 Transcript")
            with st.expander("View Transcript", expanded=False):
                st.text_area("", transcript, height=200)
            
            st.subheader("📊 FTIR Parameters")
            
            # Organized display
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Basic Information**")
                st.json({
                    "customer_complaint": ftir_data.get("customer_complaint"),
                    "defect_location": ftir_data.get("defect_location"), 
                    "defect_conditions": ftir_data.get("defect_conditions"),
                    "dtc_code": ftir_data.get("dtc_code")
                })
                
            with col2:
                st.markdown("**Repair Status**") 
                st.json({
                    "vehicle_repaired": ftir_data.get("vehicle_repaired"),
                    "problem_solved": ftir_data.get("problem_solved"),
                    "repair_method_or_reason": ftir_data.get("repair_method_or_reason")
                })
            
            st.markdown("**Complete FTIR JSON**")
            st.json(ftir_data)
            
            # Download
            json_string = json.dumps(ftir_data, indent=2)
            st.download_button(
                "📥 Download JSON",
                json_string,
                f"ftir_{audio.name.split('.')[0]}.json",
                "application/json"
            )
        else:
            progress.progress(100)
            status.text("❌ Transcription failed")

# Footer
st.markdown("---")
st.caption(f"☁️ **Cloud Engine**: Whisper-{WHISPER_SIZE} on CPU • LLM: Groq API")
