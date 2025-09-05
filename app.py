# app.py  – Streamlit FTIR extractor (whisper + local LLM)
import streamlit as st, subprocess, tempfile, json, os
from faster_whisper import WhisperModel
from openai import OpenAI   # ✔ works: LocalAI/Ollama expose OpenAI-style API

# ──────────────── CONFIG ────────────────
PROMPT_TEXT = """
You are an automotive QA assistant. Extract these FTIR parameters and
return VALID JSON; omit keys that are unknown.

{
 "customer_complaint": string,
 "defect_location": string,
 "defect_conditions": string,
 "dtc_code": string,
 "causal_part_number": string,
 "vehicle_repaired": "Yes"|"No",
 "problem_solved": "Yes"|"No",
 "repair_method_or_reason": string,
 "defective_part_available": "Yes"|"No",
 "incident_part_sent": "Yes"|"No",
 "part_dispatch_date": "YYYY-MM-DD",
 "courier_docket_no": string,
 "ftir_coordinator_mobile": string,
 "driving_phase": [ "EngineStart"|"Accelerating"|... ],
 "speed_related": "Yes"|"No",
 "speed_from_kmh": int,
 "speed_to_kmh": int,
 "rpm_related": "Yes"|"No",
 "rpm_from": int,
 "rpm_to": int,
 "road_condition": [ "CityStreet"|"Gravel"|... ]
}
"""

LLM_BASE_URL = "http://localhost:8080/v1"           # LocalAI/Ollama endpoint
LLM_MODEL    = "mistral-7b-instruct-v0.2-q4"
WHISPER_SIZE = "small"                             # small / medium / large
DEVICE       = "cpu"                               # or "cuda" if GPU present

# ──────────────── INIT ────────────────
whisper_model = WhisperModel(WHISPER_SIZE, device=DEVICE)
llm_client    = OpenAI(base_url=LLM_BASE_URL, api_key="local-key")  # dummy key

# ──────────────── HELPERS ────────────────
def transcribe_audio(raw: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(raw); tmp.flush()
        fixed = tmp.name.replace(".wav", "_16k.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp.name, "-ar", "16000", "-ac", "1", fixed],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    segments, _ = whisper_model.transcribe(fixed)
    return " ".join(s.text.strip() for s in segments)

def extract_ftir(transcript: str) -> dict:
    rsp = llm_client.chat.completions.create(
        model=LLM_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": PROMPT_TEXT},
            {"role": "user",   "content": transcript}
        ],
    )
    return json.loads(rsp.choices[0].message.content)

# ──────────────── STREAMLIT UI ────────────────
st.set_page_config(page_title="FTIR Extractor", page_icon="🚗")
st.title("Audio ➜ FTIR-Form Extractor (100 % local)")

audio = st.file_uploader("Upload customer-support call",
                         type=["wav", "mp3", "m4a"])

if audio:
    with st.spinner("Transcribing with faster-whisper…"):
        transcript = transcribe_audio(audio.read())
    st.subheader("Transcript")
    st.text_area("", transcript, height=200)

    with st.spinner("Extracting FTIR parameters via local LLM…"):
        ftir = extract_ftir(transcript)

    st.subheader("FTIR JSON")
    st.json(ftir)
    st.download_button(
        "Download JSON",
        json.dumps(ftir, indent=2),
        "ftir_output.json",
        "application/json"
    )

# ──────────────── FOOTER ────────────────
st.caption(
    "Engine: faster-whisper-%(size)s on %(dev)s • LLM: %(model)s via LocalAI"
    % {"size": WHISPER_SIZE, "dev": DEVICE, "model": LLM_MODEL}
)
