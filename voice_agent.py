from __future__ import annotations
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

@dataclass(frozen=True)
class AppConfig:
    whisper_model: str
    whisper_language: str
    whisper_device: str
    whisper_compute_type: str
    whisper_cpu_threads: int
    groq_api_key: str
    groq_model_id: str
    system_prompt: str
    gtts_lang: str

def load_config() -> AppConfig:
    def _get_int(name: str, default: int) -> int:
        raw = os.getenv(name, "").strip()
        return int(raw) if raw.isdigit() else default

    return AppConfig(
        whisper_model=os.getenv("WHISPER_MODEL", "base.en").strip(),
        whisper_language=os.getenv("WHISPER_LANGUAGE", "en").strip(),
        whisper_device=os.getenv("WHISPER_DEVICE", "cpu").strip(),
        whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip(),
        whisper_cpu_threads=_get_int("WHISPER_CPU_THREADS", 4),
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        groq_model_id=os.getenv("GROQ_MODEL_ID", "llama-3.1-8b-instant").strip(),
        system_prompt=os.getenv("SYSTEM_PROMPT", "You are a helpful voice assistant. Keep replies short.").strip(),
        gtts_lang=os.getenv("GTTS_LANG", "en").strip(),
    )

CFG = load_config()

@st.cache_resource(show_spinner=False)
def get_whisper_model():
    from faster_whisper import WhisperModel
    return WhisperModel(CFG.whisper_model, device=CFG.whisper_device, compute_type=CFG.whisper_compute_type)

def transcribe_wav_bytes(wav_bytes: bytes) -> str:
    model = get_whisper_model()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp_path = f.name
    try:
        segments, _ = model.transcribe(tmp_path, language=CFG.whisper_language, vad_filter=True)
        return "".join(seg.text for seg in segments).strip()
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

def groq_chat_completion(messages: List[Dict[str, str]]) -> str:
    import requests
    headers = {"Authorization": f"Bearer {CFG.groq_api_key}", "Content-Type": "application/json"}
    payload = {"model": CFG.groq_model_id, "messages": messages, "temperature": 0.5}
    r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def text_to_speech_bytes(text: str) -> bytes:
    from gtts import gTTS
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tts = gTTS(text=text, lang=CFG.gtts_lang)
        tts.save(f.name)
        with open(f.name, "rb") as rf: audio_data = rf.read()
    if os.path.exists(f.name): os.remove(f.name)
    return audio_data

st.set_page_config(page_title="VoiceBridge AI", layout="centered", page_icon="🎙️")
st.title("🎙️ VoiceBridge Speech-to-Speech")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] # Format: {"role": "...", "content": "...", "audio": bytes}
if "last_processed_audio" not in st.session_state:
    st.session_state.last_processed_audio = None


with st.sidebar:
    st.header("⚙️ Settings")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_processed_audio = None
        st.rerun()
    st.divider()
    st.caption(f"LLM: {CFG.groq_model_id}")

for i, message in enumerate(st.session_state.chat_history):
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        
       
        if message["role"] == "assistant" and "audio" in message:
           
            is_latest = (i == len(st.session_state.chat_history) - 1)
            st.audio(message["audio"], format="audio/mp3", autoplay=is_latest)


audio_input = st.audio_input("Speak now...", key="voice_mic_input")

audio_input = st.audio_input("Speak now...", key="voice_mic_input")

if audio_input is not None:
    audio_bytes = audio_input.getvalue()

    if audio_bytes != st.session_state.last_processed_audio:
        st.session_state.last_processed_audio = audio_bytes

        try:
            with st.status("VoiceBridge is processing...", expanded=True) as status:
                st.write("🔍 Transcribing...")
                transcript = transcribe_wav_bytes(audio_bytes)

                if not transcript:
                    status.update(label="No voice detected!", state="error")
                    st.session_state.last_processed_audio = None
                    st.stop()

                st.write("🧠 Thinking...")

                api_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chat_history[-5:]
                ]

                messages = [{"role": "system", "content": CFG.system_prompt}]
                messages.extend(api_history)
                messages.append({"role": "user", "content": transcript})

                reply_text = groq_chat_completion(messages)

                st.write("🔊 Generating Voice...")
                reply_audio = text_to_speech_bytes(reply_text)

                st.session_state.chat_history.append({
                    "role": "user",
                    "content": transcript
                })
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": reply_text,
                    "audio": reply_audio
                })

                status.update(label="Done!", state="complete", expanded=False)

            st.rerun()

        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.last_processed_audio = None
