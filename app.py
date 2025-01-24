# Load model directly
import streamlit as st
import torch
import numpy as np
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
from io import BytesIO

# Configure Streamlit page settings
st.set_page_config(
    page_title="Transcribe with Whisper",
    page_icon=":rocket:",
    layout="centered"
)

st.title("🎙️ Whisper Audio Transcriber")
st.divider()

# Set up device and data type
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the Whisper model and processor
with st.spinner("🚀 Loading Whisper model... please wait!"):
    model_name = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_name)

# Initialize ASR pipeline
asr_pipe = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

st.markdown("Upload your audio files, and let the Whisper model transcribe them instantly. 🚀")

# File uploader for audio files
uploaded_files = st.file_uploader("📂 Select audio files to transcribe", type=["wav","mp3"], accept_multiple_files=True)

# Transcription button and result display
if uploaded_files:
    if st.button("✍️ Transcribe"):
        results = []

        for idx, audio_file in enumerate(uploaded_files):
            try:
                # Read audio file
                audio_data, sr = librosa.load(BytesIO(audio_file.read()), sr=16000)

                # Run ASR pipeline
                result = asr_pipe(audio_data)
                transcription = result['text']
                results.append((audio_file.name, transcription))
            except Exception as e:
                st.error(f"Error processing '{audio_file.name}':{e}")

        # Display results
        st.subheader("Transcriptions")
        for filename, transcription in results:
            st.text_area(f"📂 **{filename}**:", value=transcription)
else:
    st.info("📤 Please upload audio files to start transcription.")
