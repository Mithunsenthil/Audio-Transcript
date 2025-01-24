import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
import os

# Save the uploaded file temporarily
def save_uploaded_file(uploaded_file):
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_file_path

# Convert MP3 to PCM WAV
def convert_audio_to_pcm_wav(mp3_file_path):
    audio = AudioSegment.from_file(mp3_file_path)
    wav_file_path = mp3_file_path.split(".")[0] + ".wav"
    
    # Export the file in PCM WAV format (16-bit, 44.1kHz)
    audio.export(wav_file_path, format="wav", parameters=["-acodec", "pcm_s16le", "-ar", "44100"])
    
    return wav_file_path

# Speech-to-text conversion using speech_recognition
def speech_to_text(wav_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_file_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Error: {str(e)}"

# Main Streamlit app function
def main():
    st.title("Speech to Text Converter")
    st.write("Upload multiple audio files and convert them to text.")

    uploaded_files = st.file_uploader("Choose audio files", type=["wav", "mp3"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.write(f"**Processing file:** {uploaded_file.name}")
            
            # Save the uploaded file temporarily
            saved_file_path = save_uploaded_file(uploaded_file)

            # If the file is MP3, convert it to PCM WAV
            if uploaded_file.type == "audio/mpeg":
                saved_file_path = convert_audio_to_pcm_wav(saved_file_path)

            # Perform speech-to-text on the PCM WAV file
            text = speech_to_text(saved_file_path)
            st.write(f"**Converted Text for {uploaded_file.name}:**")
            st.write(text)

            # Clean up by removing the saved files
            os.remove(saved_file_path)

if __name__ == "__main__":
    main()
