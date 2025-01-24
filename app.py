import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment

from pydub import AudioSegment

def convert_to_wav(input_file):
    audio = AudioSegment.from_file(input_file)
    audio.export("output.wav", format="wav")


def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Error: {str(e)}"

def main():
    st.title("Speech to Text Converter")
    st.write("Upload an audio file and convert it to text.")

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

    if uploaded_file is not None:
        file_details = {"Filename": uploaded_file.name, "FileType": uploaded_file.type}
        st.write(file_details)

        if uploaded_file.type == "audio/mp3":
            uploaded_file = convert_audio_to_wav(uploaded_file)

        text = speech_to_text(uploaded_file)
        st.write("Converted Text:")
        st.write(text)

if __name__ == "__main__":
    main()
