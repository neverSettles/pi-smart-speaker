import pyaudio
import wave
import os
import openai
from gtts import gTTS
import numpy as np
from dotenv import load_dotenv
import tempfile
from pydub import AudioSegment
from pydub.playback import play

# Load environment variables and OpenAI API key
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Audio settings
AUDIO_CONFIG = {
    'chunk': 1024,
    'format': pyaudio.paInt16,
    'channels': 1,
    'rate': 16000,
    'threshold': 1000,
    'silence_duration': 5
}

# Initialize PyAudio
audio = pyaudio.PyAudio()

def record_audio():
    """Records audio until silence is detected."""
    stream = audio.open(format=AUDIO_CONFIG['format'], channels=AUDIO_CONFIG['channels'],
                        rate=AUDIO_CONFIG['rate'], input=True, frames_per_buffer=AUDIO_CONFIG['chunk'])
    print("Listening for the wake word...")

    frames, silent_count, recording = [], 0, False

    while True:
        data = stream.read(AUDIO_CONFIG['chunk'], exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        amplitude = np.max(np.abs(audio_data))

        if not recording and amplitude > AUDIO_CONFIG['threshold']:
            if detect_wake_word(audio_data):
                print("Wake word detected. Start speaking...")
                recording, silent_count = True, 0

        if recording:
            frames.append(data)
            silent_count = silent_count + 1 if amplitude < AUDIO_CONFIG['threshold'] else 0

            if silent_count > AUDIO_CONFIG['rate'] / AUDIO_CONFIG['chunk'] * AUDIO_CONFIG['silence_duration']:
                print("Silence detected. Ending recording.")
                break

    stream.stop_stream()
    stream.close()
    return frames

def detect_wake_word(audio_data):
    """Detects 'Hey Bubby' (placeholder logic)."""
    return True

def save_audio_to_wav(frames, filename="output.wav"):
    """Save recorded audio to a WAV file."""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(AUDIO_CONFIG['channels'])
        wf.setsampwidth(audio.get_sample_size(AUDIO_CONFIG['format']))
        wf.setframerate(AUDIO_CONFIG['rate'])
        wf.writeframes(b''.join(frames))
    print(f"Audio saved to {filename}")

def transcribe_audio(filename):
    """Transcribe audio using OpenAI Whisper API."""
    print(f"Transcribing audio from {filename}...")
    try:
        with open(filename, "rb") as audio_file:
            transcription = openai.Audio.transcribe(model="whisper-1", file=audio_file)
            return transcription['text']
    except Exception as e:
        print(f"Error during transcription: {e}")
        raise

def generate_gpt4_response(prompt):
    """Send transcription to GPT-4 and get a response."""
    print(f"Generating response for: {prompt}")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

def play_audio_from_text(text):
    """Convert text to speech using gTTS and play it using pydub."""
    print("Converting text to speech...")
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as temp_audio_file:
        gTTS(text=text, lang='en').save(temp_audio_file.name)
        audio_segment = AudioSegment.from_file(temp_audio_file.name, format="mp3")
        play(audio_segment)
    print("Audio playback completed.")

def main():
    while True:
        audio_frames = record_audio()
        save_audio_to_wav(audio_frames)

        transcription = transcribe_audio("output.wav")
        if transcription:
            print(f"You said: {transcription}")

            response = generate_gpt4_response(transcription)
            print(f"GPT-4 Response: {response}")

            play_audio_from_text(response)

if __name__ == "__main__":
    main()
