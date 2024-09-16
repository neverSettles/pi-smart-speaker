import pyaudio
import wave
import os
import openai  # Correct import for OpenAI
from gtts import gTTS  # Import gTTS for text-to-speech
import time
import numpy as np
from dotenv import load_dotenv
import tempfile  # To save audio temporarily
from pydub import AudioSegment  # For handling MP3 playback
from pydub.playback import play  # For playing MP3

# Load API key from .env file
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Audio settings
CHUNK = 1024  # Buffer size
FORMAT = pyaudio.paInt16  # Format for recording
CHANNELS = 1  # Mono audio
RATE = 16000  # Sampling rate
THRESHOLD = 1000  # Amplitude threshold for speech detection
SILENCE_DURATION = 5  # Seconds of silence to determine end of speech

# Initialize pyaudio
audio = pyaudio.PyAudio()

def record_audio():
    """Records audio until there is a silence longer than SILENCE_DURATION."""
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("Listening for the wake word...")

    frames = []
    silent_count = 0
    recording = False

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        amplitude = np.max(np.abs(audio_data))

        if not recording and amplitude > THRESHOLD:
            # Detect the wake word ("Hey Bubby")
            if detect_wake_word(audio_data):
                print("Wake word detected. Start speaking...")
                recording = True
                silent_count = 0  # Reset silence count

        if recording:
            frames.append(data)
            if amplitude < THRESHOLD:
                silent_count += 1
            else:
                silent_count = 0

            if silent_count > RATE / CHUNK * SILENCE_DURATION:
                print("Silence detected. Ending recording.")
                break

    stream.stop_stream()
    stream.close()

    return frames

def detect_wake_word(audio_data):
    """Implement basic logic to detect 'Hey Bubby'"""
    return True

def save_audio_to_wav(frames, filename="output.wav"):
    """Save recorded audio to a WAV file."""
    print(f"Saving audio to WAV file: {filename}")
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Audio saved to {filename}")

def transcribe_audio(filename):
    """Use OpenAI Whisper API to transcribe audio."""
    print(f"Attempting to transcribe audio from file: {filename}")
    try:
        with open(filename, "rb") as audio_file:
            print(f"File {filename} opened successfully.")
            print("Calling OpenAI Whisper model for transcription...")
            transcription = openai.Audio.transcribe(
                model="whisper-1",  # Specify the Whisper model
                file=audio_file
            )
            print(f"Transcription successful. Text: {transcription['text']}")
        return transcription['text']
    except AttributeError as e:
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during transcription: {e}")
        raise

def generate_gpt4_response(prompt):
    """Send prompt to GPT-4 and get a response."""
    print(f"Sending prompt to GPT-4: {prompt}")
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Use GPT-4
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    print("GPT-4 response received.")
    return response['choices'][0]['message']['content']

def play_audio_from_text(text):
    """Convert text to speech using gTTS and play it using pydub."""
    print(f"Converting text to speech: {text}")
    tts = gTTS(text=text, lang='en')

    # Use a temporary file to store the generated speech
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as temp_audio_file:
        tts.save(temp_audio_file.name)
        print("Text converted to speech and saved temporarily.")

        # Play the audio using pydub
        audio_segment = AudioSegment.from_file(temp_audio_file.name, format="mp3")
        print("Playing audio...")
        play(audio_segment)

    print("Audio playback completed.")

def main():
    while True:
        # Record audio when wake word is detected
        print("Recording audio...")
        audio_frames = record_audio()
        save_audio_to_wav(audio_frames)

        # Transcribe the audio
        print("Transcribing the audio...")
        transcription = transcribe_audio("output.wav")
        if transcription:
            print(f"You said: {transcription}")

            # Send transcription to GPT-4 API
            response = generate_gpt4_response(transcription)
            print(f"GPT-4 Response: {response}")

            # Convert GPT-4 response to speech and play it
            play_audio_from_text(response)

if __name__ == "__main__":
    main()
