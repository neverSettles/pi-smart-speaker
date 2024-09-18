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
import time

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

WAKE_WORD_TIMEOUT = 120  # 2 minutes timeout

def record_audio_for_wake_word():
    """Records audio to detect the wake word."""
    stream = audio.open(format=AUDIO_CONFIG['format'], channels=AUDIO_CONFIG['channels'],
                        rate=AUDIO_CONFIG['rate'], input=True, frames_per_buffer=AUDIO_CONFIG['chunk'])
    print("Listening for the wake word...")

    frames = []
    while len(frames) < 5 * AUDIO_CONFIG['rate'] / AUDIO_CONFIG['chunk']:  # Limit to 5 seconds
        data = stream.read(AUDIO_CONFIG['chunk'], exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    return frames

def detect_wake_word(audio_data):
    """Detects 'Hey Bubby' using Whisper API transcription."""
    temp_filename = "temp_wake_word.wav"
    save_audio_to_wav(audio_data, temp_filename)
    
    transcription = transcribe_audio(temp_filename)
    print(f"Transcription for wake word detection: {transcription}")
    
    if "hey bubby" in transcription.lower():
        return True
    return False

def record_audio():
    """Records audio until silence is detected."""
    stream = audio.open(format=AUDIO_CONFIG['format'], channels=AUDIO_CONFIG['channels'],
                        rate=AUDIO_CONFIG['rate'], input=True, frames_per_buffer=AUDIO_CONFIG['chunk'])

    print("Listening for your speech...")

    frames, silent_count, recording = [], 0, False

    while True:
        data = stream.read(AUDIO_CONFIG['chunk'], exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        amplitude = np.max(np.abs(audio_data))

        if not recording and amplitude > AUDIO_CONFIG['threshold']:
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

def simple_responses(question):
    """Check for predefined responses to common questions."""
    predefined_responses = {
        "who is the first president": "George Washington, do you want to hear more?"
    }
    question_lower = question.strip().lower()
    return predefined_responses.get(question_lower)

def generate_gpt4_response(prompt):
    """Send transcription to GPT-4 and get a response."""
    print(f"Generating response for: {prompt}")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who is brief."},
            {"role": "user", "content": prompt + ' be brief in your reply, around 2-3 setences should be good.'}
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
    wake_word_detected = False
    last_interaction_time = time.time()

    while True:
        current_time = time.time()
        
        # If the wake word was detected more than WAKE_WORD_TIMEOUT seconds ago, listen for the wake word again
        if not wake_word_detected or (current_time - last_interaction_time > WAKE_WORD_TIMEOUT):
            print("Waiting for the wake word...")
            wake_word_audio = record_audio_for_wake_word()
            
            if detect_wake_word(wake_word_audio):
                print("Wake word detected! Entering interactive mode.")
                wake_word_detected = True
                last_interaction_time = current_time

        # Record the user's speech
        print("Listening for user input...")
        audio_frames = record_audio()
        save_audio_to_wav(audio_frames)

        transcription = transcribe_audio("output.wav")
        if transcription:
            print(f"You said: {transcription}")
            last_interaction_time = current_time  # Reset the timer for interactions

            # Check for predefined responses
            simple_response = simple_responses(transcription)
            if simple_response:
                print(f"Predefined Response: {simple_response}")
                play_audio_from_text(simple_response)
            else:
                # Send transcription to GPT-4 API for general queries
                response = generate_gpt4_response(transcription)
                print(f"GPT-4 Response: {response}")
                play_audio_from_text(response)

if __name__ == "__main__":
    main()
