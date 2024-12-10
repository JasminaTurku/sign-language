from gtts import gTTS
import pygame
import io
import threading
import time

# A global variable to store the last text and debounce timer
_last_text = None
_debounce_timer = None
_lock = threading.Lock()

def text_to_speech(text, lang='sr'):
    """
    Convert the provided text to speech in the specified language.

    Parameters:
        text (str): The text to convert to speech.
        lang (str): The language for the speech (default is Serbian - 'sr').
    """
    try:
        # Generate speech using gTTS and save to a BytesIO object
        tts = gTTS(text, lang=lang)
        audio_stream = io.BytesIO()
        tts.write_to_fp(audio_stream)
        audio_stream.seek(0)

        # Initialize pygame mixer
        pygame.mixer.init()

        # Load and play the audio from the stream
        pygame.mixer.music.load(audio_stream, "mp3")
        pygame.mixer.music.play()

        # Wait for the playback to finish
        while pygame.mixer.music.get_busy():
            continue

    except Exception as e:
        print(f"An error occurred: {e}")

def _debounce_and_speak(text, lang='sr'):
    global _last_text, _debounce_timer
    with _lock:
        _last_text = text

        # Cancel the previous timer if it's running
        if _debounce_timer is not None:
            _debounce_timer.cancel()

        # Start a new timer
        def speak_after_delay():
            with _lock:
                if _last_text == text:
                    text_to_speech(text, lang)

        _debounce_timer = threading.Timer(0.2, speak_after_delay) # probaj i sa 100ms debounce, pa vidi sta bolje radi
        _debounce_timer.start()

def text_to_speech_threaded(text, lang='sr'):
    """
    Run the text-to-speech function in a separate thread with debouncing.

    Parameters:
        text (str): The text to convert to speech.
        lang (str): The language for the speech (default is Serbian - 'sr').
    """
    threading.Thread(target=_debounce_and_speak, args=(text, lang)).start()

# Example usage in main.py
# from text_to_speech_module import text_to_speech_threaded
