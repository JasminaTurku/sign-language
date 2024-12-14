from gtts import gTTS
import pygame
import io
import threading

# A global variable to store the last text and debounce timer
prev_character = None
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

prev_character = ''
_accumulated_characters = ''
_character_timer = None  # Timer for per-character debounce (0.2s)
_word_timer = None  # Timer for word pause detection (1s)

def _debounce_and_speak(new_character, lang='sr'):
    global prev_character, _accumulated_characters, _character_timer, _word_timer

    with _lock:
        # Skip if the new character is the same as the previous one
        if prev_character == new_character:
            return

        prev_character = new_character

        # Cancel previous character debounce timer
        if _character_timer is not None:
            _character_timer.cancel()

        # Start a new character debounce timer (0.2s)
        def process_character():
            global _accumulated_characters, _word_timer

            with _lock:
                # Append the new character to the accumulated buffer
                _accumulated_characters += new_character

                # Cancel any previous word timer
                if _word_timer is not None:
                    _word_timer.cancel()

                # Start a new word timer (1 second) to detect pause
                def process_word():
                    global _accumulated_characters
                    with _lock:
                        if _accumulated_characters:
                            # Pronounce the accumulated word
                            text_to_speech(_accumulated_characters, lang)
                            _accumulated_characters = ''  # Clear the buffer

                _word_timer = threading.Timer(1.0, process_word)
                _word_timer.start()

        _character_timer = threading.Timer(0.2, process_character)
        _character_timer.start()

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
