import speech_recognition
from PIL import Image
import os
import matplotlib.pyplot as plt
import threading


def speech_to_text():
    recognizer = speech_recognition.Recognizer()
    while True:
        try:
            with speech_recognition.Microphone() as mic:

                recognizer.adjust_for_ambient_noise(mic, duration=0.9)
                audio = recognizer.listen(mic)

                text = recognizer.recognize_google(audio, None, "sr-Latn-CS")
                text = text.lower()

                for i in range(len(text)):
                    if text[i] == ' ':
                        continue

                    path = f"./data/{text[i].upper()}/0.jpg"
                    if os.path.exists(path):
                        print(f"Opening {path} for {text[i]}")

                        # Open image using matplotlib for better control
                        with Image.open(path) as im:
                            fig, ax = plt.subplots()
                            ax.imshow(im)
                            ax.axis('off')  # Hide axes
                            plt.title(f"Displaying {text[i].upper()}")  # Optional title
                            plt.show(block=False)  # Non-blocking display
                            plt.pause(1)  # Display for 1 second
                            plt.close(fig)  # Close the current image window

                    else:
                        print(f"File not found: {path}")

                print(f"Recognized: {text}")

        except speech_recognition.UnknownValueError:
            recognizer = speech_recognition.Recognizer()
            continue
        except Exception as e:
            print(f"An error occurred: {e}")

def speech_to_text_threaded():
    threading.Thread(target=speech_to_text, args=()).start()