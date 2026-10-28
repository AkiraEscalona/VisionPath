import time
import pyttsx3

print("Initializing speech engine...")

engine = pyttsx3.init("sapi5")
engine.setProperty("rate", 170)
engine.setProperty("volume", 1.0)

messages = [
    "Audio test one.",
    "Audio test two.",
    "Audio test three.",
]

for message in messages:
    print("Speaking:", message)
    engine.say(message)
    engine.runAndWait()
    time.sleep(1)

engine.stop()

print("Test finished.")