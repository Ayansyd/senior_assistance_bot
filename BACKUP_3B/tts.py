# tts.py
import os
import uuid
from gtts import gTTS
from playsound import playsound

class GTTS_TTS:
    def __init__(self, language="en"):
        self.language = language

    def speak(self, text: str):
        """
        Convert text to speech with gTTS, then play the resulting MP3.
        """
        if not text:
            return
        
        # Generate a unique MP3 filename
        filename = f"tts_output_{uuid.uuid4()}.mp3"
        tts = gTTS(text=text, lang=self.language)
        tts.save(filename)

        try:
            playsound(filename)
        finally:
            # Clean up the MP3 file after playing
            if os.path.exists(filename):
                os.remove(filename)
