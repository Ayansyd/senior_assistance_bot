# stt.py
import sys
import queue
import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer

class LocalSTT:
    def __init__(self, model_path: str, sample_rate: int = 16000):
        """
        :param model_path: Path to your local Vosk model folder
        :param sample_rate: Audio sampling rate (default 16000)
        """
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, sample_rate)
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.listening = False

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(f"STT status: {status}", file=sys.stderr)
        self.audio_queue.put(bytes(indata))

    def listen_once(self, duration: float = 5.0) -> str:
        """
        Record audio for a fixed duration (in seconds), transcribe it, and return text.
        """
        self.listening = True
        with sd.RawInputStream(samplerate=self.sample_rate,
                               blocksize=8000,
                               dtype='int16',
                               channels=1,
                               callback=self._audio_callback):
            print("Listening for speech...")
            sd.sleep(int(duration * 1000))

        transcript = ""
        while not self.audio_queue.empty():
            data = self.audio_queue.get()
            if self.recognizer.AcceptWaveform(data):
                result_json = self.recognizer.Result()
                text = self._extract_text(result_json)
                transcript += text
            else:
                partial_json = self.recognizer.PartialResult()
                # partial_text = self._extract_partial_text(partial_json)
                # You could handle partial transcripts here if desired.

        self.listening = False
        return transcript.strip()

    def _extract_text(self, result_json: str) -> str:
        """
        Extract the 'text' field from the Vosk JSON result.
        """
        parsed = json.loads(result_json)
        return parsed.get("text", "")
