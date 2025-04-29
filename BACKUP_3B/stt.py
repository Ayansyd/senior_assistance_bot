# stt.py
import sys
import numpy as np
import sounddevice as sd
import whisper
import time

class WhisperSTT:
    def __init__(self, model_name: str = "base.en", sample_rate: int = 16000, device: str = None):
        """
        Uses OpenAI's Whisper model for Speech-to-Text.

        :param model_name: Name of the Whisper model to use
                           (e.g., "tiny.en", "base.en", "small.en", "medium.en",
                            "tiny", "base", "small", "medium", "large").
        :param sample_rate: Audio sampling rate (Whisper expects 16000 Hz).
        :param device: Device to run the model on ("cuda", "cpu", or None for auto).
        """
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.device = device

        # Ensure sample rate is supported (optional check)
        try:
            sd.check_input_settings(samplerate=self.sample_rate, channels=1, dtype='int16')
        except Exception as e:
            print(f"Error: Input device doesn't support {self.sample_rate} Hz sample rate or int16 dtype: {e}", file=sys.stderr)
            # You might want to list available settings or handle this more gracefully
            # sd.query_devices() might be helpful here
            sys.exit(1)


        try:
            print(f"Loading Whisper model '{self.model_name}'...")
            # Explicitly set weights_only=True for security, aligning with future PyTorch defaults
            self.model = whisper.load_model(self.model_name, device=self.device)
            print("Whisper model loaded successfully.")
        except FileNotFoundError:
             print(f"Error: Whisper model '{self.model_name}' not found.", file=sys.stderr)
             print("It might need to be downloaded first, or the name is incorrect.", file=sys.stderr)
             sys.exit(1)
        except Exception as e:
            print(f"Error loading Whisper model '{self.model_name}': {e}", file=sys.stderr)
            print("Please ensure the model name is correct and whisper/pytorch are installed correctly.", file=sys.stderr)
            sys.exit(1) # Exit if model loading fails

        if ".en" in model_name:
            print("Using English-only Whisper model.")
        else:
            print("Using multilingual Whisper model.")

    def listen_once(self, duration: float = 5.0) -> str:
        """
        Record audio for a fixed duration using sd.rec(), transcribe using Whisper,
        and return text.
        """
        print(f"Listening for {duration:.1f} seconds...")

        # Calculate number of frames
        num_frames = int(duration * self.sample_rate)

        try:
            # Record audio directly into a NumPy array (blocking call)
            # This is simpler and less prone to threading/queue issues
            audio_data_int16 = sd.rec(num_frames, samplerate=self.sample_rate, channels=1, dtype='int16')
            sd.wait()  # Wait until recording is finished

        except sd.PortAudioError as e:
             print(f"\nError during audio recording: {e}", file=sys.stderr)
             print("Please check your audio device configuration.", file=sys.stderr)
             return "[Error during recording]"
        except Exception as e:
            print(f"\nAn unexpected error occurred during recording: {e}", file=sys.stderr)
            return "[Error during recording]"


        if audio_data_int16 is None or audio_data_int16.size == 0:
            print("No audio data captured.")
            return ""

        # --- Data Processing and Transcription ---
        try:
            # Ensure it's a numpy array (sd.rec should return one, but double check)
            if not isinstance(audio_data_int16, np.ndarray):
                 print("Warning: sd.rec did not return a NumPy array.", file=sys.stderr)
                 # Attempt conversion if possible, otherwise fail
                 audio_data_int16 = np.array(audio_data_int16, dtype=np.int16)


            # Check shape - should be (num_frames, 1)
            if audio_data_int16.shape[1] != 1:
                 print(f"Warning: Unexpected audio shape {audio_data_int16.shape}, expected ({num_frames}, 1). Attempting to reshape.", file=sys.stderr)
                 # If it's flat, try reshaping; otherwise, it might be problematic
                 if len(audio_data_int16.shape) == 1:
                      audio_data_int16 = audio_data_int16.reshape(-1, 1)
                 else:
                      print("Error: Cannot handle unexpected multi-channel audio shape.", file=sys.stderr)
                      return "[Error: Unexpected audio format]"


            # Convert int16 data to float32 and normalize to [-1.0, 1.0] (Whisper expects this)
            # Squeeze to remove the channel dimension -> (num_frames,)
            audio_float32 = audio_data_int16.astype(np.float32).flatten() / 32768.0


            print("Transcribing audio...")
            start_time = time.time()

            # Set language if using an English-only model, otherwise let Whisper detect
            language_option = "en" if ".en" in self.model_name else None

            # Set fp16=True only if using CUDA and GPU supports it
            use_fp16 = (self.device == "cuda" and torch.cuda.is_available()) # Requires importing torch

            result = self.model.transcribe(
                audio_float32,
                language=language_option,
                # fp16=use_fp16 # Set based on device (optional)
                fp16=False # Keep False for now to ensure CPU compatibility unless specifically enabled
                )

            transcript = result.get('text', '').strip()
            end_time = time.time()
            print(f"Transcription complete in {end_time - start_time:.2f}s")
            # Add a check for empty transcription which can happen with silence
            if not transcript:
                 print("[Transcription resulted in empty text (likely silence)]")
                 return "" # Return empty string for silence

            return transcript

        # Catch memory errors specifically if possible (though Python OOM often just crashes)
        except MemoryError as e:
             print(f"\nMemoryError during transcription: {e}", file=sys.stderr)
             print("The system may not have enough RAM for this model or audio duration.", file=sys.stderr)
             return "[Error: Out of memory during transcription]"
        except Exception as e:
            print(f"\nError during Whisper processing/transcription: {e}", file=sys.stderr)
            # Optionally print traceback for debugging unexpected errors
            # import traceback
            # traceback.print_exc()
            return "[Error during transcription]"

# You might need to import torch if you uncomment the fp16 logic based on CUDA availability
# import torch