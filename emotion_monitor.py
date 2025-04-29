# emotion_monitor.py
import threading
import time
import requests
import logging
import queue
from requests.exceptions import RequestException
import json

class EmotionMonitor:
    def __init__(self,
                 endpoint_url: str,
                 poll_interval_seconds: int,
                 proactive_queue: queue.Queue,
                 # state_lock: threading.Lock, # REMOVED
                 # get_assistant_busy_func, # REMOVED
                 confidence_threshold: float = 0.6,
                 target_emotions: list = None):
        """
        Monitors an emotion detection endpoint in a background thread.
        Puts detected target emotions onto the proactive_queue.

        Args:
            endpoint_url: The URL of the facial emotion recognition endpoint.
            poll_interval_seconds: How often to poll the endpoint (in seconds).
            proactive_queue: A queue.Queue to put detected target emotions onto.
            confidence_threshold: Minimum confidence score to consider an emotion valid.
            target_emotions: List of emotions (e.g., ['sad', 'happy']) to trigger proactive interaction.
        """
        self.endpoint_url = endpoint_url
        self.poll_interval = poll_interval_seconds
        self.proactive_queue = proactive_queue
        # self.state_lock = state_lock # REMOVED
        # self.get_assistant_busy = get_assistant_busy_func # REMOVED
        self.confidence_threshold = confidence_threshold
        self.target_emotions = target_emotions if target_emotions else ["sad", "happy"]

        self._thread = None
        self._running = False
        self.last_triggered_emotion = None # Avoid spamming for the same emotion
        self.last_trigger_time = 0
        self.min_time_between_triggers = 120 # e.g., 2 minutes

    def _poll_emotion(self):
        """The main loop for the background thread."""
        logging.info("[EmotionMonitor] Monitor loop starting.")
        while self._running:
            next_check_time = time.time() + self.poll_interval
            try:
                # --- Check Emotion Endpoint ---
                response = requests.get(self.endpoint_url, timeout=5)
                response.raise_for_status()

                try:
                    data = response.json()
                    dominant_emotion = data.get("dominant_emotion", "").lower()
                    confidence = data.get("confidence", 0.0)
                    logging.debug(f"[EmotionMonitor] Detected: {dominant_emotion} (Conf: {confidence:.2f})")

                    # --- Check if Emotion is Target and Confident ---
                    if dominant_emotion in self.target_emotions and confidence >= self.confidence_threshold:
                        current_time = time.time()
                        is_new_trigger = (dominant_emotion != self.last_triggered_emotion or
                                          (current_time - self.last_trigger_time) > self.min_time_between_triggers)

                        if is_new_trigger:
                            # --- REMOVED Busy Check ---
                            # Just put the emotion onto the queue
                            logging.info(f"[EmotionMonitor] Target emotion '{dominant_emotion}' detected. Queuing trigger.")
                            self.proactive_queue.put(dominant_emotion)
                            self.last_triggered_emotion = dominant_emotion
                            self.last_trigger_time = current_time
                        else:
                             logging.debug(f"[EmotionMonitor] '{dominant_emotion}' detected again recently. Not queuing.")

                    # --- Reset last triggered if emotion changes or becomes non-target ---
                    elif dominant_emotion != self.last_triggered_emotion and dominant_emotion not in self.target_emotions:
                         if self.last_triggered_emotion is not None:
                              logging.debug(f"[EmotionMonitor] Resetting last triggered emotion from '{self.last_triggered_emotion}'.")
                              self.last_triggered_emotion = None

                except json.JSONDecodeError:
                    logging.warning(f"[EmotionMonitor] Received non-JSON response from {self.endpoint_url}")
                except Exception as e_parse:
                     logging.error(f"[EmotionMonitor] Error processing emotion data: {e_parse}", exc_info=True)

            except RequestException as e_req:
                logging.warning(f"[EmotionMonitor] Could not connect to {self.endpoint_url}: {e_req}")
            except Exception as e_poll:
                logging.error(f"[EmotionMonitor] Unexpected error in polling loop: {e_poll}", exc_info=True)

            # --- Wait for the next poll interval (interruptible sleep) ---
            while time.time() < next_check_time and self._running:
                 time.sleep(0.5)

        logging.info("[EmotionMonitor] Polling thread stopped.")

    # --- start() and stop() methods remain the same ---
    def start(self):
        if self._thread is not None and self._thread.is_alive():
            logging.warning("[EmotionMonitor] Thread already running.")
            return
        logging.info("[EmotionMonitor] Starting background polling thread...")
        self._running = True
        self._thread = threading.Thread(target=self._poll_emotion, daemon=True)
        self._thread.start()

    def stop(self):
        logging.info("[EmotionMonitor] Stopping background polling thread...")
        self._running = False
