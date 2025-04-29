# heart_rate_monitor.py
import threading
import time
import logging
import queue

class HeartRateMonitor:
    def __init__(self,
                 poll_interval_seconds: int,
                 proactive_queue: queue.Queue,
                 # state_lock: threading.Lock, # REMOVED
                 # get_assistant_busy_func, # REMOVED
                 get_heart_rate_func,     # Function to get latest HR reading
                 threshold_bpm: int = 120):
        """
        Monitors heart rate readings in a background thread.
        Puts high heart rate alerts onto the proactive_queue.

        Args:
            poll_interval_seconds: How often to check the heart rate (in seconds).
            proactive_queue: A queue.Queue to put high heart rate alerts onto.
            get_heart_rate_func: Callable returning the latest heart rate reading dict or None.
            threshold_bpm: Heart rate threshold to trigger an alert.
        """
        self.poll_interval = poll_interval_seconds
        self.proactive_queue = proactive_queue
        # self.state_lock = state_lock # REMOVED
        # self.get_assistant_busy = get_assistant_busy_func # REMOVED
        self.get_latest_reading = get_heart_rate_func
        self.threshold = threshold_bpm

        self._thread = None
        self._running = False
        self.alert_active = False # Track if an alert is already pending/active
        self.last_alert_time = 0
        self.min_time_between_alerts = 300 # e.g., 5 minutes

    def _monitor_loop(self):
        """The main loop for the background thread."""
        logging.info("[HeartRateMonitor] Monitor loop starting.")
        while self._running:
            next_check_time = time.time() + self.poll_interval
            try:
                reading = self.get_latest_reading()
                current_time = time.time()
                logging.debug(f"[HeartRateMonitor] Checking HR. Current reading: {reading}. Alert active: {self.alert_active}. Last alert: {self.last_alert_time:.0f}")

                if reading and reading.get("heart_rate") is not None:
                    current_hr = reading["heart_rate"]
                    # --- Check Threshold ---
                    if current_hr > self.threshold:
                        # --- Check Cooldown and Active Alert ---
                        time_since_last_alert = current_time - self.last_alert_time
                        if not self.alert_active and time_since_last_alert > self.min_time_between_alerts:
                            # --- REMOVED Busy Check ---
                            # Just put the alert onto the queue
                            logging.info(f"[HeartRateMonitor] High heart rate ({current_hr} bpm) detected. Queuing trigger.")
                            self.proactive_queue.put({"type": "high_hr", "value": current_hr})
                            self.alert_active = True # Set flag
                            self.last_alert_time = current_time # Record time
                        else:
                            logging.debug(f"[HeartRateMonitor] High HR ({current_hr}) detected, but skipping queue. Alert active: {self.alert_active}, Time since last: {time_since_last_alert:.0f}s.")

                    # --- Reset Alert Flag if HR drops below threshold ---
                    elif current_hr <= self.threshold and self.alert_active:
                        logging.info(f"[HeartRateMonitor] Heart rate ({current_hr} bpm) back in normal range. Resetting alert flag.")
                        self.alert_active = False
                else:
                    logging.debug("[HeartRateMonitor] No recent heart rate reading available.")

            except Exception as e_poll:
                logging.error(f"[HeartRateMonitor] Unexpected error in monitoring loop: {e_poll}", exc_info=True)

            # --- Wait for the next poll interval (interruptible sleep) ---
            while time.time() < next_check_time and self._running:
                 time.sleep(0.5)

        logging.info("[HeartRateMonitor] Monitoring thread stopped.")

    # --- start() and stop() methods remain the same ---
    def start(self):
        if self._thread is not None and self._thread.is_alive():
            logging.warning("[HeartRateMonitor] Thread already running.")
            return
        logging.info("[HeartRateMonitor] Starting background monitoring thread...")
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        logging.info("[HeartRateMonitor] Stopping background monitoring thread...")
        self._running = False
