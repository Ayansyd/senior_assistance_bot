# reminder_manager.py

import time
import threading
import re
from datetime import datetime, timedelta
from typing import Callable, List, Optional, Tuple  # Added Tuple for type annotations
from math import floor

class Reminder:
    def __init__(self, message: str, due_time: datetime):
        self.message = message
        self.due_time = due_time
        self.acknowledged = False    # Track if user acknowledged or completed this reminder
        self.last_reminded: Optional[datetime] = None  # Track last time we reminded user

class ReminderManager:
    def __init__(self, speak_callback: Callable[[str], None], check_interval: float = 2.0):
        """
        Manages scheduled reminders in a background thread.

        :param speak_callback: A function that takes a string (the reminder message)
                               and announces it (e.g., TTS).
        :param check_interval: How often (in seconds) to check reminders.
        """
        self.reminders: List[Reminder] = []
        self.running = True
        self.speak_callback = speak_callback
        self.check_interval = check_interval

        # Start the background loop
        self.thread = threading.Thread(target=self._reminder_loop, daemon=True)
        self.thread.start()

    def schedule_reminder(self, message: str, offset_minutes: float):
        """
        Schedule a reminder message to be spoken at now + offset_minutes.
        Stores the first 'due_time' and repeats every 10 seconds until acknowledged.
        """
        due_time = datetime.now() + timedelta(minutes=offset_minutes)
        reminder = Reminder(message, due_time)
        self.reminders.append(reminder)
        print(f"[ReminderManager] Scheduled reminder: '{message}' at {due_time.isoformat()}")

    def acknowledge_reminder(self, user_text: str = ""):
        """
        Mark any active reminder(s) as acknowledged if the user indicates they've completed it.
        This is a naive approach that acknowledges all unacknowledged reminders if the user says
        something like 'I took my meds' or 'I did it' or 'Done.'

        :param user_text: The user's input text to potentially match specific reminders.
        """
        for r in self.reminders:
            if not r.acknowledged:
                r.acknowledged = True
                print(f"[ReminderManager] Acknowledged reminder: '{r.message}'")

    def _reminder_loop(self):
        """
        Background loop to check reminders.
        - If a reminder is due and not acknowledged, remind the user.
        - Then every 10 seconds, remind again unless the user acknowledges.
        """
        repeat_interval = 10.0  # seconds

        while self.running:
            now = datetime.now()

            for r in self.reminders:
                # If user hasn't acknowledged
                if not r.acknowledged:
                    # If the current time is >= due_time
                    if now >= r.due_time:
                        # Check if it's time to remind (no last_reminded or last_reminded >= 10s ago)
                        if r.last_reminded is None or (now - r.last_reminded).total_seconds() >= repeat_interval:
                            # Speak the reminder
                            reminder_text = f"Reminder: {r.message}"
                            self.speak_callback(reminder_text)
                            r.last_reminded = now

            # Remove reminders that are acknowledged
            self.reminders = [r for r in self.reminders if not r.acknowledged]

            time.sleep(self.check_interval)

    def stop(self):
        """ Stop the background thread. """
        self.running = False
        self.thread.join()

    def parse_reminder_offset_and_message(self, text: str) -> Tuple[float, str]:
        """
        Attempt to find an integer or float minute offset in the user's text.
        Remains flexible: "in 1 minute", "in 30 seconds", "in 1.5 minutes", etc.
        If no offset is found, defaults to 1 minute.
        Returns (offset_in_minutes, final_message).

        Example user text:
          - "to take my meds in 10 minutes"
          - "in 10 minutes to take my meds"
          - "take my meds after 15 mins"
          - "take my meds 15 minutes from now"
          - "take my meds in 30 seconds"
        """
        offset_minutes = 1.0  # Default offset if not found
        text_lower = text.lower()

        # Regex pattern to match various time expressions
        offset_pattern = re.compile(
            r"(?:in|after)?\s*(\d+(?:\.\d+)?)\s*(?:min|minutes|mins|sec|secs|seconds)\b(?:\s*from\s+now)?",
            re.IGNORECASE
        )

        match = offset_pattern.search(text_lower)
        if match:
            raw_value = match.group(1).strip()  # Could be int or float
            number = float(raw_value)

            # Determine if the time unit is seconds or minutes
            segment = match.group(0).lower()
            if any(token in segment for token in ["sec", "second"]):
                # Convert seconds to minutes
                offset_minutes = number / 60.0
            else:
                # It's minutes
                offset_minutes = number

        # Remove the matched offset phrase from the final message
        cleaned_message = offset_pattern.sub("", text_lower)
        cleaned_message = cleaned_message.replace("to ", "", 1).strip()
        # Optionally, further clean the message by removing common filler words
        cleaned_message = re.sub(r"\band\b", "", cleaned_message).strip()

        return offset_minutes, cleaned_message
