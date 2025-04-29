# user_profile.py
import json
import os
from datetime import datetime
from collections import deque
import logging # Use logging for profile messages

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class UserProfileManager:
    # Keep maxlen relatively small for recent mood, but store more activities/notes
    DEFAULT_MOOD_HISTORY_LEN = 10
    DEFAULT_ACTIVITY_HISTORY_LEN = 50
    DEFAULT_NOTES_LEN = 50

    def __init__(self, profile_path="user_profile.json", user_name: str = "User", language: str = "en"):
        self.profile_path = profile_path
        self.data = {
            "user_name": user_name,
            "language": language,
            "likes": [], # Keep simple likes/dislikes as quick reference
            "dislikes": [],
            # Store more structured activity/event data
            "activities": list(deque(maxlen=self.DEFAULT_ACTIVITY_HISTORY_LEN)), # [{"name": str, "sentiment": str, "details": str, "timestamp": isoformat}, ...]
            "events": list(deque(maxlen=self.DEFAULT_ACTIVITY_HISTORY_LEN)),     # [{"description": str, "sentiment": str, "timestamp": isoformat}, ...]
            "mood_history": list(deque(maxlen=self.DEFAULT_MOOD_HISTORY_LEN)), # [{"timestamp", "score", "label"}, ...]
            "llm_summary_notes": list(deque(maxlen=self.DEFAULT_NOTES_LEN)), # Store LLM-generated insights
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }
        self.load_profile()

    def _ensure_deque(self, key, maxlen):
        """Ensure a key holds a deque with the correct maxlen."""
        if key not in self.data or not isinstance(self.data[key], deque):
            self.data[key] = deque(self.data.get(key, []), maxlen=maxlen)
        elif self.data[key].maxlen != maxlen:
            # Recreate deque with new maxlen if it changed
             self.data[key] = deque(list(self.data[key]), maxlen=maxlen)


    def load_profile(self):
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, "r", encoding="utf-8") as f:
                    loaded_data = json.load(f)
                    # Convert lists back to deques after loading
                    for key, maxlen in [("activities", self.DEFAULT_ACTIVITY_HISTORY_LEN),
                                        ("events", self.DEFAULT_ACTIVITY_HISTORY_LEN),
                                        ("mood_history", self.DEFAULT_MOOD_HISTORY_LEN),
                                        ("llm_summary_notes", self.DEFAULT_NOTES_LEN)]:
                        if key in loaded_data and isinstance(loaded_data[key], list):
                             loaded_data[key] = deque(loaded_data[key], maxlen=maxlen)

                    self.data.update(loaded_data)
                    logging.info(f"[UserProfile] Loaded profile for {self.data.get('user_name', 'Unknown')} from {self.profile_path}")
            except (json.JSONDecodeError, IOError, TypeError) as e:
                logging.error(f"[UserProfile Error] Failed to load or parse profile from {self.profile_path}: {e}")
        else:
            logging.info(f"[UserProfile] No existing profile found at {self.profile_path}. Creating default.")
            self.save_profile() # Save default profile immediately

        # Ensure all deque structures exist with correct maxlen after load/init
        self._ensure_deque("activities", self.DEFAULT_ACTIVITY_HISTORY_LEN)
        self._ensure_deque("events", self.DEFAULT_ACTIVITY_HISTORY_LEN)
        self._ensure_deque("mood_history", self.DEFAULT_MOOD_HISTORY_LEN)
        self._ensure_deque("llm_summary_notes", self.DEFAULT_NOTES_LEN)


    def save_profile(self):
        self.data["last_updated"] = datetime.now().isoformat()
        # Convert deques to lists for JSON serialization
        data_to_save = self.data.copy()
        for key in ["activities", "events", "mood_history", "llm_summary_notes"]:
            if key in data_to_save and isinstance(data_to_save[key], deque):
                 data_to_save[key] = list(data_to_save[key])

        try:
            with open(self.profile_path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=4)
        except IOError as e:
            logging.error(f"[UserProfile Error] Failed to save profile to {self.profile_path}: {e}")
        except TypeError as e:
             logging.error(f"[UserProfile Error] Failed to serialize profile data: {e}")


    # --- Getters/Setters for basic info ---
    def get_user_name(self) -> str: return self.data.get("user_name", "User")
    def set_user_name(self, name: str): self.data["user_name"] = name
    def get_language(self) -> str: return self.data.get("language", "en")
    def set_language(self, language_code: str): self.data["language"] = language_code

    # --- Methods to update profile ---
    def add_like(self, item: str):
        item = item.strip().lower()
        if item and item not in self.data.get("likes", []):
            if "likes" not in self.data: self.data["likes"] = []
            self.data["likes"].append(item)
            logging.info(f"[UserProfile] Added like: {item}")

    def add_dislike(self, item: str):
        item = item.strip().lower()
        if item and item not in self.data.get("dislikes", []):
            if "dislikes" not in self.data: self.data["dislikes"] = []
            self.data["dislikes"].append(item)
            logging.info(f"[UserProfile] Added dislike: {item}")

    def add_activity(self, name: str, sentiment: str, details: str):
         """Adds a structured activity entry."""
         timestamp = datetime.now().isoformat()
         # Avoid adding duplicates based on name and recent timestamp? (Optional complexity)
         self.data["activities"].append({
             "name": name.strip().lower(),
             "sentiment": sentiment.lower(),
             "details": details.strip(),
             "timestamp": timestamp
         })
         logging.info(f"[UserProfile] Added activity: {name} ({sentiment})")

    def add_event(self, description: str, sentiment: str):
         """Adds a structured event entry."""
         timestamp = datetime.now().isoformat()
         self.data["events"].append({
             "description": description.strip(),
             "sentiment": sentiment.lower(),
             "timestamp": timestamp
         })
         logging.info(f"[UserProfile] Added event: {description} ({sentiment})")

    def record_mood(self, sentiment_score: float, sentiment_label: str):
        timestamp = datetime.now().isoformat()
        mood_entry = {"timestamp": timestamp, "score": sentiment_score, "label": sentiment_label}
        self.data["mood_history"].append(mood_entry)
        logging.info(f"[UserProfile] Recorded mood: {sentiment_label} ({sentiment_score:.2f})")

    def add_llm_note(self, note: str):
        """Adds a summary note generated by the LLM."""
        if note and note not in self.data["llm_summary_notes"]:
            self.data["llm_summary_notes"].append(note.strip())
            logging.info(f"[UserProfile] Added LLM Note: {note[:80]}...") # Log truncated note

    # --- Get summary for LLM ---
    def get_profile_summary(self) -> dict:
        """Returns a dictionary summary, preferring recent/relevant items."""
        # Get recent mood
        latest_mood = self.data["mood_history"][-1] if self.data.get("mood_history") else {"label": "Unknown"}

        # Get recent positive activities (Example: last 5)
        positive_activities = [
            act['name'] for act in list(self.data['activities'])[-5:]
            if act.get('sentiment') == 'positive'
        ]
        # Get recent important events (Example: last 3)
        recent_events = [
            evt['description'] for evt in list(self.data['events'])[-3:]
        ]
        # Get recent notes (Example: last 5)
        recent_notes = list(self.data['llm_summary_notes'])[-5:]

        return {
            "name": self.get_user_name(),
            "recent_mood_label": latest_mood.get("label", "Unknown"),
            "likes": self.data.get("likes", []),
            "dislikes": self.data.get("dislikes", []),
            "recent_positive_activities": positive_activities,
            "recent_events": recent_events, # Could add sentiment here too
            "summary_notes": recent_notes,
        }
    

    