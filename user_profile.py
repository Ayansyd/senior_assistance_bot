# user_profile.py

class UserProfileManager:
    def __init__(self, user_name: str = "User", language: str = "en"):
        """
        Manages user preferences, such as language and TTS settings.
        
        :param user_name: The name of the user (for personalization).
        :param language: The default language code for TTS or prompts.
        """
        self.user_name = user_name
        self.language = language
        # Add other preferences if needed, e.g., TTS voice, volume, etc.

    def get_user_name(self) -> str:
        return self.user_name

    def set_user_name(self, name: str):
        self.user_name = name

    def get_language(self) -> str:
        return self.language

    def set_language(self, language_code: str):
        self.language = language_code
