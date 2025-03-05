# conversation.py

from collections import deque
from typing import List

class ConversationManager:
    def __init__(self, max_history: int = 5):
        """
        Manages multi-turn conversation context by storing recent exchanges.
        
        :param max_history: Maximum number of turns to keep in memory.
        """
        self.conversation_history = deque(maxlen=max_history)

    def add_user_utterance(self, text: str):
        """
        Add user utterance to conversation history.
        """
        self.conversation_history.append(("user", text))

    def add_system_utterance(self, text: str):
        """
        Add system (assistant) utterance to conversation history.
        """
        self.conversation_history.append(("system", text))

    def get_history_as_text(self) -> str:
        """
        Return conversation history as a text block that can be fed into the LLM prompt.
        This is a simple version. More advanced methods could format differently.
        """
        lines = []
        for speaker, content in self.conversation_history:
            prefix = "User:" if speaker == "user" else "Assistant:"
            lines.append(f"{prefix} {content}")
        return "\n".join(lines)

    def clear_history(self):
        """
        Clear the conversation history.
        """
        self.conversation_history.clear()
