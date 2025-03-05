# nlp_pipeline.py
import re

class NLPProcessor:
    def __init__(self):
        pass

    def parse_user_input(self, text: str, conversation_context: str = "") -> dict:
        """
        Attempt to detect intent and extract simple entities.
        We can also leverage conversation_context if needed.
        """

        text_lower = text.lower().strip()

        # Check for fetch object intent
        if ("bring me" in text_lower) or ("fetch me" in text_lower):
            match = re.search(r"(?:bring me|fetch me)\s+a\s+(.*)", text_lower)
            if match:
                return {
                    "intent": "fetch_object",
                    "object": match.group(1).strip()
                }

        # Check for reminder intent
        if "remind me" in text_lower:
            # We'll just capture anything after "remind me"
            reminder_phrase = text_lower.split("remind me", 1)[1].strip()
            return {
                "intent": "set_reminder",
                "reminder_text": reminder_phrase,
                "raw_text": text
            }

        # NEW: check for user acknowledging a reminder
        # This is naive; you can add more phrases or a small pattern
        if any(phrase in text_lower for phrase in ["i took my meds", "i did it", "done", "i have done that"]):
            return {
                "intent": "ack_reminder",
                "details": text
            }

        # Check for weather intent
        if "weather" in text_lower:
            return {
                "intent": "get_weather",
                "details": text
            }

        # Check for joke intent
        if "joke" in text_lower:
            return {
                "intent": "tell_joke",
                "details": text
            }

        # Fallback
        return {"intent": "unknown", "text": text}

    def build_llm_prompt(self, intent: str, entities: dict, conversation_history: str) -> str:
        """
        Build a prompt to feed into the LLM, using both the parsed intent/entities
        and the existing conversation history.
        The assistant's persona is 'Alice, an elderly care assistant.'
        """

        # Start with the conversation history
        prompt = (
            "You are Alice, an elderly care assistant. "
            "Respond helpfully, compassionately, and concisely.\n\n"
        )
        prompt += conversation_history
        prompt += "\nAssistant: "

        if intent == "fetch_object":
            obj = entities.get("object", "something")
            prompt += (
                "The user has requested: fetch an object.\n"
                f"Object requested: '{obj}'.\n"
                "Please generate a short, friendly response to confirm you understood."
            )
        elif intent == "set_reminder":
            reminder_text = entities.get("reminder_text", "")
            prompt += (
                "The user wants to set a reminder.\n"
                f"Reminder details: {reminder_text}.\n"
                "Please generate a short, friendly acknowledgement."
            )
        elif intent == "ack_reminder":
            prompt += (
                "The user says they have completed a previously set reminder.\n"
                "Please provide a short, friendly response acknowledging their completion."
            )
        elif intent == "get_weather":
            prompt += (
                "The user asked for the weather.\n"
                "Please provide a short, simple response (you can simulate a response)."
            )
        elif intent == "tell_joke":
            prompt += (
                "The user wants to hear a joke.\n"
                "Please respond with a short joke or witty line."
            )
        else:
            raw_text = entities.get("text", "")
            prompt += (
                f"The user said: '{raw_text}'.\n"
                "Provide a polite, helpful response based on the conversation so far."
            )

        return prompt
