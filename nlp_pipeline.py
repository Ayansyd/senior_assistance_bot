# nlp_pipeline.py
import re
import json
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# Import the LLM interface to make calls from here
from llm_inference import OllamaLLM # Assuming this is correctly defined elsewhere

class NLPProcessor:
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        # Optional: Initialize LLM engine here if only used within NLPProcessor
        # self.llm_engine = OllamaLLM(...)

    # --- Sentiment analysis --- (Keep as is)
    def _analyze_sentiment(self, text: str) -> tuple[float, str]:
        vs = self.sentiment_analyzer.polarity_scores(text)
        score = vs['compound']
        if score >= 0.05: label = "positive"
        elif score <= -0.05: label = "negative"
        else: label = "neutral"
        return score, label

    # --- MODIFIED: Improved basic preference extraction ---
    def _extract_preferences(self, text_lower: str) -> dict:
        preferences = {"likes": [], "dislikes": []}
        min_pref_length = 3 # Minimum characters for a preference item

        # Regex to capture content after like/dislike verbs, stopping at punctuation or certain conjunctions
        # (?i) makes it case-insensitive within the group if needed later, but text_lower handles most cases
        # Using a non-greedy match '.+?' and explicit terminators
        like_match = re.search(r"i (?:like|love|enjoy|prefer)\s+(.+?)(?:\.|\!|\?|,| because| but| although|$)", text_lower)
        if like_match:
            potential_prefs = like_match.group(1).strip()
            # Split by 'and' or ',', then clean up
            items = re.split(r'\s+and\s+|\s*,\s*', potential_prefs)
            for item in items:
                cleaned_item = item.strip()
                # Basic filter: not too short, not just digits, not a pronoun like 'it' or 'them'
                if len(cleaned_item) >= min_pref_length and not cleaned_item.isdigit() and cleaned_item not in ['it', 'them', 'this', 'that']:
                    preferences["likes"].append(cleaned_item)

        # Similar pattern for dislikes
        dislike_match = re.search(r"i (?:dislike|hate|don't like|do not like)\s+(.+?)(?:\.|\!|\?|,| because| but| although|$)", text_lower)
        if dislike_match:
            potential_prefs = dislike_match.group(1).strip()
            items = re.split(r'\s+and\s+|\s*,\s*', potential_prefs)
            for item in items:
                cleaned_item = item.strip()
                if len(cleaned_item) >= min_pref_length and not cleaned_item.isdigit() and cleaned_item not in ['it', 'them', 'this', 'that']:
                     preferences["dislikes"].append(cleaned_item)

        return preferences
    # --- END MODIFICATION ---


    # --- MODIFIED: LLM-based Profile Information Extraction with Stricter JSON and Preference Instructions ---
    def extract_profile_updates_with_llm(self, utterance: str, llm_engine: OllamaLLM) -> dict:
        """
        Uses the LLM to analyze an utterance and extract structured information
        suitable for updating the user profile. Returns a dictionary.
        """
        # MODIFIED: Stricter JSON and preference instructions
        extraction_prompt = f"""
Analyze the following user utterance for user profile insights (activities, events, preferences, notes).
Format the output STRICTLY as a valid JSON object using **double quotes** for all keys and string values.
Example format:
{{
  "activities": [{{"name": "...", "sentiment": "positive/negative/neutral", "details": "..."}}],
  "events": [{{"description": "...", "sentiment": "positive/negative/neutral"}}],
  "new_likes": ["specific item", "another preference"],
  "new_dislikes": ["thing disliked"],
  "summary_note": "Brief observation about the user or conversation context."
}}
**Important for Preferences:** For "new_likes" and "new_dislikes", ONLY list specific nouns or short noun phrases representing the core items mentioned. Avoid full sentences, questions, explanations, or vague terms. If no clear new preferences are stated, use empty lists [].
Ensure the entire output is ONLY the JSON object, starting with {{ and ending with }}.

User Utterance: "{utterance}"

JSON Analysis:
"""
        # END MODIFICATION

        logging.info("[NLP Processor] Requesting profile extraction from LLM...")
        try:
            # Use lower temperature for more deterministic JSON extraction
            response = llm_engine.generate_response(extraction_prompt, max_tokens=256, temperature=0.1) # Lowered temp further
            logging.info(f"[NLP Processor] LLM Extraction Raw Response: {response}")

            # Attempt to parse the JSON response - look for start/end markers robustly
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                logging.info(f"[NLP Processor] Attempting to parse JSON: {json_str}")
                extracted_data = json.loads(json_str)
                # Basic validation (check if it's a dictionary)
                if isinstance(extracted_data, dict):
                    return extracted_data
                else:
                    logging.warning("[NLP Processor] LLM extraction response parsed but was not a valid JSON object (dict).")
                    return {}
            else:
                logging.warning("[NLP Processor] Could not find JSON object markers {{...}} in LLM extraction response.")
                return {}

        except json.JSONDecodeError as e:
            logging.error(f"[NLP Processor] Failed to decode JSON from LLM extraction response: {e}\nResponse was: {response}")
            return {} # Return empty dict on JSON error
        except Exception as e:
            logging.error(f"[NLP Processor] Error during LLM profile extraction call: {e}", exc_info=True)
            return {} # Return empty dict on other errors

    # --- Basic Intent Parsing --- (Keep as is)
    def parse_user_input(self, text: str) -> dict:
        """
        Performs basic parsing: sentiment, simple preferences, intent keywords.
        LLM extraction should be called separately.
        """
        text_lower = text.lower().strip()
        analysis_result = {"raw_text": text}

        sentiment_score, sentiment_label = self._analyze_sentiment(text)
        analysis_result["sentiment_score"] = sentiment_score
        analysis_result["sentiment_label"] = sentiment_label

        # Use the *modified* function here
        preferences = self._extract_preferences(text_lower)
        analysis_result["likes"] = preferences["likes"]
        analysis_result["dislikes"] = preferences["dislikes"]

        intent = "unknown"
        entities = {}
        # --- Intent keyword spotting --- (Keep as is)
        if ("bring me" in text_lower) or ("fetch me" in text_lower):
             match = re.search(r"(?:bring me|fetch me)\s+(?:a|an|the)\s+(.*)", text_lower)
             if match: intent = "fetch_object"; entities["object"] = match.group(1).strip()
        elif "remind me" in text_lower:
             reminder_phrase = text_lower.split("remind me", 1)[1].strip()
             intent = "set_reminder"; entities["reminder_text"] = reminder_phrase
        elif any(phrase in text_lower for phrase in ["i took my meds", "i did it", "done", "i have done that", "acknowledged", "reminder done"]):
             intent = "ack_reminder"; entities["details"] = text
        elif any(phrase in text_lower for phrase in ["weather", "forecast", "temperature", "rain"]):
             intent = "get_weather"; entities["details"] = text
        elif any(phrase in text_lower for phrase in ["joke", "funny"]):
             intent = "tell_joke"; entities["details"] = text
        elif any(phrase in text_lower for phrase in ["bored", "what should i do", "suggest something", "ideas"]):
             intent = "request_suggestion"; entities["details"] = text
        elif text.strip().endswith("?") or any(q in text_lower for q in ["what is", "tell me about", "explain", "who is", "where is", "how do i", "can you"]): # Expanded question triggers
             intent = "question" # Generic question intent
             entities["query"] = text # The full question text

        analysis_result["intent"] = intent
        analysis_result["entities"] = entities
        return analysis_result


    # --- Build LLM Prompt --- (Keep the previous enhanced version)
    def build_llm_prompt(self,
                         intent: str,
                         analysis: dict,
                         conversation_history: str,
                         user_profile_summary: dict,
                         rag_context: str | None = None # Use | None type hint
                         ) -> str:
        # --- [Keep the existing enhanced prompt building logic here from previous steps] ---
        # ... (Includes system_persona, RAG instructions, profile_text, history, task description) ...
        # Make sure this part uses the previously refined version with strong RAG instructions etc.
        # --- [ The existing logic should already be good here ] ---

        # --- Profile Summary ---
        profile_text = (
            "[User Profile Summary]\n"
            f"- Name: {user_profile_summary.get('name', 'User')}\n"
            f"- Recent Mood: {user_profile_summary.get('recent_mood_label', 'Unknown')}\n"
            f"- Known Likes: {', '.join(user_profile_summary.get('likes', [])) or 'None'}\n"
            f"- Known Dislikes: {', '.join(user_profile_summary.get('dislikes', [])) or 'None'}\n"
            f"- Recently Enjoyed Activities: {', '.join(user_profile_summary.get('recent_positive_activities', [])) or 'None'}\n"
            f"- Assistant Notes: {'; '.join(user_profile_summary.get('summary_notes', [])) or 'None'}\n"
            "[End User Profile Summary]\n\n"
        )

        # --- MODIFIED: Persona and Core Instructions ---
        system_persona = (
            "You are Alice, an elderly care assistant. Your primary goal is to be helpful, compassionate, and engaging. "
            "Use the provided User Profile Summary to personalize your responses, showing awareness of their mood, preferences, and past experiences. "
            "Keep responses concise and friendly. "
            # --- STRONGER RAG INSTRUCTION ---
            "**IMPORTANT: When answering questions or providing suggestions related to health, medical conditions (like arthritis or pain), medication, or doctor's advice, you MUST prioritize information found in the [Retrieved Information] block if it is present and relevant. Mention specific advice or prescriptions found there directly in your response.** "
            # --- END STRONGER RAG INSTRUCTION ---
            "If the user seems bored, sad, or asks for general suggestions unrelated to health (intent 'request_suggestion'), proactively suggest an activity they might enjoy based on their profile. "
        )

        # --- RAG Context Instructions & Insertion ---
        rag_instructions = ""
        rag_content_block = ""
        if rag_context and not rag_context.startswith("[Error"):
            rag_instructions = (
                "**You have been provided with potentially relevant information below from a knowledge base.** "
                "Use this information according to the IMPORTANT health-related instruction above. "
                "If the query is not health-related but the information seems relevant, use it *briefly* to enhance your answer. "
                "Do not mention the retrieval process itself. "
                "If the retrieved information is clearly not relevant, ignore it.\n\n"
            )
            rag_content_block = (
                "[Retrieved Information - Use if Relevant]\n"
                f"{rag_context}\n"
                "[End Retrieved Information]\n\n"
            )
        elif rag_context:
             rag_content_block = (
                  f"[Note: There was an error retrieving information: {rag_context}]\n\n"
             )

        # --- Combine Prompt Sections ---
        prompt = system_persona + rag_instructions + profile_text + rag_content_block

        # --- History ---
        prompt += "[Conversation History]\n"
        prompt += conversation_history if conversation_history else "(No previous conversation turns)\n"
        prompt += "[End Conversation History]\n\n"

        # --- Current Task ---
        prompt += "[Current Task]\n"
        entities = analysis.get("entities", {})
        raw_text = analysis.get("raw_text", "")
        sentiment_label = analysis.get("sentiment_label", "unknown")

        # (Keep the intent-specific task descriptions from the previous version)
        if intent == "question":
            query = entities.get("query", raw_text)
            prompt += (f"User asked: '{query}'. "
                       "**Provide a concise answer.** Prioritize the [Retrieved Information] if relevant (especially for health topics). Otherwise, use profile/history or general knowledge briefly.")
        elif intent == "request_suggestion":
             prompt += (f"User asked: '{raw_text}'. They want a suggestion. "
                        "**Check [Retrieved Information] first for relevant health advice/prescriptions and state it concisely if found.** "
                        "Then, *briefly* suggest ONE relevant activity based on their profile (positive activities/notes) suitable for their mood. Ask if they'd like to try it.")
        elif intent == "fetch_object":
             obj = entities.get("object", "something"); prompt += f"User requested fetching '{obj}'. Provide a *very short* confirmation."
        elif intent == "set_reminder":
             reminder_text = entities.get("reminder_text", ""); prompt += f"User wants to set a reminder: '{reminder_text}'. Provide a *short* acknowledgement."
        elif intent == "ack_reminder":
             prompt += "User acknowledged a reminder. Give a *brief*, positive acknowledgement."
        elif intent == "get_weather":
             prompt += "User asked for the weather. Give a *brief*, simple simulated weather update."
        elif intent == "tell_joke":
             prompt += "User wants a joke. Tell a *short* joke or witty line."
        else: # Unknown intent or general chat
            prompt += (f"User said: '{raw_text}' (Mood: {sentiment_label}). "
                       "Provide a polite, helpful, and **concise** response based on profile/history. Use [Retrieved Information] briefly if relevant.")

        prompt += "\n[End Current Task]\n\n"
        prompt += "Assistant:"

        return prompt