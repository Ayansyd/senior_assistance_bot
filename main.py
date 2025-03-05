# main.py
import time
from stt import LocalSTT
from llm_inference import OllamaLLM
from nlp_pipeline import NLPProcessor
from tts import GTTS_TTS
from conversation import ConversationManager
from user_profile import UserProfileManager
from logger_module import InteractionLogger

from reminder_manager import ReminderManager

def main():
    # 1. Initialize modules
    vosk_model_path = "vosk-model-small-en-us-0.15"  # Example Vosk model
    ollama_model_name = "llama3.2:latest"           # Ensure this matches your model's exact name

    # Speech-to-Text
    stt_engine = LocalSTT(model_path=vosk_model_path, sample_rate=16000)

    # Local LLM via Ollama
    llm_engine = OllamaLLM(model_name=ollama_model_name)

    # NLP Processor
    nlp_processor = NLPProcessor()

    # User Profile
    user_profile = UserProfileManager(user_name="Elderly User", language="en")
    tts_engine = GTTS_TTS(language=user_profile.get_language())

    # Conversation Manager
    convo_manager = ConversationManager(max_history=5)

    # Logger
    logger = InteractionLogger(log_file="assistant_log.txt")

    # 2. Reminder Manager
    reminder_manager = ReminderManager(speak_callback=tts_engine.speak, check_interval=2.0)

    print("=== Expanded Voice Assistant POC (with Repeated Reminders) ===")
    print("Press Ctrl+C to exit.")
    time.sleep(1)

    try:
        # 3. Main loop
        while True:
            print("\n--- Waiting for your command (5 seconds) ---")
            user_text = stt_engine.listen_once(duration=5.0)

            if not user_text:
                print("[No speech detected, or speech was inaudible]")
                continue

            print(f"[User said]: {user_text}")

            # Add user utterance to the conversation manager
            convo_manager.add_user_utterance(user_text)

            # 4. NLP Parsing
            conversation_context = convo_manager.get_history_as_text()
            analysis = nlp_processor.parse_user_input(user_text, conversation_context=conversation_context)
            intent = analysis.get("intent", "unknown")

            if intent == "set_reminder":
                reminder_text = analysis.get("reminder_text", "")
                offset_minutes, final_reminder_msg = reminder_manager.parse_reminder_offset_and_message(reminder_text)

                if not final_reminder_msg.strip():
                    final_reminder_msg = "Reminder with no details"

                reminder_manager.schedule_reminder(final_reminder_msg, offset_minutes)
                local_msg = f"Got it. I'll remind you in {offset_minutes:.2f} minute(s) about: {final_reminder_msg}"
                print(local_msg)

            elif intent == "ack_reminder":
                # Acknowledge all unacknowledged reminders
                reminder_manager.acknowledge_reminder(user_text)
                print("[User acknowledged reminders]")

            # 5. Build LLM prompt
            prompt = nlp_processor.build_llm_prompt(intent, analysis, conversation_history=conversation_context)
            print(f"[LLM Prompt]: {prompt}")

            # 6. Get LLM response via Ollama
            llm_response = llm_engine.generate_response(prompt)
            print(f"[LLM Response]: {llm_response}")

            # Add system response to conversation context
            convo_manager.add_system_utterance(llm_response)

            # 7. TTS the response
            tts_engine.speak(llm_response)

            # 8. Log the interaction
            logger.log_interaction(user_text, llm_response)

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nExiting... Goodbye!")
    except Exception as e:
        print(f"[Error]: {e}")
        logger.log_error(str(e))
        time.sleep(2)
    finally:
        # Stop the reminder manager thread gracefully
        reminder_manager.stop()

if __name__ == "__main__":
    main()
