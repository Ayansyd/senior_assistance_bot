# main.py
import time
import sys
import logging
import os # For format_docs path handling
import rag_setup # Import the setup module
import threading # <-- Add threading import
import queue     # <-- Add queue import
from emotion_monitor import EmotionMonitor # <-- Import the emotion monitor class
import heartbeat_server # <-- Import the server module
from heart_rate_monitor import HeartRateMonitor # <-- Import the HR monitor

# Core application imports
from stt import WhisperSTT # Assuming WhisperSTT is in stt.py
from llm_inference import OllamaLLM
from nlp_pipeline import NLPProcessor
from tts import GTTS_TTS
from conversation import ConversationManager
from user_profile import UserProfileManager
from logger_module import InteractionLogger # Keep this for general logging
from reminder_manager import ReminderManager
from langchain_core.documents import Document # To format retrieved docs


# --- Configuration ---
WHISPER_MODEL_NAME = "base.en" # Or your preferred Whisper model
OLLAMA_MODEL_NAME = "llama3.2:latest" # Or your specific Ollama model
USER_PROFILE_PATH = "elderly_user_profile.json" # Profile save location
KNOWLEDGE_BASE_DIR = rag_setup.KNOWLEDGE_BASE_DIR # Get from rag_setup
VECTOR_DB_DIR = rag_setup.PERSIST_DIRECTORY # Get from rag_setup
STT_SAMPLE_RATE = 16000
STT_LISTEN_DURATION = 7.0 # Adjust as needed
CONVERSATION_HISTORY_LENGTH = 10 # How many turns to keep in short-term memory
REMINDER_CHECK_INTERVAL = 2.0 # How often reminder thread checks (seconds)
MIN_UTTERANCE_LEN_FOR_EXTRACTION = 15 # Minimum user utterance length to trigger LLM profile extraction
RAG_NUM_DOCS = 3 # Number of documents to retrieve

# --- Emotion Monitor Configuration ---
FER_ENDPOINT_URL = "http://192.168.0.243:5000/emotion" # User provided endpoint
FER_POLL_INTERVAL = 35 # Seconds (between 30-40s)
FER_CONFIDENCE_THRESHOLD = 0.6
FER_TARGET_EMOTIONS = ["sad", "happy"]

# --- Heart Rate Monitor Configuration ---
HEARTBEAT_SERVER_HOST = '0.0.0.0' # Listen on all interfaces for the heartbeat server
HEARTBEAT_SERVER_PORT = 5002     # Port for the heartbeat server
HR_MONITOR_POLL_INTERVAL = 30    # Check stored HR every 30 seconds
HR_THRESHOLD_BPM = 120           # Threshold for high heart rate alert

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# --- Shared State for TTS Access ---
# Lock to ensure only one thread uses TTS (speaking) at a time
tts_lock = threading.Lock()
# Queues for background monitors to signal the main thread
proactive_emotion_queue = queue.Queue()
proactive_heart_rate_queue = queue.Queue()

# --- Helper Function to Format RAG Docs ---
def format_docs(docs: list[Document]) -> str | None: # Return None if no docs
    """Convert retrieved LangChain Documents into a simple string format for the prompt."""
    if not docs:
        return None
    formatted = []
    for i, doc in enumerate(docs):
        content = doc.page_content
        # Attempt to get source metadata gracefully
        metadata = doc.metadata or {} # Ensure metadata is a dict
        source = metadata.get('source', 'Unknown Source')
        page = metadata.get('page', None)
        try:
             # Replace backslashes for consistency if paths might come from Windows
             source_path = str(source).replace("\\", "/")
             source_name = os.path.basename(source_path) if source else 'Unknown Source'
        except Exception as e:
             logging.warning(f"Could not get basename for source '{source}': {e}")
             source_name = 'Invalid Source Path'
        source_str = f"Source: {source_name}"
        if page is not None:
             # Ensure page is presented cleanly
             try:
                 source_str += f" (Page {int(page) + 1})" # Often 0-indexed, display as 1-indexed
             except (ValueError, TypeError):
                 source_str += f" (Page {page})" # Display as is if not convertible to int
        formatted.append(f"{i+1}. {source_str}\n   Content: {content}")
    # Join with clear separators
    return "\n---\n".join(formatted)


# --- Main Function ---
def main():
    # No global needed for tts_lock as it's passed/used directly

    logging.info("=== Initializing Voice Assistant (Always Listening Logic) ===")
    # Initialize handles to None for cleanup safety in finally block
    emotion_monitor = None
    heart_rate_monitor = None
    reminder_manager = None
    user_profile = None

    # --- Initialize Core Modules ---
    try:
        # LLM Engine
        llm_engine = OllamaLLM(model_name=OLLAMA_MODEL_NAME, api_url="http://localhost:11434")
        # User Profile
        user_profile = UserProfileManager(profile_path=USER_PROFILE_PATH, user_name="Valued User")
        # Speech-to-Text
        stt_engine = WhisperSTT(model_name=WHISPER_MODEL_NAME, sample_rate=STT_SAMPLE_RATE)
        # NLP Processor
        nlp_processor = NLPProcessor()
        # Text-to-Speech
        tts_engine = GTTS_TTS(language=user_profile.get_language())
        # Conversation Manager
        convo_manager = ConversationManager(max_history=CONVERSATION_HISTORY_LENGTH)
        # Interaction Logger
        interaction_logger = InteractionLogger(log_file="knowledge_base/assistant_interaction_log.txt")

        # Reminder Manager Callback using non-blocking tts_lock
        def safe_speak_callback_reminder(text):
            """Wrapper for Reminder TTS callback using non-blocking lock."""
            if tts_lock.acquire(blocking=False): # Try to get lock without waiting
                logging.info(f"[Reminder Speaking]: {text}")
                try:
                    tts_engine.speak(text)
                except Exception as e:
                    logging.error(f"[Reminder TTS Error] {e}", exc_info=True)
                    interaction_logger.log_error(f"Rem TTS failed: {e}")
                finally:
                    tts_lock.release() # Release lock after speaking attempt
            else:
                 # If lock couldn't be acquired, TTS is busy with main response or proactive msg
                 logging.warning(f"[Reminder Skipped] TTS busy, could not speak reminder: {text}")

        reminder_manager = ReminderManager(speak_callback=safe_speak_callback_reminder, check_interval=REMINDER_CHECK_INTERVAL)

        # Initialize RAG Components
        force_recreate = "--force-rag-recreate" in sys.argv # Check command line arg
        rag_retriever = rag_setup.initialize_rag(force_recreate_db=force_recreate)
        if rag_retriever is None: logging.warning("RAG system failed to initialize.")

        # Start Heartbeat Server in background thread
        heartbeat_server.start_server(host=HEARTBEAT_SERVER_HOST, port=HEARTBEAT_SERVER_PORT)

        # Initialize and Start Emotion Monitor (no lock needed)
        emotion_monitor = EmotionMonitor(
            endpoint_url=FER_ENDPOINT_URL,
            poll_interval_seconds=FER_POLL_INTERVAL,
            proactive_queue=proactive_emotion_queue,
            confidence_threshold=FER_CONFIDENCE_THRESHOLD,
            target_emotions=FER_TARGET_EMOTIONS
        )
        emotion_monitor.start()

        # Initialize and Start Heart Rate Monitor (no lock needed)
        heart_rate_monitor = HeartRateMonitor(
            poll_interval_seconds=HR_MONITOR_POLL_INTERVAL,
            proactive_queue=proactive_heart_rate_queue,
            get_heart_rate_func=heartbeat_server.get_latest_heart_rate_reading,
            threshold_bpm=HR_THRESHOLD_BPM
        )
        heart_rate_monitor.start()

    except Exception as e:
        logging.critical(f"[Initialization Error] Failed to initialize modules: {e}", exc_info=True)
        heartbeat_server.stop_server() # Attempt cleanup
        sys.exit(1)

    rag_status_active = rag_retriever is not None
    logging.info(f"--- Assistant Ready for {user_profile.get_user_name()} (RAG Active: {rag_status_active}) ---")
    print("Press Ctrl+C to exit.")
    time.sleep(1)

    # --- Main Loop ---
    try:
        while True:
            proactive_message = None

            # --- Check Proactive Queues ---
            # Check emotion queue first
            if not proactive_emotion_queue.empty():
                try:
                    detected_emotion = proactive_emotion_queue.get_nowait()
                    logging.info(f"[Main Loop] Found proactive emotion trigger: {detected_emotion}")
                    if detected_emotion == "sad": proactive_message = f"{user_profile.get_user_name()}, I noticed you might be looking a little sad. Is everything alright?"
                    elif detected_emotion == "happy": proactive_message = f"{user_profile.get_user_name()}, you seem happy right now! What's putting a smile on your face?"
                except queue.Empty: pass # Queue became empty between check and get
                except Exception as e: logging.error(f"Error handling emotion queue: {e}", exc_info=True)

            # Check HR queue only if emotion was empty
            elif not proactive_heart_rate_queue.empty():
                try:
                    hr_alert = proactive_heart_rate_queue.get_nowait()
                    if hr_alert.get("type") == "high_hr":
                        hr_value = hr_alert.get("value")
                        logging.info(f"[Main Loop] Found proactive high heart rate trigger: {hr_value} bpm")
                        proactive_message = f"{user_profile.get_user_name()}, I noticed your heart rate seems a bit high, around {hr_value} beats per minute. Are you feeling okay? Maybe take a few deep breaths?"
                except queue.Empty: pass
                except Exception as e: logging.error(f"Error handling heart rate queue: {e}", exc_info=True)

            # --- Attempt to Speak Proactive Message (Non-Blocking) ---
            if proactive_message:
                if tts_lock.acquire(blocking=False): # Try to get lock without waiting
                    logging.info(f"[Assistant Proactive - Speaking]: {proactive_message}")
                    convo_manager.add_system_utterance(proactive_message) # Add to history
                    try:
                        # Speak the proactive message
                        tts_engine.speak(proactive_message)
                    except Exception as e_tts:
                        logging.error(f"[TTS Error - Proactive]: {e_tts}", exc_info=True)
                    finally:
                        tts_lock.release() # Release lock immediately after speaking attempt
                    interaction_logger.log_interaction("[System Proactive]", proactive_message) # Log it
                else:
                    # TTS is busy (likely with main response or reminder), skip proactive message this cycle
                    logging.info(f"[Assistant Proactive - Skipped] TTS busy, could not speak: {proactive_message}")
                    # Consider putting back on queue or dropping based on desired behavior

            # --- Listening Phase (Always runs) ---
            print(f"\n--- Listening ({STT_LISTEN_DURATION:.1f}s)... ---")
            user_text = None
            stt_error = False
            try:
                # Listening itself doesn't need the TTS lock
                user_text = stt_engine.listen_once(duration=STT_LISTEN_DURATION)
                if not user_text or user_text.startswith("[Error"):
                    logging.warning(f"STT Result: {user_text or '[No clear speech detected]'}")
                    stt_error = True
            except Exception as e_stt:
                 logging.error(f"[STT Error] Error during listening: {e_stt}", exc_info=True)
                 stt_error = True

            # --- Processing Phase (Only if STT was successful) ---
            if stt_error:
                continue # Go back to start of loop (check proactive, listen again)

            # If we have valid user text, process it
            logging.info(f"[User]: {user_text}")

            # --- Profile Update ---
            analysis = {}
            try:
                analysis = nlp_processor.parse_user_input(user_text)
                user_profile.record_mood(analysis["sentiment_score"], analysis["sentiment_label"])
                for like in analysis.get("likes", []): user_profile.add_like(like)
                for dislike in analysis.get("dislikes", []): user_profile.add_dislike(dislike)
                if len(user_text) >= MIN_UTTERANCE_LEN_FOR_EXTRACTION:
                    extracted_profile_data = nlp_processor.extract_profile_updates_with_llm(user_text, llm_engine)
                    if extracted_profile_data:
                        logging.info(f"[Main Loop] Updating profile with LLM extracted data: {extracted_profile_data}")
                        activities_list = extracted_profile_data.get("activities", []); events_list = extracted_profile_data.get("events", [])
                        if isinstance(activities_list, list):
                            for activity in activities_list:
                                if isinstance(activity, dict) and activity.get("name"): user_profile.add_activity(activity.get("name"), activity.get("sentiment","neutral"), activity.get("details",""))
                                else: logging.warning(f"Skipping malformed activity item: {activity}")
                        if isinstance(events_list, list):
                             for event in events_list:
                                 if isinstance(event, dict) and event.get("description"): user_profile.add_event(event.get("description"), event.get("sentiment","neutral"))
                                 else: logging.warning(f"Skipping malformed event item: {event}")
                        for like in extracted_profile_data.get("new_likes", []): user_profile.add_like(like)
                        for dislike in extracted_profile_data.get("new_dislikes", []): user_profile.add_dislike(dislike)
                        if extracted_profile_data.get("summary_note"): user_profile.add_llm_note(extracted_profile_data["summary_note"])
            except Exception as e:
                 logging.error(f"[NLP/Profile Update Error]", exc_info=True)
                 analysis = analysis or {"intent": "unknown", "sentiment_label": "unknown"}

            convo_manager.add_user_utterance(user_text)
            conversation_context = convo_manager.get_history_as_text()

            # --- RAG Retrieval ---
            rag_context_str = None; analysis_intent = analysis.get("intent", "unknown"); user_query_lower = user_text.lower()
            trigger_keywords = ["pain", "doctor", "medicine", "medication", "health", "arthritis", "appointment", "condition", "symptom", "side effect", "what did", "tell me about", "how to"]
            should_trigger_rag = analysis_intent in ["question", "request_suggestion", "unknown"] or any(keyword in user_query_lower for keyword in trigger_keywords)
            perform_rag = rag_retriever is not None and should_trigger_rag
            if perform_rag:
                try:
                    logging.info(f"[RAG] Retrieving documents for query: {user_text}")
                    start_rag_time = time.time(); retrieved_docs = rag_retriever.get_relevant_documents(user_text); end_rag_time = time.time()
                    if retrieved_docs:
                        rag_context_str = format_docs(retrieved_docs)
                        if rag_context_str: logging.info(f"[RAG] Retrieved/formatted {len(retrieved_docs)} docs in {end_rag_time - start_rag_time:.2f}s.")
                        else: logging.warning("[RAG] Formatting retrieved documents failed."); rag_context_str = "[Error formatting retrieved information.]"
                        # print(f"\n--- DEBUG: Retrieved RAG Context ---\n{rag_context_str}\n------------------------------------\n") # Keep commented unless debugging
                    else: logging.info("[RAG] No relevant documents found.")
                except Exception as e: logging.error("[RAG Error]", exc_info=True); rag_context_str = "[Error retrieving information.]"
            else: logging.info(f"[RAG] Skipping retrieval for intent '{analysis_intent}'.")

            # --- Intent Handling & Response Generation ---
            intent = analysis_intent; logging.info(f"[Analysis]: Intent={intent}, Sentiment={analysis.get('sentiment_label','N/A')}")
            response_text = ""; skip_llm = False
            # Handle simple intents first
            if intent == "set_reminder":
                 reminder_text = analysis.get("entities", {}).get("reminder_text", "")
                 try:
                    offset_minutes, final_reminder_msg = reminder_manager.parse_reminder_offset_and_message(reminder_text);
                    if not final_reminder_msg.strip(): final_reminder_msg = "Reminder set"
                    reminder_manager.schedule_reminder(final_reminder_msg, offset_minutes)
                    response_text = f"Okay {user_profile.get_user_name()}, reminder set for '{final_reminder_msg}' in about {offset_minutes:.1f} minutes."; skip_llm = True
                 except Exception as e: logging.error(f"[Reminder Error] {e}"); response_text = "Sorry, I had trouble setting that reminder."; skip_llm = True
            elif intent == "ack_reminder":
                 acknowledged = reminder_manager.acknowledge_reminder(user_text)
                 response_text = f"Okay, {user_profile.get_user_name()}." if not acknowledged else f"Great, thanks for letting me know, {user_profile.get_user_name()}!"; skip_llm = True

            # Generate LLM response if not handled by simple intent
            if not skip_llm:
                try:
                    profile_summary = user_profile.get_profile_summary()
                    # Build prompt using potentially updated context
                    prompt = nlp_processor.build_llm_prompt(
                        intent, analysis, conversation_context, profile_summary, rag_context_str
                    )
                    rag_included_in_prompt = rag_context_str is not None and not rag_context_str.startswith("[Error")
                    logging.info(f"[LLM Prompt Build] Intent: {intent}, RAG context included: {rag_included_in_prompt}")
                    # print(f"\n--- DEBUG: Final LLM Prompt ---\n{prompt}\n-------------------------------\n") # Keep commented unless debugging

                    # Call LLM with adjusted parameters for conciseness
                    llm_response = llm_engine.generate_response(prompt, max_tokens=150, temperature=0.5)
                    logging.info(f"[LLM Raw Response]: {llm_response}")
                    response_text = llm_response.strip()
                    # Handle empty LLM response
                    if not response_text: response_text = "I'm not quite sure how to respond."
                except Exception as e:
                    logging.error(f"[LLM Error]", exc_info=True)
                    response_text = "Sorry, I had an issue generating a response."

            # --- Respond & Log ---
            logging.info(f"[Assistant]: {response_text}")
            convo_manager.add_system_utterance(response_text) # Add assistant response to history

            # --- Speak Main Response (Blocking Lock Acquisition) ---
            if tts_lock.acquire(): # Wait here if TTS is busy (e.g., from proactive)
                try:
                    # Speak the main response generated above
                    tts_engine.speak(response_text)
                except Exception as e:
                    logging.error(f"[TTS Error - Main Response]", exc_info=True)
                finally:
                     tts_lock.release() # Release lock AFTER speaking main response
            else:
                 # This should theoretically not happen with blocking acquire, but log if it does
                 logging.error("[TTS Lock Error] Failed to acquire lock for main response.")

            interaction_logger.log_interaction(user_text, response_text) # Log interaction turn
            time.sleep(0.5) # Small pause before next loop iteration

    except KeyboardInterrupt:
        logging.info("\nCtrl+C received. Exiting gracefully...")
    except Exception as e:
        # Catch any unexpected critical errors in the main loop
        logging.critical(f"\n[Critical Error] An unexpected error occurred in main loop: {e}", exc_info=True)
    finally:
        # --- Cleanup ---
        logging.info("Stopping background tasks and saving profile...")
        if emotion_monitor is not None: emotion_monitor.stop(); logging.info("Emotion monitor stopped.")
        if heart_rate_monitor is not None: heart_rate_monitor.stop(); logging.info("Heart rate monitor stopped.")
        if reminder_manager is not None: reminder_manager.stop(); logging.info("Reminder manager stopped.")
        heartbeat_server.stop_server() # Stop the Flask server thread
        if user_profile is not None: user_profile.save_profile(); logging.info(f"User profile saved to {user_profile.profile_path}")
        else: logging.warning("User profile object not found, could not save.")

        logging.info("Goodbye!")
        print("\nAssistant shut down.")

if __name__ == "__main__":
    main()
