# 🧓 Senior Assistance Bot

An always-listening, proactive AI voice assistant designed for elderly users. It monitors health signals, detects emotions, manages reminders, and answers questions using a personal knowledge base — all running locally on-device.

---

## How It Works
```
Microphone (always listening)
        │
        ▼
┌───────────────────┐
│   Whisper STT     │  Speech → Text
└────────┬──────────┘
         │
         ▼
┌───────────────────────────────────────────┐
│              NLP Pipeline                 │
│  Intent detection │ Sentiment │ Entities  │
└────────┬──────────────────────────────────┘
         │
         ├──► RAG Retrieval (ChromaDB + knowledge base)
         │         └── prescriptions, logs, docs
         │
         ▼
┌───────────────────┐
│   Llama 3.2 LLM   │  Context-aware response
│   (via Ollama)    │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│     gTTS TTS      │  Text → Speech (spoken aloud)
└───────────────────┘

Background threads (proactive monitoring):
  ├── Emotion Monitor   → detects sad/happy via FER endpoint
  ├── Heart Rate Monitor → alerts if BPM > threshold
  └── Reminder Manager  → scheduled voice reminders
```

---

## Features

- **Always-listening** — continuously captures audio in 7s windows using Whisper
- **RAG-powered answers** — retrieves from personal knowledge base (prescriptions, interaction logs, docs) via ChromaDB
- **Proactive emotion detection** — polls a facial emotion recognition endpoint; speaks if user looks sad or happy
- **Heart rate monitoring** — receives BPM data from a heartbeat server; alerts if too high
- **Voice reminders** — set reminders by voice; spoken back at the right time
- **User profile** — builds and persists a profile (likes, dislikes, mood history, activities, events) across sessions
- **Thread-safe TTS** — lock-based system ensures reminders, proactive alerts, and main responses never overlap
- **LLM profile extraction** — automatically extracts profile updates from natural conversation

---

## Project Structure
```
senior_assistance_bot/
├── main.py                     # Entry point — main loop, threading, orchestration
├── stt.py                      # Speech-to-text (Whisper)
├── tts.py                      # Text-to-speech (gTTS)
├── llm_inference.py            # Ollama LLM wrapper
├── nlp_pipeline.py             # Intent detection, sentiment, entity extraction
├── conversation.py             # Conversation history manager
├── rag_setup.py                # ChromaDB vector store setup and retrieval
├── query_db.py                 # Direct vector DB query utility
├── user_profile.py             # User profile load/save/update
├── reminder_manager.py         # Scheduled voice reminder system
├── emotion_monitor.py          # Background FER emotion polling thread
├── heart_rate_monitor.py       # Background heart rate monitoring thread
├── heartbeat_server.py         # Flask server receiving BPM from wearable
├── logger_module.py            # Interaction logger
├── soundtest.py                # Audio device testing utility
├── elderly_user_profile.json   # Persisted user profile (generated at runtime)
├── requirements.txt
└── knowledge_base/
    └── hospital_priscription.docx   # RAG source document (add your own docs here)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10 |
| LLM | Ollama — Llama 3.2 (local) |
| STT | OpenAI Whisper |
| TTS | gTTS |
| Vector DB | ChromaDB |
| RAG | LangChain |
| Emotion Detection | External FER endpoint (HTTP) |
| Heart Rate | Custom Flask heartbeat server |

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Whisper model
Whisper downloads automatically on first run. Default model: `base.en`

### 3. Download and run Llama 3.2 via Ollama
```bash
ollama pull llama3.2:latest
ollama serve
```

### 4. Add your knowledge base documents
Place `.txt`, `.pdf`, or `.docx` files into the `knowledge_base/` folder.  
These are indexed into ChromaDB on first run.

### 5. Run
```bash
python main.py

# Force rebuild vector DB if you add new docs:
python main.py --force-rag-recreate
```

---

## Configuration

Key settings at the top of `main.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `WHISPER_MODEL_NAME` | `base.en` | Whisper model size |
| `OLLAMA_MODEL_NAME` | `llama3.2:latest` | LLM model |
| `STT_LISTEN_DURATION` | `7.0s` | Listening window per loop |
| `FER_ENDPOINT_URL` | `http://192.168.0.243:5000/emotion` | Facial emotion recognition server |
| `HR_THRESHOLD_BPM` | `120` | Heart rate alert threshold |
| `HEARTBEAT_SERVER_PORT` | `5002` | Port for incoming BPM data |

---

## Status

✅ Voice loop (STT → LLM → TTS) — working  
✅ RAG with ChromaDB — working  
✅ Proactive emotion monitoring — working  
✅ Heart rate alerts — working  
✅ Reminder system — working  
🔄 Facial recognition integration — in progress  

---

## Author

**Mohammed Ayan Syed**

---

*Vosk STT support was explored in earlier versions but the current build uses Whisper.*
