# SMAAI — Smart Meeting AI

An AI-powered meeting assistant built as a 4th-year B.Tech IT final-year project.
Upload (or record) a meeting, get an accurate transcript, then search it, summarise it,
extract action items, ask questions about it, and ask questions about your own documents.

```
Audio → Speech-to-Text → Transcript → MongoDB → Search
                                         ├──→ AI Summary
                                         ├──→ Action Items
                                         ├──→ Meeting Q&A
                                         └──→ Read aloud (Text-to-Speech)
Documents (PDF/DOCX/TXT) → Chunks → MongoDB → RAG Question Answering
```

---

## 1. Objectives

1. Turn meeting recordings into searchable text automatically.
2. Save the time needed to re-listen to recordings: summaries, action items and Q&A.
3. Keep every AI answer **grounded** in the real transcript or document (no invented content).
4. Keep the code small and simple enough to be fully explained in a viva.

## 2. Features

| # | Feature | How it works |
|---|---------|--------------|
| 1 | Audio upload + microphone recording | WAV / MP3 / M4A upload, or record with the browser microphone |
| 2 | Speech-to-text | Faster-Whisper runs on your own computer (offline after the first model download) |
| 3 | Meeting history | Title, date, duration, language, transcript and audio file name stored in MongoDB |
| 4 | Searchable transcripts | Case-insensitive search in titles and transcripts, with a highlighted snippet and match count |
| 5 | AI summarization | LLM writes Overview / Key Points / Decisions; long meetings are summarised in parts then merged |
| 6 | Action items | LLM extracts task, owner, deadline and the exact quote as evidence; owner/deadline are shown as "Not mentioned" when not said |
| 7 | Meeting Q&A | Answers use only the selected meeting's transcript |
| 8 | Documents + RAG | Upload PDF/DOCX/TXT, ask questions; only the most relevant chunks are sent to the LLM |
| 9 | Text-to-speech | Read a summary or transcript aloud |
| 10 | Meeting management | View, search, select, download and delete meetings (and documents) |

Summaries and action items are **cached in MongoDB**, so opening them again costs no extra API calls.

## 3. System Architecture and Workflow

```
┌───────────────────────── app.py (Streamlit UI) ─────────────────────────┐
│ Dashboard │ Upload │ History │ Search │ Summaries │ Actions │ Ask │ Docs │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    ▼
                          meeting_manager.py   ← all the logic ("brain")
        ┌──────────────┬────────────┼──────────────┬──────────────┐
        ▼              ▼            ▼              ▼              ▼
 transcription.py  database.py  llm_service.py  document_rag.py  text_to_speech.py
 (Faster-Whisper)  (MongoDB)    (LLM API)       (chunk+TF-IDF)   (gTTS)
```

**Workflow when a meeting is processed**

1. User enters a title and uploads/records audio.
2. `meeting_manager.process_meeting()` validates the input and saves the audio in `uploads/`.
3. `transcription.py` converts the audio to text with Faster-Whisper.
4. `database.py` stores the meeting (transcript, duration, language…) in MongoDB.
5. The meeting is now available in History and Search, and can be summarised, questioned, or read aloud.

**Workflow for RAG (document question answering)**

1. **Extract & chunk** – text is read from the PDF/DOCX/TXT and split into chunks of ~200 words (40-word overlap) and stored in MongoDB.
2. **Retrieve** – for a question, TF-IDF + cosine similarity ranks all chunks of the selected documents; the best 4 are kept.
3. **Generate** – only those 4 chunks and the question go to the LLM, which is told to answer *only* from them, or say it could not find the answer.

## 4. Project Structure

```text
AI-Meeting-Assistant/
├── app.py               # Streamlit user interface (all pages)
├── meeting_manager.py   # Business logic connecting everything
├── transcription.py     # Speech-to-text (Faster-Whisper)
├── database.py          # MongoDB operations
├── llm_service.py       # LLM calls: summary, action items, Q&A
├── document_rag.py      # Document reading, chunking, retrieval (RAG)
├── text_to_speech.py    # Text-to-speech (gTTS)
├── requirements.txt     # Python dependencies
├── .env.example         # Template for settings/API keys (copy to .env)
├── .gitignore           # Keeps .env, audio files and venv out of Git
├── README.md            # This file
└── uploads/             # Saved audio files (created automatically)
```

### What each file does

- **`app.py`** – Draws the 8 pages (Dashboard, Upload & Transcribe, Meeting History, Search, Summaries, Action Items, Ask Questions, Documents). It contains no AI or database code; every button calls a function in `meeting_manager.py`. It also checks the MongoDB connection at startup and shows a clear message if MongoDB is not running.
- **`meeting_manager.py`** – The "brain". Contains `process_meeting` (validate → save audio → transcribe → store), search with snippets, cached summary/action-item functions, meeting Q&A, document upload and RAG Q&A, delete, and dashboard numbers.
- **`transcription.py`** – Loads the Whisper model once and exposes `transcribe_audio(path)` → `(text, duration, language)`.
- **`database.py`** – The only file that talks to MongoDB. Functions for meetings (add / list / get / search / update / delete / count) and documents (add / list / delete / count).
- **`llm_service.py`** – `summarize_transcript`, `extract_action_items`, `answer_question`. Uses the `openai` client library against any OpenAI-compatible provider (Groq by default). Long text is split into ~10,000-character parts to respect rate limits. Prompts forbid inventing information.
- **`document_rag.py`** – `extract_text` (PDF/DOCX/TXT), `split_into_chunks`, `find_relevant_chunks` (TF-IDF retrieval).
- **`text_to_speech.py`** – `text_to_mp3(text, language)` using gTTS; strips markdown symbols and limits length to 5,000 characters.
- **`requirements.txt`** – External packages needed.
- **`.env.example`** – Names of all settings with placeholder values.

## 5. Technology Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.10+** | Programming language |
| **Streamlit** | Builds the web UI in pure Python; provides file upload, microphone recording (`st.audio_input`), audio playback |
| **Faster-Whisper** | Speech-to-text. A fast version of OpenAI's Whisper that runs locally on CPU (decodes MP3/M4A itself, no separate FFmpeg install needed) |
| **MongoDB + PyMongo** | Stores meetings and documents as flexible JSON-like documents. Fits well because a meeting holds a transcript, a summary and a list of action items together |
| **openai (client library)** | Talks to the LLM. Many providers (Groq, OpenAI, Google Gemini, local Ollama) expose the same API, so changing provider only needs `.env` changes, not code changes |
| **python-dotenv** | Reads settings and API keys from `.env` so no secrets are in the code |
| **scikit-learn** | TF-IDF vectoriser and cosine similarity for RAG retrieval |
| **pypdf** | Reads text from PDF files |
| **python-docx** | Reads text from Word (.docx) files |
| **gTTS** | Converts text to spoken MP3 audio (needs internet) |

## 6. Setup (Windows)

### Step 1 – Install prerequisites
- **Python 3.10 – 3.12** from python.org (tick *"Add Python to PATH"* during installation).
- **MongoDB Community Server** from mongodb.com/try/download/community. Use the installer's defaults and keep **"Install MongoDB as a Service"** ticked so it starts automatically. (Optional: install *MongoDB Compass*, a viewer that lets you see the saved data — great for demos.)
  *Alternative:* a free MongoDB Atlas cluster; paste its connection string into `MONGODB_URI`.

### Step 2 – Get an LLM API key
The default is **Groq** (free tier, no credit card): create an account at console.groq.com → *API Keys* → create a key.

| Provider | `LLM_BASE_URL` | `LLM_MODEL` (check the provider's docs for current names) |
|----------|---------------|--------------|
| Groq (default) | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| OpenAI | `https://api.openai.com/v1` | e.g. `gpt-4o-mini` |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | a Gemini model name |
| Ollama (local, no key) | `http://localhost:11434/v1` | a model you pulled, e.g. `llama3.2` (set `LLM_API_KEY` to any text) |

### Step 3 – Install and run
Open **Command Prompt** in the project folder:

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
notepad .env
streamlit run app.py
```

In `.env`, replace `your_api_key_here` with your real key and save. The app opens at `http://localhost:8501`.

> The first transcription downloads the Whisper model once (~150 MB for `base`), so internet is needed that first time.

### `.env` settings

| Variable | Meaning | Default |
|----------|---------|---------|
| `MONGODB_URI` | MongoDB address | `mongodb://localhost:27017` |
| `MONGODB_DB` | Database name | `smaai` |
| `LLM_API_KEY` | Your LLM provider key (**required** for AI features) | – |
| `LLM_BASE_URL` | LLM provider address | Groq |
| `LLM_MODEL` | Model name | `llama-3.3-70b-versatile` |
| `WHISPER_MODEL` | `tiny` / `base` / `small` / `medium` | `base` |

Never share or commit `.env` (it is already in `.gitignore`).

## 7. How to Use Each Feature

1. **Dashboard** – Totals (meetings, audio minutes, summarised meetings, documents), the 5 latest meetings, and a *System status* box showing whether MongoDB and the API key are OK.
2. **Upload & Transcribe** – Enter a title, then either *Upload a file* (WAV/MP3/M4A) or *Record with microphone*, and click Transcribe. The transcript appears and is saved automatically.
3. **Meeting History** – Pick a meeting to play its audio, read/download the transcript, listen to it with **Read aloud**, or **delete** it (tick the confirmation box first; this also removes the audio file).
4. **Search** – Type a word such as `database`. All matching meetings are listed with a snippet and how many times the word appears.
5. **Summaries** – Pick a meeting → **Generate summary**. Download it or **Read aloud**. *Regenerate* makes a new one.
6. **Action Items** – Pick a meeting → **Extract action items**. Shows a table with Task, Owner, Deadline and the quote from the transcript as evidence. Download as CSV.
7. **Ask Questions** –
   - *Ask about a meeting*: pick a meeting, type a question. Expand "Text used to answer" to see the evidence.
   - *Ask about documents (RAG)*: choose which uploaded documents to search and ask.
8. **Documents** – Upload PDF/DOCX/TXT files for RAG; list or delete them.

## 8. Database Structure and Data Flow

Database `smaai` (created automatically by MongoDB on first insert) has two collections.

**`meetings`**

| Field | Type | Description |
|-------|------|-------------|
| `_id` | ObjectId | Unique id (generated by MongoDB) |
| `title` | string | Meeting title |
| `created_at` | datetime | When it was processed |
| `transcript` | string | Full speech-to-text transcript |
| `audio_filename` | string | File saved inside `uploads/` |
| `duration_seconds` | number | Audio length |
| `language` | string | Language detected by Whisper (e.g. `en`) |
| `summary` | string / null | Cached AI summary (`null` until generated) |
| `action_items` | list / null | Cached list of `{task, owner, deadline, quote}` (`null` until generated; `[]` means none found) |

**`documents`**

| Field | Type | Description |
|-------|------|-------------|
| `_id` | ObjectId | Unique id |
| `name` | string | Original file name |
| `uploaded_at` | datetime | Upload time |
| `chunks` | list of strings | The document text split into ~200-word chunks |

**Data flow:** `app.py` → `meeting_manager.py` → (`transcription.py` / `llm_service.py` / `document_rag.py` / `text_to_speech.py`) → `database.py` → MongoDB. Audio files are stored on disk in `uploads/`; only their file names are in the database.

## 9. Troubleshooting

| Problem | Cause / Fix |
|---------|-------------|
| "Cannot connect to MongoDB" | MongoDB service is not running. Open *Services* (Win+R → `services.msc`), start **MongoDB Server**, or check `MONGODB_URI` in `.env`. |
| "LLM_API_KEY is not set" | Create `.env` (copy from `.env.example`) and add your key, then restart Streamlit. |
| "The AI service request failed … 401" | API key is wrong or expired. |
| "… 429" / rate limit | Free-tier limits reached (long meetings use several requests). Wait a minute, or set `LLM_MODEL` to a smaller/faster model offered by your provider. |
| "… model not found / decommissioned" | Providers retire models. Pick a current name from the provider's model list and put it in `LLM_MODEL`. |
| "Transcription failed: No speech was detected" | The audio is silent or too quiet/short. |
| Transcription is slow | Normal on CPU for long recordings. Use `WHISPER_MODEL=tiny` for speed or `small` for accuracy. |
| Audio file cannot be decoded | Re-export the recording as WAV or MP3. Installing FFmpeg is rarely needed, but can help for unusual files. |
| `pip install` fails on a very new Python version | Use Python 3.10–3.12. |
| Read aloud fails | gTTS needs internet access. |
| PDF says "No readable text" | The PDF is a scan (images only). Use a PDF with selectable text or a DOCX/TXT file. |
| Microphone button missing | Update Streamlit: `pip install -U streamlit` (needs 1.40+). Allow microphone access in the browser. |
| Old Phase 1 meetings are missing | SMAAI now uses MongoDB. The old SQLite file (`data/meetings.db`) is not read; re-upload those recordings (the audio files remain in `uploads/`). |

## 10. Implemented Features vs Limitations / Future Scope

**Implemented:** all ten features listed in section 2.

**Known limitations (be ready to discuss these in the viva):**
- **Accuracy of speech-to-text** depends on audio quality and the Whisper model size; no speaker identification (who said what).
- **AI features need internet and an API key** (the LLM and gTTS are online services; Whisper is local).
- **Retrieval is keyword-based (TF-IDF)**, so a question that uses completely different words than the document ("car" vs "automobile") can miss the right passage. Embedding-based retrieval would fix this.
- **Search is a substring match** over all transcripts (fine for hundreds of meetings; MongoDB text/Atlas search would be needed for very large collections).
- **LLM output is not guaranteed perfect** – that is why action items show their quote as evidence and answers show the text they were based on. Long meetings are processed in parts, which can lose some cross-part context.
- Text-to-speech is limited to the first 5,000 characters and is not offline.
- No user login: anyone with access to the app sees all meetings.
- The Q&A chat history lives only in the browser session (answers are not saved in MongoDB).

**Future scope:** speaker diarization, embedding-based RAG with a vector database, user accounts, exporting summaries to PDF/Word, calendar integration, live (real-time) transcription.

## 11. Quick Demo Checklist

1. Start MongoDB, run `streamlit run app.py`, open **Dashboard** → *System status* should be all ✅.
2. **Upload & Transcribe**: a 1–2 minute recording in which someone mentions a task, a person and a day.
3. **Meeting History** → transcript + audio. **Search** → a word from the transcript.
4. **Summaries** → Generate, then Read aloud. **Action Items** → Extract.
5. **Ask Questions** → "When is the deadline?" and also something *not* said, to show it answers "I could not find this".
6. **Documents** → upload a PDF → ask a question about it.
7. Show the stored data in MongoDB Compass (`smaai` → `meetings`).

## 12. Likely Viva Questions

- **Why Faster-Whisper?** Free, runs locally, very accurate, and faster than the original Whisper on CPU.
- **Why MongoDB?** One meeting = one document holding transcript, summary and action items, with no fixed table schema.
- **What is RAG?** Retrieve the most relevant text first, then let the LLM answer only from it, so answers are grounded and prompts stay small.
- **How do you stop the AI inventing things?** Low temperature, prompts that forbid invention, "Not mentioned" for missing owner/deadline, quotes as evidence, and an explicit "I could not find this" answer.
- **How are long meetings handled?** Split into ~10,000-character parts, summarise each, then combine (map-reduce); Q&A on long meetings uses retrieval of the top chunks instead of the whole transcript.
- **Why TF-IDF and not embeddings?** No model download, easy to explain and test; embeddings are listed as future scope.
- **Where are the secrets?** In `.env`, which is excluded from Git; only `.env.example` with placeholders is shared.

## 13. Testing Status (honest summary)

The development sandbox had no internet, so MongoDB, Streamlit, Faster-Whisper, the LLM API and gTTS could not be installed or called there. Everything else was tested, using small stand-ins for those five services:

- **Tested with real files and real libraries:** PDF/DOCX/TXT text extraction, chunking, TF-IDF retrieval (including scanned/empty PDFs and unrelated questions).
- **Tested against stand-ins:** the full flow upload → transcribe → save → history → search → summary → action items → Q&A → documents → TTS → delete, error handling (empty title, no audio, wrong format, silent audio, missing API key, LLM failure, MongoDB down), caching, and every Streamlit page rendering with and without data.
- **Not tested – please verify on your PC:** real MongoDB, real Whisper transcription of your audio, real LLM answers (quality depends on the model), real gTTS speech, and the real Streamlit look and behaviour. Use the checklist in section 11.
