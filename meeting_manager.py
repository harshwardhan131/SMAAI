"""
meeting_manager.py
-------------------
The "brain" that connects everything. app.py (the UI) calls the functions here,
and these functions in turn use database.py, transcription.py, llm_service.py
and document_rag.py. All the "what happens when I click this button" logic is
in this one readable file.
"""

import os
import re
from datetime import datetime

import database
import document_rag
import llm_service
import transcription

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a"}

# Transcripts shorter than this are sent to the LLM completely; longer ones use
# retrieval (only the most relevant chunks are sent).
FULL_TRANSCRIPT_LIMIT = 12000


# ---------------------------------------------------------------------------
# UPLOAD + TRANSCRIBE
# ---------------------------------------------------------------------------

def process_meeting(file_name, file_bytes, title):
    """
    Full pipeline for a new meeting:
    validate -> save audio -> speech-to-text -> save in MongoDB.
    Returns (meeting_id, transcript). Raises ValueError with a friendly message on problems.
    """
    if not title or not title.strip():
        raise ValueError("Meeting title cannot be empty.")
    if not file_bytes:
        raise ValueError("No audio was provided. Upload a file or record from the microphone.")

    extension = os.path.splitext(file_name)[1].lower()
    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        raise ValueError("Unsupported audio format. Please use a .wav, .mp3 or .m4a file.")

    # Save the audio file with a timestamp so two uploads never overwrite each other.
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    saved_name = f"{datetime.now():%Y%m%d_%H%M%S}_{os.path.basename(file_name)}"
    saved_path = os.path.join(UPLOAD_DIR, saved_name)
    with open(saved_path, "wb") as audio_file:
        audio_file.write(file_bytes)

    try:
        transcript, duration, language = transcription.transcribe_audio(saved_path)
    except Exception as error:
        os.remove(saved_path)  # do not keep audio that failed to transcribe
        raise ValueError(f"Transcription failed: {error}")

    meeting_id = database.add_meeting(title.strip(), transcript, saved_name, duration, language)
    return meeting_id, transcript


# ---------------------------------------------------------------------------
# MEETING MANAGEMENT: list / view / search / delete
# ---------------------------------------------------------------------------

def list_meetings():
    return database.get_all_meetings()


def get_meeting(meeting_id):
    return database.get_meeting(meeting_id)


def audio_path(meeting):
    """Full path of the saved audio file, or None if the file is missing."""
    if not meeting.get("audio_filename"):
        return None
    path = os.path.join(UPLOAD_DIR, meeting["audio_filename"])
    return path if os.path.exists(path) else None


def delete_meeting(meeting_id):
    """Delete the meeting from MongoDB and its audio file from disk."""
    meeting = database.get_meeting(meeting_id)
    if meeting is None:
        return False
    path = audio_path(meeting)
    if path:
        os.remove(path)
    return database.delete_meeting(meeting_id)


def _make_snippet(text, query, radius=80):
    """Return the text around the first match of `query`, with the match in **bold**."""
    match = re.search(re.escape(query), text, flags=re.IGNORECASE)
    if match is None:
        return None
    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    before = text[start:match.start()]
    found = text[match.start():match.end()]
    after = text[match.end():end]
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{before}**{found}**{after}{suffix}"


def search(query):
    """
    Search titles and transcripts. Each result gets:
      snippet     - a short piece of the transcript around the first match
      match_count - how many times the query appears in the transcript
    """
    if not query or not query.strip():
        return []
    query = query.strip()
    results = database.search_meetings(query)
    for meeting in results:
        meeting["snippet"] = _make_snippet(meeting["transcript"], query)
        meeting["match_count"] = len(re.findall(re.escape(query), meeting["transcript"], flags=re.IGNORECASE))
    return results


# ---------------------------------------------------------------------------
# AI FEATURES (results are cached in MongoDB so we do not pay for the same AI call twice)
# ---------------------------------------------------------------------------

def get_summary(meeting_id, regenerate=False):
    meeting = database.get_meeting(meeting_id)
    if meeting is None:
        raise ValueError("Meeting not found.")
    if meeting["summary"] and not regenerate:
        return meeting["summary"]
    summary = llm_service.summarize_transcript(meeting["transcript"])
    database.update_meeting(meeting_id, {"summary": summary})
    return summary


def get_action_items(meeting_id, regenerate=False):
    meeting = database.get_meeting(meeting_id)
    if meeting is None:
        raise ValueError("Meeting not found.")
    if meeting["action_items"] is not None and not regenerate:
        return meeting["action_items"]
    items = llm_service.extract_action_items(meeting["transcript"])
    database.update_meeting(meeting_id, {"action_items": items})
    return items


def ask_about_meeting(meeting_id, question):
    """
    Answer a question using only this meeting's transcript.
    Returns (answer, passages_used) so the UI can show the evidence.
    """
    if not question or not question.strip():
        raise ValueError("Please type a question first.")
    meeting = database.get_meeting(meeting_id)
    if meeting is None:
        raise ValueError("Meeting not found.")

    transcript = meeting["transcript"]
    if len(transcript) <= FULL_TRANSCRIPT_LIMIT:
        passages = [transcript]  # short meeting: the LLM can read all of it
    else:
        chunks = document_rag.split_into_chunks(transcript)
        passages = document_rag.find_relevant_chunks(question, chunks, top_k=6)
        if not passages:
            return "I could not find this in the meeting transcript.", []

    answer = llm_service.answer_question(question.strip(), passages, "meeting transcript")
    return answer, passages


# ---------------------------------------------------------------------------
# DOCUMENTS + RAG
# ---------------------------------------------------------------------------

def add_document(file_name, file_bytes):
    """Extract text, split it into chunks and store it. Returns (document_id, number_of_chunks)."""
    text = document_rag.extract_text(file_name, file_bytes)
    chunks = document_rag.split_into_chunks(text)
    document_id = database.add_document(file_name, chunks)
    return document_id, len(chunks)


def list_documents():
    return database.get_all_documents()


def delete_document(document_id):
    return database.delete_document(document_id)


def ask_about_documents(question, document_ids):
    """
    RAG: find the most relevant chunks in the chosen documents, then let the LLM
    answer from them. Returns (answer, sources) where sources = [(document_name, chunk_text), ...].
    """
    if not question or not question.strip():
        raise ValueError("Please type a question first.")
    if not document_ids:
        raise ValueError("Please select at least one document.")

    # Collect the chunks of the selected documents, remembering which file each came from.
    chunk_texts, chunk_sources = [], []
    for document in database.get_all_documents():
        if document["id"] in document_ids:
            for chunk in document["chunks"]:
                chunk_texts.append(chunk)
                chunk_sources.append(document["name"])

    best_chunks = document_rag.find_relevant_chunks(question, chunk_texts, top_k=4)
    if not best_chunks:
        return "I could not find this in the documents.", []

    answer = llm_service.answer_question(question.strip(), best_chunks, "documents")
    sources = [(chunk_sources[chunk_texts.index(chunk)], chunk) for chunk in best_chunks]
    return answer, sources


# ---------------------------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------------------------

def dashboard_stats():
    """Numbers for the dashboard cards plus the 5 most recent meetings."""
    meetings = database.get_all_meetings()
    total_seconds = sum(m.get("duration_seconds") or 0 for m in meetings)
    summarised = sum(1 for m in meetings if m.get("summary"))
    return {
        "total_meetings": len(meetings),
        "total_documents": database.count_documents(),
        "total_minutes": total_seconds / 60,
        "summarised": summarised,
        "recent": meetings[:5],
    }
