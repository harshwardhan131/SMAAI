"""
database.py
------------
All MongoDB operations for SMAAI live in this one file.

Two collections are used:

  meetings   - one document per processed meeting (title, date, transcript,
               audio info, and the cached AI summary / action items)
  documents  - one document per uploaded reference document, with its text
               already split into small chunks (used for RAG question answering)

Every other file talks to the database only through the functions below.
Connection settings come from environment variables (see .env.example).
"""

import os
import re
from datetime import datetime

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient

_client = None  # created once, then reused (MongoClient is thread-safe)


def get_db():
    """Return the MongoDB database object, creating the connection on first use."""
    global _client
    if _client is None:
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        # Fail after 3 seconds instead of hanging if MongoDB is not running.
        _client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    return _client[os.getenv("MONGODB_DB", "smaai")]


def check_connection():
    """Ping the server. Raises an exception if MongoDB cannot be reached."""
    get_db().command("ping")


def _to_object_id(text_id):
    """Convert the string id used in the UI into a MongoDB ObjectId (or None if invalid)."""
    try:
        return ObjectId(text_id)
    except (InvalidId, TypeError):
        return None


def _clean(doc):
    """Turn MongoDB's ObjectId '_id' into a plain string 'id' so the UI can use it easily."""
    if doc is None:
        return None
    doc["id"] = str(doc.pop("_id"))
    return doc


# ---------------------------------------------------------------------------
# MEETINGS
# ---------------------------------------------------------------------------

def add_meeting(title, transcript, audio_filename, duration_seconds, language):
    """Insert a new meeting and return its id (as a string)."""
    meeting = {
        "title": title,
        "created_at": datetime.now(),
        "transcript": transcript,
        "audio_filename": audio_filename,
        "duration_seconds": duration_seconds,
        "language": language,
        "summary": None,        # filled later by the Summaries page
        "action_items": None,   # filled later by the Action Items page
    }
    result = get_db().meetings.insert_one(meeting)
    return str(result.inserted_id)


def get_all_meetings():
    """Return all meetings, newest first. The (large) transcript is left out for speed."""
    cursor = get_db().meetings.find({}, {"transcript": 0}).sort("created_at", -1)
    return [_clean(doc) for doc in cursor]


def get_meeting(meeting_id):
    """Return one full meeting (including transcript), or None if not found."""
    object_id = _to_object_id(meeting_id)
    if object_id is None:
        return None
    return _clean(get_db().meetings.find_one({"_id": object_id}))


def search_meetings(query):
    """
    Find meetings whose title OR transcript contains the query text
    (case-insensitive substring match). Returns full documents, newest first.
    """
    pattern = {"$regex": re.escape(query), "$options": "i"}
    cursor = get_db().meetings.find(
        {"$or": [{"title": pattern}, {"transcript": pattern}]}
    ).sort("created_at", -1)
    return [_clean(doc) for doc in cursor]


def update_meeting(meeting_id, fields):
    """Update some fields of a meeting, e.g. update_meeting(id, {"summary": "..."})."""
    object_id = _to_object_id(meeting_id)
    if object_id is None:
        raise ValueError("Invalid meeting id.")
    get_db().meetings.update_one({"_id": object_id}, {"$set": fields})


def delete_meeting(meeting_id):
    """Delete a meeting. Returns True if something was deleted."""
    object_id = _to_object_id(meeting_id)
    if object_id is None:
        return False
    result = get_db().meetings.delete_one({"_id": object_id})
    return result.deleted_count > 0


def count_meetings():
    return get_db().meetings.count_documents({})


# ---------------------------------------------------------------------------
# DOCUMENTS (for RAG)
# ---------------------------------------------------------------------------

def add_document(name, chunks):
    """Store an uploaded document (already split into text chunks). Returns its id."""
    document = {
        "name": name,
        "uploaded_at": datetime.now(),
        "chunks": chunks,  # list of plain strings
    }
    result = get_db().documents.insert_one(document)
    return str(result.inserted_id)


def get_all_documents():
    """Return all documents, newest first (including chunks, needed for retrieval)."""
    cursor = get_db().documents.find({}).sort("uploaded_at", -1)
    return [_clean(doc) for doc in cursor]


def delete_document(document_id):
    """Delete an uploaded document. Returns True if something was deleted."""
    object_id = _to_object_id(document_id)
    if object_id is None:
        return False
    result = get_db().documents.delete_one({"_id": object_id})
    return result.deleted_count > 0


def count_documents():
    return get_db().documents.count_documents({})
