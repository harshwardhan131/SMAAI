"""
llm_service.py
---------------
Everything that needs a large language model (LLM) lives here:
  - summarize_transcript()    -> meeting summary
  - extract_action_items()    -> tasks / owners / deadlines
  - answer_question()         -> answers grounded in the text we give it

We use the `openai` Python package because many providers (Groq, OpenAI,
Google Gemini, a local Ollama server...) offer an OpenAI-compatible API.
Which provider is used is decided only by three environment variables:
LLM_API_KEY, LLM_BASE_URL and LLM_MODEL (see .env.example).
"""

import json
import os

from openai import OpenAI

# Long transcripts are cut into pieces of about this many characters, so each
# request stays small enough for free-tier rate limits.
MAX_CHARS_PER_REQUEST = 10000

SYSTEM_PROMPT = (
    "You are a careful meeting assistant. Use ONLY information that is present in the "
    "text you are given. Never invent names, numbers, decisions, tasks or dates. "
    "The text comes from automatic speech-to-text, so it may contain small errors."
)


class LLMError(Exception):
    """Raised with a user-friendly message when the AI request cannot be completed."""


def _get_client():
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key:
        raise LLMError("LLM_API_KEY is not set. Copy .env.example to .env and add your API key.")
    base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    # max_retries: the library automatically waits and retries when the
    # provider answers "too many requests" (rate limit).
    return OpenAI(api_key=api_key, base_url=base_url, max_retries=5)


def _ask_llm(user_prompt):
    """Send one prompt to the LLM and return its text answer."""
    client = _get_client()
    model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,  # low = more factual, less creative
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as error:
        raise LLMError(f"The AI service request failed: {error}")
    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        raise LLMError("The AI service returned an empty answer. Please try again.")
    return answer.strip()


def _split_text(text, max_chars=MAX_CHARS_PER_REQUEST):
    """Cut long text into pieces of at most max_chars, breaking only between words."""
    pieces, current = [], ""
    for word in text.split():
        if current and len(current) + len(word) + 1 > max_chars:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        pieces.append(current)
    return pieces


# ---------------------------------------------------------------------------
# 1. SUMMARY
# ---------------------------------------------------------------------------

SUMMARY_FORMAT = (
    "Use exactly this format:\n"
    "**Overview:** 2-3 sentences describing what the meeting was about.\n"
    "**Key Points:**\n- bullet points of the main topics discussed\n"
    "**Decisions:**\n- bullet points of decisions that were made, or 'None mentioned'"
)


def summarize_transcript(transcript):
    """Return a concise summary. Long transcripts are summarised in parts first (map-reduce)."""
    pieces = _split_text(transcript)

    if len(pieces) == 1:
        return _ask_llm(f"Summarize this meeting transcript.\n{SUMMARY_FORMAT}\n\nTranscript:\n{pieces[0]}")

    # Step 1 (map): summarise every part separately.
    partial_summaries = []
    for number, piece in enumerate(pieces, start=1):
        partial = _ask_llm(
            f"This is part {number} of {len(pieces)} of a meeting transcript. "
            f"Summarize it in bullet points, keeping names, decisions and tasks.\n\n{piece}"
        )
        partial_summaries.append(partial)

    # Step 2 (reduce): merge the partial summaries into one final summary.
    combined = "\n\n".join(partial_summaries)
    return _ask_llm(
        f"Below are summaries of consecutive parts of ONE meeting. "
        f"Combine them into a single summary.\n{SUMMARY_FORMAT}\n\nPart summaries:\n{combined}"
    )


# ---------------------------------------------------------------------------
# 2. ACTION ITEMS
# ---------------------------------------------------------------------------

ACTION_ITEM_PROMPT = (
    "Extract the action items (tasks that someone must do) from this meeting transcript.\n"
    "Rules:\n"
    "- Only include tasks that are clearly stated or agreed in the transcript.\n"
    "- 'owner' is the person responsible ONLY if a name or role is mentioned, otherwise null.\n"
    "- 'deadline' is the due date/time ONLY if one is mentioned, otherwise null.\n"
    "- Do NOT guess missing owners or deadlines.\n"
    "- 'quote' is a short sentence copied word-for-word from the transcript that proves the task.\n"
    "- If there are no action items, return [].\n"
    "Return ONLY a JSON list, with no extra text, like this:\n"
    '[{"task": "...", "owner": "... or null", "deadline": "... or null", "quote": "..."}]\n\n'
    "Transcript:\n"
)

_EMPTY_WORDS = {"", "null", "none", "n/a", "not mentioned", "unknown", "unspecified"}


def _clean_field(value):
    """Turn 'null' / 'none' / empty values into a real None."""
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in _EMPTY_WORDS else text


def _parse_action_items(answer):
    """Read the JSON list returned by the LLM and clean every item."""
    start, end = answer.find("["), answer.rfind("]")
    if start == -1 or end == -1:
        raise LLMError("The AI did not return a valid list of action items. Please try again.")
    try:
        raw_items = json.loads(answer[start:end + 1])
    except json.JSONDecodeError:
        raise LLMError("The AI response could not be read as JSON. Please try again.")

    items = []
    for raw in raw_items:
        if not isinstance(raw, dict) or not _clean_field(raw.get("task")):
            continue
        items.append({
            "task": _clean_field(raw.get("task")),
            "owner": _clean_field(raw.get("owner")),
            "deadline": _clean_field(raw.get("deadline")),
            "quote": _clean_field(raw.get("quote")),
        })
    return items


def extract_action_items(transcript):
    """Return a list of dicts: task, owner, deadline, quote (owner/deadline may be None)."""
    all_items, seen_tasks = [], set()
    for piece in _split_text(transcript):
        for item in _parse_action_items(_ask_llm(ACTION_ITEM_PROMPT + piece)):
            key = item["task"].lower()
            if key not in seen_tasks:  # avoid duplicates when parts overlap in topic
                seen_tasks.add(key)
                all_items.append(item)
    return all_items


# ---------------------------------------------------------------------------
# 3. QUESTION ANSWERING (used for both meeting Q&A and document RAG)
# ---------------------------------------------------------------------------

def answer_question(question, passages, source_name):
    """
    Answer a question using ONLY the given text passages.
    source_name is a label such as "meeting transcript" or "documents".
    """
    context = "\n\n---\n\n".join(passages)
    return _ask_llm(
        f"Answer the question using ONLY the {source_name} excerpts below.\n"
        f"If the answer is not in the excerpts, reply exactly: "
        f"\"I could not find this in the {source_name}.\"\n"
        f"Keep the answer short and clear.\n\n"
        f"Excerpts:\n{context}\n\nQuestion: {question}"
    )
