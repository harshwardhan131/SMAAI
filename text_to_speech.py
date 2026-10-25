"""
text_to_speech.py
------------------
Turns text into spoken audio (MP3 bytes) using gTTS (Google Text-to-Speech).
gTTS needs an internet connection.
"""

import io
import re

from gtts import gTTS

MAX_CHARACTERS = 5000  # keep generation quick; longer text is cut with a note in the UI


def _remove_markdown(text):
    """Remove symbols such as ** and # so they are not read aloud."""
    text = re.sub(r"[*#_`>]", "", text)
    return re.sub(r"^\s*-\s+", "", text, flags=re.MULTILINE)


def text_to_mp3(text, language="en"):
    """
    Return MP3 audio bytes that read `text` aloud.
    Falls back to English if gTTS does not support the requested language.
    """
    clean_text = _remove_markdown(text).strip()[:MAX_CHARACTERS]
    if not clean_text:
        raise ValueError("There is no text to read aloud.")

    try:
        speech = gTTS(text=clean_text, lang=language)
    except ValueError:  # unsupported language code
        speech = gTTS(text=clean_text, lang="en")

    buffer = io.BytesIO()
    speech.write_to_fp(buffer)
    return buffer.getvalue()
