"""
transcription.py
-----------------
Speech-to-text using faster-whisper (a fast version of OpenAI's Whisper model
that runs fully offline on your own computer).

The model is loaded once (on first use) and reused afterwards, because loading
it takes a few seconds.
"""

import os

from faster_whisper import WhisperModel

_model = None


def _get_model():
    """Load the Whisper model once, then reuse it for every transcription."""
    global _model
    if _model is None:
        # Model size can be changed in .env: tiny / base / small / medium.
        # Bigger = more accurate but slower. "base" is fine on a normal laptop.
        model_size = os.getenv("WHISPER_MODEL", "base")
        _model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _model


def transcribe_audio(file_path):
    """
    Convert the audio file at file_path into text.

    Returns:
        transcript (str)      - the full text
        duration (float)      - audio length in seconds
        language (str)        - detected language code, e.g. "en"

    Raises:
        ValueError if no speech could be found in the audio.
    """
    model = _get_model()
    # vad_filter skips long silences, which makes transcription faster and cleaner.
    segments, info = model.transcribe(file_path, vad_filter=True)

    # 'segments' is a generator: the real work happens while we loop over it.
    parts = [segment.text.strip() for segment in segments]
    transcript = " ".join(part for part in parts if part).strip()

    if not transcript:
        raise ValueError("No speech was detected. The audio may be silent, too short or corrupted.")

    return transcript, info.duration, info.language
