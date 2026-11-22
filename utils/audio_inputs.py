"""Encode real audio for Qwen Omni's OpenAI-compatible input_audio parts."""

import base64
from pathlib import Path
from urllib.parse import urlparse


AUDIO_MIME_TYPES = {
    "wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac",
    "ogg": "audio/ogg", "m4a": "audio/mp4", "aac": "audio/aac",
    "amr": "audio/amr", "aiff": "audio/aiff", "aif": "audio/aiff",
    "opus": "audio/opus", "3gp": "audio/3gpp", "3gpp": "audio/3gpp",
}
AUDIO_EXTENSIONS = {"." + name for name in AUDIO_MIME_TYPES}


def audio_format(source):
    """Infer the container from a local filename or HTTP(S) URL path."""
    name = urlparse(source).path if source.startswith(("https://", "http://")) else source
    kind = Path(name).suffix.lower().lstrip(".")
    if kind not in AUDIO_MIME_TYPES:
        raise ValueError("Audio source needs a supported file extension: " + ", ".join(sorted(AUDIO_EXTENSIONS)))
    return kind


def input_audio(source):
    """Use public URLs directly; upload local bytes as a complete Data URI."""
    kind = audio_format(source)
    if source.startswith(("https://", "http://")):
        data = source
    else:
        content = Path(source).expanduser().read_bytes()
        if not content:
            raise ValueError("Audio file is empty.")
        encoded = base64.b64encode(content).decode("ascii")
        data = f"data:{AUDIO_MIME_TYPES[kind]};base64,{encoded}"
    return {"type": "input_audio", "input_audio": {"data": data, "format": kind}}
