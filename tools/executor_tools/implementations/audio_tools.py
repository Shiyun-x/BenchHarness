
from __future__ import annotations
from typing import Dict, Any, List, Optional,Tuple
from pathlib import Path
from uuid import uuid4
import random
import threading
import os
import copy
# from TTS.api import TTS
from functools import lru_cache

from utils.audio_processing.library_paths import resolve_noise_root

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = str(_PROJECT_ROOT / "workspace" / "audio")
NOISE_ROOT = str(resolve_noise_root())

# ================ XTTS Configuration ================

# _TTS_INIT_LOCK = threading.Lock()   
# _TTS_RUN_LOCK = threading.Lock()    
# _TTS_INSTANCE: Optional[TTS] = None
# _AVAILABLE_SPEAKERS: List[str] = []

# def _get_tts() -> TTS:
#     global _TTS_INSTANCE, _AVAILABLE_SPEAKERS
#     with _TTS_INIT_LOCK:
#         if _TTS_INSTANCE is None:
#             tts = TTS(
#                 model_name="tts_models/multilingual/multi-dataset/xtts_v2"
#             ).to("cuda")
#             _TTS_INSTANCE = tts

#             if hasattr(tts, "speakers") and isinstance(tts.speakers, (list, tuple)):
#                 _AVAILABLE_SPEAKERS = list(tts.speakers)
#             else:
#                 _AVAILABLE_SPEAKERS = []

#         return _TTS_INSTANCE

FEMALE_SPEAKER_NAMES = {
    "Claribel Dervla",
    "Daisy Studious",
    "Gracie Wise",
    "Tammie Ema",
    "Alison Dietlinde",
    "Ana Florence",
    "Annmarie Nele",
    "Asya Anara",
    "Brenda Stern",
    "Gitta Nikolina",
    "Henriette Usha",
    "Sofia Hellen",
    "Tammy Grit",
    "Tanja Adelina",
    "Vjollca Johnnie",
    "Nova Hogarth",
    "Maja Ruoho",
    "Uta Obando",
    "Lidiya Szekeres",
    "Szofi Granger",
    "Camilla Holmström",
    "Lilya Stainthorpe",
    "Zofija Kendrick",
    "Narelle Moon",
    "Barbora MacLean",
    "Alexandra Hisakawa",
    "Alma María",
    "Rosemary Okafor",
}

def merge_dialog(dialog_list):
    if not dialog_list:
        return []

    has_speaker = any("speaker" in d and d.get("speaker") for d in dialog_list)

    if not has_speaker:
        merged_text = " ".join(d.get("text", "") for d in dialog_list if d.get("text"))
        return [{"text": merged_text}]

    merged = []
    last = None

    for item in dialog_list:
        txt = item.get("text", "")
        spk = item.get("speaker")

        if not spk:
            spk = "__NO_SPEAKER__"

        if last is None:
            last = {"speaker": spk, "text": txt}
            continue

        if last["speaker"] == spk:
            last["text"] = last["text"] + " " + txt
        else:
            merged.append(last)
            last = {"speaker": spk, "text": txt}

    if last:
        merged.append(last)

    for m in merged:
        if m["speaker"] == "__NO_SPEAKER__":
            m.pop("speaker", None)

    return merged

# def _pick_voice_for_gender(gender: Optional[str]) -> str:

#     tts = _get_tts()
#     global _AVAILABLE_SPEAKERS
#     speakers = _AVAILABLE_SPEAKERS or getattr(tts, "speakers", None) or []

#     if not speakers:
#         return "default"

#     g = (gender or "auto").lower()

#     if g == "female":
#         pool = [s for s in speakers if s in FEMALE_SPEAKER_NAMES]
#     elif g == "male":
#         pool = [s for s in speakers if s not in FEMALE_SPEAKER_NAMES]
#     else:  # auto or other
#         pool = speakers

#     if not pool:
#         pool = speakers

#     return random.choice(pool)

# def _synthesize_segment(
#     text: str,
#     language: str,
#     speaker_name: str,
#     output_path: str,
# ) -> str:

#     text = (text or "").strip()
#     if not text:
#         raise ValueError("_synthesize_segment: text is empty")

#     tts = _get_tts()
#     out = Path(output_path)
#     out.parent.mkdir(parents=True, exist_ok=True)

#     with _TTS_RUN_LOCK:
#         tts.tts_to_file(
#             text=text,
#             speaker=speaker_name,
#             language=language,
#             file_path=str(out),
#         )

#     return str(out)



# ================ Add Environmental Noise Configuration ================
INTENSITY_TO_SNR = {
    "low": 15.0,
    "medium": 10.0,
    "high": 5.0,
}


@lru_cache(maxsize=8)
def _load_tut_noise_index(noise_root: str) -> Dict[str, List[Path]]:
    """
    Parse meta.txt from the TUT noise library and index entries by label:
        {
          "residential_area": [Path(...), Path(...), ...],
          "city_center": [...],
          ...
        }
    """
    root = Path(noise_root)
    meta_path = root / "meta.txt"
    if not meta_path.exists():
        raise FileNotFoundError(f"TUT meta file not found: {meta_path}")

    index: Dict[str, List[Path]] = {}
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            rel_audio, label = parts[0], parts[1]
            audio_path = root / rel_audio  # resolve to absolute path
            index.setdefault(label, []).append(audio_path)

    return index


def _tile_or_trim(noise: AudioSegment, target_ms: int) -> AudioSegment:
    from pydub import AudioSegment
    if len(noise) == 0:
        return AudioSegment.silent(duration=target_ms)

    if len(noise) < target_ms:
        repeats = (target_ms // len(noise)) + 1
        noise = noise * repeats

    return noise[:target_ms]


def _mix_with_snr(
    clean: AudioSegment,
    noise: AudioSegment,
    intensity: str,
) -> Tuple[AudioSegment, float]:
    snr_target = INTENSITY_TO_SNR.get(intensity, 10.0)  

    clean_db = clean.dBFS if clean.dBFS is not None else -20.0
    noise_db = noise.dBFS if noise.dBFS is not None else -20.0

    target_noise_db = clean_db - snr_target
    gain = target_noise_db - noise_db
    noise_adj = noise.apply_gain(gain)

    mixed = clean.overlay(noise_adj)

    noise_adj_db = noise_adj.dBFS if noise_adj.dBFS is not None else target_noise_db
    actual_snr = clean_db - noise_adj_db

    return mixed, actual_snr


# ================ Tools Interface ================
def text2speech(
    dialog: Optional[List[Dict[str, Any]]] = None,
    *args, segments=None, **kwargs
) -> Dict[str, Any]:
    """Reserve independent segment paths; dialog remains a legacy compatibility input.

    No speech model or API is invoked here. The executor retains dialog in tool
    memory; MiniMax or XTTS delivery produces the WAV files after content acceptance.
    """
    from utils.audio_processing.audio_jobs import normalize_dialog, named_audio_plan
    if segments is not None:
        plan = named_audio_plan([{'kind': 'tts', 'segments': segments}])
        root = Path(OUTPUT_DIR) / ('tts_' + uuid4().hex)
        return {'status': 'pending', 'segments': [
            {'id': s['id'], 'label': s['label'], 'audio_path': str((root / (s['id'] + '.wav')).resolve())}
            for s in plan['segments']]}
    dialog = normalize_dialog(dialog)
    output_root_path = Path(OUTPUT_DIR)
    output_root_path.mkdir(parents=True, exist_ok=True)

    final_name = f"tts_{uuid4().hex}.wav"

    final_path = output_root_path / final_name

    return {
        "merged_audio_path": str(final_path.resolve()),
        "segments": [{'id': turn['id'], 'audio_path': str(
            (output_root_path / (final_name[:-4] + '_segments') / (turn['id'] + '.wav')).resolve())}
            for turn in dialog],
        "status": "pending",
    }


def overlay_audio(segments, tracks=None, output_path=None, *, outputs=None):
    """Schedule named outputs; legacy tracks/path calls render existing WAV files."""
    if outputs is not None:
        from utils.audio_processing.audio_jobs import (
            safe_audio_id, validate_overlay, validate_noise, speaker_label,
            validate_background_music, audio_effect_suffix)
        ids = {s['id'] for s in segments}
        by_id = {s['id']: s for s in segments}
        seen = set()
        if not isinstance(outputs, list) or not outputs:
            raise ValueError('overlay_audio requires nonempty outputs')
        root = Path(OUTPUT_DIR) / ('audio_' + uuid4().hex)
        scheduled = []
        for output in outputs:
            if (not isinstance(output, dict) or set(output) - {'id', 'tracks', 'noise', 'background_music'} or
                    not safe_audio_id(output.get('id')) or output['id'] in seen or output['id'] in ids):
                raise ValueError('audio outputs require unique safe ids and tracks')
            seen.add(output['id'])
            validate_overlay(output.get('tracks'), ids, require_all=False)
            validate_noise(output.get('noise'))
            validate_background_music(output.get('background_music'))
            tracks = copy.deepcopy(output['tracks'])
            for index, track in enumerate(tracks):
                segment = by_id[track['segment_id']]
                following = by_id[tracks[index + 1]['segment_id']] if index + 1 < len(tracks) else None
                pause = (150 if speaker_label(segment) == speaker_label(following) else 300) if following else 0
                track.setdefault('pause_after_ms', segment.get('pause_after_ms', pause))
            scheduled.append({**output, 'tracks': tracks,
                              'output_path': str((root / (output['id'] + '.wav')).resolve())})
            suffix = audio_effect_suffix(output)
            if suffix:
                scheduled[-1]['final_audio_path'] = str((root / (output['id'] + suffix + '.wav')).resolve())
        return {'status': 'pending', 'outputs': scheduled}
    from utils.audio_processing.audio_rendering import overlay_audio as render_overlay
    return render_overlay(segments, tracks, output_path)


def speech2text(
    audio_path: str,
    *args, **kwargs
):
    """
    Transcribe an audio file to text (ASR).

    audio_path: path to the input audio file
    returns: transcribed text string
    """
    pass


def add_background_music(audio_path: str, intensity: Optional[str] = None,
                         speech_to_music_db: Optional[float] = None) -> Dict[str, Any]:
    """Deferred interface only: reserve a path without loading music or audio.

    Defaults to a 0 dB speech/music ratio. Actual selection and rendering live
    in utils.audio_processing.add_music.
    """
    from utils.audio_processing.audio_jobs import background_music_settings
    background_music_settings(intensity, speech_to_music_db)
    source = Path(audio_path)
    target = source.with_name(f'{source.stem}_music_{uuid4().hex[:8]}.wav')
    return {'music_audio_path': str(target.resolve()), 'status': 'pending'}


def add_environmental_noise(
    audio_path: str,
    noise_type: str,
    intensity: str = "medium",
    noise_root: str = NOISE_ROOT,
    seed: Optional[int] = None,
    *args, **kwargs
) -> Dict[str, Any]:
    """
    add environmental_noise to audio file using TUT noise dataset

    parameters
    ----------
        audio_path: path to clean audio file
        noise_type: type of noise, as in TUT meta.txt labels
        intensity:  'low' | 'medium' | 'high', default 'medium'
        noise_root: root directory of TUT noise dataset (with audio/ and meta/meta.txt)
        seed:       random seed for selecting noise file (optional)
    returns
        dict with keys:
            "noisy_audio_path": path to noisy audio file (absolute path)
          
    """
    audio_path = Path(str(audio_path))  
    # if not audio_path.exists():
    #     raise FileNotFoundError(f"audio not found: {audio_path}")

    # index = _load_tut_noise_index(noise_root)
    # if noise_type not in index:
    #     noise_type = random.choice(list(index.keys()))

    # noise_candidates = index[noise_type]
    # if not noise_candidates:
    #     raise ValueError(f"No noise files for type '{noise_type}' in TUT index.")

    # rnd = random.Random(seed)
    # noise_path = rnd.choice(noise_candidates)

    # clean = AudioSegment.from_file(audio_path)
    # noise = AudioSegment.from_file(noise_path)

    # noise = _tile_or_trim(noise, target_ms=len(clean))
    # mixed, actual_snr = _mix_with_snr(clean, noise, intensity=intensity)

    noise_type = noise_type.replace("/", "_")
    out_root = audio_path.parent
    out_name = (
        f"{audio_path.stem}_noisy_{noise_type}_{intensity}_{uuid4().hex[:8]}.wav"
    )
    out_path = out_root / out_name

    # mixed.export(out_path, format="wav")

    return  {
        "noisy_audio_path": str(out_path.resolve())
    }

