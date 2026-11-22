import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _extract_audio_tool_args(sample: Dict[str, Any]) -> Tuple[Dict, Dict]:
    """Extract text2speech and add_environmental_noise tool args from a sample."""
    messages = sample.get("_pure_tool_messages", {})
    tts_args: Dict = {}
    noise_args: Dict = {}
    if isinstance(messages, dict):
        tts_args = messages.get("text2speech_tool_args") or {}
        noise_args = messages.get("add_environmental_noise_tool_args") or {}
    elif isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if not tts_args:
                tts_args = msg.get("text2speech_tool_args") or {}
            if not noise_args:
                noise_args = msg.get("add_environmental_noise_tool_args") or {}
    return tts_args, noise_args


def _extract_overlay_tool_args(sample: Dict[str, Any]) -> Dict:
    messages = sample.get('_pure_tool_messages') or []
    if isinstance(messages, dict):
        messages = [messages]
    return next((message['overlay_audio_tool_args'] for message in messages
                 if isinstance(message, dict) and message.get('overlay_audio_tool_args')), {})


def _extract_music_tool_args(sample: Dict[str, Any]) -> Dict:
    messages = sample.get('_pure_tool_messages') or []
    if isinstance(messages, dict):
        messages = [messages]
    return next((m['add_background_music_tool_args'] for m in messages
                 if isinstance(m, dict) and m.get('add_background_music_tool_args')), {})


def _has_audio_args(verified_buffer: Dict[str, Any]) -> bool:
    """Return True if any sample in the buffer contains audio tool args."""
    for items in verified_buffer.values():
        if not isinstance(items, list):
            continue
        for item in items:
            sample = item.get("sample") or {}
            tts, noise = _extract_audio_tool_args(sample)
            if tts or noise or _extract_music_tool_args(sample):
                return True
    return False


def _find_audio_path_in_final(current: Dict[str, Any]) -> Optional[str]:
    """Scan current.input for a string value that looks like an audio file path."""
    try:
        inp = current.get("input") or {}
    except Exception:
        return None
    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    for v in inp.values():
        if isinstance(v, str) and Path(v).suffix.lower() in audio_exts:
            return v
    return None


def _sort_key(path: Path) -> int:
    m = re.search(r"_(\d+)$", path.stem)
    return int(m.group(1)) if m else 0


def export_evaluation(cache_path: str, subtask_ids: Optional[List[str]] = None) -> str:
    """
    Read all verified_transformed_buffer_st_*.json files from {cache_path}/verify_log/,
    auto-detect whether audio tool args are present, and write:
      - evaluation.json                      (no audio)
      - evaluation_with_audio_settings.json  (audio args present)

    Returns the path of the written file.
    """
    verify_dir = Path(cache_path) / "verify_log"
    if subtask_ids is None:
        files = sorted(verify_dir.glob('verified_transformed_buffer_st_*.json'), key=_sort_key)
    else:
        files = [verify_dir / f'verified_transformed_buffer_{sid}.json' for sid in subtask_ids]
        files = [p for p in files if p.is_file()]

    if not files:
        print(f"[export_results] No verified buffer files found in {verify_dir}")
        return ""

    # Load all verified buffers
    buffers: Dict[str, Dict] = {}
    for fp in files:
        key = fp.stem.removeprefix('verified_transformed_buffer_')
        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        buffers[key] = data.get("verified_buffer") or {}

    # Detect whether any sample has audio tool args
    has_audio = any(_has_audio_args(buf) for buf in buffers.values())

    evaluations: List[Dict] = []
    for subtask_id, verified_buffer in buffers.items():
        for dataset_id, items in verified_buffer.items():
            if not isinstance(items, list):
                continue
            for item in items:
                sample = item.get("sample") or {}
                original = sample.get("original") or {}
                current = sample.get("current") or {}

                record: Dict[str, Any] = {
                    "subtask_id": subtask_id,
                    "dataset_id": dataset_id,
                    "idx": item.get("idx", -1),
                    "id": original.get("id", ""),
                    "sample": current,
                }
                generation = sample.get('_generation') or {}
                if generation:
                    record['_media_status'] = generation.get('media_status', 'ready')
                    record['_validation_phase'] = generation.get('validation_phase', 'content')

                if has_audio:
                    tts_args, noise_args = _extract_audio_tool_args(sample)
                    music_args = _extract_music_tool_args(sample)
                    if tts_args:
                        if 'segment_paths' in tts_args:
                            record['_speech_segments'] = tts_args['segments']
                            record['_tts_segments'] = tts_args['segment_paths']
                        else:
                            record["_transcript"] = tts_args.get("dialog", [])
                        if tts_args.get('segments') and 'segment_paths' not in tts_args:
                            record['_tts_segments'] = tts_args['segments']

                    overlay_args = _extract_overlay_tool_args(sample)
                    if overlay_args.get('outputs'):
                        record['_audio_outputs'] = overlay_args['outputs']
                    elif overlay_args:
                        record['_overlay_audio_args'] = overlay_args

                    noise_type = noise_args.get("noise_type", "")
                    if noise_type:
                        record["_add_environmental_noise_args"] = {
                            "noise_type": noise_type,
                            "intensity": noise_args.get("intensity", "medium"),
                        }
                        # Noise consumes the assembled speech, including any overlay.
                        tts_path = noise_args.get("audio_path")
                        if tts_path:
                            record['_noise_input_path'] = tts_path
                            record["_tts_output_path"] = tts_args.get('output_path') or tts_path
                        # final audio path in the sample = noisy wav
                        final_audio = None if 'segment_paths' in tts_args else _find_audio_path_in_final(current)
                        if final_audio:
                            record["_final_audio_path"] = final_audio
                    else:
                        # Without noise, final audio is sequential TTS or its overlay.
                        final_audio = None if 'segment_paths' in tts_args else _find_audio_path_in_final(current)
                        if final_audio:
                            record["_tts_output_path"] = tts_args.get('output_path') or final_audio
                            record['_final_audio_path'] = final_audio

                    if music_args:
                        record['_add_background_music_args'] = {
                            k: v for k, v in music_args.items() if k in ('intensity', 'speech_to_music_db')}
                        record['_music_input_path'] = music_args['audio_path']
                        record['_tts_output_path'] = tts_args.get('output_path') or music_args['audio_path']
                        final_audio = _find_audio_path_in_final(current)
                        if final_audio:
                            record['_final_audio_path'] = final_audio

                evaluations.append(record)

    # Always write evaluation.json (clean version without audio args)
    eval_records = [
        {k: v for k, v in rec.items() if not k.startswith("_")}
        for rec in evaluations
    ]
    eval_path = os.path.join(cache_path, "evaluation.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_records, f, ensure_ascii=False, indent=2)
    print(f"[export_results] Exported {len(eval_records)} samples -> {eval_path}")

    if has_audio:
        audio_path = os.path.join(cache_path, "evaluation_with_audio_settings.json")
        with open(audio_path, "w", encoding="utf-8") as f:
            json.dump(evaluations, f, ensure_ascii=False, indent=2)
        print(f"[export_results] Exported {len(evaluations)} samples -> {audio_path}")
        return audio_path

    return eval_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export verified buffers to evaluation JSON(s)")
    parser.add_argument("--cache-path", required=True,
                        help="Path to cache/{topic_id}/ directory containing verify_log/")
    _args = parser.parse_args()
    export_evaluation(_args.cache_path)
