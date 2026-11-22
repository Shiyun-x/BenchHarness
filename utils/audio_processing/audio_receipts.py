"""Bind delivered audio artifacts to the script/layout that produced them."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4


def manifest_fingerprint(record):
    settings = {key: record.get(key) for key in (
        '_transcript', '_tts_segments', '_tts_output_path', '_overlay_audio_args',
        '_add_environmental_noise_args', '_noise_input_path', '_final_audio_path')}
    if record.get('_add_background_music_args'):
        settings.update({k: record.get(k) for k in ('_add_background_music_args', '_music_input_path')})
    if '_speech_segments' in record:
        settings.update({k: record.get(k) for k in ('_speech_segments', '_audio_outputs')})
    return hashlib.sha256(json.dumps(settings, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def write_receipt(record, audio_path, *, dependencies=None):
    path = Path(audio_path)
    target = path.with_suffix(path.suffix + '.delivery.json')
    value = {'manifest_fingerprint': manifest_fingerprint(record),
             'audio_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    if dependencies:
        value['dependencies'] = dependencies
    temporary = target.with_name(uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def check_receipt(record, audio_path):
    path = Path(audio_path)
    receipt = path.with_suffix(path.suffix + '.delivery.json')
    if not record.get('_tts_segments') and not receipt.is_file():
        return
    if not receipt.is_file():
        raise ValueError('batch audio delivery receipt missing')
    value = json.loads(receipt.read_text(encoding='utf-8'))
    if (not isinstance(value, dict) or value.get('manifest_fingerprint') != manifest_fingerprint(record) or
            value.get('audio_sha256') != hashlib.sha256(path.read_bytes()).hexdigest()):
        raise ValueError('batch audio is stale or changed; rerun audio delivery')
    for dependency in value.get('dependencies', []):
        if hashlib.sha256(Path(dependency['path']).read_bytes()).hexdigest() != dependency['sha256']:
            raise ValueError('background music source changed; rerun audio delivery')


def delivery_audio_paths(record):
    """Check every named output plus directly referenced speech segments."""
    if '_speech_segments' not in record:
        return [record.get('_final_audio_path') or record.get('_tts_output_path')]
    outputs = [o.get('final_audio_path', o['output_path']) for o in record.get('_audio_outputs', [])]
    segments = {s['audio_path'] for s in record.get('_tts_segments', [])}
    def strings(value):
        if isinstance(value, dict):
            return [s for v in value.values() for s in strings(v)]
        if isinstance(value, list):
            return [s for v in value for s in strings(v)]
        return [value] if isinstance(value, str) else []
    direct = [s for s in strings(record.get('sample', {})) if s in segments]
    return list(dict.fromkeys(outputs + direct))
