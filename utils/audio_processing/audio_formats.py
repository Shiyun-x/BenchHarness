"""Apply a delivery codec to generated assets without changing source manifests."""
from __future__ import annotations

import copy
from pathlib import Path

from .audio_rendering import require_flac_codec


def audio_format(config):
    value = config.get('audio_format', 'wav')
    if value not in ('wav', 'flac'):
        raise ValueError('audio_format must be wav or flac')
    if value == 'flac':
        require_flac_codec()
    return value


def _owned_paths(record):
    paths = []
    for key in ('_tts_output_path', '_final_audio_path', '_noise_input_path', '_music_input_path'):
        if record.get(key):
            paths.append(record[key])
    paths.extend(s['audio_path'] for s in record.get('_tts_segments', []))
    for output in record.get('_audio_outputs', []):
        paths.append(output['output_path'])
        if output.get('final_audio_path'):
            paths.append(output['final_audio_path'])
    overlay = record.get('_overlay_audio_args') or {}
    if overlay.get('output_path'):
        paths.append(overlay['output_path'])
    paths.extend(s['audio_path'] for s in overlay.get('segments', []))
    return paths


def _rewrite(record, replacements):
    def rewrite(value):
        if isinstance(value, dict):
            return {key: rewrite(child) for key, child in value.items()}
        if isinstance(value, list):
            return [rewrite(child) for child in value]
        if isinstance(value, str):
            return replacements.get(value, value)
        return value
    return rewrite(record)


def format_record(record, kind):
    """Rewrite only owned asset paths and their exact references, not question text."""
    paths = _owned_paths(record)
    if not paths:
        return copy.deepcopy(record)
    replacements = {}
    for source in paths:
        if not isinstance(source, str) or Path(source).suffix.lower() not in ('.wav', '.flac'):
            raise ValueError('generated audio paths must end in .wav or .flac')
        replacements[source] = str(Path(source).with_suffix('.' + kind))

    return _rewrite(record, replacements)


def relocate_records(records, source_dir, output_dir):
    """Reserve fresh experiment paths, preserving the existing manifests and media."""
    source, target = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source == target or source.is_relative_to(target) or target.is_relative_to(source / 'media'):
        raise ValueError('output_dir must be separate from the source directory and existing media')
    replacements = {}
    for record in records:
        for name in _owned_paths(record):
            path = Path(name).resolve()
            if not path.is_relative_to(source):
                raise ValueError('generated asset is outside the source directory: ' + name)
            relocated = (target / path.relative_to(source)).resolve()
            if not relocated.is_relative_to(target):
                raise ValueError('relocated asset escaped output_dir')
            replacements[name] = str(relocated)
    return [_rewrite(record, replacements) for record in records]
