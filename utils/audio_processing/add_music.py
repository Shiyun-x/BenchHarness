"""Render deferred English vocal background music from JamendoLyrics.

Agent-facing stubs and schemas live in audio_tools.py and tools.yaml. This module
owns library access, lyric-window selection, mixing, caching and batch delivery.
"""
from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from uuid import uuid4

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import soundfile as sf
from pydub import AudioSegment

from utils.audio_processing.audio_jobs import (
    MAX_TIMELINE_MS, background_music_settings, validate_background_music, validate_noise)
from utils.audio_processing.audio_rendering import read_audio, write_audio
from utils.audio_processing.audio_receipts import write_receipt
from utils.audio_processing.library_paths import resolve_noise_root


DEFAULT_MUSIC_ROOT = Path(__file__).resolve().parents[2].parent / 'datasets' / 'data' / 'JamendoLyrics'
INTENSITY_TO_DB = {'low': 0.0, 'medium': -2.5, 'high': -5.0}
MIN_VOCAL_COVERAGE = .5


def resolve_music_root(music_root=None):
    return Path(music_root or os.environ.get('MUSIC_ROOT') or DEFAULT_MUSIC_ROOT).resolve()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode('utf-8')).hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _spans(values):
    result = []
    for value in values:
        start, end = value.get('start'), value.get('end')
        if (all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                for v in (start, end)) and 0 <= start < end):
            result.append((float(start), float(end)))
    return sorted(result)


def _merge(spans, gap=0):
    result = []
    for start, end in sorted(spans):
        if result and start <= result[-1][1] + gap:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def _coverage(spans, start, end):
    return sum(max(0, min(right, end) - max(left, start)) for left, right in spans) / (end - start)


def load_music_library(music_root=None):
    """Preflight metadata, source paths and codecs before any paid TTS request."""
    root = resolve_music_root(music_root)
    manifest = root / 'metadata.jsonl'
    if not manifest.is_file():
        raise ValueError(f'JamendoLyrics metadata.jsonl missing: {manifest}; pass --music-root or MUSIC_ROOT')
    tracks = []
    for line in manifest.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('language') != 'en' or not row.get('text'):
            continue
        vocals = _merge(_spans(row.get('words') or []))
        lines = _spans(row.get('lines') or [])
        if not vocals or not lines:
            continue
        relative = row.get('file_name')
        if not isinstance(relative, str):
            raise ValueError('music file_name must be a relative path')
        audio = (root / relative).resolve()
        if not audio.is_relative_to(root) or not audio.is_file():
            raise ValueError(f'music file missing or outside the library: {relative}')
        try:
            info = sf.info(str(audio))
        except (RuntimeError, sf.LibsndfileError) as exc:
            raise ValueError(f'Cannot decode music {relative}; MP3 requires a libsndfile build with MPEG support') from exc
        if (info.channels not in (1, 2) or info.frames <= 0 or
                vocals[-1][1] > info.duration + .5 or lines[-1][1] > info.duration + .5):
            raise ValueError(f'invalid music audio or lyric timestamps: {relative}')
        tracks.append({'id': row.get('name') or audio.stem, 'path': audio,
                       'vocals': vocals, 'lines': lines, 'duration': info.duration,
                       'source_url': row.get('url'), 'license_type': row.get('license_type')})
    if not tracks:
        raise ValueError('No English songs with timed lyrics found in JamendoLyrics')
    return {'metadata_path': manifest, 'tracks': sorted(tracks, key=lambda t: str(t['path']))}


def _select_window(library, duration, seed):
    """Prefer a continuous song window; loop a vocal phrase if a long window is sparse."""
    eligible = []
    for track in library['tracks']:
        candidates = []
        for start, _ in track['lines']:
            end = min(start + duration, track['vocals'][-1][1], track['duration'])
            if end <= start:
                continue
            coverage = _coverage(track['vocals'], start, end)
            if coverage >= MIN_VOCAL_COVERAGE and end - start >= min(duration, 3):
                candidates.append((start, end, coverage))
        # Avoid selecting a very short tail when a full-length vocal window exists.
        full = [c for c in candidates if c[1] - c[0] >= duration - .001]
        if full:
            candidates = full
        if not candidates:
            for start, end in _merge(track['vocals'], gap=1):
                end = min(end, start + duration)
                if end - start >= min(duration, 1) and _coverage(track['vocals'], start, end) >= MIN_VOCAL_COVERAGE:
                    candidates.append((start, end, _coverage(track['vocals'], start, end)))
        if candidates:
            eligible.append((track, candidates))
    if not eligible:
        raise ValueError('No usable English vocal window in the music library')
    rng = random.Random(seed)
    track, candidates = rng.choice(eligible)
    return track, rng.choice(candidates)


def _decode_clip(path, start, end, rate, channels):
    with sf.SoundFile(str(path)) as source:
        source.seek(round(start * source.samplerate))
        raw = source.buffer_read(round((end - start) * source.samplerate), dtype='int16')
        segment = AudioSegment(bytes(raw), sample_width=2,
                               frame_rate=source.samplerate, channels=source.channels)
    segment = segment.set_frame_rate(rate).set_channels(channels)
    values = np.frombuffer(segment.raw_data, dtype='<i2').reshape(-1, channels).astype(np.float64)
    if len(values) < 2:
        raise ValueError('music/noise excerpt is empty or too short')
    return values


def _fit_audio(clip, frames, rate):
    """Repeat with short crossfades; preserve the exact foreground frame count."""
    result = np.empty((frames, clip.shape[1]), dtype=np.float64)
    initial = min(frames, len(clip))
    result[:initial] = clip[:initial]
    cursor = initial
    overlap = min(round(rate * .05), len(clip) // 4)
    while cursor < frames:
        count = min(overlap, cursor)
        fade = np.linspace(0, 1, count, endpoint=False)[:, None]
        if count:
            result[cursor-count:cursor] *= 1 - fade
            result[cursor-count:cursor] += clip[:count] * fade
        append = min(len(clip) - count, frames - cursor)
        result[cursor:cursor+append] = clip[count:count+append]
        cursor += append
    fade_frames = min(round(rate * .02), frames // 4)
    if fade_frames:
        fade = np.linspace(0, 1, fade_frames, endpoint=False)[:, None]
        result[:fade_frames] *= fade
        result[-fade_frames:] *= fade[::-1]
    return result


def _speech_mask(clean, rate):
    block = max(1, round(rate * .02))
    power = np.mean(clean * clean, axis=1)
    chunks = [float(np.mean(power[i:i+block])) for i in range(0, len(clean), block)]
    maximum = max(chunks)
    if maximum <= 1e-12:
        raise ValueError('cannot mix background music into silent foreground audio')
    return np.repeat(np.asarray(chunks) >= max(maximum * .01, 1e-12), block)[:len(clean)]


def add_background_music(audio_path, output_path, intensity=None, *, speech_to_music_db=None,
                         music_root=None, seed=None, noise=None, noise_root=None):
    """Real renderer: English lyrics + accompaniment, stable level, optional TUT noise.

    Music defaults to 0 dB relative to active foreground RMS (allowed: -5 to 0).
    Both backgrounds use the same active foreground RMS. Sum in float and attenuate
    once to prevent clipping; no ducking, speech modification or source overwrites.
    """
    settings = background_music_settings(intensity, speech_to_music_db)
    validate_noise(noise)
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise ValueError('music seed must be an integer')
    source, target = Path(audio_path).resolve(), Path(output_path).resolve()
    if source == target:
        raise ValueError('music output must not overwrite foreground audio')
    if target.suffix.lower() not in ('.wav', '.flac'):
        raise ValueError('music output extension must be .wav or .flac')
    params, samples = read_audio(source)
    rate, channels, _ = params
    clean = np.asarray(samples, dtype=np.float64).reshape(-1, channels)
    duration = len(clean) / rate
    if duration * 1000 > MAX_TIMELINE_MS:
        raise ValueError('foreground audio must not exceed one hour')
    mask = _speech_mask(clean, rate)
    library = load_music_library(music_root)
    if target.is_relative_to(library['metadata_path'].parent):
        raise ValueError('music output must not overwrite files inside the music library')
    source_hash = _sha256(source)
    seed = int(source_hash[:16], 16) if seed is None else seed
    track, (start, end, coverage) = _select_window(library, duration, seed)
    dependency_paths = [source, library['metadata_path'], track['path']]
    if target in dependency_paths:
        raise ValueError('music output must not overwrite library assets')
    noise_path = None
    if noise:
        from utils.audio_processing.add_noise import _load_tut_noise_index
        index = _load_tut_noise_index(str(resolve_noise_root(noise_root)))
        if noise['noise_type'] not in index or not index[noise['noise_type']]:
            raise ValueError('noise type not found in TUT library: ' + noise['noise_type'])
        noise_path = random.Random(f'{seed}:noise').choice(sorted(index[noise['noise_type']]))
        dependency_paths.append(noise_path.resolve())
        if target == noise_path.resolve():
            raise ValueError('music output must not overwrite noise library assets')
    dependencies = [{'path': str(p), 'sha256': _sha256(p)} for p in dependency_paths]
    ratio = float(settings['speech_to_music_db'] if 'speech_to_music_db' in settings
                  else INTENSITY_TO_DB[settings['intensity']])
    recipe = {'version': 1, 'dependencies': dependencies, 'seed': seed,
              'start': start, 'end': end, 'speech_to_music_db': ratio,
              'noise': noise, 'output_format': target.suffix.lower()}
    key = _fingerprint(recipe)
    sidecar = target.with_suffix(target.suffix + '.music.json')
    if target.is_file() and sidecar.is_file():
        try:
            previous = json.loads(sidecar.read_text(encoding='utf-8'))
            if previous.get('recipe_fingerprint') == key and previous.get('audio_sha256') == _sha256(target):
                return {**previous['result'], 'status': 'cached'}
        except (ValueError, KeyError):
            pass
    background = _fit_audio(_decode_clip(track['path'], start, end, rate, channels), len(clean), rate)
    speech_rms = float(np.sqrt(np.mean(clean[mask] ** 2)))
    music_rms = float(np.sqrt(np.mean(background[mask] ** 2)))
    if not music_rms:
        raise ValueError('selected vocal music excerpt is silent')
    gain = speech_rms / music_rms / (10 ** (ratio / 20))
    mixed = clean + background * gain
    noise_info = None
    if noise_path is not None:
        info = sf.info(str(noise_path))
        noise_audio = _fit_audio(_decode_clip(noise_path, 0, min(duration, info.duration), rate, channels), len(clean), rate)
        noise_rms = float(np.sqrt(np.mean(noise_audio[mask] ** 2)))
        if not noise_rms:
            raise ValueError('selected environmental noise is silent')
        from utils.audio_processing.add_noise import INTENSITY_TO_SNR
        noise_ratio = INTENSITY_TO_SNR[noise.get('intensity', 'medium')]
        noise_gain = speech_rms / noise_rms / (10 ** (noise_ratio / 20))
        mixed += noise_audio * noise_gain
        noise_info = {'path': str(noise_path), 'actual_snr': noise_ratio,
                      'gain_db': 20 * math.log10(noise_gain)}
    peak = float(np.max(np.abs(mixed)))
    scale = min(1, 32767 / peak) if peak else 1
    output = np.clip(np.rint(mixed * scale), -32768, 32767).astype(np.int16)
    write_audio(target, params, array('h', output.reshape(-1)))
    result = {'status': 'generated', 'music_audio_path': str(target),
              'duration_ms': round(duration * 1000), 'music_id': track['id'],
              'music_source_url': track['source_url'], 'license_type': track['license_type'],
              'seed': seed, 'start_ms': round(start * 1000), 'end_ms': round(end * 1000),
              'annotated_vocal_coverage': coverage, 'looped': end - start < duration - 1 / rate,
              'speech_to_music_db': ratio, 'music_gain_db': 20 * math.log10(gain),
              'peak_scale': scale, 'noise': noise_info, 'dependencies': dependencies}
    _write_json(sidecar, {'recipe_fingerprint': key, 'audio_sha256': _sha256(target), 'result': result})
    return result


def music_jobs(record):
    """Stable per-item/output seeds, independent of output folder and delivery codec."""
    identity = {key: record.get(key) for key in ('subtask_id', 'dataset_id', 'idx', 'id')}
    if not any(v is not None for v in identity.values()):
        identity['script'] = record.get('_speech_segments') or record.get('_transcript')
    outputs = record.get('_audio_outputs', []) if '_speech_segments' in record else []
    jobs = []
    for output in outputs:
        if output.get('background_music'):
            validate_background_music(output['background_music'])
            validate_noise(output.get('noise'))
            jobs.append({'audio_path': output['output_path'], 'output_path': output['final_audio_path'],
                         **output['background_music'], 'noise': output.get('noise'),
                         'seed': int(_fingerprint([identity, output['id']])[:16], 16)})
    settings = record.get('_add_background_music_args')
    if settings and '_speech_segments' not in record:
        validate_background_music(settings)
        validate_noise(record.get('_add_environmental_noise_args'))
        jobs.append({'audio_path': record.get('_music_input_path') or record['_tts_output_path'],
                     'output_path': record['_final_audio_path'], **settings,
                     'noise': record.get('_add_environmental_noise_args'),
                     'seed': int(_fingerprint(identity)[:16], 16)})
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--eval-json', type=Path, help='Run deferred background_music outputs after TTS/assembly.')
    mode.add_argument('--audio-path', type=Path, help='Mix one existing foreground WAV/FLAC.')
    parser.add_argument('--output-path', type=Path)
    parser.add_argument('--music-root', type=Path, help='JamendoLyrics directory with metadata.jsonl; or MUSIC_ROOT.')
    parser.add_argument('--noise-root', type=Path)
    level = parser.add_mutually_exclusive_group()
    level.add_argument('--intensity', choices=tuple(INTENSITY_TO_DB),
                       help='Speech/music ratio: low=0, medium=-2.5, high=-5 dB.')
    level.add_argument('--speech-to-music-db', type=float, help='Speech/music ratio from -5 to 0 dB; defaults to 0.')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args(argv)
    try:
        if args.audio_path:
            if not args.output_path:
                parser.error('--audio-path requires --output-path')
            result = add_background_music(args.audio_path, args.output_path, args.intensity,
                speech_to_music_db=args.speech_to_music_db, music_root=args.music_root, seed=args.seed)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            records = json.loads(args.eval_json.read_text(encoding='utf-8-sig'))
            if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
                raise ValueError('evaluation manifest must be an array of objects')
            jobs = [(record, job) for record in records for job in music_jobs(record)]
            # Check the library even before the first selected output is mixed.
            if jobs:
                load_music_library(args.music_root)
            for index, (record, job) in enumerate(jobs, 1):
                result = add_background_music(**job, music_root=args.music_root, noise_root=args.noise_root)
                write_receipt(record, job['output_path'], dependencies=result['dependencies'])
                print(f'[{index}/{len(jobs)}] {result["status"]}: {job["output_path"]}')
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
