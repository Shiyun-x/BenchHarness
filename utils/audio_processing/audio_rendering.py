"""Render ordered/overlapping WAV and lossless FLAC without a model or FFmpeg."""
from __future__ import annotations

from array import array
import os
from pathlib import Path
import sys
import wave
from uuid import uuid4

from .audio_jobs import MAX_TIMELINE_MS, validate_overlay


def read_wav(path):
    with wave.open(str(path), 'rb') as audio:
        params = (audio.getframerate(), audio.getnchannels(), audio.getsampwidth())
        frames = audio.getnframes()
        raw = audio.readframes(frames)
    if params[0] <= 0 or params[1] not in (1, 2) or params[2] != 2 or frames <= 0:
        raise ValueError('audio must be a nonempty 16-bit mono/stereo PCM WAV')
    if len(raw) != frames * params[1] * params[2]:
        raise ValueError('truncated WAV data')
    samples = array('h')
    samples.frombytes(raw)
    if sys.byteorder != 'little':
        samples.byteswap()
    return params, samples


def write_wav(path, params, samples):
    """Replace only after a complete WAV has been written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(uuid4().hex + '.tmp')
    data = array('h', samples)
    if sys.byteorder != 'little':
        data.byteswap()
    try:
        with wave.open(str(temporary), 'wb') as audio:
            audio.setframerate(params[0])
            audio.setnchannels(params[1])
            audio.setsampwidth(params[2])
            audio.writeframes(data.tobytes())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def require_flac_codec():
    try:
        import soundfile
    except ImportError as exc:
        raise ValueError('FLAC delivery requires soundfile; install the project dependencies.') from exc
    return soundfile


def read_audio(path, *, expected_format=None):
    """Inspect actual container bytes, including temporary files without an extension."""
    with Path(path).open('rb') as source:
        header = source.read(42)
    kind = ('wav' if header[:4] == b'RIFF' and header[8:12] == b'WAVE' else
            'flac' if header[:4] == b'fLaC' else None)
    if kind is None or (expected_format is not None and kind != expected_format):
        raise ValueError('audio container does not match the requested WAV/FLAC format')
    if kind == 'wav':
        return read_wav(path)
    sf = require_flac_codec()
    declared_frames = int.from_bytes(header[18:26], 'big') & ((1 << 36) - 1)
    reader = sf.SoundFile
    if not declared_frames:
        class SequentialFlac(sf.SoundFile):
            # SoundFile otherwise seeks after each read; libsndfile cannot
            # reliably seek in FLAC streams with an unknown sample count.
            def seekable(self):
                return False
        reader = SequentialFlac
    try:
        with reader(str(path)) as audio:
            if (audio.format != 'FLAC' or audio.subtype != 'PCM_16' or
                    audio.channels not in (1, 2) or audio.samplerate <= 0):
                raise ValueError('audio must be nonempty 16-bit mono/stereo FLAC')
            params = (audio.samplerate, audio.channels, 2)
            # MiniMax may leave STREAMINFO.total_samples at zero (unknown).
            # libsndfile then reports an enormous frame count. Never use that
            # metadata to allocate: read into a bounded buffer until actual EOF.
            samples = array('h')
            block = array('h', [0]) * (65536 * audio.channels)
            max_samples = round(audio.samplerate * MAX_TIMELINE_MS / 1000) * audio.channels
            while True:
                frames = audio.buffer_read_into(block, dtype='int16')
                if not frames:
                    break
                count = frames * audio.channels
                if len(samples) + count > max_samples:
                    raise ValueError('decoded audio must not exceed one hour')
                samples.extend(block[:count])
        if not samples:
            raise ValueError('audio must be nonempty 16-bit mono/stereo FLAC')
        if declared_frames and len(samples) != declared_frames * params[1]:
            raise ValueError('truncated FLAC data')
        return params, samples
    except RuntimeError as exc:
        raise ValueError(f'invalid FLAC audio: {exc}') from exc


def write_audio(path, params, samples):
    """Encode from 16-bit PCM, preserving sample rate, channels and sample values."""
    target = Path(path)
    if target.suffix.lower() == '.wav':
        return write_wav(target, params, samples)
    if target.suffix.lower() != '.flac':
        raise ValueError('audio output extension must be .wav or .flac')
    sf = require_flac_codec()
    rate, channels, width = params
    if rate <= 0 or channels not in (1, 2) or width != 2:
        raise ValueError('audio output requires 16-bit mono/stereo PCM')
    data = array('h', samples)
    if not data or len(data) % channels:
        raise ValueError('audio output requires complete, nonempty frames')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(uuid4().hex + '.tmp')
    try:
        with sf.SoundFile(str(temporary), 'w', samplerate=rate, channels=channels,
                          format='FLAC', subtype='PCM_16') as audio:
            audio.buffer_write(data.tobytes(), dtype='int16')
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def render_audio(segments, output_path, tracks=None):
    """Append by default; explicit start_ms positions overlay. Attenuate peaks globally."""
    if not segments:
        raise ValueError('audio rendering requires segments')
    by_id = {segment['id']: segment for segment in segments}
    if len(by_id) != len(segments):
        raise ValueError('duplicate audio segment id')
    target = Path(output_path).resolve()
    if any(Path(segment['audio_path']).resolve() == target for segment in segments):
        raise ValueError('audio output must not overwrite a source segment')
    if tracks is None:
        tracks = [{'segment_id': segment['id']} for segment in segments]
    validate_overlay(tracks, set(by_id), require_all=False)
    params, placed, cursor, total = None, [], 0, 0
    for track in tracks:
        segment = by_id[track['segment_id']]
        current_params, samples = read_audio(segment['audio_path'])
        if params is not None and current_params != params:
            raise ValueError('all audio segments must have matching PCM parameters')
        params = current_params
        rate, channels, _ = params
        start = round(track['start_ms'] * rate / 1000) if 'start_ms' in track else cursor
        end = start + len(samples) // channels
        if end * 1000 / rate > MAX_TIMELINE_MS:
            raise ValueError('rendered audio must not exceed one hour')
        pause = track.get('pause_after_ms', segment.get('pause_after_ms', 0))
        if not isinstance(pause, int) or isinstance(pause, bool) or not 0 <= pause <= MAX_TIMELINE_MS:
            raise ValueError('invalid pause_after_ms')
        cursor = max(cursor, end + round(pause * rate / 1000))
        total = max(total, end)
        placed.append((start, samples, 10 ** (track.get('gain_db', 0) / 20)))
    mixed = array('f', [0]) * (total * params[1])
    for start, samples, gain in placed:
        offset = start * params[1]
        for index, sample in enumerate(samples):
            mixed[offset + index] += sample * gain
    peak = max(abs(value) for value in mixed)
    scale = min(1, 32767 / peak) if peak else 1
    write_audio(output_path, params, (round(value * scale) for value in mixed))
    return {'merged_audio_path': str(Path(output_path).resolve()),
            'duration_ms': round(total * 1000 / params[0]), 'peak_scale': scale}


def overlay_audio(segments, tracks, output_path):
    """The real overlay backend, shared by the registered tool and delivery CLI."""
    return render_audio(segments, output_path, tracks)
