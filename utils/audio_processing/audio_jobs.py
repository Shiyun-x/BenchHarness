"""Provider-independent contracts for deferred speech and optional audio overlay."""
from __future__ import annotations

import copy
import re


ASSET_AUDIO_TOOLS = frozenset({'text2speech', 'overlay_audio', 'add_environmental_noise', 'add_background_music'})
MAX_TIMELINE_MS = 3_600_000
TURN_FIELDS = {'id', 'text', 'speaker', 'label', 'language', 'gender',
               'speed', 'vol', 'pitch', 'pause_after_ms'}


def canonicalize_media_jobs(jobs):
    """Accept the registered speech tool name as an exact TTS job-kind alias."""
    result = copy.deepcopy(jobs)
    if isinstance(result, list):
        for job in result:
            if isinstance(job, dict) and job.get('kind') == 'text2speech':
                job['kind'] = 'tts'
    return result


def _number(value, low, high, *, integer=False):
    return (isinstance(value, int if integer else (int, float)) and
            not isinstance(value, bool) and low <= value <= high)


def speaker_label(turn):
    """Labels identify voices, never a question's target participant."""
    return turn.get('label', turn.get('speaker', '1'))


def safe_audio_id(value):
    return isinstance(value, str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}', value) is not None


def validate_noise(noise):
    if noise is not None and (not isinstance(noise, dict) or
            not isinstance(noise.get('noise_type'), str) or not noise['noise_type'].strip() or
            noise.get('intensity', 'medium') not in ('low', 'medium', 'high') or
            set(noise) - {'noise_type', 'intensity'}):
        raise ValueError('invalid noise parameters')


def validate_background_music(music):
    """Small authoring contract; assets, offsets and seeds belong to delivery."""
    if music is None:
        return
    if (not isinstance(music, dict) or not music or
            set(music) - {'intensity', 'speech_to_music_db'} or
            ('intensity' in music and music['intensity'] not in ('low', 'medium', 'high')) or
            ('speech_to_music_db' in music and not _number(music['speech_to_music_db'], -5, 0)) or
            ('intensity' in music and 'speech_to_music_db' in music)):
        raise ValueError('invalid background_music parameters; use intensity OR speech_to_music_db (-5 to 0 dB)')


def background_music_settings(intensity=None, speech_to_music_db=None):
    """Resolve optional level arguments, defaulting to equal speech/music RMS."""
    settings = {key: value for key, value in (
        ('intensity', intensity), ('speech_to_music_db', speech_to_music_db)) if value is not None}
    if not settings:
        settings = {'speech_to_music_db': 0.0}
    validate_background_music(settings)
    return settings


def audio_effect_suffix(output):
    return ('_music' if output.get('background_music') else '') + ('_noise' if output.get('noise') else '')


def normalize_dialog(dialog):
    """Validate turns, preserving legacy scripts and assigning stable segment IDs."""
    if not isinstance(dialog, list) or not dialog:
        raise ValueError('TTS job requires nonempty dialogue turns')
    result, ids, genders = [], set(), {}
    for index, turn in enumerate(dialog, 1):
        if not isinstance(turn, dict) or set(turn) - TURN_FIELDS:
            raise ValueError('unsupported dialogue parameters')
        if not isinstance(turn.get('text'), str) or not turn['text'].strip():
            raise ValueError('TTS job requires nonempty dialogue turns')
        value = copy.deepcopy(turn)
        value.setdefault('id', f'turn_{index}')
        if not isinstance(value['id'], str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}', value['id']):
            raise ValueError('dialogue id must be a safe segment ID, e.g. turn_1')
        if value['id'] in ids:
            raise ValueError('duplicate dialogue segment id')
        ids.add(value['id'])
        if any(k in value and (not isinstance(value[k], str) or not value[k].strip())
               for k in ('speaker', 'label', 'language', 'gender')):
            raise ValueError('dialogue metadata must be nonempty strings')
        if 'speaker' in value and 'label' in value and value['speaker'] != value['label']:
            raise ValueError('speaker and label must identify the same person')
        if value.get('gender', 'auto') not in ('male', 'female', 'auto'):
            raise ValueError('invalid voice gender')
        speaker, gender = speaker_label(value), value.get('gender', 'auto')
        if gender != 'auto':
            if speaker in genders and genders[speaker] != gender:
                raise ValueError('one speaker cannot have conflicting voice genders')
            genders[speaker] = gender
        for key, low, high, integer in (('speed', .9, 1.8, False), ('vol', .01, 10, False),
                                         ('pitch', -12, 12, True),
                                         ('pause_after_ms', 0, MAX_TIMELINE_MS, True)):
            if key in value and not _number(value[key], low, high, integer=integer):
                raise ValueError(f'invalid dialogue {key}: expected {low} to {high}')
        result.append(value)
    return result


def validate_overlay(tracks, segment_ids, *, require_all=True):
    """Named outputs may select and reuse segments; legacy tracks cover all once."""
    if not isinstance(tracks, list) or not tracks:
        raise ValueError('overlay_audio requires nonempty tracks')
    seen = set()
    for track in tracks:
        if not isinstance(track, dict) or set(track) - {'segment_id', 'start_ms', 'gain_db', 'pause_after_ms'}:
            raise ValueError('unsupported overlay track parameters')
        sid = track.get('segment_id')
        if not isinstance(sid, str) or sid not in segment_ids or (require_all and sid in seen):
            raise ValueError('overlay tracks must reference each existing TTS segment exactly once')
        seen.add(sid)
        if 'start_ms' in track and not _number(track['start_ms'], 0, MAX_TIMELINE_MS, integer=True):
            raise ValueError('start_ms must be an integer between 0 and 3600000')
        if 'gain_db' in track and not _number(track['gain_db'], -60, 12):
            raise ValueError('gain_db must be between -60 and 12')
        if 'pause_after_ms' in track and not _number(track['pause_after_ms'], 0, MAX_TIMELINE_MS, integer=True):
            raise ValueError('pause_after_ms must be an integer between 0 and 3600000')
    if require_all and seen != set(segment_ids):
        raise ValueError('overlay tracks must reference each existing TTS segment exactly once')


def named_audio_plan(jobs):
    """Compile scripts and named outputs without running tools or reserving files.

    Return None for the original dialog/tracks protocol, which remains readable.
    New segments never implicitly become a concatenated conversation.
    """
    jobs = canonicalize_media_jobs(jobs)
    if not isinstance(jobs, list):
        raise ValueError('media_jobs must be an array')
    if not any(isinstance(job, dict) and ('segments' in job or 'outputs' in job) for job in jobs):
        return None
    segments, outputs = [], []
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError('each media job must be an object')
        if job.get('kind') == 'tts':
            if set(job) - {'kind', 'segments'} or not isinstance(job.get('segments'), list) or not job['segments']:
                raise ValueError('TTS job requires a nonempty segments array of id, label and text')
            for segment in job['segments']:
                if (not isinstance(segment, dict) or not safe_audio_id(segment.get('id')) or
                        not isinstance(segment.get('label'), str) or not segment['label'].strip() or
                        'speaker' in segment):
                    raise ValueError('each speech segment requires a safe id and a nonempty speaker label')
            segments.extend(job['segments'])
        elif job.get('kind') == 'overlay_audio':
            if set(job) - {'kind', 'outputs'} or not isinstance(job.get('outputs'), list) or not job['outputs']:
                raise ValueError('overlay_audio requires a nonempty outputs array')
            outputs.extend(job['outputs'])
        else:
            raise ValueError('unsupported media job; expected tts or overlay_audio')
    segments = normalize_dialog(segments)
    if len({s['id'].casefold() for s in segments}) != len(segments):
        raise ValueError('segment ids must have distinct filenames, including case')
    segment_ids, output_ids = {s['id'] for s in segments}, set()
    for output in outputs:
        if (not isinstance(output, dict) or set(output) - {'id', 'tracks', 'noise', 'background_music'} or
                not safe_audio_id(output.get('id'))):
            raise ValueError('each audio output requires a safe id and tracks')
        if output['id'] in output_ids or output['id'] in segment_ids:
            raise ValueError('audio output ids must be unique and distinct from segment ids')
        output_ids.add(output['id'])
        validate_overlay(output.get('tracks'), segment_ids, require_all=False)
        validate_noise(output.get('noise'))
        validate_background_music(output.get('background_music'))
    names = [name.casefold() for output in outputs for name in
             ([output['id'], output['id'] + audio_effect_suffix(output)]
              if audio_effect_suffix(output) else [output['id']])]
    if len(set(names)) != len(names):
        raise ValueError('audio outputs require distinct clean and noisy filenames, including case')
    return {'segments': segments, 'outputs': copy.deepcopy(outputs)}


def split_media_jobs(jobs):
    """One TTS job, followed by at most one independent overlay job; legacy TTS is valid."""
    jobs = canonicalize_media_jobs(jobs)
    if not isinstance(jobs, list):
        raise ValueError('media_jobs must be an array')
    if not jobs:
        return None, None
    if len(jobs) > 2 or not isinstance(jobs[0], dict) or jobs[0].get('kind') != 'tts':
        raise ValueError('media_jobs requires one TTS job followed by optional overlay_audio')
    tts = jobs[0]
    if set(tts) - {'kind', 'dialog', 'noise', 'background_music'}:
        raise ValueError('unsupported TTS job parameters')
    validate_background_music(tts.get('background_music'))
    dialog = normalize_dialog(tts.get('dialog'))
    noise = tts.get('noise')
    if noise is not None and (not isinstance(noise, dict) or
            not isinstance(noise.get('noise_type'), str) or not noise['noise_type'].strip() or
            noise.get('intensity', 'medium') not in ('low', 'medium', 'high') or
            set(noise) - {'noise_type', 'intensity'}):
        raise ValueError('invalid noise parameters')
    overlay = jobs[1] if len(jobs) == 2 else None
    if len(jobs) == 2:
        if (not isinstance(overlay, dict) or overlay.get('kind') != 'overlay_audio' or
                set(overlay) - {'kind', 'tracks'}):
            raise ValueError('unsupported media job; expected overlay_audio')
        validate_overlay(overlay.get('tracks'), {turn['id'] for turn in dialog})
    return tts, overlay
