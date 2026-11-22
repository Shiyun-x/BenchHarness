"""Deterministic content/media contracts, independent of model clients."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import random
from pathlib import Path

from utils.audio_processing.audio_jobs import (
    normalize_dialog, split_media_jobs, named_audio_plan, speaker_label, audio_effect_suffix)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    default=str).encode('utf-8')).hexdigest()


def normalize_reference(path):
    # Accept root prefixes and nonnegative numeric array-index aliases only.
    if isinstance(path, str):
        path = path.strip()
        path = path[2:] if path.startswith('$.') else path
        if re.fullmatch(r'[A-Za-z_]\w*(?:(?:\.[A-Za-z_]\w*|\.\d+)|\[(?:0|[1-9]\d*)\])*', path):
            path = re.sub(r'\[(0|[1-9]\d*)\]', r'.\1', path)
        return path
    return path


def get_path(value, path):
    if not isinstance(path, str):
        return None
    for key in normalize_reference(path).split('.'):
        if isinstance(value, dict):
            value = value.get(key)
        elif isinstance(value, list) and key.isdigit() and int(key) < len(value):
            value = value[int(key)]
        else:
            return None
    return value


def original_reference_examples(original, limit=4):
    """A few real paths, without source values, IDs or exhaustive manifests."""
    result = []
    def visit(value, path, depth):
        if len(result) >= limit or value is None:
            return
        if isinstance(value, dict) and depth < 3:
            for key, child in value.items():
                if path == 'original' and key == 'id':
                    continue
                if isinstance(key, str) and re.fullmatch(r'[A-Za-z_]\w*', key):
                    visit(child, path+'.'+key, depth+1)
                if len(result) >= limit:
                    break
        elif isinstance(value, list):
            for index, child in enumerate(value[:2]):
                if child is not None and len(result) < limit:
                    result.append(path+'.'+str(index))
        elif path != 'original':
            result.append(path)
    visit(original, 'original', 0)
    return result


def fields(section):
    return section.get('fields', section) if isinstance(section, dict) else {}


def media_fields(contract, kind):
    found = []
    for key, spec in fields(contract.get('sample_schema', {}).get('input', {})).items():
        dtype = str(spec.get('dtype', '')).lower() if isinstance(spec, dict) else ''
        if dtype == kind or kind in key.lower():
            found.append(key)
    return found


def has_modality(contract, kind):
    """Use the current subtask's declared modalities and actual schema fields."""
    return kind in contract.get('modalities', []) or bool(media_fields(contract, kind))



def final_minor_edit_fields(contract):
    """Actual textual input fields that may receive an exact final correction."""
    specs = fields(contract.get('sample_schema', {}).get('input', {}))
    reserved = set(media_fields(contract, 'image') + media_fields(contract, 'audio'))
    return {name for name, spec in specs.items()
            if name not in reserved and not name.lower().endswith(('path', 'url')) and
            isinstance(spec, dict) and str(spec.get('dtype', '')).lower() not in ('image', 'audio', 'video') and
            str(spec.get('type', 'str')).lower() in ('str', 'string', 'text', 'list', 'array')}


def normalize_candidate(candidate, contract):
    """Only unambiguous formatting normalization; never add missing content."""
    candidate = copy.deepcopy(candidate)
    if not isinstance(candidate, dict) or not isinstance(candidate.get('output'), dict):
        return candidate
    answer = candidate['output'].get('answer')
    if isinstance(answer, str):
        if contract['answer_type'] == 'binary':
            mapping = {'true': 'yes', 'false': 'no', 'y': 'yes', 'n': 'no'}
            candidate['output']['answer'] = mapping.get(answer.strip().lower(), answer.strip().lower())
        elif contract['answer_type'] == 'choice' and re.fullmatch(r'[A-Z][.)]?', answer.strip()):
            candidate['output']['answer'] = answer.strip().rstrip('.)')
    return candidate


def choice_labels(candidate):
    inp = candidate.get('input', {})
    options = inp.get('options')
    if isinstance(options, dict):
        return list(options)
    if isinstance(options, list):
        return [chr(65+i) for i in range(len(options))]
    question = options if isinstance(options, str) else inp.get('question', '')
    if isinstance(question, str):
        return re.findall(r'(?:^|\n)\s*([A-Z])[.)：:]\s*', question)
    return []


def shuffle_choices(candidate, seed):
    """Shuffle content only, preserving stem and mapping gold exactly."""
    candidate = copy.deepcopy(candidate)
    inp = candidate['input']
    options = inp.get('options')
    answer = candidate['output']['answer']
    labels = choice_labels(candidate)
    order = list(range(len(labels)))
    random.Random(seed).shuffle(order)
    mapping = {labels[old]: chr(65+new) for new, old in enumerate(order)}
    if isinstance(options, list):
        inp['options'] = [options[i] for i in order]
    elif isinstance(options, dict):
        inp['options'] = {chr(65+n): options[labels[i]] for n, i in enumerate(order)}
    else:
        question = options if isinstance(options, str) else inp['question']
        matches = list(re.finditer(r'(?:^|\n)\s*([A-Z])[.)：:]\s*', question))
        texts = [question[m.end():(matches[n+1].start() if n+1 < len(matches) else len(question))].strip()
                 for n, m in enumerate(matches)]
        stem = question[:matches[0].start()].rstrip()
        rebuilt = stem + '\n' + '\n'.join(f'{chr(65+n)}. {texts[i]}' for n, i in enumerate(order))
        inp['options' if isinstance(options, str) else 'question'] = rebuilt.lstrip('\n')
    candidate['output']['answer'] = mapping[answer]
    return candidate


def check_content(candidate, support, jobs, contract, reference_root):
    """Return concrete errors; deferred audio has a separate content contract."""
    errors = []
    schema = contract['sample_schema']
    if not isinstance(candidate, dict):
        return ['candidate must be an object']
    if set(candidate) != set(schema):
        errors.append('candidate sections do not match schema')
    types = {'str': str, 'string': str, 'text': str, 'list': list, 'array': list,
             'str|list': (str, list),
             'dict': dict, 'object': dict, 'int': int, 'integer': int,
             'float': (int, float), 'number': (int, float), 'bool': bool, 'boolean': bool}
    for section, description in schema.items():
        actual = candidate.get(section)
        expected = fields(description)
        if not isinstance(actual, dict):
            errors.append(f'{section} must be an object')
            continue
        if set(actual) != set(expected):
            errors.append(f'{section} fields do not match schema')
        for name, spec in expected.items():
            value = actual.get(name)
            if value is None or (name in ('question', 'answer') and isinstance(value, str) and not value.strip()):
                errors.append(f'{section}.{name} is missing or empty')
            typename = str(spec.get('type', '')).lower() if isinstance(spec, dict) else ''
            if typename in types and (not isinstance(value, types[typename]) or
                                      typename in ('int', 'integer', 'number', 'float') and isinstance(value, bool)):
                errors.append(f'{section}.{name} has incorrect type')
            if isinstance(spec, dict) and isinstance(spec.get('enum'), list) and value not in spec['enum']:
                errors.append(f'{section}.{name} violates enum')
    answer = get_path(candidate, 'output.answer')
    if answer is None or answer == '':
        errors.append('gold answer missing')
    if contract['answer_type'] == 'choice':
        labels = choice_labels(candidate)
        if len(labels) < 2 or len(labels) != len(set(labels)) or answer not in labels:
            errors.append('choice options/answer label invalid; embed options on separate lines when no options field exists')
        requested = contract.get('option_count')
        if requested and len(labels) != requested:
            errors.append('choice count does not match task')
    if contract['answer_type'] == 'binary' and answer not in ('yes', 'no'):
        errors.append('binary answer must be yes or no')
    if not isinstance(support, dict) or not isinstance(support.get('answer_refs'), list) or not support.get('answer_refs'):
        errors.append('answer support references missing')
    else:
        for ref in support['answer_refs']:
            ref = normalize_reference(ref)
            if not isinstance(ref, str) or not ref.startswith(('original.', 'evidence.')) or get_path(reference_root, ref) is None:
                errors.append(f'invalid support reference: {ref}')
    try:
        audio_plan = named_audio_plan(jobs)
        tts, overlay = split_media_jobs(jobs) if audio_plan is None else (None, None)
    except ValueError as exc:
        errors.append(str(exc))
        return errors
    if tts:
        audio_keys = media_fields(contract, 'audio')
        if len(audio_keys) != 1 or get_path(candidate, f'input.{audio_keys[0]}') != 'media:audio':
            errors.append('deferred TTS must bind exactly one schema audio field to media:audio')
    deferred_refs = ({'media:audio:' + item['id'] for item in
                      audio_plan['segments'] + audio_plan['outputs']} if audio_plan else set())
    for kind in ('audio', 'image'):
        keys = media_fields(contract, kind)
        if kind in contract.get('modalities', []) and not keys:
            errors.append(f'{kind} modality missing schema field')
        for key in keys:
            value = get_path(candidate, f'input.{key}')
            if value == 'media:audio' and kind == 'audio' and jobs and audio_plan is None:
                continue
            refs = value if isinstance(value, list) else [value]
            def source_strings(node):
                if isinstance(node, dict):
                    return [s for v in node.values() for s in source_strings(v)]
                if isinstance(node, list):
                    return [s for v in node for s in source_strings(v)]
                return [node] if isinstance(node, str) else []
            allowed_media = source_strings(reference_root.get('original', {}))
            if not refs or any(not isinstance(v, str) or not (
                    (kind == 'audio' and v in deferred_refs) or
                    (v in allowed_media and (v.startswith(('http://', 'https://')) or Path(v).is_file()))) for v in refs):
                errors.append(f'input.{key} does not reference existing media or a deferred job')
    return errors


def bind_media(candidate, jobs, contract, output_root, item_key):
    """Assign deterministic paths WITHOUT invoking synthesis or creating media."""
    candidate = copy.deepcopy(candidate)
    if not jobs:
        return candidate, []
    plan = named_audio_plan(jobs)
    if plan is not None:
        return _bind_named_audio(candidate, plan, contract, output_root, item_key)
    tts, overlay = split_media_jobs(jobs)
    root = Path(output_root).resolve() / 'media'
    tts_path = str(root / f'{item_key}_tts.wav')
    overlay_path = str(root / f'{item_key}_overlay.wav') if overlay else None
    clean_path = overlay_path or tts_path
    suffix = audio_effect_suffix(tts)
    final_path = str(root / f'{item_key}{suffix}.wav') if suffix else clean_path
    candidate['input'][media_fields(contract, 'audio')[0]] = final_path
    segment_root = root / f'{item_key}_tts_segments'
    dialog = normalize_dialog(tts['dialog'])
    segments = []
    for index, turn in enumerate(dialog):
        following = dialog[index + 1] if index + 1 < len(dialog) else None
        default_pause = (150 if turn.get('speaker', '1') == following.get('speaker', '1') else 300) if following else 0
        segments.append({'id': turn['id'], 'audio_path': str(segment_root / (turn['id'] + '.wav')),
                         'pause_after_ms': turn.get('pause_after_ms', default_pause)})
    messages = [{'text2speech_tool_args': {'dialog': copy.deepcopy(tts['dialog']),
                                         'output_path': tts_path, 'segments': segments}}]
    if overlay:
        messages.append({'overlay_audio_tool_args': {'segments': copy.deepcopy(segments),
                        'tracks': copy.deepcopy(overlay['tracks']), 'output_path': overlay_path}})
    if tts.get('noise'):
        messages.append({'add_environmental_noise_tool_args': {
            'audio_path': clean_path, **tts['noise']}})
    if tts.get('background_music'):
        messages.append({'add_background_music_tool_args': {
            'audio_path': clean_path, **tts['background_music']}})
    return candidate, messages


def _bind_named_audio(candidate, plan, contract, output_root, item_key):
    root = Path(output_root).resolve() / 'media' / item_key
    paths = [{'id': s['id'], 'audio_path': str(root / 'segments' / (s['id'] + '.wav'))}
             for s in plan['segments']]
    by_id = {s['id']: s for s in plan['segments']}
    outputs = []
    for output in plan['outputs']:
        tracks = copy.deepcopy(output['tracks'])
        for index, track in enumerate(tracks):
            segment = by_id[track['segment_id']]
            following = by_id[tracks[index + 1]['segment_id']] if index + 1 < len(tracks) else None
            pause = (150 if speaker_label(segment) == speaker_label(following) else 300) if following else 0
            track.setdefault('pause_after_ms', segment.get('pause_after_ms', pause))
        assembled = {'id': output['id'], 'tracks': tracks,
                     'output_path': str(root / (output['id'] + '.wav'))}
        if output.get('noise'):
            assembled['noise'] = copy.deepcopy(output['noise'])
        if output.get('background_music'):
            assembled['background_music'] = copy.deepcopy(output['background_music'])
        suffix = audio_effect_suffix(output)
        if suffix:
            assembled['final_audio_path'] = str(root / (output['id'] + suffix + '.wav'))
        outputs.append(assembled)
    mapping = {'media:audio:' + s['id']: s['audio_path'] for s in paths}
    mapping.update({'media:audio:' + o['id']: o.get('final_audio_path', o['output_path']) for o in outputs})
    for key in media_fields(contract, 'audio'):
        value = candidate['input'][key]
        def bind(ref):
            if isinstance(ref, str) and ref.startswith('media:audio') and ref not in mapping:
                raise ValueError('unknown deferred audio reference: ' + ref)
            return mapping.get(ref, ref)
        candidate['input'][key] = [bind(ref) for ref in value] if isinstance(value, list) else bind(value)
    messages = [{'text2speech_tool_args': {'segments': copy.deepcopy(plan['segments']), 'segment_paths': paths}}]
    if outputs:
        messages.append({'overlay_audio_tool_args': {'segments': paths, 'outputs': outputs}})
    return candidate, messages
