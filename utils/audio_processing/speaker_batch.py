"""Batch one item's same-voice turns, then split at verified pause boundaries."""
from __future__ import annotations

from array import array
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import time
import unicodedata
from urllib.request import urlopen
from uuid import uuid4

from .audio_rendering import read_audio, write_audio

SPLIT_VERSION = 3


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def synthesis_mode(config, override=None):
    value = override if override is not None else config.get('synthesis_mode', 'segment')
    if value not in ('segment', 'speaker'):
        raise ValueError('synthesis_mode must be segment or speaker')
    pause = config.get('speaker_pause_s', 1.0)
    if (isinstance(pause, bool) or not isinstance(pause, (int, float)) or
            not math.isfinite(pause) or not 0.8 <= pause <= 3):
        raise ValueError('speaker_pause_s must be between 0.8 and 3 seconds')
    if type(config.get('speaker_fallback', True)) is not bool:
        raise ValueError('speaker_fallback must be true or false')
    return value


def prepare_groups(plan, config):
    """Group within one record by the complete resolved voice/language settings."""
    synthesis_mode(config)
    separator = f"\n<#{config.get('speaker_pause_s', 1.0):.2f}#>\n"
    buckets = {}
    for segment in plan['segments']:
        settings = {k: v for k, v in segment['request'].items() if k != 'text'}
        buckets.setdefault(digest(settings), []).append(segment)
    groups = []
    for bucket in buckets.values():
        current, size = [], 0
        for segment in bucket:
            added = len(segment['request']['text']) + (len(separator) if current else 0)
            if current and size + added >= 10000:
                groups.append(_group(current, separator))
                current, size = [], 0
                added = len(segment['request']['text'])
            current.append(segment)
            size += added
        if current:
            groups.append(_group(current, separator))
    return groups


def _group(segments, separator):
    request = copy.deepcopy(segments[0]['request'])
    if len(segments) > 1:
        request.update(text=separator.join(s['request']['text'] for s in segments),
                       subtitle_enable=True, subtitle_type='word')
    return {'segments': segments, 'request': request}


def _canonical(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'<#[0-9.]+#>', '', text)
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold() if c.isalnum())


def _subtitle_edges(subtitles, texts):
    """Use timestamps only when subtitle text exactly covers every submitted turn."""
    if isinstance(subtitles, dict):
        subtitles = subtitles.get('subtitles', subtitles.get('data'))
    if not isinstance(subtitles, list):
        return None
    words = []
    for item in subtitles:
        nested = item.get('timestamped_words') if isinstance(item, dict) else None
        words.extend(nested if isinstance(nested, list) and nested else [item])
    # MiniMax repeats a source word for each spoken part of numbers/acronyms
    # (e.g. "4,000" -> "four", "thousand"). Sentence text retains the exact
    # source spelling, so prefer it whenever turn boundaries align with it.
    for entries in (subtitles, words):
        edges = _edges_from_entries(entries, texts)
        if edges is not None:
            return edges
    return None


def _edges_from_entries(entries, texts):
    spans, joined = [], ''
    for item in entries:
        if not isinstance(item, dict):
            return None
        text = _canonical(item.get('text', item.get('word', '')))
        if not text:
            continue
        start, end = item.get('time_begin'), item.get('time_end')
        if (isinstance(start, bool) or isinstance(end, bool) or
                not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or
                not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start):
            return None
        spans.append((len(joined), len(joined) + len(text), start, end))
        joined += text
    normalized = [_canonical(text) for text in texts]
    if not all(normalized) or joined != ''.join(normalized):
        return None
    edges, cursor = [], 0
    for text in normalized[:-1]:
        cursor += len(text)
        left = [s for s in spans if s[1] == cursor]
        right = [s for s in spans if s[0] == cursor]
        if not left or not right or left[-1][3] > right[0][2] + 100:
            return None
        edges.append((left[-1][3], right[0][2]))
    return edges


def split_intervals(params, samples, texts, *, pause_s=1.0, subtitles=None):
    """Cut inside measured silence; never estimate speech timing from text lengths."""
    from pydub import AudioSegment
    from pydub.silence import detect_silence
    rate, channels, width = params
    duration_ms = len(samples) * 1000 / channels / rate
    if len(texts) == 1:
        return [(0, len(samples) // channels)], {'method': 'single', 'boundaries': []}
    # PCM is native-endian in arrays, while AudioSegment expects little-endian.
    import sys
    pcm = array('h', samples)
    if sys.byteorder != 'little':
        pcm.byteswap()
    audio = AudioSegment(pcm.tobytes(), sample_width=width, frame_rate=rate, channels=channels)
    minimum = round(pause_s * 650)
    silences = detect_silence(audio, min_silence_len=minimum, silence_thresh=-50, seek_step=10)
    candidates = [(a, b) for a, b in silences if a > 30 and b < duration_ms - 30
                  and b - a <= pause_s * 1000 + 1500]
    edges = _subtitle_edges(subtitles, texts)
    chosen = []
    if edges is not None:
        for left, right in edges:
            midpoint = (left + right) / 2
            hits = [(a, b) for a, b in candidates if a - 150 <= midpoint <= b + 150
                    and (not chosen or a > chosen[-1][1])]
            if len(hits) != 1:
                raise ValueError('subtitle boundary does not identify exactly one separator silence')
            chosen.append(hits[0])
        method = 'subtitles+silence'
    else:
        if len(candidates) != len(texts) - 1:
            raise ValueError(f'expected {len(texts)-1} separator silences, found {len(candidates)}; '
                             'subtitle text could not verify the boundaries')
        chosen, method = candidates, 'silence-count'
    intervals, boundaries, start = [], [], 0
    for a, b in chosen:
        # Preserve a little silence at both speech edges, remove the middle only.
        end = round((a + 40) * rate / 1000)
        next_start = round((b - 40) * rate / 1000)
        # Long-window silence detection can include a quiet word tail. Extend
        # the preserved clips inward until the actual 10 ms cut edges are quiet.
        window = max(1, round(rate / 100))
        midpoint = round((a + b) * rate / 2000)
        quiet_square = (32768 * 10 ** (-50 / 20)) ** 2
        def quiet(first, last):
            values = samples[first * channels:last * channels]
            return bool(values) and sum(v*v for v in values) / len(values) <= quiet_square
        while end < midpoint and not quiet(end - window, end):
            end += window
        while next_start > midpoint and not quiet(next_start, next_start + window):
            next_start -= window
        if not quiet(end - window, end) or not quiet(next_start, next_start + window):
            raise ValueError('separator has no quiet cut edges')
        if end <= start or next_start <= end:
            raise ValueError('invalid or empty speech interval')
        intervals.append((start, end))
        boundaries.append({'silence_start_ms': a, 'silence_end_ms': b,
                           'left_cut_ms': end * 1000 / rate,
                           'right_cut_ms': next_start * 1000 / rate,
                           'removed_ms': (next_start - end) * 1000 / rate})
        start = next_start
    intervals.append((start, len(samples) // channels))
    for a, b in intervals:
        clip = samples[a * channels:b * channels]
        if b <= a or not clip or max(abs(v) for v in clip) <= 100:
            raise ValueError('split produced an empty or nearly silent speech segment')
    return intervals, {'method': method, 'boundaries': boundaries,
                       'batch_duration_ms': duration_ms,
                       'removed_separator_ms': sum(b['removed_ms'] for b in boundaries)}


def _paths(group, config):
    key = digest({'request': group['request'], 'base_url': config.get('base_url'),
                  'ids': [s['id'] for s in group['segments']]})
    base = Path(group['segments'][0]['audio_path']).parent / '_batches'
    kind = group['request']['audio_setting']['format']
    return key, base / (key[:24] + '.' + kind), base / (key[:24] + '.json')


def _sidecar(segment):
    path = Path(segment['audio_path'])
    return path.with_suffix('.flac.request.json' if path.suffix.lower() == '.flac' else '.request.json')


def synthesize_group(group, config, api_key, synthesize):
    from . import minimax_batch as batch
    started, attempts = time.monotonic(), []
    local = {**config, '_on_minimax_attempt': lambda value: attempts.append(value)}
    if len(group['segments']) == 1:
        status = batch._synthesize_segment(group['segments'][0], local, api_key, synthesize)
        return {'status': status, 'segments': 1, 'api_requests': len(attempts),
                'method': 'single', 'elapsed_s': time.monotonic() - started}
    key, raw_path, metadata_path = _paths(group, config)
    source_key = digest({'group': key, 'split_version': SPLIT_VERSION})
    valid = []
    for segment in group['segments']:
        try:
            path = Path(segment['audio_path'])
            info = json.loads(_sidecar(segment).read_text(encoding='utf-8'))
            read_audio(path, expected_format=group['request']['audio_setting']['format'])
            valid.append(info.get('speaker_source') == source_key and
                         info.get('audio_sha256') == hashlib.sha256(path.read_bytes()).hexdigest())
        except (OSError, ValueError):
            valid.append(False)
    if all(valid):
        return {'status': 'cached', 'segments': len(valid), 'api_requests': 0,
                'method': 'speaker-cache', 'elapsed_s': time.monotonic() - started}
    metadata = {}
    if metadata_path.is_file() and raw_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata.get('request_fingerprint') != key or metadata.get('audio_sha256') != hashlib.sha256(raw_path.read_bytes()).hexdigest():
            metadata = {}
    if not metadata:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        captured = {}
        def observe(response):
            captured.update(extra_info=response.get('extra_info') or {}, trace_id=response.get('trace_id'),
                            subtitle_url=(response.get('data') or {}).get('subtitle_file'))
        raw = synthesize(group['request'], api_key, {**local, '_on_minimax_response': observe})
        temporary = raw_path.with_name(uuid4().hex + '.tmp')
        try:
            temporary.write_bytes(raw)
            temporary.replace(raw_path)
        finally:
            temporary.unlink(missing_ok=True)
        metadata = {'request_fingerprint': key, 'request': group['request'],
                    'audio_sha256': hashlib.sha256(raw).hexdigest(), 'api_attempts': attempts, **captured}
        batch.write_json(metadata_path, metadata)
    if metadata.get('subtitle_url') and 'subtitles' not in metadata:
        try:
            with urlopen(metadata['subtitle_url'], timeout=min(30, config.get('timeout_s', 120))) as response:
                metadata['subtitles'] = json.load(response)
            batch.write_json(metadata_path, metadata)
        except (OSError, ValueError) as exc:
            metadata['subtitle_error'] = type(exc).__name__
            batch.write_json(metadata_path, metadata)
    try:
        params, samples = read_audio(raw_path, expected_format=group['request']['audio_setting']['format'])
        if params != (group['request']['audio_setting']['sample_rate'], 1, 2):
            raise ValueError('speaker batch PCM parameters do not match the request')
        intervals, metrics = split_intervals(params, samples, [s['request']['text'] for s in group['segments']],
                                            pause_s=config.get('speaker_pause_s', 1.0),
                                            subtitles=metadata.get('subtitles'))
        for segment, (a, b) in zip(group['segments'], intervals):
            path = Path(segment['audio_path'])
            write_audio(path, params, samples[a * params[1]:b * params[1]])
            batch.write_json(_sidecar(segment), {'speaker_source': source_key,
                'audio_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'request': segment['request'],
                'source_batch': str(raw_path), 'start_frame': a, 'end_frame': b})
        metadata.update(split_version=SPLIT_VERSION, split=metrics,
                        segments=[{'id': s['id'], 'start_frame': a, 'end_frame': b}
                                  for s, (a, b) in zip(group['segments'], intervals)])
        metadata.pop('split_error', None)
        batch.write_json(metadata_path, metadata)
        return {'status': 'generated', 'segments': len(intervals), 'api_requests': len(attempts),
                'batch_file': str(raw_path), 'extra_info': metadata.get('extra_info', {}),
                'elapsed_s': time.monotonic() - started, **metrics}
    except (OSError, ValueError) as exc:
        metadata['split_error'] = str(exc)
        batch.write_json(metadata_path, metadata)
        if not config.get('speaker_fallback', True):
            raise
        print(f'[MiniMax] Speaker batch split failed; falling back to individual turns: {exc}', flush=True)
        for segment in group['segments']:
            batch._synthesize_segment(segment, local, api_key, synthesize)
        return {'status': 'generated', 'segments': len(group['segments']), 'method': 'fallback',
                'reason': str(exc), 'api_requests': len(attempts), 'elapsed_s': time.monotonic() - started}


def process_speaker_records(records, config, api_key, *, concurrency=None, dry_run=False,
                            synthesize, on_record=None, on_segment=None, plans=None, rpm=None):
    from . import minimax_batch as batch
    synthesis_mode(config)
    workers = batch.resolve_concurrency(config, concurrency)
    config = batch._runtime_config({**config, 'rpm': batch.resolve_rpm(config, rpm)})
    plans = [batch.plan_record(r, config) for r in records] if plans is None else plans
    if len(plans) != len(records):
        raise ValueError('one audio plan is required per record')
    groups_by_record = [prepare_groups(plan, config) for plan in plans]
    if dry_run:
        results = [{'status': 'planned', **plan, 'speaker_groups': groups}
                   for plan, groups in zip(plans, groups_by_record)]
        for index, result in enumerate(results):
            if on_record:
                on_record(index, result)
        return results
    if not api_key:
        raise ValueError('Set the speech API key before synthesis.')
    jobs, paths = {}, {}
    for index, groups in enumerate(groups_by_record):
        for group in groups:
            key = tuple(str(Path(s['audio_path']).resolve()).casefold() for s in group['segments'])
            for path in key:
                if path in paths and paths[path] != key:
                    raise ValueError('different speaker batches share one audio path')
                paths[path] = key
            if key in jobs:
                if jobs[key]['group']['request'] != group['request']:
                    raise ValueError('different speech requests share one audio path')
                jobs[key]['owners'].append(index)
            else:
                jobs[key] = {'group': group, 'owners': [index]}
    remaining = [len(groups) for groups in groups_by_record]
    errors, details, results = {}, [[] for _ in records], [None] * len(records)
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {executor.submit(synthesize_group, job['group'], config, api_key, synthesize): job
                   for job in jobs.values()}
        for count, future in enumerate(as_completed(futures), 1):
            job, error = futures[future], None
            try:
                detail = future.result()
                status = detail['status']
            except Exception as exc:
                status, error = 'failed', f'{type(exc).__name__}: {exc}'
                detail = {'status': status, 'reason': error}
                print(f'[MiniMax] Speaker group {count}/{len(jobs)} failed: {error}', flush=True)
            if on_segment:
                on_segment(count, len(jobs), status)
            for index in job['owners']:
                details[index].append(detail)
                if error:
                    errors.setdefault(index, error)
                remaining[index] -= 1
                if remaining[index]:
                    continue
                if index in errors:
                    result = {'status': 'failed', 'reason': errors[index]}
                else:
                    try:
                        result = batch._assemble_record(records[index], plans[index])
                    except Exception as exc:
                        result = {'status': 'failed', 'reason': f'{type(exc).__name__}: {exc}'}
                result.update(synthesis_mode='speaker', speaker_groups=details[index])
                results[index] = result
                if on_record:
                    on_record(index, result)
    except BaseException:
        config['_request_limiter'].cancelled.set()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return results
