"""Deliver deferred TTS scripts with MiniMax, then optionally overlay their segments.

python -m utils.audio_processing.minimax_batch --eval-json manifest.json --config voices.json
The API key is read only from MINIMAX_API_KEY. --dry-run exports requests without API calls.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
from http.client import IncompleteRead
import json
import os
from pathlib import Path
import sys
from threading import Event, Lock
import time
import wave
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.audio_processing.audio_jobs import (
    normalize_dialog, validate_overlay, speaker_label, safe_audio_id, validate_noise, validate_background_music)
from utils.audio_processing.audio_rendering import read_audio, render_audio, overlay_audio
from utils.audio_processing.audio_formats import audio_format, format_record
from utils.audio_processing.audio_receipts import write_receipt
from utils.audio_processing.speaker_batch import synthesis_mode, prepare_groups, process_speaker_records


DEFAULT_BASE_URL = 'https://api.minimax.cn/v1'
DEFAULT_VOICE_CONFIG = Path(__file__).resolve().parents[1] / 'resources' / 'minimax_voices.json'
DEFAULT_CONCURRENCY = 20
LANGUAGES = dict(zip(
    'zh en ar ru es fr pt de tr nl uk vi id ja it ko th pl ro el cs fi hi bg da he ms fa sk sv hr tl hu no sl ca nn ta af'.split(),
    'Chinese English Arabic Russian Spanish French Portuguese German Turkish Dutch Ukrainian Vietnamese '
    'Indonesian Japanese Italian Korean Thai Polish Romanian Greek Czech Finnish Hindi Bulgarian Danish '
    'Hebrew Malay Persian Slovak Swedish Croatian Filipino Hungarian Norwegian Slovenian Catalan Nynorsk '
    'Tamil Afrikaans'.split()))
LANGUAGES.update({'zh-cn': 'Chinese', 'zh-tw': 'Chinese', 'yue': 'Chinese,Yue', 'auto': 'auto'})


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def voice_map(dialog, config, *, sample_key=None):
    """Choose voices per item, preserving person identity within that item and retries."""
    speakers = list(dict.fromkeys(speaker_label(turn) for turn in dialog))
    explicit = config.get('speaker_voices', {})
    pools = config.get('voice_pools', {})
    if not isinstance(explicit, dict) or not isinstance(pools, dict):
        raise ValueError('speaker_voices and voice_pools must be objects')
    mapping = {speaker: explicit[speaker] for speaker in speakers if speaker in explicit}
    if any(not isinstance(voice, str) or not voice.strip() for voice in mapping.values()):
        raise ValueError('speaker_voices values must be nonempty voice IDs')
    if len(set(mapping.values())) != len(mapping):
        raise ValueError('different speakers require distinct voice IDs')
    candidates = {}
    for speaker in speakers:
        if speaker in mapping:
            continue
        gender = next((turn['gender'] for turn in dialog if speaker_label(turn) == speaker
                       and turn.get('gender', 'auto') != 'auto'), 'auto')
        pool = pools.get(gender, pools.get('auto', []))
        if not isinstance(pool, list) or any(not isinstance(v, str) or not v.strip() for v in pool):
            raise ValueError('voice_pools values must be lists of voice IDs')
        candidates[speaker] = list(dict.fromkeys(pool))

    def rank(speaker, voice=None):
        return fingerprint({'sample': sample_key, 'seed': config.get('voice_assignment_seed', 0),
                            'label': speaker, 'voice': voice})

    # Assign constrained gender pools before flexible auto voices consume them.
    for speaker in sorted(candidates, key=lambda s: (len(candidates[s]), rank(s))):
        available = [voice for voice in candidates[speaker] if voice not in mapping.values()]
        if available:
            mapping[speaker] = min(available, key=lambda voice: rank(speaker, voice))
        elif len(speakers) == 1 and config.get('default_voice_id'):
            voice = config['default_voice_id']
            if not isinstance(voice, str) or not voice.strip():
                raise ValueError('default_voice_id must be a nonempty string')
            mapping[speaker] = voice
        else:
            raise ValueError(f'no distinct configured voice for speaker {speaker!r}')
    return mapping


def build_request(turn, voice_id, config):
    turn = normalize_dialog([turn])[0]
    if len(turn['text']) >= 10000:
        raise ValueError('MiniMax turn text must be less than 10000 characters; split the script turn')
    language = turn.get('language', config.get('default_language', 'en'))
    language_tag = language.lower().replace('_', '-')
    boost = LANGUAGES.get(language_tag) or LANGUAGES.get(language_tag.split('-')[0])
    if boost is None:
        boost = next((v for v in LANGUAGES.values() if v.lower() == language.lower()), None)
    if boost is None:
        raise ValueError(f'unsupported MiniMax language: {language}')
    rate = config.get('sample_rate', 32000)
    if rate not in (8000, 16000, 22050, 24000, 32000, 44100) or isinstance(rate, bool):
        raise ValueError('unsupported MiniMax sample_rate')
    settings = {'speed': 1.1, 'vol': 1, 'pitch': 0, **config.get('voice_defaults', {})}
    if set(settings) - {'speed', 'vol', 'pitch'}:
        raise ValueError('unsupported voice_defaults')
    settings.update({key: turn[key] for key in ('speed', 'vol', 'pitch') if key in turn})
    normalize_dialog([{'text': turn['text'], **settings}])
    return {'model': config.get('model', 'speech-2.8-hd'), 'text': turn['text'],
            'stream': False, 'language_boost': boost, 'output_format': 'hex',
            'voice_setting': {'voice_id': voice_id, **settings},
            'audio_setting': {'sample_rate': rate, 'format': audio_format(config), 'channel': 1}}


def resolve_rpm(config, rpm=None):
    value = config.get('rpm') if rpm is None else rpm
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
        raise ValueError('rpm must be a positive integer')
    return value


class RequestLimiter:
    """Pace all attempts in a batch, including retries, and share server cooldowns."""

    def __init__(self, rpm=None, *, clock=None, sleep=None):
        resolve_rpm({}, rpm)
        self.interval = 60.0 / rpm if rpm else 0
        self.clock = clock or time.monotonic
        self.sleep = sleep or time.sleep
        self.lock = Lock()
        self.cancelled = Event()
        self.abort_reason = 'Audio batch interrupted'
        self.next_request = 0
        self.blocked_until = 0

    def wait(self, seconds):
        deadline = self.clock() + seconds
        while True:
            if self.cancelled.is_set():
                raise RuntimeError(self.abort_reason)
            remaining = deadline - self.clock()
            if remaining <= 0:
                return
            self.sleep(min(remaining, 1.0))

    def acquire(self):
        while True:
            if self.cancelled.is_set():
                raise RuntimeError(self.abort_reason)
            with self.lock:
                now = self.clock()
                delay = max(self.next_request, self.blocked_until) - now
                if delay <= 0:
                    self.next_request = now + self.interval
                    return
            self.wait(delay)

    def defer(self, seconds):
        with self.lock:
            now = self.clock()
            newly_blocked = self.blocked_until <= now
            self.blocked_until = max(self.blocked_until, now + seconds)
            return newly_blocked

    def abort(self, reason):
        self.abort_reason = reason
        self.cancelled.set()


def _retry_after(headers):
    value = headers.get('Retry-After') if headers else None
    if value is not None:
        try:
            seconds = float(value)
            if 0 <= seconds < float('inf'):
                return seconds
        except (TypeError, ValueError):
            try:
                target = parsedate_to_datetime(value)
                if target.tzinfo is None:
                    target = target.replace(tzinfo=timezone.utc)
                return max(0, (target - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                pass
    return 0


def _runtime_config(config):
    return {**config, '_request_limiter': RequestLimiter(resolve_rpm(config))}


def request_minimax(payload, api_key, config):
    """Fetch non-streaming audio bytes; business failures never become audio files."""
    endpoint = config.get('base_url', DEFAULT_BASE_URL).rstrip('/') + '/t2a_v2'
    request = Request(endpoint, data=json.dumps(payload).encode('utf-8'),
                      headers={'Authorization': 'Bearer ' + api_key, 'Content-Type': 'application/json'})
    limiter = config.get('_request_limiter') or RequestLimiter(resolve_rpm(config))
    max_attempts = config.get('max_attempts', 6)
    if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts <= 0:
        raise ValueError('max_attempts must be a positive integer')

    def rate_limited(reason, attempt, headers=None):
        delay = max(60.0, _retry_after(headers))
        if limiter.defer(delay):
            print(f'[MiniMax] Rate limited; pausing all new requests for at least {delay:g}s. '
                  'Completed audio remains cached.', flush=True)
        if attempt == max_attempts - 1:
            raise RuntimeError(f'{reason}; exhausted {max_attempts} attempts')

    for attempt in range(max_attempts):
        limiter.acquire()
        try:
            observer = config.get('_on_minimax_attempt')
            if observer:
                observer({'attempt': attempt + 1, 'time': time.time(), 'text_chars': len(payload['text'])})
            with urlopen(request, timeout=config.get('timeout_s', 120)) as response:
                response_headers = getattr(response, 'headers', None)
                result = json.load(response)
            status = result.get('base_resp') or {}
            if status.get('status_code') in (1002, 1039):
                rate_limited(f'MiniMax error {status["status_code"]}: {status.get("status_msg", "rate limit")}',
                             attempt, response_headers)
                continue
            if status.get('status_code') != 0:
                if status.get('status_code') == 2056:
                    limiter.abort('MiniMax account quota exhausted (2056); pending requests cancelled')
                raise ValueError(f'MiniMax error {status.get("status_code")}: {status.get("status_msg", "unknown")}')
            data = result.get('data') or {}
            if data.get('status') != 2 or not isinstance(data.get('audio'), str) or not data['audio']:
                raise ValueError('MiniMax returned incomplete or empty audio')
            observer = config.get('_on_minimax_response')
            if observer:
                observer(result)
            return bytes.fromhex(data['audio'])
        except HTTPError as exc:
            if exc.code == 429:
                rate_limited('MiniMax HTTP 429: rate limit', attempt, exc.headers)
                continue
            if exc.code not in (500, 502, 503, 504) or attempt == max_attempts - 1:
                raise RuntimeError(f'MiniMax HTTP {exc.code}') from None
        except (URLError, TimeoutError, IncompleteRead):
            if attempt == max_attempts - 1:
                raise RuntimeError('MiniMax connection failed or timed out') from None
        limiter.wait(min(2 ** attempt, 30))
    raise RuntimeError('MiniMax retries exhausted')


def plan_record(record, config):
    named = '_speech_segments' in record
    dialog = normalize_dialog(record.get('_speech_segments') if named else record.get('_transcript'))
    tts_path = record.get('_tts_output_path')
    if not named and (not isinstance(tts_path, str) or not tts_path):
        raise ValueError('record requires _tts_output_path')
    identity = {key: record[key] for key in ('subtask_id', 'dataset_id', 'idx', 'id')
                if key in record and record[key] is not None}
    if not identity:
        identity = {'tts_output_path': tts_path,
                    'segment_paths': [segment.get('audio_path') for segment in record.get('_tts_segments', [])]}
    mapping = voice_map(dialog, config, sample_key=identity)
    supplied = record.get('_tts_segments') or []
    paths = {item['id']: item['audio_path'] for item in supplied}
    if supplied and (len(paths) != len(supplied) or set(paths) != {t['id'] for t in dialog}):
        raise ValueError('_tts_segments does not match the script')
    if named and not supplied:
        raise ValueError('named audio requires _tts_segments paths')
    directory = Path(tts_path).parent / (Path(tts_path).stem + '_segments') if tts_path else None
    segments = []
    for index, turn in enumerate(dialog):
        kind = audio_format(config)
        path = paths[turn['id']] if turn['id'] in paths else str(directory / (turn['id'] + '.' + kind))
        if Path(path).suffix.lower() != '.' + kind:
            raise ValueError('audio path extension must match audio_format; prepare the manifest with format_record first')
        next_turn = dialog[index + 1] if index + 1 < len(dialog) else None
        default_pause = (150 if speaker_label(turn) == speaker_label(next_turn) else 300) if next_turn else 0
        segments.append({'id': turn['id'], 'audio_path': path,
                         'pause_after_ms': turn.get('pause_after_ms', default_pause),
                         'request': build_request(turn, mapping[speaker_label(turn)], config)})
    overlay = record.get('_overlay_audio_args')
    if overlay:
        validate_overlay(overlay.get('tracks'), {s['id'] for s in segments})
        if not overlay.get('output_path'):
            raise ValueError('overlay requires output_path')
    outputs = record.get('_audio_outputs', []) if named else []
    if not isinstance(outputs, list):
        raise ValueError('_audio_outputs must be an array')
    output_ids = set()
    for output in outputs:
        if (not isinstance(output, dict) or not safe_audio_id(output.get('id')) or
                output['id'] in output_ids or output['id'] in paths or not output.get('output_path')):
            raise ValueError('invalid or duplicate named audio output')
        output_ids.add(output['id'])
        validate_overlay(output.get('tracks'), set(paths), require_all=False)
        validate_noise(output.get('noise'))
        validate_background_music(output.get('background_music'))
        if (output.get('noise') or output.get('background_music')) and not output.get('final_audio_path'):
            raise ValueError('processed output requires final_audio_path')
    output_paths = [segment['audio_path'] for segment in segments] + ([tts_path] if tts_path else [])
    output_paths.extend(output['output_path'] for output in outputs)
    output_paths.extend(output['final_audio_path'] for output in outputs
                        if output.get('noise') or output.get('background_music'))
    if overlay:
        output_paths.append(overlay['output_path'])
    if len({str(Path(p).resolve()).lower() for p in output_paths}) != len(output_paths):
        raise ValueError('audio segment and output paths must be distinct')
    return {'voice_map': mapping, 'segments': segments, 'tts_output_path': tts_path, 'overlay': overlay,
            'named': named, 'outputs': outputs}


def resolve_concurrency(config, concurrency=None):
    value = config.get('concurrency', DEFAULT_CONCURRENCY) if concurrency is None else concurrency
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError('concurrency must be a positive integer')
    return value


def _synthesize_segment(segment, config, api_key, synthesize):
    path = Path(segment['audio_path'])
    # Keep existing WAV cache names and avoid evicting them when generating FLAC.
    sidecar = path.with_suffix('.flac.request.json' if path.suffix.lower() == '.flac' else '.request.json')
    key = fingerprint({'base_url': config.get('base_url', DEFAULT_BASE_URL), 'request': segment['request']})
    if sidecar.is_file() and path.is_file():
        try:
            metadata = json.loads(sidecar.read_text(encoding='utf-8'))
            read_audio(path, expected_format=segment['request']['audio_setting']['format'])
            if metadata.get('fingerprint') == key and metadata.get('audio_sha256') == hashlib.sha256(path.read_bytes()).hexdigest():
                return 'cached'
        except (OSError, ValueError, wave.Error, EOFError):
            pass
    raw = synthesize(segment['request'], api_key, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(uuid4().hex + '.tmp')
    try:
        temporary.write_bytes(raw)
        params, _ = read_audio(temporary, expected_format=segment['request']['audio_setting']['format'])
        if params != (segment['request']['audio_setting']['sample_rate'], 1, 2):
            raise ValueError('MiniMax audio does not match requested audio parameters')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    write_json(sidecar, {'fingerprint': key, 'audio_sha256': hashlib.sha256(raw).hexdigest(),
                         'request': segment['request']})
    return 'generated'


def process_record(record, config, api_key=None, *, dry_run=False, synthesize=request_minimax):
    resolve_rpm(config)
    plan = plan_record(record, config)
    if synthesis_mode(config) == 'speaker':
        return process_records([record], config, api_key, dry_run=dry_run, synthesize=synthesize, plans=[plan])[0]
    if dry_run:
        return {'status': 'planned', **plan}
    if not api_key:
        raise ValueError('set MINIMAX_API_KEY before synthesis')
    config = _runtime_config(config)
    for segment in plan['segments']:
        _synthesize_segment(segment, config, api_key, synthesize)
    return _assemble_record(record, plan)


def _assemble_record(record, plan):
    if plan['named']:
        outputs = []
        for output in plan['outputs']:
            result = overlay_audio(plan['segments'], output['tracks'], output['output_path'])
            write_receipt(record, output['output_path'])
            outputs.append({'id': output['id'], **result})
        for segment in plan['segments']:
            write_receipt(record, segment['audio_path'])
        return {'status': 'generated', 'voice_map': plan['voice_map'], 'outputs': outputs}
    result = render_audio(plan['segments'], plan['tts_output_path'])
    write_receipt(record, plan['tts_output_path'])
    if plan['overlay']:
        result = overlay_audio(plan['segments'], plan['overlay']['tracks'], plan['overlay']['output_path'])
        write_receipt(record, plan['overlay']['output_path'])
    return {'status': 'generated', 'voice_map': plan['voice_map'], **result}


def process_records(records, config, api_key=None, *, concurrency=None, dry_run=False,
                    synthesize=request_minimax, on_record=None, on_segment=None, plans=None, rpm=None):
    """Share one bounded pool across all samples; assemble only completed dependencies.

    Worker threads write distinct speech files. Reporting and assembly run in the
    calling thread, and results retain input order regardless of completion order.
    """
    if synthesis_mode(config) == 'speaker':
        return process_speaker_records(records, config, api_key, concurrency=concurrency,
            dry_run=dry_run, synthesize=synthesize, on_record=on_record, on_segment=on_segment,
            plans=plans, rpm=rpm)
    workers = resolve_concurrency(config, concurrency)
    config = {**config, 'rpm': resolve_rpm(config, rpm)}
    plans = [plan_record(record, config) for record in records] if plans is None else plans
    if len(plans) != len(records):
        raise ValueError('one audio plan is required per record')
    results = [None] * len(records)

    def finish(index, result):
        results[index] = result
        if on_record is not None:
            on_record(index, result)

    if dry_run:
        for index, plan in enumerate(plans):
            finish(index, {'status': 'planned', **plan})
        return results
    if not api_key:
        raise ValueError('set MINIMAX_API_KEY before synthesis')

    jobs = {}
    remaining = [len(plan['segments']) for plan in plans]
    errors = {}
    for index, plan in enumerate(plans):
        for segment in plan['segments']:
            path = str(Path(segment['audio_path']).resolve()).casefold()
            if path in jobs:
                if jobs[path]['segment']['request'] != segment['request']:
                    raise ValueError('different speech requests share one audio path: ' + segment['audio_path'])
                jobs[path]['owners'].append(index)
            else:
                jobs[path] = {'segment': segment, 'owners': [index]}

    config = _runtime_config(config)
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {}
    try:
        for job in jobs.values():
            future = executor.submit(_synthesize_segment, job['segment'], config, api_key, synthesize)
            futures[future] = job
        for count, future in enumerate(as_completed(futures), 1):
            job = futures.pop(future)
            error = None
            try:
                status = future.result()
            except Exception as exc:
                status, error = 'failed', f'segment {job["segment"]["id"]}: {type(exc).__name__}: {exc}'
                print(f'[MiniMax] TTS {count}/{len(jobs)} failed: {error}', flush=True)
            if on_segment is not None:
                on_segment(count, len(jobs), status)
            for index in job['owners']:
                if error is not None:
                    errors.setdefault(index, error)
                remaining[index] -= 1
                if remaining[index]:
                    continue
                if index in errors:
                    result = {'status': 'failed', 'reason': errors[index]}
                else:
                    try:
                        result = _assemble_record(records[index], plans[index])
                    except Exception as exc:
                        result = {'status': 'failed', 'reason': str(exc)}
                finish(index, result)
    except BaseException:
        config['_request_limiter'].cancelled.set()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-json', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=DEFAULT_VOICE_CONFIG,
                        help='Defaults to utils/resources/minimax_voices.json.')
    parser.add_argument('--dry-run', action='store_true', help='Export planned requests; do not call MiniMax')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--concurrency', type=int, help='Maximum simultaneous TTS requests; default from config or 20.')
    parser.add_argument('--rpm', type=int, help='Maximum TTS requests per minute across the entire batch, including retries.')
    parser.add_argument('--synthesis-mode', choices=('segment', 'speaker'), help='Request one turn or batch same-voice turns.')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    records = json.loads(args.eval_json.read_text(encoding='utf-8'))
    if (not isinstance(config, dict) or not isinstance(records, list) or
            any(not isinstance(record, dict) for record in records)):
        parser.error('config must be an object and evaluation must be an array')
    try:
        config['synthesis_mode'] = synthesis_mode(config, args.synthesis_mode)
        concurrency = resolve_concurrency(config, args.concurrency)
        rpm = resolve_rpm(config, args.rpm)
        kind = audio_format(config)
        records = [format_record(record, kind) for record in records]
    except ValueError as exc:
        parser.error(str(exc))
    api_key = os.environ.get('MINIMAX_API_KEY')
    if not args.dry_run and not api_key:
        parser.error('set MINIMAX_API_KEY before synthesis')
    if not args.dry_run and kind != 'wav':
        converted = args.eval_json.with_name(args.eval_json.stem + '.' + kind + '.json')
        write_json(converted, records)
        print(f'Delivery manifest: {converted}', flush=True)
    report_path = args.report or args.eval_json.with_suffix('.minimax-report.json')
    selected = [(index, record) for index, record in enumerate(records)
                if record.get('_transcript') or record.get('_speech_segments')]
    completed = {}

    def save_result(position, result):
        index = selected[position][0]
        completed[index] = {'record_idx': index, **result}
        write_json(report_path, [completed[i] for i in sorted(completed)])
        detail = ': ' + result['reason'] if result['status'] == 'failed' else ''
        print(f'[{len(completed)}/{len(selected)}] sample {index}: {result["status"]}{detail}', flush=True)

    print(f'TTS concurrency: {concurrency}; RPM: {rpm if rpm is not None else "unpaced"}', flush=True)
    results = process_records([record for _, record in selected], config, api_key,
                              concurrency=concurrency, dry_run=args.dry_run, on_record=save_result, rpm=rpm)
    if not selected:
        write_json(report_path, [])
    print(f'Report: {report_path}')
    return int(any(item['status'] == 'failed' for item in results))


if __name__ == '__main__':
    raise SystemExit(main())
