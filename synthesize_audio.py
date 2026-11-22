"""MiniMax delivery: python synthesize_audio.py [--eval-json FILE] [--concurrency 20] [--dry-run]."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from tools.shared.finalize_media import finalize_evaluation
from utils.audio_processing.audio_receipts import write_receipt
from utils.audio_processing.audio_formats import audio_format, format_record, relocate_records
from utils.audio_processing.library_paths import resolve_noise_root
from utils.audio_processing.speaker_batch import synthesis_mode as resolve_synthesis_mode, prepare_groups
from utils.audio_processing.minimax_batch import (
    DEFAULT_VOICE_CONFIG, plan_record, process_records, resolve_concurrency, resolve_rpm, write_json)


ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = ROOT / 'cache' / 'user_queries' / 'user_query_07'


def _windows_environment(name):
    """Read settings added after this terminal started, without printing secrets."""
    if os.name != 'nt':
        return None
    import winreg
    for hive, path in (
            (winreg.HKEY_CURRENT_USER, 'Environment'),
            (winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\Session Manager\Environment')):
        try:
            with winreg.OpenKey(hive, path) as key:
                value, _ = winreg.QueryValueEx(key, name)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        except OSError:
            continue
    return None


def get_api_key(api_env=None):
    selected = (api_env or os.environ.get('MINIMAX_API_ENV', 'MINIMAX_API')).strip()
    if selected not in ('MINIMAX_API', 'MINIMAX_API_2'):
        raise ValueError('MINIMAX_API_ENV must be MINIMAX_API or MINIMAX_API_2.')
    names = (selected,) if selected == 'MINIMAX_API_2' else ('MINIMAX_API', 'MINIMAX_API_KEY')
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            print(f'API key source: {name} (process environment)', flush=True)
            return value.strip()
    for name in names:
        value = _windows_environment(name)
        if value:
            print(f'API key source: {name} (Windows environment)', flush=True)
            return value
    raise ValueError(f'Set {selected} in your environment before synthesis.')


def noise_jobs(record):
    if '_speech_segments' in record:
        return [{'audio_path': output['output_path'], 'output_path': output['final_audio_path'],
                 **output['noise']} for output in record.get('_audio_outputs', [])
                if output.get('noise') and not output.get('background_music')]
    if record.get('_add_background_music_args'):
        return []  # The music renderer sums both backgrounds against the clean reference.
    settings = record.get('_add_environmental_noise_args')
    if settings:
        return [{'audio_path': record.get('_noise_input_path') or record['_tts_output_path'],
                 'output_path': record['_final_audio_path'], **settings}]
    return []


def read_records(path):
    records = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise ValueError('Evaluation manifest must be an array of objects: ' + str(path))
    return records


def has_speech(record):
    return bool(record.get('_speech_segments') or record.get('_transcript'))


def attach_audio_settings(records, manifest):
    """Recover scripts for exported selections, requiring an exact public-record match."""
    missing = [index for index, record in enumerate(records)
               if not has_speech(record) and record.get('sample', {}).get('input', {}).get('audio_url')]
    if not missing:
        return records
    if not manifest.is_file():
        raise ValueError('Selected audio records have no scripts; audio settings manifest missing: ' + str(manifest))

    def public_key(record):
        return json.dumps({key: value for key, value in record.items() if not key.startswith('_')},
                          ensure_ascii=False, sort_keys=True)

    candidates = {}
    for record in read_records(manifest):
        if has_speech(record):
            candidates.setdefault(public_key(record), []).append(record)
    enriched = list(records)
    for index in missing:
        matches = candidates.get(public_key(records[index]), [])
        if len(matches) != 1:
            raise ValueError(f'Selected record {index} matches {len(matches)} audio scripts in {manifest}; '
                             'expected exactly one unchanged record.')
        enriched[index] = matches[0]
    return enriched


def run_batch(input_cache_dir=DEFAULT_CACHE_DIR, config_path=None, *, dry_run=False,
              noise_root=None, api_key=None, synthesize=None, limit=None, concurrency=None, eval_json=None,
              rpm=None, synthesis_mode=None, output_dir=None, music_root=None, api_env=None):
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0):
        raise ValueError('--limit must be a positive integer.')
    cache_dir = Path(input_cache_dir).resolve()
    selection = Path(eval_json).resolve() if eval_json is not None else None
    batch_dir = selection.parent if selection else cache_dir / 'generation' / 'planned-v1'
    account_policy = batch_dir/'audio_delivery.json'
    if account_policy.is_file():
        policy = json.loads(account_policy.read_text(encoding='utf-8-sig'))
        if not isinstance(policy, dict):
            raise ValueError('audio_delivery.json must be an object.')
        if api_env is None:
            api_env = policy.get('api_env')
        if synthesis_mode is None:
            synthesis_mode = policy.get('synthesis_mode')
    if api_env is not None and api_env not in ('MINIMAX_API', 'MINIMAX_API_2'):
        raise ValueError('Audio account must be MINIMAX_API or MINIMAX_API_2.')
    manifest = batch_dir / 'evaluation_with_audio_settings.json'
    source_manifest = manifest
    config_path = Path(config_path).resolve() if config_path else DEFAULT_VOICE_CONFIG
    records = read_records(selection or manifest)
    if selection:
        records = attach_audio_settings(records, manifest)
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    if not isinstance(config, dict):
        raise ValueError('Voice config must be an object.')
    kind = audio_format(config)
    config['synthesis_mode'] = resolve_synthesis_mode(config, synthesis_mode)
    # Keep the original question selection and scripts reusable across codecs.
    records = [format_record(record, kind) for record in records]
    if output_dir is not None:
        target = Path(output_dir).resolve()
        records = relocate_records(records, batch_dir, target)
        batch_dir = target
        manifest = batch_dir / 'evaluation_with_audio_settings.json'
    concurrency = resolve_concurrency(config, concurrency)
    rpm = resolve_rpm(config, rpm)
    pending = [(i, r) for i, r in enumerate(records) if has_speech(r)]
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        raise ValueError('No speech scripts found in ' + str(selection or manifest))
    limit_suffix = f'.limit{limit}' if limit is not None else ''
    output_name = selection.stem.removesuffix('_with_audio_settings') if selection else 'evaluation'
    suffix = (f'.{output_name}' if selection else '') + limit_suffix
    ready_path = batch_dir / f'{output_name}_ready{limit_suffix}.json'
    if selection:
        manifest = batch_dir / f'{output_name}_with_audio_settings{limit_suffix}.json'
        if ready_path == selection:
            raise ValueError('Selected input filename conflicts with a generated output; use another filename.')
    if manifest.resolve() == source_manifest.resolve() and kind != 'wav':
        manifest = manifest.with_name(manifest.stem + '.' + kind + manifest.suffix)

    # Validate every selected sample before spending on the first API request.
    plans = [(i, plan_record(record, config)) for i, record in pending]
    all_music = []
    music_backend = None
    if any(record.get('_add_background_music_args') or any(o.get('background_music')
           for o in record.get('_audio_outputs', [])) for _, record in pending):
        from utils.audio_processing.add_music import add_background_music, load_music_library, music_jobs, resolve_music_root
        music_root = resolve_music_root(music_root)
        load_music_library(music_root)
        all_music = [(record, job) for _, record in pending for job in music_jobs(record)]
        music_backend = add_background_music
    total_segments = sum(len(plan['segments']) for _, plan in plans)
    total_requests = sum(len(prepare_groups(plan, config)) for _, plan in plans) if config['synthesis_mode'] == 'speaker' else total_segments
    print(f'Samples: {len(pending)}; speech segments: {total_segments}; concurrency: {concurrency}; '
          f'RPM: {rpm if rpm is not None else "unpaced"}', flush=True)
    print(f'Voice config: {config_path}; audio format: {kind}', flush=True)
    print(f'Synthesis mode: {config["synthesis_mode"]}; planned requests: {total_requests}', flush=True)
    if dry_run:
        report_path = batch_dir / f'minimax_dry_run_report{suffix}.json'
        write_json(report_path, [{'record_idx': i, 'status': 'planned', **plan,
                    **({'speaker_groups': prepare_groups(plan, config)} if config['synthesis_mode'] == 'speaker' else {})}
                    for i, plan in plans])
        print(f'Dry run passed. No API calls or audio files created. Report: {report_path}')
        return 0

    all_noise = [(record, job) for _, record in pending for job in noise_jobs(record)]
    noise_backend = None
    if all_noise or any(job.get('noise') for _, job in all_music):
        noise_root = resolve_noise_root(noise_root)
        if not (noise_root / 'meta.txt').is_file():
            raise ValueError(f'Noise library meta.txt missing: {noise_root}; pass --noise-root with your TUT library directory.')
        from utils.audio_processing.add_noise import add_environmental_noise
        noise_backend = add_environmental_noise
    api_key = api_key or (get_api_key(api_env) if api_env is not None else get_api_key())
    if limit is not None or kind != 'wav' or output_dir is not None or (selection and manifest != selection):
        if not selection and limit is not None:
            manifest = batch_dir / f'evaluation_with_audio_settings{limit_suffix}.json'
        write_json(manifest, [record for _, record in pending])
    report_path = batch_dir / f'minimax_synthesis_report{suffix}.json'
    report = []
    write_json(report_path, report)
    completed = {}

    def save_result(position, result):
        index = pending[position][0]
        completed[index] = {'record_idx': index, **result}
        report[:] = [completed[i] for i in sorted(completed)]
        write_json(report_path, report)
        detail = ': ' + result['reason'] if result['status'] == 'failed' else ''
        print(f'[{len(completed)}/{len(pending)}] sample {index}: {result["status"]}{detail}', flush=True)

    def speech_progress(count, total, status):
        print(f'[TTS {count}/{total}] {status}', flush=True)

    kwargs = {'synthesize': synthesize} if synthesize is not None else {}
    process_records([record for _, record in pending], config, api_key, concurrency=concurrency,
                    plans=[plan for _, plan in plans], on_record=save_result,
                    on_segment=speech_progress, rpm=rpm, **kwargs)
    if any(result['status'] == 'failed' for result in report):
        print(f'Some samples failed. Inspect {report_path}, then rerun this script to resume.')
        return 1

    # All speech and composition finish before independent background delivery.
    for count, (record, job) in enumerate(all_music, 1):
        print(f'Adding vocal background music [{count}/{len(all_music)}] ...', flush=True)
        try:
            result = music_backend(**job, music_root=music_root, noise_root=noise_root)
            write_receipt(record, job['output_path'], dependencies=result['dependencies'])
            report.append({'stage': 'background_music', 'status': result['status'], **result})
            write_json(report_path, report)
        except Exception as exc:
            report.append({'stage': 'background_music', 'status': 'failed', 'reason': str(exc),
                           'output_path': job['output_path']})
            write_json(report_path, report)
            print(f'Music delivery failed. Inspect {report_path}, then rerun this script.')
            return 1

    for count, (record, job) in enumerate(all_noise, 1):
        print(f'Adding final noise [{count}/{len(all_noise)}] ...', flush=True)
        try:
            noise_backend(**job, noise_root=str(noise_root))
            write_receipt(record, job['output_path'])
        except Exception as exc:
            report.append({'stage': 'noise', 'status': 'failed', 'reason': str(exc),
                           'output_path': job['output_path']})
            write_json(report_path, report)
            print(f'Noise delivery failed. Inspect {report_path}, then rerun this script.')
            return 1

    delivery = finalize_evaluation(manifest, ready_path)
    print(f'Ready: {delivery["ready"]}; rejected: {len(delivery["rejected"])}')
    print(f'Audio files: {batch_dir / "media"}')
    print(f'Evaluation: {ready_path}')
    print(f'Synthesis report: {report_path}')
    return int(bool(delivery['rejected']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-cache-dir', type=Path, default=DEFAULT_CACHE_DIR,
                        help='Query cache directory; defaults to user_query_07.')
    parser.add_argument('--eval-json', type=Path,
                        help='Process this evaluation JSON only; recover scripts from same-directory audio settings.')
    parser.add_argument('--config', type=Path, help='Defaults to shared utils/resources/minimax_voices.json.')
    parser.add_argument('--api-env', choices=('MINIMAX_API', 'MINIMAX_API_2'),
                        help='Select a synthesis account without changing the global process environment.')
    parser.add_argument('--dry-run', action='store_true', help='Validate scripts and voices without API calls.')
    parser.add_argument('--limit', type=int, help='Process only the first N audio samples; save separate trial results.')
    parser.add_argument('--concurrency', type=int, help='Maximum simultaneous speech requests; default from config or 20.')
    parser.add_argument('--rpm', type=int, help='Maximum speech requests per minute, including retries; defaults to voice config.')
    parser.add_argument('--noise-root', type=Path,
                        help='TUT library; defaults to NOISE_ROOT or ../datasets/data/TUT beside the repository.')
    parser.add_argument('--music-root', type=Path, help='JamendoLyrics library for configured vocal music outputs; or MUSIC_ROOT.')
    parser.add_argument('--synthesis-mode', choices=('segment', 'speaker'), help='One request per turn or same-speaker batch.')
    parser.add_argument('--output-dir', type=Path, help='Write fresh manifests and media to a separate directory.')
    args = parser.parse_args(argv)
    try:
        return run_batch(args.input_cache_dir, args.config, dry_run=args.dry_run,
                         noise_root=args.noise_root, limit=args.limit, concurrency=args.concurrency,
                         eval_json=args.eval_json, rpm=args.rpm,
                         synthesis_mode=args.synthesis_mode, output_dir=args.output_dir, music_root=args.music_root,
                         api_env=args.api_env)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('\nInterrupted. Rerun this script to reuse completed speech segments.', file=sys.stderr)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
