"""Run overlay_audio jobs after an external TTS backend has produced the WAV segments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import wave

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.audio_processing.audio_rendering import overlay_audio
from utils.audio_processing.audio_receipts import write_receipt


def overlay_record(record):
    if '_speech_segments' in record:
        outputs = []
        for job in record.get('_audio_outputs', []):
            result = overlay_audio(record['_tts_segments'], job['tracks'], job['output_path'])
            write_receipt(record, job['output_path'])
            outputs.append({'id': job['id'], **result})
        return {'status': 'generated' if outputs else 'skipped', 'outputs': outputs}
    job = record.get('_overlay_audio_args')
    if not job:
        return {'status': 'skipped'}
    result = overlay_audio(job['segments'], job['tracks'], job['output_path'])
    write_receipt(record, job['output_path'])
    return {'status': 'generated', **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-json', required=True, type=Path)
    args = parser.parse_args()
    records = json.loads(args.eval_json.read_text(encoding='utf-8'))
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        parser.error('evaluation must be an array of records')
    failed = 0
    for index, record in enumerate(records):
        try:
            result = overlay_record(record)
            print(f'[{index}] {result["status"]}')
        except (OSError, ValueError, KeyError, wave.Error, EOFError) as exc:
            failed += 1
            print(f'[{index}] failed: {exc}')
    return int(failed > 0)


if __name__ == '__main__':
    raise SystemExit(main())
