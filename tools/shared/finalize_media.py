"""Delivery check after the existing batch TTS/noise scripts have run."""
from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

from utils.audio_processing.audio_receipts import check_receipt, delivery_audio_paths
from utils.audio_processing.audio_rendering import read_audio


def finalize_evaluation(input_path, output_path=None):
    source = Path(input_path)
    records = json.loads(source.read_text(encoding='utf-8'))
    if not isinstance(records, list):
        raise ValueError('evaluation input must be a list')
    ready, rejected = [], []
    for record in records:
        if record.get('_media_status') == 'pending':
            try:
                paths = delivery_audio_paths(record)
                if not paths:
                    raise ValueError('batch audio outputs missing')
                for path in paths:
                    if not path or not Path(path).is_file():
                        raise ValueError('batch audio not produced')
                    check_receipt(record, path)
                    read_audio(path, expected_format=Path(path).suffix.lower().lstrip('.'))
                record['_media_status'] = 'ready'
                record['_validation_phase'] = 'delivery'
            except (OSError, ValueError, wave.Error, EOFError) as exc:
                rejected.append({'subtask_id': record.get('subtask_id'), 'dataset_id': record.get('dataset_id'),
                                 'idx': record.get('idx'), 'reason': str(exc)})
                continue
        public_record = {k: v for k, v in record.items() if not k.startswith('_')}
        if '_speech_segments' in record:
            public_record['transcript'] = record['_speech_segments']
        elif '_transcript' in record:
            public_record['transcript'] = record['_transcript']
        ready.append(public_record)
    output = Path(output_path) if output_path else source.with_name('evaluation_ready.json')
    if output.resolve() == source.resolve():
        raise ValueError('output must not overwrite the media job manifest')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(ready, ensure_ascii=False, indent=2), encoding='utf-8')
    report = {'ready': len(ready), 'rejected': rejected,
              'unchecked_properties': ['actual_audio_semantics'] if any(
                  r.get('_transcript') or r.get('_speech_segments') for r in records) else []}
    output.with_suffix('.report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description='Check batch-generated WAV/FLAC artifacts and export ready records')
    parser.add_argument('--eval-json', required=True)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    report = finalize_evaluation(args.eval_json, args.output)
    print(f'Ready: {report["ready"]}; missing/invalid audio: {len(report["rejected"])}')


if __name__ == '__main__':
    main()
