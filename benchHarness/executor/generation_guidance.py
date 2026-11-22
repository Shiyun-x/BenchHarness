"""Optional author guidance selected by exact benchmark and subtask IDs."""
from pathlib import Path
import hashlib
import json


_GUIDANCE_ROOT = Path(__file__).resolve().parent / 'guidance'
_TASK_GUIDANCE = {
    'visual_evidence_under_textual_misdirection_v1': 'query_04.md',
    'text_rule_conflicts_exceptions_and_conclusion_revision_v1': 'query_40.md',
}
_SUBTASK_GUIDANCE = {
    ('visual_answerability_and_evidence_boundaries_v1', 'st_02'): 'query_05_st_02.md',
    ('diagram_structure_and_constraint_reasoning_v1', 'st_01'): 'query_10_st_01.md',
}


def load_generation_guidance_file(path, benchmark_task_id, subtask_id, source_mode='original'):
    """Read exactly the selected Markdown/text or JSON file, without discovery or copying."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f'Guidance file does not exist: {path}')
    raw = path.read_bytes()
    text = raw.decode('utf-8-sig')
    profile = path.stem
    construction = None
    if path.suffix.lower() == '.json':
        value = json.loads(text)
        if not isinstance(value, dict) or not isinstance(value.get('instructions'), str):
            raise ValueError('Guidance JSON must contain a string instructions field.')
        for key, expected in (('benchmark_task_id', benchmark_task_id), ('subtask_id', subtask_id),
                              ('source_mode', source_mode)):
            if key in value and value[key] != expected:
                raise ValueError(f'Guidance {key} does not match this run: {value[key]!r} != {expected!r}')
        profile, text = value.get('profile', profile), value['instructions']
        if 'construction' in value:
            from .rule_ledger import validate_assistance
            construction = validate_assistance(value['construction'], benchmark_task_id)
    if not isinstance(profile, str) or not profile.strip() or not text.strip():
        raise ValueError('Guidance profile and instructions must not be empty.')
    guidance = {'profile': profile.strip(), 'instructions': text.strip()}
    if construction is not None:
        guidance['construction'] = construction
    return {'guidance': guidance,
            'source': {'kind': 'file', 'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}}


def load_generation_guidance_map(path, benchmark_task_id):
    """Load {subtask_id: file_path}, resolving relative files beside the map JSON."""
    path = Path(path).expanduser().resolve()
    try:
        mapping = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'Cannot read generation guidance mapping {path}: {exc}') from exc
    if not isinstance(mapping, dict):
        raise ValueError('Generation guidance mapping must be a JSON object of subtask IDs to file paths.')
    guidance = {}
    for subtask_id, filename in mapping.items():
        if not subtask_id.strip() or not isinstance(filename, str) or not filename.strip():
            raise ValueError('Generation guidance mapping requires non-empty subtask IDs and file path strings.')
        guidance_path = Path(filename).expanduser()
        if not guidance_path.is_absolute():
            guidance_path = path.parent / guidance_path
        try:
            guidance[subtask_id] = load_generation_guidance_file(
                guidance_path, benchmark_task_id, subtask_id)['guidance']
        except (OSError, ValueError) as exc:
            raise ValueError(f'Invalid generation guidance for {subtask_id!r} ({guidance_path}): {exc}') from exc
    return guidance


def load_generation_guidance(benchmark_task_id, subtask_id=None):
    """Read on each batch so edits participate in the existing cache signature."""
    filenames = [_TASK_GUIDANCE.get(benchmark_task_id),
                 _SUBTASK_GUIDANCE.get((benchmark_task_id, subtask_id))]
    profiles = []
    for filename in dict.fromkeys(name for name in filenames if name):
        instructions = (_GUIDANCE_ROOT / filename).read_text(encoding='utf-8').strip()
        if not instructions:
            raise ValueError(f'Generation guidance is empty: {filename}')
        profiles.append({'profile': Path(filename).stem, 'instructions': instructions})
    if not profiles:
        return None
    # Retain the exact existing query-only shape/text and cache identity.
    if len(profiles) == 1:
        return profiles[0]
    return {'profile': '+'.join(p['profile'] for p in profiles),
            'instructions': '\n\n'.join(p['instructions'] for p in profiles)}
