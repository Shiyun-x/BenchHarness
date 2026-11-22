"""Deterministic dataset-card views and representative material for Grounding."""
import copy
import hashlib
import json
import re
from pathlib import Path

from utils.audio_processing.audio_jobs import ASSET_AUDIO_TOOLS


def digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def file_version(path):
    try:
        info = Path(path).stat()
        return [info.st_size, info.st_mtime_ns]
    except OSError:
        return None


def manifest_valid(manifest):
    return all(file_version(path) == version for path, version in (manifest or {}).items())


def task_view(task):
    return {key: copy.deepcopy(task[key]) for key in
            ('name', 'description', 'sample_schema', 'answer_type') if key in task}


def asset_tools(context, task):
    """Expose relevant enabled tools actually supported by the planned generator."""
    configured = context.get('tools_list') or []
    if isinstance(configured, dict):
        configured = configured.get('tools', list(configured.values()))
    modalities = task.get('modalities') or context.get('modalities') or []
    modalities = {modalities} if isinstance(modalities, str) else set(modalities)
    result = []
    for tool in configured:
        if not isinstance(tool, dict) or tool.get('name') not in ASSET_AUDIO_TOOLS:
            continue
        supported = tool.get('modalities') or []
        if supported and modalities and not modalities.intersection(supported):
            continue
        item = copy.deepcopy(tool)
        # Cached catalogs may still contain the unused legacy YAML configuration.
        item.pop('memory', None)
        item['execution'] = 'After content generation, execute the media job during batch audio delivery.'
        result.append(item)
    if not any(tool['name'] == 'text2speech' for tool in result):
        result = [tool for tool in result if tool['name'] == 'text2speech']
    return result


def capability_view(tools):
    return [{key: copy.deepcopy(tool[key]) for key in ('name', 'description', 'io', 'execution') if key in tool}
            for tool in tools]


def brief_text(value, limit=240):
    text = ' '.join(str(value or '').split())
    if len(text) <= limit:
        return text
    head = text[:limit - 1]
    return (head.rsplit(' ', 1)[0] if ' ' in head else head).rstrip(' ,;:') + '…'


def candidate_view(row, card):
    meta = card.get('meta') or {}
    # A short material-oriented catalog entry; assess/plan read the full card and samples.
    source = meta.get('meta_structure_description') or card.get('description') or ''
    sentence = re.split(r'(?<=[.!?])\s+', ' '.join(str(source).split()), maxsplit=1)[0]
    fields = []
    for role in list(meta.get('fields') or {})[:6]:
        if len(', '.join(fields + [str(role)])) > 64:
            break
        fields.append(str(role))
    suffix = '; fields: ' + ', '.join(fields) if fields else ''
    summary = brief_text(sentence, 240 - len(suffix)) + suffix
    return {'dataset_id': str(row['dataset_id']), 'name': str(card.get('name') or row['dataset_id']),
            'summary': summary.lstrip('; '),
            'matched_by': list(row.get('retrieval_routes') or [])}


def source_identity(cards):
    """Use source versions in the session key without loading every dataset."""
    from tools.shared.dataset_paths import resolve_dataset_source
    versions = {}
    for did, card in cards.items():
        source = (card.get('raw_meta') or {}).get('source_json')
        try:
            path = resolve_dataset_source(source)
            versions[did] = {'path': str(path), 'version': file_version(path)}
        except (OSError, ValueError, TypeError):
            versions[did] = {'path': source, 'version': None}
    return versions


_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
_AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')


def _media_strings(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _media_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _media_strings(child)
    elif isinstance(value, str) and value.lower().split('?')[0].endswith(_IMAGE_EXTENSIONS + _AUDIO_EXTENSIONS):
        yield value


def _replace_paths(value, mapping):
    if isinstance(value, dict):
        return {key: _replace_paths(child, mapping) for key, child in value.items()}
    if isinstance(value, list):
        return [_replace_paths(child, mapping) for child in value]
    return mapping.get(value, value) if isinstance(value, str) else value


def read_material(dataset_id, card, task, *, sample_count=3, max_chars=18000):
    """Read stable, distributed samples, retaining full QA and explicit view scope."""
    from tools.shared.dataset_paths import resolve_dataset_source
    from tools.shared.media_paths import resolve_image_paths
    from benchHarness.executor.generation_sources import build_source_view
    source = resolve_dataset_source((card.get('raw_meta') or {}).get('source_json'))
    raw = json.loads(source.read_text(encoding='utf-8-sig'))
    rows = raw if isinstance(raw, list) else raw.get('data', raw.get('samples')) if isinstance(raw, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError('Source must contain a nonempty data/samples list.')
    count = min(max(1, int(sample_count)), len(rows))
    indexes = list(dict.fromkeys(round(i * (len(rows) - 1) / max(1, count - 1)) for i in range(count)))
    samples = {}
    scope, images, manifest = [], [], {str(source): file_version(source)}
    for index in indexes:
        sample = rows[index]
        if not isinstance(sample, dict):
            raise ValueError(f'Source sample {index} is not an object.')
        media = list(dict.fromkeys(_media_strings(sample)))
        image_paths = [path for path in media if path.lower().split('?')[0].endswith(_IMAGE_EXTENSIONS)]
        resolved = resolve_image_paths(image_paths, dataset_json_path=str(source), dataset_id=dataset_id)
        mapping = dict(zip(image_paths, resolved)) if len(resolved) == len(image_paths) else {}
        sample = _replace_paths(sample, mapping)
        for path in _media_strings(sample):
            if path.lower().split('?')[0].endswith(_AUDIO_EXTENSIONS):
                scope.append({'sample_id': f's{index}', 'unread_audio': path,
                              'reason': 'Audio is not decoded here; accompanying text/annotations are available.'})
            elif path.startswith(('http://', 'https://')) or Path(path).is_file():
                images.append(path)
                if not path.startswith(('http://', 'https://')):
                    manifest[path] = file_version(path)
            else:
                scope.append({'sample_id': f's{index}', 'unread_image': path, 'reason': 'Image file unavailable.'})
        view, selection = build_source_view(sample, task, force_excerpt=True,
                                             passage_budget_chars=max(2000, max_chars // count - 1500))
        samples[f's{index}'] = view
        if selection:
            scope.append({'sample_id': f's{index}', 'source_view': selection})
    # Keep a complete representative sample rather than silently cutting JSON/QA.
    while len(json.dumps(samples, ensure_ascii=False)) > max_chars and len(samples) > 1:
        removed = next(reversed(samples))
        samples.pop(removed)
        scope.append({'omitted_sample': removed, 'reason': 'Material request budget.'})
    if len(json.dumps(samples, ensure_ascii=False)) > max_chars:
        raise ValueError('Representative material exceeds the request budget; cannot truncate QA safely.')
    meta = card.get('meta') or {}
    material = {'dataset_id': dataset_id, 'card': {
        key: copy.deepcopy(card[key]) for key in ('name', 'description', 'tasks', 'modalities') if card.get(key)}}
    if meta.get('fields'):
        material['card']['fields'] = copy.deepcopy(meta['fields'])
    if meta.get('meta_structure_description'):
        material['card']['meta_structure_description'] = meta['meta_structure_description']
    material['samples'] = samples
    if scope:
        material['scope'] = scope
    retained = set(_media_strings(samples))
    images = list(dict.fromkeys(path for path in images if path in retained))
    return {'material': material, 'images': images, 'manifest': manifest,
            'sample_indexes': [int(key[1:]) for key in samples]}


def resolve_reference(root, path):
    if not isinstance(path, str):
        raise ValueError('A material reference must be a string.')
    value = root
    for component in path.split('.'):
        if isinstance(value, dict) and component in value:
            value = value[component]
        elif isinstance(value, list) and component.isdigit() and int(component) < len(value):
            value = value[int(component)]
        else:
            raise ValueError(f'Unknown material reference: {path}')
    return value


def material_reference_examples(material, limit=2):
    """Small, real path examples; do not enumerate the full source schema in the prompt."""
    examples = []
    for sample_id, sample in (material.get('samples') or {}).items():
        if not isinstance(sample, dict):
            continue
        for field, value in sample.items():
            path = f'material.samples.{sample_id}.{field}'
            while isinstance(value, (dict, list)) and value:
                key = next(iter(value)) if isinstance(value, dict) else 0
                path += f'.{key}'
                value = value[key]
            try:
                resolve_reference({'material': material}, path)
            except ValueError:
                continue
            examples.append(path)
            if len(examples) >= limit:
                return examples
    return examples


def normalize_material_reference(reference, material):
    """Normalize path notation only, then require the exact target to exist."""
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError('Each refs entry must be a material path string, not prose or an object.')
    path = reference.strip()
    if len(path) > 1 and path[0] == path[-1] and path[0] in ('`', '"', "'"):
        path = path[1:-1].strip()
    if path.startswith('/'):
        path = path[1:].replace('/', '.')
    if path.startswith('$'):
        path = path[1:]
    path = re.sub(r'''\[\s*(?:"([\w-]+)"|'([\w-]+)'|(\d+))\s*\]''',
                  lambda match: '.' + next(group for group in match.groups() if group is not None), path)
    path = path.removeprefix('.')
    if any(char in path for char in ('[', ']', '*')):
        raise ValueError(f'Invalid material reference {reference!r}; use dot paths with numeric array indexes.')
    head = path.split('.', 1)[0]
    if head in ('samples', 'card', 'scope', 'dataset_id'):
        path = 'material.' + path
    elif head != 'material' and head in (material.get('samples') or {}):
        path = 'material.samples.' + path
    if not path.startswith('material.'):
        examples = json.dumps(material_reference_examples(material), ensure_ascii=False)
        raise ValueError(f'Invalid material reference {reference!r}; use a full path rooted at material, '
                         f'for example {examples}. Do not cite bare schema paths such as input.context.')
    resolve_reference({'material': material}, path)
    return path
