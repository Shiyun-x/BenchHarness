"""Resolve relocated dataset-card paths without searching by filename alone."""
from pathlib import Path, PurePosixPath


def resolve_dataset_source(source, project_root=None):
    if not isinstance(source, str) or not source.strip():
        raise FileNotFoundError('dataset card raw_meta.source_json is missing')
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    path = Path(source)
    direct = path if path.is_absolute() else root / path
    if direct.is_file():
        return direct.resolve()
    parts = PurePosixPath(source.replace(chr(92), '/')).parts
    tail = None
    for marker in ('General-Bench-Openset', 'datasets'):
        if marker in parts:
            tail = parts[parts.index(marker)+1:]
            break
    candidates = [direct]
    if tail and tail[0] in ('nlp', 'image') and '..' not in tail:
        candidates.extend([
            root / 'data' / 'datasets' / Path(*tail),
            root.parent / 'datasets' / 'data' / 'General-Bench-Openset' / Path(*tail),
        ])
    for candidate in candidates[1:]:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f'dataset source not found: {source}; checked: ' + ', '.join(str(p) for p in candidates))
