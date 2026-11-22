"""Shared audio-library locations for batch delivery and standalone tools."""
from __future__ import annotations

import os
from pathlib import Path


DEFAULT_NOISE_ROOT = Path(__file__).resolve().parents[2].parent / 'datasets' / 'data' / 'TUT'


def resolve_noise_root(noise_root=None):
    """Explicit argument, then NOISE_ROOT, then the sibling datasets directory."""
    return Path(noise_root or os.environ.get('NOISE_ROOT') or DEFAULT_NOISE_ROOT).resolve()
