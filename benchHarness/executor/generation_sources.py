"""Exact source excerpts for long inputs; full source/QA remain in checkpoints.

Selection is deterministic and does not invent summaries. This is a request view,
not replacement benchmark evidence: the final candidate must expose its own clues.
"""
from __future__ import annotations
import copy
import math
import re
from collections import Counter

SOURCE_VIEW_POLICY_VERSION = 'source-view-v3'
LONG_SOURCE_CHARS = 12000
PASSAGE_BUDGET_CHARS = 8000
SOURCE_VIEW_GUIDE = "Source view: budget-limited exact passages selected for the active design or task coverage, not the old gold. Full QA/annotations remain. This is incomplete material, not a summary or proof that omitted facts are absent. Use visible passages/verified claims only; if the selected route lacks a premise, change it to a supported route rather than guess. Do not claim to have examined the full source. References still use existing original/evidence paths."
_TEXT_FIELDS = {'context', 'raw_text', 'passage', 'article', 'text', 'document', 'transcript'}
_STOP = set('the a an and or of to in on for from with by is are was were be been as that this these those it its which what who how why does did can could would should has have had at into about'.split())


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def _tokens(text):
    tokens = {token for token in re.findall(r'[^\W_]+', text.casefold())
              if len(token) > 1 and token not in _STOP}
    for run in re.findall(r'[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]+', text):
        tokens.update(run[index:index+2] for index in range(len(run)-1))
    return tokens


def _passages(text, target=1200):
    """Group complete sentences; retain exact offsets, including their whitespace."""
    spans, start = [], 0
    for match in re.finditer(r'(?<=[.!?])\s+|(?<=[。！？])\s*', text):
        end = match.end()
        if end - start >= target:
            spans.append((start, end))
            start = end
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _select(text, query, budget):
    """Rank by active route and distributed coverage, never old gold."""
    spans = _passages(text)
    tokens = [_tokens(text[start:end]) for start, end in spans]
    frequencies = Counter(token for passage in tokens for token in passage)
    query_tokens = _tokens(query)
    scores = [sum(1 + math.log((len(spans) + 1) / (frequencies[token] + 1))
                  for token in passage & query_tokens) for passage in tokens]
    ranked = sorted(range(len(spans)), key=lambda index: (-scores[index], index))
    selected = set()
    def add(index):
        if index < 0 or index >= len(spans) or index in selected:
            return
        proposed = selected | {index}
        used = sum(spans[i][1]-spans[i][0] for i in proposed) + 2 * (len(proposed)-1)
        if used <= budget:
            selected.add(index)
    if ranked and scores[ranked[0]] > 0:
        add(ranked[0])
        add(ranked[0]-1)
        add(ranked[0]+1)
        for index in ranked:
            if scores[index] > 0:
                add(index)
    slots = max(2, min(len(spans), budget // 1200))
    for slot in range(slots):
        add(round(slot * (len(spans)-1) / (slots-1)))
    for index in ranked:
        add(index)
    ordered = [spans[i] for i in sorted(selected)]
    if not ordered:
        return None, []
    return '\n\n'.join(text[start:end] for start, end in ordered), ordered


def build_source_view(original, contract, *, force_excerpt=False, design=None,
                      passage_budget_chars=PASSAGE_BUDGET_CHARS):
    """Full facts by default; excerpt only explicitly or when over budget."""
    view = copy.deepcopy(original)
    if (contract.get('context_policy', 'reconstruct') != 'reconstruct' or
            contract.get('source_view_policy') == 'full' or not isinstance(view.get('input'), dict)
            or not (force_excerpt or contract.get('source_view_policy') == 'excerpt')):
        return view, None
    eligible = [(key, value) for key, value in view['input'].items()
                if key.lower() in _TEXT_FIELDS and isinstance(value, str) and len(value) > LONG_SOURCE_CHARS
                and not value.lstrip().startswith((chr(96)*3, '{', '[', 'def ', 'class ', 'function ',
                                                   'import ', '#include', '<!DOCTYPE', '<html', '<?xml'))]
    if not eligible:
        return view, None
    if isinstance(design, dict):
        query = ' '.join(_strings({k: design[k] for k in ('goal', 'strategy') if k in design}))
        basis = 'active_design'
    else:
        query = ' '.join(_strings({k: contract[k] for k in ('name', 'description', 'user_requirements') if k in contract}))
        basis = 'task_and_coverage'
    budget = max(1000, int(passage_budget_chars) // len(eligible))
    fields = []
    for key, text in eligible:
        excerpt, spans = _select(text, query, budget)
        if excerpt is None or len(excerpt) >= len(text):
            continue
        view['input'][key] = excerpt
        fields.append({'path': 'original.input.' + key, 'full_chars': len(text),
                       'view_chars': len(excerpt), 'spans': [list(span) for span in spans]})
    metadata = {'policy': SOURCE_VIEW_POLICY_VERSION, 'selection_basis': basis, 'fields': fields}
    return view, metadata if fields else None
