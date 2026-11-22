"""Inspection response protocol checks; never infer or rewrite model judgments."""
from __future__ import annotations

INSPECTION_PROTOCOL_VERSION = 'inspection-v6'
REASONING_LEVELS = {'direct_lookup', 'local_fact_check', 'relational_inference', 'integrated_inference'}
QUALITY_GAPS = {'reasoning_dependency', 'distractor_hardness'}


def _validate_legacy_inspection_response(value, review_mode, answer_type):
    """Return a concise error for invalid structure/declared contradictions, else None.

    This does not measure difficulty or discover omitted defects. The model must
    ground those judgments; validation only keeps its report and action coherent.
    """
    def text(value):
        return isinstance(value, str) and bool(value.strip())

    required = {'quality_assessment', 'decision', 'issues', 'refinement_plan', 'minor_fixes', 'reason'}
    if not isinstance(value, dict) or set(value) != required:
        return 'inspection requires exactly six top-level keys: ' + ', '.join(sorted(required))
    if value['decision'] not in ('accept', 'revise', 'reject', 'inconclusive'):
        return 'inspection decision must be accept, revise, reject or inconclusive'
    if not text(value['reason']) or not isinstance(value['issues'], list) or not isinstance(value['minor_fixes'], list):
        return 'inspection requires a nonempty reason and issues/minor_fixes arrays'
    quality = value['quality_assessment']
    quality_keys = {'reasoning_necessity', 'distractor_hardness', 'evidence_sufficiency', 'shortcut_test', 'gaps'}
    if not isinstance(quality, dict) or set(quality) != quality_keys:
        return 'quality_assessment requires five sibling fields; do not nest decision/issues or hardness in reasoning_necessity'
    reasoning, hardness, shortcut = (quality[k] for k in ('reasoning_necessity', 'distractor_hardness', 'shortcut_test'))
    for name, node in (('reasoning_necessity', reasoning), ('distractor_hardness', hardness)):
        if not isinstance(node, dict) or set(node) != {'level', 'evidence'} or not text(node['evidence']):
            return name + ' requires only level and a nonempty evidence string'
    if not isinstance(reasoning['level'], str) or reasoning['level'] not in REASONING_LEVELS:
        return 'invalid reasoning_necessity.level'
    levels = {'low', 'moderate', 'high'} if answer_type == 'choice' else {'not_applicable'}
    if not isinstance(hardness['level'], str) or hardness['level'] not in levels:
        return 'invalid distractor_hardness.level for this answer type'
    if not text(quality['evidence_sufficiency']):
        return 'evidence_sufficiency requires a nonempty string'
    if (not isinstance(shortcut, dict) or set(shortcut) !=
            {'minimal_evidence', 'relation_required', 'bypasses_target_relation'} or
            not text(shortcut['minimal_evidence']) or not text(shortcut['relation_required']) or
            not isinstance(shortcut['bypasses_target_relation'], bool)):
        return 'shortcut_test requires minimal_evidence, relation_required strings and bypasses_target_relation boolean'
    if shortcut['bypasses_target_relation'] and reasoning['level'] in ('relational_inference', 'integrated_inference'):
        return 'a solution bypassing the target relation cannot be rated relational/integrated_inference'
    gaps = quality['gaps']
    if (not isinstance(gaps, list) or any(not isinstance(g, str) or g not in QUALITY_GAPS for g in gaps) or
            len(set(gaps)) != len(gaps)):
        return 'gaps must be a unique list of reasoning_dependency/distractor_hardness defects, or []'
    if answer_type != 'choice' and 'distractor_hardness' in gaps:
        return 'non-choice tasks cannot have a distractor_hardness gap'
    ids = set()
    issue_keys = {'id', 'code', 'path', 'problem', 'impact', 'suggested_change', 'preserve',
                  'acceptance_check', 'evidence_refs', 'remedy_scope'}

    def issues_error(issues, actionable=False):
        for issue in issues:
            if not isinstance(issue, dict) or not issue_keys.issubset(issue):
                return 'every issue/change requires id, code, path, problem, impact, suggested_change, preserve, acceptance_check, evidence_refs, remedy_scope'
            if any(not text(issue[k]) for k in ('id', 'code', 'path', 'problem', 'impact')):
                return 'issue identity/path/problem/impact must be nonempty strings'
            if issue['id'] in ids:
                return 'issue/change ids must be unique'
            ids.add(issue['id'])
            if issue['remedy_scope'] not in ('format', 'local_content', 'redesign', 'none'):
                return 'invalid issue remedy_scope'
            for key in ('preserve', 'evidence_refs'):
                if not isinstance(issue[key], list) or any(not text(v) for v in issue[key]):
                    return 'issue ' + key + ' must be a string array'
            if not isinstance(issue['suggested_change'], str) or not isinstance(issue['acceptance_check'], str):
                return 'issue suggested_change/acceptance_check must be strings'
            if actionable and (issue['remedy_scope'] == 'none' or not text(issue['suggested_change']) or not text(issue['acceptance_check'])):
                return 'actionable issue requires a supported change and observable acceptance_check'
        return None

    decision = value['decision']
    error = issues_error(value['issues'], actionable=decision == 'revise')
    if error:
        return error
    if decision == 'accept' and value['issues']:
        return 'inspection accepted with unresolved issues'
    if decision == 'revise' and not value['issues']:
        return 'revise requires at least one actionable issue'
    if review_mode == 'refinement_strategy':
        if value['minor_fixes']:
            return 'initial inspection minor_fixes must be []'
        if decision == 'accept' and value['refinement_plan'] is not None:
            plan = value['refinement_plan']
            if not isinstance(plan, dict) or not text(plan.get('goal')) or not isinstance(plan.get('changes'), list) or not plan['changes']:
                return 'initial inspection missing actionable refinement_plan'
            error = issues_error(plan['changes'], actionable=True)
            if error:
                return error
        elif value['refinement_plan'] is not None:
            return 'non-accept initial review uses issues only; refinement_plan must be null'
        if gaps and decision in ('accept', 'revise'):
            if decision != 'revise' or not set(gaps).issubset({i['code'] for i in value['issues']}):
                return 'every declared quality gap requires a matching actionable issue in initial revise'
    elif review_mode == 'final_validation':
        if value['refinement_plan'] is not None:
            return 'final inspection refinement_plan must be null'
        if gaps and decision in ('accept', 'revise'):
            return 'unresolved reasoning/distractor gaps require final rejection, not acceptance or minor fixes'
        if decision == 'revise' and not value['minor_fixes']:
            return 'final revise requires explicit minor_fixes'
        if decision != 'revise' and value['minor_fixes']:
            return 'minor_fixes are allowed only for final revise'
    else:
        return 'invalid inspection review_mode'
    return None


def _validate_compact_inspection(value, review_mode):
    """Keep only actionable information; do not judge creativity from keywords."""
    def text(value):
        return isinstance(value, str) and bool(value.strip())
    if value.get('decision') not in ('accept', 'revise', 'reject', 'inconclusive') or not text(value.get('reason')):
        return 'inspection requires a valid decision and a concrete reason'
    if review_mode == 'refinement_strategy':
        if not {'decision', 'reason', 'changes'}.issubset(value) or not isinstance(value['changes'], list):
            return 'initial inspection requires decision, reason and changes array'
        if value['decision'] == 'revise' and not value['changes']:
            return 'initial revise requires concrete changes'
        for change in value['changes']:
            if not isinstance(change, dict) or not {'path', 'problem', 'change', 'check'}.issubset(change):
                return 'each change requires path, problem, change and check'
            if any(not text(change[k]) for k in ('path', 'problem', 'change', 'check')):
                return 'each change needs a concrete problem, replacement approach and observable check'
            path = change['path']
            if not (path == 'candidate' or path.startswith('candidate.') or
                    path == 'media_jobs' or path.startswith('media_jobs.')):
                return 'changes target candidate or planned media_jobs, not trusted facts/task constraints'
    elif review_mode == 'final_validation':
        if not {'decision', 'reason', 'minor_fixes'}.issubset(value) or not isinstance(value['minor_fixes'], list):
            return 'final inspection requires decision, reason and minor_fixes array'
        fixes = value['minor_fixes']
        if value['decision'] != 'revise' and fixes:
            return 'only final revise may supply minor_fixes'
        if value['decision'] == 'revise' and not 1 <= len(fixes) <= 4:
            return 'final revise requires 1-4 safe exact minor_fixes'
        for fix in fixes:
            if (not isinstance(fix, dict) or not {'kind', 'path', 'before', 'after'}.issubset(fix)):
                return 'minor fix requires kind, path, before, after and optional evidence_refs'
            if fix['kind'] not in ('format', 'typo', 'factual_wording', 'reference_format') or not text(fix['path']):
                return 'invalid minor fix kind/path'
            if (not text(fix['before']) or not isinstance(fix['after'], str) or
                    max(len(fix['before']), len(fix['after'])) > 80 or fix['before'] == fix['after']):
                return 'minor fix must be a changed exact replacement of at most 80 characters'
            refs = fix.get('evidence_refs', [])
            if not isinstance(refs, list) or any(not text(ref) for ref in refs) or fix['kind'] == 'factual_wording' and not refs:
                return 'factual_wording requires existing evidence_refs; other fixes may omit them'
    else:
        return 'invalid inspection review_mode'
    return None


def validate_inspection_response(value, review_mode, answer_type):
    """Compact review format; initial changes feed at most one repair step."""
    if isinstance(value, dict) and ('changes' in value or 'minor_fixes' in value and 'quality_assessment' not in value):
        return _validate_compact_inspection(value, review_mode)
    return _validate_legacy_inspection_response(value, review_mode, answer_type)


def normalize_inspection_response(value, review_mode):
    """Map concise model output to existing controller fields, without new claims.

    Checkpoints keep the original small JSON. Derived ids/scopes are controller
    bookkeeping; no synthetic difficulty ratings or evidence are introduced.
    """
    import copy
    value = copy.deepcopy(value)
    if 'changes' not in value and ('minor_fixes' not in value or 'quality_assessment' in value):
        return value
    result = {'decision': value['decision'], 'reason': value['reason'],
              'issues': [], 'refinement_plan': None, 'minor_fixes': []}
    if review_mode == 'refinement_strategy':
        changes = [{'id': f'change_{i}', 'code': 'quality_improvement', 'path': change['path'],
                    'problem': change['problem'], 'suggested_change': change['change'],
                    'acceptance_check': change['check'], 'remedy_scope': 'local_content'}
                   for i, change in enumerate(value['changes'])]
        if value['decision'] == 'accept' and changes:
            result['refinement_plan'] = {'goal': value['reason'], 'changes': changes}
        else:
            result['issues'] = changes
    else:
        for i, fix in enumerate(value['minor_fixes']):
            issue_id = f'minor_{i}'
            result['issues'].append({'id': issue_id, 'code': fix['kind'], 'path': fix['path'],
                'problem': value['reason'], 'remedy_scope': 'format' if fix['kind'] in ('format', 'reference_format') else 'local_content'})
            result['minor_fixes'].append({**fix, 'issue_id': issue_id})
    return result
