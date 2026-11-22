"""Lossless source handling and stage-specific views of generation context."""
from __future__ import annotations

import copy
import ntpath
from pathlib import Path

from utils.audio_processing.audio_jobs import ASSET_AUDIO_TOOLS

PAYLOAD_POLICY_VERSION = 'role-focused-v18-image-mapping'
_DERIVED_POLICIES = {
    'source_policy': 'Read original content with its correct QA and annotations as trusted supervision; wrong choices are not facts.',
    'media_phase': 'Audio is synthesized later in batch. Content acceptance is not delivery acceptance.',
}


def image_attachment_manifest(images, payload):
    """Map actual attachment positions to source and solver-visible image fields.

    Review receives source images in source order, even when the candidate uses a
    reordered subset. The candidate's image index is therefore not its attachment
    index. Matching never makes an omitted source image solver-visible.
    """
    def identity(value):
        if value.startswith(('http://', 'https://', 'data:')):
            return value
        return ntpath.normcase(ntpath.normpath(value))

    def references(value, prefix, found):
        if isinstance(value, str):
            found.setdefault(identity(value), []).append(prefix)
        elif isinstance(value, dict):
            for key, child in value.items():
                references(child, prefix + '.' + str(key), found)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                references(child, prefix + '.' + str(index), found)

    source_refs, candidate_refs = {}, {}
    references(payload.get('original'), 'original', source_refs)
    candidate = payload.get('candidate')
    if isinstance(candidate, dict):
        references(candidate.get('input'), 'candidate.input', candidate_refs)
    result = []
    for index, image in enumerate(images or []):
        key = identity(str(image))
        visible_refs = candidate_refs.get(key, [])
        result.append({'attachment_index': index + 1, 'path': str(image),
                       'original_refs': source_refs.get(key, []),
                       'candidate_refs': visible_refs,
                       'solver_visible': bool(visible_refs)})
    return result


def media_tool_specs(allowed_tools, definitions=None):
    """Expose only enabled deferred tools; usage/parameters come from the catalog."""
    allowed = set(allowed_tools) & ASSET_AUDIO_TOOLS
    if not allowed:
        return []
    if definitions is None:
        # Direct run_sample callers can supply a custom catalog. Production
        # batches pass their already-loaded tool YAML definitions explicitly.
        from tools.shared.build_gen_input import _load_transformation_tools
        catalog = Path(__file__).resolve().parents[2]/'utils/resources/tools.yaml'
        definitions = _load_transformation_tools({'modalities': ['audio']}, str(catalog))
    result = []
    seen = set()
    for tool in definitions:
        if not isinstance(tool, dict) or tool.get('name') not in allowed or tool['name'] in seen:
            continue
        result.append({k: copy.deepcopy(tool[k]) for k in
                       ('name', 'summary', 'description', 'modalities', 'io', 'behavior', 'params', 'typical_uses')
                       if k in tool})
        seen.add(tool['name'])
    return result


def media_tool_context(stage, allowed_tools, specs, *, media_jobs=None):
    """Project asset capabilities for planners and execution contracts for Construct.

    Review also needs the meaning of effects actually used by those jobs.
    Keep canonical definitions intact for cache identity and custom catalogs.
    """
    allowed = set(allowed_tools) & ASSET_AUDIO_TOOLS
    if stage == 'inspect':
        used = set()
        for job in media_jobs if isinstance(media_jobs, list) else []:
            if not isinstance(job, dict):
                continue
            name = {'tts': 'text2speech', 'overlay_audio': 'overlay_audio'}.get(job.get('kind'))
            if name:
                used.add(name)
            outputs = job.get('outputs')
            for item in [job] + (outputs if isinstance(outputs, list) else []):
                if not isinstance(item, dict):
                    continue
                if item.get('background_music'):
                    used.add('add_background_music')
                if item.get('noise'):
                    used.add('add_environmental_noise')
        effects = {tool['name']: tool['summary'] for tool in specs
                   if tool.get('name') in used & allowed
                   and isinstance(tool.get('summary'), str) and tool['summary'].strip()}
        return {'active_media_effects': effects} if effects else {}
    if stage not in ('proposal', 'research', 'design', 'replan', 'construct'):
        return {}
    result = {'available_media_tools': sorted(allowed)}
    if stage == 'construct':
        # Job envelopes and media reference conventions currently live in
        # typical_uses, so params alone would lose required execution details.
        keys = ('name', 'summary', 'params', 'typical_uses')
        field = 'available_media_tool_specs'
    else:
        # Planning needs the authored capability summary, not backend details.
        # Custom catalogs without summary expose only the tool name; never
        # silently substitute a long description or execution behavior.
        keys = ('name', 'summary')
        field = 'available_media_capabilities'
    entries = [{key: copy.deepcopy(tool[key]) for key in keys if key in tool}
               for tool in specs if tool.get('name') in allowed]
    if entries:
        result[field] = entries
    return result


def compact_dataset_guidance(value):
    """Project optional dataset advice, never an executable operation contract."""
    if isinstance(value, dict):
        source = value.get('blueprint', value)
        if not isinstance(source, dict):
            return {}
        strategy = source.get('strategy')
        tools = source.get('tools', [])
    elif isinstance(value, list):
        # Compatibility with saved Transformability plans. Paths, invocation
        # params and cleanup steps do not belong in authoring instructions.
        suggestions, tools = [], []
        for op in value:
            if not isinstance(op, dict):
                continue
            name = op.get('step_name') or op.get('operation')
            note = op.get('notes') or op.get('description')
            if str(op.get('tool_type', '')).upper() == 'PURE':
                if name not in ('web_search', 'cleanup_interim_fields'):
                    tools.append({'name': name, 'purpose': note})
            elif isinstance(note, str) and note.strip() and note.strip() not in suggestions:
                suggestions.append(note.strip())
        # Only legacy advice is bounded; trusted source material is untouched.
        strategy = ' '.join(suggestions)[:1600].strip()
    else:
        return {}
    result = {}
    if isinstance(strategy, str) and strategy.strip():
        result['strategy'] = strategy.strip()
    uses, seen = [], set()
    for tool in tools if isinstance(tools, list) else []:
        if not isinstance(tool, dict):
            continue
        name = tool.get('name')
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        if name in ('web_search', 'cleanup_interim_fields') or name in seen:
            continue
        use = {'name': name}
        if isinstance(tool.get('purpose'), str) and tool['purpose'].strip():
            use['purpose'] = tool['purpose'].strip()
        uses.append(use)
        seen.add(name)
    if uses:
        result['tools'] = uses
    return result


def design_summary(design, stage='inspect'):
    """Keep task intent; full author plan remains in design checkpoints/metadata."""
    if not isinstance(design, dict):
        return copy.deepcopy(design)
    result = {k: copy.deepcopy(design[k]) for k in
              ('goal', 'source_refs', 'strategy', 'constraints', 'quality_requirements') if k in design}
    quality = design.get('quality_plan')
    if isinstance(quality, dict) and quality.get('reasoning_dependency'):
        result['reasoning_dependency'] = copy.deepcopy(quality['reasoning_dependency'])
    else:
        # Old plans may not yet describe the dependency explicitly.
        result.update({k: copy.deepcopy(design[k]) for k in
                       ('answer_basis', 'item_outline') if k in design})
    if stage in ('construct', 'repair', 'replan') and isinstance(quality, dict):
        result.update({k: copy.deepcopy(quality[k]) for k in
                       ('context_strategy', 'competing_interpretations', 'leakage_guards') if k in quality})
    return result


def validation_issues(issues):
    """Final review needs the defect and acceptance condition, not old edit prose."""
    if not isinstance(issues, list):
        return copy.deepcopy(issues)
    result = []
    for issue in issues:
        if not isinstance(issue, dict):
            result.append(copy.deepcopy(issue))
            continue
        keys = ('id', 'code', 'path', 'problem', 'preserve',
                'acceptance_check', 'evidence_refs', 'remedy_scope')
        item = {k: copy.deepcopy(issue[k]) for k in keys if k in issue}
        if not isinstance(issue.get('acceptance_check'), str) or not issue['acceptance_check'].strip():
            # Never remove the only actionable guidance from a legacy issue.
            item.update({k: copy.deepcopy(issue[k]) for k in
                         ('impact', 'suggested_change') if k in issue})
        result.append(item)
    return result


def prepare_generation_payload(stage, payload):
    """Keep source facts intact; author roles receive only the requested task fields."""
    result = copy.deepcopy(payload)
    # Only the current role may add a media-tool view; never inherit a catalog
    # from another stage or from a saved request.
    for key in ('available_media_tools', 'available_media_tool_specs', 'available_media_capabilities',
                'active_media_effects'):
        result.pop(key, None)
    advice = result.pop('dataset_guidance', result.pop('recipe_hint', None))
    if stage in ('proposal', 'research', 'design'):
        guidance = compact_dataset_guidance(advice)
        if guidance:
            result['dataset_guidance'] = guidance
    contract = result.get('task_contract')
    if isinstance(contract, dict):
        # The full text is injected into system instructions once for every role.
        contract.pop('generation_guidance', None)
        if stage in ('proposal', 'design', 'construct'):
            if contract.get('user_requirements'):
                result['benchmark_goal'] = copy.deepcopy(contract['user_requirements'])
            result['task_contract'] = {key: copy.deepcopy(contract[key]) for key in
                                       ('name', 'description', 'sample_schema', 'answer_type')
                                       if key in contract}
            if stage == 'proposal':
                result.pop('research_state', None)
        for key, known_value in _DERIVED_POLICIES.items():
            if contract.get(key) == known_value:
                contract.pop(key)
    design = result.get('design')
    if stage in ('inspect', 'repair') and design is not None:
        # Preserve intended capability/dependency without copying legacy author
        # audits. Review judges actual content; the plan is intent, not evidence.
        result['design'] = design_summary(design, 'inspect')
    elif stage == 'replan' and design is not None:
        result['design'] = design_summary(design, stage)
    elif stage == 'construct' and isinstance(design, dict):
        # V3 plans already carry one complete strategy. Historical plans keep
        # the route, presentation and alternatives, without duplicate audits.
        result['design'] = design_summary(design, stage)
    if stage == 'repair' and 'issues' in result:
        result['changes'] = improvement_changes(result.pop('issues'))
    if stage == 'inspect':
        if 'previous_issues' in result:
            result['previous_changes'] = improvement_changes(result.pop('previous_issues'))
        result.pop('revision_context', None)
        if result.get('validation_phase') == 'content':
            result.pop('validation_phase')
    if stage == 'replan':
        revision = result.get('revision_context')
        if (isinstance(revision, dict) and revision.get('issues')
                and result.get('failure') == revision['issues']):
            result.pop('failure')  # The same strategy is already present with the draft.
    return result


def improvement_changes(issues):
    """One problem/action/check per change; no repeated ids, impacts or policy."""
    if not isinstance(issues, list):
        return copy.deepcopy(issues)
    result = []
    for issue in issues:
        if not isinstance(issue, dict):
            result.append(copy.deepcopy(issue))
            continue
        item = {'path': issue.get('path', 'candidate'), 'problem': issue.get('problem', ''),
                'change': issue.get('suggested_change', ''), 'check': issue.get('acceptance_check', '')}
        # Keep custom preservation/ref constraints from old clients; never lose
        # their only actionable guidance during compatibility projection.
        for key in ('preserve', 'evidence_refs'):
            if issue.get(key):
                item[key] = copy.deepcopy(issue[key])
        result.append(item)
    return result
