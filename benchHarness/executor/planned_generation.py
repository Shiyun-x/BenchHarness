"""Research alternatives, synthesize a design, construct, optionally refine and inspect.

Audio synthesis runs later. Accepted content is queued for batch TTS.
Model/search clients are injectable so control flow can be tested offline.
"""
from __future__ import annotations

import copy
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from uuid import uuid4
from utils.model_config import load_model_config

from utils.audio_processing.audio_jobs import ASSET_AUDIO_TOOLS, canonicalize_media_jobs

from .generation_contracts import (fingerprint, get_path, check_content, fields, media_fields,
                                   normalize_candidate, bind_media, shuffle_choices, normalize_reference,
                                   final_minor_edit_fields, original_reference_examples, has_modality)
from .generation_prompts import (PROMPT_VERSION, PROPOSAL_PROMPT, RESEARCH_PROMPT, DESIGN_PROMPT, CONSTRUCT_PROMPT, INSPECT_PROMPT,
                                 REPAIR_PROMPT, REPLAN_PROMPT)
from .generation_research import (ResearchAgent, DEFAULT_SEARCH_BUDGET, RESEARCH_POLICY_VERSION,
                                  native_evidence, NATIVE_EVIDENCE_POLICY_VERSION)
from .generation_quality import build_generation_prompt
from .generation_guidance import load_generation_guidance
from .generation_responses import (validate_generation_response, normalize_generation_response,
                                   GENERATION_RESPONSE_POLICY_VERSION)
from .generation_sources import build_source_view, SOURCE_VIEW_GUIDE, SOURCE_VIEW_POLICY_VERSION
from .generation_payloads import (prepare_generation_payload, compact_dataset_guidance,
                                  media_tool_specs, media_tool_context, PAYLOAD_POLICY_VERSION)
from .generation_inspection import (validate_inspection_response, normalize_inspection_response,
                                    INSPECTION_PROTOCOL_VERSION)
from .generation_progress import GenerationProgress

VERSION = 'planned-v1'
GENERATION_FLOW_VERSION = 'native-proposal-synthesis-v3'
CHECKPOINT_VERSION = 3

# The v1.3 prompt change extends deferred audio only. Keep this exact v1.2
# instruction to verify old non-audio signatures, never to relabel old results.
_V12_MEDIA_CONTENT = '''Media execution: reuse original media; no external image acquisition or replacement is available. Never invent files. TTS/noise run last in batch. For deferred audio put media:audio in the schema audio field; add a top-level media_jobs array with one entry {"kind":"tts","dialog":[{"text":"<utterance>","speaker":"<speaker>","language":"<language>","gender":"<gender>"}],"noise":null}. Optional noise has noise_type/intensity. Update dialogue/jobs with changed QA; actual synthesized audio remains unchecked.'''


def _compatible_v12_signature(signature_input):
    """Allow only the known audio-only prompt update for non-audio tasks."""
    from .generation_prompts import MEDIA_CONTENT
    contract = signature_input['contract']
    if (signature_input['prompt_version'] != 'v1.3'
            or 'audio' in contract.get('modalities', [])
            or media_fields(contract, 'audio')):
        return None
    previous = {**signature_input, 'prompt_version': 'v1.2',
                'tools': [tool for tool in signature_input['tools'] if tool != 'overlay_audio'],
                'prompts': [prompt.replace(MEDIA_CONTENT, _V12_MEDIA_CONTENT)
                            for prompt in signature_input['prompts']]}
    return fingerprint(previous)


def _read_accepted_cache(path, signature):
    try:
        cached = json.loads(path.read_text(encoding='utf-8'))
        result = cached.get('result') if isinstance(cached, dict) else None
        if (cached.get('signature') == signature and isinstance(result, dict)
                and result.get('status') == 'accepted'
                and cached.get('result_digest') == fingerprint(result)):
            return result
    except (ValueError, OSError, AttributeError, TypeError):
        pass
    return None


class GenerationInfrastructureError(RuntimeError):
    """Configuration/authentication failure; changing source samples cannot fix it."""


class GenerationProtocolError(RuntimeError):
    """Malformed stage response; retry the call rather than discard the source."""


class InspectionProtocolError(GenerationProtocolError):
    """Invalid inspection response; retry the call, not the source transformation."""


def task_contract(subtask, user_requirements='', *, benchmark_task_id=None, generation_guidance=None):
    schema = subtask.get('sample_schema')
    if not isinstance(schema, dict) or not schema.get('input') or not schema.get('output'):
        raise ValueError('subtask needs complete input/output sample_schema')
    contract = {k: copy.deepcopy(subtask[k]) for k in (
        'name', 'description', 'answer_type', 'modalities', 'sample_schema',
        'difficulty', 'option_count', 'quality_requirements',
        'external_knowledge_policy', 'synthetic_content_policy',
        'context_policy', 'require_independent_sources', 'source_view_policy') if k in subtask}
    if 'evaluation_target' in subtask:
        contract['evaluation_target'] = copy.deepcopy(subtask['evaluation_target'])
    # Cached scalar/list schemas must not force a sample's media count.
    for name in media_fields(contract, 'image') + media_fields(contract, 'audio'):
        spec = fields(contract['sample_schema']['input'])[name]
        if isinstance(spec, dict):
            spec['type'] = 'str|list'
    contract.setdefault('modalities', ['text'])
    contract.setdefault('context_policy', 'reconstruct')
    contract.setdefault('require_independent_sources', False)
    contract['user_requirements'] = user_requirements
    contract['source_policy'] = 'Read original content with its correct QA and annotations as trusted supervision; wrong choices are not facts.'
    contract['media_phase'] = 'Audio is synthesized later in batch. Content acceptance is not delivery acceptance.'
    guidance = (generation_guidance if generation_guidance is not None else
                load_generation_guidance(benchmark_task_id, subtask.get('id')))
    if guidance:
        contract['generation_guidance'] = copy.deepcopy(guidance)
        if guidance.get('construction') is not None:
            from .rule_ledger import validate_assistance
            contract['generation_guidance']['construction'] = validate_assistance(
                guidance['construction'], benchmark_task_id)
    return contract


def recipe_hint(plan):
    """Keep semantic instructions, omit old controller/path machinery."""
    if not isinstance(plan, list):
        raise ValueError('tool_plan must be a list')
    result = []
    for op in plan:
        if not isinstance(op, dict):
            raise ValueError('tool_plan operation must be an object')
        name = op.get('step_name') or op.get('operation')
        if str(op.get('tool_type', '')).upper() == 'PURE' and name not in (
            'web_search', 'text2speech', 'overlay_audio', 'add_environmental_noise',
            'add_background_music', 'cleanup_interim_fields'):
            raise ValueError(f'planned path does not support required tool: {name}')
        result.append({k: op[k] for k in
                       ('step_name', 'operation', 'tool_type', 'description', 'notes') if k in op})
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A deep checkpoint may fit Windows MAX_PATH while its old temp name did not.
    temporary = path.with_name(uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class StageCheckpoints:
    """Atomic per-call records; failed calls are recorded but never replayed."""

    def __init__(self, directory, namespace, resume=True):
        self.directory = Path(directory) if directory is not None else None
        self.namespace = namespace
        self.resume = resume

    def execute(self, name, request, invoke, reusable):
        if self.directory is None:
            return invoke(), False
        path = self.directory / (name + '.json')
        signature = fingerprint({'version': CHECKPOINT_VERSION,
                                 'namespace': self.namespace, 'request': request})
        if self.resume and path.is_file():
            try:
                cached = json.loads(path.read_text(encoding='utf-8'))
                if (isinstance(cached, dict) and cached.get('signature') == signature
                        and cached.get('reusable') is True
                        and cached.get('result_digest') == fingerprint(cached.get('result'))
                        and reusable(cached.get('result'))):
                    return copy.deepcopy(cached['result']), True
            except (OSError, ValueError, TypeError, KeyError):
                pass
        record = {'version': CHECKPOINT_VERSION, 'signature': signature,
                  'request': request, 'completed_at': time.time()}
        try:
            result = invoke()
        except Exception as exc:
            record.update(reusable=False, error_type=type(exc).__name__)
            write_json(path, record)
            raise
        record.update(result=result, result_digest=fingerprint(result),
                      reusable=bool(reusable(result)), completed_at=time.time())
        write_json(path, record)
        return result, False


def _validate_design(design, original, allowed_tools, search_budget, evidence=None):
    if not isinstance(design, dict) or not isinstance(design.get('goal'), str) or not design['goal'].strip():
        raise ValueError('design requires a concrete goal')
    design = copy.deepcopy(design)
    refs = design.get('source_refs')
    if isinstance(refs, list):
        refs = [normalize_reference(ref) for ref in refs]
        design['source_refs'] = refs
    root = {'original': original, 'evidence': evidence or {}}

    def valid_ref(ref):
        if not isinstance(ref, str):
            return False
        if ref.startswith('original.'):
            return get_path(root, ref) is not None
        parts = ref.split('.')
        return (evidence is not None and len(parts) >= 4 and
                parts[0] == 'evidence' and parts[2] == 'claims' and
                parts[3].isdigit() and get_path(root, ref) is not None)

    if not isinstance(refs, list) or not refs:
        raise ValueError('design requires valid original or supported evidence claim source_refs')
    invalid = [r for r in refs if not valid_ref(r)]
    if invalid:
        bad = json.dumps(invalid[:3], ensure_ascii=False, default=str)[:300]
        examples = json.dumps(original_reference_examples(original), ensure_ascii=False)
        raise ValueError('design requires valid original or supported evidence claim source_refs; '
                         f'invalid paths: {bad}. Existing original examples: {examples}. '
                         'Use dot paths with numeric list indexes, e.g. .0; missing fields and '
                         'out-of-range indexes cannot be repaired by changing facts.')
    if not any(r.startswith('original.') for r in refs):
        raise ValueError('design requires at least one original source_refs anchor')
    requests = design.get('evidence_requests', [])
    if not isinstance(requests, list) or len(requests) > search_budget:
        raise ValueError('evidence requests exceed search budget')
    ids = set()
    for request in requests:
        if 'web_search' not in allowed_tools:
            raise ValueError('web_search is not registered/enabled')
        if not isinstance(request, dict) or not isinstance(request.get('query'), str) or not request['query'].strip():
            raise ValueError('evidence request requires a query')
        key = request.get('id')
        if not isinstance(key, str) or not key.isidentifier() or key in ids:
            raise ValueError('evidence request ids must be unique identifiers')
        ids.add(key)
        if not isinstance(request.get('image_paths', []), list):
            raise ValueError('image_paths must be a list')
    return design



def _apply_final_minor_fixes(candidate, support, inspection, reference_root, contract):
    """Apply explicit small edits already prescribed by the single final review."""
    issues, fixes = inspection.get('issues'), inspection.get('minor_fixes')
    if (not isinstance(issues, list) or not issues or
            any(not isinstance(i, dict) or i.get('remedy_scope') not in ('format', 'local_content')
                for i in issues) or not isinstance(fixes, list) or not 1 <= len(fixes) <= 4):
        raise ValueError('final review requires major changes or lacks explicit minor fixes')
    ids = [i.get('id') for i in issues]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('final minor fixes require unique issue ids')
    root = {'candidate': copy.deepcopy(candidate), 'support': copy.deepcopy(support)}
    covered = set()
    for fix in fixes:
        if not isinstance(fix, dict) or fix.get('issue_id') not in ids:
            raise ValueError('minor fix must identify a reviewed issue')
        path = fix.get('path')
        if not isinstance(path, str):
            raise ValueError('minor fix requires an existing path')
        path = normalize_reference(path)
        parts = path.split('.')
        is_text = (len(parts) >= 3 and parts[:2] == ['candidate', 'input'] and
                   parts[2] in final_minor_edit_fields(contract) and
                   (len(parts) == 3 or len(parts) == 4 and parts[3].isdigit()))
        is_ref = (len(parts) == 3 and parts[:2] == ['support', 'answer_refs'] and parts[2].isdigit())
        kind = fix.get('kind')
        if not (is_text and kind in ('format', 'typo', 'factual_wording') or
                is_ref and kind == 'reference_format'):
            raise ValueError('minor fix exceeds allowed text/reference scope')
        before, after, value = fix.get('before'), fix.get('after'), get_path(root, path)
        if (not isinstance(value, str) or not isinstance(before, str) or not before or
                not isinstance(after, str) or max(len(before), len(after)) > 80 or
                value.count(before) != 1 or before == after):
            raise ValueError('minor fix requires a unique short exact replacement')
        if kind == 'factual_wording':
            refs = fix.get('evidence_refs')
            if not isinstance(refs, list) or not refs or any(
                not isinstance(r, str) or not normalize_reference(r).startswith(('original.', 'evidence.')) or
                get_path(reference_root, r) is None for r in refs):
                raise ValueError('factual wording correction requires existing evidence refs')
        updated = value.replace(before, after, 1)
        if is_ref and (value != before or not updated.startswith(('original.', 'evidence.')) or
                       get_path(reference_root, updated) is None):
            raise ValueError('reference correction must resolve to existing evidence')
        parent = get_path(root, '.'.join(parts[:-1]))
        if isinstance(parent, list):
            parent[int(parts[-1])] = updated
        elif isinstance(parent, dict):
            parent[parts[-1]] = updated
        else:
            raise ValueError('minor fix parent is unavailable')
        covered.add(fix['issue_id'])
    if covered != set(ids):
        raise ValueError('not all final issues have explicit minor fixes')
    return root['candidate'], root['support']


def _source_images(value):
    result = []
    if isinstance(value, dict):
        for child in value.values():
            result.extend(_source_images(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(_source_images(child))
    elif isinstance(value, str) and value.lower().split('?')[0].endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
        result.append(value)
    return list(dict.fromkeys(result))


def resolve_images(original, source_json, dataset_id):
    from tools.shared.media_paths import resolve_image_paths
    original = copy.deepcopy(original)
    image_refs = _source_images(original)
    resolved = resolve_image_paths(image_refs, dataset_json_path=source_json, dataset_id=dataset_id)
    mapping = dict(zip(image_refs, resolved)) if len(image_refs) == len(resolved) else {}

    def rewrite(value):
        if isinstance(value, dict):
            return {k: rewrite(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rewrite(v) for v in value]
        return mapping.get(value, value) if isinstance(value, str) else value

    return rewrite(original)


def run_sample(original, contract, recipe, output_root, item_key, models,
               allowed_tools=('web_search',), search_budget=DEFAULT_SEARCH_BUDGET, llm=None, search=None,
               model_config_path=None, max_prompt_chars=30000,
               checkpoint_dir=None, resume_checkpoints=True, cache_namespace=None, on_stage=None,
               media_tool_definitions=None, argument_repair=None):
    from tools.executor_tools.implementations.web_evidence import EVIDENCE_POLICY_VERSION, evidence_failure_details
    if llm is None:
        from utils.llm_caller import llm_call_json
        llm = llm_call_json
    if search is None and model_config_path:
        from .generation_research import configured_search
        search = configured_search(model_config_path)
    audio_task = has_modality(contract, 'audio')
    allowed_tools = tuple(tool for tool in allowed_tools if audio_task or tool not in ASSET_AUDIO_TOOLS)
    tool_specs = media_tool_specs(allowed_tools, media_tool_definitions) if audio_task else []
    # Explicit custom retrievers retain their injectable multi-turn adapter.
    # The normal generator uses one native Responses proposal request.
    use_native_research = search is None
    native_research_response = None
    events, evidence = [], {}
    search_failures, source_views = [], {}
    effective_search_budget = max(0, int(search_budget)) if 'web_search' in allowed_tools else 0
    grounding = None
    planning_recovery_used = False
    format_recovery_used = False
    argument_repair_used = False
    argument_repairs = []
    refinement_used = False
    quality_repair_used = False
    initial_inspection = None
    checkpoint_phase = ''
    images = _source_images(original)
    if any(not p.startswith(('https://', 'http://')) and not Path(p).is_file() for p in images):
        return {'status': 'rejected', 'reason': 'source image unavailable', 'events': events}
    if not isinstance(original, dict):
        return {'status': 'rejected', 'reason': 'original sample must be an object', 'events': events}
    checkpoints = StageCheckpoints(checkpoint_dir, cache_namespace or {
        'version': VERSION, 'flow': GENERATION_FLOW_VERSION, 'prompt_version': PROMPT_VERSION,
        'item_key': item_key, 'models': models,
        'payload_policy': PAYLOAD_POLICY_VERSION, 'source_view_policy': SOURCE_VIEW_POLICY_VERSION,
        'inspection_protocol': INSPECTION_PROTOCOL_VERSION,
        'evidence_policy': EVIDENCE_POLICY_VERSION,
        'generation_response_policy': GENERATION_RESPONSE_POLICY_VERSION,
        'research_policy': RESEARCH_POLICY_VERSION,
        'research_backend': 'native_responses' if use_native_research else 'external_retriever',
        'native_evidence_policy': NATIVE_EVIDENCE_POLICY_VERSION,
        'model_config_path': str(model_config_path),
        **({'media_tool_specs': tool_specs} if tool_specs else {}),
        'media': {p: [Path(p).stat().st_size, Path(p).stat().st_mtime_ns]
                  for p in images if Path(p).is_file()}}, resume_checkpoints)
    construction = (contract.get('generation_guidance') or {}).get('construction')
    fixture = None
    if construction:
        from .rule_ledger import build_fixture
        max_prompt_chars = max(max_prompt_chars, construction['max_prompt_chars'])
        fixture, fixture_hit = checkpoints.execute('construction_fixture',
            {'settings': construction, 'item_key': item_key, 'fixture_format': 2},
            lambda: build_fixture(construction, item_key),
            lambda value: isinstance(value, dict) and value.get('type') == construction['type'])
        events.append({'stage': 'construction_fixture', 'cache_hit': fixture_hit,
                       'type': fixture['type'], 'seed': fixture['seed'], 'usage': {}})

    def content_errors(candidate, support, jobs, contract, reference_root):
        errors = check_content(candidate, support, jobs, contract, reference_root)
        if fixture is not None and not errors:
            from .rule_ledger import fixture_errors
            errors.extend(fixture_errors(candidate, fixture))
        return errors

    def checkpoint_name(stage):
        name = {'research': 'decision', 'construct': 'construction', 'inspect': 'inspection'}.get(stage, stage)
        return (checkpoint_phase + '/' if checkpoint_phase else '') + name

    def call(stage, prompt, payload, attach_images=False):
        nonlocal native_research_response
        native_proposal = stage == 'research' and use_native_research
        source_plan = payload.get('design')
        payload_chars_before = len(json.dumps(payload, ensure_ascii=False, separators=(',', ':')))
        prompt_stage = 'proposal' if native_proposal else stage
        payload = prepare_generation_payload(prompt_stage, payload)
        if fixture is not None:
            if stage in ('construct', 'repair', 'inspect'):
                payload['construction_fixture'] = copy.deepcopy({key: value for key, value in fixture.items()
                    if key != 'reference_values' and (stage != 'inspect' or key not in ('ledger', 'semantics'))})
            else:
                payload['construction_fixture_available'] = {
                    'type': fixture['type'], 'layers': fixture['layers'],
                    'derived_atoms': fixture['checked_derived_atoms'],
                    'provenance': fixture['provenance']}
        if audio_task:
            payload.update(media_tool_context(prompt_stage, allowed_tools, tool_specs,
                                              media_jobs=payload.get('media_jobs')))
        prompt = build_generation_prompt(prompt_stage, prompt, contract,
                                         review_mode=payload.get('review_mode'),
                                         search_budget=effective_search_budget if native_proposal else None,
                                         available_media_tools=allowed_tools,
                                         has_existing_media_jobs=(isinstance(payload.get('media_jobs'), list)
                                                                  and bool(payload['media_jobs'])))
        if fixture is not None:
            prompt += ('\n\nExecutor construction assistance: a local tool has already generated and replayed '
                       'a complete fictional policy fixture. It is authored scenario material, not a factual '
                       'claim extracted from the original. Keep source facts accurate and make this distinction '
                       'explicit. Proposal/design should plan a source-appropriate presentation; the full '
                       'fixture is supplied to construction, repair and inspection. In construction/repair, '
                       'put the exact marker fixture:rule_ledger once in input.context, with source-related '
                       'presentation outside it. The runtime expands this marker into the full semantics '
                       'and ledger BEFORE inspection; do not manually copy or abbreviate the long block. '
                       'A revised context may also preserve an already expanded block verbatim. '
                       'Use its privately supplied reference '
                       'status to map the CURRENT choice labels. Do not expose private reference_values or '
                       'reference_status as hints. The fixture is already expanded; no manual recreation '
                       'or unsupported claim of independent model verification is required. Review the actual '
                       'candidate, its source presentation, question and gold under the ordinary quality rules; '
                       'fictional stipulated policies do not need to have appeared in the source.')
            if stage == 'inspect':
                prompt += ('\nReview-stage distinction: you receive the FINAL EXPANDED candidate. The marker '
                           'is an author-side transport mechanism and MUST NOT remain in the final input; '
                           'its absence is expected, not an error. construction_fixture/query_audit are '
                           'private author metadata, NOT solver-visible facts. Read FACTS only from the '
                           'candidate ledger. A base->g rule supplies raw support; if base->!g also applies, '
                           'neither is established until conflict resolution. A numerically later '
                           'unconditional priority is not a shortcut: all earlier local guards must fail '
                           'before it controls. Check the ordered audit against the actual ledger. '
                           'Continue to flag real source, wording, semantic, leakage or alignment defects.')
        if stage in ('research', 'design', 'replan'):
            examples = original_reference_examples(payload.get('original', original))
            if native_proposal and examples:
                prompt = prompt.replace('"original.input.<field>"', json.dumps(examples[0]), 1)
            prompt += ('\n\nReference paths: follow the actual original structure; do not assume '
                       'original.input exists. Prefer numeric dot indexes (.0); [0] is an accepted '
                       'equivalent. Existing original examples: ' + json.dumps(examples, ensure_ascii=False) +
                       '. Examples are not an exhaustive list; every referenced field/index must exist.')
        if attach_images and stage in ('inspect', 'repair') and images:
            from .generation_payloads import image_attachment_manifest
            payload['image_attachments'] = image_attachment_manifest(images, payload)
            prompt += ('\n\nImage attachment positions follow image_attachments. Match candidate image '
                       'paths through candidate_refs; a candidate index is not an attachment index. '
                       'Only attachments marked solver_visible support solver-visible evidence. '
                       'Other source images are author context, even when they depict the missing event.')
        original_view, source_view = build_source_view(original, contract, design=source_plan)
        encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        if source_view is None and 'original' in payload and len(prompt) + len(encoded) > max_prompt_chars:
            # Reserve room for metadata/QA; retain as much exact source as this request allows.
            original_chars = len(json.dumps(original, ensure_ascii=False, separators=(',', ':')))
            available_chars = max(1000, max_prompt_chars - len(prompt) - len(SOURCE_VIEW_GUIDE)
                                  - (len(encoded) - original_chars) - 1000)
            original_view, source_view = build_source_view(
                original, contract, force_excerpt=True, design=source_plan,
                passage_budget_chars=available_chars)
        if source_view and 'original' in payload:
            payload['original'] = copy.deepcopy(original_view)
            payload['source_view'] = source_view
            prompt += '\n\n' + SOURCE_VIEW_GUIDE
            source_views[checkpoint_name(stage)] = copy.deepcopy(source_view)
            checkpoints.execute(checkpoint_name(stage) + '_source_view',
                {'original': original, 'selection': source_view},
                lambda: {'original_view': original_view, 'selection': source_view},
                lambda value: isinstance(value, dict) and value.get('selection') == source_view)
        encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        if len(prompt) + len(encoded) > max_prompt_chars:
            details = {'stage': stage, 'input_chars': len(prompt) + len(encoded),
                       'limit_chars': max_prompt_chars, 'source_view': source_view,
                       'system_chars': len(prompt), 'payload_chars': len(encoded)}
            checkpoints.execute(checkpoint_name(stage) + '_budget',
                {'stage': stage, 'details': details, 'payload': payload},
                lambda: {'blocked': True, **details}, lambda value: False)
            raise ValueError(f'prompt budget exceeded at {stage}: {len(prompt)+len(encoded)} > '
                             f'{max_prompt_chars}; source was not silently truncated')
        start = time.monotonic()
        model = models.get(stage, models.get('design', models['construct'])
                           if stage == 'research' else models['construct'])
        max_tokens = 4000
        create_params = {'timeout': 180, 'request_timeout': 180, 'num_retries': 0}
        json_tool_name = None
        if (stage == 'research' and not native_proposal and 'gemini' in model.lower()
                and load_model_config(model_config_path).get('request_options', {}).get('research_decision_tool')):
            from .generation_research import RESEARCH_DECISION_TOOL
            json_tool_name = RESEARCH_DECISION_TOOL['function']['name']
            create_params.update(tools=[copy.deepcopy(RESEARCH_DECISION_TOOL)],
                tool_choice={'type': 'function', 'function': {'name': json_tool_name}})
            prompt += '\nDeliver the specified JSON object as arguments of submit_research_decision.'
        if model.startswith('gpt-'):
            create_params['custom_llm_provider'] = 'openai'
        if native_proposal:
            create_params.pop('custom_llm_provider', None)
            create_params.pop('request_timeout', None)
            if effective_search_budget:
                # The configured gateway rejects max_tool_calls. Keep native
                # search as in the successful standalone probe; constrain it
                # in the prompt and validate the returned trace afterward.
                create_params.update(tools=[{'type': 'web_search', 'search_context_size': 'low'}],
                                     tool_choice='auto', include=['web_search_call.action.sources'])
        request = {'stage': stage, 'system_prompt': prompt, 'payload': payload,
                   'images': images if attach_images else None, 'model': model,
                   'max_attempts': 2, 'max_tokens': max_tokens, 'create_params': create_params,
                   'generation_flow': GENERATION_FLOW_VERSION, 'prompt_version': PROMPT_VERSION,
                   'json_root_type': 'object',
                   'generation_response_policy': GENERATION_RESPONSE_POLICY_VERSION,
                   'payload_policy': PAYLOAD_POLICY_VERSION, 'source_view_policy': SOURCE_VIEW_POLICY_VERSION,
                   'inspection_protocol': INSPECTION_PROTOCOL_VERSION,
                   'api_mode': 'responses' if native_proposal else 'completion',
                   **({'json_tool_name': json_tool_name} if json_tool_name else {}),
                   'input_sizes': {'system_chars': len(prompt), 'payload_chars': len(encoded),
                                   'payload_chars_before': payload_chars_before,
                                   'original_chars_full': len(json.dumps(original, ensure_ascii=False)),
                                   'original_chars_used': len(json.dumps(payload.get('original', original), ensure_ascii=False))}}
        def validator(value):
            state = payload.get('research_state', {})
            error = validate_generation_response(
                value, 'proposal' if native_proposal else stage, payload.get('review_mode'), contract.get('answer_type'),
                available_tools=allowed_tools,
                search_budget=state.get('remaining_searches', 0) if stage == 'research' else 0,
                source_images=images, search_history=state.get('search_history'))
            if not error and ((stage == 'research' and value.get('action') == 'finish') or
                              (stage == 'design' and value.get('decision') == 'proceed')):
                try:
                    plans = value['proposals'] if stage == 'research' else [value['design']]
                    reference_evidence = ({'e1': {'claims': value.get('evidence', [])}}
                                          if native_proposal else evidence)
                    for plan in plans:
                        _validate_design(plan, original, allowed_tools, 0, evidence=reference_evidence)
                except ValueError as exc:
                    label = 'proposal' if stage == 'research' else stage
                    error = f'{label} reference validation: {exc}'
            return error

        def reusable(response):
            valid = (isinstance(response, dict) and response.get('ok') and
                    isinstance(response.get('json'), dict) and
                    validator(response['json']) is None)
            if valid and native_proposal:
                try:
                    bind_native_response(response)
                except ValueError:
                    return False
            return valid

        def bind_native_response(response):
            facts, failures, status = native_evidence(response['json'], response, effective_search_budget)
            for proposal in response['json'].get('proposals', []):
                _validate_design(proposal, original, allowed_tools, 0, evidence=facts)
            return facts, failures, status

        def invoke():
            response = llm(system_prompt=prompt, user_prompt=encoded,
                           images=request['images'], model=model,
                           model_config_path=model_config_path, max_attempts=2,
                           max_tokens=max_tokens, extra_create_params=create_params,
                           response_validator=validator, json_root_type='object',
                           **({'json_tool_name': json_tool_name} if json_tool_name else {}),
                           **({'api_mode': 'responses'} if native_proposal else {}))
            if isinstance(response, dict) and response.get('ok'):
                normalized, corrections = normalize_generation_response(
                    response.get('json'), stage,
                    previous_media_jobs=payload.get('media_jobs') if stage == 'repair' else None)
                if corrections:
                    response = {**response, 'json': normalized, 'format_normalizations': corrections}
            return response

        if on_stage is not None:
            label = ('final_inspection' if payload.get('review_mode') == 'final_validation'
                     else 'initial_inspection') if stage == 'inspect' else (
                         'proposal' if stage == 'research' else stage)
            on_stage(label)
        response, cache_hit = checkpoints.execute(checkpoint_name(stage), request, invoke, reusable)
        events.append({'stage': stage, 'model': model, 'prompt_version': PROMPT_VERSION,
                       **({'research_turn': 1 if native_proposal else payload['research_state']['turn']}
                          if stage == 'research' else {}),
                       'input_chars': len(prompt)+len(encoded),
                       'input_sizes': request['input_sizes'],
                       'seconds': round(time.monotonic()-start, 3),
                       'usage': {} if cache_hit else response.get('usage', {}),
                       'attempts': 0 if cache_hit else response.get('attempts', 1),
                       'cache_hit': cache_hit,
                       'output_chars': len(response.get('raw_text') or ''),
                       'finish_reason': response.get('finish_reason'),
                       'attempt_records': [] if cache_hit else response.get('attempt_records', []),
                       **({'native_web_search_calls': response.get('web_search_calls', []),
                           'citation_count': len(response.get('citations', []))} if native_proposal else {}),
                       **({'format_normalizations': response['format_normalizations']}
                          if response.get('format_normalizations') else {})})
        result = response.get('json')
        validation_error = validator(result) if response.get('ok') else response.get('validation_error')
        if not response.get('ok') or validation_error:
            protocol_failure = bool(validation_error) or response.get('response_error_kind') in (
                'json_parse', 'response_validation', 'output_limit')
            error_type = (InspectionProtocolError if stage == 'inspect' else GenerationProtocolError) if protocol_failure else RuntimeError
            message = validation_error or response.get('error') or 'invalid or incomplete model response'
            raise error_type(f'{stage} failed: {message}')
        if native_proposal:
            try:
                bind_native_response(response)
            except ValueError as exc:
                raise GenerationProtocolError(f'research grounding failed: {exc}') from exc
            native_research_response = response
        if stage == 'inspect':
            result = normalize_inspection_response(result, payload.get('review_mode'))
        if stage == 'replan' and result.get('decision') not in ('proceed', 'skip'):
            raise ValueError('replan response missing/invalid decision; expected proceed or skip')
        return result

    def reject(reason, inspection=None, retryable=False):
        return {'status': 'rejected', 'reason': reason, 'events': events,
                'inspection': inspection, 'initial_inspection': initial_inspection,
                'evidence': evidence, 'retryable': retryable,
                **({'argument_repairs': argument_repairs} if argument_repairs else {}),
                **({'grounding_status': grounding['status']} if grounding else {}),
                **({'search_failures': search_failures} if search_failures else {}),
                **({'source_views': source_views} if source_views else {}),
                'planning_recovery_used': planning_recovery_used,
                'format_recovery_used': format_recovery_used, 'refinement_used': refinement_used}

    try:
        researcher = ResearchAgent(original, contract, recipe, allowed_tools,
                                   effective_search_budget, evidence, search_failures)

        def decide(payload, turn):
            nonlocal checkpoint_phase
            checkpoint_phase = f'research/turn_{turn:02d}'
            return call('research', RESEARCH_PROMPT, payload, True)

        def retrieve(request):
            if on_stage is not None:
                on_stage('search')
            start = time.monotonic()
            packet, cache_hit = checkpoints.execute('research/evidence/' + request['id'],
                {'query': request['query'], 'image_paths': request['image_paths'],
                 'evidence_policy': EVIDENCE_POLICY_VERSION},
                lambda: search(query=request['query'], image_paths=request['image_paths'],
                               model_config_path=model_config_path),
                lambda r: isinstance(r, dict) and r.get('status') in ('supported', 'partial', 'insufficient')
                          and isinstance(r.get('claims'), list))
            if not isinstance(packet, dict):
                raise RuntimeError('invalid search evidence packet')
            diagnostics = evidence_failure_details(packet)
            events.append({'stage': 'search', 'request_id': request['id'], 'status': packet.get('status'),
                           'model': packet.get('search_model'),
                           'source_failures': diagnostics['source_failures'],
                           'seconds': round(time.monotonic()-start, 3),
                           'usage': {} if cache_hit else packet.get('usage', {}),
                           'attempts': 0 if cache_hit else packet.get('attempts', 1),
                           'cache_hit': cache_hit or packet.get('search_cache_hit', False),
                           'disk_cache_hit': cache_hit,
                           'page_verification': packet.get('page_verification', {})})
            return packet

        def save_research_state(turn, state):
            checkpoints.execute(f'research/turn_{turn:02d}/state', state,
                lambda: copy.deepcopy(state), lambda result: result == state)

        if use_native_research:
            checkpoint_phase = 'research/turn_01'
            planned = call('research', PROPOSAL_PROMPT, {
                'task_contract': contract, 'original': original, 'recipe_hint': recipe,
                'research_state': {'mode': 'native_responses', 'turn': 1,
                                   'tool_call_budget': effective_search_budget}}, True)
            facts, failures, grounding_status = native_evidence(
                planned, native_research_response, effective_search_budget)
            evidence.update(facts)
            search_failures.extend(failures)
            native_trace = {'web_search_calls': native_research_response.get('web_search_calls', []),
                            'citations': native_research_response.get('citations', []),
                            'usage': native_research_response.get('usage', {})}
            checkpoints.execute('research/native_search', native_trace,
                lambda: copy.deepcopy(native_trace), lambda value: value == native_trace)
        else:
            planned = researcher.run(decide, retrieve, save_research_state)
            grounding_status = researcher.grounding_status
        checkpoint_phase = ''
        grounding_data = {'status': grounding_status, 'evidence': evidence,
                          'search_failures': search_failures}
        grounding, _ = checkpoints.execute('grounding', grounding_data,
            lambda: copy.deepcopy(grounding_data), lambda value: value == grounding_data)
        if planned['action'] == 'skip':
            return reject(planned['reason'])
        proposals = [_validate_design(p, original, allowed_tools, 0, evidence=evidence)
                     for p in planned['proposals']]
        proposal_data = {'proposals': proposals, 'evidence': evidence, 'grounding_status': grounding_status}
        checkpoints.execute('research/proposals', proposal_data,
            lambda: copy.deepcopy(proposal_data), lambda value: value == proposal_data)
        synthesized = call('design', DESIGN_PROMPT, {
            'task_contract': contract, 'original': original, 'proposals': proposals,
            'evidence': evidence, 'recipe_hint': recipe, 'grounding_status': grounding_status,
            **({'search_failures': search_failures} if search_failures else {})}, True)
        if synthesized['decision'] == 'skip':
            return reject(synthesized['reason'])
        design = _validate_design(synthesized['design'], original, allowed_tools, 0, evidence=evidence)
        # Hand over only the selected evidence packets. Preserve claim indexes
        # and source/qualification context within them; all search results stay
        # available in research/grounding checkpoints for audit.
        selected_ids = {r.split('.')[1] for r in design['source_refs'] if r.startswith('evidence.')}
        evidence = {key: packet for key, packet in evidence.items() if key in selected_ids}
        handoff = {'design': design, 'evidence': evidence,
                   'grounding_status': grounding_status}
        checkpoints.execute('research/handoff', handoff,
            lambda: copy.deepcopy(handoff), lambda value: value == handoff)
        reference_root = {'original': original, 'evidence': evidence}
        context = {'task_contract': contract, 'design': design, **reference_root}
        previous_issues = []
        produced = call('construct', CONSTRUCT_PROMPT, context, True)
        while True:  # At most one refinement, with separately bounded preflight recovery.
            if produced.get('status') not in ('ready', 'blocked'):
                stage = events[-1]['stage'] if events else 'construct'
                return reject(f'{stage} response missing/invalid status; expected ready or blocked')
            if produced.get('status') != 'ready':
                if planning_recovery_used or refinement_used:
                    return reject(str(produced.get('blocking_issue') or 'construction blocked'))
                planning_recovery_used = True
                checkpoint_phase = 'recovery'
                revised = call('replan', REPLAN_PROMPT, {
                    **context, 'failure': produced.get('blocking_issue') or 'construction blocked'}, True)
                if revised.get('decision') != 'proceed':
                    return reject(revised.get('reason') or 'construction unsupported')
                design = _validate_design(revised.get('design'), original, allowed_tools, 0, evidence=evidence)
                context['design'] = design
                produced = call('construct', CONSTRUCT_PROMPT, context, True)
                continue
            candidate = normalize_candidate(produced.get('candidate'), contract)
            if fixture is not None:
                from .rule_ledger import bind_fixture
                candidate = bind_fixture(candidate, fixture)
            support = copy.deepcopy(produced.get('support'))
            if isinstance(support, dict) and isinstance(support.get('answer_refs'), list):
                support['answer_refs'] = [normalize_reference(ref) for ref in support['answer_refs']]
            jobs = canonicalize_media_jobs(produced.get('media_jobs', []))
            errors = content_errors(candidate, support, jobs, contract, reference_root)
            if errors and argument_repair and not argument_repair_used:
                from .argument_repair import AUDIO_ERRORS, audio_argument_schema, repair_audio_arguments, same_audio_execution
                if all(error.startswith(AUDIO_ERRORS) for error in errors):
                    argument_repair_used = True
                    repair_request = {'media_jobs': jobs, 'errors': errors, 'policy': argument_repair,
                                      'schema': audio_argument_schema()}
                    repaired, hit = checkpoints.execute(checkpoint_name('argument_repair'), repair_request,
                        lambda: repair_audio_arguments(jobs, errors, policy=argument_repair),
                        lambda value: isinstance(value, dict) and value.get('ok') is True
                                      and same_audio_execution(jobs, value.get('arguments')))
                    argument_repairs.append(repaired)
                    events.append({'stage': 'argument_repair', 'model': repaired.get('model'),
                                   'ok': repaired.get('ok'), 'cache_hit': hit,
                                   'usage': repaired.get('usage') or {}})
                    if repaired.get('ok'):
                        jobs = canonicalize_media_jobs(repaired['arguments'])
                        produced = {**produced, 'media_jobs': jobs}
                        errors = content_errors(candidate, support, jobs, contract, reference_root)
            if jobs and 'text2speech' not in allowed_tools:
                errors.append('TTS task not enabled')
            if (isinstance(jobs, list) and any(isinstance(job, dict) and job.get('kind') == 'overlay_audio'
                                             for job in jobs) and 'overlay_audio' not in allowed_tools):
                errors.append('overlay_audio task not enabled')
            if (isinstance(jobs, list) and any(isinstance(job, dict) and any(
                    isinstance(o, dict) and o.get('noise') for o in (job.get('outputs') or [])) for job in jobs)
                    and 'add_environmental_noise' not in allowed_tools):
                errors.append('noise task not enabled')
            if isinstance(jobs, list) and jobs and isinstance(jobs[0], dict) and jobs[0].get('noise') and 'add_environmental_noise' not in allowed_tools:
                errors.append('noise task not enabled')
            if (isinstance(jobs, list) and any(isinstance(job, dict) and (
                    job.get('background_music') or any(isinstance(o, dict) and o.get('background_music')
                    for o in (job.get('outputs') or []))) for job in jobs)
                    and 'add_background_music' not in allowed_tools):
                errors.append('background music task not enabled')
            # Cheap checks are rerun rather than trusted from a previous process.
            checkpoints.execute(checkpoint_name('contract_check'),
                {'candidate': candidate, 'support': support, 'media_jobs': jobs,
                 'contract': contract}, lambda: {'errors': errors}, lambda r: False)
            if errors:
                inspection = {'decision': 'revise', 'issues': [
                    {'id': f'contract_{i}', 'code': 'contract_error', 'problem': e,
                     'remedy_scope': 'format'} for i, e in enumerate(errors)]}
                if format_recovery_used or refinement_used:
                    return reject('contract validation failed after bounded correction', inspection)
                format_recovery_used = True
                checkpoint_phase = 'format_recovery'
                produced = call('repair', REPAIR_PROMPT, {
                    **context, 'candidate': candidate, 'support': support, 'media_jobs': jobs,
                    'issues': inspection['issues']}, True)
                continue
            # Randomize once before initial review. Its feedback refers to that
            # draft's labels; reshuffling Repair's output would silently change
            # them while carrying the same previous_issues into final review.
            if contract['answer_type'] == 'choice' and not refinement_used:
                candidate = shuffle_choices(candidate, int(fingerprint(item_key)[:16], 16))
            review_payload = {
                **{k: v for k, v in context.items() if k != 'revision_context'},
                **({'previous_issues': previous_issues} if previous_issues else {}),
                'candidate': candidate, 'support_refs': support.get('answer_refs', []),
                'media_jobs': jobs, 'validation_phase': 'content'}
            final_review = refinement_used
            inspection = call('inspect', INSPECT_PROMPT, {
                **review_payload,
                'review_mode': 'final_validation' if final_review else 'refinement_strategy'}, True)
            if not refinement_used:
                initial_inspection = copy.deepcopy(inspection)
                if (inspection.get('decision') == 'accept' and not inspection.get('issues')
                        and inspection.get('refinement_plan') is None):
                    # No worthwhile edit: preserve the exact candidate and still
                    # obtain the final release decision, without a Repair call.
                    checkpoint_phase = 'unrefined_final'
                    final_review = True
                    inspection = call('inspect', INSPECT_PROMPT, {
                        **review_payload, 'review_mode': 'final_validation'}, True)
            decision = inspection.get('decision')
            final_minor_fixes = []
            if final_review and decision == 'revise':
                try:
                    candidate, support = _apply_final_minor_fixes(candidate, support, inspection, reference_root, contract)
                    final_errors = content_errors(candidate, support, jobs, contract, reference_root)
                    if final_errors:
                        return reject('final minor correction failed contract checks: ' + '; '.join(final_errors), inspection)
                except ValueError as exc:
                    return reject(str(exc), inspection)
                final_minor_fixes = copy.deepcopy(inspection['minor_fixes'])
                checkpoints.execute(checkpoint_name('final_minor_fixes'),
                    {'inspection': inspection},
                    lambda: {'candidate': candidate, 'support': support, 'errors': final_errors}, lambda r: False)
                decision = 'accept'
            if final_review and decision == 'accept' and (not inspection.get('issues') or final_minor_fixes):
                final, tool_messages = bind_media(candidate, jobs, contract, output_root, item_key)
                wrapper = {'original': original, 'current': {'final': final},
                           '_generation': {'version': VERSION, 'prompt_version': PROMPT_VERSION,
                                           'validation_phase': 'content',
                                           'media_status': 'pending' if jobs else 'ready',
                                           'support': support, 'design': design, 'evidence': evidence,
                                           'generation_flow': GENERATION_FLOW_VERSION,
                                           'grounding_status': grounding_status,
                                           'media_jobs': jobs, 'events': events,
                                           **({'source_views': source_views} if source_views else {}),
                                           **({'search_failures': search_failures} if search_failures else {}),
                                           'recovery_used': planning_recovery_used or format_recovery_used or quality_repair_used or bool(final_minor_fixes),
                                           'planning_recovery_used': planning_recovery_used,
                                           'format_recovery_used': format_recovery_used,
                                           'refinement_used': refinement_used,
                                           'initial_inspection': initial_inspection, 'inspection': inspection,
                                           'final_minor_fixes': final_minor_fixes,
                                           'finalization_status': 'accepted_after_minor_fixes' if final_minor_fixes else 'accepted'}}
                wrapper['_generation']['unchecked_properties'] = (
                    ['actual_audio_semantics'] if jobs or 'audio' in contract.get('modalities', []) else [])
                if argument_repairs:
                    wrapper['_generation']['argument_repairs'] = argument_repairs
                if contract.get('generation_guidance'):
                    guidance = contract['generation_guidance']
                    wrapper['_generation']['generation_guidance'] = {
                        'profile': guidance['profile'], 'sha256': fingerprint(guidance)}
                if fixture is not None:
                    wrapper['_generation']['construction_assistance'] = {
                        'type': fixture['type'], 'seed': fixture['seed'],
                        'layers': fixture['layers'], 'ledger_sha256': fingerprint(fixture['ledger']),
                        'reference_status': fixture['reference_status'],
                        'binding': 'Explicit fixture marker expanded before inspection'}
                if tool_messages:
                    wrapper['_pure_tool_messages'] = tool_messages
                return {'status': 'accepted', 'sample': wrapper, 'events': events}
            if final_review or decision not in ('accept', 'revise'):
                return reject(inspection.get('reason') or f'content {decision}', inspection)
            issues = inspection.get('issues')
            if decision == 'accept':
                if issues:
                    return reject('inspection accepted with unresolved issues', inspection)
                plan = inspection.get('refinement_plan')
                issues = plan.get('changes') if isinstance(plan, dict) else None
                if not isinstance(issues, list) or not issues:
                    return reject('initial inspection missing actionable refinement_plan', inspection)
                if any(not isinstance(i, dict) or
                       not isinstance(i.get('suggested_change'), str) or not i['suggested_change'].strip() or
                       not isinstance(i.get('acceptance_check'), str) or not i['acceptance_check'].strip()
                       for i in issues):
                    return reject('initial refinement_plan requires concrete changes and acceptance checks', inspection)
            if not isinstance(issues, list) or not issues or any(
                not isinstance(i, dict) or i.get('remedy_scope') not in ('format', 'local_content', 'redesign') for i in issues):
                return reject('issue requires unsupported redesign or missing evidence', inspection)
            quality_repair_used = decision == 'revise'
            refinement_used = True
            checkpoint_phase = 'refinement'
            previous_issues = copy.deepcopy(issues)
            # Initial review supplies the improvement strategy; execute it in
            # one repair call, including legacy design-level suggestions.
            produced = call('repair', REPAIR_PROMPT, {
                **context, 'candidate': candidate, 'support': support, 'media_jobs': jobs,
                'issues': issues}, True)
    except (ValueError, RuntimeError, TypeError, KeyError) as exc:
        if 'key limit exceeded' in str(exc).lower():
            raise GenerationInfrastructureError('Model gateway key budget limit exceeded') from exc
        if any(marker in str(exc).lower() for marker in (
            'authenticationerror', 'invalid_api_key', 'unauthorized', 'permissiondeniederror')):
            raise GenerationInfrastructureError('Model/search authentication or permission failure') from exc
        result = reject(f'{type(exc).__name__}: {exc}', retryable=isinstance(exc, RuntimeError))
        if isinstance(exc, ValueError) and 'prompt budget exceeded' in str(exc):
            result['failure_kind'] = 'prompt_budget'
        if isinstance(exc, GenerationProtocolError):
            result['failure_kind'] = 'inspection_protocol' if isinstance(exc, InspectionProtocolError) else 'generation_protocol'
        return result


def collect_verified(results):
    verified, rejected = {}, {}
    fixed = 0
    retryable_failed = 0
    for result in results:
        did = result['dataset_id']
        if result.get('status') == 'accepted':
            item = {'dataset_id': did, 'idx': result['idx'], 'sample': copy.deepcopy(result['sample'])}
            item['sample']['current'] = item['sample']['current']['final']
            was_fixed = item['sample']['_generation']['recovery_used']
            item['sample']['status'] = 'fixed' if was_fixed else 'ok'
            fixed += int(was_fixed)
            verified.setdefault(did, []).append(item)
        else:
            retryable_failed += int(bool(result.get('retryable')))
            rejected.setdefault(did, []).append({
                'dataset_id': did, 'idx': result['idx'], 'errors': [result.get('reason', 'rejected')],
                'diagnostics': {k: result[k] for k in ('events', 'inspection', 'evidence', 'retryable', 'failure_kind', 'grounding_status', 'search_failures', 'source_views') if k in result}})
    n_ok = sum(map(len, verified.values()))
    n_rejected = sum(map(len, rejected.values()))
    return {'verified_buffer': verified, 'rejected_buffer': rejected,
            'stats': {'total': n_ok+n_rejected, 'ok': n_ok-fixed, 'fixed': fixed, 'rejected': n_rejected,
                      'terminal_rejected': n_rejected-retryable_failed, 'retryable_failed': retryable_failed}}


def planned_transform_batch(context, cache_root, max_workers=4):
    from utils.model_config import get_tool_model
    from utils.model_config import get_api_base_url
    from tools.executor_tools.implementations.web_evidence import EVIDENCE_POLICY_VERSION
    from tools.shared.dataset_paths import resolve_dataset_source
    ctx = dict(context)
    sid = ctx['current_subtask_id']
    subtask = next(s for s in ctx['subtasks'] if s['id'] == sid)
    config = ctx.get('model_config_path')
    from utils.model_config import load_model_config
    search_settings = load_model_config(config).get('executor_search')
    models = {'research': get_tool_model('transform_stage0', config),
              'design': get_tool_model('transform_stage0', config),
              'construct': get_tool_model('transform_stage2', config),
              'inspect': get_tool_model('verify', config)}
    provider = get_api_base_url(config)
    contract = task_contract(subtask, ctx.get('description', ''),
                             benchmark_task_id=ctx.get('benchmark_task_id') or ctx.get('task_id'),
                             generation_guidance=(ctx.get('generation_guidance') or {}).get(sid))
    if contract.get('generation_guidance'):
        print(f'[PlannedGeneration] {sid}: supplementary guidance='
              f'{contract["generation_guidance"]["profile"]}', flush=True)
    search_budget = max(0, int(ctx.get('max_web_searches', DEFAULT_SEARCH_BUDGET)))
    cfg_tools = ctx.get('tools_list') or []
    if isinstance(cfg_tools, dict):
        cfg_tools = cfg_tools.get('tools', list(cfg_tools.values()))
    configured = {t.get('name') for t in cfg_tools if isinstance(t, dict)}
    # Native proposal search is controlled by its budget, not the asset catalog.
    allowed = sorted((configured.intersection(ASSET_AUDIO_TOOLS) if has_modality(contract, 'audio') else set())
                     | ({'web_search'} if search_budget > 0 else set()))
    tool_specs = media_tool_specs(allowed, cfg_tools)
    root = Path(ctx['generation_output_root'])
    jobs = []
    cached_count = 0
    reused_count = 0
    results = list(ctx.get('generation_results', {}).get(sid, []))
    seen = {(r['dataset_id'], r['idx']) for r in results}
    for pair in ctx['current_pairs']:
        did = pair['dataset_id']
        source = ctx['dataset_cards'][did].get('raw_meta', {}).get('source_json')
        try:
            assessment = (subtask.get('transformability') or {}).get(did) or {}
            blueprint = pair.get('blueprint') or assessment.get('blueprint')
            recipe = compact_dataset_guidance(blueprint if blueprint else
                                             recipe_hint(pair.get('tool_plan') or []))
            if not recipe.get('strategy') and not blueprint and assessment.get('overall_notes'):
                recipe['strategy'] = str(assessment['overall_notes']).strip()[:1600]
            source_path = resolve_dataset_source(source)
            if str(source_path) != str(source):
                print(f'[PlannedGeneration] {did}: source resolved to {source_path}')
            raw = json.loads(source_path.read_text(encoding='utf-8'))['data']
            if not isinstance(raw, list):
                raise ValueError('source data must be a list')
        except OSError as exc:
            raise GenerationInfrastructureError(f'Dataset {did} cannot be read: {exc}') from exc
        except (ValueError, KeyError, TypeError) as exc:
            for idx in pair.get('idx_todo', []):
                results.append({'dataset_id': did, 'idx': idx, 'status': 'rejected', 'reason': str(exc)})
            continue
        for idx in pair.get('idx_todo', []):
            key = (did, idx)
            if key in seen:
                continue
            seen.add(key)
            if not isinstance(idx, int) or isinstance(idx, bool) or not 0 <= idx < len(raw):
                results.append({'dataset_id': did, 'idx': idx, 'status': 'rejected', 'reason': 'invalid sample index'})
                continue
            original = resolve_images(raw[idx], str(source_path), did)
            media_identity = {p: (Path(p).stat().st_size, Path(p).stat().st_mtime_ns)
                              for p in _source_images(original) if Path(p).is_file()}
            signature_input = {'version': VERSION, 'flow': GENERATION_FLOW_VERSION,
                                     'prompt_version': PROMPT_VERSION,
                                     'contract': contract, 'original': original,
                                     'recipe': recipe, 'models': models, 'tools': allowed,
                                     **({'media_tool_specs': tool_specs} if tool_specs else {}),
                                     'provider': provider,
                                     **({'executor_search': search_settings} if search_settings else {}),
                                     'search_budget': search_budget, 'research_policy': RESEARCH_POLICY_VERSION,
                                     'media': media_identity,
                                     'checkpoint_version': CHECKPOINT_VERSION,
                                     'generation_response_policy': GENERATION_RESPONSE_POLICY_VERSION,
                                     'payload_policy': PAYLOAD_POLICY_VERSION, 'source_view_policy': SOURCE_VIEW_POLICY_VERSION,
                                     'inspection_protocol': INSPECTION_PROTOCOL_VERSION,
                                     'native_evidence_policy': NATIVE_EVIDENCE_POLICY_VERSION,
                                     'evidence_policy': EVIDENCE_POLICY_VERSION,
                                     # Fingerprint the conditional Repair instructions too;
                                     # live requests derive the flag from their actual jobs.
                                     'prompts': [build_generation_prompt(stage, prompt, contract,
                                                                        available_media_tools=allowed,
                                                                        has_existing_media_jobs=stage in ('inspect', 'repair'))
                                                 for stage, prompt in (
                                                     ('proposal', PROPOSAL_PROMPT), ('design', DESIGN_PROMPT),
                                                     ('construct', CONSTRUCT_PROMPT),
                                                     ('inspect', INSPECT_PROMPT), ('repair', REPAIR_PROMPT),
                                                     ('replan', REPLAN_PROMPT))]}
            signature = fingerprint(signature_input)
            item_key = fingerprint([sid, did, idx, signature])[:24]
            path = root / 'samples' / (item_key + '.json')
            if cache_root:
                cached_result = _read_accepted_cache(path, signature)
                if cached_result is None and ctx.get('reuse_accepted_cache'):
                    previous_signature = _compatible_v12_signature(signature_input)
                    if previous_signature:
                        previous_key = fingerprint([sid, did, idx, previous_signature])[:24]
                        previous_path = root / 'samples' / (previous_key + '.json')
                        previous_result = _read_accepted_cache(previous_path, previous_signature)
                        if (previous_result is not None
                                and previous_result.get('dataset_id') == did
                                and previous_result.get('idx') == idx
                                and get_path(previous_result, 'sample._generation.prompt_version') == 'v1.2'
                                and not get_path(previous_result, 'sample._generation.media_jobs')):
                            cached_result = previous_result
                            reused_count += 1
                if cached_result is not None:
                    results.append(cached_result)
                    cached_count += 1
                    continue
            jobs.append((did, idx, original, recipe, item_key, path, signature))

    if reused_count:
        counts = ctx.setdefault('generation_reused_accepted', {})
        counts[sid] = counts.get(sid, 0) + reused_count
        print(f'[PlannedGeneration] {sid}: reused {reused_count} accepted v1.2 samples; '
              'non-audio inputs/settings match, original provenance preserved.', flush=True)

    def worker(job):
        did, idx, original, recipe, key, path, signature = job
        progress.stage(key, 'preparing')
        resume = bool(cache_root)
        if resume and path.is_file():
            try:
                previous = json.loads(path.read_text(encoding='utf-8'))
                previous_result = previous.get('result', {})
                if (previous.get('signature') == signature
                        and previous.get('result_digest') == fingerprint(previous_result)
                        and previous_result.get('status') == 'rejected'
                        and not previous_result.get('retryable')):
                    resume = False
            except (OSError, ValueError, TypeError, AttributeError):
                pass
        result = run_sample(original, contract, recipe, root, key, models,
                            allowed_tools=allowed, search_budget=search_budget, model_config_path=config,
                            checkpoint_dir=root / 'stages' / key,
                            resume_checkpoints=resume, cache_namespace=signature,
                            media_tool_definitions=tool_specs,
                            argument_repair=ctx.get('executor_argument_repair'),
                            on_stage=lambda name: progress.stage(key, name))
        result.update(dataset_id=did, idx=idx)
        write_json(path, {'signature': signature, 'result': result, 'result_digest': fingerprint(result)})
        return result

    with GenerationProgress(sid, len(results) + len(jobs), results, cached_count,
                            interactive=ctx.get('generation_progress_live')) as progress:
        if jobs:
            worker_count = max(1, min(max_workers, len(jobs)))
            print(f'[PlannedGeneration] {sid}: model/sample concurrency={worker_count}, jobs={len(jobs)}')
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                futures = {pool.submit(worker, job): job for job in jobs}
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        result = future.result()
                    except GenerationInfrastructureError:
                        for pending in futures:
                            pending.cancel()
                        raise
                    except Exception as exc:
                        result = {'dataset_id': job[0], 'idx': job[1], 'status': 'rejected',
                                  'reason': f'worker failed: {type(exc).__name__}: {exc}'}
                    results.append(result)
                    progress.finish(job[4], result)
    results.sort(key=lambda r: (r['dataset_id'], str(r['idx'])))
    ctx.setdefault('generation_results', {})[sid] = results
    ctx.setdefault('transformed_buffer', {})[sid] = [
        {'dataset_id': r['dataset_id'], 'idx': r['idx'], 'sample': r['sample']}
        for r in results if r.get('status') == 'accepted']
    ctx.setdefault('generation_verified', {})[sid] = collect_verified(results)
    print(f'[PlannedGeneration] {sid}: {ctx["generation_verified"][sid]["stats"]}')
    return ctx
