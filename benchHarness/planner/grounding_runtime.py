"""Adaptive per-subtask Grounding with round barriers and one shared model limit."""
import asyncio
import copy
import inspect
import json
import re
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from benchHarness.planner.grounding_prompts import GROUNDING_PROMPT, ASSESS_PROMPT, BLUEPRINT_PROMPT
from benchHarness.planner.grounding_decisions import DECISION_TOOL, decision_schema, unpack_decision
from tools.planner_tools.grounding.adaptive_tools import GROUNDING_TOOLS, grounding_tool_schemas
from tools.planner_tools.grounding.grounding_materials import (
    asset_tools, capability_view, candidate_view, digest, file_version, manifest_valid,
    read_material, source_identity, task_view,
    material_reference_examples, normalize_material_reference,
    brief_text,
)

GROUNDING_VERSION = 'adaptive-v1'
INTERFACE_VERSION = 'working-note-planned-approval-v1'
WORKING_NOTE_MAX_CHARS = 1200  # Soft history target; retained name for compatibility.
MAX_PLANNED_DATASETS = 5
DEFAULT_BUDGETS = {'searches_left': 3, 'assessments_left': 12, 'plans_left': 6, 'steps_left': 12}
_BUDGET_KEY = {'search_datasets': 'searches_left', 'assess_dataset': 'assessments_left',
               'plan_dataset_transformation': 'plans_left'}


def _brief(value, limit=220):
    text = ' '.join(str(value).split())
    return text if len(text) <= limit else text[:limit - 3] + '...'


def _progress(message, subtask_id=None, logger=None):
    prefix = '[Grounding]' + (f'[{subtask_id}]' if subtask_id is not None else '')
    line = f'{prefix} {message}'
    if logger is not None:
        logger.info(line, title='Grounding')
    else:
        print(f'[{datetime.now():%H:%M:%S}] {line}', flush=True)


def _request_label(stage, request):
    dataset_id = request.get('payload', {}).get('material', {}).get('dataset_id')
    return stage + (f' dataset={dataset_id}' if dataset_id else '')


class GroundingError(RuntimeError):
    def __init__(self, code, message, retryable=False):
        super().__init__(message)
        self.code, self.retryable = code, retryable


def _error(error):
    return {'error': {'code': getattr(error, 'code', 'tool_execution_failed'),
                      'message': str(error), 'retryable': getattr(error, 'retryable', True)}}


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid4().hex[:8] + '.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _safe(value):
    return re.sub(r'[^\w.-]+', '_', str(value))[:100] or 'unknown'


def _prepare_cache_directory(cache_path, subtask_id, signature, progress, *, reuse_existing=False, fresh=False):
    """Keep the active task readable; retain signatures in metadata and old runs in history."""
    base = Path(cache_path) / 'grounding'
    task_name = _safe(subtask_id)
    root = base / task_name
    if root.resolve().parent != base.resolve() or task_name in ('_history', GROUNDING_VERSION):
        raise ValueError('Invalid Grounding subtask cache directory.')
    info_path = root / 'cache_info.json'
    info = _read(info_path)
    matches = isinstance(info, dict) and (info.get('signature') == signature or
        bool(signature and info.get('legacy_signature_prefix') == signature[:24]))
    if root.exists() and reuse_existing:
        if signature is not None and not matches:
            progress('using selected cache despite changed input signature')
        return root
    if root.exists() and (fresh or not matches):
        previous = base / '_history' / task_name / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        # Verify both absolute paths before moving a directory on Windows.
        if (not previous.resolve().is_relative_to(base.resolve()) or
                not previous.resolve().is_relative_to((base / '_history' / task_name).resolve())):
            raise ValueError('Grounding history directory escapes its cache root.')
        previous.parent.mkdir(parents=True, exist_ok=True)
        root.replace(previous)
        progress(f'cache {"reset requested" if fresh else "inputs changed"}: saved previous run to {previous}')
        info = None
    if not root.exists() and not fresh:
        legacy_parent = base / GROUNDING_VERSION / task_name
        legacy = legacy_parent / signature[:24] if signature else None
        if reuse_existing and (legacy is None or not legacy.is_dir()):
            candidates = [path for path in legacy_parent.glob('*') if path.is_dir()]
            if candidates:
                legacy = max(candidates, key=lambda path: max(
                    (item.stat().st_mtime_ns for item in path.rglob('*.json')), default=0))
        if legacy is not None and legacy.is_dir():
            if not legacy.resolve().is_relative_to(base.resolve()):
                raise ValueError('Legacy Grounding directory escapes its cache root.')
            root.parent.mkdir(parents=True, exist_ok=True)
            info = {'subtask_id': str(subtask_id), 'format_version': GROUNDING_VERSION,
                    'migrated_from': legacy.relative_to(base).as_posix()}
            if signature and legacy.name == signature[:24]:
                info['signature'] = signature
            else:
                info['legacy_signature_prefix'] = legacy.name
            # Move metadata together with the files, so an interruption cannot detach their identity.
            _write(legacy / 'cache_info.json', info)
            legacy.replace(root)
            progress(f'migrated existing cache to {root}')
            # Remove only empty containers; unrelated old signatures remain available.
            for parent in (legacy.parent, base / GROUNDING_VERSION):
                try:
                    parent.rmdir()
                except OSError:
                    pass
    if (not isinstance(info, dict) or info.get('signature') != signature) and not (reuse_existing and root.exists()):
        info = {**(info or {}), 'subtask_id': str(subtask_id), 'signature': signature,
                'format_version': GROUNDING_VERSION}
        _write(info_path, info)
    return root


def _native_prompt(prompt, enabled):
    if enabled:
        return prompt
    # Capability configuration belongs to code, not an extra model input field.
    return re.sub(r'Generator Proposal can\s+(?:use|retrieve)[^.]*native web search[^.]*\.',
                  'Generator Proposal uses source material and stable knowledge in this run; '
                  'native web search is unavailable.', prompt)


async def _default_decide(**request):
    from litellm import acompletion
    from benchHarness.planner.grounding_responses import responses_request, responses_reply
    from utils.model_config import get_api_base_url, get_api_key, get_output_token_budget
    from utils.llm_caller import _retryable_call_error
    config = request.pop('model_config_path', None)
    require_history = request.pop('_require_history', False)
    request.update(timeout=180, num_retries=0, max_tokens=get_output_token_budget(2000, config),
                   tool_choice='required', parallel_tool_calls=False)
    base, key = get_api_base_url(config), get_api_key(config)
    if base:
        request['base_url'] = base
    if key:
        request['api_key'] = key
    if request['model'].startswith('gpt'):
        request['custom_llm_provider'] = 'openai'
    use_responses = request['model'].startswith(('gpt-', 'openai/gpt-', 'openai/responses/'))
    format_failures = []
    for attempt in range(2):
        value = None
        try:
            if use_responses:
                from litellm import aresponses
                result = await aresponses(**responses_request(request))
                raw = result.model_dump() if hasattr(result, 'model_dump') else dict(result)
                value = responses_reply(raw)
            else:
                result = await acompletion(**request)
                value = result.model_dump() if hasattr(result, 'model_dump') else dict(result)
            parsed = _main_message(value)
            if 'decision' not in parsed:
                raise ValueError('Use one submit_grounding_decision call with history and actions.')
            if require_history and not parsed['decision']['history']:
                raise ValueError('Include nonempty prior confirmed history from working_note and the '
                                 'preceding action results. An empty history is only valid for the initial Search.')
            history_length = len(parsed['decision']['history'])
            if not attempt and history_length > WORKING_NOTE_MAX_CHARS:
                raise ValueError(f'history has {history_length} characters; aim for '
                    f'{WORKING_NOTE_MAX_CHARS}. Compress repetition while preserving key findings '
                    'and the intended business actions.')
            # Preserve both returned attempts and their usage if a format retry
            # was needed; app-level diagnostics otherwise see only the last call.
            if format_failures:
                value['decision_attempts'] = format_failures + [{'ok': True,
                    'usage': copy.deepcopy(value.get('usage') or {})}]
                usages = [item['usage'] for item in value['decision_attempts']]
                value['usage'] = {key: sum(u.get(key, 0) or 0 for u in usages)
                                  for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
            return value
        except Exception as error:
            if attempt or not (isinstance(error, ValueError) or _retryable_call_error(error)):
                raise
            if isinstance(error, ValueError):
                if value is not None:
                    format_failures.append({'ok': False, 'error': str(error),
                        'usage': copy.deepcopy(value.get('usage') or {}),
                        'message': copy.deepcopy((value.get('choices') or [{}])[0].get('message'))})
                request['messages'] = request['messages'] + [{'role': 'user', 'content':
                    'Correct the decision format in the same request: ' + str(error) +
                    ' Return one submit_grounding_decision call with history and business actions.'}]
            else:
                await asyncio.sleep(5)


async def _default_json(**request):
    from utils.llm_caller import llm_call_json
    request.pop('stage', None)
    payload = request.pop('payload')
    return await asyncio.to_thread(llm_call_json, user_prompt=json.dumps(payload, ensure_ascii=False), **request)


def _models(config, main_model):
    from utils.model_config import get_grounding_model, get_tool_model, load_model_config
    from tools.planner_tools.grounding.dataset_preference_tools import PREFERENCE_MODEL
    tools = load_model_config(config).get('tools') or {}
    fallback = get_tool_model('transformability_check', config)
    return {'main': main_model or get_grounding_model(config),
            'preference': tools.get('plan_dataset_preference', PREFERENCE_MODEL),
            'assess': tools.get('assess_dataset', fallback),
            'plan': tools.get('plan_dataset_transformation', fallback)}


class StageStore:
    """Content-addressed calls; only validated successes can be replayed."""
    def __init__(self, root, resume=True, progress=None):
        self.root, self.resume = Path(root), resume
        self.progress = progress

    def log(self, message):
        if self.progress is not None:
            self.progress(message)

    async def call(self, stage, request, produce, validate):
        label = _request_label(stage, request)
        key = digest(request)
        path = self.root / stage / (key[:24] + '.json')
        previous = _read(path) if self.resume else None
        if (isinstance(previous, dict) and previous.get('ok') and previous.get('input') == request
                and previous.get('output_digest') == digest(previous.get('output'))):
            try:
                value = validate(previous['output'])
            except (ValueError, TypeError, KeyError):
                pass
            else:
                self.log(f'{label}: cache hit')
                return value, key, True
        started = time.monotonic()
        output = None
        try:
            output = await produce()
            value = validate(output)
        except Exception as error:
            failed = {'input': request, 'ok': False, **_error(error), 'seconds': time.monotonic() - started}
            if output is not None:
                failed['output'] = output  # Keep raw_text/attempt diagnostics even when validation fails.
            _write(path, failed)
            self.log(f'{label}: failed after {time.monotonic() - started:.1f}s; '
                     f'{type(error).__name__}: {_brief(error)}')
            raise
        _write(path, {'input': request, 'ok': True, 'output': output, 'output_digest': digest(output),
                      'seconds': time.monotonic() - started})
        self.log(f'{label}: completed in {time.monotonic() - started:.1f}s')
        return value, key, False


def _refs(reply, material, required=False):
    refs = reply.get('refs', [])
    if isinstance(refs, str):
        refs = [refs]
    if not isinstance(refs, list) or required and not refs:
        raise ValueError('Nonempty material refs are required for this result.')
    return list(dict.fromkeys(normalize_material_reference(ref, material) for ref in refs))


def _assessment(reply, material):
    if not isinstance(reply, dict) or not isinstance(reply.get('answer'), str) or not reply['answer'].strip():
        raise ValueError('Assessment requires a nonempty answer string responding to the question.')
    if not isinstance(reply.get('summary'), str) or not reply['summary'].strip():
        raise ValueError('Assessment requires a nonempty summary.')
    refs = _refs(reply, material, True)
    result = {'answer': reply['answer'].strip(), 'summary': reply['summary'].strip()}
    if refs:
        result['refs'] = refs
    if reply.get('limitations'):
        if not isinstance(reply['limitations'], list) or any(not isinstance(x, str) or not x.strip()
                                                            for x in reply['limitations']):
            raise ValueError('limitations must contain specific string findings.')
        result['limitations'] = reply['limitations']
    return result


def _repair_plan_json(text):
    """Recover only a missing final object brace; never complete model content."""
    try:
        value = json.loads(text.strip() + '}')
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict) or value.get('status') != 'ready':
        return None
    if 'blueprint' in value:
        # Legacy output sometimes left refs inside blueprint and omitted the
        # root brace. Move only that known field, without changing its content.
        if set(value) - {'status', 'blueprint', 'refs'}:
            return None
        blueprint = value['blueprint']
        if not isinstance(blueprint, dict) or set(blueprint) - {'strategy', 'tools', 'refs'}:
            return None
        if 'refs' in blueprint:
            if 'refs' in value:
                return None
            value['refs'] = blueprint.pop('refs')
    else:
        if set(value) - {'status', 'strategy', 'tools', 'refs'}:
            return None
        blueprint = value
    if (not isinstance(blueprint.get('strategy'), str) or not blueprint['strategy'].strip()
            or not isinstance(value.get('refs'), list) or not value['refs']
            or any(not isinstance(ref, str) or not ref.strip() for ref in value['refs'])):
        return None
    return value


def _blueprint(reply, material, tools):
    if not isinstance(reply, dict) or reply.get('status') not in ('ready', 'blocked'):
        raise ValueError('Blueprint requires status: ready/blocked.')
    if reply['status'] == 'blocked':
        if not isinstance(reply.get('reason'), str) or not reply['reason'].strip():
            raise ValueError('Blocked requires a concrete reason.')
        return {'status': 'blocked', 'reason': reply['reason'].strip()}
    # The model uses a flat object; keep old cached nested replies compatible.
    blueprint = reply.get('blueprint', reply)
    if not isinstance(blueprint, dict) or not isinstance(blueprint.get('strategy'), str) or not blueprint['strategy'].strip():
        raise ValueError('Ready requires a nonempty strategy and refs at the top level.')
    result = {'strategy': blueprint['strategy'].strip()}
    allowed = {tool['name'] for tool in tools}
    uses = blueprint.get('tools', [])
    if not isinstance(uses, list):
        raise ValueError('tools must be an array when present.')
    selected, seen = [], set()
    for use in uses:
        if (not isinstance(use, dict) or use.get('name') not in allowed
                or not isinstance(use.get('purpose'), str) or not use['purpose'].strip()):
            raise ValueError('Each tool needs an available exact name and a purpose.')
        if use['name'] not in seen:
            selected.append({'name': use['name'], 'purpose': use['purpose'].strip()})
            seen.add(use['name'])
    if selected:
        result['tools'] = selected
    return {'status': 'ready', 'blueprint': result, 'refs': _refs(reply, material, True)}


def _check_arguments(value, schema):
    kind = schema.get('type')
    if kind == 'string' and not isinstance(value, str):
        raise ValueError('Expected a string argument.')
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError('Argument is outside the allowed values.')
    if kind == 'array':
        if not isinstance(value, list):
            raise ValueError('Expected an array argument.')
        for item in value:
            _check_arguments(item, schema['items'])
    if kind == 'object':
        if not isinstance(value, dict):
            raise ValueError('Expected object arguments.')
        properties = schema['properties']
        if set(value) - set(properties) or set(schema.get('required', [])) - set(value):
            raise ValueError('Unknown arguments or missing required arguments; follow the tool schema.')
        for key, item in value.items():
            _check_arguments(item, properties[key])


def _main_message(reply):
    if hasattr(reply, 'model_dump'):
        reply = reply.model_dump()
    if not isinstance(reply, dict):
        raise ValueError('Invalid main-agent response.')
    choices = reply.get('choices') or []
    if choices and choices[0].get('finish_reason') in ('length', 'incomplete'):
        raise ValueError('Decision output was incomplete; return a complete history/actions object.')
    message = choices[0].get('message') if choices else reply
    if not isinstance(message, dict):
        raise ValueError('Main-agent response lacks a message.')
    calls = copy.deepcopy(message.get('tool_calls') or [])
    if not isinstance(calls, list):
        raise ValueError('tool_calls must be an array.')
    ids = set()
    for index, call in enumerate(calls):
        if not isinstance(call, dict) or not isinstance(call.get('function'), dict):
            raise ValueError('Invalid function tool call.')
        if not call.get('id') or call['id'] in ids:
            call['id'] = f'call_{index}_{uuid4().hex[:12]}'
        ids.add(call['id'])
        call['type'] = 'function'
    result = {'role': 'assistant', 'content': message.get('content'), 'tool_calls': calls}
    if any(call['function'].get('name') == DECISION_TOOL for call in calls):
        if len(calls) != 1:
            raise ValueError('Return exactly one submit_grounding_decision call per response.')
        result['decision'], result['tool_calls'] = unpack_decision(calls[0])
    return result


class GroundingSession:
    def __init__(self, task, context, cards, store, models, gate, decide, json_client, retriever=None,
                 progress=None):
        self.task, self.context, self.cards = copy.deepcopy(task), context, cards
        self.sid, self.store, self.models = str(task['id']), store, models
        self.log = progress or (lambda message: _progress(message, self.sid))
        if self.store.progress is None:
            self.store.progress = self.log
        self.gate, self.decide, self.json_client = gate, decide, json_client
        self.retriever = retriever
        self.tools = asset_tools(context, task)
        self.native_search = int(context.get('max_web_searches', 3)) > 0
        self.budget = {key: max(0, int((context.get('grounding_budgets') or {}).get(key, value)))
                       for key, value in DEFAULT_BUDGETS.items()}
        self.candidates, self.assessments, self.blueprints, self.searches = {}, {}, {}, {}
        self.discarded, self.search_preferences = [], {}
        self.statuses, self.reserved, self.assessment_answers = {}, {}, {}
        self.pending_decisions, self.operation_errors = [], {}
        self.plan_reports = {}
        self.approvals = {}
        self.working_note = ''
        self.working_note_update_required = False
        self.search_discard_offset = 0
        self.plan_since_search = False
        self.preference, self.attempts, self.failures = {}, [], []
        self.materials, self.result = {}, None
        self.last_round = []
        self.last_actions = []
        self.history_saved = False
        self.no_progress_rounds = 0
        self.seen_report_reads = []
        self.action_schemas = grounding_tool_schemas()  # Legacy dispatch remains cache/test compatible.
        self.schemas = [decision_schema(WORKING_NOTE_MAX_CHARS)]
        self.functions = {function.__name__: function for function in GROUNDING_TOOLS}

    def view(self):
        result = {'task': task_view(self.task), 'budget': copy.deepcopy(self.budget)}
        if self.working_note:
            result['working_note'] = self.working_note
        if self.working_note_update_required:
            result['working_note_update_required'] = True
        if self.last_actions:
            feedback = {'actions': copy.deepcopy(self.last_actions)}
            if self.history_saved:
                feedback['history_saved'] = True
            if self.no_progress_rounds:
                feedback['no_progress_rounds'] = self.no_progress_rounds
                feedback['guidance'] = (
                    'No new source result or status change. Note text does not execute its plans. '
                    'Choose a substantive action: investigate a concrete gap, develop a route, '
                    'mark a sound planned dataset ready, discard an unselected source, or Finish '
                    'when the pool is resolved. Do not repeat notes, no-op actions or the same report read.')
                if self.no_progress_rounds >= 2:
                    feedback['guidance'] = 'Repeated lack of progress; execute a decision now. ' + feedback['guidance']
            if any(item.get('error') for item in self.last_actions):
                feedback['reconcile_note'] = 'Some requested actions failed; correct intended progress against actual state before retrying.'
            result['last_round_feedback'] = feedback
        result['budget']['max_planned_datasets'] = MAX_PLANNED_DATASETS
        if self.tools:
            result['tool_capabilities'] = capability_view(self.tools)
        if self.context.get('description'):
            result['benchmark_goal'] = self.context['description']
        state = {}
        candidates, reserved = [], []
        for did, item in self.candidates.items():
            entry = {'dataset_id': did, 'status': self.statuses.get(did, 'candidate')}
            if item.get('name'):
                entry['name'] = item['name']
            if did in self.reserved:
                entry['status'] = 'reserved'
                entry['progress'] = self.reserved[did]
                reserved.append(entry)
                continue
            report = self.assessments.get(did)
            plan = self.plan_reports.get(did)
            if self.is_ready(did):
                entry['route'] = brief_text(plan['blueprint']['strategy'], 160)
                if plan['blueprint'].get('tools'):
                    entry['tools'] = [tool['name'] for tool in plan['blueprint']['tools']]
            elif entry['status'] == 'planned' and plan and plan.get('status') == 'ready':
                entry['blueprint'] = copy.deepcopy(plan['blueprint'])
                # A follow-up investigation must reach the main agent even
                # while the existing unaccepted blueprint stays planned.
                if report and plan.get('_assessment_ref') != report.get('assessment_ref'):
                    entry['answer'] = report['answer']
                    if report.get('_question'):
                        entry['question'] = report['_question']
            elif report:
                # Explicit cache reuse can expose an old conclusion, without inventing a new answer.
                entry['answer'] = report.get('answer') or report.get('summary', '')
                if report.get('_question'):
                    entry['question'] = report['_question']
            elif item.get('summary'):
                entry['summary'] = brief_text(item['summary'], 240)
            if did in self.operation_errors:
                entry['error'] = copy.deepcopy(self.operation_errors[did])
            if plan and plan.get('status') == 'blocked':
                entry['blocker'] = plan['reason']
            candidates.append(entry)
        if candidates:
            state['candidates'] = candidates
        if reserved:
            state['reserved'] = reserved
        global_errors = [value for key, value in self.operation_errors.items() if key not in self.candidates]
        if global_errors:
            state['errors'] = copy.deepcopy(global_errors)
        if self.discarded:
            state['discarded'] = list(self.discarded)
        if not self.candidates and self.search_preferences:
            audit = _read(self.store.root / 'discard_log.json') or []
            feedback = audit[self.search_discard_offset:][-10:]
            if feedback:
                state['retrieval_feedback'] = [
                    {'name': self.cards.get(item['dataset_id'], {}).get('name', item['dataset_id']),
                     'reason': brief_text(item['reason'], 200)} for item in feedback]
        if state:
            result['state'] = state
        return result

    def snapshot(self):
        return {'interface_version': INTERFACE_VERSION, **{key: copy.deepcopy(getattr(self, key)) for key in
                ('candidates', 'assessments', 'blueprints', 'plan_reports', 'searches', 'preference',
                 'attempts', 'last_round', 'discarded', 'search_preferences',
                 'statuses', 'reserved', 'assessment_answers', 'pending_decisions', 'operation_errors',
                 'plan_since_search', 'approvals', 'working_note', 'working_note_update_required',
                 'search_discard_offset', 'last_actions', 'no_progress_rounds', 'seen_report_reads',
                 'history_saved')}}

    def _progress_digest(self):
        # Notes, execution errors, budgets and logs alone do not advance source work.
        return digest({key: getattr(self, key) for key in
                       ('candidates', 'assessments', 'plan_reports', 'statuses', 'reserved',
                        'approvals', 'discarded', 'search_preferences')})

    def _record_round_feedback(self, prepared, results, before, visible_blueprints):
        actions, new_information = [], False
        for (call, args, ids, _), outcome in zip(prepared, results):
            name = call['function'].get('name')
            error = outcome.get('error')
            status = ('error' if error else 'blocked' if outcome.get('status') == 'blocked'
                      else self.statuses.get(args.get('dataset_id')) or outcome.get('status', 'completed'))
            item = {'tool': name, 'status': status}
            if args.get('dataset_id'):
                item['dataset_id'] = args['dataset_id']
            elif ids:
                item['dataset_ids'] = list(ids)
            if error:
                item['error'] = {'code': error['code'], 'message': brief_text(error['message'], 160)}
            actions.append(item)
            if name == 'read_assessment' and not error:
                key = digest({'dataset': args['dataset_id'], 'type': args.get('report_type', 'assessment'),
                              'report': outcome.get('report')})
                already_visible = (args.get('report_type') == 'blueprint' and
                    args['dataset_id'] in visible_blueprints and
                    outcome.get('report', {}).get('blueprint') == visible_blueprints[args['dataset_id']])
                if key not in self.seen_report_reads:
                    new_information |= not already_visible
                    self.seen_report_reads.append(key)
        self.last_actions = actions
        progressed = before != self._progress_digest() or new_information or self.result is not None
        self.no_progress_rounds = 0 if progressed else self.no_progress_rounds + 1
        if any(item.get('error') and item['tool'] != 'update_working_note' for item in actions):
            self.working_note_update_required = True
        if self.no_progress_rounds:
            self.log(f'no progress for {self.no_progress_rounds} round(s): '
                     'request an actual decision; note changes alone do not execute actions')

    def _note_state_digest(self):
        """Detect source changes, excluding budgets, log growth and the note itself."""
        state = {key: getattr(self, key) for key in
                 ('candidates', 'assessments', 'plan_reports', 'statuses', 'reserved',
                  'approvals', 'discarded', 'search_preferences')}
        state['operation_errors'] = {key: value for key, value in self.operation_errors.items()
                                     if key != '__update_working_note'}
        return digest(state)

    def assessment_ref(self, dataset_id):
        report = self.assessments.get(dataset_id)
        if not report:
            return None
        return report.get('assessment_ref') or 'legacy/' + digest(report)

    def recent_tool_context(self):
        """Keep paired receipts; substantive answers/blueprints appear once in state."""
        if self.last_round and self.last_round[0].get('decision'):
            # New decisions already have compact action receipts in state. Only
            # explicitly requested full reports need a one-shot detail supplement.
            reports = []
            for message in self.last_round[1:]:
                if message.get('role') == 'tool' and message.get('name') == 'read_assessment':
                    value = json.loads(message['content'])
                    if value.get('report'):
                        reports.append(value)
            return ([{'role': 'user', 'content': json.dumps({'requested_reports': reports}, ensure_ascii=False)}]
                    if reports else [])
        messages = copy.deepcopy(self.last_round)
        # Successful discards are already represented by excluded IDs in state.
        # Keep their original arguments/reasons in disk traces, not the next request.
        discarded_calls = {message['tool_call_id'] for message in messages
                           if message.get('role') == 'tool' and (
                               message.get('name') == 'discard_datasets' and
                               json.loads(message['content']).get('status') == 'discarded' or
                               message.get('name') == 'update_working_note' and
                               json.loads(message['content']).get('status') == 'updated')}
        compact = []
        for message in messages:
            if message.get('role') == 'assistant' and message.get('tool_calls'):
                message['tool_calls'] = [call for call in message['tool_calls'] if call['id'] not in discarded_calls]
                if not message['tool_calls']:
                    message.pop('tool_calls')
                    if not message.get('content'):
                        continue
            if message.get('role') != 'tool':
                compact.append(message)
                continue
            if message['tool_call_id'] in discarded_calls:
                continue
            data = json.loads(message['content'])
            receipt = {key: data[key] for key in ('dataset_id', 'dataset_ids', 'status') if key in data}
            if 'error' in data:
                receipt['status'] = 'error'
                receipt['error'] = copy.deepcopy(data['error'])
            # Explicit local rereads are one-shot detail, not permanent state.
            if message.get('name') == 'read_assessment' and data.get('report'):
                receipt['report'] = data['report']
            message['content'] = json.dumps(receipt, ensure_ascii=False)
            compact.append(message)
        return compact

    def restore(self, saved):
        if isinstance(saved, dict):
            for key in self.snapshot():
                if key != 'interface_version' and key in saved:
                    setattr(self, key, copy.deepcopy(saved[key]))
            # Older checkpoints stored a reason per ID; retain only exclusions.
            self.discarded = list(dict.fromkeys(self.discarded))
            if saved.get('interface_version') != INTERFACE_VERSION:
                self.last_round = []  # Old calls used removed tools/arguments.
                old_statuses = copy.deepcopy(self.statuses)
                self.statuses = {did: ('planned' if self.plan_reports.get(did, {}).get('status') == 'ready'
                                      else 'assessed' if did in self.assessments else 'candidate')
                                 for did in self.candidates}
                # Only old final selections prove main-agent acceptance.
                for did, status in old_statuses.items():
                    report = self.plan_reports.get(did, {})
                    if status == 'selected' and report.get('status') == 'ready':
                        self.approvals[did] = report['blueprint_ref']
                        self.statuses[did] = 'selected'
                self.reserved = {did: self.statuses[did] for did in self.reserved if did in self.candidates}
                self.operation_errors = {key: value for key, value in self.operation_errors.items()
                                         if value.get('code') not in ('unassigned_candidates', 'search_before_plan')}
            self.reserved = {did: status for did, status in self.reserved.items() if did in self.candidates}
            self.pending_decisions = []  # Kept only for old checkpoint compatibility; no round coverage rule.
            if 'working_note_update_required' not in saved:
                # Old partial runs cannot prove that their results reached the note.
                self.working_note_update_required = bool(self.candidates or self.discarded)
            if 'plan_since_search' not in saved:
                self.plan_since_search = any(report.get('status') in ('ready', 'blocked')
                                             for report in self.plan_reports.values())

    async def model_call(self, function, **request):
        label = _request_label(request.get('stage', 'main'), request)
        if self.gate.locked():
            self.log(f'{label}: waiting for shared model slot')
        async with self.gate:
            self.log(f'{label}: request started; model={request.get("model", "unknown")}')
            if inspect.iscoroutinefunction(function):
                return await function(**request)
            # Keep the global permit until the synchronous request actually ends.
            task = asyncio.create_task(asyncio.to_thread(function, **request))
            try:
                reply = await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                raise
            return await reply if inspect.isawaitable(reply) else reply

    async def json_call(self, stage, prompt, payload, validator, images=None, max_tokens=1600):
        config_path = self.context.get('model_config_path')
        if stage == 'preference' and config_path:
            from utils.model_config import load_model_config
            preference = load_model_config(config_path).get('retrieval_preference') or {}
            if preference.get('model_config_path'):
                path = Path(preference['model_config_path'])
                config_path = str(path if path.is_absolute() else Path(config_path).resolve().parent/path)
        if stage in ('assess', 'plan'):
            examples = material_reference_examples(payload['material'])
            prompt += ('\nReference path examples from THIS input: ' + json.dumps(examples, ensure_ascii=False)
                       + '. Cite the paths that actually support your findings; use the supplied sample IDs '
                       'and field names, not URLs, prose, or bare schema paths such as input.context.')
        request = {'system_prompt': _native_prompt(prompt, self.native_search), 'payload': payload,
                   'model': self.models[stage], 'max_tokens': max_tokens}
        if images:
            request['images'] = images
        if len(json.dumps(request, ensure_ascii=False)) > 45000:
            raise GroundingError('request_budget_exceeded', 'Material and tool context exceed the request budget.')
        def validate_response(reply):
            if not isinstance(reply, dict) or not reply.get('ok'):
                raise GroundingError('llm_failed', str(reply.get('error', 'Invalid model response'))
                                     if isinstance(reply, dict) else 'Invalid model response', True)
            return validator(reply.get('json'))
        def local_validator(value):
            try:
                validator(value)
            except (ValueError, TypeError, KeyError) as error:
                return str(error)
            return None
        async def produce():
            repair_options = {'json_format_repair': _repair_plan_json} if stage == 'plan' else {}
            return await self.model_call(self.json_client, stage=stage, **request,
                model_config_path=config_path, max_attempts=2,
                json_root_type='object', response_validator=local_validator,
                **repair_options,
                extra_create_params={'timeout': 180, 'request_timeout': 180, 'num_retries': 0})
        cache_request = copy.deepcopy(request)
        if images:
            cache_request['image_versions'] = {path: file_version(path) for path in images
                                               if not path.startswith(('http://', 'https://'))}
        return await self.store.call(stage, cache_request, produce, validate_response)

    def check_search_direction(self, guidance):
        if not (self.search_preferences or self.searches):
            return
        if self.planned_dataset_ids():
            raise GroundingError('search_not_needed',
                'Usable planned/ready routes already exist. Judge them and finish; do not search to fill a quota.')
        if self.candidates:
            raise GroundingError('unresolved_sources',
                'Develop or explicitly discard the current sources before searching again. '
                'Reserved sources are waiting for Plan, not permission to search elsewhere.')
        if not guidance.strip():
            raise GroundingError('missing_search_gap',
                'Explain the observed material gap from the previous pool and give a changed retrieval direction.')
        if guidance.strip() in self.search_preferences:
            raise GroundingError('unchanged_search_direction',
                'Use the observed material gap to change retrieval guidance, rather than repeating the previous direction.')

    async def search(self, guidance):
        self.check_search_direction(guidance)
        from tools.planner_tools.grounding.dataset_preference_tools import (
            DATASET_PREFERENCE_SYSTEM_PROMPT, CANONICAL_ROLES,
            normalize_dataset_preference,
        )
        from tools.planner_tools.grounding.dataset_retriever_tools import build_meta_query
        key = guidance.strip()
        if key in self.searches:
            self.log(f'search_datasets: cache hit; candidates={len(self.searches[key]["candidates"])}')
            return copy.deepcopy(self.searches[key])
        prompt = DATASET_PREFERENCE_SYSTEM_PROMPT.split('\nSubtask:')[0].format(
            canonical_roles=', '.join(CANONICAL_ROLES))
        payload = {'task': task_view(self.task), 'tool_capabilities': capability_view(self.tools)}
        if key:
            payload['guidance'] = key
        if key and self.preference:
            payload['previous_preference'] = self.preference
        prompt += '\nUse guidance and previous_preference, when provided, to adjust retrieval cues while preserving the target.'
        prompt += '\nUse tool_capabilities for the asset transformations actually available in this run.'
        if key in self.search_preferences:
            preference = copy.deepcopy(self.search_preferences[key])
        else:
            preference, _, _ = await self.json_call('preference', prompt, payload, normalize_dataset_preference, max_tokens=1200)
        if self.retriever is None:
            from tools.planner_tools.grounding.dataset_retriever_tools import retrieve_candidates
            retriever = retrieve_candidates
        else:
            retriever = self.retriever
        excluded = set(self.discarded) | set(self.reserved)
        pool = [card for did, card in self.cards.items() if did not in excluded]
        rows = await asyncio.to_thread(retriever, self.task, pool, preference)
        candidates = [candidate_view(row, self.cards[row['dataset_id']]) for row in rows
                      if row.get('dataset_id') in self.cards and row['dataset_id'] not in excluded]
        # Retrieval output is preserved too, including the concrete material query.
        _write(self.store.root / 'search' / (digest([key, preference, sorted(excluded)])[:24] + '.json'),
               {'guidance': key, 'preference': preference, 'material_query': build_meta_query(preference), 'rows': rows,
                **({'excluded_dataset_ids': sorted(excluded)} if excluded else {})})
        for candidate in candidates:
            previous = self.candidates.get(candidate['dataset_id'])
            if previous:
                candidate['matched_by'] = list(dict.fromkeys(previous['matched_by'] + candidate['matched_by']))
            else:
                self.statuses[candidate['dataset_id']] = 'candidate'
                self.require_direction(candidate['dataset_id'])
            self.candidates[candidate['dataset_id']] = candidate
        self.preference = preference
        self.search_preferences[key] = copy.deepcopy(preference)
        result = {'candidates': candidates}
        self.searches[key] = copy.deepcopy(result)
        self.plan_since_search = False
        self.search_discard_offset = len(_read(self.store.root / 'discard_log.json') or [])
        self.log(f'search_datasets: retrieved={len(candidates)}, candidate_pool={len(self.candidates)}')
        return result

    def candidate(self, dataset_id):
        if dataset_id in self.discarded:
            raise GroundingError('discarded_candidate', 'This dataset was removed from active memory for this subtask.')
        if dataset_id not in self.candidates or dataset_id not in self.cards:
            raise GroundingError('unknown_candidate', 'Use a dataset ID returned by search_datasets.')
        return self.cards[dataset_id]

    def is_ready(self, dataset_id):
        return (self.statuses.get(dataset_id) in ('ready', 'selected') and
                self.plan_reports.get(dataset_id, {}).get('status') == 'ready' and
                self.approvals.get(dataset_id) == self.plan_reports[dataset_id].get('blueprint_ref'))

    def require_direction(self, dataset_id):
        # Old callers/checkpoints may still mention this helper. Progress is
        # persistent; datasets no longer need a new instruction every round.
        return

    async def update_working_note(self, note):
        if not isinstance(note, str) or not note.strip():
            raise GroundingError('invalid_working_note',
                'Provide one nonempty replacement note; '
                'keep decisions, open questions and next plans, not a growing log.')
        if len(note.strip()) > WORKING_NOTE_MAX_CHARS:
            self.log(f'history length warning: {len(note.strip())} characters exceeds soft target '
                     f'{WORKING_NOTE_MAX_CHARS}; preserving full history and continuing actions')
        self.working_note = note.strip()
        self.working_note_update_required = False
        _write(self.store.root / 'working_note.json', {'note': self.working_note})
        _write(self.store.root / 'state.json', self.checkpoint())
        self.log(f'working note updated: {len(self.working_note)} characters')
        return {'status': 'updated'}

    def current_blueprint(self, dataset_id):
        self.candidate(dataset_id)
        current = self.plan_reports.get(dataset_id) or {}
        blueprint = self.blueprints.get(current.get('blueprint_ref'))
        if (current.get('status') != 'ready' or not blueprint or blueprint.get('dataset_id') != dataset_id
                or not manifest_valid(blueprint.get('manifest') or {})):
            raise GroundingError('invalid_source', 'Source must have a current ready blueprint before main-agent approval.')
        error = self.operation_errors.get(dataset_id, {})
        if not self.is_ready(dataset_id) and error.get('operation') == 'plan_dataset_transformation':
            raise GroundingError('unresolved_plan_error', 'Retry the failed Plan with guidance before accepting its route.')
        return current

    def check_ready_ids(self, dataset_ids):
        self.check_candidate_ids(dataset_ids)
        for did in dataset_ids:
            self.current_blueprint(did)

    def activate(self, dataset_id):
        if dataset_id in self.reserved:
            self.statuses[dataset_id] = self.reserved.pop(dataset_id)
            self.searches.clear()

    def check_candidate_ids(self, dataset_ids):
        if (not isinstance(dataset_ids, list) or not dataset_ids or
                any(not isinstance(did, str) or not did.strip() for did in dataset_ids)
                or len(set(dataset_ids)) != len(dataset_ids)):
            raise GroundingError('invalid_dataset_ids', 'Provide a nonempty list of unique dataset IDs.')
        for did in dataset_ids:
            self.candidate(did)

    async def manage(self, dataset_ids, action):
        self.check_candidate_ids(dataset_ids)
        if action not in ('reserve', 'activate', 'ready'):
            raise GroundingError('invalid_candidate_action', 'Use reserve, activate or ready.')
        if action == 'ready':
            self.check_ready_ids(dataset_ids)
        elif action == 'reserve' and any(self.is_ready(did) for did in dataset_ids):
            raise GroundingError('source_already_ready', 'Ready sources are accepted and retained; reserve only sources waiting for Plan.')
        for did in dataset_ids:
            if action == 'reserve':
                if did not in self.reserved:
                    self.reserved[did] = self.statuses.get(did, 'candidate')
                self.statuses[did] = 'reserved'
                self.pending_decisions = [value for value in self.pending_decisions if value != did]
            elif action == 'activate':
                self.activate(did)
                self.require_direction(did)
            else:
                self.reserved.pop(did, None)
                self.approvals[did] = self.plan_reports[did]['blueprint_ref']
                self.statuses[did] = 'ready'
                self.operation_errors.pop(did, None)
        self.searches.clear()
        self.log(f'manage_candidates: {action}; datasets={", ".join(dataset_ids)}')
        _write(self.store.root / 'state.json', self.checkpoint())
        return {'status': {'reserve': 'reserved', 'activate': 'activated', 'ready': 'ready'}[action],
                'dataset_ids': list(dataset_ids)}

    def check_discard_ids(self, dataset_ids):
        if (not isinstance(dataset_ids, list) or not dataset_ids or
                any(not isinstance(did, str) or not did for did in dataset_ids) or
                len(set(dataset_ids)) != len(dataset_ids)):
            raise GroundingError('invalid_discard_ids', 'Provide a nonempty list of unique candidate ids.')
        for did in dataset_ids:
            if did not in self.discarded:
                self.candidate(did)

    def check_discard_decisions(self, datasets):
        if not isinstance(datasets, list) or not datasets:
            raise GroundingError('invalid_discard_ids', 'Provide a nonempty array of {dataset_id, reason}.')
        if any(not isinstance(item, dict) or set(item) != {'dataset_id', 'reason'} for item in datasets):
            raise GroundingError('invalid_discard_reason', 'Each discarded dataset needs dataset_id and a concise reason.')
        if any(not isinstance(item['reason'], str) or not item['reason'].strip() for item in datasets):
            raise GroundingError('invalid_discard_reason', 'Give a nonempty decision reason for every discarded dataset.')
        dataset_ids = [item['dataset_id'] for item in datasets]
        self.check_discard_ids(dataset_ids)
        return dataset_ids

    async def discard(self, datasets):
        dataset_ids = self.check_discard_decisions(datasets)
        audit_path = self.store.root / 'discard_log.json'
        audit = _read(audit_path)
        if audit is None and audit_path.exists():
            raise GroundingError('invalid_discard_log', 'Cannot read the existing discard audit log; preserve it before retrying.')
        if audit is not None and not isinstance(audit, list):
            raise GroundingError('invalid_discard_log', 'The discard audit log must be an array.')
        timestamp = datetime.now().astimezone().isoformat(timespec='milliseconds')
        entries = [{'time': timestamp, 'dataset_id': item['dataset_id'], 'reason': item['reason'].strip(),
                    'previous_status': self.statuses.get(item['dataset_id'], 'discarded')}
                   for item in datasets]
        _write(audit_path, (audit or []) + entries)
        for did in dataset_ids:
            if did not in self.discarded:
                self.discarded.append(did)
            for collection in (self.candidates, self.assessments, self.plan_reports, self.materials,
                               self.statuses, self.reserved, self.assessment_answers, self.operation_errors, self.approvals):
                collection.pop(did, None)
        self.pending_decisions = [did for did in self.pending_decisions if did not in self.discarded]
        self.blueprints = {ref: value for ref, value in self.blueprints.items()
                           if value['dataset_id'] not in self.discarded}
        # The next identical search can reuse its preference but refresh the reduced dataset pool.
        self.searches.clear()
        self.log(f'discarded {len(dataset_ids)} dataset(s): {", ".join(dataset_ids)}; '
                 f'active_candidates={len(self.candidates)}')
        for entry in entries:
            self.log(f'discarded dataset={entry["dataset_id"]}: {entry["reason"]}')
        _write(self.store.root / 'state.json', self.checkpoint())
        return {'status': 'discarded', 'dataset_ids': list(dataset_ids)}

    async def read_assessment(self, dataset_id, report_type='assessment'):
        self.candidate(dataset_id)
        if report_type == 'blueprint':
            report = self.plan_reports.get(dataset_id)
            if not report:
                raise GroundingError('missing_blueprint', 'This source has no Plan report.')
            return {'dataset_id': dataset_id, 'status': self.statuses.get(dataset_id, 'planned'),
                    'report': {key: copy.deepcopy(report[key]) for key in ('blueprint', 'reason') if key in report}}
        if report_type != 'assessment':
            raise GroundingError('invalid_report_type', 'Use assessment or blueprint.')
        report = self.assessments.get(dataset_id)
        if not report:
            raise GroundingError('missing_assessment', 'This dataset has no saved assessment; ask a question first.')
        visible = {key: copy.deepcopy(report[key]) for key in ('answer', 'summary', 'limitations') if key in report}
        if report.get('_question'):
            visible['question'] = report['_question']
        self.log(f'read_assessment dataset={dataset_id}: report cache hit (no model call)')
        return {'dataset_id': dataset_id, 'status': self.statuses.get(dataset_id, 'assessed'), 'report': visible}

    async def material(self, dataset_id):
        card = self.candidate(dataset_id)
        previous = self.materials.get(dataset_id)
        if previous and manifest_valid(previous['manifest']):
            return previous
        try:
            packet = await asyncio.to_thread(read_material, dataset_id, card, self.task,
                                             sample_count=self.context.get('grounding_sample_count', 3))
        except (OSError, ValueError, TypeError) as error:
            failure = GroundingError('material_read_failed', str(error), True)
            _write(self.store.root / 'material_errors' / (digest(dataset_id)[:16] + '.json'),
                   {'dataset_id': dataset_id, **_error(failure)})
            raise failure from error
        packet['material_ref'] = 'materials/' + _safe(dataset_id) + '/' + digest(packet)[:20]
        _write(self.store.root / 'materials' / (digest(packet['material_ref'])[:24] + '.json'), packet)
        self.materials[dataset_id] = packet
        self.log(f'material dataset={dataset_id}: samples={packet["sample_indexes"]}, '
                 f'images={len(packet["images"])}')
        return packet

    async def assess(self, dataset_id, question):
        self.candidate(dataset_id)
        if self.is_ready(dataset_id):
            raise GroundingError('source_already_ready', 'Ready sources are accepted; do not Assess or Plan them again.')
        if not isinstance(question, str) or not question.strip():
            raise GroundingError('missing_question', 'Ask a concrete question; use read_assessment for a local reread.')
        question = question.strip()
        self.activate(dataset_id)
        material = await self.material(dataset_id)
        previous = self.assessments.get(dataset_id)
        if (previous and previous.get('_question') == question
                and previous.get('material_ref') == material['material_ref'] and previous.get('answer')):
            self.statuses[dataset_id] = 'planned' if self.statuses.get(dataset_id) == 'planned' else 'assessed'
            self.require_direction(dataset_id)
            self.log(f'assess dataset={dataset_id}: cache hit (same question)')
            return {key: copy.deepcopy(value) for key, value in previous.items() if not key.startswith('_')}
        payload = {'task': task_view(self.task), 'material': material['material'], 'tools': self.tools}
        if self.context.get('description'):
            payload['benchmark_goal'] = self.context['description']
        if question:
            payload['question'] = question
        if (question and previous and previous.get('material_ref') == material['material_ref']
                and previous.get('_question') != question):
            payload['previous_assessment'] = {k: v for k, v in previous.items() if not k.startswith('_')}
        report, key, _ = await self.json_call('assess', ASSESS_PROMPT, payload,
            lambda reply: _assessment(reply, material['material']), material['images'])
        version = digest({'request': key, 'report': report})
        report.update(dataset_id=dataset_id, assessment_ref=f'grounding/{_safe(self.sid)}/{_safe(dataset_id)}/assessment/{version[:20]}',
                      material_ref=material['material_ref'])
        self.assessments[dataset_id] = {**report, '_question': question}
        answers = [item for item in self.assessment_answers.get(dataset_id, []) if item['question'] != question]
        # Answers tied to replaced source material are not valid carry-forward facts.
        if previous and previous.get('material_ref') != material['material_ref']:
            answers = []
        self.assessment_answers[dataset_id] = answers + [{'question': question, 'answer': report['answer']}]
        self.statuses[dataset_id] = 'planned' if self.statuses.get(dataset_id) == 'planned' else 'assessed'
        self.require_direction(dataset_id)
        return report

    def planned_dataset_ids(self):
        return {did for did, report in self.plan_reports.items()
                if did in self.candidates and report.get('status') == 'ready'}

    async def plan(self, dataset_id, guidance=''):
        self.candidate(dataset_id)
        if self.is_ready(dataset_id):
            raise GroundingError('source_already_ready', 'Ready sources are accepted; revise only unaccepted planned routes.')
        if len(self.planned_dataset_ids() | {dataset_id}) > MAX_PLANNED_DATASETS:
            raise GroundingError('dataset_limit_exceeded',
                f'Keep at most {MAX_PLANNED_DATASETS} dataset routes per subtask; '
                'revise an existing route or discard one before planning another dataset.')
        self.activate(dataset_id)
        material = await self.material(dataset_id)
        payload = {'task': task_view(self.task), 'material': material['material'], 'tools': self.tools}
        if self.context.get('description'):
            payload['benchmark_goal'] = self.context['description']
        assessment = self.assessments.get(dataset_id)
        if assessment and assessment.get('material_ref') == material['material_ref']:
            payload['assessment'] = {k: v for k, v in assessment.items() if not k.startswith('_')}
        if guidance:
            payload['guidance'] = guidance
        previous = self.plan_reports.get(dataset_id)
        saved = self.blueprints.get(previous.get('blueprint_ref')) if previous else None
        if (saved and previous.get('material_ref') == material['material_ref']
                and previous.get('_guidance', '') == guidance
                and previous.get('_assessment_ref') == (assessment or {}).get('assessment_ref')
                and manifest_valid(saved.get('manifest') or {})):
            self.statuses[dataset_id] = 'planned'
            self.require_direction(dataset_id)
            self.log(f'plan dataset={dataset_id}: cache hit (current blueprint)')
            self.plan_since_search = True
            self.operation_errors.pop(dataset_id, None)
            return {key: copy.deepcopy(value) for key, value in previous.items() if not key.startswith('_')}
        if previous and previous.get('status') == 'ready':
            payload['previous_blueprint'] = previous['blueprint']
        elif previous and previous.get('reason'):
            payload['previous_blocker'] = previous['reason']
        try:
            report, key, _ = await self.json_call('plan', BLUEPRINT_PROMPT, payload,
                lambda reply: _blueprint(reply, material['material'], self.tools), material['images'])
        except Exception as error:
            failure = _error(error)['error']
            self.operation_errors[dataset_id] = {'operation': 'plan_dataset_transformation',
                'code': failure['code'], 'message': brief_text(failure['message'], 400),
                'retryable': failure.get('retryable', True)}
            raise
        self.operation_errors.pop(dataset_id, None)
        report.update(dataset_id=dataset_id, material_ref=material['material_ref'])
        if report['status'] == 'ready':
            ref = f'grounding/{_safe(self.sid)}/{_safe(dataset_id)}/blueprint/{key[:20]}'
            report['blueprint_ref'] = ref
            self.blueprints[ref] = {**report, 'manifest': material['manifest']}
        self.plan_reports[dataset_id] = {**copy.deepcopy(report), '_guidance': guidance,
                                         '_assessment_ref': (assessment or {}).get('assessment_ref')}
        self.plan_since_search = True
        if report['status'] == 'ready':
            self.statuses[dataset_id] = 'planned'
        else:
            self.statuses[dataset_id] = ('planned' if self.statuses.get(dataset_id) == 'planned' else
                                         'assessed' if dataset_id in self.assessments else 'candidate')
            self.operation_errors[dataset_id] = {'operation': 'plan_dataset_transformation', 'code': 'blueprint_blocked',
                                                'message': brief_text(report['reason'], 400), 'retryable': False}
        self.require_direction(dataset_id)
        return report

    def finish_sources(self, dataset_ids, *, discarded_ids=(), approved_ids=(), require_resolved=True):
        if not dataset_ids:
            raise GroundingError('missing_sources', 'ready requires at least one selected source.')
        if (not isinstance(dataset_ids, list) or any(not isinstance(did, str) for did in dataset_ids)
                or len(set(dataset_ids)) != len(dataset_ids)):
            raise GroundingError('invalid_source', 'Select unique dataset IDs only.')
        if len(dataset_ids) > MAX_PLANNED_DATASETS:
            raise GroundingError('dataset_limit_exceeded', f'Select at most {MAX_PLANNED_DATASETS} datasets per subtask.')
        if set(dataset_ids).intersection(discarded_ids):
            raise GroundingError('conflicting_selection', 'A source cannot be both selected and discarded; no finalization actions executed.')
        selected = []
        for did in dataset_ids:
            current = self.current_blueprint(did)
            if not self.is_ready(did) and did not in approved_ids:
                raise GroundingError('source_not_accepted', 'Judge planned routes and explicitly mark them ready before Finish.')
            selected.append({'dataset_id': did, 'blueprint_ref': current['blueprint_ref']})
        if require_resolved and set(self.candidates) - set(discarded_ids) != set(dataset_ids):
            raise GroundingError('unresolved_candidates', 'Discard unselected active/reserved sources before finishing: '
                                 + ', '.join(sorted(set(self.candidates) - set(discarded_ids) - set(dataset_ids))))
        return selected

    async def finish(self, status, reason, dataset_ids=None):
        if status not in ('ready', 'incomplete') or not isinstance(reason, str) or not reason.strip():
            raise GroundingError('invalid_finish', 'Use ready/incomplete and a concise reason.')
        if status == 'ready' and not dataset_ids:
            raise GroundingError('missing_sources', 'ready requires at least one returned ready blueprint.')
        if dataset_ids is not None and (not isinstance(dataset_ids, list) or
                any(not isinstance(did, str) for did in dataset_ids) or len(set(dataset_ids)) != len(dataset_ids)):
            raise GroundingError('invalid_source', 'Select unique dataset IDs only.')
        if len(dataset_ids or []) > MAX_PLANNED_DATASETS:
            raise GroundingError('dataset_limit_exceeded',
                f'Select at most {MAX_PLANNED_DATASETS} datasets per subtask.')
        selected = self.finish_sources(dataset_ids, require_resolved=status == 'ready') if dataset_ids else []
        result = {'subtask_id': self.sid, 'status': status, 'reason': reason.strip()}
        if selected:
            result['sources'] = selected
        self.result = result
        for source in selected:
            self.activate(source['dataset_id'])
            self.statuses[source['dataset_id']] = 'selected'
        self.pending_decisions = []
        return result

    async def execute_round(self, calls, *, history_recorded=False):
        """Batch independent calls; validate final selections before any cleanup."""
        progress_before = self._progress_digest()
        visible_blueprints = {item['dataset_id']: item['blueprint']
                              for item in self.view().get('state', {}).get('candidates', [])
                              if 'blueprint' in item}
        names = [call['function'].get('name') for call in calls]
        self.log(f'dispatch {len(calls)} tool call(s): ' + ', '.join(str(name) for name in names))
        stages = set(names).intersection(('search_datasets', 'assess_dataset',
                                         'plan_dataset_transformation', 'finish_grounding'))
        mixed = len(stages) > 1 or names.count('search_datasets') > 1 or names.count('finish_grounding') > 1
        schemas = {tool['function']['name']: tool['function']['parameters'] for tool in self.action_schemas}
        prepared, occupied = [], set()
        remaining = copy.deepcopy(self.budget)
        for call in calls:
            name, args, ids = call['function'].get('name'), {}, []
            try:
                if mixed and name != 'update_working_note':
                    raise GroundingError('dependent_calls', 'Run search, assessment, planning and finish in separate rounds; local decisions may accompany them.')
                if name not in self.functions:
                    raise GroundingError('unknown_tool', 'Use a registered Grounding tool.')
                raw = call['function'].get('arguments', '{}')
                args = json.loads(raw) if isinstance(raw, str) else raw
                _check_arguments(args, schemas[name])
                if name == 'search_datasets':
                    self.check_search_direction(args.get('guidance', ''))
                if args.get('dataset_id'):
                    self.candidate(args['dataset_id'])
                    ids = [args['dataset_id']]
                elif name == 'discard_datasets':
                    ids = self.check_discard_decisions(args['datasets'])
                elif name == 'manage_candidates':
                    self.check_candidate_ids(args['dataset_ids'])
                    ids = args['dataset_ids']
                    if args['action'] == 'ready':
                        self.check_ready_ids(ids)
                    elif args['action'] == 'reserve' and any(self.is_ready(did) for did in ids):
                        raise GroundingError('source_already_ready', 'Reserve sources waiting for Plan, not accepted ready routes.')
                elif name == 'finish_grounding':
                    if not args['reason'].strip():
                        raise GroundingError('missing_reason', 'Explain the final outcome briefly.')
                    ids = args.get('dataset_ids') or []
                    if ids:
                        self.check_candidate_ids(ids)
                if name in ('assess_dataset', 'plan_dataset_transformation') and self.is_ready(args['dataset_id']):
                    raise GroundingError('source_already_ready', 'Ready sources are accepted; do not Assess or Plan them again.')
                if name == 'assess_dataset' and not args['question'].strip():
                    raise GroundingError('missing_question', 'Ask a concrete question; use read_assessment for cached detail.')
                if name == 'read_assessment':
                    collection = self.plan_reports if args.get('report_type') == 'blueprint' else self.assessments
                    if args['dataset_id'] not in collection:
                        raise GroundingError('missing_report', 'No saved report of the requested type exists for this source.')
                if name == 'update_working_note' and names.count(name) > 1:
                    raise GroundingError('duplicate_note_update', 'Replace the working note at most once in a response.')
                # Finish reads the selected IDs after local acceptance/cleanup.
                if name != 'finish_grounding':
                    if occupied.intersection(ids):
                        raise GroundingError('duplicate_dataset', 'Give each dataset one independent action per round; Assess/Plan already activate reserves.')
                    occupied.update(ids)
                key = _BUDGET_KEY.get(name)
                if key:
                    if remaining[key] <= 0:
                        raise GroundingError('budget_exhausted', f'No remaining budget for {name}.')
                    remaining[key] -= 1
                prepared.append((call, args, list(ids), None))
            except Exception as error:
                prepared.append((call, args if isinstance(args, dict) else {}, list(ids), _error(error)))

        discarded_ids = {did for call, _, ids, invalid in prepared
                         if call['function'].get('name') == 'discard_datasets' and invalid is None for did in ids}
        planned_ids = self.planned_dataset_ids() - discarded_ids
        for index, (call, args, ids, invalid) in enumerate(prepared):
            if invalid is None and call['function'].get('name') == 'plan_dataset_transformation':
                if len(planned_ids | set(ids)) > MAX_PLANNED_DATASETS:
                    prepared[index] = (call, args, ids, _error(GroundingError('dataset_limit_exceeded',
                        f'Keep at most {MAX_PLANNED_DATASETS} planned/ready dataset routes per subtask.')))
                    remaining['plans_left'] += 1
                else:
                    planned_ids.update(ids)

        final = [(index, args) for index, (call, args, _, _) in enumerate(prepared)
                 if call['function'].get('name') == 'finish_grounding' and args.get('status') == 'ready']
        if final:
            related = [index for index, (call, args, _, _) in enumerate(prepared)
                       if call['function'].get('name') in ('finish_grounding', 'discard_datasets') or
                       call['function'].get('name') == 'manage_candidates' and args.get('action') == 'ready']
            try:
                invalid = next((prepared[index][3] for index in related if prepared[index][3]), None)
                if invalid:
                    raise GroundingError('finalization_rejected', 'No acceptance/cleanup/Finish executed: ' + invalid['error']['message'])
                approved_ids = {did for call, args, ids, _ in prepared
                                if call['function'].get('name') == 'manage_candidates' and args.get('action') == 'ready' for did in ids}
                self.finish_sources(final[0][1].get('dataset_ids'), discarded_ids=discarded_ids, approved_ids=approved_ids)
            except Exception as error:
                failure = _error(error)
                for index in related:
                    call, args, ids, _ = prepared[index]
                    prepared[index] = (call, args, ids, copy.deepcopy(failure))
        self.budget.update(remaining)

        note_changes = set()

        async def execute(item):
            call, args, ids, invalid = item
            name = call['function'].get('name')
            did = args.get('dataset_id')
            label = str(name) + (f' dataset={did}' if did else '')
            started = time.monotonic()
            note_before = (self._note_state_digest()
                           if name not in ('update_working_note', 'finish_grounding') else None)
            if invalid:
                result = invalid
            else:
                detail = args.get('question') or args.get('guidance') or args.get('action') or args.get('reason') or ', '.join(ids)
                self.log(f'{label}: started' + (f'; {_brief(detail)}' if detail else ''))
                try:
                    result = await self.functions[name](**args, context_variables={'grounding_session': self})
                except Exception as error:
                    result = _error(error)
            error_key = did or '__' + str(name)
            if 'error' in result:
                error = result['error']
                self.operation_errors[error_key] = {'operation': name, 'code': error['code'],
                    'message': brief_text(error['message'], 400), 'retryable': error.get('retryable', False)}
                self.failures.append(error)
                summary = f'failed; {error["code"]}: {_brief(error["message"])}'
            else:
                if result.get('status') != 'blocked':
                    self.operation_errors.pop(error_key, None)
                status = self.statuses.get(did) or result.get('status') or 'completed'
                finding = result.get('answer') or result.get('reason')
                summary = status + (f'; {_brief(finding)}' if finding else '')
            if note_before is not None and note_before != self._note_state_digest():
                # Local decisions precede note updates; parallel remote returns
                # follow them and can only set, never clear, a peer's pending flag.
                self.working_note_update_required = True
                note_changes.add(name)
            self.log(f'{label}: {summary} ({time.monotonic() - started:.1f}s)')
            return result

        note_was_pending = self.working_note_update_required
        results = [None] * len(prepared)
        # Clean only unselected IDs, then accept, read/update the note, then finish.
        for group in ({'discard_datasets'}, {'manage_candidates'}, {'read_assessment', 'update_working_note'}):
            for index, item in enumerate(prepared):
                if item[0]['function'].get('name') in group:
                    results[index] = await execute(item)
        remote = [index for index, value in enumerate(results) if value is None]
        for index, result in zip(remote, await asyncio.gather(*(execute(prepared[index]) for index in remote))):
            results[index] = result

        note_updated = history_recorded or any(item[0]['function'].get('name') == 'update_working_note'
                           and result.get('status') == 'updated'
                           for item, result in zip(prepared, results))
        local_note_changes = note_changes - {'search_datasets', 'assess_dataset', 'plan_dataset_transformation'}
        if (note_was_pending or local_note_changes) and not note_updated:
            self.log('working note update missed: update_working_note is required alongside '
                     'the next decision; keeping the pending flag')

        messages = []
        for (call, args, ids, _), result in zip(prepared, results):
            name = call['function'].get('name')
            error = result.get('error')
            status = (error['code'] if error else
                      'blocked' if result.get('status') == 'blocked' else
                      self.statuses.get(args.get('dataset_id')) or result.get('status') or 'completed')
            attempt = {'tool': name, 'result': status}
            if args.get('dataset_id'):
                attempt['dataset_id'] = args['dataset_id']
            elif ids:
                attempt['dataset_ids'] = ids
            self.attempts.append(attempt)
            if error:
                public = {'status': 'error', 'error': {key: copy.deepcopy(error[key])
                          for key in ('code', 'message', 'retryable') if key in error}}
                if args.get('dataset_id'):
                    public['dataset_id'] = args['dataset_id']
            elif name == 'read_assessment':
                public = result
            elif name == 'finish_grounding':
                public = {'status': result['status']}
                if result.get('sources'):
                    public['dataset_ids'] = [source['dataset_id'] for source in result['sources']]
            elif args.get('dataset_id'):
                public = {'dataset_id': args['dataset_id'], 'status': status}
            else:
                public = {'status': result.get('status', 'searched')}
                if ids:
                    public['dataset_ids'] = ids
            messages.append({'role': 'tool', 'tool_call_id': call['id'], 'name': name,
                             'content': json.dumps(public, ensure_ascii=False)})
        self._record_round_feedback(prepared, results, progress_before, visible_blueprints)
        return messages

    def manifest(self):
        result = {}
        for packet in self.materials.values():
            result.update(packet['manifest'])
        for packet in self.blueprints.values():
            result.update(packet.get('manifest') or {})
        return result

    def checkpoint(self):
        value = {'state': self.snapshot(), 'manifest': self.manifest(), 'result': self.result}
        return {**value, 'digest': digest(value)}

    async def run(self):
        started = time.monotonic()
        max_steps = self.budget['steps_left']
        while self.budget['steps_left'] > 0 and self.result is None:
            self.budget['steps_left'] -= 1
            self.log(f'round={max_steps - self.budget["steps_left"]}/{max_steps}; '
                     f'candidates={len(self.candidates)} '
                     f'(active={len(self.candidates) - len(self.reserved)}, reserved={len(self.reserved)}), '
                     f'assessed={len(self.assessments)}, '
                     f'blueprints={len(self.blueprints)}, '
                     f'note_update_required={self.working_note_update_required}')
            messages = [{'role': 'system', 'content': _native_prompt(GROUNDING_PROMPT, self.native_search)},
                        {'role': 'user', 'content': json.dumps(self.view(), ensure_ascii=False)}] + self.recent_tool_context()
            request = {'model': self.models['main'], 'messages': messages, 'tools': self.schemas,
                       'tool_choice': 'required', 'parallel_tool_calls': False}
            async def produce():
                # Request artifacts intentionally omit API credentials.
                return await self.model_call(self.decide, model=self.models['main'], messages=messages,
                    tools=self.schemas, model_config_path=self.context.get('model_config_path'),
                    _require_history=bool(self.attempts or self.working_note))
            try:
                def validate(reply):
                    message = _main_message(reply)
                    if ('decision' in message and not message['decision']['history']
                            and (self.attempts or self.working_note)):
                        raise ValueError('Include prior confirmed history after the initial decision.')
                    return message
                message, _, _ = await self.store.call('main', request, produce, validate)
            except Exception as error:
                self.failures.append(_error(error)['error'])
                break
            if message['tool_calls']:
                if 'decision' in message:
                    history = message['decision']['history']
                    self.history_saved = bool(history)
                    if history:
                        await self.update_working_note(history)
                outputs = await self.execute_round(message['tool_calls'], history_recorded='decision' in message)
                self.last_round = [message] + outputs
            else:
                self.log('main: no tool calls; requesting a tool decision next round')
                self.last_round = [{'role': 'assistant', 'content': message.get('content') or ''},
                                   {'role': 'user', 'content': 'Use the registered tools to complete or finish Grounding.'}]
            _write(self.store.root / 'state.json', self.checkpoint())
        if self.result is None:
            reason = self.failures[-1]['message'] if self.failures else 'Grounding decision budget exhausted.'
            self.result = {'subtask_id': self.sid, 'status': 'incomplete', 'reason': reason,
                           'retryable': bool(self.failures)}
        elif self.result['status'] == 'incomplete' and self.failures:
            self.result['retryable'] = any(error.get('retryable') for error in self.failures)
        _write(self.store.root / 'state.json', self.checkpoint())
        self.log(f'finished: status={self.result["status"]}, sources={len(self.result.get("sources", []))}, '
                 f'elapsed={time.monotonic() - started:.1f}s; reason={_brief(self.result["reason"])}')
        return self.result


class AdaptiveGroundingModule:
    """AgentModule-compatible boundary; N tasks have N independent main agents."""
    def __init__(self, cache_path, model=None, model_config_path=None, *, decide=None,
                 json_client=None, models=None, retriever=None, logger=None, cache_selector=None):
        self.cache_path, self.model, self.model_config_path = Path(cache_path), model, model_config_path
        self.decide, self.json_client = decide or _default_decide, json_client or _default_json
        self.models, self.retriever = models, retriever
        self.logger = logger
        self.cache_selector = cache_selector

    def log(self, message, subtask_id=None):
        _progress(message, subtask_id, self.logger)

    def has_cache(self):
        if (self.cache_path / 'agents/grounding_agent.json').is_file():
            return True
        base = self.cache_path / 'grounding'
        return base.is_dir() and any(path.is_dir() and path.name != '_history' and
            any(path.rglob('*.json')) for path in base.iterdir())

    def completed_cache(self):
        path = self.cache_path / 'agents/grounding_agent.json'
        packet = _read(path)
        if not isinstance(packet, dict) or packet.get('version') != GROUNDING_VERSION:
            return None
        context = packet.get('context_variables') or {}
        outcome, tasks = context.get('grounding_result') or {}, context.get('subtasks')
        if outcome.get('status') != 'accepted' or not isinstance(tasks, list) or not tasks:
            return None
        if any(not isinstance(task, dict) or (task.get('grounding') or {}).get('status') != 'ready'
               for task in tasks):
            return None
        # An interrupted newer run may have checkpoints newer than the previous summary.
        base = self.cache_path / 'grounding'
        states = list(base.glob('*/state.json')) + list(base.glob(f'{GROUNDING_VERSION}/*/*/state.json'))
        if any(state.stat().st_mtime_ns > path.stat().st_mtime_ns for state in states):
            return None
        return packet

    async def __call__(self, messages, context_variables, force_recompute=False, cache_mode=None, **kwargs):
        from tools.planner_tools.grounding.dataset_preference_tools import DATASET_PREFERENCE_SYSTEM_PROMPT
        mode = 'fresh' if force_recompute else (cache_mode or
            (self.cache_selector(self.has_cache()) if self.cache_selector else 'auto'))
        if mode not in ('auto', 'use', 'resume', 'fresh'):
            raise ValueError('Grounding cache mode must be auto/use/resume/fresh.')
        explicit_reuse = mode in ('use', 'resume')
        force_recompute = mode == 'fresh'
        self.log(f'cache choice: {mode}')
        if explicit_reuse:
            packet = self.completed_cache()
            if packet is not None:
                saved = packet['context_variables']
                for task in saved['subtasks']:
                    legacy = self.cache_path / 'grounding' / GROUNDING_VERSION / _safe(task['id'])
                    if legacy.is_dir():
                        _prepare_cache_directory(self.cache_path, task['id'], None,
                            lambda message, sid=task['id']: self.log(message, sid), reuse_existing=True)
                context_variables.update(copy.deepcopy(saved))
                context_variables.pop('grounding_feedback', None)
                self.log('using selected completed cache; skipping Grounding model calls and signature invalidation')
                return list(messages) + copy.deepcopy(packet.get('messages') or []), context_variables
        context = {**context_variables, 'model_config_path': self.model_config_path}
        started = time.monotonic()
        self.log(f'start: subtasks={len(context.get("subtasks") or [])}, '
                 f'model_concurrency={max(1, int(context.get("max_workers_grounding", 4)))}, '
                 f'force_recompute={force_recompute}')
        models = self.models or _models(self.model_config_path, self.model)
        raw_cards = context.get('id2card') or context.get('dataset_cards') or {}
        if isinstance(raw_cards, list):
            cards = {str(card['dataset_id']): card for card in raw_cards if isinstance(card, dict) and card.get('dataset_id')}
        else:
            cards = {str(key): value for key, value in raw_cards.items() if isinstance(value, dict)}
        versions = await asyncio.to_thread(source_identity, cards)
        gate = asyncio.Semaphore(max(1, int(context.get('max_workers_grounding', 4))))
        tasks = context.get('subtasks') or []
        if len({str(task.get('id')) for task in tasks}) != len(tasks):
            raise ValueError('Grounding requires unique subtask IDs.')
        sessions = []

        async def one(task):
            progress = lambda message: self.log(message, task['id'])
            progress(f'start: {_brief(task.get("name") or task["id"])}')
            signature = digest({'version': GROUNDING_VERSION, 'task': task_view(task),
                'retrieval': {key: task.get(key) for key in ('id', 'modalities', 'keywords', 'domain', 'task')},
                'goal': context.get('description'), 'cards': cards, 'sources': versions,
                'tools': asset_tools(context, task), 'models': models,
                'model_config_path': str(self.model_config_path),
                'native_search': int(context.get('max_web_searches', 3)) > 0,
                'sample_count': context.get('grounding_sample_count', 3),
                'budgets': context.get('grounding_budgets') or DEFAULT_BUDGETS,
                'prompts': [GROUNDING_PROMPT, ASSESS_PROMPT, BLUEPRINT_PROMPT, DATASET_PREFERENCE_SYSTEM_PROMPT],
                'tool_schemas': [decision_schema(WORKING_NOTE_MAX_CHARS)]})
            root = _prepare_cache_directory(self.cache_path, task['id'], signature, progress,
                                           reuse_existing=explicit_reuse, fresh=force_recompute)
            store = StageStore(root, resume=not force_recompute, progress=progress)
            session = GroundingSession(task, context, cards, store, models, gate,
                                       self.decide, self.json_client, self.retriever, progress=progress)
            sessions.append(session)
            saved = _read(root / 'state.json') if not force_recompute else None
            valid_digest = (isinstance(saved, dict) and saved.get('digest') ==
                            digest({key: value for key, value in saved.items() if key != 'digest'}))
            if valid_digest and (explicit_reuse or manifest_valid(saved.get('manifest'))):
                session.restore(saved.get('state'))
                result = saved.get('result')
                if isinstance(result, dict) and result.get('status') == 'ready':
                    session.result = result
                    progress(f'cache hit: status=ready, sources={len(result.get("sources", []))}; '
                             'skipping model calls')
                    return result
                progress(f'resuming saved progress: candidates={len(session.candidates)}, '
                         f'assessed={len(session.assessments)}, blueprints={len(session.blueprints)}')
            return await session.run()

        results = await asyncio.gather(*(one(task) for task in tasks))
        session_by_id = {session.sid: session for session in sessions}
        updated = []
        for task in tasks:
            item = copy.deepcopy(task)
            session = session_by_id[str(task['id'])]
            result = session.result
            for key in ('scored_candidates', 'scored_status', 'transformability', 'selected_candidate_ids',
                        'candidate_selection_done', 'grounding_blueprints'):
                item.pop(key, None)
            item['grounding'] = {'version': GROUNDING_VERSION, **copy.deepcopy(result)}
            item['dataset_preference'] = copy.deepcopy(session.preference)
            item['retrieval_result'] = list(session.candidates.values())
            item['retrieval_searched'] = bool(session.searches)
            selected = result.get('sources') or []
            item['grounding_blueprints'] = {source['dataset_id']: copy.deepcopy(session.blueprints[source['blueprint_ref']])
                                             for source in selected}
            updated.append(item)
        accepted = bool(results) and all(result['status'] == 'ready' for result in results)
        outcome = {'version': GROUNDING_VERSION, 'status': 'accepted' if accepted else 'incomplete', 'subtasks': results}
        if not accepted:
            outcome['reason'] = '; '.join(f"{result['subtask_id']}: {result['reason']}"
                                          for result in results if result['status'] != 'ready') or 'No subtasks to ground.'
            outcome['retryable'] = any(result.get('retryable', False) for result in results)
        context_variables.update(subtasks=updated, grounding_result=outcome)
        if accepted:
            context_variables.pop('grounding_feedback', None)
        else:
            context_variables['grounding_feedback'] = outcome['reason']
        traces = [message for session in sessions for message in session.last_round]
        _write(self.cache_path / 'agents' / 'grounding_agent.json', {'version': GROUNDING_VERSION,
            'messages': traces, 'context_variables': {'subtasks': updated, 'grounding_result': outcome}})
        self.log(f'finished: status={outcome["status"]}, '
                 f'ready={sum(result["status"] == "ready" for result in results)}/{len(results)}, '
                 f'elapsed={time.monotonic() - started:.1f}s; '
                 f'cache={self.cache_path / "agents" / "grounding_agent.json"}')
        return list(messages) + traces, context_variables
