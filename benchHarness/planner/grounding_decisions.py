"""One model decision contains history plus business actions, never a note action."""
import copy
import json

from tools.planner_tools.grounding.adaptive_tools import grounding_tool_schemas


DECISION_TOOL = 'submit_grounding_decision'


def business_schemas():
    return [item for item in grounding_tool_schemas()
            if item['function']['name'] != 'update_working_note']


def decision_schema(history_limit=1200):
    variants = []
    for item in business_schemas():
        function = item['function']
        variants.append({'type': 'object', 'description': function['description'],
            'properties': {'name': {'type': 'string', 'enum': [function['name']]},
                           'arguments': copy.deepcopy(function['parameters'])},
            'required': ['name', 'arguments'], 'additionalProperties': False})
    return {'type': 'function', 'function': {'name': DECISION_TOOL,
        'description': 'Return one complete Grounding decision: compact prior history and the actual '
            'business actions you choose now. History is saved automatically, not executed as an action. '
            'Use an empty history only for the initial Search. Choose actions using their schemas below.',
        'parameters': {'type': 'object', 'properties': {
            'history': {'type': 'string',
                'description': 'Start from the previous history in working_note, merge the preceding '
                    'round\'s actual action results, and return the full updated summary by retained '
                    'dataset ID. Preserve relevant earlier findings, update changed findings/progress/'
                    'failures, and remove discarded entries. Current actions have not run; distinguish '
                    'any intention from completed results. No full reports or growing transcript. '
                    f'Aim for {history_limit} characters; length is a soft target. Initially may be empty.'},
            'actions': {'type': 'array', 'minItems': 1, 'items': {'anyOf': variants},
                        'description': 'Business actions selected by the agent; independent calls of one '
                            'stage may be batched, with local decisions alongside. No note-saving action.'}},
            'required': ['history', 'actions'], 'additionalProperties': False}}}


def unpack_decision(call):
    raw = call['function'].get('arguments')
    value = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(value, dict) or set(value) != {'history', 'actions'}:
        raise ValueError('Decision requires exactly history and actions.')
    history, actions = value['history'], value['actions']
    if not isinstance(history, str):
        raise ValueError('history must be a string.')
    if not isinstance(actions, list) or not actions:
        raise ValueError('actions must be a nonempty array of business actions.')
    allowed = {item['function']['name'] for item in business_schemas()}
    calls = []
    for index, action in enumerate(actions):
        if (not isinstance(action, dict) or set(action) != {'name', 'arguments'}
                or action.get('name') not in allowed or not isinstance(action.get('arguments'), dict)):
            raise ValueError('Each action requires an available business name and arguments object; '
                             'update_working_note is not an action.')
        calls.append({'id': f'{call["id"]}_action_{index}', 'type': 'function',
            'function': {'name': action['name'], 'arguments': json.dumps(action['arguments'], ensure_ascii=False)}})
    return {'history': history.strip(), 'actions': copy.deepcopy(actions)}, calls
