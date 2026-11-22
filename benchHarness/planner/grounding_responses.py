"""Responses transport adaptation; Grounding decisions and state stay unchanged."""
import copy

from utils.native_responses import responses_input


def responses_request(request):
    params = copy.deepcopy(request)
    model = params['model']
    if model.startswith('openai/responses/'):
        model = 'openai/' + model[len('openai/responses/'):]
    elif model.startswith('gpt-'):
        model = 'openai/' + model
    params['model'] = model
    params.pop('custom_llm_provider', None)
    inputs = []
    for message in params.pop('messages'):
        if message['role'] == 'tool':
            inputs.append({'type': 'function_call_output', 'call_id': message['tool_call_id'],
                           'output': message['content']})
        else:
            if message.get('content'):
                inputs.extend(responses_input([message]))
            for call in message.get('tool_calls') or []:
                inputs.append({'type': 'function_call', 'call_id': call['id'],
                               **call['function']})
    params['input'] = inputs
    params['max_output_tokens'] = params.pop('max_tokens')
    if 'base_url' in params:
        params['api_base'] = params.pop('base_url')
    # Keep existing optional action fields; Responses otherwise defaults to strict.
    params['tools'] = [{'type': 'function', **tool['function'], 'strict': False}
                       for tool in params['tools']]
    params['drop_params'] = False
    return params


def responses_reply(raw):
    """Normalize function calls for existing validation, preserving the native reply."""
    if raw.get('status') != 'completed':
        raise ValueError('Decision output was incomplete; return a complete history/actions object.')
    calls, texts = [], []
    for item in raw.get('output', []):
        if item.get('status') not in (None, 'completed'):
            raise ValueError('Decision output was incomplete; return a complete history/actions object.')
        if item.get('type') == 'function_call':
            calls.append({'id': item.get('call_id') or item.get('id'), 'type': 'function',
                'function': {'name': item.get('name'), 'arguments': item.get('arguments')}})
        elif item.get('type') == 'message':
            texts.extend(part['text'] for part in item.get('content', [])
                         if part.get('type') == 'output_text')
    usage = copy.deepcopy(raw.get('usage') or {})
    usage.update(prompt_tokens=usage.get('input_tokens', 0),
                 completion_tokens=usage.get('output_tokens', 0))
    return {'id': raw.get('id'), 'model': raw.get('model'), 'usage': usage,
        'provider_response': raw,
        'choices': [{'finish_reason': 'tool_calls' if calls else 'stop',
            'message': {'role': 'assistant', 'content': '\n'.join(texts) or None,
                        'tool_calls': calls}}]}
