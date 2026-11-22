"""Responses protocol helpers; no SDK, credentials or network side effects."""
from __future__ import annotations

def responses_input(messages):
    """Reuse media loading while emitting native text/image/Qwen audio parts."""
    result = []
    for message in messages:
        content = message['content']
        if isinstance(content, list):
            parts = []
            for part in content:
                if part['type'] == 'text':
                    parts.append({'type': 'input_text', 'text': part['text']})
                elif part['type'] == 'image_url':
                    image = part['image_url']
                    parts.append({'type': 'input_image', 'image_url': image['url'],
                                  **({'detail': image['detail']} if 'detail' in image else {})})
                elif part['type'] == 'input_audio':
                    audio = part['input_audio']
                    field = 'audio_url' if audio['data'].startswith(('https://', 'http://')) else 'data'
                    parts.append({'type': 'input_audio', field: audio['data'], 'format': audio['format']})
                else:
                    raise ValueError('unsupported Responses input part: ' + part['type'])
            content = parts
        result.append({'role': message['role'], 'content': content})
    return result


def inspect_native_response(raw):
    """Keep commentary out of JSON and retain actual hosted-tool provenance."""
    output = raw.get('output') or []
    messages = [v for v in output if isinstance(v, dict) and v.get('type') == 'message']
    finals = [v for v in messages if v.get('phase') == 'final_answer']
    candidates = finals or [v for v in messages if not v.get('phase')]
    final = candidates[-1] if candidates else {}
    text = '\n'.join(p['text'] for p in final.get('content') or []
                     if p.get('type') == 'output_text' and isinstance(p.get('text'), str))
    calls = [v for v in output if isinstance(v, dict) and v.get('type') == 'web_search_call']
    citations = {}

    def add(source, origin):
        if not isinstance(source, dict):
            return
        url = source.get('url')
        if isinstance(url, str) and url.startswith(('https://', 'http://')):
            citations[url] = {'url': url,
                              'title': source.get('title') or citations.get(url, {}).get('title', ''),
                              'origin': origin}

    for call in calls:
        action = call.get('action') or {}
        if (call.get('status') != 'completed' or call.get('error') or
                not isinstance(action, dict) or action.get('error')):
            continue
        # A completed page open is retrieved provenance even when its URL is
        # absent from search snippets and the final JSON has no annotations.
        # This records a visited source, not independent factual verification.
        if action.get('type') == 'open_page':
            add({'url': action.get('url')}, 'native_opened_page')
        for source in action.get('sources') or []:
            if isinstance(source, dict):
                add(source, 'native_search_source')
    for message in messages:
        for part in message.get('content') or []:
            for annotation in part.get('annotations') or []:
                if annotation.get('type') == 'url_citation':
                    add(annotation.get('url_citation', annotation), 'provider_citation')
    status = raw.get('status')
    incomplete = status != 'completed' or final.get('status', 'completed') != 'completed'
    details = raw.get('incomplete_details') or {}
    finish = 'length' if details.get('reason') == 'max_output_tokens' else 'incomplete' if incomplete else 'stop'
    return {'raw_text': text, 'finish_reason': finish, 'response_status': status,
            'usage': raw.get('usage') or {}, 'citations': list(citations.values()),
            'web_search_calls': calls, 'provider_response': raw,
            'incomplete_details': raw.get('incomplete_details')}


def sum_usage(values):
    """Keep cached/reasoning token details as well as top-level token totals."""
    result = {}
    for value in values:
        for key, number in value.items():
            if isinstance(number, dict):
                result[key] = sum_usage([result.get(key, {}), number])
            elif isinstance(number, (int, float)) and not isinstance(number, bool):
                result[key] = result.get(key, 0) + number
    return result
