"""Native proposal evidence binding and the custom-retriever compatibility adapter.

The default flow lets the provider manage search context inside one Responses
request. ResearchAgent remains for clients explicitly injecting a retriever.
"""
from __future__ import annotations

import copy
import re

from tools.executor_tools.implementations.web_evidence import evidence_failure_details


DEFAULT_SEARCH_BUDGET = 3
RESEARCH_POLICY_VERSION = 'native-proposals-v3'
NATIVE_EVIDENCE_POLICY_VERSION = 'native-cited-claims-v2'


def configured_search(model_config_path):
    """Expose the separately configured search model as a tool for the base model."""
    from utils.model_config import load_model_config
    from tools.executor_tools.implementations.web_evidence import web_search

    settings = load_model_config(model_config_path).get('executor_search')
    if not settings:
        return None
    from pathlib import Path
    config = Path(settings['model_config_path'])
    if not config.is_absolute():
        config = Path(model_config_path).resolve().parent / config

    def retrieve(query, image_paths=None, **kwargs):
        return web_search(query=query, image_paths=image_paths,
                          model=settings['model'], model_config_path=str(config),
                          api_mode=settings.get('api_mode', 'responses'))

    return retrieve


def native_evidence(value, response, search_budget):
    """Bind proposed external premises to actual provider retrieval, preserving indexes.

    This is provider citation support, not an independent page/quote verification.
    A generated URL or the model's knowledge alone cannot populate the fact store.
    """
    from tools.executor_tools.implementations.web_evidence import canonical_url
    from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

    def source_key(url):
        parts = urlsplit(canonical_url(url))
        query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                 if not k.lower().startswith('utm_') and k.lower() not in ('gclid', 'fbclid', 'msclkid')]
        return urlunsplit((parts.scheme, parts.netloc, parts.path or '/', urlencode(query), ''))
    calls = response.get('web_search_calls', [])
    if len(calls) > search_budget:
        raise ValueError('provider exceeded the native tool-call budget')
    completed = any(c.get('type') == 'web_search_call' and c.get('status') == 'completed' for c in calls)
    sources = {}
    for citation in response.get('citations', []) if completed else []:
        url = citation['url']
        key = source_key(url)
        sources[key] = {'id': sources[key]['id'] if key in sources else 's' + str(len(sources) + 1), 'url': url,
                        'title': citation.get('title', ''), 'origin': citation.get('origin', 'provider_citation'),
                        'verification': 'native_retrieval', 'page_checked': False}
    claims, failures = [], []
    for index, claim in enumerate(value.get('evidence', [])):
        supported = [sources[source_key(url)]['id'] for url in claim['source_urls']
                     if source_key(url) in sources]
        if supported:
            claims.append({'id': 'c' + str(index + 1), 'statement': claim['statement'],
                           'scope': claim.get('scope', ''), 'source_ids': list(dict.fromkeys(supported)),
                           'verification': 'native_citation'})
        else:
            # Empty positions cannot be referenced as supported statements; never shift indexes.
            claims.append({})
            failures.append({'reference': f'evidence.e1.claims.{index}.statement',
                             'reason': 'No completed native search with this URL in provider provenance'})
    used_ids = {sid for claim in claims for sid in claim.get('source_ids', [])}
    evidence = {'e1': {'claims': claims, 'sources': [s for s in sources.values() if s['id'] in used_ids],
                       'missing_requirements': [], 'conflicts': [],
                       'verification': 'native_citation', 'policy_version': NATIVE_EVIDENCE_POLICY_VERSION}} if used_ids else {}
    status = (('partial' if failures else 'supported') if evidence else
              'unavailable' if failures or any(c.get('status') != 'completed' for c in calls) else 'original_only')
    return evidence, failures, status


def query_key(query, images=()):
    return (re.sub(r'\s+', ' ', query).strip().casefold(), tuple(sorted(images)))


class ResearchAgent:
    def __init__(self, original, contract, recipe, tools, search_budget,
                 evidence, failures):
        self.original = original
        self.contract = contract
        self.recipe = recipe
        self.tools = list(tools)
        self.budget = search_budget
        self.evidence = evidence
        self.failures = failures
        self.history = []
        self.memory = ''

    @property
    def grounding_status(self):
        if not self.history:
            return 'original_only'
        return 'supported' if not self.failures else 'partial' if self.evidence else 'unavailable'

    def state(self, turn):
        return {'turn': turn, 'search_budget': self.budget,
                'remaining_searches': self.budget - len(self.history),
                'memory': self.memory, 'search_history': copy.deepcopy(self.history)}

    def run(self, decide, retrieve, save_state):
        # Each search consumes one budget unit. A final decision is always
        # allowed after the last result, including when no searches are allowed.
        for turn in range(1, self.budget + 2):
            payload = {'task_contract': self.contract, 'original': self.original,
                       'recipe_hint': self.recipe, 'available_tools': self.tools,
                       'research_state': self.state(turn),
                       'evidence': copy.deepcopy(self.evidence)}
            decision = decide(payload, turn)
            if decision['action'] != 'search':
                save_state(turn, {**self.state(turn), 'decision': decision,
                                  'grounding_status': self.grounding_status})
                return decision
            if len(self.history) >= self.budget:
                raise RuntimeError('research agent exceeded the search budget')
            query, images = decision['query'], decision.get('image_paths', [])
            if any(query_key(query, images) == query_key(h['query'], h['image_paths'])
                   for h in self.history):
                raise RuntimeError('research agent repeated a completed search')
            self.memory = decision['memory']
            request = {'id': 'e' + str(len(self.history) + 1),
                       'query': query, 'image_paths': images}
            packet = retrieve(request)
            if not isinstance(packet, dict):
                raise RuntimeError('invalid research evidence packet')
            diagnostics = evidence_failure_details(packet)
            usable = packet.get('status') in ('supported', 'partial') and bool(packet.get('claims'))
            observation = {**request, 'status': packet.get('status', 'insufficient')}
            if usable:
                self.evidence[request['id']] = {k: copy.deepcopy(packet.get(k, [])) for k in (
                    'claims', 'sources', 'missing_requirements', 'conflicts')}
                self.evidence[request['id']]['page_verification'] = copy.deepcopy(packet.get('page_verification', {}))
            if not usable or packet.get('status') == 'partial' or packet.get('missing_requirements'):
                # Never place unverified statements in the agent's fact store.
                failure = {'request': request, 'status': observation['status'], 'diagnostics': diagnostics}
                self.failures.append(copy.deepcopy(failure))
                observation['diagnostics'] = diagnostics
            self.history.append(observation)
            save_state(turn, {**self.state(turn), 'evidence': self.evidence,
                              'grounding_status': self.grounding_status})
        raise RuntimeError('research agent did not deliver proposals')
RESEARCH_DECISION_TOOL = {'type': 'function', 'function': {
    'name': 'submit_research_decision',
    'description': 'Return the next research decision for the application executor.',
    'parameters': {'type': 'object', 'properties': {
        'action': {'type': 'string', 'enum': ['search', 'finish', 'skip']},
        'memory': {'type': 'string'}, 'query': {'type': 'string'},
        'image_paths': {'type': 'array', 'items': {'type': 'string'}},
        'reason': {'type': 'string'},
        'proposals': {'type': 'array', 'items': {'type': 'object', 'properties': {
            'id': {'type': 'string'}, 'goal': {'type': 'string'},
            'source_refs': {'type': 'array', 'items': {'type': 'string'}},
            'strategy': {'type': 'string'}}, 'required': ['id', 'goal', 'source_refs', 'strategy']}}},
        'required': ['action']}}}

