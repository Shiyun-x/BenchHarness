"""Focused web evidence with actual provider citations and page quote checks."""
from __future__ import annotations

import ipaddress
import os
import socket
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError

EVIDENCE_POLICY_VERSION = 'claim-coverage-v3'

SEARCH_PROMPT = """Research requested facts using actual web search.
Keep research focused on the requested facts. Prefer one or two relevant primary-source pages; stop when the facts are supported rather than compiling a broad survey. Return only claims needed by the query.
Prefer authoritative primary sources responsible for each fact: official
institutions, original publications, collection records or data publishers.
Find facts supporting reconstruction, not a generic essay. Use supplied image
anchors, but distinguish entity identification from its attributes.
Return JSON {answer:short summary, claims:[{id,statement,scope,source_urls:[],
excerpt:short exact supporting passage from one cited page}],
missing_requirements:[], conflicts:[]}.
Each claim needs retrieved source URLs and an exact supporting passage.
Use accessible primary pages. If one institution blocks access, consider a relevant
accessible primary source; supply an alternative URL only if it supports the same
exact excerpt. Do not expand research beyond the requested fact.
Preserve dates, units and qualifications. Never invent citations, quotations or
missing facts. Report insufficient/conflicting evidence. Not finding a claim
does not make it false. Include source citations in provider annotations."""


def canonical_url(url):
    parts = urlsplit(url)
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ''))


def _public_url(url):
    parts = urlsplit(url)
    if parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or parts.password:
        raise ValueError('source must be a public HTTP(S) page')
    for addr in socket.getaddrinfo(parts.hostname, parts.port or (443 if parts.scheme == 'https' else 80)):
        if not ipaddress.ip_address(addr[4][0]).is_global:
            raise ValueError('non-public source address')


class _PublicRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _PageText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


class SourcePageError(ValueError):
    """An accessible response that cannot serve as evidence."""
    def __init__(self, message, failure_kind, http_status=None):
        super().__init__(message)
        self.failure_kind = failure_kind
        self.http_status = http_status


def _challenge_page(text):
    head = text[:6000].casefold()
    return any(marker in head for marker in (
        "verify that you're not a robot", "verify that you are not a robot",
        "verify you are human", "enable javascript and cookies to continue",
        "performing security verification", "checking your browser before accessing"))


def _source_error(exc):
    """Keep actionable page diagnostics without treating access failure as falsity."""
    status = getattr(exc, 'code', None) if isinstance(exc, HTTPError) else getattr(exc, 'http_status', None)
    reason = getattr(exc, 'reason', exc)
    if isinstance(exc, SourcePageError):
        kind = exc.failure_kind
    elif status in (401, 403, 429):
        kind = 'access_blocked'
    elif status in (404, 410):
        kind = 'not_found'
    elif status in (408, 504) or isinstance(reason, (TimeoutError, socket.timeout)):
        kind = 'timeout'
    elif status and status >= 500:
        kind = 'http_server_error'
    elif isinstance(exc, HTTPError):
        kind = 'http_error'
    elif isinstance(exc, ValueError):
        kind = 'source_validation'
    else:
        kind = 'network_error'
    result = {'error': type(exc).__name__, 'failure_kind': kind,
              'error_detail': ' '.join(str(reason).split())[:240]}
    if isinstance(status, int):
        result['http_status'] = status
    return result


def evidence_failure_details(packet):
    """Compact recovery input; failed statements remain diagnostics, never evidence."""
    sources = packet.get('sources', [])
    unverified = packet.get('unverified_claims', [])
    return {
        'source_failures': [
            {k: source[k] for k in ('url', 'failure_kind', 'http_status', 'error_detail') if k in source}
            for source in sources if isinstance(source, dict) and not source.get('page_checked')],
        'unverified_claims': [
            {'statement': claim.get('statement', '')[:240],
             'failure_reasons': claim.get('failure_reasons', [])}
            for claim in unverified if isinstance(claim, dict)],
        'missing_requirements': packet.get('missing_requirements', []),
        'conflicts': packet.get('conflicts', []),
        'page_verification': packet.get('page_verification', {}),
    }


def fetch_source_text(url):
    _public_url(url)
    request = Request(url, headers={'User-Agent': 'BenchmarkAgentEvidence/1.0',
                                   'Accept': 'text/html,application/xhtml+xml,text/plain'})
    with build_opener(_PublicRedirect()).open(request, timeout=15) as response:
        content_type = response.headers.get('Content-Type', '')
        status = getattr(response, 'status', None)
        if not any(t in content_type for t in ('text/html', 'text/plain', 'application/xhtml')):
            raise SourcePageError('unsupported source format', 'unsupported_format', status)
        data = response.read(524289)
        if len(data) > 524288:
            raise SourcePageError('source exceeds page budget', 'page_budget', status)
        text = data.decode(response.headers.get_content_charset() or 'utf-8', errors='replace')
    if 'html' in content_type:
        parser = _PageText()
        parser.feed(text)
        text = ' '.join(parser.parts)
    text = ' '.join(text.split())
    if _challenge_page(text):
        raise SourcePageError('page requires browser/robot verification', 'access_blocked', status)
    return text


def normalize_evidence(payload, citations, fetch_page=fetch_source_text, max_sources=4):
    """Verify exact passages; retain failed claims and distinguish their failure reasons."""
    budget = min(4, max(0, int(max_sources)))
    raw_claims = payload.get('claims', [])
    raw_claims = raw_claims if isinstance(raw_claims, list) else []
    requested, candidates = [], {}
    for claim in raw_claims:
        if not isinstance(claim, dict) or not isinstance(claim.get('statement'), str):
            continue
        quote = claim.get('excerpt')
        quote = ' '.join(quote.split()) if isinstance(quote, str) else ''
        urls = set()
        for raw_url in claim.get('source_urls', []) if isinstance(claim.get('source_urls'), list) else []:
            try:
                url = canonical_url(raw_url)
                parts = urlsplit(url)
                if parts.scheme in ('http', 'https') and parts.hostname and not parts.username and not parts.password:
                    urls.add(url)
            except (ValueError, TypeError):
                continue
        valid_quote = 20 <= len(quote) <= 1500
        requested.append({'claim': claim, 'quote': quote, 'all_urls': urls,
                          'urls': urls if valid_quote else set(), 'source_ids': [],
                          'failures': [] if valid_quote else ['invalid_excerpt']})
    needed_urls = set().union(*(c['urls'] for c in requested)) if requested else set()
    for citation in citations:
        try:
            url = canonical_url(citation['url'])
        except (KeyError, ValueError, TypeError):
            continue
        if url in needed_urls and url not in candidates:
            candidates[url] = citation

    sources = []
    remaining = dict(candidates)
    while remaining and len(sources) < budget:
        uncovered = [c for c in requested if not c['source_ids']]
        if not uncovered:
            break
        url = max(remaining, key=lambda u: (
            sum(u in c['urls'] for c in uncovered),
            remaining[u].get('origin', 'provider_citation') == 'provider_citation'))
        if not any(url in c['urls'] for c in uncovered):
            break
        citation = remaining.pop(url)
        source = {'id': f's{len(sources)+1}', 'url': url, 'title': citation.get('title', ''),
                  'retrieved_at': datetime.now(timezone.utc).isoformat(),
                  'origin': citation.get('origin', 'provider_citation')}
        try:
            page = ' '.join(fetch_page(url).split())
            if _challenge_page(page):
                raise SourcePageError('page requires browser/robot verification', 'access_blocked')
            source['page_checked'] = True
        except Exception as exc:
            page = ''
            source.update(page_checked=False, **_source_error(exc))
        sources.append(source)
        for item in uncovered:
            if url not in item['urls']:
                continue
            if source['page_checked'] and item['quote'].casefold() in page.casefold():
                item['source_ids'].append(source['id'])
            else:
                item['failures'].append(source.get('failure_kind', 'quote_mismatch'))

    claims, unverified = [], []
    for item in requested:
        claim = item['claim']
        if item['source_ids']:
            claims.append({'id': f'c{len(claims)+1}', 'statement': claim['statement'],
                           'scope': claim.get('scope', ''), 'source_ids': item['source_ids'],
                           'support_excerpt': item['quote']})
        else:
            reasons = list(dict.fromkeys(item['failures']))
            if any(url in remaining for url in item['urls']) and len(sources) >= budget:
                reasons.append('source_budget')
            if not reasons:
                reasons.append('no_eligible_source')
            unverified.append({'statement': claim['statement'], 'scope': claim.get('scope', ''),
                               'source_urls': sorted(item['all_urls']), 'excerpt': item['quote'],
                               'failure_reasons': reasons})
    missing = payload.get('missing_requirements', [])
    conflicts = payload.get('conflicts', [])
    missing = list(missing) if isinstance(missing, list) else []
    missing.extend('No verified supporting page for claim: ' + c['statement'][:240] for c in unverified)
    conflicts = conflicts if isinstance(conflicts, list) else []
    status = 'conflicting' if conflicts else 'partial' if claims and missing else 'supported' if claims else 'insufficient'
    failure_counts = Counter(source['failure_kind'] for source in sources if not source['page_checked'])
    return {'status': status, 'claims': claims, 'sources': sources,
            'unverified_claims': unverified,
            'missing_requirements': missing, 'conflicts': conflicts,
            'insufficient_info': status in ('insufficient', 'conflicting'),
            'answer': '\n'.join('- '+c['statement'] for c in claims),
            'page_verification': {
                'policy_version': EVIDENCE_POLICY_VERSION, 'max_sources': budget,
                'attempted_sources': len(sources), 'eligible_sources': len(candidates),
                'claims_requested': len(requested), 'claims_supported': len(claims),
                'failure_counts': dict(failure_counts),
                'stop_reason': ('no_claims' if not requested else
                                'covered' if len(claims) == len(requested) else
                                'source_budget' if len(sources) >= budget else 'candidates_exhausted')}}


def source_candidates(payload, citations):
    """Gateways may strip annotations. Proposed URLs require actual page checks."""
    result = list(citations)
    known = {c.get('url') for c in result if isinstance(c, dict)}
    for claim in payload.get('claims', []) if isinstance(payload.get('claims'), list) else []:
        if not isinstance(claim, dict) or not isinstance(claim.get('source_urls'), list):
            continue
        for url in claim['source_urls']:
            if isinstance(url, str) and url.startswith(('https://', 'http://')) and url not in known:
                result.append({'url': url, 'origin': 'proposed_url_page_check'})
                known.add(url)
    return result


def web_search(query, model='openai/responses/gpt-6.1-sol', force_search=True,
               search_context_size='medium', image_paths=None, max_output_chars=3000,
               model_config_path=None, api_mode='completion', **kwargs):
    from utils.llm_caller import llm_call_json
    if not isinstance(query, str) or not query.strip():
        raise ValueError('query must be a nonempty string')
    if search_context_size not in ('low', 'medium', 'high'):
        search_context_size = 'medium'
    result = llm_call_json(
        model=model, system_prompt=SEARCH_PROMPT, user_prompt=query.strip(),
        images=image_paths or None, model_config_path=model_config_path,
        api_mode=api_mode,
        max_attempts=2, max_tokens=3000,
        extra_create_params={'timeout': int(os.getenv('WEB_SEARCH_REQUEST_TIMEOUT_S', '180')),
                             'request_timeout': int(os.getenv('WEB_SEARCH_REQUEST_TIMEOUT_S', '180')),
                             'tools': [{'type': 'web_search', 'search_context_size': search_context_size}],
                             'tool_choice': 'required' if force_search else 'auto'})
    if not isinstance(result, dict) or not result.get('ok') or not isinstance(result.get('json'), dict):
        raise RuntimeError(result.get('error') if isinstance(result, dict) else 'web search failed')
    packet = normalize_evidence(result['json'], source_candidates(result['json'], result.get('citations', [])))
    packet.update(usage=result.get('usage', {}), attempts=result.get('attempts', 1),
                  search_model=model, api_mode=api_mode)
    packet['answer'] = packet['answer'][:max(200, int(max_output_chars))]
    return packet
