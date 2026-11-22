"""Minimal generation response checks, before retry/cache success decisions.

These validate transport shape, not sample quality or source truth. Content
constraints and supported references remain the existing controller's responsibility.
"""
import copy

from .generation_inspection import validate_inspection_response


GENERATION_RESPONSE_POLICY_VERSION = 'generation-shape-v7'


def normalize_generation_response(value, stage, *, previous_media_jobs=None):
    """Apply protocol defaults; retain raw output separately for audit."""
    if (stage == 'replan' and isinstance(value, dict) and value.get('decision') == 'proceed'
            and value.get('evidence_requests') == [] and isinstance(value.get('design'), dict)
            and ('evidence_requests' not in value['design'] or value['design']['evidence_requests'] == [])):
        normalized = copy.deepcopy(value)
        normalized.pop('evidence_requests')
        normalized['design']['evidence_requests'] = []
        return normalized, ['moved_empty_evidence_requests_into_design']
    if (stage == 'repair' and isinstance(value, dict) and value.get('status') == 'ready'
            and 'media_jobs' not in value and isinstance(previous_media_jobs, list)):
        # Omission preserves the existing production plan. Explicit arrays,
        # including [], replace it; malformed explicit values remain errors.
        normalized = copy.deepcopy(value)
        normalized['media_jobs'] = copy.deepcopy(previous_media_jobs)
        return normalized, ['inherited_media_jobs']
    return value, []


def _validate_evidence_requests(requests, available_tools=None, search_budget=None, source_images=None):
    if not isinstance(requests, list):
        return 'evidence_requests must be an array'
    if search_budget is not None and len(requests) > search_budget:
        return 'evidence requests exceed search budget'
    if requests and available_tools is not None and 'web_search' not in available_tools:
        return 'web_search is not registered/enabled'
    identifiers = set()
    for request in requests:
        if not isinstance(request, dict):
            return 'each evidence request must be an object with id/query and optional image_paths'
        key, query = request.get('id'), request.get('query')
        if not isinstance(key, str) or not key.isidentifier() or key in identifiers:
            return 'evidence request ids must be unique identifier strings such as e1'
        identifiers.add(key)
        if not isinstance(query, str) or not query.strip():
            return 'evidence request query must be a nonempty string'
        images = request.get('image_paths', [])
        if not isinstance(images, list) or any(not isinstance(path, str) for path in images):
            return 'evidence request image_paths must be an array of strings'
        if source_images is not None and any(path not in source_images for path in images):
            return 'search requested an image outside original source'
    return None


def validate_generation_response(value, stage, review_mode=None, answer_type=None,
                                 available_tools=None, search_budget=None, source_images=None,
                                 search_history=None):
    value, _ = normalize_generation_response(value, stage)
    if stage == 'inspect':
        return validate_inspection_response(value, review_mode, answer_type)
    if not isinstance(value, dict):
        return stage + ' requires a complete top-level JSON object, not an internal array or fragment'
    if stage == 'proposal':
        if value.get('action') not in ('finish', 'skip'):
            return 'native proposal agent returns finish or skip; use the hosted search tool, not action search'
        error = validate_generation_response(value, 'research')
        if error:
            return error
        claims = value.get('evidence', [])
        if not isinstance(claims, list):
            return 'proposal evidence must be an array, empty when no external premises are used'
        from urllib.parse import urlsplit
        for claim in claims:
            if not isinstance(claim, dict) or not isinstance(claim.get('statement'), str) or not claim['statement'].strip():
                return 'each external evidence claim requires a nonempty statement'
            urls = claim.get('source_urls')
            if not isinstance(urls, list) or not urls:
                return 'each external evidence claim requires actual retrieved source_urls'
            for url in urls:
                try:
                    parts = urlsplit(url) if isinstance(url, str) else None
                    if parts is None or parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or parts.password:
                        return 'external source_urls must be public HTTP(S) URLs'
                except ValueError:
                    return 'invalid external source URL'
            if 'scope' in claim and not isinstance(claim['scope'], str):
                return 'external evidence scope must be a string'
        return None
    if stage == 'research':
        action = value.get('action')
        if action == 'skip':
            return None if isinstance(value.get('reason'), str) and value['reason'].strip() else 'research skip requires a concrete reason'
        if action == 'search':
            if any(k in value for k in ('design', 'candidate', 'decision', 'evidence_requests')):
                return 'research search uses action/memory/query/image_paths, not a plan or request list'
            if not isinstance(value.get('memory'), str) or not value['memory'].strip():
                return 'research search requires brief working memory explaining the evidence need'
            error = _validate_evidence_requests(
                [{'id': 'next', 'query': value.get('query'), 'image_paths': value.get('image_paths', [])}],
                available_tools, search_budget, source_images)
            if error:
                return error
            from .generation_research import query_key
            key = query_key(value['query'], value.get('image_paths', []))
            if any(key == query_key(h['query'], h.get('image_paths', [])) for h in search_history or []):
                return 'query already completed; use its result, choose a different query or finish'
            return None
        if action == 'finish':
            if any(k in value for k in ('design', 'query', 'evidence_requests', 'candidate', 'decision')):
                return 'research finish delivers action/proposals, not a final design, search or sample'
            proposals = value.get('proposals')
            if not isinstance(proposals, list) or not 2 <= len(proposals) <= 3:
                return 'research finish requires 2-3 distinct proposals'
            identifiers, routes = set(), set()
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    return 'each proposal requires id, goal, source_refs and strategy'
                identifier = proposal.get('id')
                if not isinstance(identifier, str) or not identifier.isidentifier() or identifier in identifiers:
                    return 'proposal ids must be unique identifiers such as p1'
                identifiers.add(identifier)
                error = validate_generation_response({'decision': 'proceed', 'design': proposal}, 'design')
                if error:
                    return 'proposal ' + identifier + ': ' + error
                route = tuple(' '.join(proposal[k].casefold().split()) for k in ('goal', 'strategy'))
                if route in routes:
                    return 'proposals must not duplicate the same goal and strategy'
                routes.add(route)
            return None
        return 'research requires action search, finish or skip'
    if stage == 'source_analysis':
        if any(key in value for key in ('decision', 'design', 'candidate', 'question', 'options', 'gold', 'answer')):
            return 'source_analysis prepares evidence only; do not return a final design/question/answer'
        if not isinstance(value.get('analysis'), str) or not value['analysis'].strip():
            return 'source_analysis requires a nonempty analysis'
        if 'evidence_requests' not in value:
            return 'source_analysis requires evidence_requests, an empty array when no search is needed'
        return _validate_evidence_requests(value['evidence_requests'], available_tools, search_budget, source_images)
    if stage in ('design', 'replan'):
        if 'evidence_requests' in value:
            if stage == 'design':
                return 'evidence_requests belongs to source_analysis; retrieval is complete before design'
            return 'evidence_requests belongs inside design.evidence_requests, not at the top level'
        if value.get('decision') == 'skip':
            return None if value.get('reason') else stage + ' skip requires a reason'
        if value.get('decision') != 'proceed':
            return stage + ' response missing/invalid decision; expected proceed or skip'
        design = value.get('design')
        if not isinstance(design, dict):
            return stage + ' requires design as a top-level object'
        if not isinstance(design.get('goal'), str) or not design['goal'].strip():
            return 'design requires a nonempty goal'
        if not isinstance(design.get('strategy'), str) or not design['strategy'].strip():
            return 'design requires a nonempty executable strategy'
        refs = design.get('source_refs')
        if not isinstance(refs, list) or not refs or any(not isinstance(r, str) or not r.strip() for r in refs):
            return 'design requires a nonempty source_refs array of strings'
        requests = design.get('evidence_requests', [])
        if not isinstance(requests, list):
            return 'design.evidence_requests must be an array'
        if requests:
            return 'design.evidence_requests must be empty during ' + stage + '; retrieval is complete, no new searches'
        return None
    if stage in ('construct', 'repair'):
        if value.get('status') == 'blocked':
            return None if value.get('blocking_issue') else stage + ' blocked requires blocking_issue'
        if value.get('status') != 'ready':
            return stage + ' response missing/invalid status; expected ready or blocked'
        candidate = value.get('candidate')
        if not isinstance(candidate, dict) or not {'input', 'output'}.issubset(candidate):
            return stage + ' requires candidate with input and output'
        support = value.get('support')
        if not isinstance(support, dict):
            return 'support must be a top-level sibling of candidate, not nested inside it'
        refs = support.get('answer_refs')
        if not isinstance(refs, list) or not refs or any(not isinstance(r, str) or not r.strip() for r in refs):
            return 'support requires a nonempty answer_refs array of strings'
        if 'answer_justification' in support and (not isinstance(support['answer_justification'], str) or not support['answer_justification'].strip()):
            return 'support requires a nonempty answer_justification'
        if 'media_jobs' in value and not isinstance(value['media_jobs'], list):
            return 'media_jobs must be an array when present'
    return None
