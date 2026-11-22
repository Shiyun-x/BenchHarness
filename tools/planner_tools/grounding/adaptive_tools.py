"""Native function tools exposed to each independent Grounding agent."""
import inspect
import re
from typing import List, Literal, Optional, TypedDict, Union, get_args, get_origin


class DatasetDiscard(TypedDict):
    dataset_id: str
    reason: str


async def search_datasets(guidance: str = "", context_variables=None):
    """Search the local dataset pool and return concise candidates.

    Start once, then develop the current pool. Search again only after all its
    sources are explicitly discarded and actual material findings justify a new
    direction. Assess may establish this without Plan. Use the working note and
    retrieval_feedback to explain the gap; API/format failures need recovery.
    Planned/ready or reserved sources prevent supplementary search. Do not search
    to fill the five-source ceiling. Run separately from Assess/Plan; this searches
    local cards, while downstream Generator handles native web research.

    Args:
        guidance: Initial direction; for supplementary search, observed material gap and changed retrieval cues.
    """
    return await context_variables['grounding_session'].search(guidance)


async def assess_dataset(dataset_id: str, question: str, context_variables=None):
    """Inspect one candidate's real material and transformation potential.

    Preferred before Plan for promising candidates, unless actual material is
    already sufficiently understood. Reuse existing findings where sufficient.
    Ask what reusable information or supervision could support the target, or
    resolve a concrete doubt affecting source choice. Judge development potential,
    not whether the original sample already passes the finished test.
    Look for useful facts, annotations or correct QA that could support the target,
    including substantial reconstruction and later factual extensions; the source
    need not already have the target structure, modality or difficulty.
    Consider how freely the material can change: reused media constrain the
    depth of inference unless actual capabilities can supply relevant changes.
    Related topics or candidate labels alone do not establish richer media content.
    Returns a short answer for you; the material summary and references remain
    available to Plan. Choose the batch that needs inspection and issue its
    independent assessments together; the whole pool is allowed, not required.
    Do not defer known independent questions to later turns. After the batch
    returns, use its findings to choose Plan/discard/reserve or a concrete
    follow-up, rather than treating more inspection as the default next step.
    A follow-up uses the prior report and the new question; it does not select
    final sources or generate a blueprint. Only the latest short question/answer
    stays in your state; full history is cached. Planned routes may be investigated
    before revising with guidance. Ready routes cannot be assessed again.

    Args:
        dataset_id: An existing candidate ID returned by search_datasets.
        question: The question to answer; required for both first inspection and follow-up. Ask about a concrete decision gap; existing findings need rereading only when the short answer omits decision-relevant information.
    """
    return await context_variables['grounding_session'].assess(dataset_id, question)


async def plan_dataset_transformation(dataset_id: str, guidance: str = "", context_variables=None):
    """Suggest a brief dataset-level strategy and useful asset-tool purposes.

    Reads real material, any assessment and available capabilities. Returns a
    ready blueprint, or a concrete blocker. Prefer prior Assess to investigate
    source potential; direct Plan is appropriate when actual material is already
    sufficiently understood, without making Assess mandatory.
    Repeating this call develops or revises the current route for this dataset;
    program code manages blueprint versions and references.
    This is advice for a sample-specializing generator, not an operation script.
    Develop usable information even when substantial material/QA reconstruction
    is needed. Use the supplied downstream capabilities when developing the route.
    After Assess returns, use its findings to select promising sources and issue
    their independent planning calls together; enough material potential and a
    plausible target route justify Plan, without proving final sample quality.
    Respect information limits of reused media found by Assess; rewriting QA or
    retrieving related facts does not itself add missing perceptual clues.
    Success sets dataset status to planned, awaiting YOUR judgment. If sound,
    explicitly mark ready with manage_candidates. Otherwise supply focused guidance
    and revise, investigating a concrete doubt first when needed. Technical failure
    retains prior work and is retryable; it does not accept the old route.
    Ready routes cannot be planned again.
    Keep at most five current dataset routes, including reserved ready routes;
    revisions of the same dataset do not add a route. Five is not a quota to fill.

    Args:
        dataset_id: An existing candidate ID to develop.
        guidance: Optional dataset-level opportunity or concrete defect; describe reusable material/target/tool relationships, not instructions for a particular sampled event or QA.
    """
    return await context_variables['grounding_session'].plan(dataset_id, guidance)


async def manage_candidates(dataset_ids: List[str], action: Literal['reserve', 'activate', 'ready'], context_variables=None):
    """Reserve for batch Plan, restore visibility, or accept a planned route locally.

    reserve means information is sufficient to wait for the next planning batch;
    direct reserve after Search is allowed. It is not indefinite uncertainty.
    Original progress survives in the compact reserve index. Assess/Plan activates
    a reserve automatically; explicit activate only restores its state.
    ready is YOUR acceptance of an existing current blueprint: check target fit,
    material basis and feasibility, not final sample quality. Use Assess findings
    about retained media's information limits when judging the route. It retains
    only a short route view; no more Assess/Plan for that source. Accept each usable
    blueprint independently; do not wait for other datasets or the final round.
    You may accept A while investigating/planning OTHER datasets. Only when the
    whole pool is resolved should you batch final acceptance/cleanup and Finish.
    A no-op activation is not progress and should not be used to pad a response.

    Args:
        dataset_ids: Nonempty unique known candidate IDs.
        action: reserve to wait for Plan; activate to restore visibility; ready to accept current blueprints.
    """
    return await context_variables['grounding_session'].manage(dataset_ids, action)


async def read_assessment(dataset_id: str, report_type: Literal['assessment', 'blueprint'] = 'assessment', context_variables=None):
    """Read a cached assessment or blueprint for a specific decision gap.

    First identify the missing fact or condition and how it could change source
    selection, reserve/discard, or dataset-level guidance. Read only affected
    datasets; each read needs its own concrete gap. Do not batch-read all reports
    as a routine step after Assess, for general confidence, or to prepare Plan:
    Plan already receives the full analysis and refs internally. Proceed directly
    when the short answer is sufficient. New investigation needs a focused
    assess_dataset question, not rereading.
    No model call or assessment budget is used, but the detailed findings enlarge
    your next request. The latest detailed report appears there once.
    Repeated reading of the same report, or reading a blueprint already displayed
    in current state, does not count as new information in round feedback.
    Does not create an assessment or automatically activate a reserve.

    Args:
        dataset_id: A known dataset with a saved assessment.
        report_type: assessment for material analysis, blueprint for full planning advice; no model call.
    """
    return await context_variables['grounding_session'].read_assessment(dataset_id, report_type)


async def update_working_note(note: str, context_variables=None):
    """Legacy/internal history saver, not exposed in the new decision actions.

    Keep a compact history of actual prior work, organized by dataset ID.
    For each retained dataset use a short entry:
    <dataset ID>: <key completed investigation/planning and findings>; <confirmed progress or unresolved issue, if any>.

    Preserve valid earlier findings without rephrasing them. Replace answered
    questions with conclusions; keep only genuine unresolved issues. Update changed
    entries and remove discarded sources/obsolete plans, retaining useful general
    retrieval lessons. Discard reasons stay in the audit cache. Keep source-specific
    history that explains current judgments, not full reports, refs or a growing log.

    Reconcile against tool returns, last_round_feedback and actual state. Preserve
    prior work when a call fails and record the real failure; do not invent successful
    revisions or ready acceptance. Current calls remain intentions until confirmed.
    A brief final next-step intention is optional; confirmed history is the main content.

    After the initial Search, place this once before actual decision calls in the
    same response, as the main decision protocol describes. It only saves text;
    it cannot execute reserve, ready, discard or Finish through note instructions.
    Return the complete replacement note; aim for 1200 characters as a soft target.

    Args:
        note: Complete replacement working note, not an incremental addition.
    """
    return await context_variables['grounding_session'].update_working_note(note)


async def discard_datasets(datasets: List[DatasetDiscard], context_variables=None):
    """Remove datasets you no longer intend to consider in this subtask.

    Batch decisions in one call, with one concise reason per dataset explaining
    the observed material obstacle or why this source is not being selected now.
    For useful sources, explain the actual priority or overlap tradeoff; do not
    invent a defect to justify selecting fewer. A native task name or missing
    detail in a short card alone does not establish unusable material.
    Removes their active assessments, blueprints and loaded
    material; historical disk files remain available. Later searches exclude them,
    including after resume, until a fresh run. Reasons are saved to the audit log;
    your subsequent context keeps only excluded IDs.
    Update the decision's history field alongside discard to remove these sources'
    entries and obsolete plans; do not retain per-source discard reasons in history.
    This is a scope decision, not a claim that the entire
    dataset has no value. You can drop redundant or lower-priority sources;
    reserve sources waiting for Plan. Large edits or a different native
    format alone do not establish a lack of material potential.
    Can run beside assess/plan calls for OTHER datasets in the same round.

    Args:
        datasets: Nonempty array of {dataset_id, reason}; unique candidate IDs and a nonempty, concise decision reason for each.
    """
    return await context_variables['grounding_session'].discard(datasets)


async def finish_grounding(status: Literal['ready', 'incomplete'], reason: str,
                           dataset_ids: Optional[List[str]] = None, context_variables=None):
    """Submit this subtask's selected sources and stop its Grounding agent.

    Use ready with one to five datasets YOU have accepted via manage_candidates
    action=ready. Once the choice is justified, submit acceptance,
    any unselected-source discards and Finish together; no new model turn is
    needed between these local actions. Plan alone does not accept.
    Use incomplete when
    useful grounding could not be established; successful partial work is kept.
    Call after assessments/plans return. Discard every remaining unselected source
    before finishing ready, optionally in this same response. Program code resolves
    blueprint references; you only choose dataset IDs.
    Ready is a per-dataset decision and may happen earlier. Finish only after all
    retained sources are ready and others explicitly discarded, with no remaining
    investigation/planning gap; final ready/discard calls may be in this response.
    This explicit call stops the subtask. A note saying "finish" does not stop it;
    there is no separate finished status. Record prior outcomes in the decision's
    history field; put cleanup/ready actions as needed before this final action.

    Args:
        status: ready when usable sources have blueprints, otherwise incomplete.
        reason: Why selection is complete or what unresolved gap remains.
        dataset_ids: Selected unique dataset IDs with current ready blueprints; required for ready.
    """
    return await context_variables['grounding_session'].finish(status, reason, dataset_ids)


GROUNDING_TOOLS = (search_datasets, assess_dataset, plan_dataset_transformation,
                   manage_candidates, read_assessment, discard_datasets, update_working_note, finish_grounding)


def _type_schema(annotation):
    base = {str: 'string', int: 'integer', float: 'number', bool: 'boolean'}
    if annotation in base:
        return {'type': base[annotation]}
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Union:
        actual = [arg for arg in args if arg is not type(None)]
        return _type_schema(actual[0]) if len(actual) == 1 else {'anyOf': [_type_schema(a) for a in actual]}
    if origin is Literal:
        return {'type': 'string', 'enum': list(args)}
    if origin is list:
        return {'type': 'array', 'items': _type_schema(args[0])}
    if hasattr(annotation, '__required_keys__'):
        return {'type': 'object', 'properties': {k: _type_schema(v) for k, v in annotation.__annotations__.items()},
                'required': list(annotation.__annotations__), 'additionalProperties': False}
    raise TypeError(f'Unsupported Grounding tool annotation: {annotation}')


def tool_schema(function):
    """Build request metadata from signatures and Google-style Args docstrings."""
    doc = inspect.getdoc(function) or ''
    description, _, argument_text = doc.partition('\nArgs:\n')
    explanations = dict(re.findall(r'^\s{4}(\w+): (.+)$', argument_text, re.M))
    properties, required = {}, []
    for name, parameter in inspect.signature(function).parameters.items():
        if name == 'context_variables':
            continue
        item = _type_schema(parameter.annotation)
        if name in explanations:
            item['description'] = explanations[name]
        properties[name] = item
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
    return {'type': 'function', 'function': {'name': function.__name__, 'description': description,
            'parameters': {'type': 'object', 'properties': properties, 'required': required,
                           'additionalProperties': False}}}


def grounding_tool_schemas():
    return [tool_schema(function) for function in GROUNDING_TOOLS]
