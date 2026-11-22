# agents/grounding_agent.py
# -----------------------------------------------------------------------------
# Default Grounding uses independent adaptive subtask agents and round scheduling.
# Legacy state helpers and the former factory remain for --generation_mode legacy.
# -----------------------------------------------------------------------------
# 1) Dataset Preference and Search.
# 2) Transformability and Grounding Validation (Score-and-Filter).
# 3) Grounding Decision: accept when every subtask has at least one valid grounding.

from typing import Dict, Any, List, Optional
import json

from utils.registry import register_agent
from utils.agent_utils import Agent

from tools.planner_tools.grounding.grounding_tools import (
    preference_construction,
    dataset_search,
    select_candidates_for_subtask,
    transformability_assessment,
    score_and_filter,
    case_resolved,
)


def _candidate_short(item: Dict[str, Any]) -> Dict[str, Any]:
    """Material-aware, bounded view for choosing candidates to assess."""
    did = item.get("dataset_id") or item.get("id") or ""
    labels = item.get("labels") or {}
    desc = (labels.get("description_snippet") or "")
    out = {
        "dataset_id": did,
        "modalities": labels.get("modalities"),
        "tasks": labels.get("tasks"),
        "domain": labels.get("domain"),
        "description_snippet": desc[:500],
    }
    if item.get("dataset_name"):
        out["name"] = item["dataset_name"]
    for key in ("material_structure", "available_material"):
        if labels.get(key):
            out[key] = labels[key]
    if item.get("retrieval_routes"):
        out["retrieval_routes"] = item["retrieval_routes"]
    return {key: value for key, value in out.items() if value}


# Max candidates to include in state to avoid truncation (per subtask).
_MAX_CANDIDATES_IN_STATE = 20


def _subtask_grounding_state(st: Dict[str, Any]) -> Dict[str, Any]:
    """Pipeline state: id, name, next_step; when need_select_candidates include candidates (dataset_id + labels), capped."""
    sid = st.get("id") or ""
    has_preference = bool(st.get("dataset_preference"))
    rr = st.get("retrieval_result")
    rr_list = rr if isinstance(rr, list) else []
    retrieval_searched = bool(st.get("retrieval_searched")) or len(rr_list) > 0
    selected_ids = st.get("selected_candidate_ids") or []
    selected_count = len(selected_ids)
    candidate_selection_done = bool(st.get("candidate_selection_done")) or selected_count > 0
    has_transformability = bool(st.get("transformability"))
    scored_status = (st.get("scored_status") or "no").strip().lower()
    passed_count = len(st.get("scored_candidates") or {})

    if not has_preference:
        next_step = "need_preference"
    elif not retrieval_searched:
        next_step = "need_dataset_search"
    elif len(rr_list) == 0:
        # Search ran but found no matching datasets; nothing to select/assess.
        next_step = "ready_ungroundable"
    elif not candidate_selection_done:
        next_step = "need_select_candidates"
    elif selected_count == 0:
        # Candidates were reviewed, but none were suitable for assessment.
        next_step = "ready_ungroundable"
    elif not has_transformability:
        next_step = "need_transformability"
    elif scored_status != "yes":
        next_step = "need_score_and_filter"
    else:
        next_step = "ready_groundable" if passed_count > 0 else "ready_ungroundable"

    out = {"id": sid, "name": st.get("name") or sid, "description": st.get("description") or "", "next_step": next_step}
    if next_step in ("need_dataset_search", "need_select_candidates"):
        out["dataset_preference"] = st["dataset_preference"]
    if next_step == "need_select_candidates" and rr:
        cands = [_candidate_short(r) for r in rr if isinstance(r, dict)]
        out["candidates"] = cands[:_MAX_CANDIDATES_IN_STATE]
        if len(cands) > _MAX_CANDIDATES_IN_STATE:
            out["candidates_note"] = f"Showing first {_MAX_CANDIDATES_IN_STATE} of {len(cands)}; you may select any dataset_id from retrieval."
    if next_step.startswith("ready_"):
        out["passed"] = passed_count
    return out


def _summarize_for_grounding(subtasks: List[Dict[str, Any]], max_chars: int = 12000) -> str:
    """Full pipeline state per subtask so the agent can decide which tool to call for which subtask."""
    view = [_subtask_grounding_state(st) for st in (subtasks or [])]
    s = json.dumps(view, ensure_ascii=False, indent=2)
    # if len(s) > max_chars:
    #     return s[:max_chars] + "\n...(truncated)"
    return s


def _format_grounding_agent_history(entries: List[Dict[str, Any]]) -> str:
    if not entries:
        return "None yet."
    lines = []
    for entry in entries[-10:]:
        tool = str(entry.get("tool") or "").strip() or "unknown_tool"
        decision_rationale = str(entry.get("decision_rationale") or "").strip() or "Not recorded."
        args = str(entry.get("args") or "{}").strip()
        result = str(entry.get("result") or "").strip() or "No result recorded."
        lines.append(
            f"- tool: {tool}\n"
            f"  decision_rationale: {decision_rationale}\n"
            f"  args: {args}\n"
            f"  result: {result}"
        )
    return "\n".join(lines)


@register_agent("get_legacy_grounding_agent")
def get_legacy_grounding_agent(model: str, **kwargs) -> Agent:

    def instructions(ctx: Dict[str, Any]) -> str:
        subtasks = ctx.get("subtasks", []) or []
        grounding_agent_history = ctx.get("grounding_agent_history", []) or []
        grounding_agent_history_str = _format_grounding_agent_history(grounding_agent_history)
        subtasks_str = _summarize_for_grounding(subtasks)

        return f"""You are the benchmark Grounding Agent. Find datasets from the local pool that can support each subtask through a feasible transformation. Keep the tested capability intact; dataset-native tasks may differ. Accept when every subtask has a passed (dataset, plan) pair, otherwise report the observed material or feasibility gap to Design.

Candidate selection
- Judge the actual source material: content, relationships, observations and annotations useful for the target. Use the description together with material_structure and available_material; a similar task name or retrieval rank alone is insufficient.
- Candidates come from native-task and material-structure retrieval. retrieval_routes describes why they entered the shortlist, not proof of suitability. Ordinary QA, dialogue or other tasks can supply useful material even when their native objective differs.
- Select promising datasets for deeper assessment, not final benchmark acceptance. A concrete plausible connection to the target is enough here; the dataset need not already contain the final QA format or all features of the transformed sample. Leave detailed construction to transformability assessment and the generator.
- Usually select 2-3 strong candidates per subtask. Prefer complementary useful materials where available, without a diversity quota or padding with weak candidates. Select fewer when justified, or [] if none are plausible. Shared datasets may support different subtasks when each use is justified.

Tools and execution
Use the current next_step to act; include a brief decision_rationale. Tool schemas provide argument details.
1. preference_construction(): generate soft data preferences for all tasks lacking them.
2. dataset_search(subtask_id): retrieve candidates for one need_dataset_search task. Then select_candidates_for_subtask(subtask_id, dataset_ids, reason) for that task. For need_select_candidates, select directly. Record a short material-based reason; selection can be empty.
3. Finish search and selection for every task before transformability_assessment(). The tool enforces this gate and assesses selected pairs in batch.
4. Finish eligible transformability work before score_and_filter(), which processes all eligible tasks in batch and preserves completed ones.
5. Once every task is ready_groundable or ready_ungroundable, call case_resolved(). accepted=True requires at least one passed grounding per subtask. Otherwise use accepted=False with failed_subtask_ids, failure_reasons and feedback_to_design.

Do not repeat completed stages just to obtain a different answer. If a task fails, distinguish no retrieved or selected material from assessed infeasibility or scoring rejection. Explain the specific gap and justified next action. A failed search does not prove the target capability is invalid; avoid automatically weakening or discarding it. Do not claim dataset availability or transformation success beyond observed tool results.

Earlier run history is auxiliary memory for avoiding repeated dead ends; current state takes precedence. Actual calls and outputs in this conversation record progress in this run.

Current subtasks:
{subtasks_str}

Earlier Grounding runs:
{grounding_agent_history_str}
"""

    tools = [
        preference_construction,
        dataset_search,
        select_candidates_for_subtask,
        transformability_assessment,
        score_and_filter,
        case_resolved,
    ]

    return Agent(
        name="Grounding Agent",
        model=model,
        instructions=instructions,
        functions=tools,
        tool_choice="required",
        parallel_tool_calls=False,
    )


@register_agent("get_grounding_agent")
def get_grounding_agent(model: str, **kwargs) -> Agent:
    """New per-subtask agent; AdaptiveGroundingModule owns its round execution."""
    from benchHarness.planner.grounding_prompts import GROUNDING_PROMPT
    from tools.planner_tools.grounding.adaptive_tools import GROUNDING_TOOLS
    return Agent(name="Grounding Agent", model=model, instructions=GROUNDING_PROMPT,
                 functions=list(GROUNDING_TOOLS), tool_choice="required", parallel_tool_calls=True)
