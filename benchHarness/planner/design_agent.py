# agents/design_agent.py
# -----------------------------------------------------------------------------
# Design Agent: shapes and stabilizes the subtask-level structure of the benchmark.
# 1) Subtask construction and selection (proposer, process_proposed_subtasks).
# 2) Iterative refinement (llm_revise_one_subtask, discard_subtask).
# When stable, call case_resolved(summary) to hand off to Grounding Agent.
# -----------------------------------------------------------------------------
from typing import Dict, Any, List, Optional
import json

from utils.registry import register_agent
from utils.agent_utils import Agent

from tools.planner_tools.design.design_tools import proposer, process_proposed_subtasks, case_resolved
from tools.planner_tools.design.subtask_refinement_tools import (
    llm_revise_one_subtask,
    discard_subtask,
)


def _format_subtasks_full(subtasks: List[Dict[str, Any]]) -> str:
    """Full view for Design Agent: id, name, description, answer_type, modalities, sample_schema, keywords."""
    view = []
    for st in subtasks or []:
        view.append({
            "id": st.get("id"),
            "name": st.get("name"),
            "description": st.get("description"),
            "answer_type": st.get("answer_type"),
            "modalities": st.get("modalities", []),
            "sample_schema": st.get("sample_schema"),
            "keywords": st.get("keywords", []),
        })
    s = json.dumps(view, ensure_ascii=False, indent=2)
    return s


def _format_design_agent_history(entries: List[Dict[str, Any]]) -> str:
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


@register_agent("get_design_agent")
def get_design_agent(model: str, **kwargs) -> Agent:

    def instructions(ctx: Dict[str, Any]) -> str:
        description = ctx.get("description", "")
        short_topic = ctx.get("short_topic", "")
        modalities = ctx.get("modalities", []) or []
        keywords = ctx.get("keywords", []) or []
        subtasks = ctx.get("subtasks", []) or []
        proposed_subtasks = ctx.get("proposed_subtasks", []) or []
        design_agent_history = ctx.get("design_agent_history", []) or []
        design_agent_history_str = _format_design_agent_history(design_agent_history)

        subtasks_str = _format_subtasks_full(subtasks) if subtasks else "Empty (call proposer() then process_proposed_subtasks(accepted_subtask_ids) to build the working set)."
        proposed_str = _format_subtasks_full(proposed_subtasks) if proposed_subtasks else "None (call proposer() to get candidates)."
        grounding_feedback = ctx.get("grounding_feedback") or ""
        allocation_feedback = ctx.get("allocation_feedback") or ""

        return f"""You are the benchmark Design Agent. Turn the user's evaluation goal into a coherent working set of 1-3 subtasks, ready for Grounding. Decide which capabilities and meaningful variants to evaluate, and specify their QA interfaces. Concrete sample construction and difficulty mechanisms belong to the generator.

Define each subtask as an individual input-to-answer task; comparisons across samples are outside the subtask requirements.

Design criteria
- Every subtask must directly exercise the user's core capability. Give each a clear evaluation purpose; together the set should cover the requested benchmark. They need not each repeat the entire requirement. Avoid detached prerequisite micro-skills, redundant tasks and splits based only on wording, topic or answer format.
- Make descriptions concise: state what capability is tested and the distinction that matters. No fixed sentence count, full samples or prescribed construction recipe. Keep enough flexibility for diverse, challenging instances.
- Use existing subtask fields: id, name, description, answer_type, modalities, sample_schema, keywords. Preserve the user's explicit requirements. The supplied topic and retrieval keywords provide context, not new user constraints; work on subtasks without rewriting the global scope fields.
- Keep schema structure minimal, not test content easy. Input requires question; add context, image_url or audio_url only when needed. For media tasks avoid redundant textual replacements for the required perception. Output contains only answer. Field dtype is text/image/audio. Image and audio fields use str|list for one path or a path list; other types are str/list. Choose binary, choice, label or span to suit the capability; prefer choice when equally suitable, without weakening the target. Choice options belong in question, not context or a separate options field.
- Consider whether a task could plausibly use existing or transformable data, but leave actual dataset validation to Grounding. Do not weaken the capability merely to make data matching easier.

Decision and tools
Use the current working state, candidates and concrete feedback to choose the next action. Tool definitions supply argument schemas; include a brief decision_rationale naming the reason for the action.
- If candidates are pending, resolve them with process_proposed_subtasks(accepted_subtask_ids): accept useful candidates or pass [] to reject them. An existing id replaces that working task; a new id adds one. Make replacement deliberate and preserve unaffected tasks.
- Call proposer(guidance=...) when the working set is empty or a specific coverage/design gap needs new candidates, with no unresolved proposal. Give concise steering about that gap. A new proposal is not mandatory for an already sound working set.
- Call llm_revise_one_subtask(subtask_id, guidance) for a specific defect in an accepted task. State the defect and intended change, including the target capability when needed; preserve unrelated content and ids. If feedback concerns retrieval terms, refine those terms before changing the evaluation goal.
- Call discard_subtask for redundancy, target mismatch or an unresolved flaw that cannot usefully be revised; check remaining coverage.
- Call case_resolved(summary) when the nonempty working set meets the design criteria, pending candidates are resolved, and no concrete design defect remains. Do not keep rewriting merely to polish wording or claim stability. Summarize the resulting coverage and any unresolved data-availability caveat for Grounding.

Grounding or Allocation feedback can concern task definition, retrieval coverage or sample quantities. Determine which it actually supports changing. Apply justified local changes, preserve the core target and unaffected tasks, and do not treat a failed search or quota shortage as proof that the capability is invalid. Use history to avoid repeating unsuccessful actions without new evidence; current state takes precedence over older history.

User requirement: {description}
Scope context: short_topic={short_topic}; modalities={modalities}; keywords={keywords}

Working subtasks:
{subtasks_str}

Pending candidates:
{proposed_str}

Earlier Design runs:
{design_agent_history_str}

Grounding feedback:
{grounding_feedback if grounding_feedback else "None."}

Allocation feedback:
{allocation_feedback if allocation_feedback else "None."}
"""

    tools = [
        proposer,
        process_proposed_subtasks,
        llm_revise_one_subtask,
        discard_subtask,
        case_resolved,
    ]

    return Agent(
        name="Design Agent",
        model=model,
        instructions=instructions,
        functions=tools,
        tool_choice="required",
        parallel_tool_calls=False,
    )
