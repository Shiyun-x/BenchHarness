"""Describe useful source material for dataset retrieval, before feasibility checks."""
from typing import Dict, Any, List
import json
import math

from utils.llm_caller import llm_call_json

CANONICAL_ROLES: List[str] = [
    "context", "question", "answer", "rationale", "dialog", "options",
    "image", "audio", "category", "bbox", "mask", "point", "entity", "span", "table",
]
PREFERENCE_MODEL = "gpt-6.1-sol"

DATASET_PREFERENCE_SYSTEM_PROMPT = r"""
Describe source material worth retrieving for this evaluation subtask. Prioritize HIGH RECALL: cover both directly useful datasets and materials that could support the target after transformation. Native tasks, answer formats and field arrangements need not match the final QA. Describe material opportunities rather than a finished sample design.

Use concise English retrieval phrases likely to appear in dataset cards. Describe useful content, relationships, observations and supervision; include style, length or structural cues in semantics_hint when helpful. Cover plausible alternative material forms without requiring them all in one dataset. Preserve the evaluation target while keeping preferences broad enough to recall transformable sources.

Later generation can rewrite roles, such as context into dialogue or annotations into QA. Text-to-speech can turn text/dialogue into audio for listening tasks, and Generator Proposal can retrieve useful external facts through native web search. Consider these opportunities when choosing source materials.

Return exactly one JSON object. The required top-level key is "preferred_roles": a nonempty object keyed by allowed role names, not an array. Each role maps to an object with only "semantics_hint" (string) and "weight" (number in [0,1]). Optional "preferred_domains" is an array of strings; omit when unnecessary. Do not return bare role entries, wrap the result in task/preference/result, or replace these keys with your own names.
Shape example only: {{"preferred_roles":{{"context":{{"semantics_hint":"<English material cues>","weight":0.8}}}}}}. Choose roles and actual cues for this target; do not copy the example as the answer. Role weights provide a soft structural score; hints provide text retrieval cues. Weight expresses relative usefulness, not required presence. Select roles with a plausible contribution, without a fixed role-count limit or filling every role. Do not add question/options merely because the final task is choice QA. Include domain tags only when useful; keep domain-general targets open.

Allowed roles: {canonical_roles}
context = source content; dialog = conversation; rationale = reference explanation; entity/span = annotations; other role names have their usual data-field meaning. Alternative roles need not all be present in a dataset.
Output only the JSON from opening {{ to closing }}: double-quoted keys/strings, escaped quotes and newlines inside strings, no comments, trailing commas, Markdown fences or surrounding explanation. Before sending, check the required preferred_roles wrapper and every value type.

Subtask: {sid} / {name}
Target: {desc}
Retrieval cues: {keywords}
Final QA schema: {sample_schema_str}
Answer type: {answer_type}
"""


def build_dataset_preference_user_prompt(subtask: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sid": subtask.get("id", ""),
        "name": subtask.get("name") or "",
        "desc": subtask.get("description") or "",
        "keywords": " ".join(subtask.get("keywords") or []),
        "sample_schema_str": json.dumps(subtask.get("sample_schema") or {}, ensure_ascii=False),
        "answer_type": subtask.get("answer_type") or "",
    }


def normalize_dataset_preference(pref: Any) -> Dict[str, Any]:
    """Clean the payload itself, including legacy preference fields used in queries."""
    if not isinstance(pref, dict) or not isinstance(pref.get("preferred_roles"), dict):
        raise ValueError("Dataset preference must contain a preferred_roles object.")
    cleaned = {}
    for role, cfg in pref["preferred_roles"].items():
        if role not in CANONICAL_ROLES or not isinstance(cfg, dict):
            continue
        try:
            weight = float(cfg.get("weight", 0.5))
        except (TypeError, ValueError):
            weight = 0.5
        weight = min(1.0, max(0.0, weight)) if math.isfinite(weight) else 0.5
        item = {"weight": weight}
        hint = cfg.get("semantics_hint")
        if isinstance(hint, str) and hint.strip():
            item["semantics_hint"] = hint.strip()
        # Existing cache files can still contribute their concise style/extent cues.
        for key in ("style_preferences", "content_extent_preferences"):
            values = cfg.get(key)
            if isinstance(values, list):
                values = [value.strip() for value in values if isinstance(value, str) and value.strip()]
                if values:
                    item[key] = list(dict.fromkeys(values))
        cleaned[role] = item
    if not cleaned:
        raise ValueError("Dataset preference contains no valid source-material roles.")
    result = {"preferred_roles": cleaned}
    domains = pref.get("preferred_domains")
    if isinstance(domains, list):
        domains = [value.strip() for value in domains if isinstance(value, str) and value.strip()]
        if domains:
            result["preferred_domains"] = list(dict.fromkeys(domains))
    return result


def plan_dataset_preference(
    subtask: Dict[str, Any],
    temperature: float = 0.1,
    max_tokens: int = 900,
) -> Dict[str, Any]:
    values = build_dataset_preference_user_prompt(subtask)
    prompt = DATASET_PREFERENCE_SYSTEM_PROMPT.format(
        **values, canonical_roles=", ".join(CANONICAL_ROLES),
    )
    response = llm_call_json(
        model=PREFERENCE_MODEL,
        system_prompt="Return dataset retrieval preferences as JSON.",
        user_prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if not isinstance(response, dict) or not response.get("ok"):
        error = response.get("error") if isinstance(response, dict) else "invalid LLM response"
        raise RuntimeError(f"[plan_dataset_preference] LLM request failed: {error}")
    return normalize_dataset_preference(response.get("json"))
