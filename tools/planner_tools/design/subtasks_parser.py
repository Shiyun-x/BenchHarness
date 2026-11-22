# tools/subtask_parser.py
# -----------------------------------------------------------------------------
# Analyst Parser tool: parse free-form benchmark description into structured subtasks.
# This tool is an LLM-based parser that transforms user requirements into a set of benchmark subtasks.
# Used as the first step in the benchmark construction pipeline.
# -----------------------------------------------------------------------------
import json
import re
from typing import Dict, Any, List, Optional
from utils.llm_caller import llm_call_json


PROMPT_TEMPLATE = r"""
Propose 1-3 useful evaluation subtasks for this benchmark goal. Each must directly test the user's core capability and have a distinct purpose; the set should collectively cover the request. Do not require every task to repeat the entire request or force a split when one task suffices. Avoid generic taxonomies, isolated prerequisite skills and differences based only on topic, wording or answer format.

Define each subtask as an individual input-to-answer task; comparisons across samples are outside the subtask requirements.

Describe WHAT each task tests and its meaningful distinction, concisely and without a fixed sentence count. Do not design final samples or prescribe how every sample must become difficult. Leave room for diverse challenging constructions. Consider plausible data use without asserting that datasets have already been found; preserve the evaluation goal rather than choosing a weaker task for easy retrieval.

Use any provided guidance and working set to complement or improve the design. New subtask ids must be st_01, st_02, etc., numbered in order after the highest existing numeric id. Keep an existing id only for an intentional replacement, with unique ids within this proposal. Respect explicit user requirements. Scope topic/keywords guide retrieval, not new hard limits; use modalities appropriate to the actual task and the user's requirements.

QA interface
- Return exactly the subtask fields shown below. Use answer_type binary, choice, label or span as appropriate; prefer choice when equally suitable, not at the expense of the tested capability.
- Input fields are question (required), context, image_url and audio_url (optional). Include only fields the task needs. Simple schema does not mean an easy test. Do not duplicate answer-bearing media with redundant textual context.
- Output has only answer. Each field has dtype (text/image/audio), subtype and type. Image and audio fields use str|list for one path or a list of paths; other fields use str/list. The answer uses dtype text and type str; answer subtypes are binary:yes_no, choice:multiple_choice, label:multi_class, span:extractive_span.
- For choice tasks, options go in question, never in context or a separate field. Do not add evidence, rationale, explanation, supporting_turns, candidates or answer_id fields.
- Give 3-6 short retrieval keywords per task, describing useful capability/task cues. Do not add unused modality fields as placeholders.

Input
task_id: {task_id}
target_size: {target_size}
User requirement:
{description}

Return only one valid JSON object with key subtasks. The following shows a minimal text-task structure; adapt the values and add allowed input fields only as needed:
{{"subtasks":[{{"id":"st_01","name":"<capability-focused name>","description":"<tested capability and meaningful distinction>","answer_type":"choice","modalities":["text"],"sample_schema":{{"input":{{"fields":{{"question":{{"dtype":"text","subtype":"question","type":"str"}}}}}},"output":{{"fields":{{"answer":{{"dtype":"text","subtype":"multiple_choice","type":"str"}}}}}}}},"keywords":["<cue 1>","<cue 2>","<cue 3>"]}}]}}
"""

SCOPE_PROMPT_TEMPLATE = r"""
You are the **Benchmark Scope Planner**, the first step in an automated benchmark construction pipeline.

A user wants to build an evaluation benchmark for large language/multimodal models. They have described the evaluation target of benchmark in natural language. 

Your task is to extract the benchmark-level scope from the description.

Input:
- task_id: {task_id}
- target_size: {target_size}
- raw_user_description:
\"\"\"{description}\"\"\"

Return STRICT JSON only, with exactly these keys:
{{
  "short_topic": "string",
  "modalities": ["text"],
  "keywords": ["keyword1", "keyword2", "keyword3"]
}}
Rules:
1. "short_topic" must be a concise benchmark topic name, not a sentence.
2. "short_topic" should describe the overall evaluation focus.
3. "modalities" must be a JSON array of unique lowercase strings.
4. Allowed modality values are: "text", "image", "audio".
5. Choose the minimal set of modalities clearly required by the user description.
6. "keywords" must contain 3-8 short retrieval-oriented phrases.
7. Keywords should help retrieve relevant datasets or benchmark resources.
8. Prefer concrete capability/task phrases over vague words.
9. If the description is underspecified, make the most conservative reasonable interpretation.
Output requirements:
- Return only one JSON object.
- No markdown.
- No explanations.
- No extra keys.
"""
from typing import Any, Dict, List, Optional

def _as_list_str(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    return [str(v)]

def _normalize_field_spec(fs: Any) -> Dict[str, Any]:
    """
    Field spec must be {"dtype": "...", "subtype": "...", type:"..."}.
    """
    out: Dict[str, Any] = {}
    if isinstance(fs, dict):
        out["dtype"] = str(fs.get("dtype") or "").strip().lower()
        out["subtype"] = str(fs.get("subtype") or "").strip().lower()
        out["type"] = str(fs.get("type") or "").strip().lower()
    else:
        out["dtype"] = ""
        out["subtype"] = ""
        out["type"] = "str"

    # basic sanitize
    if out["dtype"] not in ("audio", "text", "image"):
        # keep empty; do not invent
        out["dtype"] = out["dtype"] or ""
    if out["dtype"] in ("image", "audio"):
        out["type"] = "str|list"
    elif out["type"] not in ("str", "list"):
        out["type"] = "str"
    out["subtype"] = out["subtype"] or ""
    return out

def _normalize_sample_schema(ss: Any) -> Dict[str, Any]:
    """
    Ensure a simple QA schema shape:
    {
      "input": {"fields": {question, optional context/modality_url}},
      "output": {"fields": {answer}}
    }
    """
    if not isinstance(ss, dict):
        return {"input": {"fields": {}}, "output": {"fields": {}}}

    inp = ss.get("input") if isinstance(ss.get("input"), dict) else {}
    outp = ss.get("output") if isinstance(ss.get("output"), dict) else {}

    in_fields = inp.get("fields") if isinstance(inp.get("fields"), dict) else {}
    out_fields = outp.get("fields") if isinstance(outp.get("fields"), dict) else {}

    norm_in: Dict[str, Any] = {}
    allowed_input_fields = {"question", "context", "image_url", "audio_url"}
    for k, v in in_fields.items():
        key = str(k).strip()
        if not key or key not in allowed_input_fields:
            continue
        norm_in[key] = _normalize_field_spec(v)

    answer_spec = out_fields.get("answer", {}) if isinstance(out_fields, dict) else {}
    norm_out: Dict[str, Any] = {"answer": _normalize_field_spec(answer_spec)}

    return {"input": {"fields": norm_in}, "output": {"fields": norm_out}}

# ---- key normalization logic below ----

_ALLOWED_ANSWER_TYPES = {"binary", "choice", "label", "span"}

def _normalize_subtask(st: Dict[str, Any]) -> Dict[str, Any]:
    st = dict(st or {})

    # basic fields
    st.setdefault("id", "st_unknown")
    st.setdefault("name", None)
    st.setdefault("description", "No description.")
    st.setdefault("sample_schema", {"input": {"fields": {}}, "output": {"fields": {}}})
    st.setdefault("keywords", [])
    st.setdefault("modalities", [])
    st.setdefault("answer_type", "choice")

    # normalize id / name / description
    st["id"] = (str(st["id"]).strip() or "st_unknown").lower().replace(" ", "_")
    st["name"] = str(st["name"] or "").strip() or st["id"]
    st["description"] = str(st["description"]).strip() or "No description."

    # normalize answer_type
    at = str(st.get("answer_type") or "").strip().lower()
    if at not in _ALLOWED_ANSWER_TYPES:
        at = "choice"
    st["answer_type"] = at

    # normalize modalities / keywords
    st["modalities"] = _as_list_str(st.get("modalities"))
    st["keywords"] = _as_list_str(st.get("keywords"))

    # normalize sample_schema
    st["sample_schema"] = _normalize_sample_schema(st.get("sample_schema"))
    in_fields = st["sample_schema"]["input"]["fields"]
    if "question" not in in_fields:
        in_fields["question"] = {"dtype": "text", "subtype": "question", "type": "str"}
    out_fields = st["sample_schema"]["output"]["fields"]

    # Safety: ensure output has answer field
    if "answer" not in out_fields:
        out_fields["answer"] = {"dtype": "text", "subtype": "", "type": "str"}

    # set a more appropriate subtype from answer_type (avoid free_text)
    ans = out_fields["answer"]
    if not ans.get("subtype"):
        if at == "binary":
            ans["subtype"] = "yes_no"
        elif at == "choice":
            ans["subtype"] = "multiple_choice"
        elif at == "label":
            ans["subtype"] = "multi_class"
        else:
            ans["subtype"] = "span"

    return st

def _assign_subtask_ids(
    subtasks: List[Dict[str, Any]],
    existing_working_subtasks: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Number additions deterministically while keeping explicit replacements stable."""
    existing_ids = {
        str(st.get("id") or "").strip()
        for st in (existing_working_subtasks or [])
    }
    numbers = [int(match.group(1)) for sid in existing_ids
               if (match := re.fullmatch(r"st_(\d{2,})", sid))]
    next_number = max(numbers, default=0) + 1
    assigned, seen = [], set()
    for subtask in subtasks:
        st = dict(subtask)
        sid = st["id"]
        if sid in existing_ids:
            if sid in seen:
                raise ValueError(f"Duplicate replacement subtask ID: {sid}")
        else:
            sid = f"st_{next_number:02d}"
            next_number += 1
        st["id"] = sid
        seen.add(sid)
        assigned.append(st)
    return assigned


def plan_benchmark_scope(
    task_id: str,
    description: str,
    target_size: int = 450,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1500,
    model_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Plan benchmark-level scope only: short_topic, modalities, keywords.
    Used before Design Agent; proposer then receives these as parameters and only proposes subtasks.
    """
    user_prompt = SCOPE_PROMPT_TEMPLATE.format(
        task_id=task_id, description=description, target_size=target_size
    )
    resp = llm_call_json(
        system_prompt="",
        user_prompt=user_prompt,
        model=model or None,
        model_config_path=model_config_path,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if not resp.get("ok"):
        raise RuntimeError(
            f"[plan_benchmark_scope] JSON parse failed: {resp.get('error')}\n"
            f"raw={resp.get('raw_text','')[:600]}"
        )
    data: Dict[str, Any] = resp["json"] or {}
    out = {
        "short_topic": str(data.get("short_topic") or "benchmark").strip(),
        "modalities": _as_list_str(data.get("modalities")),
        "keywords": _as_list_str(data.get("keywords")),
    }
    return out


def parse_topic_to_subtasks(
    task_id: str,
    description: str,
    target_size: int = 3000,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 3000,
    modalities: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    short_topic: Optional[str] = None,
    proposal_guidance: Optional[str] = None,
    existing_working_subtasks: Optional[List[Dict[str, str]]] = None,
    model_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Decompose user requirement into subtasks. When modalities/keywords/short_topic are provided
    (e.g. from plan_benchmark_scope), they are injected into the prompt as benchmark-level guidance.
    proposal_guidance can be supplied by the Design Agent to steer a fresh proposal after reviewing
    an earlier candidate set. existing_working_subtasks can be supplied so the proposer knows what
    is already in the current working set and can avoid naive duplication.
    """
    user_prompt = PROMPT_TEMPLATE.format(
        task_id=task_id, description=description, target_size=target_size
    )
    if modalities is not None or keywords is not None or short_topic is not None:
        user_prompt += "\n\n[Benchmark scope guidance]\n"
        user_prompt += "Use this scope as guidance when proposing subtasks. Stay aligned with it, but optimize for a strong candidate set for the user's benchmark goal.\n"
        user_prompt += f"- short_topic: {short_topic or 'benchmark'}\n"
        user_prompt += f"- modalities: {modalities or []}\n"
        user_prompt += f"- keywords: {keywords or []}\n"
    if proposal_guidance is not None and str(proposal_guidance).strip():
        user_prompt += "\n\n[Design guidance for this proposal]\n"
        user_prompt += str(proposal_guidance).strip() + "\n"
    if existing_working_subtasks:
        user_prompt += "\n\n[Current working set summary]\n"
        user_prompt += (
            "These subtasks are already in the current working set. "
            "Use them as context so your proposal can complement, improve, replace, or rethink the current set rather than naively repeating it.\n"
        )
        user_prompt += json.dumps(existing_working_subtasks, ensure_ascii=False, indent=2) + "\n"
    resp = llm_call_json(
        system_prompt="",
        user_prompt=user_prompt,
        model=model or None,
        model_config_path=model_config_path,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if not resp.get("ok"):
        raise RuntimeError(
            f"[subtask_parser] JSON parse failed: {resp.get('error')}\n"
            f"raw={resp.get('raw_text','')[:600]}"
        )

    data: Dict[str, Any] = resp["json"] or {}
    data.setdefault("subtasks", [])
    if not isinstance(data.get("subtasks"), list):
        data["subtasks"] = []

    if short_topic is not None:
        data["short_topic"] = str(short_topic).strip()
    else:
        data["short_topic"] = str(data.get("short_topic") or "benchmark").strip()
    if modalities is not None:
        data["modalities"] = _as_list_str(modalities)
    else:
        data["modalities"] = _as_list_str(data.get("modalities"))
    if keywords is not None:
        data["keywords"] = _as_list_str(keywords)
    else:
        data["keywords"] = _as_list_str(data.get("keywords"))

    # Keep proposer output compact and aligned with prompt contract.
    data["subtasks"] = _assign_subtask_ids(
        [_normalize_subtask(st) for st in data["subtasks"][:3]],
        existing_working_subtasks,
    )
    return data
