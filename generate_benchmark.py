from typing import List, Dict, Any, Union, Tuple
import os
import argparse
import asyncio
import json
import copy
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent

from utils.benchmark_flow import FlowModule, ToolModule, AgentModule, _load_agent_context_output_keys, _make_postprocess_fn, _select_grounding_cache_mode
from utils.constant import CONTEXT_KEYS_EXCLUDED_FROM_CACHE
from utils.logger import MetaChainLogger
from utils.schema.metadata_core import TopicMetadata
from utils.agent_utils import _messages_to_tool_trace

# tools
from tools.planner_tools.design.subtasks_parser import plan_benchmark_scope
from tools.planner_tools.design.design_tools import _carry_forward_grounding_state_for_unchanged_subtasks
from tools.planner_tools.allocation.importance_and_quota_tool import llm_plan_quota
from tools.planner_tools.allocation.allocation_tools import _build_allocation_orders, _check_allocation_gap
from tools.shared.build_gen_input import (
    build_generator_inputs,
    _load_dataset_cards,
    _load_dataset_cards_from_config,
    _build_id2card,
    _compute_gap_per_subtask,
    _refill_idx_todo_for_replenishment,
    _load_transformation_tools,
    attach_grounding_blueprints,
)

# agents
from benchHarness.planner.design_agent import get_design_agent
from benchHarness.planner.grounding_agent import get_legacy_grounding_agent
from benchHarness.planner.grounding_runtime import AdaptiveGroundingModule
from benchHarness.planner.allocation_agent import get_allocation_agent
from benchHarness.executor.sample_realization import (
    _save_transform_cache,
    _load_transformed_buffer_from_file,
    iterative_transform_batch,
)
from benchHarness.executor.verification import verify_one_subtask
from tools.shared.export_results import export_evaluation

# model config
from utils.model_config import (
    get_design_model,
    get_grounding_model,
    get_allocation_model,
    get_tool_model,
    load_model_config,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TOPIC_ID = "topic_01"
DEFAULT_MAX_WORKERS_TRANSFORM = 20
DEFAULT_MAX_VERIFY_REPLENISH_ROUNDS = 1
DEFAULT_MAX_DESIGN_GROUNDING_ROUNDS = 4
CACHE_FNAME_VERIFIED_BUFFER = "verify_log/verified_transformed_buffer_{subtask_id}.json"


def _load_topic(task_id: str, description: str, target_size: int) -> dict:
    return {"task_id": task_id, "target_size": target_size, "description": description}


class BenchmarkFlow(FlowModule):
    """
    Full benchmark pipeline orchestrator.

    Flow: scope parsing → [Design → Grounding]* → Quota → [Allocation]* → Transform → Verify
    Retry loops handle grounding rejection (inner) and allocation gap (outer).
    All agent/tool calls are wrapped with caching; backtracking triggers force_recompute.
    """

    def __init__(
        self,
        cache_path: str,
        log_path: Union[str, None, MetaChainLogger] = None,
        model: str = None,
        model_config_path: str = None,
        agent_cache_config_path: str = "utils/resources/agent_cache_config.yaml",
    ):
        self.model_config = load_model_config(model_config_path)
        if model is None:
            model = get_design_model(model_config_path)
        super().__init__(cache_path, log_path, model, model_config_path=model_config_path)
        self.cache_path = cache_path
        self.model_config_path = model_config_path

        self._load_tp = ToolModule(_load_topic, cache_path)
        self._plan_scope = ToolModule(plan_benchmark_scope, cache_path)
        self._init_importance_and_quota = ToolModule(llm_plan_quota, cache_path)
        self._generator_input = ToolModule(build_generator_inputs, cache_path)

        def _output_keys(agent):
            return _load_agent_context_output_keys(agent_cache_config_path, agent.name)

        # Each AgentModule: caches messages + context_variables; postprocess_fn appends
        # run history and strips stage-irrelevant subtask fields before writing cache.
        design_agent = get_design_agent(model=get_design_model(model_config_path))
        self.design_agent = AgentModule(
            design_agent, self.client, cache_path,
            exclude_context_keys_from_cache=CONTEXT_KEYS_EXCLUDED_FROM_CACHE,
            context_output_keys=_output_keys(design_agent),
            postprocess_fn=_make_postprocess_fn("design", "design"),
        )
        self.grounding_agent = AdaptiveGroundingModule(
            cache_path, model=get_grounding_model(model_config_path), model_config_path=model_config_path,
            logger=getattr(self.client, 'logger', None), cache_selector=_select_grounding_cache_mode)
        grounding_agent = get_legacy_grounding_agent(model=get_grounding_model(model_config_path))
        self.legacy_grounding_agent = AgentModule(
            grounding_agent, self.client, os.path.join(cache_path, "legacy_grounding"),
            exclude_context_keys_from_cache=CONTEXT_KEYS_EXCLUDED_FROM_CACHE,
            context_output_keys=_output_keys(grounding_agent),
            postprocess_fn=_make_postprocess_fn("grounding", "grounding"),
        )
        allocation_agent = get_allocation_agent(model=get_allocation_model(model_config_path))
        self.allocation_agent = AgentModule(
            allocation_agent, self.client, cache_path,
            exclude_context_keys_from_cache=CONTEXT_KEYS_EXCLUDED_FROM_CACHE,
            context_output_keys=_output_keys(allocation_agent),
            postprocess_fn=_make_postprocess_fn("allocation", "allocation"),
        )

    # ============ Design Agent ============
    async def _run_design(
        self,
        metadata: TopicMetadata,
        context_variables: Dict[str, Any],
        force_recompute: bool = False,
    ) -> TopicMetadata:
        messages = [{
            "role": "user",
            "content": (
                "From the user requirement, propose and refine the subtask set. "
                "Use proposer if you need initial or alternative candidates; use revise/discard to refine. "
                "When the design is stable, call case_resolved(summary)."
            )
        }]
        messages_out, context_variables = await self.design_agent(
            messages,
            context_variables,
            force_recompute=force_recompute,
        )
        context_variables["design_tool_trace"] = _messages_to_tool_trace(messages_out)
        metadata.topic = context_variables.get("short_topic", metadata.topic)
        metadata.global_modalities = context_variables.get("modalities", metadata.global_modalities or [])
        metadata.keywords = context_variables.get("keywords", metadata.keywords or [])
        metadata.subtasks = context_variables.get("subtasks", metadata.subtasks or [])
        return metadata

    # ============ Grounding Agent ============
    async def _run_grounding(
        self,
        metadata: TopicMetadata,
        context_variables: Dict[str, Any],
        force_recompute: bool = False,
    ) -> bool:
        """Run independent adaptive agents, or the explicit legacy compatibility path."""
        messages = [{
            "role": "user",
            "content": (
                "Find promising sources for the assigned targets. Inspect candidates as needed, "
                "obtain brief dataset-level blueprints and submit the selected sources."
            )
        }]
        runner = self.grounding_agent
        if context_variables.get("generation_mode") == "legacy":
            runner = self.legacy_grounding_agent
            for task in context_variables.get("subtasks", []):
                task.pop("grounding", None)
                task.pop("grounding_blueprints", None)
        _, context_variables = await runner(
            messages,
            context_variables,
            force_recompute=force_recompute,
        )
        result = context_variables.get("grounding_result") or {}
        metadata.subtasks = context_variables.get("subtasks", metadata.subtasks)
        status = result.get("status", "")
        if status == "accepted":
            return True
        if status in ("rejected", "incomplete"):
            context_variables["grounding_feedback"] = result.get("feedback_to_design") or result.get("reason", "")
            return False
        # Defensive fallback: unexpected status treated as rejected so retry logic can stop gracefully.
        fb = result.get("feedback_to_design") or result.get("reason") or str(result)
        print(
            "[BenchmarkFlow] Grounding returned an unexpected status; treating as rejected. "
            f"Feedback: {fb[:200]}..."
        )
        context_variables["grounding_feedback"] = fb
        return False

    # ============ Allocation Agent ============
    async def _run_allocation(
        self,
        metadata: TopicMetadata,
        context_variables: Dict[str, Any],
        force_recompute: bool = False,
    ) -> TopicMetadata:
        messages = [{
            "role": "user",
            "content": "Please allocate dataset samples across subtasks."
        }]
        cache_name = None
        if (context_variables.get("grounding_result") or {}).get("version") == "adaptive-v1":
            from tools.planner_tools.grounding.grounding_materials import digest
            cache_name = "adaptive-v1/" + digest(context_variables.get("allocation_config") or {})[:20]
        _, context_variables = await self.allocation_agent(
            messages,
            context_variables,
            force_recompute=force_recompute,
            cache_name=cache_name,
        )
        last_alloc = context_variables.get("last_allocation", {})
        context_variables["allocation"] = {
            "subtasks": last_alloc.get("subtasks", {}),
            "datasets": last_alloc.get("datasets", {}),
            "ok": last_alloc.get("ok", False),
            "unmet_total": int(last_alloc.get("unmet_total", 0) or 0),
        }
        if "last_allocation" in context_variables:
            del context_variables["last_allocation"]
        metadata.allocation = context_variables.get("allocation", {})
        return metadata

    # ============ step 4: Generator (execute + verify + replenishment) ============
    async def _run_generator(
        self,
        metadata: TopicMetadata,
        context_variables: Dict[str, Any],
        topic_id: str = "topic_01",
        force_recompute: bool = False,
        **kwargs: Any,
    ) -> TopicMetadata:
        """Execute transform batch for all subtasks, then verify (single pass in testing mode)."""
        subtask_order = list(context_variables["subtask_order"])
        requested_subtask = kwargs.get('generation_subtask')
        if requested_subtask:
            if requested_subtask not in subtask_order:
                raise ValueError(f'Unknown or unallocated generation subtask: {requested_subtask}. '
                                 f'Available: {subtask_order}')
            subtask_order = [requested_subtask]
            context_variables = {**context_variables, 'subtask_order': subtask_order}
            print(f'[BenchmarkFlow] Generation limited to {requested_subtask}; other subtasks are skipped.')
        max_workers = int(kwargs.get("max_workers_transform", DEFAULT_MAX_WORKERS_TRANSFORM))
        max_replenish_rounds = int(kwargs.get("max_verify_replenish_rounds", DEFAULT_MAX_VERIFY_REPLENISH_ROUNDS))
        soft_minimum = kwargs.get('min_accepted_samples')
        if self.model_config_path:
            policy_path = Path(self.model_config_path).resolve().parent/'generation_policy.json'
            if policy_path.is_file():
                policy = json.loads(policy_path.read_text(encoding='utf-8-sig'))
                if policy.get('executor_argument_repair'):
                    context_variables['executor_argument_repair'] = copy.deepcopy(policy['executor_argument_repair'])
                max_workers = int(policy.get('executor_workers', max_workers))
                if max_workers < 1:
                    raise ValueError('executor_workers must be positive')
                if context_variables.get('generation_guidance'):
                    if soft_minimum is None:
                        soft_minimum = policy.get('iteration_minimum_samples')
                else:
                    max_replenish_rounds = int(policy.get('initial_generation_rounds', max_replenish_rounds))
        allocation_subtasks = (metadata.allocation or {}).get("subtasks") or {}

        mode = kwargs.get('generation_mode', 'planned')
        if mode not in ('planned', 'legacy'):
            raise ValueError('generation_mode must be planned or legacy')
        if not subtask_order:
            print('[BenchmarkFlow] No allocated samples to generate.')
            return metadata
        context_variables['generation_mode'] = mode
        context_variables['max_web_searches'] = int(kwargs.get('max_web_searches', 3))
        if mode == 'planned':
            # Never read legacy transformation/verification caches in the new path.
            output_root = os.path.join(self.cache_path, 'generation', 'planned-v1')
            context_variables['generation_output_root'] = output_root
            context_variables['transformed_buffer'] = {}
            context_variables['generation_verified'] = {}
            context_variables['generation_results'] = {}
            context_variables['reuse_accepted_cache'] = bool(kwargs.get('reuse_accepted_cache', False))
            context_variables['generation_reused_accepted'] = {}
            # A manually shortened cached todo is a fixed test selection, not
            # a request to refill back to the allocation's original quota.
            todo_counts = {sid: len({(p['dataset_id'], idx)
                for p in context_variables['pairs_by_subtask'][sid]
                for idx in p.get('idx_todo', [])}) for sid in subtask_order}
            sampled_todo = any(todo_counts[sid] < max(
                int(allocation_subtasks.get(sid, {}).get('quota', 0) or 0),
                sum(int(p.get('need_n', 0) or 0)
                    for p in context_variables['pairs_by_subtask'][sid]))
                for sid in subtask_order)
            if sampled_todo:
                max_replenish_rounds = 1
                allocation_subtasks = {sid: {'quota': todo_counts[sid]} for sid in subtask_order}
                print(f'[BenchmarkFlow] Sampled idx_todo: {todo_counts}; automatic refill disabled.')
            active_subtasks = list(subtask_order)
            for replenish_round in range(max(1, max_replenish_rounds)):
                for subtask_number, subtask_id in enumerate(active_subtasks, 1):
                    print(f'[BenchmarkFlow] Generation round={replenish_round + 1}/{max(1, max_replenish_rounds)}; '
                          f'subtask={subtask_number}/{len(active_subtasks)} ({subtask_id})', flush=True)
                    context_variables['current_subtask_id'] = subtask_id
                    context_variables['current_pairs'] = context_variables['pairs_by_subtask'][subtask_id]
                    context_variables['model_config_path'] = self.model_config_path
                    context_variables = iterative_transform_batch(
                        context_variables=context_variables,
                        cache_root=None if force_recompute else output_root,
                        max_workers=max_workers)
                    _save_transform_cache(output_root, context_variables, subtask_id)
                    result = context_variables['generation_verified'][subtask_id]
                    from benchHarness.executor.planned_generation import write_json
                    write_json(os.path.join(output_root, CACHE_FNAME_VERIFIED_BUFFER.format(subtask_id=subtask_id)), result)
                gaps = _compute_gap_per_subtask(subtask_order, context_variables['generation_verified'], allocation_subtasks)
                accepted_count = sum(sum(len(rows) for rows in value.get('verified_buffer', {}).values())
                                     for value in context_variables['generation_verified'].values())
                if soft_minimum is not None and accepted_count >= int(soft_minimum):
                    print(f'[BenchmarkFlow] Soft sample target met: {accepted_count} accepted; '
                          f'minimum={soft_minimum}.')
                    break
                if not gaps or replenish_round + 1 >= max(1, max_replenish_rounds):
                    break
                if not _refill_idx_todo_for_replenishment(context_variables, gaps, allocation_subtasks):
                    break
                active_subtasks = list(gaps)
            if gaps:
                print(f'[BenchmarkFlow] Remaining content quota gap: {gaps}')
            # Preserve the actual source cohort after refill for feedback iteration.
            write_json(os.path.join(output_root, 'generator_inputs_final.json'), {
                'benchmark_task_id': context_variables.get('benchmark_task_id') or context_variables.get('task_id'),
                'pairs_by_subtask': context_variables['pairs_by_subtask'],
                'dataset_index_pool': context_variables.get('dataset_index_pool', {}),
                'subtask_order': subtask_order,
                'allocation': metadata.allocation,
            })
            write_json(os.path.join(output_root, 'generation_summary.json'), {
                'version': 'planned-v1', 'quota_gaps': gaps,
                'initial_todo_counts': todo_counts, 'sampled_todo': sampled_todo,
                'reused_accepted_cache': context_variables.get('generation_reused_accepted', {}),
                'stats': {sid: result['stats'] for sid, result in context_variables['generation_verified'].items()}})
            from benchHarness.executor.generation_contracts import fingerprint
            results = context_variables['generation_results']
            write_json(os.path.join(output_root, 'generation_results_final.json'),
                       {'results': results, 'digest': fingerprint(results)})
            # These results already passed the short inspector; do not inspect again.
            export_evaluation(output_root, subtask_ids=subtask_order)
            print(f'[BenchmarkFlow] Planned content results saved under: {output_root}')
            return metadata

        if force_recompute:
            context_variables.pop("transformed_buffer", None)
        transform_cache_root = None if force_recompute else self.cache_path
        for subtask_id in subtask_order:
            context_variables["current_subtask_id"] = subtask_id
            context_variables["current_pairs"] = context_variables["pairs_by_subtask"][subtask_id]
            context_variables["model_config_path"] = self.model_config_path
            context_variables = iterative_transform_batch(
                context_variables=context_variables,
                cache_root=transform_cache_root,
                max_workers=max_workers,
            )
            _save_transform_cache(self.cache_path, context_variables, subtask_id)

        transformed_buffer = context_variables.get("transformed_buffer") or {}
        if (not force_recompute) and (not transformed_buffer):
            transformed_buffer = _load_transformed_buffer_from_file(
                self.cache_path, context_variables["subtask_order"][-1]
            )

        verified: Dict[str, Dict[str, Any]] = {}

        # Replenishment loop is disabled for testing — single verify pass only.
        # To re-enable: uncomment the for-loop below and the gap/refill logic inside it.
        #
        # for replenish_round in range(max_replenish_rounds):

        for subtask_id in subtask_order:
            transformed_items = transformed_buffer.get(subtask_id, [])
            if not isinstance(transformed_items, list):
                transformed_items = []
            subtask = next((st for st in context_variables["subtasks"] if st.get("id") == subtask_id), None)
            if not subtask:
                continue
            verified_one = verify_one_subtask(
                subtask=subtask,
                transformed_items=transformed_items,
                topic=topic_id,
                checkpoint_dir=os.path.join(self.cache_path, "verify_log"),
                model_config_path=self.model_config_path,
                topic_user_requirements=str(context_variables.get("description") or "").strip() or None,
                topic_short_topic=str(context_variables.get("short_topic") or "").strip() or None,
                dataset_cards=context_variables.get("dataset_cards"),
            )
            verified[subtask_id] = verified_one
            vpath = os.path.join(self.cache_path, CACHE_FNAME_VERIFIED_BUFFER.format(subtask_id=subtask_id))
            os.makedirs(os.path.dirname(vpath), exist_ok=True)
            with open(vpath, "w", encoding="utf-8") as f:
                json.dump(verified_one, f, ensure_ascii=False, indent=2)
            stats = verified_one.get("stats", {})
            print(f"[BenchmarkFlow] Verified {subtask_id} -> ok={stats.get('ok', 0)} fixed={stats.get('fixed', 0)} rejected={stats.get('rejected', 0)}")

            # gap_per_subtask = _compute_gap_per_subtask(subtask_order, verified, allocation_subtasks)
            # if not gap_per_subtask:
            #     print("[BenchmarkFlow] Verification done; all subtask quotas met.")
            #     break
            # if replenish_round + 1 >= max_replenish_rounds:
            #     print(f"[BenchmarkFlow] Verification stopped after {max_replenish_rounds} round(s); remaining gap: {gap_per_subtask}")
            #     break
            # total_added = _refill_idx_todo_for_replenishment(context_variables, gap_per_subtask, allocation_subtasks)
            # if total_added == 0:
            #     print(f"[BenchmarkFlow] Replenishment: no more indices in pool; remaining gap: {gap_per_subtask}")
            #     break
            # print(f"[BenchmarkFlow] Replenishment round {replenish_round + 1}: added {total_added} indices; re-running transform for: {list(gap_per_subtask.keys())}")
            # pairs_by_subtask = context_variables.get("pairs_by_subtask") or {}
            # for subtask_id in gap_per_subtask:
            #     pairs = pairs_by_subtask.get(subtask_id) or []
            #     if not any((p.get("idx_todo") or []) for p in pairs):
            #         continue
            #     context_variables["current_subtask_id"] = subtask_id
            #     context_variables["current_pairs"] = pairs
            #     context_variables["model_config_path"] = self.model_config_path
            #     context_variables = iterative_transform_batch(
            #         context_variables=context_variables, cache_root=self.cache_path, max_workers=max_workers,
            #     )
            #     transformed_buffer = context_variables.get("transformed_buffer") or transformed_buffer
            #     _save_transform_cache(self.cache_path, context_variables, subtask_id)

        print(f"[BenchmarkFlow] Verified buffers saved under: {self.cache_path}")
        export_evaluation(self.cache_path, subtask_ids=subtask_order)

        return metadata

    def _reinject_context_after_cache_load(
        self, context_variables: Dict[str, Any], tool_config_path: str = "./utils/resources/tools.yaml"
    ) -> None:
        context_variables["cache_root"] = self.cache_path
        context_variables["cache_path"] = self.cache_path
        context_variables["model_config_path"] = self.model_config_path
        if "id2card" not in context_variables and context_variables.get("dataset_cards"):
            context_variables["id2card"] = _build_id2card(context_variables["dataset_cards"])
        if "tools_list" not in context_variables:
            context_variables["tools_list"] = _load_transformation_tools(context_variables, tool_config_path)

    async def _run_design_grounding_allocation(
        self,
        metadata: TopicMetadata,
        context_variables: Dict[str, Any],
        tool_config_path: str,
        **kwargs: Any,
    ) -> Tuple[TopicMetadata, Dict[str, Any]]:
        """Design → Grounding (until accepted) → Quota → Allocation; loop on allocation gap."""
        max_design_grounding_rounds = DEFAULT_MAX_DESIGN_GROUNDING_ROUNDS
        max_allocation_rounds = int(kwargs.get("max_allocation_rounds", 3))
        allocation_unmet_ratio_threshold = float(kwargs.get("allocation_unmet_ratio_threshold", 0.0))
        allocation_feedback = ""
        initial_force_recompute = bool(kwargs.get("force_recompute", False))
        force_recompute_chain = initial_force_recompute

        # Outer loop: retry Design+Grounding+Allocation when allocation gap is too large.
        for allocation_round in range(max_allocation_rounds):
            backtracking_happened = allocation_round > 0
            if allocation_round > 0:
                # Clear stale results so agents run fresh; pass feedback to guide redesign.
                context_variables["allocation_feedback"] = allocation_feedback
                context_variables.pop("design_result", None)
                context_variables.pop("grounding_result", None)
                context_variables.pop("grounding_feedback", None)
                context_variables.pop("allocation_result", None)
                context_variables.pop("last_allocation", None)
                context_variables.pop("last_diagnosis", None)
            # Inner loop: retry Design+Grounding when grounding rejects the subtask set.
            for round_index in range(max_design_grounding_rounds):
                if round_index > 0:
                    context_variables.pop("design_result", None)
                    context_variables.pop("grounding_result", None)
                is_backtracking = force_recompute_chain or (allocation_round > 0) or (round_index > 0)
                backtracking_happened = backtracking_happened or is_backtracking
                force_recompute_chain = force_recompute_chain or is_backtracking
                subtasks_before_design = copy.deepcopy(context_variables.get("subtasks", []) or [])
                metadata = await self._run_design(
                    metadata,
                    context_variables,
                    force_recompute=is_backtracking,
                )
                self._reinject_context_after_cache_load(context_variables, tool_config_path)
                if is_backtracking and (not initial_force_recompute):
                    # Preserve grounding progress for subtasks whose design didn't change.
                    context_variables["subtasks"] = _carry_forward_grounding_state_for_unchanged_subtasks(
                        old_subtasks=subtasks_before_design,
                        new_subtasks=list(context_variables.get("subtasks", []) or []),
                    )
                    # Disable stage file caches so changed subtasks recompute from scratch.
                    context_variables["_use_grounding_stage_cache"] = False
                design_result = context_variables.get("design_result") or {}
                if design_result.get("status") != "stabilized":
                    print("[BenchmarkFlow] Design did not stabilized.")
                    context_variables["_benchmark_flow_status"] = "design_not_stabilized"
                    context_variables["_benchmark_flow_feedback"] = str(design_result)
                    return metadata, context_variables
                grounding_accepted = await self._run_grounding(
                    metadata,
                    context_variables,
                    force_recompute=is_backtracking,
                )
                self._reinject_context_after_cache_load(context_variables, tool_config_path)
                if grounding_accepted:
                    break
                if (context_variables.get("grounding_result") or {}).get("status") == "incomplete":
                    context_variables["_benchmark_flow_status"] = "grounding_incomplete"
                    context_variables["_benchmark_flow_feedback"] = context_variables.get("grounding_feedback", "")
                    return metadata, context_variables
            else:
                fb = str(context_variables.get("grounding_feedback", "") or "").strip()
                print("[BenchmarkFlow] Grounding was rejected too many times (reached max_design_grounding_rounds).")
                context_variables["_benchmark_flow_status"] = "grounding_rejected"
                context_variables["_benchmark_flow_feedback"] = fb
                return metadata, context_variables

            quota_model = get_tool_model("importance_and_quota", self.model_config_path)
            quota_cache = None
            if (context_variables.get("grounding_result") or {}).get("version") == "adaptive-v1":
                from tools.planner_tools.grounding.grounding_materials import digest, task_view
                quota_cache = "adaptive-v1/" + digest({"target_size": metadata.target_size,
                    "tasks": [{"id": task["id"], **task_view(task)} for task in context_variables.get("subtasks", [])],
                    "model": quota_model})[:20]
            subtasks, notes = self._init_importance_and_quota({
                "short_topic": metadata.topic,
                "target_size": metadata.target_size,
                "subtasks": context_variables.get("subtasks", []),
                "model": quota_model,
                "model_config_path": self.model_config_path,
            }, force_recompute=force_recompute_chain or backtracking_happened, cache_name=quota_cache)
            # Quota caches own importance/quota, never the Grounding products.
            if quota_cache:
                quotas = {task["id"]: task for task in subtasks}
                for task in context_variables.get("subtasks", []):
                    if task["id"] not in quotas:
                        raise ValueError("Quota result does not cover current Grounding subtasks.")
                    for key in ("importance", "quota"):
                        task[key] = quotas[task["id"]][key]
            else:
                context_variables["subtasks"] = subtasks
            context_variables["quota_notes"] = notes
            context_variables["allocation_config"] = _build_allocation_orders(context_variables)
            metadata = await self._run_allocation(
                metadata,
                context_variables,
                force_recompute=force_recompute_chain or backtracking_happened,
            )
            self._reinject_context_after_cache_load(context_variables, tool_config_path)

            allocation_accepted, allocation_feedback = _check_allocation_gap(
                metadata, context_variables, unmet_ratio_threshold=allocation_unmet_ratio_threshold
            )
            if allocation_accepted:
                context_variables["_force_recompute_downstream"] = force_recompute_chain or backtracking_happened
                context_variables["_benchmark_flow_status"] = "ok"
                return metadata, context_variables
            print(f"[BenchmarkFlow] Allocation gap too large (round {allocation_round + 1}/{max_allocation_rounds}), re-running Design+Grounding. Feedback: {allocation_feedback[:200]}...")
            force_recompute_chain = True

        print("[BenchmarkFlow] Allocation did not become acceptable within max_allocation_rounds.")
        context_variables["_force_recompute_downstream"] = force_recompute_chain
        context_variables["_benchmark_flow_status"] = "allocation_gap_rejected"
        context_variables["_benchmark_flow_feedback"] = allocation_feedback
        return metadata, context_variables

    # ============ main entry ============
    async def forward(
        self,
        task_id: str,
        description: str,
        target_size: int = 2000,
        dataset_card_dir: str = "./dataset_cards",
        topic_id: str = DEFAULT_TOPIC_ID,
        *args,
        **kwargs,
    ) -> TopicMetadata:
        # This value is passed explicitly below; do not forward it again in kwargs.
        tool_config_path = kwargs.pop("tool_config_path", "./utils/resources/tools.yaml")
        guidance_path = kwargs.pop("generation_guidance_path", None)
        generation_guidance = {}
        if guidance_path is not None:
            if kwargs.get("generation_mode", "planned") != "planned":
                raise ValueError('generation_guidance_path requires planned generation mode')
            from benchHarness.executor.generation_guidance import load_generation_guidance_map
            generation_guidance = load_generation_guidance_map(guidance_path, task_id)
        force_recompute_chain = bool(kwargs.get("force_recompute", False))
        context_variables: Dict[str, Any] = {
            "task_id": task_id,
            "topic_id": topic_id,
            "description": description,
            "target_size": target_size,
            "dataset_card_dir": dataset_card_dir,
            "cache_root": self.cache_path,
            "cache_path": self.cache_path,
            "generation_mode": kwargs.get("generation_mode", "planned"),
            "max_web_searches": int(kwargs.get("max_web_searches", 3)),
            "max_workers_grounding": int(kwargs.get("max_workers_grounding", 4)),
            "grounding_sample_count": int(kwargs.get("grounding_sample_count", 3)),
            "grounding_budgets": kwargs.get("grounding_budgets") or {},
        }
        topic_dict = self._load_tp({
            "task_id": task_id,
            "target_size": target_size,
            "description": description,
        })
        metadata = TopicMetadata(**topic_dict)

        # Step 1: load dataset cards and parse description → short_topic, modalities, keywords.
        dataset_cards = kwargs.pop("dataset_cards", None) or _load_dataset_cards(dataset_card_dir)
        context_variables["dataset_cards"] = dataset_cards
        id2card = _build_id2card(dataset_cards)
        context_variables["id2card"] = id2card

        scope_model = get_tool_model("parse_subtasks", self.model_config_path)
        scope = self._plan_scope({
            "task_id": task_id,
            "description": description,
            "target_size": target_size,
            "model": scope_model,
            "model_config_path": self.model_config_path,
        }, force_recompute=force_recompute_chain)
        context_variables.update({
            "short_topic": scope.get("short_topic", "benchmark"),
            "modalities": scope.get("modalities", []),
            "keywords": scope.get("keywords", []),
            "subtasks": [],
            "proposed_subtasks": [],
            "change_log": [],
            "design_agent_history": [],
            "grounding_agent_history": [],
        })
        context_variables["model_config_path"] = self.model_config_path
        context_variables["tools_list"] = _load_transformation_tools(context_variables, tool_config_path)

        # Steps 2–4: Design → Grounding → Allocation (with retry loops; see _run_design_grounding_allocation).
        metadata, context_variables = await self._run_design_grounding_allocation(
            metadata, context_variables, tool_config_path, **kwargs
        )
        force_recompute_chain = force_recompute_chain or bool(
            context_variables.get("_force_recompute_downstream", False)
        )
        if context_variables.get("_benchmark_flow_status") and context_variables["_benchmark_flow_status"] != "ok":
            status = context_variables.get("_benchmark_flow_status")
            fb = context_variables.get("_benchmark_flow_feedback") or ""
            fb = (fb[:300] + "...") if isinstance(fb, str) and len(fb) > 300 else fb
            print(f"[BenchmarkFlow] Flow stopped early: {status}. Feedback: {fb}")
            return metadata

        # Step 5: strip heavy grounding fields, build (dataset, subtask) input pairs for the generator.
        for st in context_variables.get("subtasks", []):
            st.pop("retrieval_result", None)
            st.pop("scored_candidates", None)
            st.pop("domain_match", None)
        context_variables.pop("dataset_cards", None)
        context_variables.pop("id2card", None)

        print("[BenchmarkFlow] Building generator inputs...")
        generator_input_args = {
            "benchmark_task_id": task_id,
            "subtasks": context_variables.get("subtasks", []),
            "id2card": id2card,
            "allocation": metadata.allocation,
        }
        generator_input = self._generator_input(generator_input_args, force_recompute=force_recompute_chain)
        if any(task.get("grounding") for task in context_variables.get("subtasks", [])):
            generator_input = attach_grounding_blueprints(generator_input, context_variables["subtasks"])
            self._generator_input.save_cache(
                tool=build_generator_inputs, tool_args=generator_input_args, tool_result=generator_input)
        context_variables.update(generator_input)
        # Restore only the dataset cards needed by the generator (subset of full id2card).
        needed_datasets = set(context_variables.get("dataset_index_pool", {}))
        needed_datasets.update(pair["dataset_id"] for pairs in context_variables.get("pairs_by_subtask", {}).values()
                               for pair in pairs)
        context_variables["dataset_cards"] = {did: id2card.get(did, {}) for did in needed_datasets}
        print("[BenchmarkFlow] Generator inputs built.")

        # Step 6: parallel transform per subtask, then single-pass verify.
        if generation_guidance:
            context_variables["generation_guidance"] = generation_guidance
        generator_kwargs = {k: v for k, v in kwargs.items() if k != 'force_recompute'}
        metadata = await self._run_generator(
            metadata, context_variables, topic_id=topic_id, force_recompute=force_recompute_chain, **generator_kwargs
        )
        return metadata


def main(args: Any):
    args.instance_path = args.instance_path + args.topic_id + ".json"
    args.cache_path = args.cache_path + args.topic_id

    with open(args.instance_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    task_id = config.get("id")
    raw_description = config.get("description")
    target_size = config.get("target_size", 2000)
    bench_id = config.get("id", "benchmark")
    dataset_cards = _load_dataset_cards_from_config(args.dataset_card_config)
    if raw_description is None:
        raise ValueError("Config JSON must contain a 'description' field")

    flow = BenchmarkFlow(
        cache_path=args.cache_path,
        log_path=f"log_{bench_id}",
        model=args.model,
        model_config_path=args.model_config_path,
    )

    metadata: TopicMetadata = asyncio.run(
        flow(
            task_id=task_id,
            description=raw_description,
            target_size=target_size,
            topic_id=args.topic_id,
            dataset_cards=dataset_cards,
            tool_config_path=args.tool_config_path,
            generation_mode=args.generation_mode,
            generation_subtask=args.generation_subtask,
            generation_guidance_path=args.generation_guidance_path,
            reuse_accepted_cache=args.reuse_accepted_cache,
            max_workers_transform=args.max_workers_transform,
            max_workers_grounding=args.max_workers_grounding,
            max_web_searches=args.max_web_searches,
            max_verify_replenish_rounds=args.max_verify_replenish_rounds,
            force_recompute=args.force_recompute,
        )
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation_mode', choices=('planned', 'legacy'), default='planned',
                        help='Planned generation with optional quality refinement and bounded preflight recovery, or the previous iterative path')
    parser.add_argument('--max_workers_transform', type=int, default=DEFAULT_MAX_WORKERS_TRANSFORM)
    parser.add_argument('--generation_subtask', default=None,
                        help='Generate only this allocated subtask ID; planner and cached todos are unchanged')
    parser.add_argument('--generation_guidance_path', default=None,
                        help='Optional JSON mapping subtask IDs to guidance files for planned generation; '
                             'relative guidance paths are resolved from the JSON directory')
    parser.add_argument('--max_workers_grounding', type=int, default=4,
                        help='Shared model-call limit across all Grounding main agents and their child calls')
    parser.add_argument('--max_web_searches', type=int, default=3,
                        help='Maximum focused evidence requests per sample in planned mode')
    parser.add_argument('--force_recompute', action='store_true')
    parser.add_argument('--reuse_accepted_cache', action='store_true',
                        help='Reuse matching accepted v1.2 non-audio samples after the v1.3 audio-only prompt update; preserve their original version')
    parser.add_argument('--max_verify_replenish_rounds', type=int, default=DEFAULT_MAX_VERIFY_REPLENISH_ROUNDS,
                        help='Total generation/verification rounds in planned mode (1 disables refill)')
    parser.add_argument(
        "--instance_path",
        type=str,
        default=str(_PROJECT_ROOT) + os.sep,
        help="Directory prefix for the user-query JSON; combined with --topic_id to form the full path (e.g. user_queries/user_query_03.json)",
    )
    parser.add_argument(
        "--topic_id",
        type=str,
        default="user_queries/user_query_03",
        help="User-query identifier (path stem without .json); selects which benchmark instance to run",
    )
    parser.add_argument("--model", type=str, default=None, help="Override default agent model (deprecated, use --model_config_path instead)")
    parser.add_argument("--model_config_path", type=str, default=None, help="Path to model configuration YAML file")
    parser.add_argument(
        "--dataset_card_config",
        type=str,
        default=str(_PROJECT_ROOT / "utils" / "resources" / "dataset_cards.yaml"),
        help="Path to dataset_cards.yaml listing the dataset IDs and card directory to use",
    )
    parser.add_argument(
        "--tool_config_path",
        type=str,
        default=str(_PROJECT_ROOT / "utils" / "resources" / "tools.yaml"),
        help="Path to tools.yaml registering LLM and deterministic transformation tools",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default=str(_PROJECT_ROOT / "cache") + os.sep,
        help="Directory prefix for intermediate artifacts; combined with --topic_id (e.g. cache/user_queries/user_query_03/)",
    )
    args = parser.parse_args()
    if args.reuse_accepted_cache and (args.force_recompute or args.generation_mode != 'planned'):
        parser.error('--reuse_accepted_cache requires planned mode without --force_recompute')
    if args.generation_guidance_path is not None and args.generation_mode != 'planned':
        parser.error('--generation_guidance_path requires planned mode')
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
