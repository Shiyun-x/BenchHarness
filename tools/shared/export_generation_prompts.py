"""Export current generator prompts offline for review; never load API credentials."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from benchHarness.executor import generation_prompts as prompts
from benchHarness.executor.generation_quality import (build_generation_prompt, MEDIA_QUALITY,
                                                        IMAGE_QUALITY, AUDIO_QUALITY)
from benchHarness.executor.generation_prompts import MEDIA_CONTENT
from benchHarness.executor.generation_payloads import PAYLOAD_POLICY_VERSION, prepare_generation_payload
from benchHarness.executor.generation_responses import GENERATION_RESPONSE_POLICY_VERSION
from benchHarness.executor.generation_research import (RESEARCH_POLICY_VERSION,
    NATIVE_EVIDENCE_POLICY_VERSION, DEFAULT_SEARCH_BUDGET)
from benchHarness.executor.generation_sources import SOURCE_VIEW_GUIDE, SOURCE_VIEW_POLICY_VERSION
from benchHarness.executor.generation_inspection import INSPECTION_PROTOCOL_VERSION
from benchHarness.executor.planned_generation import GENERATION_FLOW_VERSION, CHECKPOINT_VERSION

ROOT = Path(__file__).resolve().parents[2]
SUFFIX = ('\n\n[IMPORTANT] Return ONLY valid JSON (a single top-level JSON object). '
          'Use double quotes for all keys/strings. No trailing commas. [IMPORTANT]')
STAGES = [
    ('proposal', '提案 agent：发散方向与按需原生检索', 'proposal', prompts.PROPOSAL_PROMPT, None,
     'original、benchmark_goal（存在时）和四字段 task_contract；有 dataset 建议时接收 dataset_guidance（strategy 与可选工具用途），可调整或替换。不传 recipe_hint 或 research_state。先从样本和目标形成粗略想法，再定向发散为 2–3 个方向，按需搜索补充；允许不同成熟度。'),
    ('design', 'Design：从灵感收敛为具体改造与难度方案', 'design', prompts.DESIGN_PROMPT, None,
     '与提案相同的四字段 task、benchmark_goal、original 和实际存在的 dataset_guidance，加上全部 proposals、来源支持的 evidence 和 grounding_status。综合材料与灵感找到落脚点，不调用搜索；制定 sample 特定方案，输出 goal/source_refs/strategy。'),
    ('construction', 'Construct：执行方案并交付完整初稿', 'construct', prompts.CONSTRUCT_PROMPT, None,
     '四字段 task、benchmark_goal、原文、定稿 design 和选中证据包。以执行为主，实现计划中的关键关系与挑战；按目标区分应给条件与待测操作，将方案转成自然成品，完成格式、具体问题、选项及答案。'),
    ('initial_inspection', '初审：基于成品找问题并制定具体改法', 'inspect', prompts.INSPECT_PROMPT, 'refinement_strategy',
     '实际 candidate、support_refs、精简 design、任务与事实。找不足和错误，给出原因、可直接执行的解决方案和效果检查；需要修正旧设计也在同一 changes 中说明，交 repair 执行。'),
    ('repair', 'Repair：按需执行一次实质改进', 'repair', prompts.REPAIR_PROMPT, None,
     '初稿、精简 design、初审 changes、事实与硬约束。直接落实整份改进策略，包括初审说明的设计调整，返回完整改进样本并逐项自检。'),
    ('final_inspection', '终审：一次定稿', 'inspect', prompts.INSPECT_PROMPT, 'final_validation',
     '最终候选、精简 design、support_refs 与 previous_changes。检查实际质量及改进结果；接受、有限小修或拒绝，不发起新设计或大改。'),
    ('replan', 'Replan：构造受阻的前置恢复', 'replan', prompts.REPLAN_PROMPT, None,
     '仅用于初稿尚未形成、construct 报告具体阻塞时的原有有界恢复。初审之后不调用此阶段，质量改进直接由 repair 完成。'),
]


def export(contract, contract_source):
    now = datetime.now(timezone.utc)
    name = now.isoformat(timespec='milliseconds').replace('+00:00', 'Z').replace(':', '-').replace('.', '-')
    folder = ROOT/'docs/prompt_review'/name
    folder.mkdir(parents=True)
    files = [
        'benchHarness/executor/' + n + '.py' for n in (
            'generation_research', 'generation_prompts', 'generation_quality', 'generation_payloads',
            'generation_responses', 'generation_sources', 'generation_inspection',
            'generation_contracts', 'planned_generation')]
    files += ['utils/llm_caller.py', 'utils/native_responses.py', 'utils/resources/tools.yaml']
    manifest = {'generated_at': now.isoformat(), 'kind': 'current_code_prompt_export', 'live_model_calls': 0,
                'prompt_version': prompts.PROMPT_VERSION, 'prompt_baseline': prompts.PROMPT_BASELINE,
                'flow': GENERATION_FLOW_VERSION, 'payload': PAYLOAD_POLICY_VERSION,
                'checkpoint_version': CHECKPOINT_VERSION,
                'response': GENERATION_RESPONSE_POLICY_VERSION, 'research': RESEARCH_POLICY_VERSION,
                'inspection_protocol': INSPECTION_PROTOCOL_VERSION,
                'default_search_budget': DEFAULT_SEARCH_BUDGET, 'source_view': SOURCE_VIEW_POLICY_VERSION,
                'evidence': NATIVE_EVIDENCE_POLICY_VERSION, 'contract_source': contract_source,
                'contract': contract, 'stages': {}, 'source_files': [
                    {'path': p, 'sha256': hashlib.sha256((ROOT/p).read_bytes()).hexdigest()} for p in files]}
    sections, rows = [], []
    for index, (key, title, stage, base, mode, inputs) in enumerate(STAGES, 1):
        system = build_generation_prompt(stage, base, contract, mode,
                                         search_budget=DEFAULT_SEARCH_BUDGET if stage == 'proposal' else None)
        first = system + SUFFIX
        (folder/(key+'_system.txt')).write_text(first+'\n', encoding='utf-8')
        manifest['stages'][key] = {'stage': stage, 'review_mode': mode,
            'built_system_chars': len(system), 'first_attempt_system_chars': len(first),
            'caller_suffix_chars': len(SUFFIX), 'max_tokens': 4000}
        rows.append(f'| {key} | {len(system)} | {len(first)} | 4000 |')
        sections.append(f'<a id="stage-{key}"></a>\n\n## {index}. {title}\n\n{inputs}\n\n```text\n{first}\n```')
    (folder/'proposal_tools.json').write_text(json.dumps([
        {'type': 'web_search', 'search_context_size': 'low'}], ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (folder/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    header = f'''# 当前 Generator 完整 Prompt 审查稿

[Prompt 与接口](../GeneratorPrompt与接口.md) · [流程说明](../Generator流程说明.md) · [文档入口](../README.md#generator)

更新：{now:%Y-%m-%d}。由当前代码离线导出，没有发起模型或网络请求。

Prompt 版本：**{prompts.PROMPT_VERSION}**，基线 **{prompts.PROMPT_BASELINE}**。当前变化及阶段边界见 [Prompt 与接口](../GeneratorPrompt与接口.md)，保留 [v1.4 指导优先级](指导优先级与设计参考_v1.4.md)、[v1.2 事实依据](样本改造职责与事实依据_v1.2.md)、[v1.1 干扰项分工](干扰项设计与初审强化_v1.1.md)；[冻结的 v1](Generator-v1版本基线.md) 留作对照。运行协议：{GENERATION_FLOW_VERSION} / {PAYLOAD_POLICY_VERSION} / {GENERATION_RESPONSE_POLICY_VERSION}。

## 当前调用结构

提案 agent 先查看具体 sample 及原始媒体，结合正确 QA 和 evaluation target 形成粗略想法，再发散为 2–3 个方向并按需原生检索补充。独立 design agent 同样查看原始媒体，吸收灵感并制定一份完整方案，再进入 construct → 初审 → 可选一次改进 → 终审。没有外部下载、换图或 PDF 取图工具；各阶段继续接收原始图片。

task 只负责说明要测什么能力，不需要预先规定如何增加难度。提案、design 和 construct 的 task 输入只保留实际存在的 name、description、sample_schema、answer_type；benchmark 整体目标单独放在 benchmark_goal，原始 sample 保留原有字段。不存在的字段不占位，不传旧 recipe_hint。提案和 design 接收实际存在的 dataset_guidance，仅含简短 strategy 与可选工具用途，可调整或替换；Construct 落实样本 design，inspection 不以遵循 dataset 建议为验收条件。提案不传 research_state；搜索预算由程序约束并在 system 中说明。内部契约保留给程序检查及现有后续审查。

沿用 v1.2 的样本改造职责与事实依据：结合内容、正确 QA 和标注理解可信监督，不能把输入中待判断的描述都当真。内部区分创作内容与准备当作真实事实引入的信息，后者需要依据；材料和 QA 怎样呈现取决于任务，gold 根据成品核验。不默认加 fictional 等构造说明。搜索、媒体和干扰项规则保持。

design 负责整个 sample 的改造方案：利用原样本哪些内容、材料与 QA 怎样协同变化、目标能力如何成为解题所必需，以及由此形成什么难度。选择题简短规划关键区别、可信混淆方向与排除依据。问题考查方向和答案成立的依据在方案层面说明，不预先写死完整问题、选项和答案标签，也不先选一道题再为它拼材料。方案需具体到能指导实现，不能只给提高复杂度的口号。选择强方向即可，不为使用所有提案而拼接；构造机制随能力和样本而定，不强加同一事件、多视角或固定跳数。仍使用 goal/source_refs/strategy，不增加字段或固定小标题。

construct 以执行为主，按 design 改造原始样本，完成材料、具体问题、选项及最终答案。自行处理自然表达、组织和实现细节，不重新选择路线。将设计关系落实到成品结构，按 evaluation target 区分已给条件与待测操作，不必把内部分析反复写进题面和每个选项。保留必要前提、选项区别、限定和有用的复杂度；任务需要时可以提供待解释的已知结果或完整解释选项。不能为缩短而削弱挑战。初审针对成品核对这层边界，具体修改仍交可选一次 Repair。缺少必要前提或媒体时报告具体阻塞。

初审和修改保留精简 design，了解起始目标与证据依赖，但将其视作参考而非不可变计划。初审已经看到具体成品，应找出不足和错误、说明原因，并给出能直接执行的具体解决方案，包括可读性。选择题逐项检查可信混淆、目标知识或推理的排除依据与表面捷径，指出具体强化办法及检查条件，不能只说提高难度。Repair 改善措辞和组织，使题目清楚、自然、容易理解，同时保持原意、必要信息和测试要求；具体改法由当前任务与样本决定。设计本身不足时也在 changes 中给出调整。全部 changes 交 repair 一次执行，不另开规划，不需要 scope 标记。

修改 agent 以 design 和初稿为起点，按初审策略直接改善成品；初审有依据的调整可以替代旧方案，但仍须符合 evaluation target、事实和硬约束。终审对最终候选及 previous_changes 作一次发布决定，设计仅作起始参考；不能仅因按意见改过就放行，也不把所有可优化之处都判作必须大改。小修仍由程序受限应用；不再开启新一轮质量改造。

正常路径为 5–6 个 Agent 阶段。原生搜索预算默认 3 次（包括打开页面），0 关闭 web_search。当前上游不支持 max_tool_calls，请求不发送该参数；预算依靠提示与返回轨迹校验，不是服务端硬限制。每个阶段最多 2 次尝试，重试可能再次产生费用。

初审鼓励提出实质改进；样本已合格且无值得修改之处时，可返回 accept、changes=[] 并说明依据，跳过 Repair，保持原候选进入终审。有修改建议时最多执行一次 Repair，再终审。construct 成品之后为初审 → 可选一次 repair → 终审，初审发现设计问题也不增加模型阶段。原有的初稿构造受阻恢复、格式恢复和传输重试另计。

规划阶段还会在实际请求中按当前 original 添加最多 4 条真实引用路径；以下静态角色导出不含这些样本专属路径示例。数字数组索引支持 .0 与 [0] 等价归一化，实际字段仍严格校验。

格式要求保留并明确：各阶段只返回一个完整 JSON 对象，字段层级和类型遵循当前模板，字符串中的引号与换行正确转义。Construct / Repair 负责最终样本格式；没有独立 options 字段时，选项在 question 中逐行以 A. / B. 等起行，说明在首项之前，末项之后不附加说明。Proposal / Design 仍遵守自己的 JSON 接口，不承担完整成品格式审查。

样本文本统一要求清楚、自然、精练，覆盖 context、问题、选项和答案；删除不服务于测试的冗余措辞、重复和说明，保留原意、必要信息与难度。长度由任务需要决定。Construct、Repair 与初审共用此规则；初审给出具体改法，Repair 根据成品落实。多个选项共享的真事实不自动是可公开的前提；待测知识、身份线索和中间关系不能为了精简被搬进题面。保持可选一次修改。

选项顺序只在初审前由程序随机化一次，Repair 后不再程序重排。初审按选项含义给建议，字母只定位当时草稿；终审从当前 candidate.output.answer 读取待核验标签，对照当前选项独立判断。previous_changes 的旧字母不代表当前标签。checkpoint 版本为 {CHECKPOINT_VERSION}，修复前缓存不会被当作新结果复用。

## 原生搜索参数与证据边界

提案使用 litellm.responses，保持配置中的模型和成对 API 凭据。tools 按预算仅提供原生 web_search，见 [proposal_tools.json](../prompt_review/{name}/proposal_tools.json)。网页搜索使用 search_context_size=low、include=["web_search_call.action.sources"]；tool_choice=auto。没有宿主下载或图片替换工具。搜索上下文 low 不保证固定 token 数。

只解析最终答案 JSON，保留原始响应及搜索轨迹。提案 evidence 的事实关联实际完成的原生检索来源，包括搜索、成功页面打开及提供方引用；生成的网址或未完成检索不能提供支持。来源关联不自动证明语义或作品身份，Design 与终审仍须检查前提、范围和答案。

原始内容与正确 QA、标注共同提供可信监督；普通可靠知识可以使用。新增真实知识按需检索，不重新查证可信 QA。研究阶段不返回旧 action=search，原生搜索由服务端执行。显式传入自定义 search 回调的兼容接口保留旧研究适配器，不是默认生成路径。

## 内部事实依据与评测作答条件

内部 evidence、source_refs 和 support 用于核验事实及 gold，不是被测模型额外可见的输入，引用和核验说明不必展示。依据最终输入/媒体和任务合理要求的知识判断可作答性；题面提供必要个案前提，把待测知识和推断留给被测模型，缺少待测知识本身不是缺陷。明确仅依据所给材料的任务仍遵守该限制。

Design 规划材料与 QA 的协同变化和难度；Construct 按计划执行，纠正写作引入的泄露、简化或错误。初审从实际解题路径诊断，给出具体协调修改，Repair 一次落实并重算答案。终审核验成品，不仅依据完成修改或作者自述放行，也不因单个选项稍弱就自动拒绝整条样本。具体措辞以本次导出的英文正文为准。

## 长度与组装

以下是代表任务的完整首次 system 文本；不重复展开长原文。字符数不是 token 数，不含动态样本、证据与图片。提案不再叠加共享构造/检查规则、详细选项或媒体执行格式，后续阶段按真实 schema 加入对应要求。

| 阶段 | 组装字符 | 首次请求字符 | 输出 token 上限 |
| --- | ---: | ---: | ---: |
''' + '\n'.join(rows) + '\n\n## 三个生成 agent 的代表性任务输入（不重复展开 original）\n\n```json\n' + json.dumps(prepare_generation_payload('proposal', {'task_contract': contract}), ensure_ascii=False, indent=2) + '\n```\n\n'
    footer = '\n\n## 条件规则与实际请求边界\n\n图片字段使用 str|list，可传一个路径或非空路径列表；旧 Planner 图片字段在 Generator 内部兼容，具体图片数量由样本设计决定。详见 [图片接口](图片字段单图与多图兼容.md)。\n\n默认路径按当前子任务的模态与 schema 加入图像/音频质量规则。音频工具说明统一来自 tools.yaml：Proposal/Research/Design/Replan 通过 available_media_capabilities 只接收 name、summary；Construct 通过 available_media_tool_specs 接收 io、behavior、params 和 typical_uses 等执行约定；Inspect/Repair 看已有脚本和 media_jobs，不接收工具目录。没有音频模态或音频字段时，不传音频工具说明。TTS 在内容验收后批量执行。下列为各条件规则的参考原文，不表示同一任务会同时收到全部规则。\n\n'
    footer += '\n\n'.join('```text\n'+p+'\n```' for p in
                          (MEDIA_QUALITY, IMAGE_QUALITY, AUDIO_QUALITY, MEDIA_CONTENT, SOURCE_VIEW_GUIDE))
    footer += f'''\n\n超预算且允许时才选择原句，并记录 source_view；必须全文保留时不得截短。构造与小修字段由真实 schema 生成。

## 审查重点与来源

检查每个方向是否基于样本独有细节、是否解释 evaluation target 与新增难度；design 是否做出完整证据安排而不是换词或拼题；最终可见输入是否需要目标能力才能解答。离线协议检查不证明实际难度或接受率已提高。

完整逐阶段导出和源码摘要见 [manifest](../prompt_review/{name}/manifest.json)。

可运行 `python -m tools.shared.export_generation_prompts` 重新导出。实际线上请求、响应和 usage 以 generation 缓存为准。
'''
    review_path = ROOT/'docs/generator/完整Prompt审查稿.md'
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(header+'\n\n'.join(sections)+footer, encoding='utf-8')
    readme = f'''# 当前 Prompt 审查导出

[返回文档入口](../README.md) · [Prompt 与接口](../GeneratorPrompt与接口.md) · [完整英文审查稿](../generator/完整Prompt审查稿.md) · [全文件索引](../全文件索引.md)

当前导出目录：[{name}]({name}/)，Prompt **{prompts.PROMPT_VERSION}**，基线 **{prompts.PROMPT_BASELINE}**；运行协议 {GENERATION_FLOW_VERSION} / {PAYLOAD_POLICY_VERSION}。包含 proposal、design、construction、initial_inspection、repair、final_inspection、replan 七类系统文本与 manifest。

当前版本的改动见 [Prompt 与接口](../GeneratorPrompt与接口.md)：Construct 将设计关系落实到样本结构，按目标区分已给条件与待测内容；初审核对这层边界。保留必要前提、复杂度和任务需要的解释型选项，不设长度上限，不新增阶段。沿用 [v1.4 指导优先级](../generator/指导优先级与设计参考_v1.4.md) 和 [v1.2 事实依据](../generator/样本改造职责与事实依据_v1.2.md)。全部 JSON 模板、流程和工具权限保持；原生搜索 schema 单列于 proposal_tools.json。

后续分工：初审诊断成品并提出策略，有实质改进点时 repair 落实一次修改；无需修改可直接终审，终审一次定稿。设计不足也由初审写进策略、repair 一次落实，不另开重规划或重新构造分流。精简设计作为起始参考，不作为事实或难度证明。

这是当前代码的离线导出，包含首次请求的 JSON 后缀；没有发起在线请求。此前目录是历史对照，不代表当前规则。重新导出命令为 `python -m tools.shared.export_generation_prompts`，随后同步全文件索引。
'''
    (ROOT/'docs/prompt_review/README.md').write_text(readme, encoding='utf-8')
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, default=ROOT/'tests/fixtures/long_climate_sample.json',
                        help='A task contract JSON, or fixture containing a contract key')
    args = parser.parse_args()
    value = json.loads(args.contract.read_text(encoding='utf-8'))
    folder = export(value.get('contract', value), str(args.contract))
    print(folder)


if __name__ == '__main__':
    main()
