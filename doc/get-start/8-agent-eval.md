# 8 · 跑实验与读报告

本篇以实验 `ex-1`（数据集 `d1`，27 条用例，`agent/v1`）为例，说明跑批、评分、报告、人工检查和问题归因。数据集与评分规则见 [7 · 指标与数据集](https://tiltwind.github.io/refund-agent/doc/get-start/7-agent-dataset.md)。

| 产物 | 位置 | 作用 |
|---|---|---|
| 实验说明 | [`evals/experiments/ex-1/README.md`](https://github.com/tiltwind/refund-agent/blob/main/evals/experiments/ex-1/README.md) | 运行命令与指标口径 |
| 实验脚本 | [`evals/experiments/ex-1/run_experiment.py`](https://github.com/tiltwind/refund-agent/blob/main/evals/experiments/ex-1/run_experiment.py) | 跑批 + 打分 + 聚合 |
| 指标结果 | [`evals/experiments/ex-1/result.json`](https://github.com/tiltwind/refund-agent/blob/main/evals/experiments/ex-1/result.json) | run 级与逐条指标，报告的数据源 |
| 导出脚本 | [`evals/experiments/ex-1/export_result.py`](https://github.com/tiltwind/refund-agent/blob/main/evals/experiments/ex-1/export_result.py) | 从 Langfuse 补拉历史 run 的指标 |
| 机器报告 | `evals/experiments/ex-1/report.html` | 打分器算出来的结论 |
| 人工报告 | [`evals/experiments/ex-1/human-eval-report.md`](https://github.com/tiltwind/refund-agent/blob/main/evals/experiments/ex-1/human-eval-report.md) | trace 抽查结果 |
| 现场记录 | [`evals/experiments/ex-1/traces/`](https://github.com/tiltwind/refund-agent/blob/main/evals/experiments/ex-1/traces/) | 四条样本 trace，md + json |

---

## 一、跑批

### 1.1 前置

| 前置 | 命令 |
|---|---|
| Milvus 已启动并灌库 | `bash scripts/milvus.sh start` + `python rag/index/seed_milvus.py` |
| `.env` 配好模型与 Langfuse 密钥 | 缺 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` 会直接退出 |
| 数据集已推上去 | `python evals/push_dataset.py` |
| 用例自检过 | `python evals/validate_cases.py` |

实验通过本地 SDK 执行 LangGraph 和工具；Langfuse 负责保存 trace 与 dataset run。

### 1.2 命令

```bash
python evals/experiments/ex-1/run_experiment.py   # 全量 27 条
```

脚本只服务 ex-1 这一个实验，因此没有命令行参数，跑法写死在文件开头的常量里：

| 常量 | 默认 | 说明 |
|---|---|---|
| `DATASET` | `refund-cases-d1` | Langfuse 上的数据集名 |
| `AGENT_VERSION` | `v1` | 被测 agent 版本，做版本对比时只换这个 |
| `CASES` | `[]`（全量） | 填 case_id 只跑这几条；跑子集时不写 `result.json` |
| `CONCURRENCY` | 4 | 并发用例数，上限受模型限速与本地重排制约 |
| `VERBOSE` | `False` | 逐轮打印用户输入、工具链、答复、落库 |

run 名取 `<agent 版本>-<git short sha>`，版本对比时才对得上是哪次改动跑出来的分。

跑批时每条用例出两行：

```
  ▶ D1-011 金牌会员窗口优待：10 天无理由退货
  [ 1/27] ✓ D1-011 金牌会员窗口优待：10 天无理由退货   12.4s · 1 轮 · 4 次工具 · 落库 1 笔
```

这里的 ✓/✗ 表示执行成功或失败，不表示评分结果。

---

## 二、SDK 实验脚本

`client.run_experiment` 要三个钩子，`run_experiment.py` 就是这三个函数加一层进度打印（`task`
读的 agent 实例、并发数这些都是模块级常量，不再逐层往下传）：

| 钩子 | 职责 | 返回 |
|---|---|---|
| `task(item)` | 跑一条用例：构造 context、逐轮 invoke、把痕迹压成结构化记录 | `{"turns": [...], "error": None}` |
| `evaluate(output, expected_output)` | 单条用例判分 | `list[Evaluation]` |
| `aggregate(item_results)` | run 级聚合 | `list[Evaluation]` |

```python
result = client.run_experiment(
    name=DATASET,
    run_name=run_name,                                  # <agent 版本>-<git short sha>
    description=f"{META['agent_version']} / prompt {META['prompt_version']}",
    data=items,
    task=task,
    evaluators=[evaluate],
    run_evaluators=[aggregate],
    max_concurrency=CONCURRENCY,
    metadata={k: str(v) for k, v in META.items()},
)
client.flush()
```

跑批需要处理以下五点：

| 点 | 做法 |
|---|---|
| 数据隔离 | 每条用例 `eval_store.begin_session()` 起一份独立数据副本 |
| 多轮上下文 | `history = result["messages"]` 累积后进下一轮 |
| 痕迹采集 | 只取本轮新增的 messages 与新增流水，压成[判分要的四样东西](https://tiltwind.github.io/refund-agent/doc/get-start/7-agent-dataset.md#71-判分的四个输入) |
| 执行失败 | `try/except` 单独返回 `error`，判分时记 `run_error`，不静默判 0 |
| 重放用例 | 按 `run.repeat` 连跑，只取第一遍的 turns 判分，流水行数与单号单独判 |

`_score_turn(actual, expected)` 返回 `{指标: (分数, 说明)}`，判据见 [7 · 七](https://tiltwind.github.io/refund-agent/doc/get-start/7-agent-dataset.md#七评分规则实现)。说明写入 Langfuse comment，包括缺失工具和实际调用序列。

### 2.1 结果落盘

分数写回 Langfuse 的同时，跑批脚本把同一份指标写进 `result.json`。报告与版本对比只读 `result.json`，不依赖 Langfuse 处于运行状态。

| 层级 | 字段 |
|---|---|
| run | `run_name` / `run_url` / `agent`（版本与模型）/ `cases_total` / `cases_passed` / `elapsed_s` / `summary`（`p0_pass_rate` 等全部 run 级分数） |
| 用例 | `case_id` / `priority` / `trace_id` / `elapsed_s` / `tokens` / `case_pass` / `failed`（没过的硬指标）/ `scores`（每项的值与判分说明） |

结果文件丢失或更换机器时，用 `export_result.py` 从 Langfuse 补拉，输出为同一 schema（`case_row` 与 `write_result` 由两个入口共用）：

```bash
python evals/experiments/ex-1/export_result.py --run v1-6425ffd
python evals/experiments/ex-1/export_result.py --run v2-abc1234 --out /tmp/v2.json   # 两份 diff 即版本对比
```

补拉版比跑批版多 `tokens`，`elapsed_s` 换成 trace 实测延迟。

---

## 三、HTML 报告

在 `result.json` 之上补充跨用例对比与失败归因，生成可归档的自包含 HTML。

### 3.1 数据来源

run 元信息、九项指标、通过与否、耗时与 token 直接读 `result.json`；证据内容、工具入参与返回等 observation 级细节仍需回到 trace：

```
GET /api/public/traces/{id}     单条 trace 的全部 observation
```

检索链路的 `rag.rewrite` / `rag.route` / `rag.recall` / `rag.rerank` / `rag.assemble` 五个 span 也在这份 observation 里，报告的检索结论（零证据、候选被阈值挡掉、证据重复）直接引它们，不用猜。

报告中的数字必须来自这两个来源，结论需关联具体用例和 observation。HTML 使用内联 CSS 与 SVG，不依赖 CDN。

### 3.2 报告内容

| 章节 | 内容 | 用途 |
|---|---|---|
| 结论 | 门禁结果、失败指标、是否退化 | 快速判断结果 |
| run 概览 | 通过率 / 错误率 / 耗时 / token 六块数 | 判断这次结论可不可信 |
| run 级指标 | 九个 `avg_*` 条形图，硬软分开标 | 看趋势，与上一个 run 对齐 |
| 用例明细 | 每行一条用例 × 十列指标，✓ / ✗ / ◐ | 定位到具体用例 |
| 失败归因 | 逐条列期望链路 vs 实际链路，给处置建议 | 区分 Agent 缺陷与判分口径问题 |
| 软指标专章 | `citation_hit` 拆成「没检索」和「检索了没召回」 | 定位检索问题的环节 |
| 检索链路专章 | 零证据次数、证据块重复比例 | 九项指标看不见的检索问题 |
| 开销 | 模型调用次数、token、并行度 | 决定下次并发怎么设 |
| 下一步 | 每条问题指明「改 Agent」「改检索」还是「开 ex-2」 | 形成后续任务 |

### 3.3 ex-1 的结论

以下是最近一次 run `v1-6425ffd`（2026-08-19）的结果。

> **门禁不通过**：`p0_pass_rate = 0.857`，低于要求的 1.0。27 条过 24 条，3 条失败都是 P0，分属两类：D1-004 执行失败（检索返回零证据），D1-020 / D1-021 挂在 `tool_sequence`。

| 数 | 值 |
|---|---|
| `p0_pass_rate`（门禁） | 85.7%（18 / 21） |
| `overall_pass_rate` | 88.9%（24 / 27） |
| `error_rate` | 3.7%（1 条执行失败） |
| 墙钟耗时 | 8 分 21 秒（用例耗时合计 1457s，并发 4，并行度约 2.9） |
| 单条耗时 | 中位 52.1s · P90 69.7s · 最长 98.0s |
| token | 562k / 164 次模型调用 / 均 20.8k 一条 |

`avg_tool_sequence = 0.889`、软指标 `avg_citation_hit = 0.673`；其余六项硬指标为 0.963，差额全部来自 D1-004 的执行失败——它没有产出答复和流水，六项硬指标按执行失败记 0，与判定退化不是一回事。

与上一轮 `v1-0a0d3c4` 相比，agent 版本、提示词、数据集、判分逻辑都没动，动的是检索链路。两次 run 的门禁结论相同，但**挂掉的用例换了人**：

| 用例 | v1-0a0d3c4 | v1-6425ffd |
|---|---|---|
| D1-004 高风险账户 | 全绿 | 执行失败（检索零证据） |
| D1-024 冒充身份 | `tool_sequence` 失败（硬否决前没检索） | 通过 |
| D1-020 / D1-021 多轮 | `tool_sequence` 失败 | 同样失败，同一口径问题 |

D1-024 的翻转要单独记一笔：上一轮把它判为「唯一一条真实的 SOP 偏离」，并据此排了改提示词的 `agent/v2`。同一版提示词、同一条用例，这一轮模型自己检索了政策。那条结论建立在单次采样上，不成立——对应 [6.2 ⑥](#62-p1结论正确但系统退化) 的 `pass^k` 缺口。

---

## 四、人工检查

现有指标不检查检索证据的内容质量，实验后抽查 trace 补上这一层。

### 4.1 Trace 归档

抽样 trace 从 Langfuse 导出到仓库，一条 trace 两个文件：

| 文件 | 内容 | 给谁 |
|---|---|---|
| `.md` | 元信息、判分表，然后按轮列出用户消息、每次工具调用的入参与返回全文、模型答复 | 人读 |
| `.json` | trace 元信息 + 全部 observation，`input` / `output` 原样保留 | 重新统计、做 diff |

导出时将重复的完整系统提示词替换为 `«system prompt: agent/v1/prompt.py»`。

ex-1 保留四条：一条全指标通过、两条多轮评分失败、一条 SOP 偏离。这四条来自 run `v1-0a0d3c4`，是那一次运行的现场记录，不随后续 run 更新。

### 4.2 装配后的证据整段重复

`D1-001` 的检索返回中，同一段条文出现两次：

```text
| 会员等级 | 无理由退货窗口 |          ← 第 1 遍
| 普通会员 | 自签收之日起 7 天 |
| 金牌会员 | 自签收之日起 15 天 |
……
| 会员等级 | 无理由退货窗口 |          ← 第 2 遍，逐字相同
| 普通会员 | 自签收之日起 7 天 |
```

对两次 run 的全部检索返回复算后得到：

| 维度 | v1-0a0d3c4 | v1-6425ffd |
|---|---|---|
| 含重复正文的证据块 | 41 / 106（38.7%） | 46 / 107（43.0%） |
| 重复掉的正文 | 19,617 / 66,904 字 = **29.3%** | 21,863 / 71,472 字 = **30.6%** |
| 最大重复 | 3 遍 | 3 遍 |
| 主要文档 | P07 占 37 块 | P07 占 43 块（另 P04 两块、P11 一块） |

P07 的小节结构是一条下面挂 2～4 个子标题，正好踩中根因。根因位于 [`rag/retrieving/pipeline/assemble.py`](https://github.com/tiltwind/refund-agent/blob/main/rag/retrieving/pipeline/assemble.py)：装配阶段按父块回填，但分组键使用 `(parent_seq, parent_id, section_path)`。同一父块的子块具有不同 `section_path`，因此同一个 `parent_id` 会登记多次，`_render` 随后重复拼接父块正文。

影响：三成的证据正文重复并计入 token。`TOKEN_BUDGET = 3000` 是全部证据的上限，`v1-6425ffd` 中 `D1-009` 单次证据为 5,236 字（986 字是重复），存在后续证据被截断的风险；两次 run 都未观察到截断。

`citation_hit` 只检查期望条款是否召回，无法发现证据内部重复。两次 run 的数字接近，装配这段代码在两次之间没有改动。

---

## 五、实验问题与归因

| # | 问题 | 类别 | 证据 | 现有指标能发现吗 | 处置 |
|---|---|---|---|---|---|
| 1 | `D1-004` 检索返回零证据，工具抛 `NoEvidenceError`，整轮中断 | **检索链路** | `rag.rerank` span：`passed=0 / candidates=20 / min_score=0.3` | 能（`error_rate`） | 零证据降级为「空证据 + warn」交回模型；`MIN_SCORE` 到 `r1` 上校准 |
| 2 | 多轮用例第 2 轮被要求重复调第 1 轮已调过的工具 | **判分口径** | `D1-020` / `D1-021` | 能，但为假阴性 | `must_call` 改为按「累计到本轮」判断；在 ex-2 中验证 |
| 3 | `D1-024` 两轮结论相反：上一轮硬否决前没检索，这一轮检索了 | **稳定性** | 两次 run 的 `tool_sequence` | 不能（单次采样） | P0 子集连跑 k=3，报 `pass^k`，确认是偶发还是常态后再决定改不改提示词 |
| 4 | run metadata 记 `anthropic:claude-sonnet-5`，实跑全在 `deepseek-v4-flash` | **埋点** | 164 次 generation | 不能 | 让 `registry.meta()` 上报实际生效的供应商与模型 |
| 5 | 装配后的证据整段重复，占正文 30.6% | **检索链路** | 46 / 107 证据块 | **不能** | `_render` 按 `parent_id` 去重；装配后加一条「同块内不应出现完全相同段落」的断言 |
| 6 | `citation_hit = 0.673`，9 条检索了但没召回期望条款 | **检索质量** | 召回集被 P02 / P07 占满（107 块中占 83 块） | 只能看趋势 | 用 [`r1` 检索评测](https://tiltwind.github.io/refund-agent/doc/get-start/5-rag-eval.md)独立评估，与 Agent 回归分开跑 |

问题 1 与 5 都在检索链路，但暴露方式相反：

- `D1-004`：重排把 20 个候选全挡在 `MIN_SCORE = 0.3` 之外（被挡的里面有 `P02` 退货窗口和 `P09` 超窗口质量问题），装配拿到空证据，工具抛异常，这一轮没有答复也没有落库。单独重跑这条通过，不是稳定复现；`rag-ex-1` 的 96 条样本中有 5 条 `empty_context`（5.2%），与这次 27 条挂 1 条（3.7%）是同一件事，阈值本身没校准过。抛异常的处理方式放大了后果：一次检索没结果，整条用例的六项硬指标一起记 0。
- 证据重复则一次都没报警，九项指标结构上看不见它。

问题 2 和 3 都表现为 `tool_sequence` 失败，但归因和修改对象不同：

- `D1-020` / `D1-021`：第 1 轮已查询会员信息和政策，第 2 轮补充信息后直接复判并执行终局动作。决策、金额、落库和单号均正确，失败来自逐轮 `must_call` 口径。`D1-022` 第 1 轮未调用工具，第 2 轮执行完整链路，因此通过。两次 run 表现一致。
- `D1-024`：订单不属于认证身份，规则引擎返回「订单不存在」。上一轮模型跳过了政策检索，这一轮调了。同一版提示词跑出两种链路，说明这条不是稳定的 SOP 偏离，先测稳定性再谈改提示词。

---

## 六、后续规划

| 顺序 | 做什么 | 动哪一边 | 验收 |
|---|---|---|---|
| 1 | 零证据降级：`search_policy` 拿不到证据时返回空证据并打 warn，不抛异常打断整轮；`MIN_SCORE` 在 `r1` 上校准并按问题类型分设 | 检索链路 | 重跑 d1，`error_rate = 0`；`rag-ex-1` 的 `empty_context` 归零 |
| 2 | 修复 `assemble` 的父块重复并增加装配断言 | 检索链路 | 重跑 d1，确认重复正文与 token 减少；`citation_hit` 仅作观察项 |
| 3 | 修复 metadata 的实际模型上报 | 埋点 | 下一个 run 记录正确的模型信息 |
| 4 | P0 子集连跑 k=3，报 `pass^k` 与不稳定用例 | 判分 | 分清 `D1-024` 这类翻转是偶发还是常态，再决定要不要改提示词 |
| 5 | 修复多轮 `must_call` 口径及 ②④⑤⑥⑧，新建 `ex-2` | 判分 | `D1-020` / `D1-021` 通过；保留 ex-1 原始结果 |
| 6 | 入参断言、落库 `reason`、handoff 维度 → `d2` + `ex-2` | 期望值 + 判分 | `D1-026` 这类「过程错但结论对」能被判负 |
| 7 | 回流线上失败样本 | 数据集 | 保留原始表述并脱敏，不改写为标准问法 |

检索评测集 `r1` 与实验 `rag-ex-1` 已经建好（[4](https://tiltwind.github.io/refund-agent/doc/get-start/4-rag-dataset.md) + [5](https://tiltwind.github.io/refund-agent/doc/get-start/5-rag-eval.md)），检索质量从 Agent 回归里摘出去了：96 条样本上 `context_precision = 0.90`、`context_recall = 0.927`、`hit@1 = 0.552`，另有 5 条 `empty_context`——就是 `D1-004` 那个失败模式的底数。

先完成 1、2（都在检索链路，且 1 是这一轮唯一的真实故障），再执行 4、5（改判分）。改 Agent 的事排在稳定性测完之后，不与判分口径改动合并到同一轮实验。

后续补充版本对比自动化（`evals/compare.py`，同集运行 v1/v2 并逐条 diff）和线上监控（复用不依赖标注的三个指标，见 [7 · 7.4](https://tiltwind.github.io/refund-agent/doc/get-start/7-agent-dataset.md#74-从轮到用例从用例到-run)）。设计见 [2 · 设计 · 6.2 / 6.4](https://tiltwind.github.io/refund-agent/doc/get-start/2-design.md#六持续评估闭环)。
