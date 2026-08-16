# 5 · 跑实验与读报告

本篇以实验 `ex-1`（数据集 `d1`，27 条用例，`agent/v1`）为例，说明跑批、评分、报告、人工检查和问题归因。数据集与评分规则见 [4 · 数据集与指标](https://tiltwind.github.io/refund-agent/doc/get-start/4-dataset.md)。

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

## 一、跑

### 1.1 前置

| 前置 | 命令 |
|---|---|
| Milvus 已启动并灌库 | `bash scripts/milvus.sh start` + `python knowledge/seed_milvus.py` |
| `.env` 配好模型与 Langfuse 密钥 | 缺 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` 会直接退出 |
| 数据集已推上去 | `python evals/push_dataset.py` |
| 用例自检过 | `python evals/validate_cases.py` |

实验通过本地 SDK 执行 LangGraph 和工具；Langfuse 负责保存 trace 与 dataset run。

### 1.2 命令

```bash
python evals/experiments/ex-1/run_experiment.py                       # 全量 27 条
python evals/experiments/ex-1/run_experiment.py --cases D1-011 D1-027 # 只跑指定用例
python evals/experiments/ex-1/run_experiment.py -v --concurrency 1    # 逐轮打印工具链和答复
python evals/experiments/ex-1/run_experiment.py --run-name v1-$(git rev-parse --short HEAD)
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dataset` | `refund-cases-d1` | Langfuse 上的数据集名 |
| `--agent` | `v1` | 被测 agent 版本，做版本对比时只换这个 |
| `--run-name` | `<agent>-<条数>cases` | 版本对比时传 git sha，UI 上才分得清哪次是哪次 |
| `--cases` | 全量 | 只跑指定 case_id，调用例时用 |
| `--concurrency` | 4 | 并发用例数；调高会撞模型限速，也会拖慢本地重排 |
| `--out` | `result.json` | 指标落盘路径；只跑 `--cases` 子集时默认不写，免得覆盖全量结果 |
| `-v` | 关 | 逐轮打印用户输入、工具链、答复、落库 |

跑批时每条用例出两行：

```
  ▶ D1-011 金牌会员窗口优待：10 天无理由退货
  [ 1/27] ✓ D1-011 金牌会员窗口优待：10 天无理由退货   12.4s · 1 轮 · 4 次工具 · 落库 1 笔
```

这里的 ✓/✗ 表示执行成功或失败，不表示评分结果。

---

## 二、SDK 实验脚本

`client.run_experiment` 要三个钩子，`run_experiment.py` 就是这三个函数加一层进度打印：

| 钩子 | 职责 | 返回 |
|---|---|---|
| `task(item)` | 跑一条用例：构造 context、逐轮 invoke、把痕迹压成结构化记录 | `{"turns": [...], "error": None}` |
| `evaluate(output, expected_output)` | 单条用例判分 | `list[Evaluation]` |
| `aggregate(item_results)` | run 级聚合 | `list[Evaluation]` |

```python
result = client.run_experiment(
    name=args.dataset,
    run_name=run_name,                                  # 版本对比时传 git sha
    description=f"{meta['agent_version']} / prompt {meta['prompt_version']}",
    data=items,
    task=make_task(args.agent, progress),
    evaluators=[evaluate],
    run_evaluators=[aggregate],
    max_concurrency=args.concurrency,
    metadata={k: str(v) for k, v in meta.items()},
)
client.flush()                                          # 短命脚本不 flush 就什么都没上去
```

跑批需要处理以下五点：

| 点 | 做法 | 不这样会怎样 |
|---|---|---|
| 数据隔离 | 每条用例 `eval_store.begin_session()` 起一份独立数据副本 | `execute_refund` 会把 `order["refunded"]` 置 True，并发跑批互相污染 |
| 多轮上下文 | `history = result["messages"]` 累积后进下一轮 | 第二轮接不上第一轮，多轮用例全成了单轮 |
| 痕迹采集 | 只取本轮新增的 messages 与新增流水，压成[判分要的四样东西](https://tiltwind.github.io/refund-agent/doc/get-start/4-dataset.md#51-判分只看这四样东西) | 拿全量会把上一轮的工具调用算进这一轮 |
| 执行失败 | `try/except` 单独返回 `error`，判分时记 `run_error` 而非静默判 0 | 环境故障被读成 Agent 退化 |
| 重放用例 | 按 `run.repeat` 连跑，只取第一遍的 turns 判分，流水行数与单号单独判 | 第二遍的痕迹混进来，落库断言必然挂 |

`_score_turn(actual, expected)` 返回 `{指标: (分数, 说明)}`，判据见 [4 · 五](https://tiltwind.github.io/refund-agent/doc/get-start/4-dataset.md#五评分规则)。说明写入 Langfuse comment，包括缺失工具和实际调用序列。

### 2.1 结果落盘

分数写回 Langfuse 的同时，跑批脚本把同一份指标写进 `result.json`。Langfuse 是本地实例，换台机器 run 页就打不开，报告和版本对比不应依赖它处于运行状态。

| 层级 | 字段 |
|---|---|
| run | `run_name` / `run_url` / `agent`（版本与模型）/ `cases_total` / `cases_passed` / `elapsed_s` / `summary`（`p0_pass_rate` 等全部 run 级分数） |
| 用例 | `case_id` / `priority` / `trace_id` / `elapsed_s` / `tokens` / `case_pass` / `failed`（没过的硬指标）/ `scores`（每项的值与判分说明） |

结果文件丢失或更换机器时，用 `export_result.py` 从 Langfuse 补拉，输出为同一 schema（`case_row` 与 `write_result` 由两个入口共用）：

```bash
python evals/experiments/ex-1/export_result.py --run v1-0a0d3c4
python evals/experiments/ex-1/export_result.py --run v2-abc1234 --out /tmp/v2.json   # 两份 diff 即版本对比
```

补拉版比跑批版多 `tokens`，`elapsed_s` 也换成 trace 实测延迟——这两个数只有 Langfuse 算得出。

---

## 三、生成 HTML 报告

在 `result.json` 之上补充跨用例对比与失败归因，生成可归档的自包含 HTML。

### 3.1 数据来源

run 元信息、九项指标、通过与否、耗时与 token 直接读 `result.json`；证据内容、工具入参与返回等 observation 级细节仍需回到 trace：

```
GET /api/public/traces/{id}     单条 trace 的全部 observation
```

报告中的数字必须来自这两个来源，结论需关联具体用例和 observation。HTML 使用内联 CSS 与 SVG，不依赖 CDN。

### 3.2 报告内容

| 章节 | 内容 | 用途 |
|---|---|---|
| 结论 | 门禁结果、失败指标、是否退化 | 快速判断结果 |
| run 概览 | 通过率 / 错误率 / 耗时 / token 六块数 | 判断这次结论可不可信 |
| run 级指标 | 九个 `avg_*` 条形图，硬软分开标 | 看趋势，与上一个 run 对齐 |
| 用例明细 | 每行一条用例 × 十列指标，✓ / ✗ / ◐ | 定位到具体用例 |
| 失败归因 | 逐条列期望链路 vs 实际链路，给处置建议 | 区分 Agent 缺陷与判分口径问题 |
| 软指标专章 | `citation_hit` 拆成「没检索」和「检索了没召回」 | 混在一个数里读不出问题在哪 |
| 开销 | 模型调用次数、token、并行度 | 决定下次并发怎么设 |
| 下一步 | 每条问题指明「改 Agent」还是「开 ex-2」 | 形成后续任务 |

### 3.3 ex-1 的结论

> **门禁不通过**：`p0_pass_rate = 0.857`，低于要求的 1.0。27 条过 24 条，3 条失败全部挂在 `tool_sequence` 一个指标上，且都是 P0。

| 数 | 值 |
|---|---|
| `p0_pass_rate`（门禁） | 85.7%（18 / 21） |
| `overall_pass_rate` | 88.9%（24 / 27） |
| `error_rate` | 0%（无环境故障干扰） |
| 墙钟耗时 | 7 分 26 秒（用例耗时合计 1416s，并发 4，并行度约 3.2） |
| 单条耗时 | 中位 53.1s · P90 70.5s · 最长 88.2s |
| token | 565k / 165 次模型调用 / 均 20.9k 一条 |

`decision_match`、`rule_consistency`、`log_match`、`receipt_in_answer`、`no_leak`、`idempotent_replay` 均为满分。失分集中在 `avg_tool_sequence = 0.889` 与软指标 `avg_citation_hit = 0.704`。

---

## 四、人工检查

现有指标不检查检索证据的内容质量，因此实验后需要抽查 trace。

### 4.1 Trace 归档

Langfuse 为本地实例，因此将抽样 trace 导出到仓库：

| 文件 | 内容 | 给谁 |
|---|---|---|
| `.md` | 元信息、判分表，然后按轮列出用户消息、每次工具调用的入参与返回全文、模型答复 | 人读 |
| `.json` | trace 元信息 + 全部 observation，`input` / `output` 原样保留 | 重新统计、做 diff |

导出时将重复的完整系统提示词替换为 `«system prompt: agent/v1/prompt.py»`。

ex-1 保留四条：一条全指标通过、两条多轮评分失败、一条 SOP 偏离。

### 4.2 发现：装配后的证据整段重复

`D1-001` 的检索返回中，同一段条文出现两次：

```text
| 会员等级 | 无理由退货窗口 |          ← 第 1 遍
| 普通会员 | 自签收之日起 7 天 |
| 金牌会员 | 自签收之日起 15 天 |
……
| 会员等级 | 无理由退货窗口 |          ← 第 2 遍，逐字相同
| 普通会员 | 自签收之日起 7 天 |
```

对全部 run 数据复算后得到：

| 维度 | 数 |
|---|---|
| 含重复正文的证据块 | 41 / 106（38.7%），分布在 24 次检索 / 23 条用例 |
| 重复掉的正文 | 19,569 字 / 66,746 字 = **29.3%** |
| 最大重复 | 重复 3 遍；单次证据最大 4,192 字，其中 1,702 字是重复 |
| 主要文档 | P07 占 37 块；其一个父块包含 2～4 个子标题 |

根因位于 [`services/rag/pipeline/assemble.py`](https://github.com/tiltwind/refund-agent/blob/main/services/rag/pipeline/assemble.py)：装配阶段按父块回填，但分组键使用 `(parent_seq, parent_id, section_path)`。同一父块的子块具有不同 `section_path`，因此同一个 `parent_id` 会登记多次，`_render` 随后重复拼接父块正文。

影响：29.3% 的证据正文重复并计入 token。`TOKEN_BUDGET = 3000` 是全部证据的上限，`D1-017` 单次证据为 4,192 字，存在后续证据被截断的风险；本次 run 未观察到截断。

`citation_hit` 只检查期望条款是否召回，无法发现证据内部重复。

---

## 五、实验问题与归因

| # | 问题 | 类别 | 证据 | 现有指标能发现吗 | 处置 |
|---|---|---|---|---|---|
| 1 | `D1-024` 拒绝前没调 `search_refund_policy` | **Agent** | `traces/D1-024-274ee602` | 能（`tool_sequence`） | 提示词补「硬否决同样要检索政策依据后再答复」，开 `agent/v2` 重跑对比 |
| 2 | 多轮用例第 2 轮被要求重复调第 1 轮已调过的工具 | **判分口径** | `D1-020` / `D1-021` | 能，但为假阴性 | `must_call` 改为按「累计到本轮」判断；在 ex-2 中验证 |
| 3 | run metadata 记 `anthropic:claude-sonnet-5`，实跑全在 `deepseek-v4-flash` | **埋点** | 165 次 generation | 不能 | 让 `registry.meta()` 上报实际生效的供应商与模型 |
| 4 | 装配后的证据整段重复，占正文 29.3% | **检索链路** | 41 / 106 证据块 | **不能** | `_render` 按 `parent_id` 去重；装配后加一条「同块内不应出现完全相同段落」的断言 |
| 5 | `citation_hit = 0.704`，7 条检索了但没召回期望条款 | **检索质量** | 召回集被 P02 / P07 占满 | 只能看趋势 | 另建 query→section retrieval 数据集，与 Agent 回归分开评估 |

问题 1 和 2 均表现为 `tool_sequence` 失败，但归因和修改对象不同：

- `D1-020` / `D1-021`：第 1 轮已查询会员信息和政策，第 2 轮补充信息后直接复判并执行终局动作。决策、金额、落库和单号均正确，失败来自逐轮 `must_call` 口径。`D1-022` 第 1 轮未调用工具，第 2 轮执行完整链路，因此通过。
- `D1-024`：订单不属于认证身份，规则引擎返回「订单不存在」，模型未检索政策。身份边界、答复和拒绝落库均正确，但拒绝答复缺少政策依据，因此应修改 Agent，不放宽用例。

---

## 六、指标缺口

对照 [4 · 4.3 的验收口径映射表](https://tiltwind.github.io/refund-agent/doc/get-start/4-dataset.md#43-验收口径--指标覆盖是不均匀的)与打分器实现，缺口分为 P0 和 P1。

### 6.1 P0：漏判真实缺陷

**① 未校验工具入参。** `_score_turn` 只读取 `names = [call["name"] for call in tools]`，未使用 `args`。例如 `D1-026` 中，即使模型把 `reason_type` 从「无理由」改为「质量问题」，只要拒绝结论不变，现有指标仍会通过。
增加 `tools.args` 断言，至少覆盖 `check_refund_eligibility` 的两个枚举参数。

**② 未统计工具调用错误。** `check_refund_eligibility` 遇到非法枚举会返回 `参数错误：…`，`_last_verdict` 会跳过这类返回。多次纠错通常表示工具描述或 schema 存在问题。
统计每轮工具错误次数，记为软指标 `tool_call_validity`。该指标可从现有记录计算，无需修改期望值。

**③ 未校验落库的 `reason`。** 流水包含七个字段，`log_match` 只比较 `decision`、`order_id` 和 `amount`。`reason` 由模型填写，但当前不参与评分。
增加断言，要求 `reason` 包含规则引擎返回的关键片段。

**④ 未检查答复中的额外数字。** `mention_hit` 只检查必需内容，未检查金额、天数和窗口是否来自工具返回。
提取答复中的金额与天数，与本轮工具返回的数字集合比较。无需调用模型。

### 6.2 P1：决策对了，但系统在退化

**⑤ 成本与延迟不参与门禁。** 当前报告记录 565k token 和 P90 70.5s，但未设置阈值。
run 级增加 `p50/p95_latency`、`tokens_per_case`，版本对比超过阈值时失败。

**⑥ 未测稳定性（`pass^k`）。** 每条用例只运行一次；工具顺序、检索 tie-break 和并发重排仍可能波动。
P0 子集运行 k=3，报告 `pass^k` 和不稳定用例。

**⑦ 未统计自动闭环率与转人工率。** 需求要求自动闭环率达到 70%，但 handoff 当前归入 `denied`。
为 `outcome` 增加 handoff，run 级增加 `automation_rate`。需要修改期望值。

**⑧ 未检查多轮 slot 漂移。** `D1-020` 要求第二轮复用订单号，但仅在漂移导致落库 `order_id` 变化时才会触发 `log_match`。
跨轮比较 `check_refund_eligibility.order_id`，并入工具入参断言。

### 6.3 实施范围

| 批次 | 缺口 | 要动什么 | 附带好处 |
|---|---|---|---|
| 只改打分器 | ②④⑤⑥⑧ | 从现有记录计算；新建 ex-2，不修改 `cases.jsonl` | 可用 ex-1 已导出的 trace 回算 |
| 补充期望值 | ①③⑦ | 在 `cases.jsonl` 增加 `tools.args` / `log.reason` / handoff；新建 d2 与 ex-2 | — |

---

## 七、后续规划

| 顺序 | 做什么 | 动哪一边 | 验收 |
|---|---|---|---|
| 1 | 修复 `assemble` 的父块重复并增加装配断言 | 检索链路 | 重跑 d1，确认重复正文与 token 减少；`citation_hit` 仅作观察项 |
| 2 | 修复 metadata 的实际模型上报 | 埋点 | 下一个 run 记录正确的模型信息 |
| 3 | 提示词增加硬否决检索要求，发布 `agent/v2` | Agent | 同集对比 v1 / v2，`D1-024` 通过且无退化 |
| 4 | 修复多轮 `must_call` 口径及 ②④⑤⑥⑧，新建 `ex-2` | 判分 | `D1-020` / `D1-021` 通过；保留 ex-1 原始结果 |
| 5 | 入参断言、落库 `reason`、handoff 维度 → `d2` + `ex-2` | 期望值 + 判分 | `D1-026` 这类「过程错但结论对」能被判负 |
| 6 | 建 retrieval 数据集（query → 应召回 section） | 新数据集 | 使用 recall@k / MRR 独立评估检索质量 |
| 7 | 回流线上失败样本 | 数据集 | 保留原始表述并脱敏，不改写为标准问法 |

先完成 1、2，避免证据重复和错误 metadata 影响后续 run 的可比性。3 修改 Agent，4 修改评分规则，分开执行以便归因。

后续补充版本对比自动化（`evals/compare.py`，同集运行 v1/v2 并逐条 diff）和线上监控（复用不依赖标注的三个指标，见 [4 · 5.4](https://tiltwind.github.io/refund-agent/doc/get-start/4-dataset.md#54-从轮到用例从用例到-run)）。设计见 [2 · 设计 · 6.2 / 6.4](https://tiltwind.github.io/refund-agent/doc/get-start/2-design.md#六持续评估闭环)。
