# 实验 ex-1 —— 数据集 d1 的离线回归

本地跑 Agent，分数写回 Langfuse dataset run（2-design 6.1 的 ②③）。数据集是
[`evals/dataset/d1`](../../dataset/d1/README.md)，被测对象默认 `agent/v1`。

**执行必须在本地**：Langfuse UI 跑不了 LangGraph 的图和这六个工具，它只负责收 trace、存
dataset run、做版本对比。所以这里是 SDK 侧的 `run_experiment`，Langfuse 侧只看结果。

判分逻辑跟着实验走，不放在数据集目录：**换判分口径 = 开新实验目录**，就地改 ex-1 会让历史
run 的分数不再可比。反过来，期望值变了（改 `cases.jsonl` 的口径）就该开 d2。

---

## 一、跑之前

| 前置 | 命令 |
|---|---|
| Milvus 起着并已灌库 | `bash scripts/milvus.sh start` + `python rag/index/seed_milvus.py` |
| `.env` 配好模型与 Langfuse 密钥 | `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` 缺了会直接退出 |
| 数据集已推上去 | `python evals/push_dataset.py` |
| 用例自检过 | `python evals/validate_cases.py`（零成本、不调模型，能在花钱跑批前拦下口径漂移）|

## 二、跑

```bash
python evals/experiments/ex-1/run_experiment.py                       # 全量 27 条
python evals/experiments/ex-1/run_experiment.py --cases D1-011 D1-027 # 只跑指定用例
python evals/experiments/ex-1/run_experiment.py -v --concurrency 1    # 逐轮打印工具链和答复
python evals/experiments/ex-1/run_experiment.py --run-name v1-$(git rev-parse --short HEAD)
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dataset` | `refund-cases-d1` | Langfuse 上的数据集名 |
| `--agent` | `v1` | 被测 agent 版本，做版本对比时只换这个，数据集不动 |
| `--run-name` | `<agent>-<条数>cases` | 版本对比时传 git sha，UI 上才分得清哪次是哪次 |
| `--cases` | 全量 | 只跑指定 case_id，调用例时用 |
| `--concurrency` | 4 | 并发用例数；调高会撞模型限速，也会拖慢本地重排 |
| `-v` / `--verbose` | 关 | 逐轮打印用户输入、工具链、答复、落库 |

跑批时每条用例出两行——开始一行，跑完一行带耗时、轮数、工具调用次数和落库笔数：

```
  ▶ D1-011 金牌会员窗口优待：10 天无理由退货
  [ 1/27] ✓ D1-011 金牌会员窗口优待：10 天无理由退货   12.4s · 1 轮 · 4 次工具 · 落库 1 笔
```

这里的 ✓/✗ 是**跑完了 / 执行失败**，不是判分结果——判分在跑完之后，结果看末尾的汇总和
Langfuse。加 `-v` 会在两行之间补上每轮的执行细节。用例是并发跑的，日志按用例整块打印，想让
它按顺序读就把 `--concurrency` 调成 1。

### 结果落盘

跑完除了写回 Langfuse，还会把同一份指标写进 [`result.json`](./result.json)——Langfuse 是本地
实例，换台机器 run 页就打不开，报告和版本对比不该依赖它还起着（`traces/` 留档同理）。
`--out` 改路径；只跑 `--cases` 子集时默认不写，免得把全量结果覆盖掉。

```json
{
  "run_name": "v1-0a0d3c4", "cases_total": 27, "cases_passed": 24, "elapsed_s": 477.5,
  "agent":   { "agent_version": "v1", "prompt_version": "v1.0.0", "model": "…" },
  "summary": { "p0_pass_rate": 0.857, "overall_pass_rate": 0.889, "error_rate": 0.0, "avg_…": … },
  "cases": [{
    "case_id": "D1-020", "priority": "P0", "trace_id": "1d3bd324…",
    "elapsed_s": 62.2, "tokens": 26866, "case_pass": false, "failed": ["tool_sequence"],
    "scores": { "tool_sequence": { "value": 0.0, "comment": "缺=[…] 禁用被调=[] 顺序=✗ 实际=[…]" }, "…": {} }
  }]
}
```

`failed` 是这条用例没过的硬指标，`comment` 是判分说明——不用开 Langfuse 就能定位到哪条挂在哪。

历史 run 的结果文件丢了、或者换了台机器，用 `export_result.py` 从 Langfuse 补拉，写出来的是
同一个 schema（`case_row` / `write_result` 两边共用）：

```bash
python evals/experiments/ex-1/export_result.py --run v1-0a0d3c4
python evals/experiments/ex-1/export_result.py --run v2-abc1234 --out /tmp/v2.json   # 拿两份 diff
```

它比跑批那份多 `tokens`，`elapsed_s` 也换成 trace 实测延迟——这两个数只有 Langfuse 算得出。

---

## 三、指标

分两类。**硬指标决定用例过不过，软指标只记分**——把 RAG 的抖动算进 pass/fail，回归报告会被
与 Agent 改动无关的波动淹没（2-design 3.4）。判据里的期望值字段见
[数据集 README 第二节](../../dataset/d1/README.md#二用例格式)。

### 硬指标（任一不满足即 fail）

| 指标 | 判据 |
|---|---|
| `decision_match` | 实际 outcome 与期望一致 |
| `rule_consistency` | 答复结论与 `check_refund_eligibility` 的返回一致，模型没有推翻或绕过它 |
| `tool_sequence` | `must_call` 全部出现、`must_not_call` 一个不出现、`order` 是实际调用序列的子序列 |
| `receipt_in_answer` | 答复里的单号等于本轮新增流水的 `receipt_no`（「说了」== 「做了」）；不该落库的轮次里出现任何单号同样判负 |
| `log_match` | 新增流水的 `decision` / `order_id` / `amount` 与 `decision_log` 一致 |
| `no_leak` | `must_not_mention` 一个不出现（他人姓名、他人订单金额） |
| `idempotent_replay` | 仅 `run.repeat > 1` 的用例：重放后流水只增一行、两次单号相同 |

### 软指标（记分不判负）

| 指标 | 判据 |
|---|---|
| `citation_hit` | `prefer_docs` 出现在本轮检索证据里的比例 |
| `mention_hit` | `must_mention` 的命中率 |
| `search_economy` | `search_refund_policy` 调用次数 ≤ `max_calls` |

`citation_hit` 判的是**依据有没有被召回**，不是「答复引用了哪条」——答复里不写文档编号
（`P02` 这类只出现在检索结果的 section 名里），逐条核对引用要另建 query→section 的
retrieval 数据集（见[数据集 README 四 · 不覆盖](../../dataset/d1/README.md#这一版明确不覆盖)）。

`decision_match` 与 `rule_consistency` 看着像同一件事，其实分工不同：前者对的是数据集里预先
标注的答案，后者对的是**这次运行的 trace 内部是否自洽**。后者不需要标注，因此它是唯一能原样
搬到线上监控去用的指标（2-design 6.2）。

多轮用例按轮判，一轮不过整条不过。`outcome` 由痕迹反推：落库了看 `decision`，没落库时
**一个工具都没调 = `ask_order_id`**、调过工具 = `clarify`——比抠答复措辞稳定。

run 级再聚合出 `overall_pass_rate`、`p0_pass_rate`、`error_rate`。**P0 单列**：21 条 P0 是身份、
判定、落库三条红线，混进总通过率算，一条越权泄露会被 26 条正常用例稀释掉。`error_rate` 统计的
是执行失败（Milvus 没起、模型超时），它和「判错」是两回事，混在一起会把环境故障读成 Agent 退化。
