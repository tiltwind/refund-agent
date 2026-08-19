# 实验 rag-ex-1 —— 数据集 r1 的检索评测

被测对象是 [`search_policy`](../../retrieving/milvus.py) 的六步链路，不经过 Agent。样本来自
[`rag/datasets/r1`](../../datasets/r1/README.md)，指标口径见
[5 · 检索评测](https://tiltwind.github.io/refund-agent/doc/get-start/5-rag-eval.md)。

判分逻辑跟着实验走：**换判分口径 = 开新实验目录**，就地改会让历史 run 的分数不再可比。
期望值变了（改 `cases.jsonl` 的口径）就该开 r2。

打分器分两处：`scorers.py` 是纯函数（三档 Recall + 两个辅助数，进门禁），`judge.py` 要调 LLM
（Context Recall + Context Relevance，默认关，`--judge` 打开，只进报告）。六步上报 Langfuse，
分数写回 dataset run。

---

## 一、跑之前

| 前置 | 命令 |
|---|---|
| Milvus 起着并已灌库 | `bash scripts/milvus.sh start` + `python rag/index/seed_milvus.py` |
| 数据集自检过 | `python rag/evals/validate_cases.py` |
| 样本带 `claims`（只有 `--judge` 要） | `python rag/evals/generate_claims.py` |

改写那一步会调一次模型（`.env` 里没配凭据就原文透传，检索照跑，排序质量掉一档）。其余
两个模型是本地的：BGE-M3 嵌入、bge-reranker-v2-m3 重排。

`--judge` 另外要一个 judge 模型，从 `.env` 取，与改写模型分开配 —— 没单独配就回落到主模型，
那是自己评自己，跑批时会打一行警告：

```bash
OPENAI_JUDGE_MODEL="deepseek-v4-pro"          # 或 ANTHROPIC_JUDGE_MODEL
REFUND_AGENT_JUDGE_STRUCTURED="json_mode"     # DeepSeek 上留着思考模式的那一档
```

## 二、跑

```bash
python rag/experiments/rag-ex-1/run_experiment.py                        # 离线，全量 102 条
python rag/experiments/rag-ex-1/run_experiment.py --cases R1-001 R1-082  # 只跑指定用例
python rag/experiments/rag-ex-1/run_experiment.py --judge --concurrency 8  # 加两个 LLM 指标
python rag/experiments/rag-ex-1/run_experiment.py --langfuse --run-name $(git rev-parse --short HEAD)
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `dataset` | `rag/datasets/r1` | 数据集目录，离线路径用 |
| `--langfuse` | 关 | 样本改从 Langfuse 数据集拉，分数写回 dataset run，六步上报 trace |
| `--dataset-name` | `retrieval-cases-r1` | Langfuse 上的数据集名 |
| `--run-name` | `<数据集>-<条数>cases` | 版本对比时传 git sha |
| `--cases` | 全量 | 只跑指定 case_id |
| `--judge` | 关 | 加算 Context Recall 与 Context Relevance，每条多两次 judge 调用 |
| `--concurrency` | 4 | 嵌入和重排的前向是串行的（同一块 GPU，且 MPS 后端不是线程安全的，见 `llm/device.py`），调高它并行的是改写和 judge 那些网络调用 —— 带 `--judge` 时值得调到 8 |
| `--out` | `result.json` | 只跑 `--cases` 子集时默认不写，免得覆盖全量结果 |

每条一行，`@1/@3/@10=001` 是三档的命中位：只有候选里有，重排后前 3 里没有。`-` 表示这一档
不适用（`multi_hop` 没有 `recall@1`）。带 `--judge` 时行尾多出 `CR`（Context Recall）与
`CRel`（Context Relevance）。

`--judge` 默认关，因为调参迭代要的是三档 Recall：多付两百多次 judge 调用、慢一个量级，而那
两个数不进门禁。出报告、做版本留档时再带上。

### 两条路径

判分逻辑只有一份，在 `run_case` 里；两条路径的区别只是样本从哪来、结果往哪写。

| | 默认 | `--langfuse` |
|---|---|---|
| 样本 | `cases.jsonl` | Langfuse 数据集（先 `python rag/evals/push_dataset.py` 推） |
| 分数 | `result.json` | `result.json` + dataset run |
| 检索 trace | 不上报（脚本把 `REFUND_AGENT_RAG_SPAN` 关掉） | 六步随 item 的 trace 上报 |

门禁走默认路径：三档 Recall 不该被一个本地 Langfuse 实例的死活卡住。版本对比用
`--langfuse`，run 页上能并排看两次跑批。

### Langfuse 上能看到什么

每条 item 的 trace 里，`rag.search_policy`（retriever）下面挂着六步：

```
retriever  rag.search_policy   in {query, top_k}          out {sections: 全文}
├── span   rag.rewrite         in {query}                 out {rewritten, needs_law, sub_queries}
├── span   rag.route                                      out {routes: 每层名额与法规层权重}
├── span   rag.recall                                     out {candidates: 20 条 × {chunk_id, section, rrf, hits}}
├── span   rag.rerank          in {query, candidates}     out {passed, min_score, dropped, evidence: [… + excerpt]}
└── span   rag.assemble        in {top_k, evidence}       out {sections: 全文}
```

三层的粒度不一样，按各自要回答的问题定：

| 层 | 记到什么程度 | 因为要回答 |
|---|---|---|
| `rag.recall` | 全部 20 条的 `chunk_id` + 小节名 + RRF 分 + 命中来源，无正文 | 捞没捞到、是哪一路捞的、排第几。20 条全文有几万字，这一步用不上 |
| `rag.rerank` | 每条证据的三个分 + 120 字摘录，外加被阈值砍掉的 `dropped` | 分数为什么是 0.98 或 0.29。重排返回空时 `dropped` 是全部 20 条，一眼看出是阈值不是召回的问题 |
| `rag.assemble` / 根节点 | `PolicySection` **全文**，含 `text`、`score`、`reason`、来源路径 | 这是真正注入模型上下文的东西。judge 接进来后判的就是这段文字，只留小节名的话掉分时无从核对；装配把同一父块拼两遍，也只有正文在场才看得出来 |

一次检索的 span 合计约 19KB。埋点在 [`milvus.py`](../../retrieving/milvus.py) 而不是在这个脚本里，
所以线上出坏 case 翻的是同一棵树（`2-design 3.4`）。

## 三、指标

| 指标 | 读什么 | 诊断 |
|---|---|---|
| `recall@10` | 召回融合后的候选（`CANDIDATE_LIMIT = 20`） | 召回层的上界，这里没有的后面救不回来 |
| `recall@3` | 重排后前 3（`DEFAULT_TOP_K = 4`，留一格余量） | 实际交付水位 |
| `recall@1` | 重排后第 1 | 头部精度，对 `MIN_SCORE` 最敏感 |
| `evidence_tokens` | 装配后证据的 token 用量 | `TOKEN_BUDGET = 3000` 的实际占用 |
| `duplicate_ratio` | 重复正文的字符占比 | 内容重复信号；是否由装配造成需结合 trace 验证 |
| `context_recall` | 参考答案的 claim 有几条被检回的上下文支撑 | 种子 ID 判负而它高 = 召回了等价条款，是标注不全不是检索坏 |
| `context_relevance` | 检回内容里跟 query 相关的句子占比 | 惩罚「多召回」。`top_k` 调大时它下降，与 Recall 一起读才知道净赚还是净亏 |

后两个挂在 `PolicySection.text` 上（那是真正注入模型上下文的东西），逐条判定理由落在
`result.json` 的 `judge` 字段里。它们调模型、有噪声，所以只记录不进门禁；judge 校准做完
之前结论不采信。claim 不在跑批时现拆，是 `cases.jsonl` 里的一个字段 —— 分母得在两次 run
之间保持一致。judge 调用失败的那条不写分数（写 0 会被均值当成检索没召回），失败清单进
`summary.judge_errors`。

命中口径是**全部种子块都进前 k 才算命中**，`multi_hop` 不给部分分：只召回一半，答案照样是错的。

`unanswerable` 的 6 条没有种子块，不进上述均值 —— 空集是任何集合的子集，算出来是满分。它们
只判一件事：链路有没有按口径抛异常（`unanswerable_raised`）。

分档按 `style` / `type` / `layer` / `kind` / `doc_id` 各切一份，落在 `result.json` 的
`breakdown` 里。总均值只用来看趋势，能指向改哪个文件的是分档。

## 四、结果

`result.json` 是唯一事实源，报告和版本对比都读它。第一次全量跑的结论见
[基线报告](./baseline-report.md)。逐条那份带 `seed_rank` ——
种子块在候选里第几、在证据里第几，两个名次差就是重排的账：

```json
{
  "case_id": "R1-046", "type": "single", "style": "colloquial", "kind": "table",
  "seed_chunk_id": ["P05#002:00"],
  "scores": { "recall@10": 1.0, "recall@3": 0.0, "recall@1": 0.0,
              "evidence_tokens": 2604, "duplicate_ratio": 0.434 },
  "seed_rank": { "candidate": { "P05#002:00": 2 }, "evidence": { "P05#002:00": 12 } }
}
```

召回层排第 2，重排压到第 12 —— 要改的是重排权重，不是切片。`multi_hop` 没有 `recall@1` 这个键
（k 小于种子块数不计分），三档各自的分母记在 `summary.counted` 里。

走 `--langfuse` 时每条还会多一个 `trace_id`，run 级多一个 `run_url`：从结果文件能直接跳到那条
trace 看六步。反过来不成立 —— Langfuse 是本地实例，换台机器就打不开，所以报告只读结果文件。

`config` 记的是这次跑批的被测参数（collection、模型、`top_k`、`MIN_SCORE`、各权重）。两次 run
的分数只有在这些值相同的前提下才可比 —— 尤其是 collection，重新灌库而 `chunk_id` 漂了，
Recall 会全线暴跌，看上去像检索退化。

### 现场记录

```bash
python rag/experiments/rag-ex-1/export_traces.py                # 抽 20 条导出到 traces/
python rag/experiments/rag-ex-1/export_traces.py --all          # 全量 102 条
python rag/experiments/rag-ex-1/export_traces.py --cases R1-046 # 只导指定用例
```

抽样不是随机的 —— 随机 20 条里大概率一条空证据都没有，而那恰恰是最该留档的。按报告里的每个
结论分桶取，最后三条是三档全中的对照：只留坏 case 的话，读的人无从判断正常的一次检索长什么样。
桶的定义在脚本的 `BUCKETS` 里，也写进 [`traces/README.md`](./traces/README.md)。

`--all` 是同一套分组、不截断，剩下的进「其余」。留全量的理由是报告的分档表：口语档、表格档
指到的是一批用例，只留 20 条抽样的话点进去多半没有对应的那一条。代价是 102 条约 6MB。

每条两个文件：`.md` 人读版（六步展开，装配那步是条款全文），`.json` 机器版（全部 observation
原样保留）。跑批带过 `--judge` 的话，`.md` 末尾还有一段 judge 判定：逐条 claim 支不支撑、
哪些内容单元被判为不相关 —— 就跟在装配的正文后面，对着看的。要跑批带过 `--langfuse`
才有 `trace_id` 可导。

### HTML 报告

```bash
python rag/experiments/rag-ex-1/report.py                       # → rag-ex-1-report.html
```

[`rag-ex-1-report.html`](./rag-ex-1-report.html) 是同一批数据的可视化：三档、四张分档表、
两个 LLM 指标、四个链路问题、run 间抖动、102 条明细（链到现场记录）。只读 `result.json`，
不连 Langfuse。结果文件里带 `--judge` 那两个键时，分档表和明细表各多两列，第三节多出
`recall@3` × `Context Recall` 的四象限 —— 那张表是这两个指标存在的理由：种子 ID 判负而
上下文撑得住，说明召回的是等价条款，只报 Recall 的话它和真的没召回长得一样。
`report.py` 里的 `HISTORY` 是人填的 —— 结果文件只存最近一次，而抖动幅度要几次才看得出来。

## 五、门禁

三档 Recall 是纯函数，所以它做门禁；两个 LLM 指标接进来之后只记录，不参与 pass / fail。

**打分器是纯函数，被测链路不是**：改写那一步调模型，同一份数据集跑几次会有一两条翻转，
`recall@10` 实测抖 ±0.02。定容差之前先看[基线报告第六节](./baseline-report.md#六门禁前要解决的一件事链路不是确定性的)。

门禁看**相对值**：`seed_chunk_id` 是下界，绝对值本来就偏低，定一个 `recall@3 ≥ 0.9` 之类的
阈值没有依据。改参数后重跑，与上一版 `result.json` 对比，任一档下降超过容差即不通过。容差
在攒够几个基线 run 之后再定，第一版只记录。
