# 实验 rag-ex-1 —— 数据集 r1 的检索评测

被测对象是 [`search_policy`](../../retrieving/milvus.py) 的六步链路，不经过 Agent。样本来自
[`rag/datasets/r1`](../../datasets/r1/README.md)，指标口径见
[5 · 检索评测](https://tiltwind.github.io/refund-agent/doc/get-start/5-rag-eval.md)。

判分逻辑跟着实验走：**换判分口径 = 开新实验目录**，就地改会让历史 run 的分数不再可比。
期望值变了（改 `cases.jsonl` 的口径）就该开 r2。

当前实现到 [9.2](https://tiltwind.github.io/refund-agent/doc/get-start/5-rag-eval.md#92-实现顺序)
的第 2 步：三档 Recall 与两个辅助数，都是纯函数。Context Recall 与 Context Relevance 要调
judge，等校准做完再接。

---

## 一、跑之前

| 前置 | 命令 |
|---|---|
| Milvus 起着并已灌库 | `bash scripts/milvus.sh start` + `python rag/index/seed_milvus.py` |
| 数据集自检过 | `python rag/evals/validate_cases.py` |

改写那一步会调一次模型（`.env` 里没配凭据就原文透传，检索照跑，排序质量掉一档）。其余
两个模型是本地的：BGE-M3 嵌入、bge-reranker-v2-m3 重排。

## 二、跑

```bash
python rag/experiments/rag-ex-1/run_experiment.py                        # 离线，全量 102 条
python rag/experiments/rag-ex-1/run_experiment.py --cases R1-001 R1-082  # 只跑指定用例
python rag/experiments/rag-ex-1/run_experiment.py --langfuse --run-name $(git rev-parse --short HEAD)
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `dataset` | `rag/datasets/r1` | 数据集目录，离线路径用 |
| `--langfuse` | 关 | 样本改从 Langfuse 数据集拉，分数写回 dataset run，六步上报 trace |
| `--dataset-name` | `retrieval-cases-r1` | Langfuse 上的数据集名 |
| `--run-name` | `<数据集>-<条数>cases` | 版本对比时传 git sha |
| `--cases` | 全量 | 只跑指定 case_id |
| `--concurrency` | 4 | 嵌入和重排是本地模型，调太高只会互相抢算力 |
| `--out` | `result.json` | 只跑 `--cases` 子集时默认不写，免得覆盖全量结果 |

每条一行，`@1/@3/@10=001` 是三档的命中位：只有候选里有，重排后前 3 里没有。`-` 表示这一档
不适用（`multi_hop` 没有 `recall@1`）。

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
retriever  rag.search_policy   input={query, top_k}  output={sections}
├── span   rag.rewrite         output={rewritten, needs_law, sub_queries}
├── span   rag.route           output={routes}
├── span   rag.recall          output={candidates: [全部 20 个 chunk_id], single_path}
├── span   rag.rerank          output={passed, min_score, evidence: [{chunk_id, score, relevance, prior}]}
└── span   rag.assemble        output={sections}
```

`rag.recall` 记的是**完整候选序列**不是 top3 —— 「种子块排在第 13 位」这类问题只有全序列答得了。
埋点在 [`milvus.py`](../../retrieving/milvus.py) 而不是在这个脚本里，所以线上出坏 case 翻的是
同一棵树（`2-design 3.4`）。

## 三、指标

| 指标 | 读什么 | 诊断 |
|---|---|---|
| `recall@10` | 召回融合后的候选（`CANDIDATE_LIMIT = 20`） | 召回层的上界，这里没有的后面救不回来 |
| `recall@3` | 重排后前 3（`DEFAULT_TOP_K = 4`，留一格余量） | 实际交付水位 |
| `recall@1` | 重排后第 1 | 头部精度，对 `MIN_SCORE` 最敏感 |
| `evidence_tokens` | 装配后证据的 token 用量 | `TOKEN_BUDGET = 3000` 的实际占用 |
| `duplicate_ratio` | 重复正文的字符占比 | 装配把同一父块拼了两遍，Recall 看不见 |

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

## 五、门禁

三档 Recall 是纯函数，同一份数据集跑两遍结果一致，所以它做门禁；两个 LLM 指标接进来之后
只记录，不参与 pass / fail。

门禁看**相对值**：`seed_chunk_id` 是下界，绝对值本来就偏低，定一个 `recall@3 ≥ 0.9` 之类的
阈值没有依据。改参数后重跑，与上一版 `result.json` 对比，任一档下降超过容差即不通过。容差
在攒够几个基线 run 之后再定，第一版只记录。
