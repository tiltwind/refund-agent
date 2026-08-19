# 实验 rag-ex-1 —— 数据集 r1 的检索评测

被测对象是 [`search_policy`](../../retrieving/milvus.py)，不经过 Agent：一个问题进去、一组
`PolicySection` 出来。样本来自 [`rag/datasets/r1`](../../datasets/r1/README.md)，两个指标的
算法见 [4 · 检索评测数据集](https://tiltwind.github.io/refund-agent/doc/get-start/4-rag-dataset.md)。
这里只写脚本怎么用，跑批结果、校准基线与门禁口径见
[5 · 检索评测](https://tiltwind.github.io/refund-agent/doc/get-start/5-rag-eval.md)。

指标分两组。LLM judge 判交付的上下文：

```
Context Precision = Σ(Precision@k × rel_k) / 相关条数   逐段判相关 + 位置加权，不要标注
Context Recall    = 被支撑的句子数 / ground_truth 句数  标准答案拆句 + 逐句归因
```

排序指标判出题的那段条文排第几，只比对 `source` 与检索链路的 ID 序列，不调模型：

```
candidate_hit   source 有没有进 20 条候选        分开「召回层漏了」和「重排/阈值挡了」
hit@1 / hit@4   重排后它排在第几
mrr             1 / 排名，掉出证据列表记 0
```

两组的可比性条件不同：judge 那组要判定口径相同（模型、思考档位、结构化方式），排序那组
只要数据集与 collection 没变。跨块样本按两段里排得最深的那个算 —— 只召回一半答不全。

判分逻辑跟着实验走：**换判分口径 = 开新实验目录**，就地改会让历史 run 的分数不再可比。
期望值变了（改 `cases.jsonl` 的口径）就该开 r2。

五个文件：`run_experiment.py` 跑批、`judge.py` 是两个 judge 指标、`rank_metrics.py` 是三个
排序指标、`report.py` 出 HTML 报告、`calibrate_judge.py` 校准 judge。

---

## 一、跑之前

| 前置 | 命令 |
|---|---|
| Milvus 起着并已灌库 | `bash scripts/milvus.sh start` + `python rag/index/seed_milvus.py` |
| 数据集自检过 | `python rag/evals/validate_cases.py` |

链路的改写那一步会调一次模型（`.env` 里没配凭据就原文透传，检索照跑，排序质量掉一档）。
其余两个是本地模型：BGE-M3 嵌入、bge-reranker-v2-m3 重排。

judge 模型从 `.env` 取，与改写模型分开配 —— 没单独配就回落到主模型，那是自己评自己，
跑批时打一行警告：

```bash
OPENAI_JUDGE_MODEL="deepseek-v4-pro"          # 或 ANTHROPIC_JUDGE_MODEL
REFUND_AGENT_JUDGE_STRUCTURED="json_mode"     # DeepSeek 上留着思考模式的那一档
REFUND_AGENT_JUDGE_REASONING="none"           # 思考档位；它换掉的是判定口径，不只是速度
```

## 二、跑

```bash
python rag/experiments/rag-ex-1/run_experiment.py                        # 离线，全量
python rag/experiments/rag-ex-1/run_experiment.py --cases R1-001 R1-081  # 只跑指定用例
python rag/experiments/rag-ex-1/run_experiment.py --langfuse --run-name $(git rev-parse --short HEAD)
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `dataset` | `rag/datasets/r1` | 数据集目录，离线路径用 |
| `--langfuse` | 关 | 样本改从 Langfuse 数据集拉，分数写回 dataset run，六步上报 trace |
| `--dataset-name` | `retrieval-cases-r1` | Langfuse 上的数据集名 |
| `--run-name` | `<数据集>-<条数>cases` | 版本对比时传 git sha |
| `--cases` | 全量 | 只跑指定 case_id |
| `--concurrency` | 8 | 本地模型的前向与 tokenizer 都被一把全局锁串起来（同一块 GPU，MPS 后端不是线程安全的；HF fast tokenizer 并发调用抛 `Already borrowed`，见 `llm/device.py`），调高它并行的是改写和 judge 那些网络调用 |
| `--out` | `result.json` | 只跑 `--cases` 子集时默认不写，免得覆盖全量结果 |

跑批同时在 `--out` 旁边落一份 `context_snapshot.json`：这一轮 judge 的输入（问题、有序
上下文全文、标准答案），校准 judge 用，见下面「judge 校准」。`result.json` 里的
`sections` 只有标题，判定复现要的是正文。

每条一行：`4 段 · CP=0.833 CR=1.000 · rank=证据列表第 2 位（共 16 条）`。最后那一段写的是
「卡在哪一步」而不是数字 —— 一个 `hit@4=0` 说明不了该改召回还是该改阈值。

检索的三种出口分开记账：

| 出口 | 判分 | 计入均值 |
|---|---|---|
| 正常返回 | 判 | 是 |
| `RetrievalError`（重排后没有可交付证据） | 上下文为空，两个 judge 指标判 0 | 是 |
| 其他异常（跑批故障） | 不判 | 否，进 `summary.failures`，行首打「跑批故障」 |

第三行与第二行分开，是因为它们混在一起时一次 tokenizer 竞态会被记成「检索没召回」——
均值里凭空多出几条 0，看上去像检索退化。口径与 judge 调用失败不写分数一致。

耗时绝大部分花在 judge 上：每条两次调用，空上下文的那几条直接判 0 不调。排序指标不花
时间 —— 它读的是同一次检索已经产出的 ID 序列。

### 两条路径

判分逻辑只有一份，在 `run_case` 里；两条路径的区别只是样本从哪来、结果往哪写。

| | 默认 | `--langfuse` |
|---|---|---|
| 样本 | `cases.jsonl` | Langfuse 数据集（先 `python rag/evals/push_dataset.py` 推） |
| 分数 | `result.json` | `result.json` + dataset run |
| 检索 trace | 不上报（脚本把 `REFUND_AGENT_RAG_SPAN` 关掉） | 六步随 item 的 trace 上报 |

调参走默认路径，不该被一个本地 Langfuse 实例的死活卡住。版本对比用 `--langfuse`，
run 页上能并排看两次跑批。

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
| `rag.recall` | 全部 20 条的 `chunk_id` + 小节名 + RRF 分 + 命中来源，无正文 | 捞没捞到、是哪一路捞的、排第几 |
| `rag.rerank` | 每条证据的三个分 + 120 字摘录，外加被阈值砍掉的 `dropped` | 分数为什么是 0.98 或 0.29。重排返回空时 `dropped` 是全部 20 条，一眼看出是阈值不是召回的问题 |
| `rag.assemble` / 根节点 | `PolicySection` **全文** | 这是两个指标真正判定的东西，掉分时要对着正文核对 |

埋点在 [`milvus.py`](../../retrieving/milvus.py) 而不是在这个脚本里，所以线上出坏 case
翻的是同一棵树（`2-design 3.4`）。

## 三、结果文件

`result.json` 是唯一事实源，报告和版本对比都读它。逐条那份带每一次判定的理由：

```json
{
  "case_id": "R1-039", "doc_id": "P03",
  "question": "…",
  "error": null, "failure": null,
  "sections": ["P03 退款到账时效 > 第二条 各渠道到账时效", "…"],
  "scores": { "context_precision": 0.5, "context_recall": 1.0,
              "candidate_hit": 1.0, "hit@1": 0.0, "hit@4": 1.0, "mrr": 0.5 },
  "retrieval": {
    "candidate_ids": ["P03#004:01", "…"],
    "evidence_ids": ["P03#004:01", "P03#004:02", "…"],
    "rank_note": "证据列表第 2 位（共 4 条）"
  },
  "judge": {
    "context_precision": { "score": 0.5, "n": 3, "hit": 1,
      "detail": [{ "text": "【P03 …】…", "hit": true, "reason": "给出了到账时效" }, "…"] }
  }
}
```

一个 0.5 分说明不了改哪里 —— 要能翻到是哪一段被判不相关、哪一句没被支撑。两条 ID 序列
落盘是同一个道理：一条样本掉分时，要能当场分清 source 是没被召回、还是被阈值挡在了
证据列表外，这两种情况的修法完全不同。

judge 调用失败的那条不写分数，从均值的分母里少掉，失败清单进 `summary.judge_errors`；
链路本身跑挂的那条一个分数都不写，进 `summary.failures`。两份清单都要报出来 ——
写 0 会被均值当成检索没召回，把故障记到检索头上。

走 `--langfuse` 时每条还会多一个 `trace_id`，run 级多一个 `run_url`。反过来不成立 ——
Langfuse 是本地实例，换台机器就打不开，所以报告只读结果文件。

`config` 记的是这次跑批的被测参数（collection、三个模型、`top_k`、`MIN_SCORE`、各权重）。
两次 run 的分数只有在这些值相同的前提下才可比 —— 尤其是 `judge_model` 与
`judge_reasoning`：换 judge 等于换判定口径。

### HTML 报告

```bash
python rag/experiments/rag-ex-1/report.py                       # → report.html
```

五节：两个 judge 指标的均值与公式、四个排序指标、分数分布、按文档分档、逐条明细。
明细里每条都能展开看逐条判定，「source 排名」那一列直接写出卡在哪一步。分布图看的是
均值掩盖的形状 —— 0.87 可能是「都在 0.87」，也可能是「八成满分、两成接近 0」，
这两种情况的修法不同。

### judge 校准

跑批的分数里混着两个变量：检索每次返回的上下文会变，judge 的判定也会变。校准把前一个
拿掉 —— 固定住上下文，只让 judge 重复判定，剩下的波动就都是它自己的。

```bash
python rag/experiments/rag-ex-1/calibrate_judge.py            # 3 轮，读 context_snapshot.json
python rag/experiments/rag-ex-1/calibrate_judge.py --rounds 5
```

不碰 Milvus、embedding、reranker，只读快照。三个数进 `calibration.json`：

| 数 | 说的是什么 | 怎么用 |
|---|---|---|
| 轮间均值极差 | 判定噪声底 | 两次 run 的指标差值大过它，才算链路真的动了 |
| verdict 翻转率 | 逐段/逐句判定在各轮之间判得不一致的比例 | 提示词的边界说清楚了没有 |
| 样本分数极差 | 哪几条判得最不稳 | 分歧明细带各轮理由，是改提示词的入口 |

空上下文的条目不参与：judge 对它们短路判 0，不调模型。

判得稳不等于判得对 —— 后者要人工标注。两个指标的判定单位不同，抽样也分开：Context
Precision 按段抽、Context Recall 按句抽，人工标 hit 之后与 judge 的判定算一致率，再看
错的方向偏哪边。上下文固定住之后，同一份标注在改过提示词的 judge 上还能接着用。

换 `judge_model`、`judge_reasoning` 或 `judge_structured` 等于换判定口径，新旧一致率
不可比 —— 快照里记着跑批那轮的 judge，对不上时脚本打一行提示。
