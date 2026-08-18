# 5 · 检索评测：指标、跑批与结果

用 [4 · 检索评测数据集](https://tiltwind.github.io/refund-agent/doc/get-start/4-rag-dataset.md)的 `r1` 评[六步检索链路](https://tiltwind.github.io/refund-agent/doc/get-start/3-rag-impl.md#七第-6-步--检索链路-ragretrieving)。被测对象是 `search_policy` 及其内部各步，不含 Agent。实验目录 `rag/experiments/rag-ex-1/`。

一次全量 102 条，`recall@1 / @3 / @10` = 0.637 / 0.760 / 0.885，跑出四个待处理的问题，见第五、六节。

---

## 一、三个指标的分工

只报一个数字就没有归因能力。三个指标各自回答一个问题：

| 指标 | 问题 | 真值 | 调模型 | 现状 |
|---|---|---|---|---|
| Recall@k | 该召回的块，召回了吗 | `seed_chunk_id`，集合比对 | 否 | 已实现 |
| Context Recall | 检回的上下文，撑得住参考答案吗 | 参考答案拆成 claim，逐条判 | 是 | 见第七节 |
| Context Relevance | 检回的东西，有多少是真有用的 | 无需标注 | 是 | 见第七节 |

Recall 不够用，因为 `seed_chunk_id` 只是种子块，不是全部相关块（[4 · 5.3](https://tiltwind.github.io/refund-agent/doc/get-start/4-rag-dataset.md#53-种子块不等于全部相关块)）。链路召回了 P07 而不是种子块 P02，两条讲的是同一个窗口规则，ID 级判它错，实际够用。

Context Relevance 管的是另一个方向：前两个指标都在问「捞得够不够全」，把 `DEFAULT_TOP_K` 从 4 改成 20，两个分数一起变好看。没有一个惩罚「多召回」的指标，调参会朝着往上下文里塞东西的方向走，而 `TOKEN_BUDGET = 3000` 是硬上限。

三者分开报。合成一个加权总分会把归因能力丢掉。

---

## 二、指标的链路位置

链路后半段有两种粒度的产物：

```
… → 4 召回融合 → Candidate（chunk 级，有 chunk_id）    ← recall@10 挂这里
       → 5 重排  → Evidence（chunk 级，有 chunk_id + 分数） ← recall@3 / @1 挂这里
              → 6 装配 → PolicySection（父块级，无 chunk_id） ← 两个 LLM 指标挂这里
```

[`PolicySection`](https://github.com/tiltwind/refund-agent/blob/main/rag/retrieving/protocol.py) 上没有 `chunk_id`：装配把命中的子块还原成完整小节，还可能合并相邻父块，一条 `PolicySection` 对应一组子块。所以 Recall@k 取重排那一层的 ID，两个 LLM 指标取 `PolicySection.text`。

分开挂是因为装配自己会出错。[`assemble`](https://github.com/tiltwind/refund-agent/blob/main/rag/retrieving/pipeline/assemble.py) 按 `(parent_seq, parent_id, section_path)` 分组回填父块，而同一父块的子块 `section_path` 各不相同，同一个 `parent_id` 因此被登记多次，父块正文被重复拼进上下文。种子块召回了，Recall 满分，注入模型的上下文里却有一大截是重复正文 —— 这类缺陷只有落在装配产物上的指标才抓得住。

### 2.1 中间产物的取法

`search_policy` 只返回 `PolicySection`，评测要的 `chunk_id` 序列在它上面没有。两个做法：

| 做法 | 代价 |
|---|---|
| 评测脚本自己按 `rewrite → route → recall → rerank → assemble` 编排一遍 | 复制了 [`milvus.py`](https://github.com/tiltwind/refund-agent/blob/main/rag/retrieving/milvus.py) 的编排逻辑，容易和线上跑偏 |
| 给 `search_policy` 加一个返回 trace 的旁路 | 多一个入口方法；评测和线上跑的是同一段代码 |

取第二种，与 `2-design 3.4` 一致：prod 与 eval 走同一条检索路径。落地成 `search_with_trace` —— 编排在它里面，`search_policy` 调它并丢掉 trace，线上调用方一行不用改。[`RetrievalTrace`](https://github.com/tiltwind/refund-agent/blob/main/rag/retrieving/protocol.py) 上加两个结构化字段 `candidate_ids` / `evidence_ids`。人读的 `steps` 字符串里也有 ID，但正则去抠一句日志，措辞改一个字打分器就静默判负。

同一处埋点还把六步各自上报成 Langfuse span，挂在 `rag.search_policy` 这个 retriever 节点下。埋在链路里而不是评测脚本里，线上出坏 case 翻的是同一棵树。三层的粒度不同：

| span | 记什么 |
|---|---|
| `rag.recall` | 全部 20 条候选的 `chunk_id`、小节名、RRF 分、命中来源，无正文 |
| `rag.rerank` | 每条证据的分数拆解、120 字摘录，加上被 `MIN_SCORE` 砍掉的名单 |
| `rag.assemble` | `PolicySection` 全文 —— 那是真正注入模型上下文的东西 |

---

## 三、Recall@k 口径

```
Recall@k = 全部种子块都在前 k 个里 → 1，否则 0
```

`multi_hop` 不给部分分：只召回一半，答案照样是错的（[4 · 5.4](https://tiltwind.github.io/refund-agent/doc/get-start/4-rag-dataset.md#54-单块答不全的问题)）。k 取三个值：

| 指标 | k | 读什么 |
|---|---|---|
| `recall@10` | 召回融合后的候选，`CANDIDATE_LIMIT = 20` 之内 | 召回层的上界 |
| `recall@3` | 重排后前 3 | 实际交付水位。`DEFAULT_TOP_K = 4`，取 3 留一格余量 |
| `recall@1` | 重排后第 1 | 头部精度，对 `MIN_SCORE = 0.30` 最敏感 |

k 小于种子块数的档次不计分：两个种子块不可能同时排第 1，`multi_hop` 的 `recall@1` 是结构性的 0，算进均值等于按 multi_hop 的占比给这一档加了个固定折扣。r1 的 16 条 multi_hop 因此只进 `@3` 与 `@10`，三档的分母各自记在结果文件的 `summary.counted` 里。

`@10` 读候选、`@3` 读重排后的证据，两个序列不同，所以「`@10` 判负而 `@3` 满分」是可能的：候选第 15 被重排提进了前 3。

### 3.1 两档之差的归因

| 现象 | 结论 | 改哪里 |
|---|---|---|
| `recall@10` 低 | 召回层就没捞到 | 切片、块头、BM25 分析器、`CANDIDATE_LIMIT`、[过滤条件](https://github.com/tiltwind/refund-agent/blob/main/rag/retrieving/pipeline/filters.py) |
| `recall@10` 高、`recall@3` 低 | 捞到了但被压下去 | [重排](https://github.com/tiltwind/refund-agent/blob/main/rag/retrieving/pipeline/rerank.py)：`RELEVANCE_WEIGHT` / `PRIOR_WEIGHT`、`DOC_PRIOR` 的 P02 偏置 |
| `recall@3` 高、`recall@1` 低 | 头部排序不稳 | `RRF_K`、tie-break |
| `formal` 档高、`colloquial` 档低 | 链路吃字面匹配 | 改写环节、稠密一路 |
| `table` 档低于 `text` 档 | 表格块切碎了或块头没带表头 | 切片 |

一个混合的「检索命中率」给不出其中任何一行。

---

## 四、跑批方式

实验目录里 `run_experiment.py` 跑批、`scorers.py` 是打分器（`recall_at_k` 只做集合比对）、`export_traces.py` 与 `report.py` 出留档。

前置：Milvus 起着并已灌库，数据集自检过。

```bash
bash scripts/milvus.sh start && python rag/index/seed_milvus.py
python rag/evals/validate_cases.py
```

两条路径，判分逻辑共用，落同一份 `result.json`：

```bash
# 离线：样本读 cases.jsonl，不连 Langfuse
python rag/experiments/rag-ex-1/run_experiment.py

# dataset run：样本从 Langfuse 数据集拉，分数写回，六步上报 trace
python rag/evals/push_dataset.py
python rag/experiments/rag-ex-1/run_experiment.py --langfuse --run-name baseline-2
```

| 路径 | 用在哪 |
|---|---|
| 默认（离线） | 调参。三档 Recall 是要进门禁的数，不该被一个本地 Langfuse 实例的死活卡住；脚本会把 `REFUND_AGENT_RAG_SPAN` 关掉，免得堆一批不挂在任何 dataset run 上的 trace |
| `--langfuse` | 留档与版本对比。run 页上能并排看两次跑批，逐条 `trace_id` 落进结果文件 |

102 条全量，离线约 75s，`--langfuse` 约 210s。跑批时每条一行：

```
  [ 46/102] R1-046 2.7s · @1/@3/@10=001 · 2604 token
  [ 47/102] R1-047 2.1s · @1/@3/@10=111 · 1708 token
```

`@1/@3/@10=001` 是三档的命中位，`-` 表示这一档不适用（`multi_hop` 没有 `recall@1`）。

跑完的产物：

| 产物 | 作用 |
|---|---|
| `result.json` | run 级与逐条指标，报告只读它，不依赖 Langfuse 在线 |
| `baseline-report.md` | 结论与归因 |
| `rag-ex-1-report.html` | 同一批数据的可视化，`python rag/experiments/rag-ex-1/report.py` 生成 |
| `traces/` | 20 条现场记录，`python rag/experiments/rag-ex-1/export_traces.py` 导出 |

`export_traces.py` 的抽样按「报告里每个结论各留一份现场」分桶，不是随机 —— 随机 20 条里大概率一条空证据都没有。最后三条是三档全中的对照。

---

## 五、跑批结果

run `baseline-2`，102 条样本、96 条计分（6 条 `unanswerable` 不进 Recall 均值），206s，无执行失败。

| 指标 | 值 | 分母 |
|---|---|---|
| `recall@1` | 0.637 | 80（不含 multi_hop） |
| `recall@3` | 0.760 | 96 |
| `recall@10` | 0.885 | 96 |
| `evidence_tokens` | 1267 | 预算 3000 |
| `duplicate_ratio` | 0.207 | 96 条中 58 条含重复正文 |

分档（`unanswerable` 除外）：

| 分档 | n | `recall@1` | `recall@3` | `recall@10` |
|---|---|---|---|---|
| `formal` | 48 | 0.725 | 0.896 | 0.938 |
| `colloquial` | 48 | 0.550 | 0.625 | 0.833 |
| `text` | 62 | 0.732 | 0.871 | 0.952 |
| `table` | 26 | 0.417 | 0.615 | 0.769 |
| `table+text` | 8 | — | 0.375 | 0.750 |
| `single` | 80 | 0.637 | 0.825 | 0.912 |
| `multi_hop` | 16 | — | 0.438 | 0.750 |

按 3.1 的表读：口语档比书面档低一大截，差距在 `@10` 上收窄 —— 召回层捞得到，是排序吃字面匹配，责任在改写与稠密一路。表格块低于正文块，那是切片的事。

`recall@10` 丢的 11 条分两类：5 条种子块根本没进候选（救不回来，要看切片和过滤），6 条排在候选第 11~19（落在 `CANDIDATE_LIMIT = 20` 内，重排还有机会）。`@10` 命中而 `@3` 丢的 15 条里，4 条的种子块被 `MIN_SCORE` 整批滤掉，其余是名次被压 —— 候选第 2 掉到证据第 13 这种。

---

## 六、暴露的问题

### 6.1 重排全滤后静默返回空列表

4 条用例拿到 0 条证据且不报错。R1-074 的 trace：

```
rag.recall     candidates[0] = P10#004:03      ← 种子块排第 1
rag.rerank     passed=0, min_score=0.3, dropped=[全部 20 条]
rag.assemble   sections=[]
```

链路对「一条候选都没有」是显式抛异常的，但「重排后一条都不剩」走的不是这条路：`assemble([])` 返回空列表，工具层拿到空证据，Agent 退回凭记忆答政策。阈值偏高和空结果不报错是两件事，前者调参，后者是口径漏洞。

### 6.2 `unanswerable` 兜底口径不成立

6 条全部没抛异常。异常只在候选为空时触发，而召回层对任何 query 都能捞回 20 条 —— 问「保价规则」照样返回一堆退货条款。两条路：给链路加「最高分低于下限即判无适用条款」，或者把这类样本的判据改成 Context Relevance。

### 6.3 重复正文占 20.7%

96 条里 58 条含重复，最高一条 75%。成因是第二节说的分组键。`recall@3` 满分的用例里照样有 0.4 以上的重复率，ID 级 Recall 对它无感 —— 这是配套记辅助数的理由。

### 6.4 链路非确定性与门禁容差

同一套参数跑四次，`recall@1` 四次相同，`@3` 抖 1 条（±0.010），`@10` 抖 2 条（±0.021）。翻转的都是种子块在候选里排 11~13 名、来回跨 `k=10` 的用例。来源是改写那一步调模型，温度 0 也不保证网关每次返回同一份拆分。

打分器是纯函数，被测链路不是。门禁口径因此用相对值：`seed_chunk_id` 是下界，定一个 `recall@3 ≥ 0.9` 之类的绝对阈值没有依据；改参数后重跑，与上一版 `result.json` 对比，任一档下降超过容差即不通过。容差要盖过上面这个抖动幅度，或者先把改写的结果按 query 缓存下来。

---

## 七、两个待接的 LLM 指标

两个都要 LLM judge，判分挂在 `PolicySection.text` 上。

```
Context Recall    = 被检回上下文支撑的 claim 数 / 参考答案中的 claim 总数
Context Relevance = 与 query 相关的内容单元数 / 检回的内容单元总数
```

Context Recall 用参考答案当代理，不需要标注 `reference_contexts`：先把 `reference_answer` 拆成 claim，再逐条判有没有被上下文支撑。拆 claim 与被测链路无关，结果要缓存，缓存键用 `case_id` 加参考答案的哈希。判定环节四条约束：温度 0 且结构化输出；不确定时判「不支撑」；只判「上下文里有没有」，不判「答案对不对」；judge 模型与链路的[改写模型](https://github.com/tiltwind/refund-agent/blob/main/rag/retrieving/pipeline/rewrite.py)分开配置。每条 claim 的判定理由要落盘。

Context Relevance 的「内容单元」取句子而不是取 `PolicySection` —— 一条 `PolicySection` 是回填后的完整小节，里面必然混有不相关的句子，那是父块回填故意带进来的（[3 · 五](https://tiltwind.github.io/refund-agent/doc/get-start/3-rag-impl.md#五第-4-步--切片-ragchunking)）。按块判会把这个设计判成缺陷。它的绝对值不该追求高，用途是横向对比：`top_k` 调大时 Recall 上升、它下降，两个一起看才知道是净赚还是净亏。

接进来之前要先校准 judge，否则两个分数只是噪声。[4 · 8.2](https://tiltwind.github.io/refund-agent/doc/get-start/4-rag-dataset.md#82-人工抽检) 的人工抽检样本在这里第二次派上用场：算 judge 与人工标注的一致率、假阳假阴的分布、同一批样本判两遍的自一致性。一致率不达标就修提示词或换 judge 模型，修好之前这两个指标的结论不采信。校准结果跟着数据集版本记录，换了 judge 模型历史分数就不可比。

两个都调模型、有噪声，因此归入观察指标：记录、进报告、看趋势，不参与 pass / fail。

三个指标齐了之后这样联合读：

| Recall@3 | Context Recall | Context Relevance | 判断 | 动哪里 |
|---|---|---|---|---|
| 低 | 低 | — | 真的没召回 | 召回层，先看 `recall@10` 分档 |
| 低 | 高 | — | 召回了等价条款，种子 ID 判负 | 检索没问题，是 `seed_chunk_id` 不全 |
| 高 | 低 | — | 种子块召回了但答案撑不住 | 切片切碎了，或装配把关键段截断了 |
| 高 | 高 | 低 | 召回对，但上下文注水 | 装配：去重、合并上限、`top_k` |
| 高 | 高 | 高 | 检索没问题 | 答复仍然错的话，问题在生成阶段 |

第二行把「检索失败」和「标注不全」区分开。只报 Recall 的话这两种情况长得一样，会导致朝着拟合标注去调参。

---

## 八、跑批时机

| 改了什么 | 要不要跑 r1 |
|---|---|
| 检索参数（`RRF_K`、重排权重、`MIN_SCORE`、`top_k`） | 要，这是主口径 |
| 切片参数、块头、BM25 分析器 | 要，且要先确认 `chunk_id` 没漂 |
| 政策文档改版、重新灌库 | 要，跑之前先过数据集自检 |
| Agent 提示词、工具描述 | 不用，链路没变 |

调参时只看 r1：一次跑批不经过 Agent，快，每一档指标直接指向一个文件。

判分口径变了就开新的实验目录，不要就地改 `rag-ex-1` —— 就地改会让历史 run 的分数不再可比，而版本对比正是这套指标的主要用途。期望值变了（改 `cases.jsonl` 的口径）则开 r2。
