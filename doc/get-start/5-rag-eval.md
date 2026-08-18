# 5 · 检索评测：三个指标与归因

用 [4 · 检索评测数据集](https://tiltwind.github.io/refund-agent/doc/get-start/4-rag-dataset.md)的 `r1` 评[六步检索链路](https://tiltwind.github.io/refund-agent/doc/get-start/3-rag-impl.md#七第-6-步--检索链路-ragretrieving)。被测对象是 `search_policy` 及其内部各步，不含 Agent。

> 三档 Recall 已实现，在 `rag/experiments/rag-ex-1/`；两个 LLM 指标待建，见[第九节](#九待建产物)。

---

## 一、为什么是三个指标

只报一个数字就没有归因能力。三个指标各自回答一个不同的问题，任意两个都不能替代第三个：

| 指标 | 问题 | 真值 | 调模型 |
|---|---|---|---|
| Recall@k | 该召回的块，召回了吗 | `seed_chunk_id`，集合比对 | 否 |
| Context Recall | 检回的上下文，撑得住参考答案吗 | 参考答案拆成 claim，逐条判 | 是 |
| Context Relevance | 检回的东西，有多少是真有用的 | 无需标注 | 是 |

Recall 不够用，因为 `seed_chunk_id` 只是种子块，不是全部相关块（[4 · 5.3](https://tiltwind.github.io/refund-agent/doc/get-start/4-rag-dataset.md#53-种子块不等于全部相关块)）。链路召回了 P07 而不是种子块 P02，两条讲的是同一个窗口规则，ID 级判它错，实际够用。Context Recall 用「标答案」替代「标全部相关文档」，覆盖的正是这段标不到的长尾。

Context Relevance 补的是另一个方向：前两个指标都在问「捞得够不够全」，把 `DEFAULT_TOP_K` 从 4 改成 20，两个分数一起变好看。没有一个惩罚「多召回」的指标，调参会朝着往上下文里塞垃圾的方向走，而 `TOKEN_BUDGET = 3000` 是硬上限，塞进去的垃圾会把真正的证据挤掉。

三者一起用，分开报。合成一个加权总分会把归因能力丢掉。

---

## 二、指标挂在链路的哪个位置

链路后半段有两种粒度的产物：

```
… → 4 召回融合 → Candidate（chunk 级，有 chunk_id）
       → 5 重排  → Evidence（chunk 级，有 chunk_id + 分数）   ← Recall@k 挂这里
              → 6 装配 → PolicySection（父块级，无 chunk_id） ← 两个 LLM 指标挂这里
                            └─ 去重 → 相邻合并 → 父块回填 → 预算截断
```

[`PolicySection`](https://github.com/tiltwind/refund-agent/blob/main/rag/retrieving/protocol.py) 上没有 `chunk_id`：装配把命中的子块还原成完整小节，还可能合并相邻父块，一条 `PolicySection` 对应的是一组子块。所以：

| 指标 | 取什么 | 理由 |
|---|---|---|
| Recall@k | [`rerank`](https://github.com/tiltwind/refund-agent/blob/main/rag/retrieving/pipeline/rerank.py) 输出的 `Evidence` 序列的 `chunk_id` | 只有这一层有 ID。它诊断的是检索器 |
| Context Recall / Relevance | `search_policy` 返回的 `PolicySection.text` | 这是真正注入模型上下文的东西。它诊断的是交付给模型的证据 |

分开挂的原因是装配阶段自己会出错。[`assemble`](https://github.com/tiltwind/refund-agent/blob/main/rag/retrieving/pipeline/assemble.py) 按 `(parent_seq, parent_id, section_path)` 分组回填父块，而同一父块的子块 `section_path` 各不相同，同一个 `parent_id` 因此被登记多次，父块正文被重复拼进上下文。这类缺陷 chunk 级 Recall 看不见：种子块召回了，Recall 满分，但注入模型的上下文里有一大截是重复正文。只有落在装配产物上的指标才抓得住。

评测脚本因此需要拿到中间产物。`search_policy` 当前只返回 `PolicySection`，实现时有两个选择：

| 做法 | 代价 |
|---|---|
| 评测脚本自己按 `rewrite → route → recall → rerank → assemble` 编排一遍 | 复制了 [`milvus.py`](https://github.com/tiltwind/refund-agent/blob/main/rag/retrieving/milvus.py) 的编排逻辑，容易和线上跑偏 |
| 给 `search_policy` 加一个返回 trace 的旁路 | 多一个入口方法；换来评测和线上跑的是同一段代码 |

取第二种，与 `2-design 3.4` 的口径一致：prod 与 eval 走同一条检索路径。落地成 `search_with_trace`：编排逻辑在它里面，`search_policy` 调它并丢掉 trace，线上调用方一行不用改。[`RetrievalTrace`](https://github.com/tiltwind/refund-agent/blob/main/rag/retrieving/protocol.py) 已经在记每一步的中间产物，两个 ID 序列 `candidate_ids` / `evidence_ids` 结构化加在它上面 —— 那些人读的 `steps` 字符串也含 ID，但正则去抠一句日志，措辞改一个字打分器就静默判负。

---

## 三、Recall@k：ID 级，纯函数

### 3.1 报三档，不报一个数

```
Recall@k = |命中的种子块| / |全部种子块|
```

`single` 类样本分母是 1，`multi_hop` 分母是 2 且要求全中（[4 · 5.4](https://tiltwind.github.io/refund-agent/doc/get-start/4-rag-dataset.md#54-单块答不全的问题)）。k 取三个值，各自对应链路的一个位置：

| 指标 | k | 读什么 |
|---|---|---|
| `recall@10` | 召回融合后的候选，`CANDIDATE_LIMIT = 20` 之内 | 召回层的上界。这里没有的，后面救不回来 |
| `recall@3` | 重排后前 3 | 实际交付水位。`DEFAULT_TOP_K = 4`，取 3 留一格余量 |
| `recall@1` | 重排后第 1 | 头部精度，对 `MIN_SCORE = 0.30` 这类阈值最敏感 |

k 小于种子块数的档次不计分。两个种子块不可能同时排第 1，`multi_hop` 的 `recall@1` 是结构性的 0，算进均值等于按 multi_hop 的占比给这一档加了个固定折扣 —— 改参数动不了它，版本间对比也读不出东西。r1 的 16 条 multi_hop 因此只进 `@3` 与 `@10`，三档的分母各自记在结果文件里。

### 3.2 归因就在两档之差

| 现象 | 结论 | 改哪里 |
|---|---|---|
| `recall@10` 低 | 召回层就没捞到 | 切片、块头、BM25 分析器、`CANDIDATE_LIMIT`、[过滤条件](https://github.com/tiltwind/refund-agent/blob/main/rag/retrieving/pipeline/filters.py)是不是把它滤掉了 |
| `recall@10` 高、`recall@3` 低 | 捞到了但被压下去 | [重排](https://github.com/tiltwind/refund-agent/blob/main/rag/retrieving/pipeline/rerank.py)：`RELEVANCE_WEIGHT` / `PRIOR_WEIGHT`、`DOC_PRIOR` 的 P02 偏置 |
| `recall@3` 高、`recall@1` 低 | 头部排序不稳 | `RRF_K`、tie-break |
| `formal` 档高、`colloquial` 档低 | 链路吃字面匹配 | 改写环节、稠密一路 |
| `law` 层显著低于 `platform` 层 | 路由或降权过头 | `LAW_K = 5`、法规层 `0.5` 降权 |

一个混合的「检索命中率」给不出其中任何一行，它只说明结果不好，不说明改哪个文件。

### 3.3 它是纯函数，所以它做门禁

集合比对，不调模型，同一份数据集跑两遍结果一致。能进门禁的指标必须是纯函数：门禁要给出明确的 pass / fail，一个每次跑都飘几个点的数字没法当发布依据。

门禁口径用相对值而不是绝对值。`seed_chunk_id` 是下界，绝对值本来就偏低，定一个 `recall@3 ≥ 0.9` 之类的阈值没有依据。改参数后重跑 r1，与上一版对比，任一档下降超过容差即不通过。容差在攒够几个基线 run 之后再定，第一版只记录。

---

## 四、Context Recall：LLM as judge

Ragas 的做法：不需要 `reference_contexts`，用参考答案当代理。

```
Context Recall = 被检回上下文支撑的 claim 数 / 参考答案中的 claim 总数
```

两步，各一次模型调用：

| 步骤 | 输入 | 输出 |
|---|---|---|
| 拆 claim | `reference_answer` | 若干条不可再分的陈述 |
| 逐条判定 | 每条 claim + 全部 `PolicySection.text` | 支撑 / 不支撑 + 一句理由 |

拆 claim 这一步与被测链路无关，结果要缓存。同一条样本的参考答案不变，claim 就不变，每次评测重拆是白花钱，而且会引入本不该有的波动。缓存键用 `case_id` + 参考答案的哈希。

### 4.1 判定环节的四条约束

| 约束 | 为什么 |
|---|---|
| 温度 0，判定结果结构化输出 | 自由文本要解析，解析失败就成了静默的 0 分 |
| 判定的默认值偏向「不支撑」 | 判定模型天然倾向于说「是」。不确定时判负，宁可低估 |
| 只判「上下文里有没有」，不判「答案对不对」 | 判定滑向事实核查，测的就不是检索了 |
| judge 模型与被测链路的[改写模型](https://github.com/tiltwind/refund-agent/blob/main/rag/retrieving/pipeline/rewrite.py)分开配置 | 同一个模型既参与检索又给检索打分，偏差没法排除 |

每条 claim 的判定理由要落盘。Context Recall 掉了以后，要能直接看到是哪条 claim 没被支撑，这是它相对于 ID 级 Recall 多出来的信息：它指得出缺的是哪句话。

### 4.2 它为什么不做门禁

调模型，有噪声，同一份输入跑两次可能不同。它归入观察指标：记录分数、进报告、看趋势，不参与 pass / fail。

降噪可以做，但收益有限：同一条判定跑 3 次取多数，成本乘 3，方差降一些，不会让它变成确定性指标。第一版只跑一次，把预算花在样本量上而不是重复判定上。

---

## 五、Context Relevance

```
Context Relevance = 与 query 相关的内容单元数 / 检回的内容单元总数
```

「内容单元」取句子而不是取 `PolicySection`。一条 `PolicySection` 是回填后的完整小节，里面必然混有不相关的句子，那是父块回填故意带进来的上下文（[3 · 五](https://tiltwind.github.io/refund-agent/doc/get-start/3-rag-impl.md#五第-4-步--切片-ragchunking)）。按块判会把这个设计判成缺陷。

这个指标的绝对值不该追求高，回填本身就会拉低它。它的用途是横向对比和防退化：

| 场景 | 期望看到 |
|---|---|
| `DEFAULT_TOP_K` 从 4 调到 8 | Recall 上升，Context Relevance 下降。两个一起看才知道这次调参是净赚还是净亏 |
| 装配把同一父块拼了两遍 | 重复的句子重复计入分母，指标下降 |
| `MERGE_MAX_PARENTS` 放宽 | 相邻合并拖进更多不相关条文，指标下降 |

配套记两个不调模型的辅助数：证据 token 用量（`TOKEN_BUDGET = 3000` 的实际占用）和重复正文占比。后者是纯字符串比对就能算的，装配重复拼接这类缺陷它零成本就能抓住，不必等人工去读 trace。

---

## 六、三个指标怎么联合读

| Recall@3 | Context Recall | Context Relevance | 判断 | 动哪里 |
|---|---|---|---|---|
| 低 | 低 | — | 真的没召回 | 召回层，先看 `recall@10` 分档 |
| 低 | 高 | — | 召回了等价条款，种子 ID 判负 | 检索没问题，是 `seed_chunk_id` 不全，属正常现象 |
| 高 | 低 | — | 种子块召回了但答案撑不住 | 切片切碎了，或装配把关键段截断了 |
| 高 | 高 | 低 | 召回对，但上下文注水 | 装配：去重、合并上限、`top_k` |
| 高 | 高 | 高 | 检索没问题 | 答复仍然错的话，问题在生成阶段，不在这条链路上 |

第二行把「检索失败」和「标注不全」区分开了。只报 Recall 的话，这两种情况长得一样，会导致朝着拟合标注去调参。

---

## 七、judge 自身也要被评估

用 LLM 判分，就得先知道它判得准不准，否则第四、五节两个指标只是噪声。

[4 · 8.2](https://tiltwind.github.io/refund-agent/doc/get-start/4-rag-dataset.md#82-人工抽检) 的人工抽检样本在这里第二次派上用场：这批样本人工判过每条 claim 支不支撑，拿它当基准，算 judge 与人工的一致率。

| 检查 | 口径 |
|---|---|
| 一致率 | judge 判定与人工标注相同的比例 |
| 偏向 | 假阳（该判不支撑却判了支撑）与假阴各占多少。偏向不对称说明提示词有倾向 |
| 自一致性 | 同一批样本判两遍，两次结果的一致比例 |

一致率不达标时先修提示词或换 judge 模型，不去调 Context Recall 的分数；修好之前这两个指标的结论不采信。这个校准结果要跟着数据集版本记录，换了 judge 模型历史分数就不可比，和换切片参数要开 r2 是一个道理。

---

## 八、跑与聚合

### 8.1 什么时候跑

r1 是改检索就跑的那一套，与 Agent 端到端回归互不干扰：

| 改了什么 | 要不要跑 r1 |
|---|---|
| 检索参数（`RRF_K`、重排权重、`MIN_SCORE`、`top_k`） | 要，这是主口径。端到端评测在这类改动上给不出归因 |
| 切片参数、块头、BM25 分析器 | 要，且要先确认 `chunk_id` 没漂 |
| 政策文档改版、重新灌库 | 要，跑之前先过数据集自检 |
| Agent 提示词、工具描述 | 不用，链路没变，分数不会动 |

调参时只看 r1：一次跑批不经过 Agent，快，而且每一档指标直接指向一个文件。参数定下来之后再跑一次端到端，确认改动传导到答复上是净收益。

### 8.2 聚合与门禁

| 层级 | 口径 |
|---|---|
| 样本 | 三个指标各出一个分。`unanswerable` 类只判「是否按预期抛异常」，不进前三个指标的均值 |
| 分档 | 按 `style`、`layer`、`kind`、`doc_id` 各切一份均值。这些切片才是可执行的信息，总均值只用来看趋势 |
| run | `recall@1/@3/@10` 三个硬指标，`context_recall`、`context_relevance` 两个观察指标，外加 token 用量与重复正文占比两个辅助数 |
| 门禁 | 只看三档 Recall 的版本间变化，见 3.3。两个 LLM 指标只记录 |

判分口径变了就开新的实验目录，不要就地改。就地改会让历史 run 的分数不再可比，而版本对比正是这套指标的主要用途。

---

## 九、待建产物

实验目录是 `rag/experiments/rag-ex-1/`，脚本、结果、报告、trace 留档都在目录内。判分口径变了开新目录，见 8.2。

| 产物 | 位置 | 作用 |
|---|---|---|
| 实验脚本 | `rag/experiments/rag-ex-1/run_experiment.py` | 跑 r1、算三个指标、聚合落盘 |
| 打分器 | `rag/experiments/rag-ex-1/scorers.py` | `recall_at_k` 纯函数；`context_recall`、`context_relevance` 调 judge |
| judge 提示词 | `rag/experiments/rag-ex-1/prompts/` | 拆 claim、判支撑、判句子相关性 |
| judge 校准 | `rag/experiments/rag-ex-1/judge_calibration.md` | 第七节的一致率结果，不达标不采信指标 |
| 结果 | `rag/experiments/rag-ex-1/result.json` | run 级与逐条指标落盘，报告只读它，不依赖 Langfuse 在线 |
| trace 留档 | `rag/experiments/rag-ex-1/traces/` | 人工要逐条读的几条链路现场，导出成文件 |
| 链路旁路 | [`rag/retrieving/milvus.py`](https://github.com/tiltwind/refund-agent/blob/main/rag/retrieving/milvus.py) | 让 `search_policy` 能返回重排后的 `chunk_id` 序列，并把六步上报 Langfuse，见第二节与 9.1 |

### 9.1 Langfuse 承担哪一段

数据集要推上去：`python rag/evals/push_dataset.py` 得到 `retrieval-cases-r1`，一条 item 一个 query，`expected_output` 里的 `seed_chunk_id` 与 `reference_answer` 就是两个 Recall 打分器的输入。`run_experiment.py --langfuse` 时样本从这份数据集拉，跑批走 `client.run_experiment`，分数写回 dataset run 做版本对比。

六步各自成 span，挂在 `rag.search_policy` 这个 retriever 节点下，随 item 的 trace 一起上报。埋点在 [`milvus.py`](https://github.com/tiltwind/refund-agent/blob/main/rag/retrieving/milvus.py) 里而不是在评测脚本里，因此线上和评测翻的是同一棵树 —— 这和 `2-design 3.4` 是同一条口径。候选与证据的 `chunk_id` **全序列**进 span 的 output：`recall@10` 判负时 UI 上直接看得到种子块排在第几，不必回头翻结果文件。judge 接进来之后，每条 claim 的判定也挂在这条 trace 上。

指标不以它为准。默认路径读 `rag/datasets/r1/cases.jsonl`，不连 Langfuse 也不上报（脚本会把 `REFUND_AGENT_RAG_SPAN` 关掉，免得堆一批不挂在任何 dataset run 上的 trace）。三档 Recall 是要进门禁的数，不该被一个本地实例的死活卡住。两条路径判分逻辑共用，落同一份 `result.json`。

### 9.2 实现顺序

前置是 Milvus 起着并已灌库，且 `python rag/evals/validate_cases.py` 通过。

| 步 | 做什么 | 完成的标志 |
|---|---|---|
| 1 | 第二节的链路旁路，把重排后的 `chunk_id` 序列结构化落进 `RetrievalTrace` | 单条 query 能取到 Evidence 序列 |
| 2 | `scorers.recall_at_k` 与离线跑批 | r1 全量跑出 `recall@1/@3/@10` 三档，写进 `result.json` |
| 3 | 接 Langfuse：推数据集，跑批写回 dataset run | run 页有分数，分档均值可查 |
| 4 | judge 提示词与第七节的校准 | 一致率达标，之后才接两个 LLM 指标 |
| 5 | 报告与 trace 留档 | 分档表 + 逐条归因 |

先做 `recall_at_k`：它不调模型，数据集一到位就能跑，也能立刻验证第 1 步的旁路改造对不对。第 1~3 步已落地，产物与第一份基线在 `rag/experiments/rag-ex-1/`。
