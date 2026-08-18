# 5 · 检索评测：指标、跑批与结果

用 [4 · 检索评测数据集](https://tiltwind.github.io/refund-agent/doc/get-start/4-rag-dataset.md)的 `r1` 评[六步检索链路](https://tiltwind.github.io/refund-agent/doc/get-start/3-rag-impl.md#七第-6-步--检索链路-ragretrieving)。被测对象是 `search_policy` 及其内部各步，不含 Agent。实验目录 `rag/experiments/rag-ex-1/`。

一次全量 102 条，`recall@1 / @3 / @10` = 0.625 / 0.760 / 0.885，`Context Recall` 0.915、
`Context Relevance` 0.238，跑出四个待处理的问题，见第五、六节。

---

## 一、三个指标的分工

只报一个数字就没有归因能力。三个指标各自回答一个问题：

| 指标 | 问题 | 真值 | 调模型 | 现状 |
|---|---|---|---|---|
| Recall@k | 该召回的块，召回了吗 | `seed_chunk_id`，集合比对 | 否 | 门禁 |
| Context Recall | 检回的上下文，撑得住参考答案吗 | 参考答案拆成 claim，逐条判 | 是 | 观察，`--judge` 开 |
| Context Relevance | 检回的东西，有多少是真有用的 | 无需标注 | 是 | 观察，`--judge` 开 |

实测：`recall@3` 判负的 23 条里，12 条的 Context Recall 是 1.0 —— 召回的是等价条款，
判负的是标注（第七节）。

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

实验目录里 `run_experiment.py` 跑批、`scorers.py` 是纯函数打分器（`recall_at_k` 只做集合比对）、`judge.py` 是两个 LLM 指标、`export_traces.py` 与 `report.py` 出留档。

前置：Milvus 起着并已灌库，数据集自检过；要算两个 LLM 指标还得先把参考答案拆成 claim。

```bash
bash scripts/milvus.sh start && python rag/index/seed_milvus.py
python rag/evals/validate_cases.py
python rag/evals/generate_claims.py        # 只有 --judge 要，增量的，跑过就不再花钱
```

两条路径，判分逻辑共用，落同一份 `result.json`：

```bash
# 离线：样本读 cases.jsonl，不连 Langfuse
python rag/experiments/rag-ex-1/run_experiment.py

# 加两个 LLM 指标：每条多两次 judge 调用
python rag/experiments/rag-ex-1/run_experiment.py --judge --concurrency 8

# dataset run：样本从 Langfuse 数据集拉，分数写回，六步上报 trace
python rag/evals/push_dataset.py
python rag/experiments/rag-ex-1/run_experiment.py --langfuse --judge --run-name baseline-4
```

`--judge` 默认关：调参迭代要的是三档 Recall，每跑一轮多付 200 多次 judge 调用不划算，而且那两个数不进门禁。出报告、做版本留档时再带上。

`--concurrency` 调高并行的是网络调用（改写、judge）。嵌入和重排的前向被一把全局锁串起来 —— PyTorch 的 MPS 后端不是线程安全的，并发跑 Metal kernel 会让进程段错误退出（实测并发 5 崩在 `exec_unary_kernel`）。串行不慢：本来就是同一块 GPU 在抢算力。

| 路径 | 用在哪 |
|---|---|
| 默认（离线） | 调参。三档 Recall 是要进门禁的数，不该被一个本地 Langfuse 实例的死活卡住；脚本会把 `REFUND_AGENT_RAG_SPAN` 关掉，免得堆一批不挂在任何 dataset run 上的 trace |
| `--langfuse` | 留档与版本对比。run 页上能并排看两次跑批，逐条 `trace_id` 落进结果文件 |

102 条全量，离线约 75s，`--langfuse` 约 210s，加 `--judge` 是 2940s —— 慢一个量级半，而且慢的全在 judge 上：

| 路径 | 耗时 | 其中检索 |
|---|---|---|
| 离线 | 75.6s | 全部 |
| `--langfuse` | 206s | 大部分，差额是上报开销 |
| `--langfuse --judge` | 2940s | 224s（7.6%） |

judge 开着思考模式，这是 7.2 那个取舍的代价：实测一次 Context Relevance 调用（104 个内容单元）输出 21168 token，其中 20963 是思考 token，占 99%。两个指标的性价比不同 —— Context Recall 只判 4~6 条 claim，Context Relevance 要逐个判上百个单元，思考量随单元数线性涨。跑批时每条一行：

```
  [ 46/102] R1-046 2.7s · @1/@3/@10=001 · 2604 token · CR=0.0 · CRel=0.812
  [ 47/102] R1-047 2.1s · @1/@3/@10=111 · 1708 token · CR=0.75 · CRel=0.23
```

`@1/@3/@10=001` 是三档的命中位，`-` 表示这一档不适用（`multi_hop` 没有 `recall@1`）。行首那个秒数只是检索的，不含 judge。

跑完的产物：

| 产物 | 作用 |
|---|---|
| `result.json` | run 级与逐条指标，报告只读它，不依赖 Langfuse 在线 |
| `baseline-report.md` | 结论与归因 |
| `rag-ex-1-report.html` | 同一批数据的可视化，`python rag/experiments/rag-ex-1/report.py` 生成 |
| `traces/` | 现场记录，`python rag/experiments/rag-ex-1/export_traces.py --all` 全量导出 |

`export_traces.py` 默认按「报告里每个结论各留一份现场」分桶抽 20 条，不是随机 —— 随机 20 条里大概率一条空证据都没有。`--all` 是同一套分组不截断，102 条全导（约 7.6MB）：报告的分档表指到的是一批用例，只留抽样的话点进去多半没有对应那一条。带过 `--judge` 的话，每条现场的末尾还有一段 judge 判定 —— 逐条 claim 支不支撑、哪些内容单元被判为不相关，就跟在装配那步的条款全文后面，对着看的。

---

## 五、跑批结果

run `baseline-4`，102 条样本、96 条计分（6 条 `unanswerable` 不进 Recall 均值），2940s，无执行失败，judge 调用 0 次失败。

| 指标 | 值 | 分母 |
|---|---|---|
| `recall@1` | 0.625 | 80（不含 multi_hop） |
| `recall@3` | 0.760 | 96 |
| `recall@10` | 0.885 | 96 |
| `evidence_tokens` | 1229 | 预算 3000 |
| `duplicate_ratio` | 0.206 | 96 条中 57 条含重复正文 |
| `context_recall` | 0.915 | 96 |
| `context_relevance` | 0.238 | 92（4 条空证据没有分母） |

分档（`unanswerable` 除外）：

| 分档 | n | `recall@1` | `recall@3` | `recall@10` | `CR` | `CRel` |
|---|---|---|---|---|---|---|
| `formal` | 48 | 0.700 | 0.896 | 0.917 | 0.995 | 0.196 |
| `colloquial` | 48 | 0.550 | 0.625 | 0.854 | 0.835 | 0.284 |
| `text` | 62 | 0.714 | 0.871 | 0.952 | 0.948 | 0.223 |
| `table` | 26 | 0.417 | 0.615 | 0.769 | 0.827 | 0.262 |
| `table+text` | 8 | — | 0.375 | 0.750 | 0.950 | 0.270 |
| `single` | 80 | 0.625 | 0.825 | 0.912 | 0.906 | 0.239 |
| `multi_hop` | 16 | — | 0.438 | 0.750 | 0.959 | 0.233 |

按 3.1 的表读：口语档比书面档低一大截，差距在 `@10` 上收窄 —— 召回层捞得到，是排序吃字面匹配，责任在改写与稠密一路。表格块低于正文块，那是切片的事。

`multi_hop` 的 `CR` 0.959 高于 `single`：两个种子块只进来一个，另一半信息常由同一小节的邻近条款带进上下文。ID 级判负、内容够用，这一档是第七节四象限第二行的主力。

`recall@10` 丢的 11 条分两类：5 条种子块根本没进候选（救不回来，要看切片和过滤），6 条排在候选第 11~17（落在 `CANDIDATE_LIMIT = 20` 内，重排还有机会）。`@10` 命中而 `@3` 丢的 16 条里，6 条的种子块被 `MIN_SCORE` 整批滤掉，其余是名次被压 —— 候选第 2 掉到证据第 13 这种。

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

### 6.2 链路给不出「无适用条款」这个结论

6 条 `unanswerable` 全部没抛异常。异常只在候选为空时触发，而召回层对任何 query 都能捞回 20 条。实际发生的是：

| | 条数 |
|---|---|
| 重排全滤，返回空证据 | 5 |
| 返回不相关的条款（`context_relevance = 0.000`） | 1 |

`MIN_SCORE` 在这类 query 上多半已经把候选全砍掉了，所以缺的不是「不硬凑证据」，而是链路说不出「语料里没有适用条款」：它返回的空列表，与 6.1 那种「有正确条款却被阈值误杀」返回的空列表是同一个值，调用方无从区分。

两条路：给链路加「最高分低于下限即判无适用条款」，或者把这类样本的判据改成 Context Relevance —— 后者现在有实测支撑，那唯一一条有证据的样本判出 0.000，正常样本均值 0.238。

### 6.3 重复正文占 20.6%

96 条里 57 条含重复，最高一条 75%。成因是第二节说的分组键。`recall@3` 满分的用例里照样有 0.4 以上的重复率，ID 级 Recall 对它无感 —— 这是配套记辅助数的理由。

Context Relevance 也抓不到它：重复率 ≥ 0.4 的 18 条 `CRel` 均值 0.326，比无重复的 35 条（0.245）还高。被拼两遍的正是命中的那一小节，两份拷贝都判「相关」。`duplicate_ratio` 因此不能由 `CRel` 取代。

### 6.4 链路非确定性与门禁容差

同一套参数跑五次，`@1` 抖 1 条（±0.012），`@3` 抖 1 条（±0.010），`@10` 抖 2 条（±0.021）。翻转的都是种子块在候选里排 11~17 名、来回跨 `k=10` 的用例。来源是改写那一步调模型，温度 0 也不保证网关每次返回同一份拆分；`baseline-4` 还出现了 1 次改写降级（模型返回的 `intent` 不在字面量里，pydantic 校验失败，整条 query 原文透传）。

`@1` 前四次纹丝不动、第五次翻了一条 —— 某一档的稳定性不能靠四个点下结论。

打分器是纯函数，被测链路不是。门禁口径因此用相对值：`seed_chunk_id` 是下界，定一个 `recall@3 ≥ 0.9` 之类的绝对阈值没有依据；改参数后重跑，与上一版 `result.json` 对比，任一档下降超过容差即不通过。容差要盖过上面这个抖动幅度，或者先把改写的结果按 query 缓存下来。

---

## 七、两个 LLM 指标

两个都要 LLM judge，判分挂在 `PolicySection.text` 上，实现在 `judge.py`（与纯函数的 `scorers.py` 分开放）。

```
Context Recall    = 被检回上下文支撑的 claim 数 / 参考答案中的 claim 总数
Context Relevance = 与 query 相关的内容单元数 / 检回的内容单元总数
```

### 7.1 claim 跟着样本走

Context Recall 用参考答案当代理，不需要标注 `reference_contexts`：先把 `reference_answer` 拆成 claim，再逐条判有没有被上下文支撑。

拆分**不在跑批时做**，claim 是 `cases.jsonl` 里的一个字段，跟 query 和参考答案同一行：

```jsonc
{
  "case_id": "R1-001",
  "query": "…哪些商品属于法定不适用七天无理由退货的情形？",
  "reference_answer": "有四类商品不适用无理由退货：消费者定作的、鲜活易腐的…",
  "claims": ["消费者定作的商品不适用无理由退货。", "鲜活易腐的商品不适用无理由退货。", "…"]
}
```

| 为什么提前拆 | 现拆会怎样 |
|---|---|
| 拆 claim 与被测链路无关，只依赖 `reference_answer` | 每改一版检索参数就重拆 96 条，花的钱全是重复的 |
| 分母必须稳定 | 同一条答案这次拆 4 条、下次拆 5 条，两次 run 的 Context Recall 不可比 —— 而版本对比正是它的用途。温度 0 挡不住这个抖动，改写那一步已经实测到（6.4） |
| claim 要能人工核对 | 参考答案本身要过人工抽检（[4 · 8.2](https://tiltwind.github.io/refund-agent/doc/get-start/4-rag-dataset.md#82-人工抽检)），claim 跟它并排放才看得出拆得对不对 |

放进 `cases.jsonl` 而不是另起一个文件：`query` 和 `reference_answer` 本来就是模型生成的，claim 不比它们更「派生」，分开放只是多一处要对齐的东西。同一行还带来两个好处 —— [`generate_cases.py`](https://github.com/tiltwind/refund-agent/blob/main/rag/evals/generate_cases.py) 生成参考答案时一并出 claim，不用为此多调一轮模型；`push_dataset.py` 把它塞进 `expected_output` 推上 Langfuse，dataset run 判分时读 item 自己带的那份，不回头读本地文件。

`formal` 与 `colloquial` 是同一个种子块的两种问法，参考答案逐字相同，一次生成、两条样本共用一份 claim。分母一致，语域分档表比的才只是检索质量。

已经生成好的样本用 [`generate_claims.py`](https://github.com/tiltwind/refund-agent/blob/main/rag/evals/generate_claims.py) 补 —— 重跑 `generate_cases.py` 会连 query 一起重写，措辞漂了历史分数就不可比。它只补缺的那些，补出来的行记一个 `meta.claims_by`（补拆走的是 judge 模型，与生成样本的模型不是同一个）。

改了参考答案而没重拆 claim，分母还是老的、分数看着正常，这是最难发现的一类错。自检里那条「数字可溯源」因此扩到 claim 上：claim 里的数字必须在参考答案里出现过。

### 7.2 judge 的配置与四条约束

judge 模型从 `.env` 取，与链路的[改写模型](https://github.com/tiltwind/refund-agent/blob/main/rag/retrieving/pipeline/rewrite.py)分开配置 —— 解析规则复用 [`llm/chat.py`](https://github.com/tiltwind/refund-agent/blob/main/llm/chat.py) 的角色机制，多一个 `judge` 角色：

```bash
OPENAI_JUDGE_MODEL="deepseek-v4-pro"          # 或 ANTHROPIC_JUDGE_MODEL
OPENAI_JUDGE_BASE_URL=""                      # 不填则跟通用的 OPENAI_BASE_URL 走
OPENAI_JUDGE_API_KEY=""                       # 同上
REFUND_AGENT_JUDGE_MODEL=""                   # 跨供应商的强制覆盖
REFUND_AGENT_JUDGE_STRUCTURED="json_mode"     # 见下
```

端点和 key 跟模型名同一套角色前缀，角色专属的优先、没配就回落到通用的那组。分开配的理由是 judge 常挂在另一个网关上，而换端点通常要换 key。

judge 该比主模型强一档：改写判错顶多让排序掉一档，judge 判错整个指标就是噪声。没单独配时它回落到主模型，等于自己评自己，跑批会打一行警告。

结构化输出这里踩的坑与改写那边同源，但结论相反。改写走 `function_calling` + 关思考；judge 走 `json_mode` + **留着思考** —— DeepSeek 的 thinking 与 `function_calling` 要设的 `tool_choice` 冲突（400 `Thinking mode does not support this tool_choice`），二者只能取一个，而判定 claim 正是该让模型多想一步的活。代价是 `json_mode` 只保证「输出是合法 JSON」，langchain 不会把 schema 传给模型，字段名得自己写进提示词，否则模型自由发挥（实测吐过 `{"useful": [...]}`）解析直接失败；那段 schema 由 pydantic 模型现生成，不手写。

四条判定约束：温度 0 且结构化输出；不确定时判「不支撑」；只判「上下文里有没有」，不判「答案对不对」；judge 与改写分开配。每条 claim 的判定理由落进 `result.json` 的 `judge` 字段 —— 一个 0.6 分说明不了改哪里，要能翻到是哪条 claim 没撑住。

judge 调用失败**不写分数**，那一条从均值里少掉，失败清单进 `summary.judge_errors` 并在跑批末尾打出来。写个 0 会被均值当成「检索没召回」，把网关抖动记到检索头上。

### 7.3 Context Relevance 的内容单元取句子

一条 `PolicySection` 是回填后的完整小节，里面必然混有不相关的句子，那是父块回填故意带进来的（[3 · 五](https://tiltwind.github.io/refund-agent/doc/get-start/3-rag-impl.md#五第-4-步--切片-ragchunking)）。按块判会把这个设计判成缺陷，分数一律接近 1。所以按句切，表格行整行作一个单元（`| 已拆封 | 不支持 |` 里没有句号，按句切只会切出没有主语的碎片）。

它的绝对值不该追求高（`baseline-4` 实测 92 条：最低 0.019、中位 0.193、均值 0.238、最高 1.0），用途是横向对比：`top_k` 调大时 Context Recall 上升、它下降，两个一起看才知道是净赚还是净亏。`unanswerable` 样本也算这个数 —— 那 6 条没有 claim，但 6.2 说的那件事正好由它量化：其中 5 条是空证据没有分母，剩下一条判出 0.000。它单列在 `summary.unanswerable_relevance`。

它抓不到的是重复正文（6.3）：被拼两遍的是命中的那一小节，两份拷贝都判「相关」。

### 7.4 三个指标联合读

| Recall@3 | Context Recall | Context Relevance | 判断 | 动哪里 |
|---|---|---|---|---|
| 低 | 低 | — | 真的没召回 | 召回层，先看 `recall@10` 分档 |
| 低 | 高 | — | 召回了等价条款，种子 ID 判负 | 检索没问题，是 `seed_chunk_id` 不全 |
| 高 | 低 | — | 种子块召回了但答案撑不住 | 切片切碎了，或装配把关键段截断了 |
| 高 | 高 | 低 | 召回对，但上下文注水 | 装配：去重、合并上限、`top_k` |
| 高 | 高 | 高 | 检索没问题 | 答复仍然错的话，问题在生成阶段 |

第二行把「检索失败」和「标注不全」区分开。只报 Recall 的话这两种情况长得一样，会导致朝着拟合标注去调参。`baseline-4` 落在这四个格子里的分布（`CR` 以 0.8 分界）：

| 象限 | n | 占比 |
|---|---|---|
| `@3` 判负 · `CR` 低 | 11 | 11% |
| `@3` 判负 · `CR` 高 | 12 | 12% |
| `@3` 命中 · `CR` 低 | 0 | 0% |
| `@3` 命中 · `CR` 高 | 73 | 76% |

`@3` 判负的 23 条里过半落在第二行，`CR` 全是 1.0：要补的是 `seed_chunk_id`，不是重排权重。

### 7.5 还没做：judge 校准

两个指标现在**只记录、进报告，不参与 pass / fail**。同一批 10 条样本连跑两遍，`context_recall` 有一条从 0.833 翻到 1.0，`context_relevance` 每条都在动（0.163→0.234、0.615→0.567、0.376→0.282）。这里面既有链路自身的非确定性（改写那一步调模型，6.4），也有 judge 的噪声，两者从跑批结果上分不开 —— 这正是校准要单独做的事：把上下文固定住，只让 judge 重复判定。

`baseline-4` 的分布本身就是要校准的证据：96 条里 85 条 `context_recall` 判满分，`@3` 命中的用例没有一条低于 0.8（上表第三行是 0）。判负的 11 条里 4 条是空证据、分子必然为 0，真正「有上下文但撑不住」的只有几条。

那几条里 R1-046 是另一类问题：`CR = 0`，两条 claim 说「拆封不支持无理由退货」，检回的正文写的是「查验性拆封不影响商品完好、可以退货」。judge 判的是「上下文里有没有」，判负没错，暴露的是参考答案与语料相互矛盾 —— 数据集的事，不是检索的事。这类只有逐条判定理由在场才看得出来。

校准之前它们的结论不采信：要拿 [4 · 8.2](https://tiltwind.github.io/refund-agent/doc/get-start/4-rag-dataset.md#82-人工抽检) 的人工抽检样本算 judge 与人工标注的一致率、假阳假阴分布、同一批样本判两遍的自一致性。一致率不达标就修提示词或换 judge 模型。

已经量到的第一个影响因素是思考档位。同一批 8 条样本、同一份上下文，只把 `REFUND_AGENT_JUDGE_REASONING` 从模型默认改成 `none`：

| | 开思考 | 关思考 |
|---|---|---|
| 8 条 16 次判定耗时 | 约 1400s | 32.8s |
| `Context Recall` 一致 | 6/8，两条变松（0.0→1.0、0.75→1.0） | |
| `Context Relevance` | 0.034 / 0.049 / 0.812 | 0.793 / 0.659 / 0.368 |

Context Recall 尚能对上大半，Context Relevance 两套数之间看不出对应关系 —— 逐个判上百个内容单元本来就是它最吃推理的地方。所以换的不只是速度，是判定口径：`config.judge_model` 之外，`config.judge_reasoning` 与 `config.judge_structured` 也一并记进每一次 run 的结果文件，否则两次跑批的 CR / CRel 会被当成可比的数。

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
