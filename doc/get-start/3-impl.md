# 3 · 实现：一步步搭出 RefundAgent v1

本文是**动手教程**：从一个空目录开始，按依赖顺序把 v1 搭出来，最终 `bash run-main.sh` 能跑通三个退款场景。
每一步都给出「建什么文件 → 关键取舍 → 怎么单独验证」，不必等整条链路成型才知道对不对。

业务口径（五步 SOP、判定规则、验收要求）见
[0 · 需求](https://github.com/tiltwind/refund-agent/blob/main/doc/get-start/0-requirement.md)；
组件构成与请求链路见
[1 · 架构](https://github.com/tiltwind/refund-agent/blob/main/doc/get-start/1-architecture.md)；
设计上的理由（为什么规则引擎不放 Agent、为什么身份不进工具 schema、为什么 RAG 不切数据源）在
[2 · 设计](https://github.com/tiltwind/refund-agent/blob/main/doc/get-start/2-design.md)，本文只在必要处引用结论。
完整代码：[tiltwind/refund-agent](https://github.com/tiltwind/refund-agent)。

---

## 一、先看终点

v1 是一个**离线可跑的完整链路**：没有 HTTP 服务外壳，客户档案与订单读本地 eval 数据，
政策检索连真实 Milvus，模型走 Anthropic 或任意 OpenAI 兼容网关。

```
用户消息
  └─ RefundContext（customer_id / request_id / request_source）—— 演示脚本直接构造，线上由认证中间件注入
     └─ Agent Loop（五步 SOP 写死在系统提示里）
        ├─ get_customer_info        → services/customer/eval.py   → evals/data/customers.json
        ├─ search_refund_policy     → services/rag/milvus.py      → Milvus（六步检索链路）
        ├─ check_refund_eligibility → services/order/eval.py      → 规则引擎副本
        └─ execute_refund / record_refund_denial → 落决策流水，返回单号
     └─ 答复（必须写明单号）
```

跑起来长这样（节选）：

```
模型: anthropic:claude-sonnet-5 → 官方端点（改写: anthropic:claude-haiku-4-5）
Langfuse: on → http://localhost:3000

======================================================================
场景：金牌会员的窗口期优待（预期：批准）
用户（C1001）：你好，订单 O2001 的耳机买回来一直没拆封，现在不想要了，想无理由退货。
----------------------------------------------------------------------
  ▶ get_customer_info({})
  ▶ search_refund_policy({'query': '金牌会员签收 10 天的耳机未拆封，无理由退货还在窗口期内吗？'})
  ▶ check_refund_eligibility({'order_id': 'O2001', 'reason_type': '无理由', 'item_condition': '未拆封'})
  ▶ execute_refund({'order_id': 'O2001', 'amount': 899.0, 'reason': '...'})

RefundAgent：您好，您的退款申请已通过审核……退款单号 R9000……
```

### 构建顺序

自底向上，每一层都能脱离上层单独验证：

| 步骤 | 建什么 | 能独立验证吗 |
|---|---|---|
| 1 | 环境与依赖 | ✅ import 得动就算过 |
| 2 | `doc/policy/` 政策语料 | ✅ frontmatter 解析 |
| 3 | `llm/` 模型层（嵌入 / 重排 / 对话） | ✅ 编码一句话、打一次分 |
| 4 | `knowledge/chunking/` 切片 | ✅ 打印块数与 token 分布 |
| 5 | Milvus + `knowledge/seed_milvus.py` 灌库 | ✅ collection 行数 |
| 6 | `services/rag/` 六步检索链路 | ✅ 单跑一次检索看 top-k |
| 7 | `app/context.py` + `services/` 业务接入层 | ✅ 直接调规则引擎打边界 |
| 8 | `agent/v1/` 提示 + 工具 + 装配 | ✅ 单跑一次 invoke |
| 9 | `main.py` / `run-main.sh` 入口 | ✅ 三个场景 |
| 10 | `services/telemetry.py` 埋点（可选） | ✅ Langfuse 上看到调用树 |

**顺序不能倒过来**：先写 Agent 再写检索，坏 case 出来时你分不清是提示词的问题还是检索的问题——
而这两者的修法完全不同。先把下层每一段各自验收掉，上层出问题时怀疑面才收得住。

---

## 二、前置条件

| 项 | 要求 | 说明 |
|---|---|---|
| Python | 3.10+ | `match` 语句与 `X \| None` 类型标注要用 |
| Docker | 任意近版 | 只用来跑 Milvus standalone |
| 磁盘 | ≈ 5 GB | BGE-M3（约 2.2 GB）+ bge-reranker-v2-m3（约 2.2 GB）权重 |
| 模型凭据 | 二选一 | `ANTHROPIC_API_KEY`，或 `OPENAI_API_KEY` + `OPENAI_BASE_URL` + `OPENAI_MODEL`（DeepSeek / Qwen / vLLM / one-api 都走这条） |
| Langfuse | 可选 | 不配则埋点静默关闭，链路照常跑 |

嵌入与重排**跑在本地**，不消耗 API 额度；CPU 也能跑，只是慢。Apple Silicon 自动走 mps，NVIDIA 走 cuda
（顺序固定在 [`llm/device.py`](https://github.com/tiltwind/refund-agent/blob/main/llm/device.py)，不做成配置项——
换设备不该成为「检索结果变了」的解释之一）。

---

## 三、第 1 步 · 环境与依赖

```bash
git clone https://github.com/tiltwind/refund-agent.git
cd refund-agent

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # 填入 ANTHROPIC_API_KEY 或 OPENAI_* 三件套
```

[`requirements.txt`](https://github.com/tiltwind/refund-agent/blob/main/requirements.txt) 只有六组依赖，每一组都有明确用途：

| 依赖 | 用途 | 备注 |
|---|---|---|
| `langchain` / `langgraph` | `create_agent` + Agent Loop | 1.x |
| `langchain-anthropic` / `langchain-openai` | 两家供应商 | 后者同时覆盖所有 OpenAI 兼容网关 |
| `pymilvus` | 向量库客户端 | **必须 2.5+**：稀疏向量与 BM25 Function 自该版本起可用 |
| `torch` / `transformers` | 本地嵌入与重排 | 首次运行会下载权重 |
| `pyyaml` | 政策文档 frontmatter、`meta.yaml` | |
| `langfuse` | 可观测 | 缺密钥时静默降级 |

注意**没有 jieba**：BM25 的分词与打分都在 Milvus 服务端，应用侧不自建倒排索引——
那份索引迟早会与 collection 漂移，而漂移不会报错。

密钥只放 `.env`（已在 [`.gitignore`](https://github.com/tiltwind/refund-agent/blob/main/.gitignore) 里），
`.env.example` 只放占位符。

**验证**：

```bash
python -c "import langchain, langgraph, pymilvus, torch, transformers, yaml; print('deps ok')"
```

---

## 四、第 2 步 · 政策语料 `doc/policy/`

Agent 要引用条款答复用户，先得有条款。这里的关键决定是：
**语料源就是 Markdown 文档本身，没有中间产物**——不要再手抄一份 `policies.json`，那等于给同一套政策留两份事实源，
文档改了 JSON 没改（或反之），Agent 就会引用一条与线上公示规则不一致的条款，而且不报错。

语料分两层（清单见 [`doc/policy/README.md`](https://github.com/tiltwind/refund-agent/blob/main/doc/policy/README.md)）：

- [`law/`](https://github.com/tiltwind/refund-agent/tree/main/doc/policy/law)：L01–L05 法律法规，**法定底线**，用于判断平台条款是否有效；
- [`platform/`](https://github.com/tiltwind/refund-agent/tree/main/doc/policy/platform)：P01–P11 平台政策，**与消费者的直接约定**，答复用户引用的就是它。

每篇文档必须带 YAML frontmatter，缺字段直接构建失败——元数据不是装饰：

```yaml
---
doc_id: P02
title: 星辰优选售后服务与退换货政策
category: 平台政策
layer: platform             # law | platform，路由与重排都要用
publisher: 星辰优选商城      # 法规层写 authority（发布机关），代码里归一成同一个字段
doc_type: 平台交易规则
effective_date: 2024-01-01  # 必填：检索前按它硬过滤
expire_date: 9999-12-31     # 必填：长期有效记 9999-12-31
version: v3.2
public_notice_date: 2023-12-25
authority_level: 3          # 1 法律，2 行政法规与部门规章，3 平台规则
retrieval_scope: 退款判定的核心条款——退货窗口、商品条件、类目限制、风控、退款时效
tags: [退货窗口, 无理由退货, 商品条件, 高风险账户]
---
```

必填五项是 `doc_id / title / layer / effective_date / expire_date`——
没有生效日期就没法在检索时排除已废止条款，没有 `layer` 就没法在法规与平台规则冲突时定序。
**一篇没有元数据的政策文档进了库，会以「看起来正常」的方式污染每一次检索。**

正文按「第 X 条」组织标题（`##` 为章 / 条，`###` 为细则），表格用 Markdown 表格——
下一步的切分逻辑完全依赖这个结构。

---

## 五、第 3 步 · 模型层 `llm/`

四个文件，与业务无关，被灌库、检索、Agent 三条链路共用。

| 文件 | 职责 | 失败时 |
|---|---|---|
| [`device.py`](https://github.com/tiltwind/refund-agent/blob/main/llm/device.py) | cuda > mps > cpu | — |
| [`embedding/bge_m3.py`](https://github.com/tiltwind/refund-agent/blob/main/llm/embedding/bge_m3.py) | 稠密向量（1024 维）+ tokenizer 计长 | **硬失败** |
| [`rerank/bge_reranker.py`](https://github.com/tiltwind/refund-agent/blob/main/llm/rerank/bge_reranker.py) | cross-encoder 重排 | 打 warn 后降级 |
| [`chat.py`](https://github.com/tiltwind/refund-agent/blob/main/llm/chat.py) | 供应商与模型名的**唯一**解析处 | 无凭据时改写降级 |

### 嵌入：三处必须写死一致

```python
MAX_LENGTH = 1024      # 灌库与检索必须同一个值，否则 query 与 passage 编码不对称
DIMENSION  = 1024      # 由 model.config.hidden_size 自检，对不上直接抛
# dense 表示取 [CLS]（hidden[:, 0]），不是 mean pooling —— 取错不报错，只让相似度整体失真
```

三件事一旦在灌库与检索之间不一致，检索结果不会报错，只会悄悄变差，这是 RAG 里最难定位的一类故障。
所以这里直接用 `transformers` 而不是 `FlagEmbedding` / `sentence-transformers`：把 pooling 与截断长度摆在明面上。

**嵌入模型没有降级实现**：拿不到就抛。换嵌入模型等于换向量空间，悄悄退回哈希嵌入那种兜底，
会让检索看起来在工作，实际召回的条款与问题无关。

### 重排：可选，但降级要出声

`reranker()` 拿不到模型返回 `None`，调用方退回「融合分 + 效力位阶加权」，并打一行 warn。
和嵌入相反的取舍：重排缺失只是排序变粗，融合分仍然可用。

### 对话模型：解析只做一次

项目里有两处调对话模型——Agent 主循环和查询改写。两处各自 `os.getenv` 拼模型名，
迟早漂移成「主模型换了供应商、改写还在打上一家的端点」，而且不报错，只是改写默默走降级。所以：

```
选哪家：REFUND_AGENT_PROVIDER 显式指定 > 哪边 key 非空走哪边 > 两边都配走 anthropic
用哪个：REFUND_AGENT_MODEL > OPENAI_MODEL / ANTHROPIC_MODEL > 调用方默认值（仅当同供应商）
端点：  跟着**模型**走，不跟着当前 provider 走
```

「两边都配了走 anthropic」不是偏好，是不改变老 `.env` 的既有行为：
**切换供应商必须是一个显式动作，而不是删掉某个 key 的副作用。**

**验证**（第一次会下载权重，耐心等；国内网络先 `export HF_ENDPOINT=https://hf-mirror.com`）：

```bash
python -c "
from llm.embedding import embedder
e = embedder()
print('device =', e.device, '| tokens =', e.count_tokens('金牌会员签收 10 天未拆封'),
      '| dim =', len(e.encode_query('还能退吗？')))
"

python -c "
from llm.rerank import reranker
m = reranker()
print(m.score('签收10天的耳机未拆封还能无理由退吗？',
              ['金牌会员无理由退货窗口为签收后 15 天。', '退款将原路退回，1-3 个工作日到账。']))
"
```

期望：`dim = 1024`；重排两条分数明显拉开（第一条高）。
对话模型的状态用 `set -a; source .env; set +a; python -c "from llm import chat; print(chat.describe())"`
看一行摘要——**到底在打哪个端点的哪个模型，要一眼可见**。

---

## 六、第 4 步 · 切片 `knowledge/chunking/`

把 16 篇 Markdown 变成可入库的父子块。四个文件：

| 文件 | 职责 |
|---|---|
| [`model.py`](https://github.com/tiltwind/refund-agent/blob/main/knowledge/chunking/model.py) | `DocMeta` / `Chunk`；块头拼什么 |
| [`markdown.py`](https://github.com/tiltwind/refund-agent/blob/main/knowledge/chunking/markdown.py) | frontmatter + 标题树 + 段落/表格/代码 |
| [`semantic.py`](https://github.com/tiltwind/refund-agent/blob/main/knowledge/chunking/semantic.py) | 超长自然段的语义切分兜底 |
| [`policy.py`](https://github.com/tiltwind/refund-agent/blob/main/knowledge/chunking/policy.py) | 编排 + 参数（320 / 512 / overlap=0） |

流程：

```
frontmatter 解析 → 按标题层级切 → 父块 = 一个 ## 小节（一章 / 一条完整规则）
  → 父块内按段落切 → 子块 = 检索单元，目标 320 / 硬上限 512 token
      表格与代码原子化（半张表会给出与完整规则相反的结论）
      超长自然段 → 语义切分兜底
  → 每个子块加块头：【文档】+【路径】
```

四个决定，照抄之前先理解它们的前提：

- **overlap = 0**。overlap 防的是「关键句正好落在切分边界上被劈成两半」，前提是切分边界与语义边界无关。
  这里恰恰相反——边界是标题和自然段，本身就是语义边界。加了只会让索引变大、top-k 里出现高度重复的相邻块。
- **块头只放文档标题和标题路径**。生效日期、tags 这类文档级常量故意不进块头：
  同一篇文档的每个块都带上它们，对文档内部的区分度是零，却会稀释短块（法规条文块正文常常只有 100 token）。
  它们进标量字段，参与过滤和排查。
- **父块粒度定在 `##` 而不是 `###`**。定在 `###`，法规层一个条文就是一个父块，父子块退化成 1:1，
  「小块检索、大块喂模型」就白做了。
- **父块不单独存储**。它就是同 `parent_id` 的子块按 `chunk_index` 拼接，检索期还原（overlap = 0 让还原精确）。
  代价换来的是没有第二个 collection、没有「父块存了子块删了」这类不一致。

还有一处容易做错：**计长必须用嵌入模型自己的 tokenizer**，不能用 `len()`。
中文 1 字 ≈ 1~1.5 token（XLM-R BPE），拿字符数当 token 数会静默超限。

**验证**（不连 Milvus，只切片）：

```bash
python -c "
from pathlib import Path
from knowledge.chunking import chunk_document
from llm.embedding import embedder

root = Path('.').resolve(); m = embedder(); total = []
for p in sorted((root / 'doc/policy').rglob('*.md')):
    if p.name == 'README.md': continue
    got = chunk_document(p, root, m.count_tokens, m.encode_documents)
    print(f'{p.name}: {len(got)} 子块 / {len({c.parent_id for c in got})} 父块')
    total += got
print('合计', len(total), '子块')
print(m.truncation_report([c.text for c in total]))
"
```

期望：当前语料切出 **353 个子块 / 174 个父块**，`truncated == 0`，p99 约 300 token。
`truncated` 只要不是 0 就必须处理——超过 `max_seq_length` 的部分**从未进入模型**，
对向量的贡献严格为 0，那种块躺在库里看起来「已经索引了」，却永远召回不到。

---

## 七、第 5 步 · 向量库与灌库

### 起 Milvus

```bash
bash scripts/milvus.sh start     # standalone + 内嵌 etcd，等待就绪后打印状态
bash scripts/milvus.sh status
```

| 用途 | 宿主机端口 | 容器端口 |
|---|---|---|
| gRPC（客户端连接） | 19530 | 19530 |
| 健康检查 / metrics | **19091** | 9091 |
| 内嵌 etcd | 2379 | 2379 |

健康端口对外映射成 19091 是为了不和本机其它服务抢 9091。数据落在 `~/.refund-agent-milvus`，
容器删了重建不丢。完整命令见
[`doc/platform/milvus.md`](https://github.com/tiltwind/refund-agent/blob/main/doc/platform/milvus.md)。

### 建表与灌库

[`knowledge/seed_milvus.py`](https://github.com/tiltwind/refund-agent/blob/main/knowledge/seed_milvus.py) 一个脚本做三件事：切片、建表、灌库。

schema 上有四个要点：

```python
# 1. 正文与可检索文本分开存
schema.add_field("body", DataType.VARCHAR, max_length=16384)   # 原文，装配后喂模型
schema.add_field("text", DataType.VARCHAR, max_length=20480,   # 块头 + 正文，进 embedding 与 BM25
                 enable_analyzer=True,
                 analyzer_params={"type": "chinese"})          # 2. 中文必须显式指定分析器

# 3. BM25 由 Milvus 服务端算：插入只给 text，sparse 由 Function 生成
schema.add_function(Function(name="bm25", function_type=FunctionType.BM25,
                             input_field_names=["text"], output_field_names=["sparse"]))

# 4. 几百个块，稠密一路直接用 FLAT 精确检索 —— 没有 nlist / ef 可抖
index.add_index(field_name="dense",  index_type="FLAT", metric_type="COSINE")
index.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
```

`analyzer_params={"type": "chinese"}` 漏了会让 BM25 **静默失效**：默认的 `standard` 按空白切词，
「金牌会员15天未拆封」会整串变成一个 term，永远匹配不上。

标量字段里有两个坑：`max_length` 的单位是**字节**不是字符（中文一字 3 字节）；
日期存 ISO 字符串（`YYYY-MM-DD` 的字典序即时间序，能直接进 filter 表达式比较）。

```bash
python knowledge/seed_milvus.py
```

脚本默认 **drop 后重建**。条款是全量小语料，增量更新省不了多少时间，而「旧版本块残留」会让检索同时召回新旧两版。
线上不该这么灌——政策按版本发布、灰度切换 collection，在同一个 collection 上原地 drop 会让检索在重建的空窗期返回空。

**验证**：

```bash
python -c "
from services.rag import store
print(store.COLLECTION, store.client().get_collection_stats(store.COLLECTION))
"
```

期望 `row_count` 与上一步的子块数一致（353）。

---

## 八、第 6 步 · 检索链路 `services/rag/`

一次 `search_refund_policy` 内部有六次数据形变，各占一个文件：

| 步骤 | 文件 | 做什么 | 出错的表现 |
|---|---|---|---|
| 1 改写 | [`pipeline/rewrite.py`](https://github.com/tiltwind/refund-agent/blob/main/services/rag/pipeline/rewrite.py) | 拆多意图、判断要不要法规层，**输出自然语言问句** | 检索到的完全是另一类问题的条款 |
| 2 路由 | [`pipeline/route.py`](https://github.com/tiltwind/refund-agent/blob/main/services/rag/pipeline/route.py) | 平台层 / 法规层各给多少名额与权重 | 该引平台条款却引了法条 |
| 3 过滤 | [`pipeline/filters.py`](https://github.com/tiltwind/refund-agent/blob/main/services/rag/pipeline/filters.py) | 生效日期 + 层级，**只做硬约束** | 明明有条款却一条没召回 |
| 4 召回融合 | [`pipeline/recall.py`](https://github.com/tiltwind/refund-agent/blob/main/services/rag/pipeline/recall.py) | 稠密 + BM25 双路 → RRF（k=20） | 召回了但排名靠后 |
| 5 重排 | [`pipeline/rerank.py`](https://github.com/tiltwind/refund-agent/blob/main/services/rag/pipeline/rerank.py) | cross-encoder + 层级/文档先验 | 候选里有正确答案但没顶上来 |
| 6 装配 | [`pipeline/assemble.py`](https://github.com/tiltwind/refund-agent/blob/main/services/rag/pipeline/assemble.py) | 父块回填 + 去重 + 相邻合并 + 预算截断 | 上下文里有证据但答复没用上 |

编排在 [`milvus.py`](https://github.com/tiltwind/refund-agent/blob/main/services/rag/milvus.py)，
Milvus 存取的薄封装在 [`store.py`](https://github.com/tiltwind/refund-agent/blob/main/services/rag/store.py)
（collection 名与字段列表**只有一个定义点**，灌库脚本和检索链路都从这里拿，改字段不会漏改一边）。

拆成六个文件不是为了好看，是因为**每一步都可能是坏 case 的源头，而它们在最终结果里长得一模一样**。
上表最后一列就是排查手册：症状对应哪一步。

### 三个容易做反的决定

**改写必须输出问句。** Agent 拼出来的 query 长这样：`金牌会员 耳机 未拆封 签收10天 无理由退货`。
同一语料、同一套参数，只把它改写成「金牌会员签收 10 天的耳机未拆封，无理由退货还在窗口期内吗？」，
重排 top-1 就从 P07 第三条（极速退款，与问题无关，只是「金牌会员」四个字高度匹配）变成 P02 第二条（正确答案），
分数 0.947 → 0.984。原因在重排：cross-encoder 判断的是「这段文字**回没回答这个问题**」，
关键词串里根本没有问题，它只能退化成算主题相似度。
**「精简成关键词」这个看起来天经地义的预处理，在带重排的链路里是反向优化。**

**必须双路召回。** BM25 不可替代的是精确 term（`7 天`、`3 次`、`90 天`、`生鲜`、`运费险`）：
问「近 90 天退款几次算高风险」，`90` 和 `3` 是低频高 IDF term，BM25 直接把 P02 第五条顶到第一；
稠密只会把它稀释成「风控相关」。反过来，用户说「拆开看了一眼就想退」，正文写的是「已开启包装，但未投入使用」，
一个查询词都不重合——这时候只有稠密捞得到。

**融合在应用层显式做，不用 Milvus 的 `hybrid_search`。** 融合分是排查坏 case 的关键中间产物：
一条条款是两路都召回了、还是只被 BM25 单路捞到（那它在 RRF 里天然吃亏，得靠重排救回来），黑盒融合看不出来。

### 降级路径都要显式

| 组件 | 不可用时 | 为什么 |
|---|---|---|
| 改写模型 | 原文透传，排序掉一档 | 改写有延迟和改坏的风险，不该拖垮整条链路 |
| 重排模型 | 融合分 + 先验加权，打 warn | 排序变粗但仍可用 |
| 嵌入模型 | **硬失败** | 换向量空间等于检索结果无意义 |
| 检索结果为空 | **抛异常** | 这是运维故障（collection 空 / 没灌库），不是「没有适用政策」；让 Agent 带着「未检索到条款」继续判定，等于把它推回「凭记忆编政策」 |

**验证**：

```bash
REFUND_AGENT_RAG_TRACE=on python -c "
from services.rag.milvus import MilvusRagService
for s in MilvusRagService().search_policy('金牌会员签收 10 天的耳机未拆封，无理由退货还在窗口期内吗？'):
    print(round(s.score, 3), s.section)
"
```

期望 top-1 命中 P02 的退货窗口条款；`REFUND_AGENT_RAG_TRACE=on` 会打印六步各自的中间产物——
「明明有条款却没召回」和「召回了但没排上来」是两种病，修法完全不同，这行输出就是分辨它们的依据。

> 权重（`0.80` 相关性 / `0.20` 先验）、阈值（`MIN_SCORE=0.30`）、RRF 的 `k=20`、法规层 `0.5` 降权——
> **这些都是未经校准的起点，不是结论**，必须在标注集（query → 应召回的 section）上调。

---

## 九、第 7 步 · 上下文与服务接入层

### `app/context.py`：身份的落地点

```python
@dataclass
class RefundContext:
    customer_id: str              # 主体身份，网关从 JWT claims.sub 提取后注入
    actor: str = "self"           # self | staff:{staff_id}，审计要用
    request_id: str = ""          # 全链路追踪 ID，兼作幂等键
    session_id: str = ""          # 一通会话共用，随 trace 上报
    request_source: str = "prod"  # prod | eval，决定 services/ 选哪个实现
```

它由 `create_agent(context_schema=...)` 传给工具层，**不会出现在发给模型的 tool schema 里**。
把 `customer_id` 放进工具参数，等于把「访问谁的数据」的决策权交给一个可被 prompt injection 操控的组件（IDOR 越权）。

`request_source` 同理**必须由服务端决定**：否则任何调用方声明一句 `request_source=eval` 就能绕开真实数据与真实风控。

### 一个服务 = 一个接口 + 按需的实现

```
services/
├── factory.py          # 按 request_source 选实现
├── eval_store.py       # 加载 evals/data + 会话隔离
├── errors.py           # EvalDataMissError
├── customer/           # protocol.py（接口）/ prod.py（留桩）/ eval.py
├── order/              # protocol.py / prod.py（留桩）/ eval.py（含规则引擎副本）
└── rag/                # 只有一个实现，不分数据源
```

工厂函数是这一层的核心，有两条纪律：

```python
def order_service(ctx: RefundContext) -> OrderService:
    match ctx.request_source:
        case "prod": return ProdOrderService()
        case "eval": return EvalOrderService()
        case other:  raise ValueError(f"unknown request_source: {other}")   # 绝不 fallback 到 prod
```

**未知取值一律抛异常**——拼错一个字母就静默连上线上库，是这类工厂最典型的事故。
而 `rag_service()` 不看 `request_source`，直接返回 Milvus 实现：检索结果本身就是被评估的对象，
为离线评估另造一份写死的条款，等于把这段逻辑整体排除在回归之外。

### 规则引擎副本与 eval 数据

线上规则引擎在订单系统里，离线连不上，所以
[`services/order/eval.py`](https://github.com/tiltwind/refund-agent/blob/main/services/order/eval.py) 维护一份等价副本。
判定分成两段，顺序不能乱：

```
第一段（硬否决，与 reason_type / item_condition 无关，命中即定案）
  归属校验 → 已退款 → 风控（近 90 天 > 3 次）→ 类目黑名单 → 超出最宽窗口（15 天）
第二段（判定确实取决于缺失参数时，才返回「需补充：…」）
  退货窗口（无理由：普通 7 / 金牌 15；质量问题：一律 15）→ 商品条件（无理由需未拆封）
```

这个顺序把「何时该追问用户」从模型的自由裁量收回到规则里：
硬否决命中时无论用户怎么回答都不会通过，追问只会白白拖长处理时间。
风控排在会员权益之前也是刻意的——金牌 + 高风险必须走风控结论。

[`evals/data/*.json`](https://github.com/tiltwind/refund-agent/tree/main/evals/data) 手工构造，每条对应一个规则分支：

```jsonc
"O2011": { "customer_id": "C1006", "signed_days_ago": 15,
           "_note": "金牌窗口边界内：15 == 15，判定条件是 > window，应通过" },
"O2012": { "customer_id": "C1006", "signed_days_ago": 16,
           "_note": "金牌窗口边界外：16 > 15，超出一天也应拒绝" }
```

两条硬要求：**时间必须相对化**（`signed_days_ago` 而非绝对时间戳，否则数据集放三个月后所有窗口判定全部失效，
而且失效得悄无声息）；**查不到必须显式失败**（抛 `EvalDataMissError`，把用例标记为 `invalid` 而非 `failed`——
它不是回归，是 eval 数据覆盖不足）。

`eval_store` 用 `contextvars` 给每条用例发一份独立数据副本：`execute_refund` 会把 `refunded` 置成 `True`，
并发跑批时两条用例会互相污染。

**验证**（不连任何外部服务）：

```bash
python -c "
from services.order.eval import EvalOrderService
s = EvalOrderService()
print(s.check_eligibility('O2001', 'C1001', '无理由', '未拆封'))   # 金牌 10 ≤ 15 → 通过
print(s.check_eligibility('O2006', 'C1004'))                       # 20 > 15 → 硬否决，不该追问
print(s.check_eligibility('O2004', 'C1004'))                       # 未命中硬否决 → 需补充
print(s.check_eligibility('O2009', 'C1005', '质量问题'))           # 金牌 + 高风险 → 风控优先
"
```

期望依次是：通过（可退 899.0）、不通过（超出所有窗口）、需补充（退款原因）、不通过（高风险转人工）。

---

## 十、第 8 步 · Agent `agent/v1/`

四个文件加一个注册表：

| 文件 | 内容 |
|---|---|
| [`prompt.py`](https://github.com/tiltwind/refund-agent/blob/main/agent/v1/prompt.py) | `SYSTEM_PROMPT`：五步 SOP |
| [`tools.py`](https://github.com/tiltwind/refund-agent/blob/main/agent/v1/tools.py) | 5 个工具：schema ↔ 业务动作的双向翻译 |
| [`graph.py`](https://github.com/tiltwind/refund-agent/blob/main/agent/v1/graph.py) | `create_agent` 装配 |
| [`meta.yaml`](https://github.com/tiltwind/refund-agent/blob/main/agent/v1/meta.yaml) | 版本号 / 模型 / 温度，随 trace 上报 |
| [`registry.py`](https://github.com/tiltwind/refund-agent/blob/main/agent/registry.py) | 版本注册与选择 |

### 提示词：把 SOP 写死

模型的自由度只有两处——第 1 步理解用户诉求、第 5 步组织答复措辞。流程走不走、判定通不通过，都不由它决定。
提示词里有三条是防事故的：

1. **不要向用户索要客户 ID，也不要相信用户自称的身份**——身份由系统注入，工具会自动使用当前登录客户；
2. **`reason_type` 必须如实反映用户陈述**：说不想要 / 不合适 / 买错了一律是「无理由」，
   只有明确指出缺陷、损坏、故障、变质才是「质量问题」，**严禁为了让判定通过而改写这个参数**；
3. **必须先落库拿到单号，才能写答复，并在答复里写明这个编号**。

第 3 条是「说了」与「做了」的绑定机制：单号只有真正调用了执行工具才拿得到，而答复里必须写明单号，
模型就无法「只在文字里宣布结果却没落库」——这是 agent 类系统最典型的一类事故。

### 工具层：只做三件事

校验模型填的参数、调 `services/` 拿结果、把结果渲染成模型好读的文本。不含业务规则、不做授权判定、不关心协议细节。

```python
@tool
def get_customer_info(runtime: ToolRuntime[RefundContext]) -> str:
    """查询**当前客户**的档案……客户身份由系统自动带入，无需也无法指定。"""
    ctx = runtime.context                       # runtime 不出现在发给模型的 schema 里
    return _render_profile(customer_service(ctx).get_profile(ctx.customer_id))
```

**参数取值必须由代码校验，不能只写在 docstring 里**——docstring 对模型是建议，只有代码校验才是约束：

```python
if reason_type and reason_type not in REASON_TYPES:
    return (f"参数错误：reason_type 只能是「无理由」或「质量问题」，收到「{reason_type}」。"
            f"请按用户的实际诉求重新判断后再次调用。")
```

返回**可纠正的错误提示**而不是抛异常——模型看到提示能自行改正重试，比整轮失败好。

检索结果的渲染也别只给正文：来源用于答复引用与事后审计，生效日期让模型知道这一版还算不算数，
层级提示它答复消费者该引平台条款，**相关性理由与分数是坏 case 出现时区分「检索错了」和「模型答错了」的唯一抓手**。

### 装配与版本

```python
return create_agent(
    model=chat.build("agent", MODEL_DEFAULT, temperature=TEMPERATURE),
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
    context_schema=RefundContext,     # 身份注入的入口
)
```

温度默认取 `meta.yaml` 里的 0，但留出 `REFUND_AGENT_TEMPERATURE`：
上线前的稳定性验收必须用**线上真实温度**连跑多轮看方差，在温度 0 下做的评估不代表线上表现。
（注意温度 0 不等于确定性输出：浮点非结合性、batch 组成、服务端硬件差异都会带来差异。它是降噪，不是消噪。）

`registry.get("v1")` 对未知版本抛异常而不 fallback——否则一次拼写错误会让整轮评估悄悄跑在旧版本上，
报告却署着新版本的名字。`agent/` 下按版本目录**并存**而非原地覆盖，是同集对比 v1/v2、定位退化用例、
灰度回滚、线上归因这四件事的前提。

**验证**（跑一轮完整链路，需要模型凭据 + Milvus 已灌库）：

```bash
set -a; source .env; set +a
python -c "
from agent import registry
from app.context import RefundContext
r = registry.get('v1').invoke(
    {'messages': [{'role': 'user', 'content': '订单 O2004 的跑鞋没拆封，想退。'}]},
    context=RefundContext(customer_id='C1004', request_id='t-1', request_source='eval'))
for m in r['messages']:
    for c in getattr(m, 'tool_calls', None) or []:
        print('▶', c['name'], c['args'])
print(r['messages'][-1].text)
"
```

期望工具调用顺序是 `get_customer_info → search_refund_policy → check_refund_eligibility → execute_refund`，
且最终答复里带着 `R9000` 这样的退款单号。

---

## 十一、第 9 步 · 入口与跑通

[`main.py`](https://github.com/tiltwind/refund-agent/blob/main/main.py) 是离线演示入口，做四件事：
启动时打印模型与埋点状态、跑三个场景、打印工具调用轨迹、最后按审计视角复盘决策流水。

三个场景各守一条规则分支：

| 场景 | 数据 | 预期 |
|---|---|---|
| 金牌会员的窗口期优待 | C1001 / O2001，签收 10 天 | **批准**（普通 7 天会拒，金牌 15 天通过） |
| 不支持退款的类目 | C1002 / O2002，生鲜 | **拒绝**（类目黑名单，签收才 2 天也不退） |
| 高风险账户转人工 | C1003 / O2003，近 90 天退款 4 次 | **拒绝并引导**（风控优先于质量问题） |

[`run-main.sh`](https://github.com/tiltwind/refund-agent/blob/main/run-main.sh) 在跑之前做凭据预检：

```bash
bash run-main.sh              # 跑三个演示场景
bash run-main.sh --trace      # 额外打印检索链路每一步的中间产物
ENV_FILE=.env.staging bash run-main.sh
```

预检只拦三种「一定跑不起来」的情况：`.env` 不存在、两个 API key 都为空、走 OpenAI 却没配 `OPENAI_MODEL`。
**脚本不回显模型端点**——供应商解析有优先级规则，脚本里复述一遍必然和实际用的那个不一致；
真正解析出来的结果由 `main.py` 启动时打印。

最后一段值得单独看：

```python
new_rows = eval_store.decision_log()[log_before:]
if not new_rows:
    print("  ⚠️ 本轮没有任何终局动作落库")
```

这就是「说了」是否等于「做了」的断言点。评估流水线里，这一条会变成一条硬指标。

**验证**：三个场景各自给出预期结论，且末尾的决策流水里有三条记录，每条都带 `actor` 与 `request_id`——
缺这两个字段的流水事后追不到人、对不上链路。

---

## 十二、第 10 步 · 埋点（可选）

[`services/telemetry.py`](https://github.com/tiltwind/refund-agent/blob/main/services/telemetry.py)：
Langfuse v3+ 底座是 OpenTelemetry，`CallbackHandler` 直接吃 LangGraph 的回调，
不需要自建 tracer provider——接一个 callback 就能拿到完整调用树。

```python
result = agent.invoke(
    {"messages": [...]},
    context=ctx,                                     # 业务身份 → 工具层
    config=telemetry.trace_config(ctx, meta, ...),   # 运行时回调 → Langfuse
)
```

**`context` 与 `config` 别混**：前者是给工具层的业务身份，后者是给 LangGraph 运行时的回调通道。

三条设计约束：

1. **缺密钥就静默降级**。埋点是旁路，不该让主链路挂掉；`trace_config()` 返回空 dict，调用方不必写 `if`。
2. **脱敏做在 SDK 的 `mask` 钩子里，不做在调用点**。所有 span 的 input/output 统一过一遍——
   线上评估的 LLM judge 读的是同一批 trace，漏一处 PII 就跟着进了 judge 的 prompt。
3. **`customer_id` 上报加盐哈希**。要的是「同一个人的多次请求能串起来」，不是「知道他是谁」。

两个「接上了但看不到数据」的常见原因：

- **短命脚本必须显式 flush**。SDK 攒批异步上报，进程跑完就退，还没到发送时机的那批 span 会随进程消失。
  `main.py` 把 `telemetry.flush()` 放在 `finally` 里——链路中途抛异常时，那条失败的 trace 恰恰是最该看到的。
- **启动时打一行状态并做 `auth_check`**。凭据写错、Langfuse 没起这类问题，在启动时报出来只要一秒。

本地起 Langfuse 见
[`doc/platform/langfuse.md`](https://github.com/tiltwind/refund-agent/blob/main/doc/platform/langfuse.md)，
把 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` 填进 `.env` 即可。

**验证**：启动行显示 `Langfuse: on → http://localhost:3000`，
Langfuse UI 上能看到 `refund-chat:*` 这条 trace，展开后图节点、工具、generation 自动成树。

---

## 十三、验收清单

按顺序过一遍，每一项都能单独定位问题：

| # | 命令 | 期望 |
|---|---|---|
| 1 | `python -c "import langchain, pymilvus, torch"` | 无报错 |
| 2 | 切片脚本（第 4 步） | 353 子块 / 174 父块，`truncated == 0` |
| 3 | `bash scripts/milvus.sh status` | 容器 healthy，19530 可连 |
| 4 | `python knowledge/seed_milvus.py` | 建表 + 灌库完成，行数与 #2 一致 |
| 5 | 检索冒烟（第 6 步） | top-1 命中 P02 退货窗口条款 |
| 6 | 规则边界（第 7 步） | 四条判定与 `_note` 标注一致 |
| 7 | `bash run-main.sh` | 三个场景结论符合预期，决策流水 3 条 |
| 8 | `bash run-main.sh --trace` | 六步链路中间产物逐行可见 |

---

## 十四、常见故障

| 症状 | 原因 | 处理 |
|---|---|---|
| `policy collection「…」检索不到任何生效条款` | collection 空 / 灌库没跑 / 条款被生效日期过滤掉 | 跑 `python knowledge/seed_milvus.py`；确认 frontmatter 的 `effective_date ≤ 今天 < expire_date` |
| 灌库报 `有 N 个块超过 1024 token` | 有超长表格未被拆开 | 调小 `POLICY_CHUNK_MAX`，或把大表拆成多张 |
| BM25 一条都召不回中文查询 | 建表漏了 `analyzer_params={"type": "chinese"}` | 重建 collection（改 analyzer 必须重灌） |
| 模型报 404 且模型名看着没错 | provider 切到 openai 后仍在用 `meta.yaml` 的 Claude 模型名 | 配 `OPENAI_MODEL`，或用 `REFUND_AGENT_MODEL=openai:xxx` 带前缀强制覆盖 |
| 改写报 `This response_format type is unavailable now` | 兼容网关不支持 `json_schema` | 已默认走 `function_calling`；仍不行设 `REFUND_AGENT_REWRITE_STRUCTURED=json_mode` |
| 改写报 `Thinking mode does not support this tool_choice` | 推理模式与 `tool_choice` 冲突 | `REFUND_AGENT_REWRITE_REASONING=none`（非推理模型不要设这个参数，它会直接报错） |
| 改写始终降级、打 warn | 无凭据 / 超时 / 结构化失败 | 不影响可用性，只掉一档排序；要彻底关掉用 `REFUND_AGENT_REWRITE=off` |
| 权重下载失败 | 网络 | `export HF_ENDPOINT=https://hf-mirror.com`，或 `BGE_M3_MODEL` / `BGE_RERANKER_MODEL` 指向本地目录 |
| 重排 warn「模型不可用」 | 权重缺失或 `REFUND_AGENT_RERANK=off` | 链路照跑，精排缺失；要恢复精度就补上模型 |
| 跑完了 Langfuse 上什么都没有 | 短命脚本没 flush / 凭据错 / `LANGFUSE_HOST` 变量名不被识别 | 看启动行的 `auth_check` 结果；确认 `flush()` 在 `finally` 里 |
| `EvalDataMissError` | eval 数据缺这个客户 | 补 `evals/data/customers.json`——这是数据覆盖不足，不是回归 |
| `unknown request_source: …` | 拼写错误 | 只接受 `prod` / `eval`，故意不 fallback |

---

## 十五、v1 之后

v1 刻意留白的部分，以及它们各自的入口：

| 待补 | 在哪落地 | 说明 |
|---|---|---|
| HTTP 服务外壳 | `app/main.py` + `app/middleware/auth.py` | 读网关注入的身份 header → `RefundContext`；**取不到必须 401，不能降级** |
| 人工审批 | `app/middleware/approval.py` | 高风险 case 用 LangGraph `interrupt` 挂起，状态由 checkpointer 持久化 |
| prod 数据源 | `services/customer/prod.py`、`services/order/prod.py` | 服务身份 + `X-Acting-User` + `traceparent`，横切关注点收敛到 `services/base.py` |
| 评估闭环 | `evals/` | `validate_cases`（数据集自检）→ `offline`（离线回归）→ `compare`（v1/v2 对比）→ `online`（线上采样打分） |
| 检索子 span | `services/rag/pipeline/` | `rag.recall` / `rag.rerank` / `rag.assemble` 是纯函数，要手工包一层才进 trace |
| v2 与灰度 | `agent/v2/` + `registry.select` | 按流量比例路由，trace 里记 `agent_version` 做线上归因 |

这些的设计取舍都写在 [2 · 设计](https://github.com/tiltwind/refund-agent/blob/main/doc/get-start/2-design.md) 的第一、四、五、六章。
先读那几章再动手，比照着代码反推便宜得多。
