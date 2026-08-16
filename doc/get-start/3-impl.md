# 3 · 实现 RefundAgent v1

按依赖顺序搭建 v1，最后用 `bash run-main.sh` 验证三个退款场景。业务、架构和设计分别见 [0 · 需求](https://tiltwind.github.io/refund-agent/doc/get-start/0-requirement.md)、[1 · 架构](https://tiltwind.github.io/refund-agent/doc/get-start/1-architecture.md) 和 [2 · 设计](https://tiltwind.github.io/refund-agent/doc/get-start/2-design.md)。

---

## 一、先看终点

v1 不含 HTTP 服务外壳。客户档案和订单读取本地 eval 数据，政策检索连接 Milvus，模型使用 Anthropic 或 OpenAI 兼容接口。

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

每完成一层先独立验证，再接入上层。

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

主要依赖：

| 依赖 | 用途 | 备注 |
|---|---|---|
| `langchain` / `langgraph` | `create_agent` + Agent Loop | 1.x |
| `langchain-anthropic` / `langchain-openai` | 两家供应商 | 后者同时覆盖所有 OpenAI 兼容网关 |
| `pymilvus` | 向量库客户端 | **必须 2.5+**：稀疏向量与 BM25 Function 自该版本起可用 |
| `torch` / `transformers` | 本地嵌入与重排 | 首次运行会下载权重 |
| `pyyaml` | 政策文档 frontmatter、`meta.yaml` | |
| `langfuse` | 可观测 | 缺密钥时静默降级 |

BM25 的分词和打分在 Milvus 服务端完成，因此不依赖 jieba。密钥写入已被 Git 忽略的 `.env`；`.env.example` 只保留占位符。

**验证**：

```bash
python -c "import langchain, langgraph, pymilvus, torch, transformers, yaml; print('deps ok')"
```

---

## 四、第 2 步 · 政策语料 `doc/policy/`

`doc/policy/` 下的 Markdown 是政策语料的唯一来源，不维护 `policies.json` 等副本。

语料分两层（清单见 [`doc/policy/README.md`](https://tiltwind.github.io/refund-agent/doc/policy/README.md)）：

- [`law/`](https://tiltwind.github.io/refund-agent/doc/policy/law)：L01–L05 法律法规，**法定底线**，用于判断平台条款是否有效；
- [`platform/`](https://tiltwind.github.io/refund-agent/doc/policy/platform)：P01–P11 平台政策，**与消费者的直接约定**，答复用户引用的就是它。

每篇文档必须包含 YAML frontmatter：

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

必填项为 `doc_id`、`title`、`layer`、`effective_date` 和 `expire_date`。缺失时构建失败。

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

灌库和检索必须使用相同的长度、维度与 pooling。嵌入模型不可用时直接报错，不切换向量空间。

### 重排：可选，但降级要出声

`reranker()` 不可用时返回 `None`，调用方退回“融合分 + 效力位阶加权”并记录警告。

### 对话模型：解析只做一次

Agent 主循环和查询改写共用 `llm/chat.py` 的模型配置：

```
选哪家：REFUND_AGENT_PROVIDER 显式指定 > 哪边 key 非空走哪边 > 两边都配走 anthropic
用哪个：REFUND_AGENT_MODEL > OPENAI_MODEL / ANTHROPIC_MODEL > 调用方默认值（仅当同供应商）
端点：  跟着**模型**走，不跟着当前 provider 走
```

两组凭据同时存在时默认使用 Anthropic；切换供应商应设置 `REFUND_AGENT_PROVIDER`。

**验证**（首次运行会下载权重；国内网络可设置 `HF_ENDPOINT=https://hf-mirror.com`）：

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
该命令会打印当前端点和模型。

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

切分参数：`overlap=0`；块头只含文档标题和标题路径；父块粒度为 `##`；父块按 `parent_id` 和 `chunk_index` 从子块还原。token 长度使用嵌入模型的 tokenizer 计算。

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

期望：353 个子块、174 个父块，`truncated == 0`，p99 约 300 token。出现截断时应调整切分。

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
[`doc/platform/milvus.md`](https://tiltwind.github.io/refund-agent/doc/platform/milvus.md)。

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

中文 BM25 必须设置 `analyzer_params={"type": "chinese"}`。`max_length` 的单位是字节；日期存为 `YYYY-MM-DD`，可直接用于过滤表达式。

```bash
python knowledge/seed_milvus.py
```

脚本默认 drop 后重建，仅用于本地环境。线上按版本新建 collection，再灰度切换。

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

编排位于 [`milvus.py`](https://github.com/tiltwind/refund-agent/blob/main/services/rag/milvus.py)，Milvus 存取封装位于 [`store.py`](https://github.com/tiltwind/refund-agent/blob/main/services/rag/store.py)。collection 名和字段只在 `store.py` 定义。

改写输出自然语言问句，便于 cross-encoder 判断段落是否回答问题。召回同时使用 BM25 和稠密向量，分别覆盖精确词项和语义匹配；RRF 融合在应用层完成，便于记录各路排名。

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

期望 top-1 命中 P02 的退货窗口条款。`REFUND_AGENT_RAG_TRACE=on` 会打印各步中间结果。

> 权重（`0.80` 相关性 / `0.20` 先验）、阈值（`MIN_SCORE=0.30`）、RRF 的 `k=20`、法规层 `0.5` 降权——
> 这些参数需要用 query → section 标注集校准。

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

`create_agent(context_schema=...)` 把它传给工具层，不暴露在 tool schema 中。`customer_id` 和 `request_source` 均由服务端设置。

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

工厂函数按数据源返回实现：

```python
def order_service(ctx: RefundContext) -> OrderService:
    match ctx.request_source:
        case "prod": return ProdOrderService()
        case "eval": return EvalOrderService()
        case other:  raise ValueError(f"unknown request_source: {other}")   # 绝不 fallback 到 prod
```

未知取值直接报错，不回退到 prod。`rag_service()` 始终返回 Milvus 实现。

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

只有硬否决全部通过后，规则引擎才会要求补充参数。风控优先于会员权益。

[`evals/data/*.json`](https://github.com/tiltwind/refund-agent/tree/main/evals/data) 手工构造，每条对应一个规则分支：

```jsonc
"O2011": { "customer_id": "C1006", "signed_days_ago": 15,
           "_note": "金牌窗口边界内：15 == 15，判定条件是 > window，应通过" },
"O2012": { "customer_id": "C1006", "signed_days_ago": 16,
           "_note": "金牌窗口边界外：16 > 15，超出一天也应拒绝" }
```

时间使用 `signed_days_ago`。查不到数据时抛 `EvalDataMissError`，用例标记为 `invalid`。

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

提示词包含三条约束：

1. **不要向用户索要客户 ID，也不要相信用户自称的身份**——身份由系统注入，工具会自动使用当前登录客户；
2. **`reason_type` 必须如实反映用户陈述**：说不想要 / 不合适 / 买错了一律是「无理由」，
   只有明确指出缺陷、损坏、故障、变质才是「质量问题」，**严禁为了让判定通过而改写这个参数**；
3. **必须先落库拿到单号，才能写答复，并在答复里写明这个编号**。

### 工具层：只做三件事

校验模型填的参数、调 `services/` 拿结果、把结果渲染成模型好读的文本。不含业务规则、不做授权判定、不关心协议细节。

```python
@tool
def get_customer_info(runtime: ToolRuntime[RefundContext]) -> str:
    """查询**当前客户**的档案……客户身份由系统自动带入，无需也无法指定。"""
    ctx = runtime.context                       # runtime 不出现在发给模型的 schema 里
    return _render_profile(customer_service(ctx).get_profile(ctx.customer_id))
```

参数取值由代码校验：

```python
if reason_type and reason_type not in REASON_TYPES:
    return (f"参数错误：reason_type 只能是「无理由」或「质量问题」，收到「{reason_type}」。"
            f"请按用户的实际诉求重新判断后再次调用。")
```

参数错误返回可纠正的提示，供模型重试。检索结果包含正文、来源、生效日期、层级、相关性理由和分数。

### 装配与版本

```python
return create_agent(
    model=chat.build("agent", MODEL_DEFAULT, temperature=TEMPERATURE),
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
    context_schema=RefundContext,     # 身份注入的入口
)
```

温度默认读取 `meta.yaml`，可用 `REFUND_AGENT_TEMPERATURE` 覆盖。稳定性测试应使用线上温度并运行多轮。`registry.get("v1")` 遇到未知版本直接报错；各版本保留独立目录。

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

预检检查 `.env`、API key 和 OpenAI 模型名。实际端点和模型由 `main.py` 打印。

```python
new_rows = eval_store.decision_log()[log_before:]
if not new_rows:
    print("  ⚠️ 本轮没有任何终局动作落库")
```

该检查用于确认终局动作已落库。

**验证**：三个场景各自给出预期结论，且末尾的决策流水里有三条记录，每条都带 `actor` 与 `request_id`——
缺这两个字段的流水事后追不到人、对不上链路。

---

## 十二、第 10 步 · 埋点（可选）

[`services/telemetry.py`](https://github.com/tiltwind/refund-agent/blob/main/services/telemetry.py) 使用 Langfuse `CallbackHandler` 接收 LangGraph 回调。

```python
result = agent.invoke(
    {"messages": [...]},
    context=ctx,                                     # 业务身份 → 工具层
    config=telemetry.trace_config(ctx, meta, ...),   # 运行时回调 → Langfuse
)
```

`context` 传递业务身份，`config` 传递运行时回调。

三条设计约束：

1. **缺密钥就静默降级**。埋点是旁路，不该让主链路挂掉；`trace_config()` 返回空 dict，调用方不必写 `if`。
2. **脱敏做在 SDK 的 `mask` 钩子里，不做在调用点**。所有 span 的 input/output 统一过一遍——
   线上评估的 LLM judge 读的是同一批 trace，漏一处 PII 就跟着进了 judge 的 prompt。
3. **`customer_id` 上报加盐哈希**。要的是「同一个人的多次请求能串起来」，不是「知道他是谁」。

注意：

- 短命脚本要显式调用 `flush()`；`main.py` 在 `finally` 中调用。
- 启动时运行 `auth_check`，检查凭据和服务状态。

本地起 Langfuse 见
[`doc/platform/langfuse.md`](https://tiltwind.github.io/refund-agent/doc/platform/langfuse.md)，
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
| 评估闭环 | `evals/` | 已有 `dataset/d1`、自检和离线实验 `experiments/ex-1`，见 [4 · 数据集与指标](https://tiltwind.github.io/refund-agent/doc/get-start/4-dataset.md)、[5 · 跑实验](https://tiltwind.github.io/refund-agent/doc/get-start/5-experiment.md)；待补版本对比与线上采样评分 |
| 检索子 span | `services/rag/pipeline/` | `rag.recall` / `rag.rerank` / `rag.assemble` 是纯函数，要手工包一层才进 trace |
| v2 与灰度 | `agent/v2/` + `registry.select` | 按流量比例路由，trace 里记 `agent_version` 做线上归因 |

相关设计见 [2 · 设计](https://tiltwind.github.io/refund-agent/doc/get-start/2-design.md) 第一、四、五、六章。
