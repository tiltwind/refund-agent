# 3 · 政策知识库：切片、灌库与检索

RefundAgent 的判定依据来自 `doc/policy/` 下的 16 篇政策文档。本篇搭出从 Markdown 到一次可用检索的整条路：环境、语料、模型层、切片、灌库、六步检索链路。Agent 本体、服务接入层和入口不在本篇范围内。

业务、架构和设计分别见 [0 · 需求](https://tiltwind.github.io/refund-agent/doc/get-start/0-requirement.md)、[1 · 架构](https://tiltwind.github.io/refund-agent/doc/get-start/1-architecture.md) 和 [2 · 设计](https://tiltwind.github.io/refund-agent/doc/get-start/2-design.md)。

---

## 一、目标

```
doc/policy/*.md
  └─ 切片（父子块）→ Milvus collection（dense 稠密 + sparse BM25）
                                    ↑
用户问句 ─→ search_refund_policy → rag/retrieving/milvus.py
              1 改写 → 2 路由 → 3 过滤 → 4 召回融合 → 5 重排 → 6 装配
           └─ 政策条款（正文 / 来源 / 生效日期 / 层级 / 相关性分数）
```

### 构建顺序

| 步骤 | 建什么 | 能独立验证吗 |
|---|---|---|
| 1 | 环境与依赖 | ✅ import 得动就算过 |
| 2 | `doc/policy/` 政策语料 | ✅ frontmatter 解析 |
| 3 | `llm/` 模型层（嵌入 / 重排 / 对话） | ✅ 编码一句话、打一次分 |
| 4 | `rag/chunking/` 切片 | ✅ 打印块数与 token 分布 |
| 5 | Milvus + `rag/index/seed_milvus.py` 灌库 | ✅ collection 行数 |
| 6 | `rag/retrieving/` 六步检索链路 | ✅ 单跑一次检索看 top-k |

---

## 二、第 1 步 · 环境与依赖

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

BM25 的分词和打分在 Milvus 服务端完成，不依赖 jieba。密钥写入已被 Git 忽略的 `.env`，`.env.example` 只保留占位符。

**验证**：

```bash
python -c "import langchain, langgraph, pymilvus, torch, transformers, yaml; print('deps ok')"
```

---

## 三、第 2 步 · 政策语料 `doc/policy/`

`doc/policy/` 下的 Markdown 是政策语料的唯一来源。

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

正文按「第 X 条」组织标题（`##` 为章 / 条，`###` 为细则），表格用 Markdown 表格，切分逻辑依赖这个结构。

---

## 四、第 3 步 · 模型层 `llm/`

四个文件，与业务无关，被灌库、检索、Agent 三条链路共用。

| 文件 | 职责 | 失败时 |
|---|---|---|
| [`device.py`](https://github.com/tiltwind/refund-agent/blob/main/llm/device.py) | cuda > mps > cpu | — |
| [`embedding/bge_m3.py`](https://github.com/tiltwind/refund-agent/blob/main/llm/embedding/bge_m3.py) | 稠密向量（1024 维）+ tokenizer 计长 | **硬失败** |
| [`rerank/bge_reranker.py`](https://github.com/tiltwind/refund-agent/blob/main/llm/rerank/bge_reranker.py) | cross-encoder 重排 | 打 warn 后降级 |
| [`chat.py`](https://github.com/tiltwind/refund-agent/blob/main/llm/chat.py) | 供应商与模型名的**唯一**解析处 | 无凭据时改写降级 |

### 嵌入：灌库与检索共用的三个参数

```python
MAX_LENGTH = 1024      # 灌库与检索取同一个值
DIMENSION  = 1024      # 由 model.config.hidden_size 自检，对不上直接抛
# dense 表示取 [CLS]（hidden[:, 0]），不用 mean pooling
```

嵌入模型不可用时直接报错，不切换向量空间。

### 重排：可选项与降级提示

`reranker()` 不可用时返回 `None`，调用方退回“融合分 + 效力位阶加权”并记录警告。

### 对话模型

Agent 主循环和查询改写共用 `llm/chat.py` 的模型配置。

**验证**（首次运行会下载权重；国内网络可设置 `HF_ENDPOINT=https://hf-mirror.com`）：

```python
from llm.embedding import embedder
e = embedder()
print('device =', e.device, '| tokens =', e.count_tokens('金牌会员签收 10 天未拆封'),
      '| dim =', len(e.encode_query('还能退吗？')))

```

```python
from llm.rerank import reranker
m = reranker()
print(m.score('签收10天的耳机未拆封还能无理由退吗？',
              ['金牌会员无理由退货窗口为签收后 15 天。', '退款将原路退回，1-3 个工作日到账。']))
```

期望：`dim = 1024`；重排两条分数明显拉开（第一条高）。
对话模型的状态用 `set -a; source .env; set +a; python -c "from llm import chat; print(chat.describe())"`
该命令会打印当前端点和模型。

---

## 五、第 4 步 · 切片 `rag/chunking/`

把 16 篇 Markdown 变成可入库的父子块。四个文件：

| 文件 | 职责 |
|---|---|
| [`model.py`](https://github.com/tiltwind/refund-agent/blob/main/rag/chunking/model.py) | `DocMeta` / `Chunk`；块头拼什么 |
| [`markdown.py`](https://github.com/tiltwind/refund-agent/blob/main/rag/chunking/markdown.py) | frontmatter + 标题树 + 段落/表格/代码 |
| [`semantic.py`](https://github.com/tiltwind/refund-agent/blob/main/rag/chunking/semantic.py) | 超长自然段的语义切分兜底 |
| [`policy.py`](https://github.com/tiltwind/refund-agent/blob/main/rag/chunking/policy.py) | 编排 + 参数（320 / 512 / overlap=0） |

流程：

```
frontmatter 解析 → 按标题层级切 → 父块 = 一个 ## 小节（一章 / 一条完整规则）
  → 父块内按段落切 → 子块 = 检索单元，目标 320 / 硬上限 512 token
      表格与代码原子化，不拆开
      超长自然段 → 语义切分兜底
  → 每个子块加块头：【文档】+【路径】
```

切分参数：`overlap=0`；块头只含文档标题和标题路径；父块粒度为 `##`；父块按 `parent_id` 和 `chunk_index` 从子块还原。token 长度使用嵌入模型的 tokenizer 计算。

**验证**（不连 Milvus，只切片）：

```python
from pathlib import Path
from rag.chunking import chunk_document
from llm.embedding import embedder

root = Path('.').resolve(); m = embedder(); total = []
for p in sorted((root / 'doc/policy').rglob('*.md')):
    if p.name == 'README.md': continue
    got = chunk_document(p, root, m.count_tokens, m.encode_documents)
    print(f'{p.name}: {len(got)} 子块 / {len({c.parent_id for c in got})} 父块')
    total += got
print('合计', len(total), '子块')
print(m.truncation_report([c.text for c in total]))
```

期望：353 个子块、174 个父块，`truncated == 0`，p99 约 300 token。出现截断时应调整切分。

---

## 六、第 5 步 · 向量库与灌库

### Milvus 启动

```bash
bash scripts/milvus.sh start     # standalone + 内嵌 etcd，等待就绪后打印状态
bash scripts/milvus.sh status
```

| 用途 | 宿主机端口 | 容器端口 |
|---|---|---|
| gRPC（客户端连接） | 19530 | 19530 |
| 健康检查 / metrics | **19091** | 9091 |
| 内嵌 etcd | 2379 | 2379 |

健康端口对外映射为 19091，避开本机 9091。数据落在 `~/.refund-agent-milvus`，
容器删除重建不丢。完整命令见
[`doc/platform/milvus.md`](https://tiltwind.github.io/refund-agent/doc/platform/milvus.md)。

### 建表与灌库

[`rag/index/seed_milvus.py`](https://github.com/tiltwind/refund-agent/blob/main/rag/index/seed_milvus.py) 一个脚本做三件事：切片、建表、灌库。

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

# 4. 稠密一路用 FLAT 精确检索，无 nlist / ef 参数
index.add_index(field_name="dense",  index_type="FLAT", metric_type="COSINE")
index.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
```

中文 BM25 必须设置 `analyzer_params={"type": "chinese"}`。`max_length` 的单位是字节；日期存为 `YYYY-MM-DD`，可直接用于过滤表达式。

```bash
python rag/index/seed_milvus.py
```

脚本默认 drop 后重建，仅用于本地环境。线上按版本新建 collection，再灰度切换。

**验证**：

```python
from rag.retrieving import store
print(store.COLLECTION, store.client().get_collection_stats(store.COLLECTION))
```

期望 `row_count` 与上一步的子块数一致（353）。

---

## 七、第 6 步 · 检索链路 `rag/retrieving/`

一次 `search_refund_policy` 内部有六次数据形变，各占一个文件：

| 步骤 | 文件 | 做什么 | 出错的表现 |
|---|---|---|---|
| 1 改写 | [`pipeline/rewrite.py`](https://github.com/tiltwind/refund-agent/blob/main/rag/retrieving/pipeline/rewrite.py) | 拆多意图、判断要不要法规层，**输出自然语言问句** | 检索到的完全是另一类问题的条款 |
| 2 路由 | [`pipeline/route.py`](https://github.com/tiltwind/refund-agent/blob/main/rag/retrieving/pipeline/route.py) | 平台层 / 法规层各给多少名额与权重 | 该引平台条款却引了法条 |
| 3 过滤 | [`pipeline/filters.py`](https://github.com/tiltwind/refund-agent/blob/main/rag/retrieving/pipeline/filters.py) | 生效日期 + 层级，**只做硬约束** | 明明有条款却一条没召回 |
| 4 召回融合 | [`pipeline/recall.py`](https://github.com/tiltwind/refund-agent/blob/main/rag/retrieving/pipeline/recall.py) | 稠密 + BM25 双路 → RRF（k=20） | 召回了但排名靠后 |
| 5 重排 | [`pipeline/rerank.py`](https://github.com/tiltwind/refund-agent/blob/main/rag/retrieving/pipeline/rerank.py) | cross-encoder + 层级/文档先验 | 候选里有正确答案但没顶上来 |
| 6 装配 | [`pipeline/assemble.py`](https://github.com/tiltwind/refund-agent/blob/main/rag/retrieving/pipeline/assemble.py) | 父块回填 + 去重 + 相邻合并 + 预算截断 | 上下文里有证据但答复没用上 |

编排位于 [`milvus.py`](https://github.com/tiltwind/refund-agent/blob/main/rag/retrieving/milvus.py)，Milvus 存取封装位于 [`store.py`](https://github.com/tiltwind/refund-agent/blob/main/rag/retrieving/store.py)。collection 名和字段只在 `store.py` 定义。

改写输出自然语言问句。召回同时使用 BM25 和稠密向量，分别覆盖精确词项和语义匹配；RRF 融合在应用层完成，各路排名记进 trace。

### 降级路径

| 组件 | 不可用时 |
|---|---|
| 改写模型 | 原文透传，排序掉一档 |
| 重排模型 | 融合分 + 先验加权，打 warn |
| 嵌入模型 | **硬失败** |
| 检索结果为空 | **抛异常**，不返回「未检索到条款」让 Agent 继续判定 |

**验证**：

```python
from rag.retrieving.milvus import MilvusRagService
for s in MilvusRagService().search_policy('金牌会员签收 10 天的耳机未拆封，无理由退货还在窗口期内吗？'):
    print(round(s.score, 3), s.section)
```

期望 top-1 命中 P02 的退货窗口条款。

> 权重（`0.80` 相关性 / `0.20` 先验）、阈值（`MIN_SCORE=0.30`）、RRF 的 `k=20`、法规层 `0.5` 降权
> 目前都是初始值，需要用 query → section 标注集校准。
