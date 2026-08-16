# RefundAgent —— 电商退款处理 Agent

面向生产环境的退款客服 Agent。核心命题是：**把「理解用户」交给模型，把「判定与执行」交给确定性系统**，
让每一笔退款决策可复现、可审计、可回溯。


## 文档

| 文档 | 内容 |
|---|---|
| [政策文档库](doc/policy/README.md) | 语料清单（法规 L01-L05 / 平台 P01-P11）、两层的关系、元数据字段 |
| [Milvus 本地服务](doc/platform/milvus.md) | `scripts/milvus.sh` 的用法与端口 |
| [Langfuse 本地服务](doc/platform/langfuse.md) | 埋点后端的启停 |

## 快速开始

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env               # 填 ANTHROPIC_API_KEY 或 OPENAI_* 三件套
bash scripts/milvus.sh start       # 启 Milvus 2.5+
python knowledge/seed_milvus.py    # 切片 doc/policy/ 并灌库（只需一次）

bash run-main.sh                   # 跑三个演示场景
bash run-main.sh --trace           # 额外打印检索链路每一步的中间产物
```

首次运行会下载约 4.4 GB 的本地嵌入与重排权重。完整步骤与验收标准见 [3 · 实现](doc/get-start/3-impl.md)。

## 当前进度

已落地：`agent/v1`（五步 SOP + 5 个工具）、`services/`（客户 / 订单的 eval 数据源 + RAG 六步检索链路）、
`llm/`（BGE-M3 嵌入 + bge-reranker-v2-m3 重排 + 供应商可切换的对话模型）、`knowledge/`（政策文档切片与灌库）、
`services/telemetry.py`（Langfuse 埋点）、`evals/`（[用例集 d1](evals/dataset/d1/README.md) 27 条 + 数据集自检脚本）。

规划中：`app/main.py` 服务外壳与认证中间件、人工审批、`services/*/prod.py` 下游接入、`evals/` 的跑批与打分、
`agent/v2` 与灰度路由。取舍见 [2 · 设计](doc/get-start/2-design.md)，落点见
[3 · 实现](doc/get-start/3-impl.md) 第十五节。
