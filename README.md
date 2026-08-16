# RefundAgent —— 电商退款处理 Agent

面向生产环境的退款客服 Agent。核心命题是：**把「理解用户」交给模型，把「判定与执行」交给确定性系统**，
让每一笔退款决策可复现、可审计、可回溯。

本文件只是索引，正文在 `doc/` 下。

## 文档

| 文档 | 内容 |
|---|---|
| [0 · 需求](doc/get-start/0-requirement.md) | 业务场景、五步处理 SOP、判定规则与优先级、政策语料范围、验收口径 |
| [1 · 架构](doc/get-start/1-architecture.md) | 整体架构与组件职责、请求链路时序、目录结构与四个目录的边界 |
| [2 · 设计](doc/get-start/2-design.md) | 身份与授权、工具设计、服务接入层与数据源切换、幂等、可观测、评估闭环、技术栈与检索链路取舍 |
| [3 · 实现](doc/get-start/3-impl.md) | 一步步搭出 v1：环境、语料、模型层、切片灌库、检索链路、服务接入层、Agent、入口、埋点，含验收清单与故障排查 |
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
`services/telemetry.py`（Langfuse 埋点）。

规划中：`app/main.py` 服务外壳与认证中间件、人工审批、`services/*/prod.py` 下游接入、`evals/` 评估流水线、
`agent/v2` 与灰度路由。取舍见 [2 · 设计](doc/get-start/2-design.md)，落点见
[3 · 实现](doc/get-start/3-impl.md) 第十五节。
