# 1 · 架构：组件、链路与目录

本文说明组件边界、请求链路和目录结构。业务口径见 [0 · 需求](https://tiltwind.github.io/refund-agent/doc/get-start/0-requirement.md)。

---

## 一、整体架构

```mermaid
flowchart TD
    Client["用户端<br/>App · H5 · 客服工作台"]

    subgraph Edge["接入层"]
        GW["API 网关<br/>JWT 验签 · strip 伪造 header<br/>限流 · 审计日志"]
    end

    subgraph AgentSvc["Agent 服务（本项目）"]
        direction TB
        Ctx["认证中间件<br/>身份 header → RefundContext<br/>不重复验签"]
        Loop["Agent Loop<br/>agent/v1 · v2 版本并存<br/>灰度路由 + 对比评估"]
        LLM["LLM<br/>低温 · 工具调用"]
        Tools["工具层<br/>schema ↔ 业务动作"]
        HITL["人工审批中间件<br/>高风险 case interrupt"]
        TEL["Telemetry 埋点<br/>OpenTelemetry SDK"]

        Ctx --> Loop
        Loop <--> LLM
        Loop --> HITL
        HITL --> Tools
        Loop -.-> TEL
        Tools -.-> TEL
        LLM -.-> TEL
    end

    subgraph SvcLayer["服务接入层 services/"]
        direction TB
        SC["customer/<br/>客户档案"]
        SU["rule/<br/>资格判定"]
        SO["order/<br/>退款执行"]
        SRC{"request_source"}
        SC --> SRC
        SU --> SRC
        SO --> SRC
    end

    subgraph RagLayer["政策知识库 rag/"]
        direction TB
        SR["retrieving/<br/>六步检索链路<br/>（不分数据源）"]
        IDX["chunking/ + index/<br/>离线切片与灌库"]
    end

    subgraph VectorDB["向量库 · prod 与 eval 共用"]
        KB["Milvus 2.5+<br/>refund_policy_chunks collection<br/>BGE-M3 稠密向量 + BM25 稀疏向量<br/>生效日期 / 层级标量过滤"]
    end

    subgraph Downstream["下游微服务 · prod"]
        direction TB
        USER["用户服务<br/>客户档案 · 会员等级"]
        RULE["规则服务<br/>退款规则引擎"]
        ORDER["订单系统<br/>订单数据 + 退款执行"]
        PAY["支付服务"]
    end

    subgraph LocalData["本地数据源 · 评估"]
        FIX["evals/data/*.json<br/>eval 数据 · 手工构造的规则边界用例<br/>customers · orders"]
    end

    Client -->|"HTTPS + Bearer Token"| GW
    GW -->|"已认证请求<br/>X-Customer-Id · X-Request-Id"| Ctx

    Tools -->|"① 检索政策"| SR
    Tools -->|"② 查客户档案"| SC
    Tools -->|"③ 判定退款资格"| SU
    Tools -->|"④ 执行退款 / 记录拒绝"| SO

    SR -->|"prod / eval 同一条路径"| KB
    IDX -.->|"离线灌库 doc/policy/"| KB
    SRC -->|"prod"| USER
    SRC -->|"prod"| RULE
    SRC -->|"prod"| ORDER
    SRC -.->|"eval"| FIX
    RULE -->|"取订单数据"| ORDER
    ORDER -->|"打款"| PAY

    subgraph Obs["可观测平台"]
        direction TB
        LF["Langfuse<br/>trace · span · generation · score"]
        EVAL["评估流水线<br/>离线回归 + 线上采样打分"]
        LF --> EVAL
    end

    TEL -.->|"OTLP 上报 trace"| LF
    Downstream -.->|"traceparent 透传<br/>下游 span 归入同一 trace"| LF
```

### 组件职责

| 组件 | 职责 | 明确**不**负责 |
|---|---|---|
| API 网关 | JWT 验签、**strip 客户端伪造的身份 header**、注入身份 header、限流、生成 request_id | 业务逻辑、授权判定 |
| 认证中间件 | 读网关注入的身份 header → 类型化 `RefundContext`；**不重复验签**，缺 header 直接 401 | 验签、授权判定 |
| Agent Loop | 编排工具调用、管理对话状态 | 业务规则 |
| 工具层 | schema ↔ 业务动作的双向翻译 | 业务规则、授权判定、协议细节 |
| **服务接入层** | 下游能力的统一入口；客户档案、资格判定与退款执行按 `request_source` 选择 prod / eval 实现 | 业务规则、政策检索 |
| **政策知识库** | `doc/policy/` 的切片灌库与六步检索链路；不分数据源，prod / eval 同一条路径 | 业务规则、下游数据 |
| Milvus 向量库 | 政策条款混合检索（稠密 + BM25）+ 生效日期过滤；**prod 与 eval 共用同一 collection** | — |
| 用户服务 | 客户档案（**自己做归属校验**） | — |
| 规则服务 | **退款规则引擎**：窗口、风控、类目黑名单、商品条件 | 订单数据的存储、资金动作 |
| 订单系统 | 订单数据 + 退款执行（**自己做归属校验**） | 退款规则 |

规则引擎独立成服务，与 Agent、订单系统各自发版。规则服务不存订单数据，判定时带 `X-Acting-User` 向订单系统取数，归属校验由数据所有者执行。

---

## 二、请求链路

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户端
    participant GW as API 网关
    participant AG as Agent 服务
    participant M as LLM
    participant KB as Milvus
    participant US as 用户服务
    participant RL as 规则服务
    participant OD as 订单系统

    U->>GW: POST /refund/chat<br/>Authorization: Bearer {JWT}
    GW->>GW: 验签 · strip 客户端伪造的 X-Customer-Id<br/>限流 · 生成 request_id
    GW->>AG: 转发 + 注入 X-Customer-Id / X-Actor / X-Request-Id
    AG->>AG: 读 header 构造 RefundContext<br/>不重复验签；缺 header 则 401

    Note over AG,M: Agent Loop 开始

    AG->>M: system_prompt + 用户消息 + 工具 schema
    M-->>AG: tool_call: get_customer_info

    AG->>US: GET /customers/me<br/>服务身份 + X-Acting-User
    US->>US: 归属校验
    US-->>AG: 会员等级 / 退款次数 / 名下订单
    AG-->>M: 工具结果

    M-->>AG: tool_call: search_refund_policy
    AG->>AG: 改写 → 路由（六步链路）
    AG->>KB: 双路召回：稠密 + BM25<br/>生效日期 / 层级标量过滤
    KB-->>AG: 两份排名列表 → RRF 融合
    AG->>AG: 重排（cross-encoder）→ 装配（父块回填）
    AG-->>AG: 条款原文 + 来源 + 生效日期 + 相关性理由
    AG-->>M: 工具结果

    M-->>AG: tool_call: check_refund_eligibility(order_id, ...)
    AG->>RL: POST /refund-eligibility<br/>X-Acting-User: {customer_id}
    RL->>OD: GET /orders/{id}
    OD->>OD: 归属校验
    OD-->>RL: 订单数据（取不到即视为不存在）
    RL->>RL: 规则引擎判定
    RL-->>AG: 通过 / 不通过 / 需补充：xxx
    AG-->>M: 工具结果

    alt 返回「需补充」
        M-->>U: 向用户追问退款原因或商品状态
        U->>AG: 补充信息 → 重新判定
    else 判定通过
        M-->>AG: tool_call: execute_refund
        AG->>OD: POST /refunds<br/>Idempotency-Key: {request_id}
        OD-->>AG: 退款单号 R9001
    else 判定不通过
        M-->>AG: tool_call: record_refund_denial
        AG->>OD: POST /refund-denials
        OD-->>AG: 受理编号 D9001
    end

    AG-->>M: 单号
    M-->>AG: 答复（含单号 + 政策条款引用）
    AG-->>U: 最终答复
```

最终答复必须包含单号，单号由终局工具返回。

---

## 三、目录结构（规划）

```
refund-agent/
├── app/                          # 服务外壳
│   ├── main.py                   # FastAPI 入口 + 版本路由
│   ├── context.py                # RefundContext 定义
│   └── middleware/
│       ├── auth.py               # 读网关注入的身份 header → RefundContext
│       └── approval.py           # 人工审批 interrupt
│
├── agent/                        # Agent 版本并存，用于对比评估与灰度
│   ├── v1/                       # 基线：当前线上版本
│   │   ├── prompt.py             # SYSTEM_PROMPT
│   │   ├── graph.py              # create_agent 装配
│   │   ├── tools.py              # 工具定义（schema ↔ 业务动作）
│   │   └── meta.yaml             # 版本号 / 模型 / 温度，随 trace 上报
│   ├── v2/                       # 候选：本次改动
│   │   └── ...
│   └── registry.py               # 版本注册、灰度选择
│
├── services/                     # 服务接入层：下游能力的统一入口
│   ├── base.py                   # 服务身份 + X-Acting-User + traceparent + 重试熔断
│   ├── factory.py                # 按 request_source 选实现（rag 除外）
│   ├── errors.py                 # EvalDataMissError
│   ├── eval_store.py             # 加载 evals/data + 会话隔离（并发跑批不互相污染）
│   ├── telemetry.py              # OTel 埋点 → Langfuse：trace 属性组装 + PII 脱敏
│   ├── customer/
│   │   ├── protocol.py           # CustomerService 接口 + 数据模型
│   │   ├── prod.py               # → 用户服务
│   │   └── eval.py               # → evals/data/customers.json
│   ├── rule/
│   │   ├── protocol.py           # RuleService 接口 + EligibilityResult
│   │   ├── prod.py               # → 规则服务（规则引擎在对侧）
│   │   └── eval.py               # → 本地规则引擎副本
│   └── order/
│       ├── protocol.py           # 退款执行 + 拒绝落库
│       ├── prod.py               # → 订单系统
│       └── eval.py               # → evals/data/orders.json（终局动作 stub）
│
├── rag/                          # 政策知识库：离线灌库 + 在线检索
│   ├── chunking/                 # doc/policy/*.md → 父子块
│   │   ├── model.py              # DocMeta / Chunk；块头拼装
│   │   ├── markdown.py           # frontmatter + 标题树 + 段落/表格/代码
│   │   ├── semantic.py           # 超长段落的语义切分兜底
│   │   └── policy.py             # 编排 + 切分参数（320 / 512 / overlap=0）
│   ├── index/
│   │   └── seed_milvus.py        # 切片 + 建表 + 灌库，入库前硬卡 token 长度
│   └── retrieving/
│       ├── protocol.py           # PolicySection（内容+来源+时间+理由）+ RetrievalTrace
│       ├── store.py              # Milvus 连接与字段定义的单一定义点
│       ├── milvus.py             # 六步链路的编排，**唯一实现**，prod / eval 共用
│       └── pipeline/             # 一步一个文件，每步可单独观测
│           ├── rewrite.py        # ① 改写：拆多意图 → 自然语言问句
│           ├── route.py          # ② 路由：平台层 / 法规层的名额与权重
│           ├── filters.py        # ③ 过滤：生效日期 + 层级，只做硬约束
│           ├── recall.py         # ④ 召回融合：稠密 + BM25 → RRF
│           ├── rerank.py         # ⑤ 重排：cross-encoder + 层级/文档先验
│           └── assemble.py       # ⑥ 装配：父块回填 + 去重 + 预算截断
│
├── llm/                          # 模型层：与业务无关，被多条链路共用
│   ├── chat.py                   # 供应商与模型名的唯一解析处
│   ├── device.py                 # cuda > mps > cpu
│   ├── embedding/bge_m3.py       # BGE-M3 稠密向量 + tokenizer 计长（切分要用）
│   └── rerank/bge_reranker.py    # bge-reranker-v2-m3，可关闭（降级打 warn）
│
├── doc/
│   ├── policy/                   # **政策语料的单一事实源**，直接被切片入库
│   │   ├── law/                  # L01-L05 法律法规：法定底线
│   │   └── platform/             # P01-P11 平台政策：与消费者的直接约定
│   ├── get-start/                # 需求 / 架构 / 设计 / 实现
│   └── platform/                 # 依赖服务的本地部署说明（Milvus / Langfuse）
│
├── evals/
│   ├── data/                     # eval 数据源（JSON）
│   │   ├── customers.json
│   │   └── orders.json           # signed_days_ago 相对天数 + _note 标注守的边界
│   ├── dataset/                  # 评估用例集，按版本目录并存
│   │   └── d1/
│   │       ├── cases.jsonl       # 27 条用例，单一事实源
│   │       └── README.md         # 绑定关系 / 字段口径 / 指标 / 覆盖矩阵
│   ├── validate_cases.py         # 数据集自检：期望值 vs 规则引擎
│   ├── offline.py                # 离线回归（单版本）
│   ├── compare.py                # 多版本对比 v1 vs v2
│   ├── online.py                 # 线上采样打分
│   └── archive/                  # 每轮实验的 trace 与报告归档
│
├── scripts/
│   └── milvus.sh                 # 本地 Milvus standalone 的启停（start/stop/status/logs）
│
├── main.py                       # 离线演示入口：跑三个典型场景
├── run-main.sh                   # 加载 .env 后跑 main.py，顺带校验模型凭据齐不齐
├── .env.example                  # 配置占位符（密钥只放 .env，已 gitignore）
└── README.md                     # 仓库索引，正文在 doc/get-start/
```

> v1 已落地的是 `agent/v1`、`services/`、`rag/`、`llm/`、`doc/policy/`、两个入口脚本，
> 以及 `evals/` 下的用例集 `dataset/d1` 与自检脚本 `validate_cases.py`；
> `app/main.py`、`app/middleware/`、`agent/v2/`、`evals/` 的跑批与打分（`offline` / `compare` / `online`）仍是规划。

**四个目录的边界**：`agent/` 放提示词、流程与工具描述，按版本并存；`services/` 放下游能力契约，跨版本共享；`rag/` 放政策知识库的灌库与检索，语料本身在 `doc/policy/`，改版走灌库而不是改代码；`evals/` 消费前三者，用同一套 eval 数据和同一个政策 collection 跑不同 agent 版本。
