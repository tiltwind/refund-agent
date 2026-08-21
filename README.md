# RefundAgent —— 电商退款处理 Agent

实现一个评估驱动进化的 Agent。

## 持续评估闭环

Agent 的每次改动都先用固定评估集与线上版本对比，通过后再发布；线上发现的新问题经脱敏和标注后回流为评估用例，防止同类问题再次出现。

```mermaid
flowchart LR
    CHANGE[改进 Agent] --> EVAL[离线评估]
    EVAL --> COMPARE[新旧版本对比]
    COMPARE -->|通过| RELEASE[发布上线]
    COMPARE -->|退化| CHANGE
    RELEASE --> MONITOR[线上监控]
    MONITOR -->|新问题回流| CASES[补充评估用例]
    CASES --> EVAL
```

闭环关注三件事：评估用例与业务规则一致、每次改动没有引入退化、线上失效模式能够沉淀为新的回归用例。完整设计见[持续评估闭环](https://tiltwind.github.io/refund-agent/doc/get-start/2-design.md#六持续评估闭环)。

## 文档

| 文档 | 内容 |
|---|---|
| [0 · 需求](doc/get-start/0-requirement.md) | 业务场景、五步处理 SOP、判定规则与优先级、政策语料范围、验收口径 |
| [1 · 架构](doc/get-start/1-architecture.md) | 整体架构与组件职责、请求链路时序、目录结构与四个目录的边界 |
| [2 · 设计](doc/get-start/2-design.md) | 身份与授权、工具设计、服务接入层与数据源切换、幂等、可观测、评估闭环、技术栈与检索链路 |
| [3 · 政策知识库](doc/get-start/3-rag-impl.md) | 环境、语料、模型层、切片、Milvus 灌库、六步检索链路，含验收清单与故障排查 |
| [4 · 检索评测数据集](doc/get-start/4-rag-dataset.md) | 从政策块反向生成 query 与参考答案、生成的四个坑、分层抽样、自检与人工抽检 |
| [5 · 检索评测](doc/get-start/5-rag-eval.md) | 两个 judge 指标与三个排序指标怎么联合归因、跑批的三种出口怎么记账、抖动从哪来 |
| [6 · 装配 Agent](doc/get-start/6-agent-impl.md) | 身份上下文、服务接入层与规则引擎副本、`agent/v1` 提示与工具、演示入口、埋点 |
| [7 · 指标与数据集](doc/get-start/7-agent-dataset.md) | 从失效模式抽出五个核心指标，再由指标反推数据集覆盖、用例字段与判分实现 |
| [8 · 跑实验](doc/get-start/8-agent-eval.md) | 跑批与打分脚本、HTML 报告、人工读 trace、ex-1 的关键问题、指标本身的缺口与排期 |
| [9 · 线上监控](doc/get-start/9-online-monitor.md) | 线上指标盘、trace 上报改造、链路内自洽打分、正确性与忠实度评估器、看板与告警、失败回流 |
| [Agent 评估核心步骤](doc/get-start/agent-eval-core-steps.md) | 评估体系的通用思路：从失效模式倒推指标、指标含义、由指标反推数据集、trace 该报哪些字段、Langfuse 要配置哪些监控 |
| [用 Langfuse 评估 Agent：核心代码](https://tiltwind.github.io/refund-agent/doc/get-start/agent-langfuse-eval-codes.md) | Langfuse 数据集、执行钩子、逐条评分、run 级聚合与实验启动的核心代码 |