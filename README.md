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
| [2 · 设计](doc/get-start/2-design.md) | 身份与授权、工具设计、服务接入层与数据源切换、幂等、可观测、评估闭环、技术栈与检索链路取舍 |
| [3 · 实现](doc/get-start/3-impl.md) | 一步步搭出 v1：环境、语料、模型层、切片灌库、检索链路、服务接入层、Agent、入口、埋点，含验收清单与故障排查 |
| [4 · 数据集与指标](doc/get-start/4-dataset.md) | 用例集格式与设计取舍、自检与推送、指标是怎么定下来的、九个指标各自怎么算 |
| [5 · 跑实验](doc/get-start/5-experiment.md) | 跑批与打分脚本、HTML 报告、人工读 trace、ex-1 的关键问题、指标本身的缺口与排期 |
