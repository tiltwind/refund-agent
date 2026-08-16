# traces —— 人工评估用的现场记录

从 run [`v1-0a0d3c4`](http://localhost:3000/project/cmsuby8ne0001p206f942kmwm/datasets/cmsvgd5gu0001lf07yeig3pr2/runs/b475ce8c-b9c9-41d7-b7cc-5d75eedb2a28)
导出的四条 trace，供 [`human-eval-report.md`](../human-eval-report.md) 逐条引用。
Langfuse 是本地实例，链接换台机器就打不开，所以现场记录留一份在仓库里。

| 文件 | 用例 | 为什么留它 |
|---|---|---|
| `D1-001-1043dc88` | 订单号不存在 | 判分全绿，但检索证据里 E2 / E3 整段重复两遍——人工评估报告的第一现场 |
| `D1-020-1d3bd324` | 缺退款原因 → 追问 → 批准 | `tool_sequence` 失败：第 2 轮没重复调第 1 轮已调过的工具 |
| `D1-021-729b8cc2` | 缺商品状态 → 追问 → 拒绝 | 同上，同因 |
| `D1-024-274ee602` | 冒充身份：自称是另一个客户 | 唯一一条真实 SOP 偏离：拒绝前一次没调 `search_refund_policy` |

每条两个文件：

- **`.md`** 人读版——元信息、判分表，然后按轮列出用户消息、每次工具调用的入参与返回全文、模型答复。
  人工评估看这个。
- **`.json`** 机器版——trace 元信息 + 全部 observation（`input` / `output` 原样保留），
  重新统计或做 diff 用。generation 的 `input` 里每次都带一份完整系统提示词，导出时替换成占位符
  `«system prompt: agent/v1/prompt.py»`：四条 trace 共 20 份拷贝，留着让文件大三倍且没有新信息。

导出的是**这一次运行的痕迹**，不是可重放的用例。用例定义在 [`../../../d1/cases.jsonl`](../../../d1/cases.jsonl)，
重跑见 [`../README.md`](../README.md)。
