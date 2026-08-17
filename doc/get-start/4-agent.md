# 4 · 装配 RefundAgent v1

接着 [3 · 政策知识库](https://tiltwind.github.io/refund-agent/doc/get-start/3-rag.md) 往上搭：身份上下文、服务接入层、Agent 本体、演示入口和埋点，最后用 `bash run-main.sh` 验证三个退款场景。业务、架构和设计分别见 [0 · 需求](https://tiltwind.github.io/refund-agent/doc/get-start/0-requirement.md)、[1 · 架构](https://tiltwind.github.io/refund-agent/doc/get-start/1-architecture.md) 和 [2 · 设计](https://tiltwind.github.io/refund-agent/doc/get-start/2-design.md)。

---

## 一、目标

客户档案和订单读取本地 eval 数据，政策检索连接 Milvus，模型使用 Anthropic 或 OpenAI 兼容接口。

```
用户消息
  └─ RefundContext（customer_id / request_id / request_source）—— 演示脚本直接构造，线上由认证中间件注入
     └─ Agent Loop（五步 SOP 写死在系统提示里）
        ├─ get_customer_info        → services/customer/eval.py   → evals/data/customers.json
        ├─ search_refund_policy     → services/rag/milvus.py      → Milvus（六步检索链路）
        ├─ check_refund_eligibility → services/rule/eval.py       → 规则引擎副本
        └─ execute_refund / record_refund_denial → 落决策流水，返回单号
     └─ 答复（必须写明单号）
```

### 构建顺序

| 步骤 | 建什么 | 能独立验证吗 |
|---|---|---|
| 1 | `app/context.py` + `services/` 业务接入层 | ✅ 直接调规则引擎打边界 |
| 2 | `agent/v1/` 提示 + 工具 + 装配 | ✅ 单跑一次 invoke |
| 3 | `main.py` / `run-main.sh` 入口 | ✅ 三个场景 |
| 4 | `services/telemetry.py` 埋点（可选） | ✅ Langfuse 上看到调用树 |

---

## 二、第 1 步 · 上下文与服务接入层

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
├── rule/               # protocol.py / prod.py（留桩）/ eval.py（含规则引擎副本）
├── order/              # protocol.py / prod.py（留桩）/ eval.py（终局动作 stub）
└── rag/                # 只有一个实现，不分数据源
```

资格判定和退款执行是两个独立服务，边界见 [1 · 架构](https://tiltwind.github.io/refund-agent/doc/get-start/1-architecture.md) 第一章。

工厂函数按数据源返回实现：

```python
def rule_service(ctx: RefundContext) -> RuleService:
    match ctx.request_source:
        case "prod": return ProdRuleService()
        case "eval": return EvalRuleService()
        case other:  raise ValueError(f"unknown request_source: {other}")   # 不 fallback 到 prod
```

未知取值直接报错，不回退到 prod。`rag_service()` 始终返回 Milvus 实现。

### 规则引擎副本与 eval 数据

[`services/rule/eval.py`](https://github.com/tiltwind/refund-agent/blob/main/services/rule/eval.py) 维护一份线上规则引擎的等价副本，供离线使用。
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

`eval_store` 用 `contextvars` 给每条用例发一份独立数据副本，并发跑批时互不影响。

**验证**（不连任何外部服务）：

```python
from services.rule.eval import EvalRuleService
s = EvalRuleService()
print(s.check_eligibility('O2001', 'C1001', '无理由', '未拆封'))   # 金牌 10 ≤ 15 → 通过
print(s.check_eligibility('O2006', 'C1004'))                       # 20 > 15 → 硬否决，不该追问
print(s.check_eligibility('O2004', 'C1004'))                       # 未命中硬否决 → 需补充
print(s.check_eligibility('O2009', 'C1005', '质量问题'))           # 金牌 + 高风险 → 风控优先
```

期望依次是：通过（可退 899.0）、不通过（超出所有窗口）、需补充（退款原因）、不通过（高风险转人工）。

---

## 三、第 2 步 · Agent `agent/v1/`

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

1. **不向用户索要客户 ID，不采信用户自称的身份**：身份由系统注入，工具自动使用当前登录客户；
2. **`reason_type` 如实反映用户陈述**：不想要 / 不合适 / 买错了一律是「无理由」，
   只有明确指出缺陷、损坏、故障、变质才是「质量问题」，**不得为了让判定通过而改写这个参数**；
3. **先落库拿到单号，再写答复，并在答复里写明这个编号**。

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

```python
from agent import registry
from app.context import RefundContext
r = registry.get('v1').invoke(
    {'messages': [{'role': 'user', 'content': '订单 O2004 的跑鞋没拆封，想退。'}]},
    context=RefundContext(customer_id='C1004', request_id='t-1', request_source='eval'))
for m in r['messages']:
    for c in getattr(m, 'tool_calls', None) or []:
        print('▶', c['name'], c['args'])
print(r['messages'][-1].text)
```

期望工具调用顺序是 `get_customer_info → search_refund_policy → check_refund_eligibility → execute_refund`，
且最终答复里带着 `R9000` 这样的退款单号。

---

## 四、第 3 步 · 入口与跑通

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

**验证**：三个场景各自给出预期结论，末尾的决策流水里有三条记录，每条都带 `actor` 与 `request_id`。

---

## 五、第 4 步 · 埋点

[`services/telemetry.py`](https://github.com/tiltwind/refund-agent/blob/main/services/telemetry.py) 使用 Langfuse `CallbackHandler` 接收 LangGraph 回调。

```python
result = agent.invoke(
    {"messages": [...]},
    context=ctx,                                     # 业务身份 → 工具层
    config=telemetry.trace_config(ctx, meta, ...),   # 运行时回调 → Langfuse
)
```

`context` 传递业务身份，`config` 传递运行时回调。

四条约束：

1. **缺密钥静默降级**：`trace_config()` 返回空 dict，调用方不必写 `if`，主链路照常跑。
2. **脱敏做在 SDK 的 `mask` 钩子里，不做在调用点**：所有 span 的 input/output 统一过一遍。
3. **`customer_id` 上报加盐哈希**：同一个人的多次请求可串联，不还原真实身份。
4. **v1 起就上报 `agent_version` / `prompt_version`**：两个值来自 `meta.yaml`，由 `registry.meta()`
   传进 `trace_config()`，既进 metadata 也进 tag。

注意：

- 短命脚本要显式调用 `flush()`；`main.py` 在 `finally` 中调用。
- 启动时运行 `auth_check`，检查凭据和服务状态。

本地起 Langfuse 见
[`doc/platform/langfuse.md`](https://tiltwind.github.io/refund-agent/doc/platform/langfuse.md)，
把 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` 填进 `.env` 即可。

**验证**：启动行显示 `Langfuse: on → http://localhost:3000`，
Langfuse UI 上能看到 `refund-chat:*` 这条 trace，展开后图节点、工具、generation 自动成树。
