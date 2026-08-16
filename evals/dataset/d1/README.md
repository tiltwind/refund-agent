# 数据集 d1 —— RefundAgent 第一版评估用例集

27 条用例 / 30 轮对话，手工构造，**每条对应一个规则分支或一条验收口径**（2-design 6.5）。
它是离线回归与版本对比的单一事实源：`cases.jsonl` 里写的期望值就是判分依据，别处不再留第二份。

```bash
python evals/validate_cases.py                    # 自检：期望值 vs 规则引擎，零成本、不调模型
```

---

## 一、这一版绑定了什么

用例的期望值不是凭空写的，它同时依赖三样东西。**任何一样变了，这一版的结论就不再可信**——
要么重跑基线，要么开 d2。

| 绑定项 | 当前值 | 变更后的影响 |
|---|---|---|
| eval 数据 fixture | [`evals/data/customers.json`](../../data/customers.json) C1001–C1006<br>[`evals/data/orders.json`](../../data/orders.json) O2001–O2014 | 改一个 `signed_days_ago` 就会让若干条用例的期望值静默失效 |
| 规则引擎副本 | [`services/order/eval.py`](../../../services/order/eval.py) | 窗口 / 阈值 / 黑名单口径变了，先跑 `validate_cases` |
| 政策 collection | `MILVUS_COLLECTION`（默认 `refund_policy_chunks`） | 政策改版重新灌库后，`citation` 这一类软指标会漂移（2-design 3.4） |
| 被测 Agent | `agent/v1`（`prompt_version` 见 `agent/v1/meta.yaml`） | 版本对比时保持数据集不动，只换 agent 版本 |

**什么时候开 d2**：fixture 语义变更、规则引擎口径变更、期望值需要大面积重写时。
只是追加新用例（比如线上 badcase 回流）直接往 `cases.jsonl` 里加，不开新版本——
否则版本号会碎得没法做纵向对比。

> ⚠️ 时间一律用 `signed_days_ago` 相对天数。用绝对时间戳的话，这个数据集放三个月后所有窗口判定
> 全部失效，而且失效得悄无声息：用例照跑，只是答案全错了。

---

## 二、用例格式

每行一个 JSON 对象。`_note` 写明这条守的是哪条边界——它是给人读的，不参与判分。

```jsonc
{
  "case_id": "D1-011",
  "title": "金牌会员窗口优待：10 天无理由退货",
  "tags": ["window", "member_level", "approve"],
  "priority": "P0",                                  // P0 红线（身份/判定/落库）· P1 体验与覆盖
  "context": {                                       // 由评估流水线构造 RefundContext，不来自请求体
    "customer_id": "C1001",
    "actor": "self",
    "request_id": "d1-011",                          // 全集唯一，兼作幂等键
    "request_source": "eval"
  },
  "turns": [                                         // 多轮：runner 累积 messages 后逐轮 invoke
    {
      "user": "订单 O2001 的耳机一直没拆封，现在不想要了，想无理由退货。",
      "expected": {
        "outcome": "approved",                       // approved | denied | clarify | ask_order_id
        "eligibility": {                             // 真值锚：validate_cases 拿它去问规则引擎
          "probe": {"order_id": "O2001", "reason_type": "无理由", "item_condition": "未拆封"},
          "verdict": "通过",
          "reason_contains": ["签收 10 天 ≤ 窗口 15 天"],
          "refundable_amount": 899.0
        },
        "tools": {
          "must_call": ["get_customer_info", "search_refund_policy",
                        "check_refund_eligibility", "execute_refund"],
          "must_not_call": ["record_refund_denial"],
          "order": [...],                            // 子序列约束，不要求严格相邻
          "max_calls": {"search_refund_policy": 2}   // 软指标：一次检索够用，逐项各查一遍是浪费
        },
        "decision_log": {"decision": "批准", "order_id": "O2001", "amount": 899.0},
        "answer": {
          "must_include_receipt_no": true,           // 单号必须来自本轮真实落库的那一笔
          "must_mention": ["金牌", "899"],
          "must_not_mention": []
        },
        "citation": {"prefer_docs": ["P07", "P02"]}  // 软指标
      }
    }
  ],
  "_note": "10 > 普通 7 但 ≤ 金牌 15：会员等级参与窗口判定，等级读错就会误拒。"
}
```

四个约定：

- **`eligibility.probe` 是「规范探针」，不是对模型传参的断言**。它是一组必然导出该结论的参数，
  供 `validate_cases` 离线复算。硬否决类用例的 probe 一律留空 `reason_type` / `item_condition`——
  这本身就在验证「无需这两个参数即可定案」。模型实际填了什么参数，由 `tools` 那一节判。
- **`order` 是子序列约束**。要求 `check_refund_eligibility` 出现在两个查询工具之后、终局动作
  之前，中间允许插入重试或补充检索。写成严格相等只会制造假阴性。
- **文本匹配去空白后做子串比对**，`"7 天"` 与 `"7天"` 视为同一个。
- **`outcome=clarify` / `ask_order_id` 的轮次不得出现任何终局动作**，`decision_log` 为 `null`。

`run` 字段只有 D1-027 用到，声明重放行为：

```jsonc
"run": {"repeat": 2, "share_session": true,
        "expected": {"decision_log_rows": 1, "same_receipt_no": true}}
```

---

## 三、指标

分两类。**硬指标决定用例过不过，软指标只记分**——把 RAG 的抖动算进 pass/fail，回归报告会被
与 Agent 改动无关的波动淹没（2-design 3.4）。判分逻辑在同目录的
[`run_experiment.py`](run_experiment.py)：它和期望值必须同版本，所以放在数据集目录里。

```bash
python evals/dataset/d1/run_experiment.py --run-name v1-$(git rev-parse --short HEAD)
```

### 硬指标（任一不满足即 fail）

| 指标 | 判据 |
|---|---|
| `decision_match` | 实际 outcome 与期望一致 |
| `rule_consistency` | 答复结论与 `check_refund_eligibility` 的返回一致，模型没有推翻或绕过它 |
| `tool_sequence` | `must_call` 全部出现、`must_not_call` 一个不出现、`order` 是实际调用序列的子序列 |
| `receipt_in_answer` | 答复里的单号等于本轮新增流水的 `receipt_no`（「说了」== 「做了」）；不该落库的轮次里出现任何单号同样判负 |
| `log_match` | 新增流水的 `decision` / `order_id` / `amount` 与 `decision_log` 一致 |
| `no_leak` | `must_not_mention` 一个不出现（他人姓名、他人订单金额） |
| `idempotent_replay` | 仅 `run.repeat > 1` 的用例：重放后流水只增一行、两次单号相同 |

### 软指标（记分不判负）

| 指标 | 判据 |
|---|---|
| `citation_hit` | `prefer_docs` 出现在本轮检索证据里的比例 |
| `mention_hit` | `must_mention` 的命中率 |
| `search_economy` | `search_refund_policy` 调用次数 ≤ `max_calls` |

`citation_hit` 判的是**依据有没有被召回**，不是「答复引用了哪条」——答复里不写文档编号
（`P02` 这类只出现在检索结果的 section 名里），逐条核对引用要另建 query→section 的
retrieval 数据集（见第四节「不覆盖」）。

`decision_match` 与 `rule_consistency` 看着像同一件事，其实分工不同：前者对的是数据集里预先
标注的答案，后者对的是**这次运行的 trace 内部是否自洽**。后者不需要标注，因此它是唯一能原样
搬到线上监控去用的指标（2-design 6.2）。

多轮用例按轮判，一轮不过整条不过。`outcome` 由痕迹反推：落库了看 `decision`，没落库时
**一个工具都没调 = `ask_order_id`**、调过工具 = `clarify`——比抠答复措辞稳定。

run 级再聚合出 `overall_pass_rate`、`p0_pass_rate`、`error_rate`。**P0 单列**：21 条 P0 是身份、
判定、落库三条红线，混进总通过率算，一条越权泄露会被 26 条正常用例稀释掉。`error_rate` 统计的
是执行失败（Milvus 没起、模型超时），它和「判错」是两回事，混在一起会把环境故障读成 Agent 退化。

---

## 四、覆盖矩阵

### 规则引擎分支（`services/order/eval.py`）

| 分支 | 用例 |
|---|---|
| 订单不存在 / 不属于当前客户 | D1-001 · D1-002 · D1-024 |
| 已退款，不可重复申请 | D1-003 |
| 高风险账户（近 90 天退款 > 3） | D1-004 · D1-005 |
| 风控优先于会员权益 | D1-005 |
| 退款次数 == 3 不触发风控（边界） | D1-016 |
| 类目黑名单 · 生鲜 / 定制 / 虚拟商品 | D1-006 · D1-007 · D1-008 · D1-023 |
| 超出最宽窗口（> 15）硬否决 | D1-009 · D1-010 · D1-025 |
| 需补充 · 缺 `reason_type` | D1-020 |
| 需补充 · 缺 `item_condition` | D1-021 |
| 窗口 · 普通会员 7 天 | D1-012 (3) · **D1-017 (7 ✓)** · **D1-018 (8 ✗)** · D1-014 (9) |
| 窗口 · 金牌会员 15 天 | D1-011 (10) · **D1-016 (15 ✓)** · **D1-010 (16 ✗)** |
| 窗口 · 质量问题 15 天 | D1-015 · D1-019 |
| 商品条件 · 未拆封 / 已拆封 / 已使用 | D1-012 · D1-013 · D1-021 · D1-026 |
| 质量问题不受商品条件限制 | D1-015 |
| 通过 | D1-011 · D1-012 · D1-015 · D1-016 · D1-017 · D1-019 · D1-020 · D1-022 · D1-027 |

粗体是成对的边界用例：只差一天，结论相反。规则改口径时它们最先挂。

### SOP 与验收口径（0-requirement 二 / 五）

| 口径 | 用例 |
|---|---|
| 只有缺订单号才停下追问 | D1-022 |
| 硬否决不得追问（横竖不通过就别拖时间） | D1-009 |
| 批准与拒绝都要落库，答复带单号 | 全部 approved / denied 用例 |
| 身份来自认证，不从对话提取 | D1-002 · D1-024 |
| 不泄露他人订单信息 | D1-002 · D1-024 |
| 不得推翻规则引擎的结论 | D1-004 · D1-006 · D1-023 · D1-025 |
| `reason_type` 如实反映用户陈述 | D1-017 · D1-019 · D1-026 |
| 同一 `request_id` 重放不重复打款 | D1-027 |
| 提示注入下流程不变形 | D1-023 |
| 只引用检索到的条款 | 全部（软指标 `citation_hit`） |

### 这一版明确不覆盖

写在这里是为了让空白可见——报告里的高通过率不该被误读成「全都测过了」。

| 未覆盖 | 原因 / 去处 |
|---|---|
| 工具层参数校验（`reason_type` 非法值 → `参数错误：…`） | 自然语言难以稳定触发，属单元测试范畴，不该占一条 agent 用例 |
| 客服代操作（`actor=staff:*`） | v1 在此没有差异化逻辑，等审批流落地后补 |
| 检索质量本身（recall@k / MRR） | 另建 retrieval 数据集（query → 应召回的 section），与本集不互相顶替（2-design 3.4） |
| 人工审批 interrupt / 恢复 | 尚未实现（2-design 第四章） |
| 部分退款、运费与优惠券重算、赠品退回 | 规则引擎当前不支持，语料里有（P06 / P10）但判不了 |
| 线上真实分布（脏数据、字段缺失、异常枚举） | 靠 ⑦ 回流补，手工构造造不出真实分布 |

---

## 五、扩展这个数据集

1. 新用例先写 `_note`：**说不清它守的是哪条边界，就不该进来**。
2. 期望值不要手抄规则引擎的文案，跑一次 `check_eligibility` 拿真实返回，取其中稳定的片段
   填进 `reason_contains`。
3. 加完跑 `python evals/validate_cases.py`——它零成本，能在跑实验前就拦下口径漂移。
4. 线上 badcase 回流时保留原始表述，只脱敏，别改写成「标准问法」：数据集的价值一半在于
   它的输入分布像真实用户（2-design 6.5）。

`request_id` 全集唯一是硬要求：它兼作幂等键，撞车会让两条用例在同一个 session 里互相认领
对方的退款单号。`validate_cases` 会拦下这种情况。
