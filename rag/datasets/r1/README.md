# 数据集 r1 —— 检索评测样本集

102 条样本，反向生成：从块出发写 query，`seed_chunk_id` 在生成时就确定。它直接喂
`search_policy`，不经过 Agent，测的是[六步检索链路](https://github.com/tiltwind/refund-agent/blob/main/rag/retrieving/milvus.py)本身。

```bash
python rag/evals/generate_cases.py       # 重新生成（抽样稳定，措辞可能有细微出入）
python rag/evals/generate_claims.py      # 给已有样本补 claims，不动 query
python rag/evals/validate_cases.py       # 自检：ID 有效性、可溯源、分层覆盖，不调模型
python rag/evals/push_dataset.py         # 推成 Langfuse 数据集 retrieval-cases-r1
```

设计口径见 [4 · 检索评测数据集](https://tiltwind.github.io/refund-agent/doc/get-start/4-rag-dataset.md)，
指标与归因见 [5 · 检索评测](https://tiltwind.github.io/refund-agent/doc/get-start/5-rag-eval.md)。

---

## 一、这一版绑定了什么

`seed_chunk_id` 由切片位置派生（`{doc_id}#{parent_seq:03d}:{chunk_index:02d}`），不是内容哈希。
下面任何一项变了，期望值都可能**静默失效** —— 用例照跑，只是 Recall 全线暴跌，看上去像检索退化。

| 绑定项 | 当前值 | 变更后 |
|---|---|---|
| 政策语料 | `doc/policy/` @ `fc6ef3b`（16 篇 / 353 个子块） | 增删小节会让该文档后续所有 ID 偏移，重新生成受影响文档的样本 |
| 切片参数 | `320 / 512 / overlap=0`（`rag/chunking/policy.py`） | 参数变了整个数据集作废，开重建数据集 |
| collection | `refund_policy_chunks` | 重新灌库后先跑自检 |
| 嵌入模型 | `BAAI/bge-m3`，`max_length=1024` | 换模型等于换向量空间，基线要重跑 |
| 生成模型 | `deepseek-v4-flash`，温度 0 | 只影响新增样本的措辞，已有样本不动 |
| claim 拆分 | `deepseek-v4-pro`，温度 0（`meta.claims_by`） | 重拆会改 Context Recall 的分母，历史分数不再可比 |
| 抽样种子 | `SEED = 20260817` | 改它等于换一批种子块，全集要重抽检 |

**什么时候开重建数据集**：切片参数变更、语料大改版、期望值大面积失效。只是补样本就直接往`cases.jsonl` 里加 —— 版本号碎了就没法做纵向对比。

---

## 二、样本格式

每行一个 JSON 对象，注释是说明，实际文件里没有。

```jsonc
{
  "case_id": "R1-082",
  "query": "我拆开包装看了一眼，没用过，这种还能七天无理由退吗？要是能退，寄回去的运费是不是得我自己掏？",
  "style": "colloquial",                          // formal | colloquial，分档报指标
  "type": "multi_hop",                            // single | multi_hop | unanswerable

  "seed_chunk_id": ["P05#002:00", "P06#001:00"],  // Recall 的真值，multi_hop 全中才算命中
  "reference_answer": "已拆封但未实际使用的商品不支持无理由退货……寄回运费由消费者承担……",
  "claims": [                                     // 参考答案拆成的原子事实
    "已拆封但未实际使用的商品不支持无理由退货。",   // Context Recall 逐条判有没有被
    "无理由退货的寄回运费由消费者承担。"            // 检回的上下文支撑，分母就是条数
  ],

  "meta": {
    "doc_id": "P05+P06",                          // 跨块样本用 + 连接
    "layer": "platform",
    "kind": "table+text",
    "section": "第二条 商品状态的三档划分 / 第二条 无理由退货的运费 > 2.1 寄回运费",
    "overlap_ratio": 0.194,                       // query 与种子块正文的实词重叠率
    "generated_by": "openai:deepseek-v4-flash",
    "claims_by": "openai:deepseek-v4-pro",        // 只有事后补拆的行才有这个字段
    "reviewed": false                             // 人工抽检过没有，见第五节
  }
}
```

`unanswerable` 样本的 `seed_chunk_id` 为空、`reference_answer` 与 `claims` 为空：语料里没有的事没有正确答案，
这类样本只判「链路的兜底行为对不对」，不进前三个指标的均值。

`claims` 与 query、参考答案在同一次生成里出齐，不在跑批时现拆 —— 它是 Context Recall 的分母，
现拆的话同一条答案这次 4 条下次 5 条，两次跑批的分数就没法比。`formal` 与 `colloquial` 是同一个
种子块的两种问法，参考答案逐字相同，共用一份 claim。当前 102 条里 96 条带 claim，共 282 条，
是用 `generate_claims.py` 事后补的（重跑 `generate_cases.py` 会连 query 一起重写），所以每行都记了
`meta.claims_by`。

---

## 三、分布

| 切面 | 分布 |
|---|---|
| 类型 | `single` 80 · `multi_hop` 16 · `unanswerable` 6 |
| 语域 | `formal` 48 · `colloquial` 54（`unanswerable` 全部口语） |
| 层 | `platform` 66 · `law` 26 · 跨层 4 · 无种子块 6 |
| 块类型 | `text` 62 · `table` 26 · 混合 8 · 无种子块 6 |
| 文档 | 16 篇每篇 ≥ 4 条；L02 / L05 / P01 / P03 / P06 / P09 / P10 / P11 各 6 条 |
| 种子块 | 40 个单块 + 8 组跨块对（16 个块），去重后 53 个，占库里 353 个块的 15% |

名额分配是「每篇保底 2 个种子块，块数 ≥ 20 的文档加 1 个」，不按块数按比例分。P02 只有 10 个块
却是判定核心，L02 有 41 个块，按比例分会让数据集复现命中分布 —— 而它的作用是暴露问题。
层内保证表格块与短块各有一个：表格是原子块，短块在 BM25 里天然吃亏，两类的检索行为都与普通正文块不同。

### 重叠率分档

`overlap_ratio` 是 query 与种子块正文的实词重叠率（中文二元组近似，见 `rag/evals/common.py`）。

| 语域 | p25 | 中位数 | p75 | 最大 |
|---|---|---|---|---|
| `formal` | 0.47 | 0.56 | 0.69 | 0.87 |
| `colloquial` | 0.07 | 0.10 | 0.19 | 0.31 |

两档差了一个数量级，这正是它们各自要压测的东西：`formal` 档给 BM25 一路送精确 term，
它的 Recall 读作**上界**；`colloquial` 档不含条文措辞，压的是稠密一路与改写环节，
它才对应线上表现。两档分数的差值反映链路对字面匹配的依赖程度。

---

## 四、生成与校验

```
353 个子块
  └─ 分层抽样（种子固定、按 chunk_id 排序后抽）→ 53 个种子块
        └─ LLM 生成（温度 0，只给 chunk.body 不给块头）→ { formal, colloquial, reference_answer, claims }
              └─ 重叠率超 0.45 的口语档打回重写一次
                    └─ 自动自检五项 → cases.jsonl
```

块头（`【文档】P02 …【路径】第二条 退货窗口`）不进生成提示词：模型会照抄进 query，
等于直接给 BM25 一个精确命中。跨块样本的两个种子块按关键词从两篇文档里各挑一个，
关键词只用于挑块，不进提示词。

自检项（`rag/evals/validate_cases.py`，不调模型）：

| 检查 | 抓什么 | 级别 |
|---|---|---|
| `seed_chunk_id` 存在性 | 切片版本漂了 | 错误 |
| `meta.overlap_ratio` 与实算一致 | 改了 query 忘了改分档字段 | 错误 |
| 参考答案里的数字可溯源 | 凭空出现的天数 / 次数 / 金额是幻觉 | 错误 |
| `claims` 里的数字在参考答案里出现过 | 改了参考答案而 claim 没重拆 | 错误 |
| 非 `unanswerable` 样本带 `claims` | 那条样本算不了 Context Recall | 警告 |
| `case_id` 唯一、种子数与 `type` 匹配 | 聚合口径算错 | 错误 |
| 分层覆盖（16 篇 / 两层 / 两种 kind / 每篇 ≥ 4 条） | 某一档没样本，测不出来 | 错误 |
| 口语档重叠率超 0.45 | query 抄了原文 | 警告，人工决定重写还是降级为 `formal` |

当前全集零错误、零警告。

---

## 五、人工抽检

自动自检只能确认「ID 有效、数字没编、分层齐全」，判不了「真人会不会这么问」。
按 10% 抽 10 条，只判三件事，判不了就打回：

1. 这个 query 真人会这么问吗；
2. 不看原文，这个 query 能唯一指向种子块吗 —— 不能说明问得太泛（「退款要多久」这种），要改写或降级为 `multi_hop`；
3. 参考答案有没有超出种子块 —— 超出的部分要么删，要么把那个块也加进 `seed_chunk_id`；
   顺带看 `claims` 拆得对不对：有没有把一句话切成半句、有没有补进答案里没有的话。

抽检过的样本的第二个用途：Context Recall 靠模型逐条判 claim，判得准不准需要一个基准，用的就是这批（5-rag-eval 七）。抽检做完之前，两个 LLM 指标的结论不采信。

两处已知现象，抽检时重点看：

- `formal` 档有 39 条重叠率超过 0.45。这是设计如此（那一档就是术语问法），但其中问得太泛的要挑出来；
- 平台层与法规层对同一件事的口径不一致时，参考答案跟着**种子块**走。例如「拆封查验后能否退货」，
  L03 第十九条说应当退货，P05 第二条说平台只受理未拆封 —— R1-012 与 R1-082 因此给出相反的答案，
  这不是生成错误，是语料本身的层级差异。

---

## 六、已知偏差

| 偏差 | 影响 | 怎么处理 |
|---|---|---|
| `seed_chunk_id` 不是全部相关块 | ID 级 Recall 系统性偏低（召回了等价条款也判负） | 按下界读，看版本间相对变化；长尾交给 Context Recall |
| 同一条规则散在多篇文档 | 几条样本问的是同一件事，种子块却各不相同 | 见下 |
| `formal` 档重叠率高 | 那一档 Recall 偏高 | 两档分开报，不合成一个数 |
| 参考答案由模型写 | 可能漏掉块里的条件 | 抽检第 3 条；数字幻觉已由自检拦住 |
| 重叠率用二元组近似 | 与 Milvus 服务端中文分析器的切词不完全一致 | 只用于分档与打回重写，不参与判分 |
| `unanswerable` 只有 6 条 | 兜底行为的样本量小 | 它只判「是否按预期抛异常」，不进均值 |
| 样本覆盖 53 / 353 个块 | 未被抽到的块从未被测过 | 分层保证了每篇文档、两种块类型都有样本，但不等于全覆盖 |

第二行最典型的一组：R1-012（L03 第十九条）、R1-034（P02 第三条）、R1-046 与 R1-082（P05 第二条）
问的都是「拆开看过没用过还能不能退」，种子块分别在三篇文档里。链路只能召回其中一两个，
另外几条按 ID 判就是漏召。自检里的重复问法检查抓不到它们（措辞差异足够大，实词距离 0.48~0.71），
也不该抓 —— 它们是合法样本，只是 Recall 的绝对值会因此偏低。
