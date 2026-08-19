# 4 · 检索评测数据集

[3 · 政策知识库](https://tiltwind.github.io/refund-agent/doc/get-start/3-rag-impl.md)里的 RRF `k=20`、重排权重 `0.80 / 0.20`、`MIN_SCORE=0.30`、法规层 `0.5` 降权都是初始值，校准它们要有一个数据集。本篇建这个集 `r1`。

顺序是：先定要评什么指标，再由指标决定标注哪些字段。反过来做会标出一堆没人读的字段。

---

## 一、评估指标

检索交付给模型的是一组条款。它只可能在两个方向上出错：

| 指标 | 问什么 | 要标什么 |
|---|---|---|
| Context Precision | 检回的条款里有多少是真有用的，有用的排得够不够靠前 | 什么都不标 |
| Context Recall | 回答这个问题需要的信息，检回的上下文里有没有 | 一个标准答案 |

```
Context Precision = Σ(Precision@k × rel_k) / 相关条数
Context Recall    = 被上下文支撑的句子数 / 标准答案的句子数
```

一个管精度，一个管覆盖，方向相反。两个都由 LLM judge 判定。

> 不评 Recall@k. ID 级的 Recall@k 要的真值是「这条问题的全部相关块」，人工在 353 个块里穷举做不到：P02 和 P07 都讲退货窗口，L02 和 P02 都讲七天无理由，漏标无法避免。

这两个指标都要调模型，判定本身带噪声，而且判的是交付出去的那几段 —— 「本该给的那段排在第几」它们答不了。所以另有一组不调模型的排序指标，见 1.5。

### 1.1 判定对象

`search_policy` 返回的是装配后的 `PolicySection` 列表，`DEFAULT_TOP_K = 4`，按重排分降序：

```
… → 4 召回融合 → 5 重排 → 6 装配 → PolicySection × 4   ← 两个指标都判这一组
```

判这一层而不是重排那一层，因为装配产物才是真正注入模型的上下文：装配把命中的子块还原成完整小节、还可能合并相邻父块，重排选对了块、装配仍可能塞进一堆无关正文。

两个指标都由 LLM judge 逐条判定，实现在 [`judge.py`](https://github.com/tiltwind/refund-agent/blob/main/rag/experiments/rag-ex-1/judge.py)。

### 1.2 Context Precision 计算

**第一步，逐段判相关性**。把问题和 4 段条款一起给 judge，每段返回 0(不相关) 或 1(相关)。

**第二步，按位置加权**。`rel_k` 是第 k 段的判定（0 或 1），`Precision@k` 是前 k 段里相关的比例，只在相关的位置上累加。

举例，4 段判成 `[相关, 不相关, 相关, 不相关]`：

| k | 判定 | Precision@k | 计入 |
|---|---|---|---|
| 1 | 相关 | 1/1 = 1.000 | ✓ |
| 2 | 不相关 | — | |
| 3 | 相关 | 2/3 = 0.667 | ✓ |
| 4 | 不相关 | — | |

`(1.000 + 0.667) / 2 = 0.833`。同样是两段相关，排在第 1、2 位得 1.000，排在第 3、4 位只得 0.583。

位置相关：检索交付给模型的是一个有序列表，把不相关的顶在前面本身就是缺陷。

### 1.3 Context Recall 计算

- **第一步，拆句**。参考答案 拆分为独立的信息单元（通常按句子）。
- **第二步，逐句归因**。对于每个信息单元，判断是否能从检索结果中找到支撑。

### 1.4 两个指标的判断

| Precision | Recall | 说明 | 改哪里 |
|---|---|---|---|
| 低 | 低 | 检索没找对东西 | 召回层：切片、块头、BM25 分析器、`CANDIDATE_LIMIT` |
| 低 | 高 | 答案撑得住，但混进了无关条款 | 重排权重、`MIN_SCORE`、`top_k` 调小 |
| 高 | 低 | 检回的都有用，就是不够 | `top_k` 调大、[改写](https://github.com/tiltwind/refund-agent/blob/main/rag/retrieving/pipeline/rewrite.py)拆子查询 |
| 高 | 高 | 检索没问题 | 答复仍然错，问题在生成阶段 |


### 1.5 三个排序指标

上面两个指标各自有一个绕不开的短板：都要调模型，判定本身带噪声；而且它们判的是**交付出去
的那几段**，判不了「本该给的那段排在第几」。所以另有一组不调模型的指标，判 `source`
在检索链路里的位次：

| 指标 | 问什么 |
|---|---|
| `candidate_hit` | 出题的条文有没有进 20 条候选 —— 分开「召回层漏了」和「重排/阈值挡了」 |
| `hit@1` / `hit@4` | 重排后它排在第几 |
| `mrr` | 1 / 排名，掉出证据列表记 0 |

它与 Recall@k 的区别在真值：Recall@k 要「这条问题的全部相关块」，这一组只要「这条问题至少
出自这一段」，而后者在反向生成时就是已知的。代价是它只能作**下界** —— 命中不代表检索完备，
没命中也可能是检回了另一段等价条文（平台层与法规层对同一件事常常各有条文，`source` 只记了
出题用的那一段）。所以两组指标并排读：排序那组指方向，判定那组兜底。

跨块样本按两段里排得最深的那个算：标准答案要两段都用上，只召回一半答不全。

实现在 [`rank_metrics.py`](https://github.com/tiltwind/refund-agent/blob/main/rag/experiments/rag-ex-1/rank_metrics.py)。
判分只比对 chunk_id，同一份检索结果重复判多少次都是同一个数。


## 二、数据集格式

```jsonc
{
  "case_id": "R1-081",
  "question": "商品已拆封但未实际使用，是否支持无理由退货？若支持，寄回运费由谁承担？",
  "ground_truth": "已拆封但未实际使用的商品不支持无理由退货，平台仅接受未拆封商品的无理由退货。若符合无理由退货条件，寄回运费由消费者承担，可自行选择快递或使用平台上门取件服务（费用从退款中扣除）。",
  "source": ["P05#002:00", "P06#001:00"]   // 出题的条文，排序指标的真值
}
```

| 字段 | 谁读它 |
|---|---|
| `question` | 直接喂 `search_policy`，不经过 Agent |
| `ground_truth` | Context Recall 的分母：按句拆开，逐句判有没有被上下文支撑 |
| `source` | 三个排序指标的真值；报告按文档分档也读它；掉分时回头看这条出自哪一段条文 |


## 三、反向生成

反向从条文出发生成数据集：

```
353 个子块
  └─ 分层抽样 → 单块种子 + 跨块对
        └─ LLM 生成（温度 0）→ { formal, colloquial, ground_truth }
              └─ 自检五项 → rag/datasets/r1/cases.jsonl
```

当前这一版：96 条样本，82 条单块 + 14 条跨块，去重后用到 52 个种子块（占 353 个块的 15%）。

问题分2类型，一个是带政策术语的规范问法，一个是消费者的口语。
同一条规则被问到两次，一次给 BM25 送精确 term，一次压稠密一路和[改写](https://github.com/tiltwind/refund-agent/blob/main/rag/retrieving/pipeline/rewrite.py)环节。

抽样每篇文档保底 2 个、块数 ≥ 20 的加 1，层内保证表格块与短块各有一个。

## 四、自检

数据集自检[`validate_cases.py`](https://github.com/tiltwind/refund-agent/blob/main/rag/evals/validate_cases.py) ：

| 检查 | 抓什么 |
|---|---|
| 四个字段齐全、`case_id` 唯一 | 跑批跑到一半才发现要重来 |
| `question` 不重复 | 同一个问题被重复计权 |
| `source` 里的 chunk_id 在库里存在 | 切片版本漂了 |
| `ground_truth` 里的数字在源块正文里出现过 | 凭空出现的天数、次数、金额是模型幻觉 |
| `ground_truth` 不以独立结论句开头 | 见 1.3 |

自检之外抽 10% 人工过一遍，判两件事：这个问题真人会不会这么问；标准答案有没有超出源块（超出的部分要么删，要么把那个块也加进 `source`）。

## 五、产物

| 产物 | 位置 | 作用 |
|---|---|---|
| 生成脚本 | [`rag/evals/generate_cases.py`](https://github.com/tiltwind/refund-agent/blob/main/rag/evals/generate_cases.py) | 分层抽样 + 反向生成 |
| 用例集 | [`rag/datasets/r1/cases.jsonl`](https://github.com/tiltwind/refund-agent/blob/main/rag/datasets/r1/cases.jsonl) | 96 条，检索评测的单一事实源 |
| 数据集说明 | [`rag/datasets/r1/README.md`](https://github.com/tiltwind/refund-agent/blob/main/rag/datasets/r1/README.md) | 绑定关系、分布、已知偏差 |
| 自检脚本 | [`rag/evals/validate_cases.py`](https://github.com/tiltwind/refund-agent/blob/main/rag/evals/validate_cases.py) | 第四节五项 |
| 排序指标 | [`rag/experiments/rag-ex-1/rank_metrics.py`](https://github.com/tiltwind/refund-agent/blob/main/rag/experiments/rag-ex-1/rank_metrics.py) | 1.0 那三个，判分只读 `source` |
| 推送脚本 | [`rag/evals/push_dataset.py`](https://github.com/tiltwind/refund-agent/blob/main/rag/evals/push_dataset.py) | 推成 Langfuse 数据集 |

```bash
python rag/evals/generate_cases.py --dry-run   # 只看抽样结果，不调模型
python rag/evals/generate_cases.py             # 生成 cases.jsonl
python rag/evals/validate_cases.py             # 五项自检
```
