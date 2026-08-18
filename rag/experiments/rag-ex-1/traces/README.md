# traces —— 现场记录

从 run `baseline-2` 导出的 20 条，供[基线报告](../baseline-report.md)逐条引用。
Langfuse 是本地实例，链接换台机器就打不开，所以现场留一份在仓库里。

重新导出：`python rag/experiments/rag-ex-1/export_traces.py`。

抽样不是随机的 —— 随机 20 条里大概率一条空证据都没有，而那恰恰是最该留档的。
按报告里的每个结论分桶取，最后三条是三档全中的对照：只留坏 case 的话，读的人无从
判断正常的一次检索长什么样。

## 空证据

重排后一条都不剩，Agent 拿到零证据（报告 5.1）

| 用例 | 读点 | 分类 |
|---|---|---|
| [R1-006](./R1-006.md) | 重排后 0 条过阈值，返回空证据 | single / colloquial / text |
| [R1-026](./R1-026.md) | 重排后 0 条过阈值，返回空证据 | single / colloquial / text |
| [R1-040](./R1-040.md) | 重排后 0 条过阈值，返回空证据 | single / colloquial / table |
| [R1-074](./R1-074.md) | 重排后 0 条过阈值，返回空证据 | single / colloquial / text |

## 召回层丢失

种子块根本没进 20 条候选，重排救不回来

| 用例 | 读点 | 分类 |
|---|---|---|
| [R1-012](./R1-012.md) | 种子块没进候选 | single / colloquial / text |
| [R1-024](./R1-024.md) | 种子块没进候选；证据里 46% 是重复正文 | single / colloquial / table |

## 候选尾部

种子块在候选里排 11~19，卡在 k=10 这条线上，也是 run 间抖动的来源

| 用例 | 读点 | 分类 |
|---|---|---|
| [R1-023](./R1-023.md) | 种子块在候选里排第 11 | single / formal / table |
| [R1-048](./R1-048.md) | 种子块在候选里排第 19；重排把它救进了前 3 | single / colloquial / text |

## 重排后掉出前 3

候选里有，重排后不在前 3 —— 压下去的、被阈值滤掉的、提了名次但不够的（报告四）

| 用例 | 读点 | 分类 |
|---|---|---|
| [R1-007](./R1-007.md) | 候选第 1 → 证据第 6，重排压下去了；证据里 49% 是重复正文 | single / formal / text |
| [R1-008](./R1-008.md) | 候选第 1，被 MIN_SCORE 滤掉 | single / colloquial / text |
| [R1-043](./R1-043.md) | 候选第 7 → 证据第 4，重排提了名次但仍不在前 3；证据里 42% 是重复正文 | single / formal / table |

## 重复正文

装配把同一父块拼了两遍以上，重复率超过 50%（报告 5.3）

| 用例 | 读点 | 分类 |
|---|---|---|
| [R1-009](./R1-009.md) | 三档全中；证据里 53% 是重复正文 | single / formal / text |
| [R1-010](./R1-010.md) | 命中，但排第 3；证据里 52% 是重复正文 | single / colloquial / text |

## unanswerable

语料里没有的问题，链路照样返回证据（报告 5.2）

| 用例 | 读点 | 分类 |
|---|---|---|
| [R1-097](./R1-097.md) | 语料里没有的问题，链路返回了证据 | unanswerable / colloquial /  |
| [R1-098](./R1-098.md) | 语料里没有的问题，链路返回了证据 | unanswerable / colloquial /  |

## multi_hop

两个种子块都要进前 3 才算命中

| 用例 | 读点 | 分类 |
|---|---|---|
| [R1-081](./R1-081.md) | 种子块在候选里排第 3/15；重排把它救进了前 3 | multi_hop / formal / table+text |
| [R1-083](./R1-083.md) | 命中，但排第 2/1；证据里 55% 是重复正文 | multi_hop / formal / table+text |

## 对照

三档全中，读「正常的一次检索长什么样」用

| 用例 | 读点 | 分类 |
|---|---|---|
| [R1-002](./R1-002.md) | 三档全中 | single / colloquial / text |
| [R1-003](./R1-003.md) | 三档全中 | single / formal / text |
| [R1-004](./R1-004.md) | 三档全中 | single / colloquial / text |

---

每条两个文件：`.md` 人读版，六步展开，装配那步是条款全文；`.json` 机器版，trace 元信息
加全部 observation，`input` / `output` 原样保留。

导出的是**这一次运行的痕迹**，不是可重放的用例。用例定义在
[`rag/datasets/r1/cases.jsonl`](../../../datasets/r1/cases.jsonl)。
