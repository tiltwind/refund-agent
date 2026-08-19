# traces —— 现场记录

从 run `r1-rerun-f5e4272` 导出的 102 条，供[HTML 报告](../rag-ex-1-report.html)逐条引用。
Langfuse 是本地实例，链接换台机器就打不开，所以现场留一份在仓库里。

重新导出：`python rag/experiments/rag-ex-1/export_traces.py --all`。

全量导出，每条用例一份。下面的分组是按报告里的结论分的，先到先得，一条只入一个组 ——
空证据、召回层丢失这些是报告逐条引用的那批，`其余` 是 `@3` 命中的常规样本。

## 空证据

重排后一条都不剩，Agent 拿到零证据（报告 5.1）

| 用例 | 读点 | 分类 |
|---|---|---|
| [R1-006](./R1-006.md) | 重排后 0 条过阈值，返回空证据 | single / colloquial / text |
| [R1-026](./R1-026.md) | 重排后 0 条过阈值，返回空证据 | single / colloquial / text |
| [R1-040](./R1-040.md) | 重排后 0 条过阈值，返回空证据 | single / colloquial / table |
| [R1-074](./R1-074.md) | 重排后 0 条过阈值，返回空证据 | single / colloquial / text |
| [R1-080](./R1-080.md) | 重排后 0 条过阈值，返回空证据 | single / colloquial / table |

## 候选尾部

种子块在候选里排 11~19，卡在 k=10 这条线上，也是 run 间抖动的来源

| 用例 | 读点 | 分类 |
|---|---|---|
| [R1-068](./R1-068.md) | 种子块在候选里排第 11；重排把它救进了前 3 | single / colloquial / text |
| [R1-081](./R1-081.md) | 种子块在候选里排第 3/13；重排把它救进了前 3 | multi_hop / formal / table+text |
| [R1-082](./R1-082.md) | 种子块在候选里排第 5/11 | multi_hop / colloquial / table+text |
| [R1-086](./R1-086.md) | 种子块在候选里排第 2/11 | multi_hop / colloquial / table+text |
| [R1-093](./R1-093.md) | 种子块在候选里排第 1/12；重排把它救进了前 3 | multi_hop / formal / text |
| [R1-094](./R1-094.md) | 种子块在候选里排第 2/11；重排把它救进了前 3 | multi_hop / colloquial / text |
| [R1-095](./R1-095.md) | 种子块在候选里排第 11/2；重排把它救进了前 3 | multi_hop / formal / text |

## 重排后掉出前 3

候选里有，重排后不在前 3 —— 压下去的、被阈值滤掉的、提了名次但不够的（报告四）

| 用例 | 读点 | 分类 |
|---|---|---|
| [R1-007](./R1-007.md) | 候选第 1 → 证据第 6，重排压下去了；证据里 49% 是重复正文 | single / formal / text |
| [R1-008](./R1-008.md) | 候选第 1，被 MIN_SCORE 滤掉 | single / colloquial / text |
| [R1-043](./R1-043.md) | 候选第 7 → 证据第 4，重排提了名次但仍不在前 3；证据里 42% 是重复正文 | single / formal / table |
| [R1-046](./R1-046.md) | 候选第 2 → 证据第 13，重排压下去了；证据里 43% 是重复正文 | single / colloquial / table |
| [R1-050](./R1-050.md) | 候选第 4 → 证据第 9，重排压下去了 | single / colloquial / table |
| [R1-070](./R1-070.md) | 候选第 3 → 证据第 8，重排压下去了；证据里 55% 是重复正文 | single / colloquial / table |
| [R1-084](./R1-084.md) | 候选第 1/2 → 证据第 5/1，重排压下去了 | multi_hop / colloquial / table+text |
| [R1-085](./R1-085.md) | 候选第 2/3 → 证据第 1/9，重排压下去了 | multi_hop / formal / table+text |
| [R1-092](./R1-092.md) | 候选第 8/5 → 证据第 4/5，重排提了名次但仍不在前 3 | multi_hop / colloquial / table+text |
| [R1-096](./R1-096.md) | 候选第 1/3，被 MIN_SCORE 滤掉 | multi_hop / colloquial / text |

## 重复正文

装配把同一父块拼了两遍以上，重复率超过 50%（报告 5.3）

| 用例 | 读点 | 分类 |
|---|---|---|
| [R1-033](./R1-033.md) | 三档全中；证据里 62% 是重复正文 | single / formal / table |
| [R1-053](./R1-053.md) | 三档全中；证据里 75% 是重复正文 | single / formal / text |
| [R1-054](./R1-054.md) | 命中，但排第 3；证据里 67% 是重复正文 | single / colloquial / text |
| [R1-060](./R1-060.md) | 三档全中；证据里 54% 是重复正文 | single / formal / table |
| [R1-069](./R1-069.md) | 三档全中；证据里 56% 是重复正文 | single / formal / table |

## unanswerable

语料里没有的问题，链路照样返回证据（报告 5.2）

| 用例 | 读点 | 分类 |
|---|---|---|
| [R1-097](./R1-097.md) | 语料里没有的问题，链路按预期抛异常 | unanswerable / colloquial /  |
| [R1-098](./R1-098.md) | 语料里没有的问题，链路按预期抛异常 | unanswerable / colloquial /  |
| [R1-099](./R1-099.md) | 语料里没有的问题，链路按预期抛异常 | unanswerable / colloquial /  |
| [R1-100](./R1-100.md) | 语料里没有的问题，链路按预期抛异常 | unanswerable / colloquial /  |
| [R1-101](./R1-101.md) | 语料里没有的问题，链路按预期抛异常 | unanswerable / colloquial /  |
| [R1-102](./R1-102.md) | 语料里没有的问题，链路按预期抛异常 | unanswerable / colloquial /  |

## multi_hop

两个种子块都要进前 3 才算命中

| 用例 | 读点 | 分类 |
|---|---|---|
| [R1-083](./R1-083.md) | 命中，但排第 2/1 | multi_hop / formal / table+text |
| [R1-089](./R1-089.md) | 命中，但排第 1/2 | multi_hop / formal / text |
| [R1-090](./R1-090.md) | 命中，但排第 1/2 | multi_hop / colloquial / text |
| [R1-091](./R1-091.md) | 命中，但排第 3/2 | multi_hop / formal / table+text |

## 对照

三档全中，读「正常的一次检索长什么样」用

| 用例 | 读点 | 分类 |
|---|---|---|
| [R1-002](./R1-002.md) | 三档全中；证据里 41% 是重复正文 | single / colloquial / text |
| [R1-003](./R1-003.md) | 三档全中 | single / formal / text |
| [R1-004](./R1-004.md) | 三档全中 | single / colloquial / text |
| [R1-005](./R1-005.md) | 三档全中 | single / formal / text |
| [R1-009](./R1-009.md) | 三档全中 | single / formal / text |
| [R1-011](./R1-011.md) | 三档全中 | single / formal / text |
| [R1-012](./R1-012.md) | 三档全中 | single / colloquial / text |
| [R1-013](./R1-013.md) | 三档全中 | single / formal / text |
| [R1-014](./R1-014.md) | 三档全中 | single / colloquial / text |
| [R1-015](./R1-015.md) | 三档全中 | single / formal / text |
| [R1-016](./R1-016.md) | 三档全中 | single / colloquial / text |
| [R1-017](./R1-017.md) | 三档全中 | single / formal / text |
| [R1-018](./R1-018.md) | 三档全中 | single / colloquial / text |
| [R1-019](./R1-019.md) | 三档全中 | single / formal / text |
| [R1-020](./R1-020.md) | 三档全中 | single / colloquial / text |
| [R1-022](./R1-022.md) | 三档全中 | single / colloquial / text |
| [R1-025](./R1-025.md) | 三档全中 | single / formal / text |
| [R1-027](./R1-027.md) | 三档全中 | single / formal / text |
| [R1-028](./R1-028.md) | 三档全中 | single / colloquial / text |
| [R1-029](./R1-029.md) | 三档全中 | single / formal / text |
| [R1-030](./R1-030.md) | 三档全中 | single / colloquial / text |
| [R1-031](./R1-031.md) | 三档全中 | single / formal / table |
| [R1-032](./R1-032.md) | 三档全中 | single / colloquial / table |
| [R1-035](./R1-035.md) | 三档全中 | single / formal / text |
| [R1-036](./R1-036.md) | 三档全中 | single / colloquial / text |
| [R1-037](./R1-037.md) | 三档全中 | single / formal / text |
| [R1-038](./R1-038.md) | 三档全中 | single / colloquial / text |
| [R1-044](./R1-044.md) | 三档全中 | single / colloquial / table |
| [R1-047](./R1-047.md) | 三档全中；证据里 47% 是重复正文 | single / formal / text |
| [R1-048](./R1-048.md) | 三档全中 | single / colloquial / text |
| [R1-049](./R1-049.md) | 三档全中；证据里 43% 是重复正文 | single / formal / table |
| [R1-051](./R1-051.md) | 三档全中 | single / formal / text |
| [R1-052](./R1-052.md) | 三档全中 | single / colloquial / text |
| [R1-055](./R1-055.md) | 三档全中 | single / formal / table |
| [R1-057](./R1-057.md) | 三档全中 | single / formal / text |
| [R1-058](./R1-058.md) | 三档全中 | single / colloquial / text |
| [R1-059](./R1-059.md) | 三档全中 | single / formal / table |
| [R1-061](./R1-061.md) | 三档全中 | single / formal / text |
| [R1-062](./R1-062.md) | 三档全中 | single / colloquial / text |
| [R1-065](./R1-065.md) | 三档全中 | single / formal / text |
| [R1-066](./R1-066.md) | 三档全中 | single / colloquial / text |
| [R1-071](./R1-071.md) | 三档全中；证据里 48% 是重复正文 | single / formal / text |
| [R1-072](./R1-072.md) | 三档全中 | single / colloquial / text |
| [R1-073](./R1-073.md) | 三档全中 | single / formal / text |
| [R1-075](./R1-075.md) | 三档全中 | single / formal / text |
| [R1-076](./R1-076.md) | 三档全中 | single / colloquial / text |
| [R1-079](./R1-079.md) | 三档全中；证据里 42% 是重复正文 | single / formal / table |
| [R1-087](./R1-087.md) | 三档全中 | single / formal / table |
| [R1-088](./R1-088.md) | 三档全中 | single / colloquial / table |

## 其余

@3 命中、@1 未中的常规样本，不属于上面任何一类

| 用例 | 读点 | 分类 |
|---|---|---|
| [R1-001](./R1-001.md) | 命中，但排第 2 | single / formal / text |
| [R1-010](./R1-010.md) | 命中，但排第 3 | single / colloquial / text |
| [R1-021](./R1-021.md) | 命中，但排第 3 | single / formal / text |
| [R1-023](./R1-023.md) | 命中，但排第 7 | single / formal / table |
| [R1-024](./R1-024.md) | 命中，但排第 × | single / colloquial / table |
| [R1-034](./R1-034.md) | 命中，但排第 3 | single / colloquial / table |
| [R1-039](./R1-039.md) | 命中，但排第 2 | single / formal / table |
| [R1-041](./R1-041.md) | 命中，但排第 2 | single / formal / text |
| [R1-042](./R1-042.md) | 命中，但排第 2 | single / colloquial / text |
| [R1-045](./R1-045.md) | 命中，但排第 2 | single / formal / table |
| [R1-056](./R1-056.md) | 命中，但排第 2 | single / colloquial / table |
| [R1-063](./R1-063.md) | 命中，但排第 2 | single / formal / table |
| [R1-064](./R1-064.md) | 命中，但排第 3 | single / colloquial / table |
| [R1-067](./R1-067.md) | 命中，但排第 2 | single / formal / text |
| [R1-077](./R1-077.md) | 命中，但排第 2 | single / formal / text |
| [R1-078](./R1-078.md) | 命中，但排第 3 | single / colloquial / text |

---

每条两个文件：`.md` 人读版，六步展开，装配那步是条款全文；`.json` 机器版，trace 元信息
加全部 observation，`input` / `output` 原样保留。

导出的是**这一次运行的痕迹**，不是可重放的用例。用例定义在
[`rag/datasets/r1/cases.jsonl`](../../../datasets/r1/cases.jsonl)。
