# 5 · 检索评测：跑批与结果

用 [4 · 检索评测数据集](https://tiltwind.github.io/refund-agent/doc/get-start/4-rag-dataset.md)的 `r1` 与那里定下的指标，评[六步检索链路](https://tiltwind.github.io/refund-agent/doc/get-start/3-rag-impl.md#七第-6-步--检索链路-ragretrieving)。被测对象是 `search_policy`，不含 Agent：一个问题进去、一组 `PolicySection` 出来。实验目录 `rag/experiments/rag-ex-1/`。

指标分两组：两个 LLM judge 指标判**交付的上下文**，三个排序指标判**出题的那段条文排第几**。
后者不调模型，只比对 `source` 与检索链路的 ID 序列。

---

## 一、跑批

```bash
# 前置：Milvus 起着并已灌库，数据集自检过
bash scripts/milvus.sh start && python rag/index/seed_milvus.py
python rag/evals/validate_cases.py

# dataset run：样本从 Langfuse 数据集拉，分数写回，六步上报 trace
python rag/evals/push_dataset.py
python rag/experiments/rag-ex-1/run_experiment.py --langfuse --run-name $(git rev-parse --short HEAD)
```

参数:
- `--langfuse` 指定上报 Langfuse trace 。
- `--concurrency` 调高并行的只有网络调用（改写、judge）。本地模型的前向与 tokenizer 都被一把
  全局锁串起来 —— PyTorch 的 MPS 后端不是线程安全的，并发跑 Metal kernel 会让进程段错误退出；
  HF 的 fast tokenizer 是 Rust 对象，同一个实例被两个线程同时调用抛 `RuntimeError: Already borrowed`。


## 二、跑批结果

96 条全量，112.4s，judge 调用 0 次失败，跑批故障 0 条。judge 是 `deepseek-v4-flash`，
与链路的改写模型同一个 —— 自己评自己，跑批开头打一行警告，换 judge 之后的分数与这里不可比。

| 指标 | 值 | 分母 |
|---|---|---|
| Context Precision | 0.900 | 96 |
| Context Recall | 0.927 | 96 |

分母里有 5 条空上下文：检索一条证据都没交付，两个指标都判 0。非空的 91 条是 0.950 / 0.978。

三个排序指标不调模型，只比对 `source` 与检索链路的 ID 序列：

| 指标 | 值 | 说的是什么 |
|---|---|---|
| candidate_hit | 0.969 | source 进了 20 条候选 |
| hit@1 | 0.552 | 重排后排第一 |
| hit@4 | 0.812 | 重排后进前四 |
| mrr | 0.675 | 1 / 排名，掉出证据列表记 0 |

candidate_hit 与 hit@4 之间的落差是重排和阈值这一段丢的，与召回层漏掉的那 3% 分开看 ——
两种情况的修法不同。


## 三、校准 judge

跑批的分数里混着两个变量：检索每次交付的上下文会变，judge 的判定也会变。
跑批保存一份上下文镜像 `context_snapshot.json`，存一轮 judge的全部输入（问题、有序上下文全文、标准答案），可以通过它来校准 judge 本身的稳定性。

```bash
python rag/experiments/rag-ex-1/calibrate_judge.py --rounds 5
```

91 条非空上下文各判 5 轮（空上下文那 5 条 judge 短路判 0、不调模型，不参与）：

| 指标 | 5 轮均值 | 轮间极差 | 判定翻转 |
|---|---|---|---|
| Context Precision | 0.948 | 0.007 | 15/301 段（5.0%） |
| Context Recall | 0.978 | 0.000 | 0/147 句 |

Context Recall 的逐句判定五轮一次没翻 —— 它问的是「上下文里有没有这个信息」，答案在文本里。
抖动全在 Context Precision 的「这一段有没有用」上：翻转的 15 段各轮理由互相矛盾，同一段
四轮判无关、一轮判相关。位置加权把大部分翻转吃掉了，5% 的段级翻转只换来 0.007 的均值波动。

判得稳不等于判得对 —— 后者要人工标注，抽样与一致率的算法见
[实验目录 README](https://github.com/tiltwind/refund-agent/blob/main/rag/experiments/rag-ex-1/README.md#judge-校准)。


## 四、门禁

排序指标可以做门禁：同一套参数连跑四次，Hit@4 是 0.802 / 0.802 / 0.792 / 0.812，MRR 是
0.670 / 0.670 / 0.671 / 0.675，空上下文四次都是同样那 5 条。这四次不受 judge 影响 ——
排序指标不调模型。残留的抖动来自改写那一步调模型（温度 0 也不保证每次返回同一份拆分），
量级在 0.020。具体阈值等这一轮 `MIN_SCORE` 调完再定 —— 当前基线本身就是要被改掉的那个。

两个 judge 指标暂不做门禁，两个原因：

1. **已经饱和**。非空的 91 条里 88 条 Context Recall 满分；Context Precision 的位置加权
   让尾部的无关条款几乎免费 —— 交付的 301 段里 77 段被判不相关，指标只掉到 0.950。
   改一次检索参数，这两个数字基本不动。
2. **判得对不对还没量**。噪声那一半有了：5 轮校准的轮间极差是 0.007（CP）和 0.000（CR），
   门禁阈值的下限就在这里。剩下的一致率要人工抽检那批来算，判得稳不等于判得对。

在那之前它们只作观察。
