# 用 Langfuse 评估 Agent：核心代码

Agent 在本地执行，Langfuse 保存数据集、trace 和分数。流程：准备数据、执行用例、计算分数、聚合结果。使用 SDK v4。

## 一、准备数据集

每条用例是一个 dataset item：`input` 放上下文和用户消息，`expected_output` 放期望结果，`metadata` 放标题和优先级。用例 id 同时作为 item id。

```python
from langfuse import Langfuse

client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
client.create_dataset(name=dataset_name)
for case in cases:
    client.create_dataset_item(
        dataset_name=dataset_name,
        id=case["id"],
        input=case["input"],
        expected_output=case["expected"],
        metadata=case["metadata"],
    )
client.flush()
```

## 二、执行用例并采集 trace

LangGraph 通过 `CallbackHandler` 上报节点、模型和工具调用。`run_experiment` 为每个 item 调用一次 `task`。

```python
from langfuse.langchain import CallbackHandler

handler = CallbackHandler()

def task(*, item, **_):
    history, turns = [], []
    sandbox.begin()
    try:
        for turn in item.input["turns"]:
            history.append({"role": "user", "content": turn["user"]})
            message_start = len(history)
            log_start = len(sandbox.logs())
            result = agent.invoke(
                {"messages": history},
                context=item.input["context"],
                config={"callbacks": [handler], "run_name": f"eval:{item.id}"},
            )
            history = result["messages"]
            turns.append(collect_trace(
                result["messages"][message_start:],
                sandbox.logs()[log_start:],
            ))
        return {"turns": turns, "error": None}
    except Exception as exc:
        return {"turns": [], "error": str(exc)}
    finally:
        sandbox.end()
```

多轮对话持续累积 `history`。`collect_trace` 提取本轮工具、答复和新增业务流水：

```python
{"tools": [...], "tool_results": {...},
 "answer": "...", "business_logs": [...]}
```

评分函数只读取这份结构。

## 三、计算逐条分数

评估器接收实际输出和期望输出，返回 `Evaluation` 列表。执行异常单独记为 `run_error`。

```python
from langfuse import Evaluation

def evaluate(*, output, expected_output, **_):
    if output["error"]:
        return [
            Evaluation(name="run_error", value=1.0,
                       comment=output["error"]),
            Evaluation(name="case_pass", value=0.0),
        ]

    scores = [score for actual, expected in zip(
        output["turns"], expected_output["turns"]
    ) for score in score_turn(actual, expected)]
    hard = [score for score in scores if score.name in HARD_METRICS]
    scores.append(Evaluation(
        name="case_pass", value=float(all(s.value == 1.0 for s in hard))))
    scores.append(Evaluation(name="run_error", value=0.0))
    return scores
```

单项指标可使用确定性代码计算。例如工具流程要求必调工具全部出现、禁用工具均未调用，并按子序列检查顺序：

```python
names = [call["name"] for call in actual["tools"]]
spec = expected["tools"]
missing = [n for n in spec["must_call"] if n not in names]
banned = [n for n in spec["must_not_call"] if n in names]
order_ok = is_subsequence(spec["order"], names)

score = Evaluation(
    name="tool_sequence",
    value=float(not missing and not banned and order_ok),
    comment=f"missing={missing}, banned={banned}, actual={names}",
)
```

结论、业务流水、回执号和调用次数也可按同样方式计算。

## 四、聚合实验指标

run 级评估器从所有 item 的评分中计算通过率、错误率和单项均值：

```python
def aggregate(*, item_results, **_):
    return [
        Evaluation(name="pass_rate",
                   value=average(item_results, "case_pass")),
        Evaluation(name="error_rate",
                   value=average(item_results, "run_error")),
    ]
```

## 五、启动实验

从 Langfuse 读取数据集，把执行、逐条评分和聚合三个钩子交给 `run_experiment`：

```python
items = client.get_dataset(dataset_name).items

result = client.run_experiment(
    name=dataset_name,
    run_name=f"{agent_version}-{commit_id}",
    data=items,
    task=task,
    evaluators=[evaluate],
    run_evaluators=[aggregate],
    max_concurrency=4,
    metadata={"agent": agent_version, "prompt": prompt_version},
)
client.flush()
```

`result.item_results` 包含逐条分数，`result.run_evaluations` 包含聚合指标，`result.dataset_run_url` 是实验页面地址。

接入顺序是：校验用例、推送数据集、运行实验、查看 dataset run。正式回归使用固定数据集、run 名和评分代码。
