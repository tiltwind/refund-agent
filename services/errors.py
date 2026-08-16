"""服务接入层的异常。"""


class EvalDataMissError(RuntimeError):
    """eval 数据源里查不到入参对应的数据（2-design 3.3）。

    Agent 是非确定性的：改完提示词，它可能用一个 eval 数据里根本没有的入参去查。
    这时**绝不能静默返回空值当作「查无此人」**，否则用例会带着一个看似合理的
    错误结论通过。抛出本异常，把用例标记为 invalid 而非 failed —— 它不是回归，
    是 eval 数据覆盖不足。
    """
