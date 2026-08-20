from contextlib import contextmanager

import langfuse

from app.context import RefundContext
from services import online_monitor
from services import telemetry
from services.telemetry import TurnTrace


def _turn(*, verdict="通过：符合退款条件", decision="批准", receipt="R9000", answer="R9000"):
    return {
        "tools": [
            {"name": "check_refund_eligibility", "args": {}},
            {"name": "execute_refund", "args": {}},
        ],
        "tool_results": {
            "check_refund_eligibility": [verdict],
            "search_refund_policy": ["[E1] 七天无理由退款"],
        },
        "answer": answer,
        "new_log": [{"decision": decision, "receipt_no": receipt}],
    }


def test_trace_output_contains_judge_fields():
    assert online_monitor.trace_output(_turn()) == {
        "answer": "R9000",
        "evidence": "[E1] 七天无理由退款",
        "rule_verdict": "通过：符合退款条件",
    }


def test_online_scores_pass_for_consistent_terminal_turn():
    scores = {name: value for name, value, _ in online_monitor.online_scores(_turn())}
    assert scores == {
        "rule_consistency": 1.0,
        "receipt_in_answer": 1.0,
        "log_structure": 1.0,
    }


def test_online_scores_reject_fabricated_receipt_and_duplicate_terminal_action():
    turn = _turn(answer="受理编号 D9999")
    turn["tools"].append({"name": "execute_refund", "args": {}})
    scores = {name: value for name, value, _ in online_monitor.online_scores(turn)}
    assert scores["receipt_in_answer"] == 0.0
    assert scores["log_structure"] == 0.0


def test_question_before_tools_is_a_consistent_ask_order_id():
    turn = {"tools": [], "tool_results": {}, "answer": "请提供订单号", "new_log": []}
    assert online_monitor.actual_outcome(turn) == "ask_order_id"
    assert online_monitor.online_scores(turn)[0][1] == 1.0


def test_turn_trace_writes_output_and_four_trace_scores():
    class Root:
        def __init__(self):
            self.output = None
            self.scores = []

        def update(self, *, output):
            self.output = output

        def set_trace_io(self, *, input, output):
            self.trace_io = {"input": input, "output": output}

        def score_trace(self, **score):
            self.scores.append(score)

    root = Root()
    trace = TurnTrace(config={}, root=root, input="我要退款")
    turn = _turn()
    trace.finish([], turn["new_log"])

    # 空消息验证 finish 自己按观察结果上报，而不是直接复用测试构造的 turn。
    assert root.output == {"answer": "", "evidence": "", "rule_verdict": ""}
    assert root.trace_io == {"input": "我要退款", "output": root.output}
    assert [score["name"] for score in root.scores] == [
        "rule_consistency",
        "receipt_in_answer",
        "log_structure",
        "outcome",
    ]


def test_trace_turn_sets_trace_attributes_once(monkeypatch):
    calls = {}
    handler = object()

    class Client:
        def create_trace_id(self, *, seed):
            calls["seed"] = seed
            return "a" * 32

        @contextmanager
        def start_as_current_observation(self, **kwargs):
            calls["observation"] = kwargs
            yield object()

    @contextmanager
    def propagate_attributes(**kwargs):
        calls["attributes"] = kwargs
        yield

    monkeypatch.setattr(telemetry, "_handler", lambda: handler)
    monkeypatch.setattr(langfuse, "get_client", lambda: Client())
    monkeypatch.setattr(langfuse, "propagate_attributes", propagate_attributes)
    monkeypatch.setenv("LANGFUSE_TRACING_ENVIRONMENT", "production")
    ctx = RefundContext(
        customer_id="C1001",
        actor="self",
        request_id="req-1",
        session_id="session-1",
        request_source="prod",
    )

    with telemetry.trace_turn(
        ctx, {"agent_version": "v1", "prompt_version": "v1.0.0"}, "我要退款"
    ) as trace:
        assert trace.config == {"callbacks": [handler], "run_name": "agent-graph"}

    assert calls["seed"] == "production:req-1"
    assert calls["observation"]["trace_context"] == {"trace_id": "a" * 32}
    assert calls["attributes"]["session_id"] == "session-1"
    assert calls["attributes"]["environment"] == "production"
    assert calls["attributes"]["metadata"] == {
        "agent_version": "v1",
        "prompt_version": "v1.0.0",
        "request_id": "req-1",
        "request_source": "prod",
        "actor": "self",
    }
