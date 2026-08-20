"""从一轮 Agent 运行中提取线上评估字段并计算确定性分数。"""

import re


RECEIPT_RE = re.compile(r"\b[RD]\d{4,}\b")
VERDICT_TO_OUTCOME = {"通过": "approved", "不通过": "denied", "需补充": "clarify"}
TERMINAL_TOOLS = {"execute_refund", "record_refund_denial"}
HANDOFF_TOOLS = {"handoff_to_human", "transfer_to_human"}


def observe(new_messages: list, new_log: list) -> dict:
    """把一轮新增消息压成评估与上报共用的结构。"""
    calls: list[dict] = []
    results: dict[str, list[str]] = {}
    answer = ""
    for message in new_messages:
        for call in getattr(message, "tool_calls", None) or []:
            calls.append({"name": call["name"], "args": call["args"]})
        kind = getattr(message, "type", "")
        if kind == "tool":
            results.setdefault(getattr(message, "name", "?"), []).append(str(message.content))
        elif kind == "ai":
            answer = getattr(message, "text", "") or answer
    return {"tools": calls, "tool_results": results, "answer": answer, "new_log": list(new_log)}


def last_verdict(turn: dict) -> str | None:
    """取规则引擎最后一次有效判定的结论。"""
    for text in reversed(turn["tool_results"].get("check_refund_eligibility", [])):
        head = text.split("：", 1)[0]
        if head in VERDICT_TO_OUTCOME:
            return head
    return None


def rule_verdict(turn: dict) -> str:
    """取规则引擎最后一次有效判定的完整文本，作为判官的真值锚。"""
    for text in reversed(turn["tool_results"].get("check_refund_eligibility", [])):
        if text.split("：", 1)[0] in VERDICT_TO_OUTCOME:
            return text
    return ""


def actual_outcome(turn: dict) -> str:
    """从终局动作、流水和工具轨迹确定本轮结果分类。"""
    names = [call["name"] for call in turn["tools"]]
    if any(name in HANDOFF_TOOLS for name in names):
        return "handoff"
    rows = turn["new_log"]
    if rows:
        return "approved" if rows[-1]["decision"] == "批准" else "denied"
    return "ask_order_id" if not names else "clarify"


def trace_output(turn: dict) -> dict:
    """生成在线评估器读取的紧凑 trace output。"""
    return {
        "answer": turn["answer"],
        "evidence": "\n\n".join(turn["tool_results"].get("search_refund_policy", [])),
        "rule_verdict": rule_verdict(turn),
    }


def online_scores(turn: dict) -> list[tuple[str, float, str]]:
    """计算三个不依赖人工标注的线上自洽分数。"""
    names = [call["name"] for call in turn["tools"]]
    rows = turn["new_log"]
    outcome = actual_outcome(turn)
    verdict = last_verdict(turn)

    if verdict is None:
        consistent = outcome == "ask_order_id"
        consistency_note = "尚未进入判定（合规）" if consistent else "没问规则引擎就下了结论"
    else:
        consistent = outcome == VERDICT_TO_OUTCOME[verdict]
        consistency_note = f"规则引擎判「{verdict}」，实际 {outcome}"

    answer = re.sub(r"\s+", "", turn["answer"] or "")
    if rows:
        receipt = rows[-1]["receipt_no"]
        receipt_ok = receipt in answer
        receipt_note = f"答复{'已' if receipt_ok else '未'}引用落库单号 {receipt}"
    else:
        stray = RECEIPT_RE.findall(answer)
        receipt_ok = not stray
        receipt_note = f"没落库却报了编号 {stray}" if stray else "无落库无单号（合规）"

    terminal = [name for name in names if name in TERMINAL_TOOLS]
    structure_ok = (len(terminal), len(rows)) in {(0, 0), (1, 1)}
    structure_note = f"终局工具 {terminal}，新增流水 {len(rows)} 行"

    return [
        ("rule_consistency", float(consistent), consistency_note),
        ("receipt_in_answer", float(receipt_ok), receipt_note),
        ("log_structure", float(structure_ok), structure_note),
    ]
