"""prod SQLite 数据库连接与初始化。"""

import os
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PROD_DIR = ROOT / "prod"
DEFAULT_DB_PATH = PROD_DIR / "sqlite.db"


def database_path() -> Path:
    configured = os.getenv("REFUND_AGENT_SQLITE_PATH", "").strip()
    return Path(configured) if configured else DEFAULT_DB_PATH


def connect() -> sqlite3.Connection:
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    return db


def initialize() -> None:
    """重建模拟线上库，并执行两份初始化 SQL。"""
    with connect() as db:
        db.executescript(
            """
            DROP TABLE IF EXISTS refund_decisions;
            DROP TABLE IF EXISTS orders;
            DROP TABLE IF EXISTS customers;
            """
        )
        for name in ("customers.sql", "orders.sql"):
            db.executescript((PROD_DIR / name).read_text(encoding="utf-8"))


def decision_log() -> list[dict]:
    with connect() as db:
        rows = db.execute(
            """
            SELECT decision, receipt_no, order_id, amount, reason, actor, idempotency_key
            FROM refund_decisions
            ORDER BY id
            """
        ).fetchall()
    return [dict(row) for row in rows]
