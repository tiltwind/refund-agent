CREATE TABLE orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    product TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL,
    signed_days_ago INTEGER NOT NULL,
    refunded INTEGER NOT NULL DEFAULT 0 CHECK (refunded IN (0, 1))
);

CREATE TABLE refund_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision TEXT NOT NULL CHECK (decision IN ('批准', '拒绝')),
    receipt_no TEXT NOT NULL UNIQUE,
    order_id TEXT NOT NULL,
    amount REAL NOT NULL,
    reason TEXT NOT NULL,
    actor TEXT NOT NULL,
    idempotency_key TEXT UNIQUE
);

INSERT INTO orders
    (order_id, customer_id, product, category, price, signed_days_ago, refunded)
VALUES
    ('O2001', 'C1001', '无线降噪耳机', '数码', 899.00, 10, 0),
    ('O2002', 'C1002', '挪威三文鱼刺身 500g', '生鲜', 128.00, 2, 0),
    ('O2003', 'C1003', '机械键盘', '数码', 459.00, 5, 0),
    ('O2004', 'C1004', '缓震跑鞋', '服饰', 399.00, 3, 0);
