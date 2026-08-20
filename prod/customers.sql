CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    level TEXT NOT NULL,
    register_date TEXT NOT NULL,
    refund_count_90d INTEGER NOT NULL
);

INSERT INTO customers
    (customer_id, name, level, register_date, refund_count_90d)
VALUES
    ('C1001', '林晓', '金牌会员', '2023-05-12', 1),
    ('C1002', '陈立', '普通会员', '2025-11-03', 0),
    ('C1003', '王强', '普通会员', '2024-01-20', 4),
    ('C1004', '赵敏', '普通会员', '2024-08-09', 0);
