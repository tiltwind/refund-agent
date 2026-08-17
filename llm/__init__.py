"""模型层 —— 本地嵌入与重排（embedding/ rerank/），远程对话模型接入（chat.py）。

这一层与业务无关，只负责「把文本变成向量 / 给 (query, passage) 打分 / 拿到一个
可调用的对话模型」。放在 services/ 之外，是因为每一项都同时被两条互不相干的链路
用到：嵌入与重排被离线灌库（rag/index/）和在线检索（rag/retrieving/）共用 —— 两边
必须是**同一份**模型与同一套编码参数，向量空间不同则检索结果毫无意义；chat.py 被
Agent 主循环（agent/）和查询改写（rag/retrieving/pipeline/）共用 —— 供应商解析散在
两处，迟早漂移成「主模型换了家、改写还在打上一家的端点」。
"""
