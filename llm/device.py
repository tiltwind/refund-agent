"""推理设备选择与推理锁 —— 嵌入与重排共用。

顺序固定为 cuda > mps > cpu。设备只影响速度，不影响结果，
所以这里不做任何可配置项：换设备不该成为「检索结果变了」的解释之一。
（`LLM_DEVICE` 是排障用的强制覆盖，不是常规配置项。）
"""

import os
import threading
from functools import lru_cache


@lru_cache(maxsize=1)
def pick_device() -> str:
    override = os.getenv("LLM_DEVICE")
    if override:
        return override

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def inference_lock() -> threading.Lock:
    """把本地模型的前向推理串起来的全局锁。嵌入与重排共用一把。

    PyTorch 的 MPS 后端不是线程安全的：多个线程同时跑 Metal kernel，
    `MetalShaderLibrary` 内部那张哈希表会被并发改写，进程直接段错误退出
    （实测跑批并发 5 时崩在 `exec_unary_kernel`，macOS 上还会弹系统崩溃窗）。
    评测脚本用线程池并发跑用例，正好踩中。

    加锁的代价接近零：嵌入和重排是同一块 GPU 上的计算，本来就在互相抢算力，
    并发跑不会更快。锁只圈住前向那一段，改写和 judge 那些网络调用照样并行。
    """
    return threading.Lock()
