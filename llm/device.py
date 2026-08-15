"""推理设备选择 —— 嵌入与重排共用。

顺序固定为 cuda > mps > cpu。设备只影响速度，不影响结果，
所以这里不做任何可配置项：换设备不该成为「检索结果变了」的解释之一。
"""

import os
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
