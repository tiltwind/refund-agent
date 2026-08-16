#!/usr/bin/env bash
# main.py 的启动脚本：加载 .env 里的环境变量后跑演示链路。
#
# 用法：
#   bash run-main.sh                    # 跑三个演示场景
#   bash run-main.sh --trace            # 额外打印检索链路每一步的中间产物
#   ENV_FILE=.env.staging bash run-main.sh
#
# 密钥只放在 .env（已 gitignore），脚本本身不硬编码任何凭据。
# 首次使用：cp .env.example .env && 填入 ANTHROPIC_API_KEY 或 OPENAI_API_KEY
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

ENV_FILE="${ENV_FILE:-.env}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ ! -f "${ENV_FILE}" ]; then
  echo "缺少 ${ENV_FILE}，先执行：cp .env.example ${ENV_FILE} 并填入密钥" >&2
  exit 1
fi

# set -a 让后续赋值自动 export；用 source 而非逐行解析，以便支持引号和 # 注释。
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

# 两家供应商任配一个即可（解析规则见 llm/chat.py）。两个都配时默认走 Anthropic，
# 要切到 OpenAI 就设 REFUND_AGENT_PROVIDER=openai —— 别靠删 key 来切换。
if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "${ENV_FILE} 里 ANTHROPIC_API_KEY 与 OPENAI_API_KEY 都为空，Agent 无法调用模型" >&2
  echo "任配一个：ANTHROPIC_API_KEY，或 OPENAI_API_KEY + OPENAI_BASE_URL + OPENAI_MODEL" >&2
  exit 1
fi

if [ "${REFUND_AGENT_PROVIDER:-}" = "openai" ] || { [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -n "${OPENAI_API_KEY:-}" ]; }; then
  # 兼容网关的模型名各家各不相同，猜不出默认值，缺了就直接说清楚
  if [ -z "${OPENAI_MODEL:-}" ] && [ -z "${REFUND_AGENT_MODEL:-}" ]; then
    echo "走 OpenAI 供应商但没配 OPENAI_MODEL（或 REFUND_AGENT_MODEL）" >&2
    exit 1
  fi
fi

# --trace 是 REFUND_AGENT_RAG_TRACE=on 的快捷写法（见 main.py 头部说明）
ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --trace) export REFUND_AGENT_RAG_TRACE=on ;;
    *) ARGS+=("${arg}") ;;
  esac
done

# 决定 trace 报不报的是两个 key，不是 BASE_URL —— 早先这里只看 BASE_URL，
# 于是明明没接埋点也显示一个地址，看着像开着。详细状态由 main.py 启动时打印。
if [ -n "${LANGFUSE_PUBLIC_KEY:-}" ] && [ -n "${LANGFUSE_SECRET_KEY:-}" ]; then
  LANGFUSE_STATUS="${LANGFUSE_BASE_URL:-cloud}"
else
  LANGFUSE_STATUS="off（未配 key）"
fi

# 模型端点不在这里回显：供应商解析有优先级规则，脚本里复述一遍必然和实际用的
# 那个不一致。main.py 启动时打印的是真正解析出来的结果。
echo "→ env=${ENV_FILE}  langfuse=${LANGFUSE_STATUS}"
exec "${PYTHON_BIN}" main.py ${ARGS[@]+"${ARGS[@]}"}
