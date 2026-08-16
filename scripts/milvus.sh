#!/usr/bin/env bash
# Milvus standalone(内嵌 etcd) 本地服务管理脚本。
# 基于官方 standalone_embed.sh 改写，差异：健康/metrics 端口对外映射为 19091（容器内仍是 9091），
# 避免与本机其它服务抢占 9091。
#
# 用法：
#   bash scripts/milvus.sh start     启动（数据目录不存在时自动创建配置）
#   bash scripts/milvus.sh stop      停止容器（保留数据）
#   bash scripts/milvus.sh restart   重启
#   bash scripts/milvus.sh status    查看状态与端口映射
#   bash scripts/milvus.sh logs      跟踪日志
#   bash scripts/milvus.sh delete    删除容器与全部数据（不可恢复）
#
# 可通过环境变量覆盖：
#   MILVUS_HOME        数据与配置目录，默认 ~/.refund-agent-milvus
#   MILVUS_PORT        gRPC 端口，默认 19530
#   MILVUS_HEALTH_PORT 健康/metrics 端口，默认 19091
#   MILVUS_ETCD_PORT   内嵌 etcd 端口，默认 2379
#   MILVUS_IMAGE       镜像，默认 milvusdb/milvus:v2.5.4
set -euo pipefail

CONTAINER="milvus-standalone"
MILVUS_HOME="${MILVUS_HOME:-${HOME}/.refund-agent-milvus}"
MILVUS_PORT="${MILVUS_PORT:-19530}"
MILVUS_HEALTH_PORT="${MILVUS_HEALTH_PORT:-19091}"
MILVUS_ETCD_PORT="${MILVUS_ETCD_PORT:-2379}"
MILVUS_IMAGE="${MILVUS_IMAGE:-milvusdb/milvus:v2.5.4}"

container_exists() {
  docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER}"
}

container_running() {
  docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"
}

init_config() {
  mkdir -p "${MILVUS_HOME}/volumes/milvus"

  if [ ! -f "${MILVUS_HOME}/embedEtcd.yaml" ]; then
    cat > "${MILVUS_HOME}/embedEtcd.yaml" <<'EOF'
listen-client-urls: http://0.0.0.0:2379
advertise-client-urls: http://0.0.0.0:2379
quota-backend-bytes: 4294967296
auto-compaction-mode: revision
auto-compaction-retention: '1000'
EOF
  fi

  if [ ! -f "${MILVUS_HOME}/user.yaml" ]; then
    echo "# empty" > "${MILVUS_HOME}/user.yaml"
  fi
}

start() {
  if container_running; then
    echo "Milvus 已在运行。"
    status
    return 0
  fi

  if container_exists; then
    # 已有容器：端口映射在创建时固化，无法修改，只能按当前配置重启。
    echo "复用已存在的容器 ${CONTAINER}（端口映射沿用创建时的设置）。"
    echo "如需切换端口，先执行：bash scripts/milvus.sh recreate"
    docker start "${CONTAINER}" > /dev/null
  else
    init_config
    docker run -d \
      --name "${CONTAINER}" \
      --security-opt seccomp:unconfined \
      -e ETCD_USE_EMBED=true \
      -e ETCD_DATA_DIR=/var/lib/milvus/etcd \
      -e ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml \
      -e COMMON_STORAGETYPE=local \
      -v "${MILVUS_HOME}/volumes/milvus:/var/lib/milvus" \
      -v "${MILVUS_HOME}/embedEtcd.yaml:/milvus/configs/embedEtcd.yaml" \
      -v "${MILVUS_HOME}/user.yaml:/milvus/configs/user.yaml" \
      -p "${MILVUS_PORT}:19530" \
      -p "${MILVUS_HEALTH_PORT}:9091" \
      -p "${MILVUS_ETCD_PORT}:2379" \
      --health-cmd="curl -f http://localhost:9091/healthz" \
      --health-interval=30s \
      --health-start-period=90s \
      --health-timeout=20s \
      --health-retries=3 \
      "${MILVUS_IMAGE}" \
      milvus run standalone > /dev/null
  fi

  echo -n "等待 Milvus 就绪"
  for _ in $(seq 1 60); do
    if curl -sf "http://localhost:${MILVUS_HEALTH_PORT}/healthz" > /dev/null 2>&1; then
      echo ""
      echo "Start successfully."
      status
      return 0
    fi
    echo -n "."
    sleep 2
  done

  echo ""
  echo "等待超时，请检查日志：bash scripts/milvus.sh logs" >&2
  return 1
}

stop() {
  container_running || { echo "Milvus 未在运行。"; return 0; }
  docker stop "${CONTAINER}" > /dev/null
  echo "Stop successfully."
}

restart() {
  stop
  start
}

# 按脚本当前的端口配置重建容器。数据在 bind mount 中，不受影响。
recreate() {
  if container_exists; then
    docker rm -f "${CONTAINER}" > /dev/null
    echo "已删除旧容器（数据保留在 ${MILVUS_HOME}/volumes/milvus）。"
  fi
  start
}

status() {
  if ! container_exists; then
    echo "容器 ${CONTAINER} 不存在。执行 bash scripts/milvus.sh start 创建。"
    return 0
  fi
  docker ps -a --filter "name=^/${CONTAINER}$" \
    --format '容器: {{.Names}}
状态: {{.Status}}
端口: {{.Ports}}'
  echo "数据: ${MILVUS_HOME}/volumes/milvus"
  echo "连接: localhost:${MILVUS_PORT}    健康检查: http://localhost:${MILVUS_HEALTH_PORT}/healthz"
}

logs() {
  docker logs -f "${CONTAINER}"
}

delete() {
  read -r -p "将删除容器与 ${MILVUS_HOME} 下的全部数据，不可恢复。确认？[y/N] " ans
  case "${ans}" in
    y|Y|yes|YES) ;;
    *) echo "已取消。"; return 0 ;;
  esac
  container_exists && docker rm -f "${CONTAINER}" > /dev/null
  rm -rf "${MILVUS_HOME}/volumes" "${MILVUS_HOME}/embedEtcd.yaml" "${MILVUS_HOME}/user.yaml"
  echo "Delete successfully."
}

case "${1:-}" in
  start)    start ;;
  stop)     stop ;;
  restart)  restart ;;
  recreate) recreate ;;
  status)   status ;;
  logs)     logs ;;
  delete)   delete ;;
  *)
    echo "用法: bash scripts/milvus.sh {start|stop|restart|recreate|status|logs|delete}" >&2
    exit 1
    ;;
esac
