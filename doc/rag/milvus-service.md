# Milvus 本地服务

本地用 Milvus standalone（内嵌 etcd）作为向量库，通过 `scripts/milvus.sh` 管理，无需再下载官方
`standalone_embed.sh`。脚本已把官方启动参数固化，唯一差异是健康/metrics 端口对外映射为 **19091**
（容器内仍是 9091），避免与本机其它服务抢占 9091。

## 端口

| 用途 | 宿主机端口 | 容器端口 |
| --- | --- | --- |
| gRPC（客户端连接） | 19530 | 19530 |
| 健康检查 / metrics | 19091 | 9091 |
| 内嵌 etcd | 2379 | 2379 |

数据与配置位于 `~/.refund-agent-milvus`，容器删除重建不影响数据。

## 常用命令

```bash
bash scripts/milvus.sh start      # 启动，等待就绪后打印状态
bash scripts/milvus.sh status     # 查看运行状态与端口映射
bash scripts/milvus.sh logs       # 跟踪容器日志
bash scripts/milvus.sh stop       # 停止容器，保留数据
bash scripts/milvus.sh restart    # 重启
```

启动成功后会输出 `Start successfully.`，此时 `localhost:19530` 可供客户端连接，
`http://localhost:19091/healthz` 返回 200。

## 修改端口

端口映射在容器创建时固化，`start` 无法改动已存在的容器。改端口需要重建容器（数据保留）：

```bash
MILVUS_HEALTH_PORT=19092 bash scripts/milvus.sh recreate
```

可覆盖的环境变量：`MILVUS_HOME`、`MILVUS_PORT`、`MILVUS_HEALTH_PORT`、`MILVUS_ETCD_PORT`、`MILVUS_IMAGE`。

## 删除数据

```bash
bash scripts/milvus.sh delete     # 删除容器与全部数据，会二次确认，不可恢复
```
