## 下载脚本：
```bash
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
```
## 启动 Milvus：

```bash
bash standalone_embed.sh start
```

启动成功后，日志中会出现 Start successfully. 的提示。此时，一个名为 milvus-standalone 的 Docker 容器已经在 19530 端口启动。

## 停止服务：
```bash
bash standalone_embed.sh stop
```

## 删除容器和数据：

```bash
bash standalone_embed.sh delete
```
