# 服务查看与排查命令

本文档用于查看 PersonTracking 服务、Docker、Redis、Gateway、算法 worker 的运行情况。

## 基本配置

当前默认配置：

```bash
项目目录：/home/wangchenhao/PersonTracking_om
算法镜像：person_tracking_full:v2
Redis 容器：gateway-redis
Redis 端口：6379
Redis 密码：redisForPersonTracking
Redis DB：1
Redis key 前缀：gateway
Gateway 端口：8130
算法容器前缀：person_tracking_om_d
```

默认 worker 端口：

```bash
npu0_worker0 -> 8131
npu1_worker0 -> 8132
npu0_worker1 -> 8133
npu1_worker1 -> 8134
```

## 启动和关闭

启动全部服务：

```bash
cd /home/wangchenhao/PersonTracking_om
./start_all_services.sh
```

关闭算法容器和 Gateway，保留 Redis：

```bash
cd /home/wangchenhao/PersonTracking_om
./stop_all_services.sh
```

关闭并删除 Redis 容器：

```bash
cd /home/wangchenhao/PersonTracking_om
STOP_REDIS=1 ./stop_all_services.sh
```

Redis 端口、密码改过后，建议用 `STOP_REDIS=1` 删除旧 Redis 容器，再重新启动。

## 查看 Docker

查看正在运行的容器：

```bash
sudo docker ps
```

查看所有容器，包括已退出的：

```bash
sudo docker ps -a
```

只看本项目相关容器：

```bash
sudo docker ps -a | grep -E 'person_tracking|gateway-redis'
```

查看算法容器日志：

```bash
sudo docker logs -f person_tracking_om_d0
sudo docker logs -f person_tracking_om_d1
```

查看最近 200 行日志：

```bash
sudo docker logs --tail=200 person_tracking_om_d0
sudo docker logs --tail=200 person_tracking_om_d1
```

进入容器：

```bash
sudo docker exec -it person_tracking_om_d0 bash
sudo docker exec -it person_tracking_om_d1 bash
```

查看容器里的服务进程：

```bash
sudo docker exec -it person_tracking_om_d0 ps -ef | grep -E 'api_server|gateway_server'
sudo docker exec -it person_tracking_om_d1 ps -ef | grep -E 'api_server|gateway_server'
```

查看容器里的 Redis 环境变量：

```bash
sudo docker exec -it person_tracking_om_d0 env | grep REDIS
sudo docker exec -it person_tracking_om_d1 env | grep REDIS
```

## 查看端口

查看 Gateway 和 worker 端口：

```bash
sudo ss -lntp | grep -E '8130|8131|8132|8133|8134'
```

查看 Redis 端口：

```bash
sudo ss -lntp | grep ':6379'
```

如果启动 Redis 报 `port is already allocated`，说明宿主机 `6379` 已经被占用，使用上面的命令确认占用进程。

## 查看 Gateway

健康检查：

```bash
curl --noproxy '*' http://127.0.0.1:8130/health
```

查看 Gateway 当前加载的 pipeline：

```bash
curl --noproxy '*' http://127.0.0.1:8130/gateway/pipelines
```

查看某个摄像头分配到了哪个 pipeline：

```bash
curl --noproxy '*' http://127.0.0.1:8130/gateway/routes/207
```

删除某个摄像头路由：

```bash
curl --noproxy '*' -X DELETE http://127.0.0.1:8130/gateway/routes/207
```

手动绑定某个摄像头到指定 pipeline：

```bash
curl --noproxy '*' -X POST http://127.0.0.1:8130/gateway/routes/207 \
  -H 'Content-Type: application/json' \
  -d '{"pipe_id":"npu0_worker0"}'
```

## 查看 Redis

Redis 容器是否存在：

```bash
sudo docker ps -a | grep gateway-redis
```

Redis 是否可连接：

```bash
sudo docker exec -it gateway-redis redis-cli -a redisForPersonTracking ping
```

进入 Redis DB 1：

```bash
sudo docker exec -it gateway-redis redis-cli -a redisForPersonTracking -n 1
```

查看 Gateway 相关 key：

```bash
sudo docker exec -it gateway-redis redis-cli -a redisForPersonTracking -n 1 keys 'gateway:*'
```

查看 camera_id 到 pipeline 的路由：

```bash
sudo docker exec -it gateway-redis redis-cli -a redisForPersonTracking -n 1 get 'gateway:camera_route:207'
```

查看某个 camera 路由 TTL：

```bash
sudo docker exec -it gateway-redis redis-cli -a redisForPersonTracking -n 1 ttl 'gateway:camera_route:207'
```

查看所有 camera 路由 hash：

```bash
sudo docker exec -it gateway-redis redis-cli -a redisForPersonTracking -n 1 hgetall 'gateway:camera_routes'
```

查看某个 pipeline 当前绑定了哪些 camera：

```bash
sudo docker exec -it gateway-redis redis-cli -a redisForPersonTracking -n 1 smembers 'gateway:pipeline_cameras:npu0_worker0'
```

删除某个 camera 的路由：

```bash
sudo docker exec -it gateway-redis redis-cli -a redisForPersonTracking -n 1 del 'gateway:camera_route:207'
```

## 查看 worker

查看 worker docs 是否正常：

```bash
curl --noproxy '*' http://127.0.0.1:8131/docs
curl --noproxy '*' http://127.0.0.1:8132/docs
curl --noproxy '*' http://127.0.0.1:8133/docs
curl --noproxy '*' http://127.0.0.1:8134/docs
```

查看 Gateway 配置文件：

```bash
cat /home/wangchenhao/PersonTracking_om/config/gateway_pipelines.json
```

## 常见问题

### curl 没有输出

服务器如果有代理，使用：

```bash
curl --noproxy '*' -v http://127.0.0.1:8130/health
```

### Redis 6379 端口被占用

查看占用：

```bash
sudo ss -lntp | grep ':6379'
sudo docker ps -a | grep -E 'redis|6379'
```

如果是旧 Redis 容器占用：

```bash
sudo docker rm -f 容器名
```

如果是系统 Redis 占用：

```bash
sudo systemctl stop redis
sudo systemctl stop redis-server
```

### 第二个 Docker 连不上 Redis

确认第二个容器是否拿到了 Redis 环境变量：

```bash
sudo docker exec -it person_tracking_om_d1 env | grep REDIS
```

正常应包含：

```bash
REDIS_HOST=host.docker.internal
REDIS_PORT=6379
REDIS_DB=1
REDIS_PASSWORD=redisForPersonTracking
REDIS_KEY_PREFIX=gateway
```

### Gateway 转发失败

先看 Gateway 能否访问：

```bash
curl --noproxy '*' http://127.0.0.1:8130/health
curl --noproxy '*' http://127.0.0.1:8130/gateway/pipelines
```

再看 worker 是否正常：

```bash
curl --noproxy '*' http://127.0.0.1:8131/docs
curl --noproxy '*' http://127.0.0.1:8132/docs
curl --noproxy '*' http://127.0.0.1:8133/docs
curl --noproxy '*' http://127.0.0.1:8134/docs
```

最后看容器日志：

```bash
sudo docker logs --tail=200 person_tracking_om_d0
sudo docker logs --tail=200 person_tracking_om_d1
```
