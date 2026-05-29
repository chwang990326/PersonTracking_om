import json
import os
import time
import uuid
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

import httpx
import redis.asyncio as redis
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


DEFAULT_GATEWAY_CONFIG = "config/gateway_pipelines.json"
DEFAULT_PIPELINES = "npu0_pipe1=http://127.0.0.1:8131,npu1_pipe1=http://127.0.0.1:8132"

DEFAULT_REDIS_HOST = "127.0.0.1"
DEFAULT_REDIS_PORT = 6379
DEFAULT_REDIS_PASSWORD = "redisForPersonTracking"
DEFAULT_REDIS_DB = 1
DEFAULT_REDIS_KEY_PREFIX = "gateway"
DEFAULT_REDIS_SOCKET_CONNECT_TIMEOUT = 2
DEFAULT_REDIS_SOCKET_TIMEOUT = 2
DEFAULT_CAMERA_ROUTE_TTL_SECONDS = 300

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}

UNLOCK_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
else
    return 0
end
"""


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


def parse_pipelines(raw: str) -> Dict[str, str]:
    """
    Parse GATEWAY_PIPELINES.

    Supported forms:
      npu0_pipe1=http://127.0.0.1:9001,npu0_pipe2=http://127.0.0.1:9002
      {"npu0_pipe1": "http://127.0.0.1:9001"}
    """
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("GATEWAY_PIPELINES is empty")

    if raw.startswith("{"):
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("GATEWAY_PIPELINES JSON must be an object")
        pipelines = {str(key): str(value).rstrip("/") for key, value in data.items()}
    else:
        pipelines = {}
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(
                    "GATEWAY_PIPELINES entries must use pipe_id=http://host:port"
                )
            pipe_id, url = item.split("=", 1)
            pipe_id = pipe_id.strip()
            url = url.strip().rstrip("/")
            if not pipe_id or not url:
                raise ValueError(f"Invalid pipeline entry: {item}")
            pipelines[pipe_id] = url

    for pipe_id, url in pipelines.items():
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Pipeline {pipe_id} url must start with http:// or https://")
    if not pipelines:
        raise ValueError("No valid pipelines configured")
    return pipelines


def load_pipelines_from_config(config_path: str) -> Dict[str, str]:
    path = Path(config_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    raw_pipelines = data.get("pipelines") if isinstance(data, dict) else data
    if isinstance(raw_pipelines, dict):
        pipelines = {
            str(pipe_id): str(url).rstrip("/")
            for pipe_id, url in raw_pipelines.items()
        }
    elif isinstance(raw_pipelines, list):
        pipelines = {}
        for index, item in enumerate(raw_pipelines):
            if isinstance(item, str):
                pipe_id = f"pipe{index}"
                url = item
            elif isinstance(item, dict):
                pipe_id = str(item.get("id") or f"pipe{index}")
                url = item.get("url")
            else:
                raise ValueError(f"Invalid pipeline item at index {index}")
            if not url:
                raise ValueError(f"Pipeline {pipe_id} has no url")
            pipelines[pipe_id] = str(url).rstrip("/")
    else:
        raise ValueError("Gateway config must contain a pipelines object or list")

    for pipe_id, url in pipelines.items():
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Pipeline {pipe_id} url must start with http:// or https://")
    if not pipelines:
        raise ValueError("No valid pipelines configured")
    return pipelines


def load_pipelines() -> tuple[Dict[str, str], str]:
    raw_pipelines = os.getenv("GATEWAY_PIPELINES")
    if raw_pipelines:
        return parse_pipelines(raw_pipelines), "env:GATEWAY_PIPELINES"

    config_path = os.getenv("GATEWAY_CONFIG", DEFAULT_GATEWAY_CONFIG)
    resolved_path = Path(config_path)
    if not resolved_path.is_absolute():
        resolved_path = Path.cwd() / resolved_path
    if resolved_path.exists():
        return load_pipelines_from_config(str(resolved_path)), str(resolved_path)

    return parse_pipelines(DEFAULT_PIPELINES), "built-in default"


class GatewaySettings:
    def __init__(self) -> None:
        self.pipelines, self.pipeline_source = load_pipelines()
        self.redis_host = _env_str("REDIS_HOST", DEFAULT_REDIS_HOST)
        self.redis_port = _env_int("REDIS_PORT", DEFAULT_REDIS_PORT)
        self.redis_db = _env_int("REDIS_DB", DEFAULT_REDIS_DB)
        self.redis_password = _env_str("REDIS_PASSWORD", DEFAULT_REDIS_PASSWORD)
        self.redis_key_prefix = _env_str("REDIS_KEY_PREFIX", DEFAULT_REDIS_KEY_PREFIX)
        self.gateway_key_prefix = ""
        self.camera_route_ttl_seconds = DEFAULT_CAMERA_ROUTE_TTL_SECONDS
        self.lock_ttl_ms = _env_int("GATEWAY_ROUTE_LOCK_TTL_MS", 5000)
        self.lock_retry_sleep_ms = _env_int("GATEWAY_ROUTE_LOCK_RETRY_SLEEP_MS", 50)
        self.lock_max_retries = _env_int("GATEWAY_ROUTE_LOCK_MAX_RETRIES", 100)
        self.backend_timeout_seconds = _env_float("GATEWAY_BACKEND_TIMEOUT_SECONDS", 120.0)


class ManualRouteRequest(BaseModel):
    pipe_id: str = Field(..., description="Pipeline id from GATEWAY_PIPELINES")


class CameraRouteStore:
    def __init__(self, client: redis.Redis, settings: GatewaySettings) -> None:
        self.client = client
        self.settings = settings
        self.pipeline_ids = list(settings.pipelines.keys())

    def _key(self, name: str) -> str:
        parts = [
            self.settings.redis_key_prefix,
            self.settings.gateway_key_prefix,
            name,
        ]
        return ":".join(part for part in parts if part)

    def route_key(self, camera_id: str) -> str:
        return self._key(f"camera_route:{camera_id}")

    def lock_key(self, camera_id: str) -> str:
        return self._key(f"camera_route_lock:{camera_id}")

    def assign_lock_key(self) -> str:
        return self._key("pipeline_assign_lock")

    def pipeline_cameras_key(self, pipe_id: str) -> str:
        return self._key(f"pipeline_cameras:{pipe_id}")

    def routes_hash_key(self) -> str:
        return self._key("camera_routes")

    async def get_route(self, camera_id: str, refresh_ttl: bool = False) -> Optional[str]:
        pipe_id = await self.client.get(self.route_key(camera_id))
        if pipe_id in self.settings.pipelines:
            if refresh_ttl:
                await self.client.expire(
                    self.route_key(camera_id),
                    self.settings.camera_route_ttl_seconds,
                )
            return pipe_id
        return None

    async def get_or_assign_route(self, camera_id: str) -> str:
        pipe_id = await self.get_route(camera_id, refresh_ttl=True)
        if pipe_id:
            return pipe_id

        token = uuid.uuid4().hex
        lock_key = self.lock_key(camera_id)

        for _ in range(self.settings.lock_max_retries):
            locked = await self.client.set(
                lock_key,
                token,
                nx=True,
                px=self.settings.lock_ttl_ms,
            )
            if locked:
                try:
                    pipe_id = await self.get_route(camera_id, refresh_ttl=True)
                    if pipe_id:
                        return pipe_id
                    return await self.assign_new_route(camera_id)
                finally:
                    await self.client.eval(UNLOCK_SCRIPT, 1, lock_key, token)

            await asyncio.sleep(self.settings.lock_retry_sleep_ms / 1000.0)
            pipe_id = await self.get_route(camera_id, refresh_ttl=True)
            if pipe_id:
                return pipe_id

        raise HTTPException(
            status_code=503,
            detail=f"Timed out while assigning route for camera_id={camera_id}",
        )

    async def assign_new_route(self, camera_id: str) -> str:
        token = uuid.uuid4().hex
        lock_key = self.assign_lock_key()

        for _ in range(self.settings.lock_max_retries):
            locked = await self.client.set(
                lock_key,
                token,
                nx=True,
                px=self.settings.lock_ttl_ms,
            )
            if locked:
                try:
                    pipe_id = await self.choose_pipeline()
                    await self.bind_route(camera_id, pipe_id)
                    return pipe_id
                finally:
                    await self.client.eval(UNLOCK_SCRIPT, 1, lock_key, token)

            await asyncio.sleep(self.settings.lock_retry_sleep_ms / 1000.0)

        raise HTTPException(
            status_code=503,
            detail="Timed out while assigning pipeline",
        )

    async def bind_route(self, camera_id: str, pipe_id: str) -> None:
        if pipe_id not in self.settings.pipelines:
            raise HTTPException(status_code=400, detail=f"Unknown pipe_id: {pipe_id}")
        await self.client.set(
            self.route_key(camera_id),
            pipe_id,
            ex=self.settings.camera_route_ttl_seconds,
        )
        await self.client.hset(self.routes_hash_key(), camera_id, pipe_id)
        await self.client.sadd(self.pipeline_cameras_key(pipe_id), camera_id)

    async def choose_pipeline(self) -> str:
        await self.cleanup_stale_routes()
        best_pipe_id = None
        best_count = None
        for pipe_id in self.pipeline_ids:
            count = await self.client.scard(self.pipeline_cameras_key(pipe_id))
            if best_count is None or count < best_count:
                best_count = count
                best_pipe_id = pipe_id
        if best_pipe_id is None:
            raise HTTPException(status_code=503, detail="No pipelines configured")
        return best_pipe_id

    async def cleanup_stale_routes(self) -> int:
        removed = 0
        for pipe_id in self.pipeline_ids:
            key = self.pipeline_cameras_key(pipe_id)
            camera_ids = await self.client.smembers(key)
            for camera_id in camera_ids:
                current_pipe_id = await self.client.get(self.route_key(camera_id))
                if current_pipe_id != pipe_id:
                    await self.client.srem(key, camera_id)
                    await self.client.hdel(self.routes_hash_key(), camera_id)
                    removed += 1
        return removed

    async def pipeline_stats(self) -> Dict[str, dict]:
        await self.cleanup_stale_routes()
        stats = {}
        for pipe_id in self.pipeline_ids:
            key = self.pipeline_cameras_key(pipe_id)
            cameras = sorted(await self.client.smembers(key))
            stats[pipe_id] = {
                "url": self.settings.pipelines[pipe_id],
                "camera_count": len(cameras),
                "camera_ids": cameras,
            }
        return stats

    async def delete_route(self, camera_id: str) -> bool:
        pipe_id = await self.get_route(camera_id)
        if pipe_id:
            await self.client.srem(self.pipeline_cameras_key(pipe_id), camera_id)
        await self.client.hdel(self.routes_hash_key(), camera_id)
        deleted = await self.client.delete(self.route_key(camera_id))
        return deleted > 0


def build_forward_headers(request: Request, camera_id: str, pipe_id: str) -> Dict[str, str]:
    headers = {
        name: value
        for name, value in request.headers.items()
        if name.lower() not in HOP_BY_HOP_HEADERS
    }
    headers["x-camera-id"] = camera_id
    headers["x-gateway-pipeline"] = pipe_id
    return headers


def build_backend_url(base_url: str, request: Request) -> str:
    path = request.url.path
    query = request.url.query
    url = f"{base_url.rstrip('/')}{path}"
    if query:
        url = f"{url}?{query}"
    return url


def create_redis_client(settings: GatewaySettings) -> redis.Redis:
    return redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password or None,
        decode_responses=True,
        socket_connect_timeout=DEFAULT_REDIS_SOCKET_CONNECT_TIMEOUT,
        socket_timeout=DEFAULT_REDIS_SOCKET_TIMEOUT,
    )


settings = GatewaySettings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_client = create_redis_client(settings)
    http_client = httpx.AsyncClient(
        timeout=settings.backend_timeout_seconds,
        trust_env=False,
    )
    app.state.redis = redis_client
    app.state.http = http_client
    app.state.routes = CameraRouteStore(redis_client, settings)
    try:
        await redis_client.ping()
        yield
    finally:
        await http_client.aclose()
        await redis_client.aclose()


app = FastAPI(title="PersonTracking Camera Gateway", version="1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    start = time.perf_counter()
    await app.state.redis.ping()
    latency_ms = (time.perf_counter() - start) * 1000.0
    return {
        "status": "ok",
        "redis_latency_ms": round(latency_ms, 2),
        "pipeline_count": len(settings.pipelines),
        "pipeline_source": settings.pipeline_source,
        "camera_route_ttl_seconds": settings.camera_route_ttl_seconds,
    }


@app.get("/gateway/pipelines")
async def list_pipelines():
    return {
        "pipelines": await app.state.routes.pipeline_stats(),
    }


@app.get("/gateway/routes/{camera_id}")
async def get_camera_route(camera_id: str):
    pipe_id = await app.state.routes.get_route(camera_id)
    if not pipe_id:
        raise HTTPException(status_code=404, detail="camera_id has no route")
    return {
        "camera_id": camera_id,
        "pipe_id": pipe_id,
        "url": settings.pipelines[pipe_id],
    }


@app.post("/gateway/routes/{camera_id}")
async def bind_camera_route(camera_id: str, route: ManualRouteRequest):
    old_pipe_id = await app.state.routes.get_route(camera_id)
    if old_pipe_id:
        await app.state.redis.srem(
            app.state.routes.pipeline_cameras_key(old_pipe_id),
            camera_id,
        )
    await app.state.routes.bind_route(camera_id, route.pipe_id)
    return {
        "camera_id": camera_id,
        "pipe_id": route.pipe_id,
        "url": settings.pipelines[route.pipe_id],
    }


@app.delete("/gateway/routes/{camera_id}")
async def delete_camera_route(camera_id: str):
    deleted = await app.state.routes.delete_route(camera_id)
    return {
        "camera_id": camera_id,
        "deleted": deleted,
    }


@app.post("/api/v1/person/detect")
async def proxy_person_detect(request: Request):
    body = await request.body()
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=400,
            content={"code": 1001, "message": "Invalid JSON body", "timestamp": int(time.time() * 1000)},
        )

    camera_id = str(payload.get("camera_id") or "").strip()
    if not camera_id:
        return JSONResponse(
            status_code=400,
            content={"code": 1002, "message": "camera_id required", "timestamp": int(time.time() * 1000)},
        )

    pipe_id = await app.state.routes.get_or_assign_route(camera_id)
    backend_base = settings.pipelines[pipe_id]
    backend_url = build_backend_url(backend_base, request)
    headers = build_forward_headers(request, camera_id, pipe_id)

    try:
        response = await app.state.http.post(
            backend_url,
            content=body,
            headers=headers,
        )
    except httpx.RequestError as exc:
        return JSONResponse(
            status_code=502,
            content={
                "code": 1502,
                "message": f"Backend request failed: {exc}",
                "camera_id": camera_id,
                "pipe_id": pipe_id,
                "timestamp": int(time.time() * 1000),
            },
        )

    response_headers = {
        name: value
        for name, value in response.headers.items()
        if name.lower() not in HOP_BY_HOP_HEADERS
    }
    response_headers["x-gateway-pipeline"] = pipe_id
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=response_headers,
        media_type=response.headers.get("content-type"),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=_env_int("GATEWAY_PORT", 8130))
