"""
SAM Load Balancer - Single port routing to multiple SAM instances.
Exposes port 8001 and load-balances to backend instances on ports 8010-8025.
"""
import asyncio
import aiohttp
import argparse
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn
import logging
from typing import List
import random
import time
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sam_lb")

app = FastAPI(title="SAM Load Balancer")

# Backend instances
BACKENDS: List[str] = []
# Track backend health and load
backend_stats = {}
# Round-robin index
rr_index = 0
# Request queue per backend for load tracking
backend_load = {}


class BackendPool:
    def __init__(self, backends: List[str]):
        self.backends = backends
        self.healthy = {b: True for b in backends}
        self.response_times = {b: deque(maxlen=10) for b in backends}
        self.rr_index = 0
        self.lock = asyncio.Lock()
    
    async def get_backend(self, strategy="round_robin") -> str:
        """Get next available backend."""
        healthy_backends = [b for b in self.backends if self.healthy[b]]
        
        if not healthy_backends:
            # Try all backends if none marked healthy
            healthy_backends = self.backends
        
        if strategy == "round_robin":
            async with self.lock:
                self.rr_index = (self.rr_index + 1) % len(healthy_backends)
                return healthy_backends[self.rr_index]
        elif strategy == "least_latency":
            # Pick backend with lowest average response time
            def avg_latency(b):
                times = self.response_times.get(b, [])
                return sum(times) / len(times) if times else 0
            return min(healthy_backends, key=avg_latency)
        elif strategy == "random":
            return random.choice(healthy_backends)
        else:
            return healthy_backends[0]
    
    def record_response(self, backend: str, latency: float, success: bool):
        """Record response metrics."""
        if backend in self.response_times:
            self.response_times[backend].append(latency)
        self.healthy[backend] = success
    
    def mark_unhealthy(self, backend: str):
        self.healthy[backend] = False
    
    def mark_healthy(self, backend: str):
        self.healthy[backend] = True


pool: BackendPool = None
http_session: aiohttp.ClientSession = None


@app.on_event("startup")
async def startup():
    global http_session
    timeout = aiohttp.ClientTimeout(total=120)
    connector = aiohttp.TCPConnector(limit=100)
    http_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    logger.info(f"Load balancer started with {len(pool.backends)} backends")


@app.on_event("shutdown")
async def shutdown():
    global http_session
    if http_session:
        await http_session.close()


@app.get("/health")
async def health():
    healthy_count = sum(1 for b in pool.backends if pool.healthy[b])
    return {
        "status": "healthy" if healthy_count > 0 else "degraded",
        "backends_total": len(pool.backends),
        "backends_healthy": healthy_count,
        "backends": {b: pool.healthy[b] for b in pool.backends}
    }


@app.get("/stats")
async def stats():
    return {
        "backends": {
            b: {
                "healthy": pool.healthy[b],
                "avg_latency_ms": sum(pool.response_times[b]) / len(pool.response_times[b]) * 1000 
                    if pool.response_times[b] else 0
            }
            for b in pool.backends
        }
    }


async def proxy_request(request: Request, path: str):
    """Proxy request to a backend."""
    backend = await pool.get_backend("round_robin")
    url = f"{backend}{path}"
    
    # Get request body
    body = await request.body()
    
    t0 = time.time()
    try:
        async with http_session.request(
            method=request.method,
            url=url,
            headers={k: v for k, v in request.headers.items() if k.lower() != 'host'},
            data=body,
        ) as resp:
            latency = time.time() - t0
            pool.record_response(backend, latency, resp.status < 500)
            
            content = await resp.read()
            # Return raw response (don't re-serialize JSON)
            from starlette.responses import Response
            return Response(
                content=content,
                status_code=resp.status,
                media_type=resp.content_type
            )
    except Exception as e:
        latency = time.time() - t0
        pool.record_response(backend, latency, False)
        logger.error(f"Backend {backend} error: {e}")
        
        # Try another backend
        for retry_backend in pool.backends:
            if retry_backend != backend and pool.healthy.get(retry_backend, True):
                try:
                    url = f"{retry_backend}{path}"
                    async with http_session.request(
                        method=request.method,
                        url=url,
                        headers={k: v for k, v in request.headers.items() if k.lower() != 'host'},
                        data=body,
                    ) as resp:
                        content = await resp.read()
                        from starlette.responses import Response
                        return Response(
                            content=content,
                            status_code=resp.status,
                            media_type=resp.content_type
                        )
                except:
                    continue
        
        raise HTTPException(status_code=503, detail=f"All backends unavailable: {e}")


# Proxy all SAM endpoints
@app.api_route("/api/v1/image/segment", methods=["POST"])
async def segment(request: Request):
    return await proxy_request(request, "/api/v1/image/segment")


@app.api_route("/segment", methods=["POST"])
async def segment_alt(request: Request):
    return await proxy_request(request, "/segment")


@app.api_route("/set_image", methods=["POST"])
async def set_image(request: Request):
    return await proxy_request(request, "/set_image")


@app.api_route("/predict", methods=["POST"])
async def predict(request: Request):
    return await proxy_request(request, "/predict")


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all(request: Request, path: str):
    return await proxy_request(request, f"/{path}")


def main():
    global pool
    
    parser = argparse.ArgumentParser(description="SAM Load Balancer")
    parser.add_argument("--port", type=int, default=8001, help="Load balancer port")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--backends", type=str, required=True,
                        help="Comma-separated backend URLs, e.g., http://localhost:8010,http://localhost:8011")
    parser.add_argument("--backend-base-port", type=int, default=8010,
                        help="Base port for auto-generated backends")
    parser.add_argument("--num-backends", type=int, default=16,
                        help="Number of backends if using auto-generation")
    args = parser.parse_args()
    
    if args.backends:
        backends = [b.strip() for b in args.backends.split(",")]
    else:
        # Auto-generate backend URLs
        backends = [f"http://localhost:{args.backend_base_port + i}" 
                    for i in range(args.num_backends)]
    
    pool = BackendPool(backends)
    
    logger.info(f"Starting load balancer on port {args.port}")
    logger.info(f"Backends: {backends}")
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
