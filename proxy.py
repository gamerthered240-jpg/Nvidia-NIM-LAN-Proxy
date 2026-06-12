"""
Nvidia NIM → OpenAI-Compatible LAN Proxy  v1.3
================================================
"""

import os
import json
import logging
import time
from typing import Optional

import httpx
from fastapi import FastAPI, Request, HTTPException, Header, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────

PROXY_API_KEY   = os.getenv("PROXY_API_KEY", "")
PROXY_HOST      = os.getenv("PROXY_HOST", "0.0.0.0")
PROXY_PORT      = int(os.getenv("PROXY_PORT", "8016"))
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com")
NVIDIA_API_KEY  = os.getenv("NVIDIA_API_KEY", "")
TIMEOUT         = int(os.getenv("REQUEST_TIMEOUT", "180"))
LOG_LEVEL       = os.getenv("LOG_LEVEL", "INFO").upper()

# Auto-strip /v1 from base URL
NVIDIA_BASE_URL = NVIDIA_BASE_URL.rstrip("/")
if NVIDIA_BASE_URL.endswith("/v1"):
    NVIDIA_BASE_URL = NVIDIA_BASE_URL[: -len("/v1")]

# Parameters to strip before forwarding
_strip_raw = os.getenv(
    "STRIP_PARAMS",
    "logprobs,top_logprobs,tools,tool_choice,response_format,n",
)
STRIP_PARAMS: set[str] = {k.strip() for k in _strip_raw.split(",") if k.strip()}

# Parameter renaming
MAP_PARAMS: dict[str, str] = {}
for _e in os.getenv("MAP_PARAMS", "max_completion_tokens:max_tokens").split(","):
    _e = _e.strip()
    if ":" in _e:
        _old, _new = _e.split(":", 1)
        _old, _new = _old.strip(), _new.strip()
        if _old and _new:
            MAP_PARAMS[_old] = _new

# Model mapping
MODELS: dict[str, str] = {}
for _e in os.getenv("MODELS", "").split(","):
    _e = _e.strip().strip("\\")
    if ":" in _e:
        _n, _i = _e.split(":", 1)
        _n, _i = _n.strip(), _i.strip().strip("\\")
        if _n and _i:
            MODELS[_n] = _i

# ─── Logging ──────────────────────────────────────────────────────

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("nim-proxy")

# ─── FastAPI ──────────────────────────────────────────────────────

app = FastAPI(title="NIM → OpenAI Proxy", version="1.3")

# CORS — answers browser OPTIONS preflight requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Request logger
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    resp = await call_next(request)
    ms = (time.time() - start) * 1000
    log.info(
        "  ← %s %s  → %s  (%.0f ms)",
        request.method,
        request.url.path,
        resp.status_code,
        ms,
    )
    return resp


async def _auth(authorization: Optional[str] = Header(None)):
    if PROXY_API_KEY:
        if not authorization or authorization != f"Bearer {PROXY_API_KEY}":
            raise HTTPException(status_code=401, detail="Unauthorized")


# ─── Core chat completions logic ──────────────────────────────────

async def _handle_chat(request: Request):
    """Shared logic for all POST paths that handle chat completions."""
    body = await request.json()

    # ── Resolve model name ──
    requested = body.get("model", "")
    resolved  = MODELS.get(requested, requested)
    body["model"] = resolved

    # ── Strip unsupported params ──
    for key in STRIP_PARAMS:
        body.pop(key, None)

    # ── Rename params ──
    for old_key, new_key in MAP_PARAMS.items():
        if old_key in body:
            body[new_key] = body.pop(old_key)

    # ── Build upstream request ──
    is_stream = body.get("stream", False)
    headers   = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type":  "application/json",
    }
    url = f"{NVIDIA_BASE_URL}/v1/chat/completions"

    log.info("▶ %s → %s  stream=%s", requested, resolved, is_stream)

    if is_stream:
        return StreamingResponse(
            _forward_stream(url, headers, body),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(url, headers=headers, json=body)
    except httpx.ReadTimeout:
        raise HTTPException(504, "Upstream timeout")

    if resp.status_code != 200:
        log.error("✖ Nvidia %s: %s", resp.status_code, resp.text[:500])
        raise HTTPException(resp.status_code, resp.text)

    return JSONResponse(content=resp.json())


# ─── Endpoints ────────────────────────────────────────────────────
# We register POST on every path a frontend might use, so it works
# no matter how you configure the API URL in JanitorAI / SillyTavern.

@app.get("/")
async def root():
    return {"status": "ok", "models": list(MODELS.keys())}


@app.get("/v1/models")
@app.get("/models")
async def list_models(_=Depends(_auth)):
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "created": 0,
                "owned_by": "nvidia",
            }
            for name in MODELS
        ],
    }


# ★ All these POST paths do the same thing — route to Nvidia NIM
@app.post("/")
@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(request: Request, _=Depends(_auth)):
    return await _handle_chat(request)


# Preflight handlers for all those paths
@app.options("/")
@app.options("/chat/completions")
@app.options("/v1/chat/completions")
@app.options("/v1/models")
@app.options("/models")
async def preflight():
    return JSONResponse(content={}, status_code=200)


# ─── Stream forwarder ─────────────────────────────────────────────

async def _forward_stream(url: str, headers: dict, body: dict):
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            async with client.stream("POST", url, headers=headers, json=body) as resp:
                if resp.status_code != 200:
                    err = await resp.aread()
                    log.error("✖ stream %s: %s", resp.status_code, err.decode()[:500])
                    payload = json.dumps(
                        {"error": {"message": err.decode(), "type": "upstream_error"}}
                    )
                    yield f"data: {payload}\n\n".encode()
                    yield b"data: [DONE]\n\n"
                    return

                async for chunk in resp.aiter_bytes():
                    yield chunk

    except httpx.ReadTimeout:
        log.error("✖ stream read timeout")
        yield b"data: [DONE]\n\n"
    except Exception as exc:
        log.error("✖ stream error: %s", exc)
        payload = json.dumps(
            {"error": {"message": str(exc), "type": "proxy_error"}}
        )
        yield f"data: {payload}\n\n".encode()
        yield b"data: [DONE]\n\n"


# ─── Entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    log.info("=" * 58)
    log.info("  Nvidia NIM → OpenAI  LAN Proxy  v1.3")
    log.info("  Listening  : http://%s:%s", PROXY_HOST, PROXY_PORT)
    log.info("  Base URL   : %s", NVIDIA_BASE_URL)
    log.info("  Upstream   : %s/v1/chat/completions", NVIDIA_BASE_URL)
    log.info("  Models     : %s", ", ".join(MODELS) or "(none)")
    for name, mid in MODELS.items():
        log.info("               %-20s → %s", name, mid)
    log.info("  POST paths : /  /chat/completions  /v1/chat/completions")
    log.info("  Strip params: %s", ", ".join(STRIP_PARAMS) or "(none)")
    log.info("  Map params : %s", ", ".join(f"{k}→{v}" for k, v in MAP_PARAMS.items()) or "(none)")
    if not NVIDIA_API_KEY:
        log.warning("  ⚠  NVIDIA_API_KEY is not set!")
    if not MODELS:
        log.warning("  ⚠  No models configured!")
    log.info("=" * 58)

    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT, log_level=LOG_LEVEL.lower())