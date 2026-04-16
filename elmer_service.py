"""
Elmer Standalone Service — Brainstem with Full Brain Capability

Runs Elmer as an independent process with:
  - Full engine (all sockets including brain sockets + Lenia)
  - Tract drain loop — reads topology deltas from the River
  - Brain buffer + async drain — Lenia steps in background
  - KISS observation — input optimization logging
  - FastAPI endpoints for health, status, and manual input

The fan-out hook (elmer_hook.py) runs Elmer in lightweight mode
inside the NeuroGraph RPC process. This service is the full organism
with dedicated resources for brain processing.

# ---- Changelog ----
# [2026-03-28] Claude Code (Opus 4.6) — Initial implementation
#   What: Standalone Elmer service with tract drain, full brains, FastAPI
#   Why:  Brain sockets need dedicated process — Lenia steps (65s+ on CPU)
#         block the fan-out RPC. Standalone service drains tract for real
#         topology data and runs brains at their own pace.
#   How:  FastAPI + uvicorn. Tract drain loop on background thread.
#         Engine starts with full brains (skip_brains=False). KISS + brain
#         buffer from engine. Health/status/process endpoints.
# [2026-04-15] Claude Code (Sonnet 4.6) — Punchlist #137: Fix bridge.drain() no-op
#   What: Replace bridge.drain() / bridge.read_tract() branches with
#         bridge.sync_state() in the tract drain loop.
#   Why:  NGTractBridge has no public drain() method. hasattr guard silently
#         skipped the entire tract drain — drain loop was a no-op for River.
#   How:  sync_state(local_state={}, module_id="elmer") calls _drain_all()
#         internally, clearing tracts and updating bridge state. Text extraction
#         loop removed — Law 7 violation deferred to punchlist #154.
# -------------------
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import signal
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

# Elmer's own directory must be importable
_elmer_dir = os.path.dirname(os.path.abspath(__file__))
if _elmer_dir not in sys.path:
    sys.path.insert(0, _elmer_dir)

from fastapi import FastAPI
from pydantic import BaseModel

from core.config import load_config
from runtime.engine import ElmerEngine
import ng_autonomic

logger = logging.getLogger("elmer.service")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)

# ── Configuration ─────────────────────────────────────────────────

SERVICE_PORT = int(os.environ.get("ELMER_SERVICE_PORT", "7438"))
TRACT_DRAIN_INTERVAL = float(os.environ.get("ELMER_DRAIN_INTERVAL", "10.0"))
BRAIN_LOAD_DELAY = float(os.environ.get("ELMER_BRAIN_DELAY", "5.0"))


# ── Global State ──────────────────────────────────────────────────

_engine: Optional[ElmerEngine] = None
_drain_thread: Optional[threading.Thread] = None
_drain_running = False


# ── Tract Drain ───────────────────────────────────────────────────

def _drain_loop():
    """Background loop that drains Elmer's tract and feeds the engine.

    Reads topology deltas deposited by NeuroGraph via the River.
    Each delta becomes a process_text() call — lightweight sockets
    process synchronously, brain sockets via the async buffer.
    """
    global _drain_running

    logger.info("Tract drain loop started (interval=%.1fs)", TRACT_DRAIN_INTERVAL)

    while _drain_running:
        try:
            if _engine and _engine._started and _engine._ecosystem:
                bridge = getattr(_engine._ecosystem, '_peer_bridge', None)
                if bridge and hasattr(bridge, 'sync_state'):
                    bridge.sync_state(local_state={}, module_id="elmer")
        except Exception as exc:
            logger.warning("Tract drain error: %s", exc)

        time.sleep(TRACT_DRAIN_INTERVAL)

    logger.info("Tract drain loop stopped")


def _extract_text_from_delta(event: Dict[str, Any]) -> Optional[str]:
    """Extract processable text from a topology delta event.

    Topology deltas from NeuroGraph contain fired nodes, synapse changes,
    prediction results, etc. Convert to a text representation that the
    sensory pipeline can process. The substrate receives this as raw
    experience (Law 7).
    """
    parts = []

    fired = event.get('fired_node_ids', [])
    if fired:
        parts.append(f"fired:{len(fired)} nodes")

    fired_he = event.get('fired_hyperedge_ids', [])
    if fired_he:
        parts.append(f"hyperedges:{len(fired_he)} active")

    pruned = event.get('synapses_pruned', 0)
    sprouted = event.get('synapses_sprouted', 0)
    if pruned or sprouted:
        parts.append(f"structural:+{sprouted}/-{pruned}")

    confirmed = event.get('predictions_confirmed', 0)
    surprised = event.get('predictions_surprised', 0)
    if confirmed or surprised:
        parts.append(f"predictions:confirmed={confirmed},surprised={surprised}")

    if not parts:
        return None

    return f"[topology_delta] {' | '.join(parts)}"


def _get_peer_ids(bridge) -> list:
    """Get list of peer IDs from the tract bridge."""
    if hasattr(bridge, 'list_peers'):
        return bridge.list_peers()
    if hasattr(bridge, '_tracts'):
        return list(bridge._tracts.keys())
    return []


# ── FastAPI App ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    global _engine, _drain_thread, _drain_running

    logger.info("Elmer standalone service starting on port %d", SERVICE_PORT)

    # Start engine with full brains (not skip_brains)
    config = load_config()
    _engine = ElmerEngine(config=config)

    # Start engine — lightweight sockets first
    try:
        _engine.start(skip_brains=True)
        logger.info("Engine started (lightweight sockets)")
    except Exception as exc:
        logger.error("Engine start failed: %s", exc)

    # Load brains after a short delay for GC
    def _load_brains():
        time.sleep(BRAIN_LOAD_DELAY)
        gc.collect()
        try:
            _engine.load_brains()
            logger.info("Brains loaded — full capability active")
        except Exception as exc:
            logger.warning("Brain load failed: %s", exc)

    brain_thread = threading.Thread(
        target=_load_brains,
        name="elmer-brain-loader",
        daemon=True,
    )
    brain_thread.start()

    # Start tract drain loop
    _drain_running = True
    _drain_thread = threading.Thread(
        target=_drain_loop,
        name="elmer-tract-drain",
        daemon=True,
    )
    _drain_thread.start()

    yield

    # Shutdown
    logger.info("Elmer standalone service shutting down")
    _drain_running = False
    if _drain_thread:
        _drain_thread.join(timeout=5)
    if _engine:
        _engine.stop()
    logger.info("Elmer standalone service stopped")


app = FastAPI(
    title="Elmer — Cognitive Substrate Service",
    version="0.2.0",
    lifespan=lifespan,
)


# ── API Models ────────────────────────────────────────────────────

class ProcessRequest(BaseModel):
    text: str


class ProcessResponse(BaseModel):
    status: str
    process_id: int = 0
    autonomic_state: str = "PARASYMPATHETIC"
    result: Dict[str, Any] = {}


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    if not _engine or not _engine._started:
        return {"status": "starting", "brains": "loading"}
    h = _engine.health()
    return {
        "status": "healthy",
        "version": h.get("version", "unknown"),
        "uptime": h.get("uptime", 0),
        "process_count": h.get("process_count", 0),
        "brains": h.get("brain", {}).get("active_brain", "none") if h.get("brain") else "none",
        "kiss": h.get("kiss", {}),
        "ecosystem_tier": h.get("ecosystem", {}).get("tier", 0) if h.get("ecosystem") else 0,
    }


@app.get("/status")
async def status():
    """Full status report."""
    if not _engine:
        return {"status": "not_started"}
    return _engine.health()


@app.post("/process", response_model=ProcessResponse)
async def process(req: ProcessRequest):
    """Process text through the full Elmer substrate.

    This is the manual input endpoint. In normal operation, Elmer
    reads from its tract (topology deltas from the River). This
    endpoint allows direct input for testing and debugging.
    """
    if not _engine or not _engine._started:
        return ProcessResponse(status="not_started")

    try:
        result = _engine.process_text(req.text)
        return ProcessResponse(
            status="ok",
            process_id=result.get("process_id", 0),
            autonomic_state=result.get("autonomic_state", "PARASYMPATHETIC"),
            result=result,
        )
    except Exception as exc:
        return ProcessResponse(status="error", result={"error": str(exc)})


@app.get("/kiss")
async def kiss_stats():
    """KISS filter statistics."""
    if not _engine:
        return {"status": "not_started"}
    return _engine._kiss.stats.to_dict()


@app.get("/brains")
async def brain_status():
    """Brain socket status including Lenia state."""
    if not _engine:
        return {"status": "not_started"}

    result = {"brains_loaded": _engine._brain_switcher is not None}

    if _engine._brain_switcher:
        result["switcher"] = _engine._brain_switcher.status()

        # Get Lenia summary from proto if available
        proto = _engine._socket_manager.get_socket("elmer:proto_unibrain")
        if proto and hasattr(proto, 'get_lenia_summary'):
            result["lenia"] = proto.get_lenia_summary()

    return result


# ── Entry Point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "elmer_service:app",
        host="127.0.0.1",
        port=SERVICE_PORT,
        log_level="info",
    )
