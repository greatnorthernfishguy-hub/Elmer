# Elmer — Cognitive Substrate Module

Elmer is an E-T Systems module that processes SubstrateSignals through
a socket-based processing architecture integrated with the NG learning
ecosystem (NG-Lite → Peer → Full SNN).

## Quick Start

```python
from elmer_hook import ElmerHook

hook = ElmerHook()
hook.start()

# Process a message through the substrate
result = hook.on_message("What patterns have you learned?")
print(result)

# Get cross-module context
context = hook.get_context("Tell me about X")
print(context)

# Health check
print(hook.health())
```

## Architecture

- **Sockets**: Pluggable processing units (ComprehensionSocket, MonitoringSocket)
- **Pipelines**: Domain-specific chains (sensory, inference, health, memory, identity)
- **Runtime**: ElmerEngine orchestrates sockets + pipelines + NG ecosystem
- **Substrate Bus**: SubstrateSignal dataclass flows between all components

## NG Ecosystem Integration

Elmer participates in the E-T Systems three-tier learning architecture:
- **Tier 1**: Standalone local Hebbian learning via NG-Lite
- **Tier 2**: Peer-pooled cross-module learning via NGPeerBridge
- **Tier 3**: Full SNN via NeuroGraph Foundation (auto-upgrade)

## Status

Phase 1 (Foundation v0.1) — Module registers, connects to ecosystem,
reports health, produces and consumes SubstrateSignals.
