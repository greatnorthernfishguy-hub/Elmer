# Changelog

## [0.2.0] - 2026-02-28

### Added
- SubstrateSignal §6.1 schema (flat scored fields, Elmer-specific extensions)
- §14 threshold constants (COHERENCE_HEALTHY=0.70, DEGRADED=0.40, CRITICAL=0.15)
- ng_autonomic.py vendored file — read-only autonomic state monitor (§7)
- ElmerSocket ABC with §5.2.1 interface (declare_requirements, load, unload, process, health)
- GraphSnapshot, SocketOutput, HardwareRequirements, SocketHealth data structures (§5.2.2)
- Autonomic-aware processing in ElmerEngine and ElmerHook (§7)
- et_module.json Appendix B schema (ecosystem, sockets sections)
- models/ directory for future model storage

### Changed
- SubstrateSignal: generic payload dict → flat scored fields per §6.1
- ElmerSocket: connect/disconnect/process(signal) → load/unload/process(GraphSnapshot, context)
- SocketManager: route_signal → route(GraphSnapshot, context)
- All pipelines: use §6.1 signal_type strings instead of SignalType enum
- et_module.json: version 0.1.0 → 0.2.0, added ecosystem/sockets sections

## [0.1.0] - 2026-02-28

### Added
- Initial Phase 1 foundation
- SubstrateSignal dataclass (Grok priority #1)
- ElmerSocket ABC with connect/disconnect/process/health_check
- SocketManager with hardware detection
- ComprehensionSocket and MonitoringSocket
- Five pipeline stubs (sensory, inference, health, memory, identity)
- ElmerEngine runtime orchestrator
- ElmerHook (OpenClawAdapter subclass)
- Five vendored files from NeuroGraph
- YAML config with dataclass loader
- 70 passing tests
