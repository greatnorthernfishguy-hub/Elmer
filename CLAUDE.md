# Elmer Repository
## Claude Code Onboarding — Repo-Specific

**You have already read the global `CLAUDE.md` and `ARCHITECTURE.md`.**
**If you have not, stop. Go read them. The Laws defined there govern this repo.**
**This document adds Elmer-specific rules on top of those Laws.**

---
## Vault Context
For full ecosystem context, read these from the Obsidian vault (`~/docs/`):
- **Module page:** `~/docs/modules/Elmer.md`
- **Concepts:** `~/docs/concepts/The Triad.md`, `~/docs/concepts/Autonomic State.md`, `~/docs/concepts/SubstrateSignal.md`, `~/docs/concepts/Competence Model.md`, `~/docs/concepts/The River.md`, `~/docs/concepts/Vendored Files.md`
- **Systems:** `~/docs/systems/NG-Lite.md`, `~/docs/systems/NG Peer Bridge.md`, `~/docs/systems/NG Tract Bridge.md`
- **Audits:** `~/docs/audits/ecosystem-test-suite-audit-2026-03-23.md`, `~/docs/audits/ecosystem-static-value-audit-2026-03-23.md`

Each vault page has a Context Map at the top linking to related docs. Follow those links for ripple effects and dependencies.

---


## What This Repo Is

Elmer is the brainstem and cerebellum of the E-T Systems digital organism. It maintains the conditions for cognition — substrate health monitoring, topology awareness, identity coherence tracking. Elmer does not think. Elmer ensures that thinking can happen.

Elmer is part of the **Triad** (Immunis, Elmer, THC). The Triad forms a closed-loop self-regulating system:
- **Immunis** detects host-level threats
- **Elmer** maintains substrate-level cognitive conditions
- **THC** diagnoses and repairs

They do not coordinate directly. The River flows. The topology reshapes itself.

**Status: Integrated (Tier 2, peer bridge).** v0.2.0. Vendored files synced to NeuroGraph canonical (2026-03-19, includes Cricket rim). Registered in `~/.et_modules/`. Not yet running as a persistent service on the VPS.

---

## 1. Repository Structure

```
~/Elmer/
├── elmer_hook.py                  # OpenClaw skill entry point (ElmerHook singleton)
├── et_module.json                 # Module manifest (v2 schema)
├── config.yaml                    # All configuration
├── SKILL.md                       # OpenClaw skill discovery
├── CHANGELOG.md                   # Version history
├── core/                          # Core domain logic
│   ├── substrate_signal.py        # SubstrateSignal dataclass + COHERENCE_* thresholds
│   ├── config.py                  # ElmerConfig dataclass + loader
│   ├── base_socket.py             # Socket ABC for processing units
│   ├── comprehension.py           # ComprehensionSocket — graph topology analysis
│   ├── monitoring.py              # MonitoringSocket — health dashboard
│   ├── myelination.py             # MyelinationSocket — oligodendrocyte (v0.4, tract myelination decisions)
│   └── socket_manager.py          # Socket lifecycle management
├── pipelines/                     # Five-stage processing pipeline
│   ├── sensory.py                 # Raw text → observation SubstrateSignal
│   ├── inference.py               # Observation → coherence signal
│   ├── health.py                  # Coherence → health assessment (§14 thresholds)
│   ├── memory.py                  # Signal buffer (bounded, 1000 max)
│   └── identity.py                # Module identity declaration
├── runtime/                       # Engine and signal encoding
│   ├── engine.py                  # ElmerEngine — orchestrates sockets + pipelines
│   ├── graph_encoder.py           # SubstrateSignal → graph-compatible encoding
│   └── signal_decoder.py          # Graph output → response dict
├── data/
│   └── constitutional_embeddings.json  # Cricket rim — 22 embeddings, 4 categories (read-only after init)
├── surgery/                       # Neural socket training (not core runtime)
│   ├── train.py, train_on_syl.py, extract_features.py, graph_io.py
│   └── elmer_brain_v0.1.pt       # Optional neural socket backbone
├── ng_lite.py                     # VENDORED — canonical from NeuroGraph
├── ng_peer_bridge.py              # VENDORED — canonical from NeuroGraph (legacy, retained until v1.0)
├── ng_tract_bridge.py             # VENDORED — canonical from NeuroGraph (v0.3+, preferred)
├── ng_ecosystem.py                # VENDORED — canonical from NeuroGraph
├── ng_autonomic.py                # VENDORED — canonical from NeuroGraph
├── openclaw_adapter.py            # VENDORED — canonical from NeuroGraph
├── ng_updater.py                  # VENDORED — auto-update + vendored file sync (runs on startup)
├── et_modules/                    # ET Module Manager integration
│   ├── __init__.py
│   └── manager.py
└── tests/                         # Test suite (70 tests)
    ├── test_config.py
    ├── test_hook.py
    ├── test_pipelines.py
    ├── test_signal_format.py
    ├── test_socket_manager.py
    └── test_sockets.py
```

---

## 2. Key Architectural Constraint: Elmer Reads, Never Writes Autonomic State

This is the single most important rule specific to Elmer.

Elmer **reads** the autonomic state via `AutonomicMonitor.read()`. Elmer **never writes** to `ng_autonomic.py`. Only Immunis, TrollGuard, and Cricket have write permission.

This boundary is enforced in:
- `config.yaml` line 44: `# Elmer reads but NEVER writes autonomic state.`
- `et_module.json`: `"autonomic_writer": false`
- `elmer_hook.py` line 156: `autonomic = self._autonomic.read()` — read only

If you are tempted to have Elmer escalate a health issue by writing SYMPATHETIC: **stop**. Elmer records the observation to the substrate. The River carries it to Immunis. Immunis decides whether to escalate. That is the correct flow.

### Exception: Cricket Rim Violations

Cricket (the constitutional constraint layer) is integrated into Elmer as its extraction boundary — Cricket IS the bucket. When the extraction pipeline detects that an input landed on a constitutional node (frozen semantic region), Elmer writes SYMPATHETIC with `threat_level: "constitutional"` and `source: "cricket_rim"`.

This is **not** Elmer deciding to escalate a health issue. This is constitutional enforcement — a mechanically distinct code path in `runtime/engine.py`. The Cricket rim write is the sole exception to Elmer's read-only autonomic rule. See `docs/prd/Cricket_Design_v0.1.md` for full design context.

---

## 3. SubstrateSignal — Elmer's Extraction Bucket

`SubstrateSignal` is defined in `core/substrate_signal.py` (Elmer-local, not vendored). It is Elmer's extraction vocabulary — the shape of Elmer's bucket when it dips into the River.

It is **not** an inter-module protocol. No module serializes a SubstrateSignal and sends it to another module. See ARCHITECTURE.md §6-7.

Previously this class lived inside the vendored `ng_ecosystem.py` (Law 2 violation). Extracted to `core/substrate_signal.py` on 2026-03-18.

### Threshold Constants (PRD §14)

```python
COHERENCE_HEALTHY  = 0.70  # System functioning well
COHERENCE_DEGRADED = 0.40  # Attention needed
COHERENCE_CRITICAL = 0.15  # Immediate concern
```

These are Elmer's extraction thresholds. They are **not** the ecosystem-wide confidence thresholds from `ng_ecosystem.py` (0.70/0.40/0.15 — same values, different purpose). Do not confuse them.

---

## 4. The Processing Pipeline

Every message flows through five pipelines in sequence:

```
Raw text
  → SensoryPipeline.process(text)     → observation SubstrateSignal
  → InferencePipeline.process(signal) → coherence SubstrateSignal
  → MemoryPipeline.store(signal)      → buffered
  → HealthPipeline.check()            → §14 threshold assessment
  → IdentityPipeline.query()          → module identity context
```

The pipeline feeds raw embeddings to NG-Lite via `NGEcosystem.record_outcome()`. Classification happens only when extracting context via `NGEcosystem.get_context()`. This is correct per Law 7.

---

## 5. Sockets

Elmer uses a socket architecture for processing units:

| Socket | Purpose | GPU Required |
|--------|---------|-------------|
| `ComprehensionSocket` | Graph topology analysis | No |
| `MonitoringSocket` | Health monitoring dashboard | No |
| `MyelinationSocket` | Oligodendrocyte — decides which tracts to myelinate | No |

Sockets are managed by `SocketManager` and registered in `runtime/engine.py:start()`. `MyelinationSocket` (v0.4, 2026-03-23) extracts pathway activity patterns from the substrate and produces myelination recommendations. Apprentice-tier heuristic counts peer events from the bridge cache. Engine applies recommendations to Elmer's own tract bridge via `myelinate_tract()`/`demyelinate_tract()`. Tracts stay dumb — myelination state is runtime-only.

---

## 6. Vendored Files

Seven vendored files synced to NeuroGraph canonical:

| File | Purpose | Last Synced |
|------|---------|-------------|
| `ng_lite.py` | Tier 1 learning substrate (includes Cricket rim) | 2026-03-19 |
| `ng_peer_bridge.py` | Tier 2 legacy fallback (JSONL-based) | 2026-03-18 |
| `ng_tract_bridge.py` | Tier 2 preferred (per-pair directional tracts, v0.3+) | 2026-03-19 |
| `ng_ecosystem.py` | Tier management lifecycle | 2026-03-18 |
| `ng_autonomic.py` | Autonomic state (**READ ONLY for Elmer**, except Cricket rim) | 2026-03-18 |
| `openclaw_adapter.py` | OpenClaw skill base class | 2026-02-22 |
| `ng_updater.py` | Auto-update + vendored file sync (runs on startup before imports) | 2026-03-19 |

**Do not modify vendored files.** If Elmer needs different behavior, that behavior lives in Elmer-specific code (`core/`, `pipelines/`, `runtime/`), not in vendored files.

---

## 7. What Elmer Does NOT Do

- Elmer **never** executes repairs — THC's domain
- Elmer **never** touches host-level threats — Immunis's domain
- Elmer **never** writes autonomic state — security modules only
- Elmer **never** calls other modules directly — Law 1

When Elmer detects something outside its domain, it records to the substrate and steps back. The River carries it to the appropriate module.

---

## 8. Graph Encoder Embedding

`runtime/graph_encoder.py` currently uses SHA256 hash-based deterministic embedding as a fallback. The comment acknowledges this: `"Production: replace with fastembed or Ollama."` This is not a bug — it's a bootstrap placeholder. The ecosystem standard embedding is `ng_embed.py` (`Snowflake/snowflake-arctic-embed-m-v1.5`, 768-dim, ONNX Runtime).

---

## 9. Historical Failure Modes — Learn From These

### SubstrateSignal Law 2 Violation (2026-03-18)
`SubstrateSignal` and `COHERENCE_*` thresholds were initially in vendored `ng_ecosystem.py`. This prevented syncing vendored files without Elmer-specific code leaking into the canonical source. **Fixed:** Extracted to `core/substrate_signal.py` (Elmer-local, not vendored). This is the correct pattern — module-specific behavior lives in module code, not vendored files.

### Autonomic Read/Write Confusion (2026-03-19)
Both `elmer_hook.py` and `runtime/engine.py` were separately trying to manage autonomic state. **Fixed:** Single integration point, both use `ng_autonomic.read_state()` (read-only). The Cricket rim write path is mechanically isolated in `runtime/engine.py`.

### Missing `_embed()` Override (2026-03-19)
`OpenClawAdapter` requires `_embed()` but it was missing from `ElmerHook`. **Fixed:** Now delegates to `ng_embed.py` (centralized ecosystem embedding) with hash fallback.

### Embedding Dimension Incident (ecosystem-wide, 2026-03-19)
A CC instance switched NeuroGraph from 768-dim to 384-dim without checking stored vectors, breaking Syl's query layer. **Resolved:** All modules now use `ng_embed.py` (vendored, `Snowflake/snowflake-arctic-embed-m-v1.5`, 768-dim). Embedding model is centralized — no per-module model references. Do not change the model without verifying dimension compatibility with stored vectors.

---

## 10. What Claude Code May and May Not Do

### Without Josh's Approval

**Permitted:**
- Read any file in the repo
- Run the test suite (`tests/`)
- Edit Elmer-specific files (core/, pipelines/, runtime/, elmer_hook.py)
- Add or modify tests
- Update documentation

**Not permitted without explicit Josh approval:**
- Modify any vendored file (including ng_updater.py)
- Delete any file
- Add autonomic write capability (except Cricket rim, which already exists)
- Restart any service
- Change the pipeline processing order
- Modify `data/constitutional_embeddings.json`
- Change the embedding model or dimension

---

## 11. Environment and Paths

| What | Where |
|------|-------|
| Repo root | `~/Elmer/` |
| Configuration | `~/Elmer/config.yaml` |
| Module manifest | `~/Elmer/et_module.json` |
| Module data (runtime) | `~/.et_modules/elmer/` |
| Shared learning JSONL | `~/.et_modules/shared_learning/elmer.jsonl` |
| Peer registry | `~/.et_modules/shared_learning/_peer_registry.json` |
| Tract files (Tier 2) | `~/.et_modules/tracts/elmer/` |

---

*E-T Systems / Elmer*
*Last updated: 2026-03-21*
*Maintained by Josh — do not edit without authorization*
*Parent documents: `~/.claude/CLAUDE.md` (global), `~/.claude/ARCHITECTURE.md`*
