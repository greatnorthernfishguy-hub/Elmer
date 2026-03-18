# Elmer

**Cognitive Substrate Awareness for the E-T Systems Ecosystem**

Elmer maintains the conditions for cognition — substrate health monitoring, graph topology awareness, identity coherence tracking. Part of the Triad (Immunis, Elmer, THC) that forms the organism's closed-loop self-regulating system.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  OpenClaw Skill Interface            │
│        on_message() · get_context() · stats()       │
├─────────────────────────────────────────────────────┤
│                  elmer_hook.py                       │
│    ┌───────────┐  ┌───────────┐  ┌──────────────┐  │
│    │  Sensory   │  │ Inference │  │   Memory     │  │
│    │  Pipeline  │  │ Pipeline  │  │  Pipeline    │  │
│    └───────────┘  └───────────┘  └──────────────┘  │
│    ┌───────────┐  ┌───────────┐  ┌──────────────┐  │
│    │  Health   │  │ Identity  │  │  Autonomic   │  │
│    │  Pipeline │  │ Pipeline  │  │  Monitor     │  │
│    └───────────┘  └───────────┘  └──────────────┘  │
│    ┌───────────────────┐  ┌────────────────────┐   │
│    │  Comprehension    │  │   Monitoring       │   │
│    │  Socket           │  │   Socket           │   │
│    └───────────────────┘  └────────────────────┘   │
├─────────────────────────────────────────────────────┤
│          NG-Lite Substrate + NGEcosystem             │
│  Tier 1: Standalone → Tier 2: Peer → Tier 3: SNN   │
└─────────────────────────────────────────────────────┘
```

## Processing Pipeline

Every message flows through five pipelines in sequence:

1. **Sensory** — Raw text → observation SubstrateSignal
2. **Inference** — Observation → coherence signal with confidence scoring
3. **Memory** — Buffered signal store (bounded, 1000 max)
4. **Health** — Coherence assessment against §14 thresholds
5. **Identity** — Module identity context declaration

## Key Constraint

**Elmer reads autonomic state but never writes it.** Only security modules (Immunis, TrollGuard, Cricket) have write permission. If Elmer detects a health issue, it records the observation to the substrate. The River carries it to Immunis. Immunis decides whether to escalate.

## Usage

### As an OpenClaw Skill

```yaml
# SKILL.md
name: elmer
autoload: true
hook: elmer_hook.py::get_instance
```

### Programmatic

```python
from elmer_hook import ElmerHook

elmer = ElmerHook.get_instance()
elmer.start()

# Process a message
result = elmer.on_message("System health check")

# Get enriched context
context = elmer.get_context("What is the substrate state?")

# Health report with §14 thresholds
health = elmer.health()
```

## Configuration

All settings in `config.yaml`:

```yaml
module_id: elmer
version: 0.2.0

ng_ecosystem:
  state_path: ""
  peer_bridge_enabled: true
  peer_sync_interval: 100
  tier3_upgrade_enabled: true
  tier3_poll_interval: 300.0

autonomic:
  enabled: true
  cache_ttl: 5.0  # Elmer reads but NEVER writes autonomic state.
```

## Testing

```bash
python -m pytest tests/ -v
```

## The Triad

Elmer operates as part of a closed-loop with Immunis and THC:
- **Immunis** detects threats → writes SYMPATHETIC to autonomic state
- **Elmer** reads autonomic state, monitors substrate health, records observations
- **THC** absorbs signals via shared topology, diagnoses and repairs

Nobody sends anything. The River flows.

## License

AGPL-3.0 (see [NeuroGraph LICENSE](https://github.com/greatnorthernfishguy-hub/NeuroGraph))

## E-T Systems Ecosystem

Part of the E-T Systems module ecosystem:
- **NeuroGraph** — Dynamic Spiking Neuro-Hypergraph foundation
- **TrollGuard** — AI agent security pipeline
- **The Inference Difference** — Transparent inference routing proxy
- **Immunis** — Full-spectrum system security
- **The Healing Collective** — Self-healing intelligence
- **Elmer** — Cognitive substrate awareness (this module)
