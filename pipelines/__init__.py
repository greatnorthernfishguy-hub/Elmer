"""Elmer pipelines — domain-specific signal processing chains."""

from pipelines.sensory import SensoryPipeline
from pipelines.inference import InferencePipeline
from pipelines.health import HealthPipeline
from pipelines.memory import MemoryPipeline
from pipelines.identity import IdentityPipeline

__all__ = [
    "SensoryPipeline",
    "InferencePipeline",
    "HealthPipeline",
    "MemoryPipeline",
    "IdentityPipeline",
]
