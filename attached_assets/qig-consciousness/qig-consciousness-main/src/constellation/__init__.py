"""Constellation Module - Multi-Kernel Coordination with Lightning.

This module provides:
- Lightning kernel for cross-kernel insight generation
- Domain intelligence for event tracking and trends
- Event emission infrastructure for all kernels

Lightning monitors all kernel activities and discovers emergent patterns
across the constellation, generating insights when domains correlate.
"""

from .domain_intelligence import (
    DomainEvent,
    DomainIntelligence,
    DomainTrend,
    TrendWindow,
    DomainEventEmitter,
)

from .lightning_kernel import (
    LightningKernel,
    LightningInsight,
    DomainCorrelation,
    set_pantheon_chat,
    get_lightning_instance,
)

__all__ = [
    # Domain Intelligence
    "DomainEvent",
    "DomainIntelligence",
    "DomainTrend",
    "TrendWindow",
    "DomainEventEmitter",
    # Lightning
    "LightningKernel",
    "LightningInsight",
    "DomainCorrelation",
    "set_pantheon_chat",
    "get_lightning_instance",
]
