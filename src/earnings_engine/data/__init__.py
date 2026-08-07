"""Data acquisition: provider protocols, concrete providers, universe handling."""

from .base import (
    EventProvider,
    FilingProvider,
    FundamentalProvider,
    PriceProvider,
    ProviderError,
)
from .registry import get_provider, list_providers, register

__all__ = [
    "EventProvider",
    "FilingProvider",
    "FundamentalProvider",
    "PriceProvider",
    "ProviderError",
    "get_provider",
    "list_providers",
    "register",
]
