"""Tiny provider registry so config can name providers as strings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_REGISTRY: dict[str, Callable[..., Any]] = {}


def register(name: str) -> Callable[[type], type]:
    """Class decorator: make a provider constructible by name."""

    def deco(cls: type) -> type:
        key = name.lower()
        if key in _REGISTRY and _REGISTRY[key] is not cls:
            raise ValueError(f"provider {name!r} is already registered")
        _REGISTRY[key] = cls
        return cls

    return deco


def get_provider(name: str, **kwargs: Any) -> Any:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown provider {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)


def list_providers() -> list[str]:
    return sorted(_REGISTRY)
