"""Concrete providers. Importing this package registers them all by name."""

from . import edgar, local, stooq, synthetic, vendor, yahoo  # noqa: F401

__all__ = ["edgar", "local", "stooq", "synthetic", "vendor", "yahoo"]
