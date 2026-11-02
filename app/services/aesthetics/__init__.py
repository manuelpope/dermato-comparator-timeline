"""Aesthetics (Botox before/after) service package.

Exposes :func:`compare`, the orchestrator behind
``POST /v1/aesthetics/compare``.
"""

from app.services.aesthetics.service import compare

__all__ = ["compare"]
