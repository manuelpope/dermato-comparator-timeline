"""Aggregate v1 router — mounts trichology + dermatology + aesthetics."""

from fastapi import APIRouter

from app.api.v1 import aesthetics, dermatology, trichology

router = APIRouter()
router.include_router(trichology.router)
router.include_router(dermatology.router)
router.include_router(aesthetics.router)
