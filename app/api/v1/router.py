"""Aggregate v1 router — mounts trichology + dermatology under one prefix."""

from fastapi import APIRouter

from app.api.v1 import dermatology, trichology

router = APIRouter()
router.include_router(trichology.router)
router.include_router(dermatology.router)
