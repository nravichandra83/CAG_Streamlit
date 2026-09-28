"""Dependency providers for the API.

get_cag_service() is a process-wide singleton: lru_cache(maxsize=1) on a
zero-argument function means the first call builds the CAGService and every
later call (every request, via Depends) gets that same instance back. That's
what we want -- the document is parsed and cached once, not per request.

In tests, swap it out with app.dependency_overrides[get_cag_service].
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from ..core.config import Settings, load_settings
from ..services.cag_service import CAGService


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


@lru_cache(maxsize=1)
def get_cag_service() -> CAGService:
    return CAGService(get_settings())


CAGServiceDep = Annotated[CAGService, Depends(get_cag_service)]
