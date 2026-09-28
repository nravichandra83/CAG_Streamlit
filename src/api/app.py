"""FastAPI application factory.

The lifespan hook warms up the singleton CAGService at startup (so the first
request doesn't pay for document parsing + cache creation) and releases the
provider cache on shutdown.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .controllers import chat_controller
from .dependencies import get_cag_service

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Resolve through app.dependency_overrides so tests can inject a fake service.
    service = app.dependency_overrides.get(get_cag_service, get_cag_service)()
    service.start()
    try:
        yield
    finally:
        service.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="CAG HR Policy API",
        description="Cache-Augmented Generation Q&A over the HR leave policy document.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(chat_controller.router)
    return app


app = create_app()
