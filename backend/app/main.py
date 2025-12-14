from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import DATA_DIR, FRONTEND_DIR
from .rag import load_resources
from .routes import register_routes

load_dotenv()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    load_resources(app)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="NepEd Bot API",
        description="Your guide to bachelor study pathways",
        version="0.3.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_routes(app)

    if FRONTEND_DIR.exists():
        logger.info("Serving static frontend from %s", FRONTEND_DIR)

    return app


app = create_app()
