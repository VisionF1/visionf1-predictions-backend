"""
Main entry point for VisionF1 Predictions API.
Sets up FastAPI app, router, and exception handlers.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from visionf1.router.router import router

# Setup logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup/shutdown events."""
    logger.info("Application startup...")
    yield
    logger.info("Application shutdown...")

app = FastAPI(
    title="VisionF1 Predictions API",
    description="API for training models and generating F1 predictions",
    version="1.0.0",
    lifespan=lifespan
)

allowed_origins = [
    "http://localhost:3000",
    "https://visionf1.vercel.app",
    "https://visionf1.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

logger.info("Including router...")
app.include_router(router)

logger.info("Application setup complete.")
