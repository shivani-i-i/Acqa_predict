"""Vercel entrypoint for the AquaPredict FastAPI service."""
from aquapredict.app.api import app

__all__ = ["app"]
