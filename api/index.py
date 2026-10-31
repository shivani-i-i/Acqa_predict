"""Small Vercel entrypoint for the public AquaPredict API.

The complete API remains available locally with ``aquapredict.app.api``.
Vercel functions must stay below the provider's 500 MB bundle limit, so the
serverless deployment exposes the lightweight, provenance-preserving health
and USGS context endpoints without importing the local ML stack.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from aquapredict.app.online_sources import OnlineSourceError, fetch_usgs_observations
from aquapredict import __version__


app = FastAPI(
    title="AquaPredict Public API",
    version=__version__,
    description=(
        "Public deployment for source-labeled AquaPredict status and USGS "
        "environmental context. Filter forecasts remain available in the "
        "full local API and Streamlit application."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "AquaPredict",
        "status": "ok",
        "docs": "/docs",
        "endpoints": ["/health", "/sources/usgs/observations"],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": __version__,
        "deployment": "vercel",
        "data": "source-labeled USGS environmental context",
    }


@app.get("/sources/usgs/observations")
def usgs_observations(site: str = "01646500"):
    try:
        return fetch_usgs_observations(site)
    except OnlineSourceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


__all__ = ["app"]
