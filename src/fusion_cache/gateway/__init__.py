"""FastAPI gateway for fusion-cache."""

from .app import create_app, app

__all__ = ["create_app", "app"]
