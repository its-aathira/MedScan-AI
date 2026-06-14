# ─────────────────────────────────────────────
#  backend/api/main.py
#  FastAPI app — serves API + React static files
# ─────────────────────────────────────────────

import os
from pathlib      import Path
from contextlib   import asynccontextmanager

from fastapi               import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles   import StaticFiles
from fastapi.responses     import FileResponse

from backend.api.routes         import router
from backend.api.model_registry import ModelRegistry


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🚀  Starting Medical AI Detector API...")
    registry = ModelRegistry()
    registry.load_available()
    app.state.registry = registry
    print("✅  API ready\n")
    yield
    print("👋  Shutting down...")


app = FastAPI(
    title       = "Medical Image Disease Detection API",
    description = "AI-powered disease detection across Chest X-Ray, Brain MRI, Skin Lesion, and Retinal scans",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# ── CORS ──────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── API routes ────────────────────────────────
app.include_router(router, prefix="/api")

@app.get("/api")
def api_root():
    return {"name": "Medical AI Detector", "version": "1.0.0", "status": "running"}


# ── Serve React frontend (production) ────────
FRONTEND_DIST = Path(__file__).parent.parent.parent / "frontend" / "dist"

if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

    @app.get("/")
    def serve_root():
        return FileResponse(str(FRONTEND_DIST / "index.html"))

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        file_path = FRONTEND_DIST / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(FRONTEND_DIST / "index.html"))
else:
    @app.get("/")
    def root():
        return {
            "name"   : "Medical AI Detector",
            "version": "1.0.0",
            "docs"   : "/docs",
            "status" : "running",
        }