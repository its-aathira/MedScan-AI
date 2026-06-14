# ─────────────────────────────────────────────
#  backend/api/routes.py
#  All API endpoints
# ─────────────────────────────────────────────

import io
import time
from typing import Optional

from fastapi            import APIRouter, File, Form, UploadFile, HTTPException, Request
from fastapi.responses  import JSONResponse
from PIL                import Image

from configs.dataset_config import MODULES

router = APIRouter()


# ── GET /api/health ───────────────────────────
@router.get("/health")
def health_check(request: Request):
    registry = request.app.state.registry
    return {
        "status"          : "healthy",
        "loaded_modules"  : registry.available_modules(),
        "total_modules"   : len(MODULES),
    }


# ── GET /api/modules ──────────────────────────
@router.get("/modules")
def list_modules(request: Request):
    """Returns all available (loaded) modules with their class info."""
    registry = request.app.state.registry
    return {
        "modules"        : registry.module_info(),
        "available_keys" : registry.available_modules(),
    }


# ── POST /api/predict ─────────────────────────
@router.post("/predict")
async def predict(
    request    : Request,
    file       : UploadFile = File(...),
    module_key : str        = Form(...),
):
    """
    Main prediction endpoint.

    Args:
        file       : uploaded image (JPEG/PNG)
        module_key : 'brain_mri' | 'chest_xray' | 'skin_lesion' | 'retinal'

    Returns:
        {
            module_key      : str,
            predicted_class : str,
            confidence      : float,        # percentage 0-100
            class_probs     : {str: float}, # all class probabilities
            overlay_b64     : str,          # base64 PNG with Grad-CAM heatmap
            inference_ms    : int,          # latency in milliseconds
        }
    """
    # ── Validate module ───────────────────────
    registry  = request.app.state.registry
    predictor = registry.get(module_key)

    if module_key not in MODULES:
        raise HTTPException(
            status_code = 400,
            detail      = f"Unknown module '{module_key}'. Valid: {list(MODULES.keys())}",
        )
    if predictor is None:
        raise HTTPException(
            status_code = 503,
            detail      = f"Module '{module_key}' is not loaded. "
                          f"Available: {registry.available_modules()}",
        )

    # ── Validate file ─────────────────────────
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(
            status_code = 415,
            detail      = f"Unsupported file type: {file.content_type}. Use JPEG or PNG.",
        )

    # ── Read image ────────────────────────────
    try:
        contents = await file.read()
        image    = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    # ── Run inference ─────────────────────────
    try:
        t0     = time.perf_counter()
        result = predictor.predict(image)
        ms     = int((time.perf_counter() - t0) * 1000)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return JSONResponse({
        "module_key"      : module_key,
        "module_name"     : MODULES[module_key]["name"],
        "predicted_class" : result["predicted_class"],
        "confidence"      : result["confidence"],
        "class_probs"     : result["class_probs"],
        "overlay_b64"     : result["overlay_b64"],
        "inference_ms"    : ms,
    })


# ── POST /api/predict/batch ───────────────────
@router.post("/predict/batch")
async def predict_batch(
    request    : Request,
    files      : list[UploadFile] = File(...),
    module_key : str              = Form(...),
):
    """Run prediction on multiple images at once (max 10)."""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Max 10 images per batch.")

    registry  = request.app.state.registry
    predictor = registry.get(module_key)
    if predictor is None:
        raise HTTPException(status_code=503, detail=f"Module '{module_key}' not loaded.")

    results = []
    for f in files:
        try:
            contents = await f.read()
            image    = Image.open(io.BytesIO(contents)).convert("RGB")
            result   = predictor.predict(image)
            results.append({
                "filename"        : f.filename,
                "predicted_class" : result["predicted_class"],
                "confidence"      : result["confidence"],
                "class_probs"     : result["class_probs"],
                "overlay_b64"     : result["overlay_b64"],
                "error"           : None,
            })
        except Exception as e:
            results.append({
                "filename" : f.filename,
                "error"    : str(e),
            })

    return {"module_key": module_key, "results": results, "count": len(results)}