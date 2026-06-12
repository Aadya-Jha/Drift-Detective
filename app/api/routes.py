from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.routing import APIRouter
import pandas as pd
import io
from app.core.scorer import score_all_features

router = APIRouter()

# in-memory baseline store
baseline_store = {}

@router.post("/upload/baseline")
async def upload_baseline(file: UploadFile = File(...)):
    """Upload training baseline CSV"""
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    baseline_store["data"] = df
    return {
        "message": "Baseline uploaded successfully",
        "rows": len(df),
        "features": list(df.select_dtypes(include=["float64", "int64"]).columns)
    }

@router.post("/upload/current")
async def upload_current(file: UploadFile = File(...)):
    """Upload current production data CSV and run drift detection"""
    if "data" not in baseline_store:
        raise HTTPException(status_code=400, detail="Upload baseline first")

    contents = await file.read()
    current_df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    baseline_df = baseline_store["data"]

    results = score_all_features(baseline_df, current_df)
    return results

@router.get("/health")
def health():
    return {"status": "ok", "baseline_loaded": "data" in baseline_store}