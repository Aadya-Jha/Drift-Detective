from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="Drift Detective",
    description="ML Model Observability and Drift Detection Platform",
    version="0.1.0"
)

app.include_router(router, prefix="/api")

@app.get("/")
def root():
    return {"message": "Drift Detective is running"}