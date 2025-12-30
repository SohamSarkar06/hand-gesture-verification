from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import router

app = FastAPI(
    title="Hand Gesture Verification API",
    version="1.0.0",
    description="Real-time hand gesture based human verification system"
)

# CORS (frontend-safe)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Health check (VERY IMPORTANT)
@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Hand Gesture Verification API"
    }

# API routes
app.include_router(router, prefix="/api", tags=["Gesture Verification"])
