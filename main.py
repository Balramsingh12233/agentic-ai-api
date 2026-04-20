import os
import sys

# We add the backend directory to python path so it can import our modules correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router as api_router
from services.health_service import health_service
from services.vision_service import vision_service

# Initialize FastAPI application
app = FastAPI(
    title="Agentic AI Backend",
    description="API for Health Predictions and Smart City Object Detection",
    version="1.0"
)

# Standard CORS Middleware so the Flutter frontend can talk to it without issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Since it's a project, allow everything initially
    allow_credentials=False, # Set to False to allow "*" origins in modern browsers
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include our routes
app.include_router(api_router)

# FastAPI Built-in Lifespan (Startup / Shutdown events)
@app.on_event("startup")
async def startup_event():
    print("Agentic AI Server starting up...")
    
    # 1. Load ML models instantly when server boots (avoids delay on first request)
    try:
        health_service.load_model()
        vision_service.load_model()
    except Exception as e:
        print(f"CRITICAL ERROR loading models on startup: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    print("Agentic AI Server shutting down...")

@app.get("/", tags=["Healthcheck"])
async def root():
    return {"message": "Agentic AI Backend is Running!"}
