from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from api.schemas import HealthRequest, HealthResponse
from services.agent_service import agent
from utils.security import verify_token

router = APIRouter()

# We add 'dependencies=[Depends(verify_token)]' to protect the routes
@router.post("/predict-health", 
             response_model=HealthResponse, 
             tags=["Health Module"], 
             dependencies=[Depends(verify_token)])
async def predict_health(request: HealthRequest):
    """
    Predicts if health readings are Normal or Abnormal based on ML model.
    Triggers an Agentic alert and logs the event if Abnormal (Encrypted).
    """
    try:
        response_data = agent.process_health_data(request.heart_rate, request.oxygen_level)
        return HealthResponse(**response_data)
    except Exception as e:
        # Standard error response
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect-objects", 
             tags=["Smart City Module"], 
             dependencies=[Depends(verify_token)])
async def detect_objects(file: UploadFile = File(...)):
    """
    Detect objects in a picture to simulate Smart City Accident/Crime monitoring.
    Requires an image file upload. Triggers Agentic alert if hazards are found (Encrypted).
    """
    # Simple validation to ensure they upload an image
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
        
    try:
        # Read the raw byte content of the uploaded file
        image_bytes = await file.read()
        
        # Pass bytes to our Agent Service
        response_data = agent.process_vision_data(image_bytes, file.filename)
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")
