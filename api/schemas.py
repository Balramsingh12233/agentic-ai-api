from pydantic import BaseModel, Field

class HealthRequest(BaseModel):
    heart_rate: int = Field(..., description="Heart rate in BPM", ge=0, le=300)
    oxygen_level: int = Field(..., description="SpO2 percentage", ge=0, le=150)

class HealthResponse(BaseModel):
    status: str
    message: str
    alert_triggered: bool
    risk_level: str = "Stable"  # Stable, Caution, Critical
    insights: list[str] = []
    treatments: list[str] = []
    recommendation: str = "Continuously monitoring vitals."


