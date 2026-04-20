import base64
from services.health_service import health_service
from services.vision_service import vision_service
from utils.logger import logger

class AgentService:
    """
    This is the "Brain" of the Backend.
    It follows the Agentic AI Pipeline: Detect -> Analyze -> Decide -> Act
    """
    
    def process_health_data(self, heart_rate: int, oxygen_level: int) -> dict:
        # 1. INITIAL ANALYSIS & DATA VALIDATION
        ml_status = health_service.predict(heart_rate, oxygen_level)
        insights = []
        treatments = []
        risk_level = "Stable"
        alert_triggered = False
        
        # --- Heart Rate Analysis ---
        if heart_rate > 150:
            insights.append(f"EXTREME RISK: Very High Heart Rate detected ({heart_rate} BPM).")
            insights.append("Hindi: Heart rate bahot jyada hai. (Severe Tachycardia)")
            treatments.extend(["Sit down and rest immediately", "Try slow, deep breathing", "Call for emergency medical assistance (102/108)"])
            risk_level = "Critical"
            alert_triggered = True
        elif heart_rate > 100:
            insights.append(f"Caution: High Heart Rate detected ({heart_rate} BPM).")
            insights.append("Hindi: Dhadkan tez hai. (Tachycardia)")
            treatments.extend(["Drink water and rest", "Avoid caffeine or stimulants", "Monitor for chest pain"])
            if risk_level != "Critical": risk_level = "Caution"
        elif heart_rate < 40:
            insights.append(f"EXTREME RISK: Very Low Heart Rate detected ({heart_rate} BPM).")
            insights.append("Hindi: Dhadkan bahot kam hai. (Severe Bradycardia)")
            treatments.extend(["Lay down on a flat surface", "Do not stand up suddenly", "Seek immediate emergency help"])
            risk_level = "Critical"
            alert_triggered = True
        elif heart_rate < 55:
            insights.append(f"Caution: Low Heart Rate detected ({heart_rate} BPM).")
            insights.append("Hindi: Dhadkan thodi kam hai. (Bradycardia)")
            treatments.extend(["Rest and stay warm", "Notify your doctor if you feel dizzy"])
            if risk_level != "Critical": risk_level = "Caution"

        # --- Oxygen (SpO2) Analysis ---
        if oxygen_level > 100:
            insights.append(f"INVALID DATA: Oxygen level cannot be {oxygen_level}%. Check Sensor placement.")
            insights.append("Hindi: Sensor sahi se check karein (Invalid Data).")
            treatments.append("Re-attach the SpO2 sensor and try again")
            risk_level = "Caution"
        elif oxygen_level < 85:
            insights.append(f"EXTREME RISK: Oxygen level is Very Low ({oxygen_level}%).")
            insights.append("Hindi: Oxygen bahot kam hai. (Severe Hypoxia)")
            treatments.extend(["Use supplemental oxygen immediately if available", "Sit in an upright 'tripod' position", "Go to Emergency ER immediately"])
            risk_level = "Critical"
            alert_triggered = True
        elif oxygen_level < 93:
            insights.append(f"Warning: Oxygen level is Low ({oxygen_level}%).")
            insights.append("Hindi: Oxygen thoda kam hai. (Mild Hypoxia)")
            treatments.extend(["Deep breathing exercises (Pranayama)", "Increase room ventilation (Fresh Air)"])
            if risk_level != "Critical": risk_level = "Caution"

        # --- AI ML Model Overrides ---
        if ml_status == "Abnormal" and risk_level == "Stable":
            insights.append("AI Pattern Sync: ML Model detected subtle abnormalities in vital rhythm.")
            risk_level = "Caution"
            alert_triggered = True
        
        # --- Final Recommendation Summary ---
        rec = "Patient is Stable. No immediate action required."
        if risk_level == "Caution":
            rec = "CONDITION UNSTABLE: Patient requires continuous monitoring and rest."
        elif risk_level == "Critical":
            rec = "CRITICAL EMERGENCY: Life-threatening vitals detected. Act now!"

        response = {
            "status": ml_status,
            "alert_triggered": alert_triggered,
            "risk_level": risk_level,
            "insights": insights if insights else ["All vitals appear normal and within healthy range."],
            "treatments": treatments if treatments else ["None required. Maintain a healthy lifestyle."],
            "recommendation": rec,
            "message": f"Agentic AI Analysis Completion: {risk_level.upper()} Status."
        }
        
        # Logging for background trace
        logger.log_event(
            module="Health Intelligence", 
            status="ALERT" if alert_triggered else "NORMAL", 
            details=f"HR: {heart_rate}, SpO2: {oxygen_level}%, State: {risk_level}"
        )
            
        return response


    def process_vision_data(self, image_bytes: bytes, filename: str) -> dict:
        # 1. DETECT / ANALYZE (Using the YOLO Model)
        vision_result = vision_service.detect_objects(image_bytes)
        detections = vision_result["detections"]
        
        # Expanded Smart City Hazards
        target_risks = ["bus", "truck", "motorcycle", "car", "person", "bicycle", "fire hydrant", "stop sign"]
        risks_found = []
        
        # 2. DECIDE
        for obj in detections:
            # Lowered threshold to 35% for better real-world detection sensitivity
            if obj["class"] in target_risks and obj["confidence"] > 35.0:
                risks_found.append(obj["class"])
                
        # Structured response for Vision
        unique_risks = list(set(risks_found))
        
        vehicles = ["bus", "truck", "motorcycle", "car"]
        has_vehicle = any(v in unique_risks for v in vehicles)
        has_person = "person" in unique_risks
        
        # Only trigger a CRITICAL alert if vehicles are present
        alert_triggered = has_vehicle
        
        if alert_triggered:
            risk_level = "Critical"
        elif has_person:
            risk_level = "Caution"
        else:
            risk_level = "Stable"
        
        # Descriptive insights
        insights = []
        if has_person and not has_vehicle:
            insights.append("Occupancy Monitor: Human presence detected in safe/indoor zone.")
            
        if alert_triggered:
            for r in unique_risks:
                if r == "person":
                    insights.append("Pedestrian detected near traffic: High risk of accident.")
                elif r in ["bus", "truck"]:
                    insights.append(f"Heavy vehicle ({r}) identified: Potential traffic congestion.")
                elif r not in ["person"]:
                    insights.append(f"Hazardous object: {r.capitalize()} detected in active zone.")
                    
        if len(insights) == 0:
            insights = ["No immediate roadway hazards or traffic obstructions detected."]

        if alert_triggered:
            treatments = ["Activate roadside warning displays", "Alert local emergency dispatch"]
            rec = "Activate emergency sirens and notify local traffic control."
        elif has_person:
            treatments = ["Continue standard occupancy monitoring"]
            rec = "Scene is stable. Monitoring human presence."
        else:
            treatments = ["Continue autonomous surveillance."]
            rec = "Roadway remains clear for autonomous transit."

        response = {
            "status": "success",
            "detected_objects": detections,
            "image_base64": vision_result["image_base64"],
            "filename": filename,
            "alert_triggered": alert_triggered,
            "risk_level": risk_level,
            "insights": insights,
            "treatments": treatments,
            "recommendation": rec,
            "message": f"Hazard Analysis: {len(unique_risks)} threats identified." if alert_triggered else "AI Monitoring: Scene Secure."
        }
        
        # 3. ACT
        logger.log_event(
            module="Smart City Vision", 
            status="ALERT" if alert_triggered else "NORMAL", 
            details=f"Hazards: {', '.join(unique_risks)}"
        )
            
        return response



# Export single instance
agent = AgentService()
