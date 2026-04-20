from ultralytics import YOLO
import cv2
import numpy as np
import base64
import io
import os

class VisionService:
    def __init__(self):
        self.model = None

    def load_model(self):
        if self.model is None:
            # We use the pre-trained yolo v8 nano for fast inference
            print("Loading YOLOv8n Object Detection model...")
            self.model = YOLO('yolov8n.pt')
            print("Successfully loaded Smart City Vision model.")

    def detect_objects(self, image_bytes: bytes) -> dict:
        if self.model is None:
            print("Vision model was not loaded yet. Lazy loading now...")
            self.load_model()
            
        # Convert raw bytes into a cv2 image format
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image from bytes. Ensure the image is valid.")

        # --- OPTIMIZATION: Resize for faster inference ---
        # Large images from modern phones can slow down YOLO significantly on CPU
        max_size = 640
        h, w = img.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            print(f"Resized image for faster detection: {w}x{h} -> {img.shape[1]}x{img.shape[0]}")

        # Run inference
        results = self.model(img)
        res = results[0]
        
        detected_objects = []
        
        for box in res.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0])
            
            # Rounding for clean API response
            conf_percent = round(confidence * 100, 2)
            
            print(f"Detected: {class_name} ({conf_percent}%)")
            
            detected_objects.append({
                "class": class_name,
                "confidence": conf_percent
            })
            
        # Custom Professional Bounding Boxes (Like real Traffic/Smart City cameras)
        annotated_img = img.copy()
        # Dynamic scaling based on actual image resolution
        img_h, img_w = annotated_img.shape[:2]
        scale_factor = max(0.35, max(img_h, img_w) / 1000.0) # Downscales nicely for low-res streams
        box_thick = max(1, int(4 * scale_factor))
        font_scale = 0.8 * scale_factor
        font_thick = max(1, int(2 * scale_factor))
        pad = int(8 * scale_factor)

        for box in res.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0])
            conf_percent = round(confidence * 100, 2)
            
            # Get coordinates safely
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Theme Colors (BGR format): Magenta for People, Cyan for Vehicles/Others
            color = (255, 50, 255) if class_name == "person" else (255, 255, 0)
            
            # 1. Draw dynamically thickened bounding box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, box_thick)
            
            # 2. Dynamic label logic
            label = f"{class_name.upper()} {conf_percent}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            
            total_th = th + pad * 2
            # If box is too close to top of frame, draw label inside the box, else above it
            label_y = y1 if y1 > total_th else y1 + total_th
            
            # Label Background
            cv2.rectangle(annotated_img, (x1, label_y - total_th), (x1 + tw + pad * 2, label_y), color, -1)
            
            # 3. Text
            cv2.putText(annotated_img, label, (x1 + pad, label_y - pad), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thick)
        
        # Convert the annotated image back to base64
        _, buffer = cv2.imencode('.jpg', annotated_img)
        base64_image = base64.b64encode(buffer).decode('utf-8')
            
        return {
            "detections": detected_objects,
            "image_base64": base64_image
        }


vision_service = VisionService()
