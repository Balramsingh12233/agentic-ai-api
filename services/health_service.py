import joblib
import os

class HealthService:
    def __init__(self):
        self.model = None

    def load_model(self):
        # We ensure the model is loaded only once when the server starts
        if self.model is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, 'models', 'health_model.pkl')
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Health model not found at {model_path}. Make sure to train it first!")
                
            self.model = joblib.load(model_path)
            print("Successfully loaded Health AI model.")

    def predict(self, heart_rate: int, oxygen_level: int) -> str:
        if self.model is None:
            print("Model was not loaded yet. Lazy loading now...")
            self.load_model()
        
        # Reshape data for predicting single sample: [[HR, SpO2]]
        prediction = self.model.predict([[heart_rate, oxygen_level]])
        
        # Based on our mapping in train_health_model.py: 0 is Normal, 1 is Abnormal
        if prediction[0] == 0:
            return "Normal"
        else:
            return "Abnormal"

health_service = HealthService()
