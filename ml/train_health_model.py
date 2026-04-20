import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def generate_health_data(num_samples=1000):
    """
    Generate synthetic health dataset.
    Normal: Heart Rate (60-100), SpO2 (95-100)
    Abnormal: Heart Rate (<60 or >100), SpO2 (<95)
    """
    np.random.seed(42)
    
    # Generate Normal Data (Label 0)
    normal_len = num_samples // 2
    normal_hr = np.random.randint(60, 101, size=normal_len)
    normal_spo2 = np.random.randint(95, 101, size=normal_len)
    normal_labels = np.zeros(normal_len)
    
    # Generate Abnormal Data (Label 1)
    abnormal_len = num_samples // 2
    abnormal_hr_low = np.random.randint(40, 60, size=abnormal_len // 2)
    abnormal_hr_high = np.random.randint(101, 140, size=abnormal_len - len(abnormal_hr_low))
    abnormal_hr = np.concatenate([abnormal_hr_low, abnormal_hr_high])
    
    abnormal_spo2 = np.random.randint(80, 95, size=abnormal_len)
    abnormal_labels = np.ones(abnormal_len)
    
    # Combine
    hr = np.concatenate([normal_hr, abnormal_hr])
    spo2 = np.concatenate([normal_spo2, abnormal_spo2])
    labels = np.concatenate([normal_labels, abnormal_labels])
    
    # Shuffle the dataset
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    df = pd.DataFrame({
        'heart_rate': hr[indices],
        'oxygen_level': spo2[indices],
        'status': labels[indices]
    })
    
    return df

def train_and_save_model():
    print("Generating synthetic health dataset...")
    df = generate_health_data()
    
    X = df[['heart_rate', 'oxygen_level']]
    y = df['status']
    
    print("Splitting generated data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Classifier model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model trained successfully. Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model in backend/models
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'health_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model()
