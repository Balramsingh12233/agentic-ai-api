# 🧠 Agentic AI for Healthcare & Smart City

## 🚀 Overview

This project is an **Agentic AI-based system** designed for **Healthcare and Smart City applications**. It uses a **FastAPI backend** to process data and a **Flutter mobile app** to display real-time results to users.

The system analyzes health-related data and provides intelligent insights, alerts, and predictions using machine learning models.

---

## 🏗️ Tech Stack

### 🔹 Backend

* FastAPI (Python)
* Machine Learning Models (Health Prediction)
* REST APIs
* Token-based Authentication

### 🔹 Frontend

* Flutter (Android App)
* API Integration
* Clean UI for data visualization

---

## ⚙️ Features

### 🏥 Healthcare Module

* Predicts health status (Normal / Abnormal)
* Uses ML model for health analysis
* Secure API endpoints with token verification
* Generates alerts for abnormal conditions

### 🌆 Smart City Integration

* Scalable backend for real-time data handling
* Can be extended for IoT-based monitoring systems

### 📱 Mobile App (Flutter)

* Displays backend data in real-time
* User-friendly interface
* API-based dynamic data fetching

---

## 🔐 Security

* Token-based authentication implemented
* Secure communication between Flutter app and backend

---

## 📂 Project Structure

```
api/
 ├── routes.py
 ├── schemas.py

services/
 ├── agent_service.py
 ├── health_service.py
 ├── vision_service.py

ml/
 ├── train_health_model.py
 ├── test_yolo_inference.py

models/
 ├── health_model.pkl

utils/
 ├── security.py

main.py
```

---

## 🔄 API Example

### POST `/predict-health`

**Request:**

```json
{
  "heart_rate": 80,
  "oxygen_level": 98
}
```

**Response:**

```json
{
  "status": "Normal",
  "confidence": 0.95
}
```

---

## 📦 Installation & Setup

### Backend (FastAPI)

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

### Flutter App

```bash
flutter pub get
flutter run
```

---

## 🎯 Future Improvements

* Real-time IoT sensor integration
* Advanced AI models
* Cloud deployment (AWS / Firebase)
* Push notifications for alerts

---

## 👨‍💻 Author

Balram Singh

---

## ⭐ Note

This project demonstrates the integration of **AI + Backend + Mobile App**, making it suitable for real-world healthcare and smart city applications.
