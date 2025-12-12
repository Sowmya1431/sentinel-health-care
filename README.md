# Sentinel Health

Sentinel Health is an AI-based video analysis system that detects human falls, unsafe postures, emotional states, and aggressive actions from uploaded videos or webcam feeds using deep learning models.

---

## Features

- Secure login-based dashboard  
- Video upload and analysis  
- Fall detection  
- Unsafe pose detection  
- Facial emotion recognition  
- Action and aggression detection  
- Visual alerts and audio beep for critical events  

---

## Models Used

- YOLOv8 – Person detection  
- YOLOv8-Pose – Pose estimation  
- FER – Facial emotion recognition  
- SlowFast R50 – Aggressive action detection  

---

## Tech Stack

- Python, Flask  
- OpenCV, NumPy  
- PyTorch  
- HTML, CSS, JavaScript  
- MongoDB (GridFS)  

---

## Setup & Run

### Clone Repository
```bash
git clone https://github.com/your-username/sentinel-health.git
cd sentinel-health
python -m venv venv
venv\Scripts\activate   # or source venv/bin/activate
pip install -r requirements.txt
python app.py
