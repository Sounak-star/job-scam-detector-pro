## Fraudulent Job Detector

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![ML](https://img.shields.io/badge/machine%20learning-scikit--learn-orange)
![Web](https://img.shields.io/badge/framework-streamlit-ff69b4)
![License](https://img.shields.io/badge/license-MIT-green)

An AI-powered tool to detect fraudulent job postings using **NLP** and machine learning.

---

## Project Structure
fraudulent-job-detector/
├── core/ # Core functionality
│ ├── data_loader.py # Loads and preprocesses job posting data
│ ├── scam_detector.py # Implements detection algorithms
│ ├── setup.py # Package configuration
│ ├── api.py # REST API endpoints (Flask/FastAPI)
│ ├── app.py # Streamlit web interface
│ ├── jobs.csv # Sample dataset (real vs fraudulent postings)
│ └── model.joblib # Serialized trained model
├── requirements.txt # Python dependencies
└── train.py # Model training pipeline