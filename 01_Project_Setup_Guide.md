# 🚀 Telecom Churn Prediction MLOps Pipeline

# Chapter 1 – Complete Project Setup Guide

> This document explains how to set up the Telecom Churn Prediction MLOps project from scratch. It is intended for interview preparation, project documentation, and GitHub reference.

---

# 📖 Table of Contents

1. Project Overview
2. Project Objectives
3. Technology Stack
4. System Requirements
5. Project Folder Structure
6. Clone the Repository
7. Create Virtual Environment
8. Install Dependencies
9. Configure Environment Variables
10. Run the Project Locally
11. Train the Machine Learning Model
12. Start the FastAPI Server
13. Test the REST API
14. Docker Setup
15. Run with Docker
16. GitHub Actions Workflow
17. Common Commands
18. Troubleshooting
19. Interview Questions

---

# 1. Project Overview

## Project Name

**Telecom Churn Prediction MLOps Pipeline**

### Problem Statement

Telecom companies lose customers every month.

This is known as **Customer Churn**.

The goal of this project is to predict whether a customer is likely to leave the telecom company before they actually do.

The prediction helps businesses:

* Reduce customer loss
* Increase customer retention
* Improve customer satisfaction
* Save marketing costs

---

# 2. Project Objectives

This project demonstrates an end-to-end MLOps workflow.

The pipeline includes:

* Data preprocessing
* Model training
* Model evaluation
* Model serialization
* REST API deployment
* Docker containerization
* GitHub Actions CI/CD
* Automated testing

Instead of only building a machine learning model, the objective is to build a production-ready application.

---

# 3. Technology Stack

| Technology          | Purpose               |
| ------------------- | --------------------- |
| Python              | Programming Language  |
| Pandas              | Data Processing       |
| NumPy               | Numerical Computation |
| Scikit-Learn        | Machine Learning      |
| Joblib              | Save Trained Model    |
| FastAPI             | REST API              |
| Uvicorn             | API Server            |
| Docker              | Containerization      |
| Git                 | Version Control       |
| GitHub              | Repository Hosting    |
| GitHub Actions      | CI/CD                 |
| Pytest              | Unit Testing          |
| MLflow *(Optional)* | Experiment Tracking   |
| DVC *(Optional)*    | Dataset Versioning    |

---

# 4. System Requirements

Minimum Requirements

* Windows / Linux / MacOS
* Python 3.11+
* Git
* Docker Desktop
* VS Code
* Internet Connection

Recommended

* 8GB RAM
* 20GB Free Storage

---

# 5. Project Folder Structure

```text
Telecom-churn-mlops/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── notebooks/
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── tests/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci.yml
│
└── README.md
```

---

# 6. Clone Repository

```bash
git clone https://github.com/RavindraM0/Telecom-churn-mlops.git
```

Move into the project folder.

```bash
cd Telecom-churn-mlops
```

---

# 7. Create Virtual Environment

Windows

```bash
python -m venv venv
```

Activate

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

# 8. Install Dependencies

Install all required libraries.

```bash
pip install -r requirements.txt
```

Typical libraries include:

```text
pandas
numpy
scikit-learn
fastapi
uvicorn
joblib
pytest
mlflow
dvc
```

Verify installation

```bash
pip list
```

---

# 9. Configure Environment

If the project uses a `.env` file:

```text
MODEL_PATH=models/model.pkl
HOST=0.0.0.0
PORT=8000
```

Load variables using:

```python
from dotenv import load_dotenv
load_dotenv()
```

---

# 10. Dataset Preparation

The dataset contains telecom customer information.

Example columns:

* gender
* SeniorCitizen
* Partner
* Dependents
* tenure
* InternetService
* Contract
* MonthlyCharges
* TotalCharges
* Churn

The target variable is:

```text
Churn
```

Where

```
Yes = Customer Leaves

No = Customer Stays
```

---

# 11. Data Preprocessing

Before training, the dataset is cleaned.

Typical preprocessing includes:

* Remove missing values
* Encode categorical columns
* Scale numerical columns
* Split train/test dataset

Output:

```
Clean Dataset
```

---

# 12. Train the Model

Run

```bash
python src/train.py
```

Training performs:

* Read Dataset
* Clean Data
* Feature Engineering
* Split Dataset
* Train Random Forest
* Evaluate Model
* Save Model

Generated files

```text
models/model.pkl
```

```text
models/preprocessor.pkl
```

---

# 13. Start FastAPI Server

Run

```bash
uvicorn app:app --reload
```

Output

```
Uvicorn running on

http://127.0.0.1:8000
```

---

# 14. API Documentation

Swagger UI

```
http://127.0.0.1:8000/docs
```

Alternative Documentation

```
http://127.0.0.1:8000/redoc
```

Swagger allows testing the API without writing frontend code.

---

# 15. Test Prediction API

Example Request

```json
{
  "gender":"Male",
  "SeniorCitizen":0,
  "Partner":"Yes",
  "Dependents":"No",
  "tenure":12,
  "MonthlyCharges":75.5,
  "TotalCharges":890
}
```

Response

```json
{
  "prediction":"No Churn"
}
```

---

# 16. Docker Setup

Build Docker Image

```bash
docker build -t telecom-churn .
```

Verify

```bash
docker images
```

---

# 17. Run Docker Container

```bash
docker run -p 8000:8000 telecom-churn
```

Application becomes available at

```
http://localhost:8000
```

---

# 18. Docker Compose (If Available)

```bash
docker compose up --build
```

Stop

```bash
docker compose down
```

---

# 19. GitHub Actions

Every push to GitHub triggers the CI pipeline.

Typical workflow:

```
Developer Pushes Code
        │
        ▼
GitHub Repository
        │
        ▼
GitHub Actions Starts
        │
        ▼
Install Dependencies
        │
        ▼
Run Tests
        │
        ▼
Build Docker Image
        │
        ▼
Pipeline Passes
```

Benefits:

* No manual testing
* Automatic validation
* Consistent builds
* Better software quality

---

# 20. Useful Commands

## Create Virtual Environment

```bash
python -m venv venv
```

Activate

```bash
venv\Scripts\activate
```

Install Packages

```bash
pip install -r requirements.txt
```

Run API

```bash
uvicorn app:app --reload
```

Run Tests

```bash
pytest
```

Build Docker

```bash
docker build -t telecom-churn .
```

Run Docker

```bash
docker run -p 8000:8000 telecom-churn
```

Push Code

```bash
git add .

git commit -m "Updated Project"

git push
```

---

# 21. Troubleshooting

### Module Not Found

```bash
pip install -r requirements.txt
```

---

### Port Already in Use

Change the port.

```bash
uvicorn app:app --port 8001
```

---

### Docker Not Running

Start Docker Desktop before executing Docker commands.

---

### Virtual Environment Not Activated

Activate:

```bash
venv\Scripts\activate
```

or

```bash
source venv/bin/activate
```

---

# 22. Complete Setup Flow

```text
Clone Repository
        │
        ▼
Create Virtual Environment
        │
        ▼
Install Dependencies
        │
        ▼
Prepare Dataset
        │
        ▼
Train Model
        │
        ▼
Save Model (.pkl)
        │
        ▼
Launch FastAPI
        │
        ▼
Test Using Swagger
        │
        ▼
Dockerize Application
        │
        ▼
Push Code to GitHub
        │
        ▼
GitHub Actions Executes CI Pipeline
```

---

# Interview Answer

### **Interviewer:** *"How did you set up your Telecom Churn Prediction project?"*

**Answer:**

> "I started by cloning the GitHub repository and creating a Python virtual environment to isolate dependencies. After installing all required packages from the `requirements.txt` file, I prepared the telecom churn dataset by cleaning missing values and encoding categorical features. I then trained a Random Forest model using Scikit-learn and saved both the trained model and preprocessing pipeline as `.pkl` files with Joblib. Next, I developed a FastAPI application that loads these artifacts and exposes a REST API for real-time churn prediction. I verified the endpoints using Swagger UI. To make the application portable, I containerized it with Docker. Finally, I configured GitHub Actions to automatically install dependencies, run tests, and build the application whenever new code is pushed to the repository. This setup mirrors a production-ready MLOps workflow rather than just a standalone machine learning notebook."

---

# Key Takeaways

✅ Production-ready ML application

✅ REST API for predictions

✅ Docker containerization

✅ Automated CI/CD

✅ Reproducible environment

✅ Easy deployment

✅ Scalable architecture

---

**Next Chapter:** `02_Project_Workflow_Deep_Dive.md` — End-to-end explanation of how every component works together, from dataset ingestion to real-time prediction.
