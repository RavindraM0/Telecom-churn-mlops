# 🚀 Telecom Churn Prediction MLOps Pipeline

# Chapter 2 – Project Workflow Deep Dive

> This document explains the complete workflow of the Telecom Churn Prediction MLOps project in a simple and interview-friendly way.

---

# 📌 Project Workflow

```text
Dataset
   │
   ▼
Data Preprocessing
   │
   ▼
Train Machine Learning Model
   │
   ▼
Evaluate Model
   │
   ▼
Save Model (.pkl)
   │
   ▼
FastAPI Loads Model
   │
   ▼
User Sends Customer Data
   │
   ▼
Prediction Generated
   │
   ▼
Response Returned to User
   │
   ▼
Docker Container
   │
   ▼
GitHub Actions CI/CD
```

---

# 1. Dataset

The project starts with the **Telecom Customer Churn dataset**.

It contains customer information such as:

* Gender
* Tenure
* Contract Type
* Internet Service
* Monthly Charges
* Total Charges
* Churn (Target)

**Goal:** Predict whether a customer will leave the company.

---

# 2. Data Preprocessing

Raw data cannot be used directly.

The preprocessing step:

* Removes missing values
* Converts categorical data into numbers
* Scales numerical values (if required)
* Splits data into training and testing sets

**Output:** Clean dataset ready for machine learning.

---

# 3. Model Training

The cleaned data is used to train a **Random Forest Classifier**.

Training steps:

* Read processed data
* Train the model
* Test the model
* Calculate accuracy and other metrics

After successful training, the model is saved as:

```text
models/model.pkl
```

The preprocessing pipeline is also saved:

```text
models/preprocessor.pkl
```

These files are reused during prediction, so the model does not need to be retrained every time.

---

# 4. FastAPI Prediction Service

FastAPI acts as the backend service.

When the server starts:

* It loads the trained model
* It loads the preprocessor
* It waits for prediction requests

The API is available through Swagger UI:

```text
http://localhost:8000/docs
```

---

# 5. Prediction Workflow

When a user sends customer details:

```text
Customer Data
      │
      ▼
FastAPI API
      │
      ▼
Preprocessor
      │
      ▼
Trained Model
      │
      ▼
Prediction
      │
      ▼
Return Result
```

Example:

Input:

```text
Tenure = 5
Monthly Charges = 95
Contract = Month-to-Month
```

Output:

```text
Customer is likely to Churn
```

or

```text
Customer is likely to Stay
```

---

# 6. Docker Workflow

Docker packages the entire application into a container.

It includes:

* Python
* Required libraries
* FastAPI application
* Trained model

Benefits:

* Same environment everywhere
* Easy deployment
* No dependency issues

Workflow:

```text
Application
      │
      ▼
Docker Image
      │
      ▼
Docker Container
      │
      ▼
Application Runs Anywhere
```

---

# 7. GitHub Actions (CI/CD)

Whenever new code is pushed to GitHub:

```text
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
Pipeline Completes
```

This automates testing and ensures the project is always in a working state.

---

# 8. Complete End-to-End Workflow

```text
1. Clone Project
        │
2. Install Dependencies
        │
3. Load Dataset
        │
4. Preprocess Data
        │
5. Train Random Forest Model
        │
6. Save model.pkl
        │
7. FastAPI Loads Model
        │
8. User Sends Customer Details
        │
9. Model Predicts Churn
        │
10. Prediction Returned
        │
11. Docker Packages Application
        │
12. GitHub Actions Automates Testing & Build
```

---

# Role of Each Component

| Component          | Contribution                                                                          |
| ------------------ | ------------------------------------------------------------------------------------- |
| **Python**         | Main programming language used to build the project                                   |
| **Pandas**         | Loads, cleans, and processes the telecom dataset                                      |
| **Scikit-learn**   | Trains the Random Forest model and performs predictions                               |
| **Joblib**         | Saves and loads the trained model (`model.pkl`)                                       |
| **FastAPI**        | Creates REST API endpoints for real-time predictions                                  |
| **Uvicorn**        | Runs the FastAPI application as a web server                                          |
| **Docker**         | Packages the application into a portable container                                    |
| **Git**            | Tracks code changes                                                                   |
| **GitHub**         | Stores the source code and enables collaboration                                      |
| **GitHub Actions** | Automatically installs dependencies, runs tests, and builds the project on every push |
| **Pytest**         | Runs automated tests to verify the project works correctly                            |

---

# Architecture Diagram

```text
                  Telecom Dataset
                        │
                        ▼
              Data Preprocessing
                        │
                        ▼
           Random Forest Training
                        │
                        ▼
              Trained Model (.pkl)
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
   FastAPI Backend               Docker Container
        │                               │
        └───────────────┬───────────────┘
                        ▼
                 REST API Endpoints
                        │
                        ▼
                  User Request
                        │
                        ▼
              Churn Prediction
                        │
                        ▼
                 Response to User
```

---

# Interview Answer (2 Minutes)

**Interviewer:** *Explain the workflow of your Telecom Churn Prediction MLOps project.*

**Answer:**

> "My project predicts whether a telecom customer is likely to leave the company. The workflow starts by loading the telecom dataset and preprocessing it by handling missing values, encoding categorical features, and preparing the data for training. I then train a Random Forest model using Scikit-learn and save both the trained model and preprocessing pipeline using Joblib. The FastAPI application loads these saved files when it starts. Whenever a user sends customer details through the API, FastAPI preprocesses the input, passes it to the trained model, and returns the prediction as either 'Churn' or 'No Churn'. To ensure the application runs consistently across environments, I containerized it using Docker. Finally, I used GitHub Actions to automate dependency installation, testing, and Docker image building whenever code is pushed to GitHub. This project demonstrates an end-to-end MLOps workflow, from model training to deployment and automation."

---

# Key Takeaways

* End-to-end Machine Learning pipeline
* Real-time prediction using FastAPI
* Model saved using Joblib
* Docker for portability
* GitHub Actions for CI/CD automation
* Production-ready project structure
