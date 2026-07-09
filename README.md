# SmartHire AI

## Project Overview

SmartHire AI is a Machine Learning based Resume Screening and Job Recommendation System.

## Features

- Resume Preprocessing
- TF-IDF Feature Extraction
- Resume Classification
- Job Recommendation
- Candidate Clustering
- Resume Fit Prediction
- Streamlit Web Application

## Technologies

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

## Project Structure

```
SmartHire/
│
├── data/
├── models/
├── src/
├── requirements.txt
└── README.md
```

## Run

Install requirements

```
pip install -r requirements.txt
```

Run Streamlit

```
streamlit run src/app/streamlit_app.py
```

## Models Used

- Logistic Regression
- Random Forest Regressor
- KMeans Clustering
- TF-IDF
- Cosine Similarity