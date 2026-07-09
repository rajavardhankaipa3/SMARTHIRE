# ==========================================================
# SmartHire Project
# File: text_features.py
# Purpose: Create TF-IDF Features from Resume Dataset
# ==========================================================

import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer

# ----------------------------------------------------------
# Dataset Path
# ----------------------------------------------------------

RESUME_PATH = "data/processed/resume_cleaned.csv"

# ----------------------------------------------------------
# Model Folder
# ----------------------------------------------------------

MODEL_FOLDER = "models"

os.makedirs(MODEL_FOLDER, exist_ok=True)

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

resume_df = pd.read_csv(RESUME_PATH, low_memory=False)

print("Resume Dataset Loaded Successfully")

print("\nColumns in Dataset:")
print(resume_df.columns.tolist())

# ----------------------------------------------------------
# Fill Missing Values
# ----------------------------------------------------------

resume_df = resume_df.fillna("")

# ----------------------------------------------------------
# Create One Combined Resume Text Column
# ----------------------------------------------------------

resume_df["combined_text"] = (

    resume_df["career_objective"].astype(str) + " " +

    resume_df["skills"].astype(str) + " " +

    resume_df["degree_names"].astype(str) + " " +

    resume_df["major_field_of_studies"].astype(str) + " " +

    resume_df["professional_company_names"].astype(str) + " " +

    resume_df["positions"].astype(str) + " " +

    resume_df["responsibilities"].astype(str) + " " +

    resume_df["languages"].astype(str) + " " +

    resume_df["certification_skills"].astype(str)

)

print("\nCombined Resume Text Created Successfully")

# ----------------------------------------------------------
# Create TF-IDF Vectorizer
# ----------------------------------------------------------

vectorizer = TfidfVectorizer(

    stop_words="english",

    max_features=5000

)

# ----------------------------------------------------------
# Convert Text into Numerical Features
# ----------------------------------------------------------

X = vectorizer.fit_transform(resume_df["combined_text"])

# ----------------------------------------------------------
# Display Information
# ----------------------------------------------------------

print("\nTF-IDF Matrix Shape")
print(X.shape)

print("\nNumber of Features")
print(len(vectorizer.get_feature_names_out()))

print("\nFirst 20 Features")
print(vectorizer.get_feature_names_out()[:20])

# ----------------------------------------------------------
# Save TF-IDF Vectorizer
# ----------------------------------------------------------

joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\nTF-IDF Vectorizer Saved Successfully")

# ----------------------------------------------------------
# Save TF-IDF Matrix
# ----------------------------------------------------------

joblib.dump(X, "models/resume_features.pkl")

print("Resume Feature Matrix Saved Successfully")