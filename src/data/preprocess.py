# ==========================================================
# SmartHire Project
# File: preprocess.py
# Purpose: Clean Resume and Job datasets
# ==========================================================

import pandas as pd
import re
import os

# ----------------------------------------------------------
# Dataset Paths
# ----------------------------------------------------------

RESUME_PATH = "data/raw/UpdatedResumeDataSet.csv"
LINKEDIN_PATH = "data/raw/linkedin.csv"
NAUKRI_PATH = "data/raw/naukri_jobs.csv"

# ----------------------------------------------------------
# Output Folder
# ----------------------------------------------------------

OUTPUT_FOLDER = "data/processed"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ----------------------------------------------------------
# Function to Load Dataset
# ----------------------------------------------------------

def load_dataset(path):
    try:
        df = pd.read_csv(path, low_memory=False)
        print(f"Loaded: {path}")
        return df
    except Exception as e:
        print(e)
        return None

# ----------------------------------------------------------
# Text Cleaning Function
# ----------------------------------------------------------

def clean_text(text):

    # If value is empty
    if pd.isna(text):
        return ""

    # Convert to string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = re.sub(r'[^a-zA-Z ]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# ----------------------------------------------------------
# Clean Dataset Function
# ----------------------------------------------------------

def clean_dataset(df):

    if df is None:
        return None

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove rows with all missing values
    df = df.dropna(how="all")

    # Fill remaining missing values
    df = df.fillna("")

    # Clean every text column
    for column in df.columns:

        if df[column].dtype == object:
            df[column] = df[column].apply(clean_text)

    return df

# ----------------------------------------------------------
# Load Datasets
# ----------------------------------------------------------

resume_df = load_dataset(RESUME_PATH)
linkedin_df = load_dataset(LINKEDIN_PATH)
naukri_df = load_dataset(NAUKRI_PATH)

# ----------------------------------------------------------
# Clean Datasets
# ----------------------------------------------------------

resume_df = clean_dataset(resume_df)
linkedin_df = clean_dataset(linkedin_df)
naukri_df = clean_dataset(naukri_df)

# ----------------------------------------------------------
# Save Cleaned Files
# ----------------------------------------------------------

resume_df.to_csv(
    "data/processed/resume_cleaned.csv",
    index=False
)

linkedin_df.to_csv(
    "data/processed/linkedin_cleaned.csv",
    index=False
)

naukri_df.to_csv(
    "data/processed/naukri_cleaned.csv",
    index=False
)

print("\nAll datasets cleaned successfully.")
print("Saved inside data/processed/")