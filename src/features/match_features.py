# ==========================================================
# SmartHire Project
# File: match_features.py
# Purpose: Create Resume-Job Matching Features
# ==========================================================

import pandas as pd
import numpy as np
import re

# ----------------------------------------------------------
# Dataset Path
# ----------------------------------------------------------

DATA_PATH = "data/processed/resume_cleaned.csv"

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

df = pd.read_csv(DATA_PATH, low_memory=False)

print("Dataset Loaded Successfully")

# Remove BOM if present
df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

# ----------------------------------------------------------
# Fill Missing Values
# ----------------------------------------------------------

df = df.fillna("")

# ----------------------------------------------------------
# Function to Clean Skills
# ----------------------------------------------------------

def clean_skill_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z0-9, ]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# ----------------------------------------------------------
# Function to Convert Skills into Set
# ----------------------------------------------------------

def skill_set(text):

    text = clean_skill_text(text)

    skills = text.split(",")

    skills = [skill.strip() for skill in skills if skill.strip() != ""]

    return set(skills)

# ----------------------------------------------------------
# Skill Overlap
# ----------------------------------------------------------

def calculate_skill_overlap(candidate_skills, required_skills):

    candidate = skill_set(candidate_skills)

    required = skill_set(required_skills)

    if len(required) == 0:
        return 0

    overlap = candidate.intersection(required)

    score = len(overlap) / len(required)

    return round(score, 2)

# ----------------------------------------------------------
# Experience Match
# ----------------------------------------------------------

def experience_score(responsibilities):

    responsibilities = str(responsibilities)

    words = responsibilities.split()

    return len(words)

# ----------------------------------------------------------
# Education Score
# ----------------------------------------------------------

def education_score(degree):

    degree = str(degree).lower()

    if "phd" in degree:
        return 4

    elif "master" in degree:
        return 3

    elif "bachelor" in degree:
        return 2

    elif "diploma" in degree:
        return 1

    else:
        return 0

# ----------------------------------------------------------
# Language Count
# ----------------------------------------------------------

def language_score(language):

    language = str(language)

    return len(language.split(","))

# ----------------------------------------------------------
# Certification Count
# ----------------------------------------------------------

def certification_score(certification):

    certification = str(certification)

    return len(certification.split(","))

# ----------------------------------------------------------
# Create Features
# ----------------------------------------------------------

df["skill_overlap"] = df.apply(

    lambda row:

    calculate_skill_overlap(

        row["skills"],

        row["skills_required"]

    ),

    axis=1

)

df["experience_score"] = df["responsibilities"].apply(experience_score)

df["education_score"] = df["degree_names"].apply(education_score)

df["language_score"] = df["languages"].apply(language_score)

df["certification_score"] = df["certification_skills"].apply(certification_score)

# ----------------------------------------------------------
# Display Sample
# ----------------------------------------------------------

print("\nGenerated Features")

print(

    df[

        [

            "skill_overlap",

            "experience_score",

            "education_score",

            "language_score",

            "certification_score"

        ]

    ].head()

)

# ----------------------------------------------------------
# Save Feature Dataset
# ----------------------------------------------------------

OUTPUT = "data/processed/match_features.csv"

df.to_csv(

    OUTPUT,

    index=False

)

print("\nFeature Dataset Saved Successfully")

print("Location :", OUTPUT)