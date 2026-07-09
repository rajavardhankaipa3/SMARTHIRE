# ==========================================================
# SmartHire Project
# File: streamlit_app.py
# ==========================================================

import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------

st.set_page_config(
    page_title="SmartHire AI",
    page_icon="💼",
    layout="wide"
)

st.title("💼 SmartHire AI")
st.subheader("Resume Screening & Job Recommendation System")

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

DATA_PATH = "data/processed/match_features.csv"

if not os.path.exists(DATA_PATH):
    st.error("match_features.csv not found.")
    st.stop()

df = pd.read_csv(DATA_PATH, low_memory=False)

# Remove BOM and extra spaces
df.columns = (
    df.columns
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
)

# Fill missing values
df = df.fillna("")

print(df.columns.tolist())

# ----------------------------------------------------------
# Create Resume Text
# ----------------------------------------------------------

resume_columns = [
    "career_objective",
    "skills",
    "degree_names",
    "major_field_of_studies",
    "professional_company_names",
    "positions",
    "responsibilities",
    "languages",
    "certification_skills"
]

df["resume_text"] = ""

for col in resume_columns:
    if col in df.columns:
        df["resume_text"] += df[col].astype(str) + " "

# ----------------------------------------------------------
# Create Job Text
# ----------------------------------------------------------

job_columns = [
    "job_position_name",
    "skills_required",
    "educationaL_requirements",
    "experiencere_requirement",
    "responsibilities.1"
]

df["job_text"] = ""

for col in job_columns:
    if col in df.columns:
        df["job_text"] += df[col].astype(str) + " "

# ----------------------------------------------------------
# Load Models
# ----------------------------------------------------------

classifier = None
vectorizer = None
fit_predictor = None

if os.path.exists("models/classifier.pkl"):
    classifier = joblib.load("models/classifier.pkl")

if os.path.exists("models/classifier_vectorizer.pkl"):
    vectorizer = joblib.load("models/classifier_vectorizer.pkl")

if os.path.exists("models/fit_predictor.pkl"):
    fit_predictor = joblib.load("models/fit_predictor.pkl")

# ----------------------------------------------------------
# Candidate Selection
# ----------------------------------------------------------

st.sidebar.header("Candidate")

candidate_index = st.sidebar.number_input(
    "Candidate Index",
    min_value=0,
    max_value=len(df)-1,
    value=0
)

candidate = df.iloc[candidate_index]

# ----------------------------------------------------------
# Candidate Information
# ----------------------------------------------------------

st.header("Candidate Information")

col1, col2 = st.columns(2)

with col1:

    st.write("### Career Objective")
    st.write(candidate["career_objective"])

    st.write("### Skills")
    st.write(candidate["skills"])

    st.write("### Degree")
    st.write(candidate["degree_names"])

with col2:

    st.write("### Experience")
    st.write(candidate["responsibilities"])

    st.write("### Languages")
    st.write(candidate["languages"])

    st.write("### Certifications")
    st.write(candidate["certification_skills"])

# ----------------------------------------------------------
# Job Role Prediction
# ----------------------------------------------------------

st.header("Predicted Job Role")

if classifier is not None and vectorizer is not None:

    feature = vectorizer.transform([candidate["resume_text"]])

    prediction = classifier.predict(feature)

    st.success(prediction[0])

else:

    st.warning("Classifier model not available.")

# ----------------------------------------------------------
# Fit Score Prediction
# ----------------------------------------------------------

st.header("Predicted Fit Score")

feature_columns = [
    "skill_overlap",
    "experience_score",
    "education_score",
    "language_score",
    "certification_score"
]

if fit_predictor is not None:

    if all(col in df.columns for col in feature_columns):

        X = df.loc[[candidate_index], feature_columns]

        score = fit_predictor.predict(X)[0]

        st.metric("Candidate Fit Score", f"{score:.2f}")

    else:

        st.error("Feature columns are missing in match_features.csv")

        st.write("Available Columns:")

        st.write(df.columns.tolist())

else:

    st.warning("Fit Predictor model not found.")

# ----------------------------------------------------------
# Job Recommendation
# ----------------------------------------------------------

st.header("Top Recommended Jobs")

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

tfidf.fit(
    pd.concat(
        [
            df["resume_text"],
            df["job_text"]
        ]
    )
)

resume_vectors = tfidf.transform(df["resume_text"])

job_vectors = tfidf.transform(df["job_text"])

similarity = cosine_similarity(
    resume_vectors[candidate_index],
    job_vectors
)

top = similarity[0].argsort()[::-1][:5]

recommendations = df.iloc[top][
    [
        "job_position_name",
        "skills_required",
        "matched_score"
    ]
].copy()

recommendations["Similarity"] = similarity[0][top]

st.dataframe(
    recommendations,
    use_container_width=True
)

# ----------------------------------------------------------
# Footer
# ----------------------------------------------------------

st.markdown("---")

st.write("SmartHire AI using Machine Learning")