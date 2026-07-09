# ==========================================================
# SmartHire Project
# File: load_data.py
# Purpose: Load all datasets required for the project
# ==========================================================

# Import required libraries
import pandas as pd
import os


# ----------------------------------------------------------
# Dataset file paths
# ----------------------------------------------------------

RESUME_PATH = "data/raw/UpdatedResumeDataSet.csv"
LINKEDIN_PATH = "data/raw/linkedin.csv"
NAUKRI_PATH = "data/raw/naukri_jobs.csv"


# ----------------------------------------------------------
# Function to load a CSV file safely
# ----------------------------------------------------------

def load_dataset(file_path):

    # Check whether the file exists
    if not os.path.exists(file_path):
        print(f"\nERROR: File not found -> {file_path}")
        return None

    try:
        # Read CSV file
        dataframe = pd.read_csv(file_path)

        print(f"\nSUCCESS: Loaded {file_path}")

        return dataframe

    except Exception as error:
        print(f"\nERROR while reading {file_path}")

        print(error)

        return None


# ----------------------------------------------------------
# Load all datasets
# ----------------------------------------------------------

resume_df = load_dataset(RESUME_PATH)

linkedin_df = load_dataset(LINKEDIN_PATH)

naukri_df = load_dataset(NAUKRI_PATH)


# ----------------------------------------------------------
# Function to display dataset information
# ----------------------------------------------------------

def dataset_information(name, dataframe):

    print("\n")
    print("=" * 70)

    print(name)

    print("=" * 70)

    if dataframe is None:

        print("Dataset not loaded.")

        return

    print("\nShape of Dataset")
    print(dataframe.shape)

    print("\nColumn Names")
    print(dataframe.columns.tolist())

    print("\nData Types")
    print(dataframe.dtypes)

    print("\nFirst Five Rows")
    print(dataframe.head())

    print("\nLast Five Rows")
    print(dataframe.tail())

    print("\nMissing Values")
    print(dataframe.isnull().sum())

    print("\nDuplicate Rows")
    print(dataframe.duplicated().sum())


# ----------------------------------------------------------
# Display information of every dataset
# ----------------------------------------------------------

dataset_information("Resume Dataset", resume_df)

dataset_information("LinkedIn Dataset", linkedin_df)

dataset_information("Naukri Dataset", naukri_df)