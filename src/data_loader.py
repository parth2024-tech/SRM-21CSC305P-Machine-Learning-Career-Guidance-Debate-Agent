"""
Data Loader & Dataset Familiarization Module (Unit 1 Syllabus & Practice)
Implements:
  - Practice 1: Devise a program to import, load and view dataset.
  - Practice 2: Create a program to display summary and statistics of the dataset.
  - Unit 1 Foundations: Discrete & continuous random variables, Quantiles, Mean,
    Variance, Covariance, Bayes Rule Prior Probabilities.
"""

import os
from typing import Any

import numpy as np
import pandas as pd

CAREER_TRACK_MAPPING = {
    # Software & Systems Engineering
    "Software Engineer": "Software & Systems Engineering",
    "Software Developer": "Software & Systems Engineering",
    "Software Systems Engineer": "Software & Systems Engineering",
    "Applications Developer": "Software & Systems Engineering",
    "Mobile Applications Developer": "Software & Systems Engineering",
    "Web Developer": "Software & Systems Engineering",
    "Programmer Analyst": "Software & Systems Engineering",
    
    # Data Science & Database Systems
    "Database Developer": "Data Science & Database Systems",
    "Database Administrator": "Data Science & Database Systems",
    "Database Manager": "Data Science & Database Systems",
    "Data Architect": "Data Science & Database Systems",
    "Business Intelligence Analyst": "Data Science & Database Systems",
    
    # Network Architecture & Cybersecurity
    "Network Engineer": "Network Architecture & Cybersecurity",
    "Network Security Engineer": "Network Architecture & Cybersecurity",
    "Network Security Administrator": "Network Architecture & Cybersecurity",
    "Systems Security Administrator": "Network Architecture & Cybersecurity",
    "Information Security Analyst": "Network Architecture & Cybersecurity",
    
    # Business, Systems & Management
    "Project Manager": "Business, Systems & Management",
    "Information Technology Manager": "Business, Systems & Management",
    "Business Systems Analyst": "Business, Systems & Management",
    "CRM Business Analyst": "Business, Systems & Management",
    "Systems Analyst": "Business, Systems & Management",
    "E-Commerce Analyst": "Business, Systems & Management",
    "Solutions Architect": "Business, Systems & Management",
    
    # Quality Assurance, UX & Operations
    "UX Designer": "Quality Assurance, UX & Operations",
    "Design & UX": "Quality Assurance, UX & Operations",
    "Software Quality Assurance (QA) / Testing": "Quality Assurance, UX & Operations",
    "Quality Assurance Associate": "Quality Assurance, UX & Operations",
    "Portal Administrator": "Quality Assurance, UX & Operations",
    "Technical Support": "Quality Assurance, UX & Operations",
    "Technical Services/Help Desk/Tech Support": "Quality Assurance, UX & Operations",
    "CRM Technical Developer": "Quality Assurance, UX & Operations",
    "Information Technology Auditor": "Quality Assurance, UX & Operations",
    "Technical Engineer": "Quality Assurance, UX & Operations"
}


def load_raw_dataset(filepath: str = "data/career_guidance_dataset.csv") -> pd.DataFrame:
    """Practice 1: Import, load and view dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at {filepath}")
    df = pd.read_csv(filepath)
    # Strip any whitespace in column names
    df.columns = [c.strip() for c in df.columns]
    return df


def compute_summary_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """
    Practice 2: Display the summary and statistics of the dataset.
    Computes mean, variance, quantiles (25%, 50%, 75%, 90%, 95%),
    covariance, and discrete/continuous variable identification.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Continuous variable metrics (Mean, Variance, Quantiles)
    stats_dict = {}
    for col in numeric_cols:
        series = df[col]
        stats_dict[col] = {
            "mean": float(series.mean()),
            "variance": float(series.var()),
            "std": float(series.std()),
            "min": float(series.min()),
            "q25": float(series.quantile(0.25)),
            "median": float(series.median()),
            "q75": float(series.quantile(0.75)),
            "q90": float(series.quantile(0.90)),
            "max": float(series.max())
        }
    
    # Covariance Matrix
    cov_matrix = df[numeric_cols].cov()
    
    # Discrete Random Variable Distributions (Prior Probabilities P(Y))
    target_priors = df['Suggested Job Role'].value_counts(normalize=True).to_dict()
    
    return {
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "feature_statistics": stats_dict,
        "covariance_matrix": cov_matrix,
        "target_priors": target_priors,
        "num_samples": len(df),
        "num_features": len(df.columns)
    }


def prepare_dataset_for_modeling(
    filepath: str = "data/career_guidance_dataset.csv"
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Full preprocessing pipeline:
      - Maps 34 fine-grained roles to 5 high-level Career Tracks
      - Computes continuous regression targets (Career Readiness Score & Expected Starting Salary in $k)
      - Encodes binary and ordinal variables
      - Performs One-Hot Encoding for categorical variables
    """
    df = load_raw_dataset(filepath).copy()
    
    # 1. Map target to Strategic Career Tracks
    df['Career_Track'] = df['Suggested Job Role'].map(CAREER_TRACK_MAPPING)
    
    # 2. Continuous Regression Targets (Unit 2 Practice 1: Linear Models for Regression)
    # Grounded composite index of career readiness (40-100 scale) and expected starting compensation ($k)
    # Academic weight (0.35) + Coding/Logical weight (0.35) + Practical/Hackathon/Speaking weight (0.30)
    acad_avg = df[[
        'Acedamic percentage in Operating Systems',
        'percentage in Algorithms',
        'Percentage in Programming Concepts',
        'Percentage in Software Engineering',
        'Percentage in Computer Networks',
        'Percentage in Electronics Subjects',
        'Percentage in Computer Architecture',
        'Percentage in Mathematics',
        'Percentage in Communication skills'
    ]].mean(axis=1)
    
    tech_skills = (
        (df['coding skills rating'] / 9.0) * 50.0 +
        (df['Logical quotient rating'] / 9.0) * 50.0
    )
    
    experience_boost = (
        (df['hackathons'] / 6.0) * 50.0 +
        (df['public speaking points'] / 9.0) * 50.0
    )
    
    # Career Readiness Score (Continuous variable bounded ~50-100)
    readiness_score = (
        0.40 * acad_avg +
        0.35 * tech_skills +
        0.25 * experience_boost
    )
    df['Career_Readiness_Score'] = np.round(readiness_score, 2)
    
    # Expected Annual Starting Salary in $k (Regression Target)
    # Baseline salary $50k + readiness multiplier + stochastic industry noise N(0, 2)
    np.random.seed(42)
    noise = np.random.normal(0, 2.5, size=len(df))
    expected_salary = 40.0 + 0.85 * (df['Career_Readiness_Score'] - 50.0) + noise
    df['Expected_Salary_k'] = np.round(np.clip(expected_salary, 42.0, 135.0), 2)
    
    # 3. Categorical Encoding
    binary_maps = {
        'can work long time before system?': {'yes': 1, 'no': 0},
        'self-learning capability?': {'yes': 1, 'no': 0},
        'Extra-courses did': {'yes': 1, 'no': 0},
        'talenttests taken?': {'yes': 1, 'no': 0},
        'olympiads': {'yes': 1, 'no': 0},
        'Taken inputs from seniors or elders': {'yes': 1, 'no': 0},
        'interested in games': {'yes': 1, 'no': 0},
        'In a Realtionship?': {'yes': 1, 'no': 0},
        'worked in teams ever?': {'yes': 1, 'no': 0},
        'Introvert': {'yes': 1, 'no': 0},
        'Management or Technical': {'Technical': 1, 'Management': 0},
        'Salary/work': {'salary': 1, 'work': 0},
        'hard/smart worker': {'smart worker': 1, 'hard worker': 0},
        'Gentle or Tuff behaviour?': {'gentle': 1, 'stubborn': 0},
        'Job/Higher Studies?': {'job': 1, 'higherstudies': 0}
    }
    
    for col, mapping in binary_maps.items():
        if col in df.columns:
            df[col + '_encoded'] = df[col].map(mapping).fillna(0).astype(int)
            
    ordinal_skill = {'poor': 1, 'medium': 2, 'excellent': 3}
    df['reading_writing_encoded'] = df['reading and writing skills'].map(ordinal_skill).fillna(2).astype(int)
    df['memory_capability_encoded'] = df['memory capability score'].map(ordinal_skill).fillna(2).astype(int)
    
    # One-hot encode high-cardinality categorical features
    ohe_cols = ['certifications', 'workshops', 'Interested subjects', 'interested career area', 'Type of company want to settle in?']
    df_encoded = pd.get_dummies(df, columns=ohe_cols, drop_first=True)
    
    metadata = {
        "total_records": len(df),
        "features": list(df_encoded.columns),
        "career_tracks": list(df['Career_Track'].unique()),
        "target_roles_count": df['Suggested Job Role'].nunique()
    }
    
    return df_encoded, metadata
