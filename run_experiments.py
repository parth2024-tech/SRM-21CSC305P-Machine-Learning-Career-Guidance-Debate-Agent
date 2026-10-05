"""
Master Experiment Runner & Evaluation Pipeline
Executes end-to-end:
  1. Data Loading & Statistical Summaries (Unit 1 Practice 1 & 2)
  2. Exploratory Data Analysis & Visualizations (Rubric 2)
  3. Regression Suite Training (Unit 2 Practice 1: OLS, Ridge, Bayesian, Poly)
  4. Classification Suite Training (Unit 2 Practice 2: Logistic, Bayes, SVM Linear & RBF)
  5. Multi-Persona Debate Simulation & Calibrated Uncertainty
  6. Synthetic Profile Stress-Testing & Metrics Reporting (Rubrics 4 & 5)
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import (
    load_raw_dataset,
    compute_summary_statistics,
    prepare_dataset_for_modeling
)
from src.eda import generate_eda_visualizations
from src.models_regression import RegressionSuite
from src.models_classification import ClassificationSuite
from src.debate_agent import MultiPersonaDebateCoordinator
from src.metrics_evaluator import (
    generate_synthetic_test_profiles,
    format_markdown_comparison_tables
)


def main():
    print("=" * 80)
    print("  PBL 1: DEBATE-BASED CAREER GUIDANCE AGENT WITH COMPLEMENTARY PERSONAS")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # Step 1: Unit 1 Practice 1 & 2 - Data Loading & Statistics
    # -------------------------------------------------------------
    print("\n[Step 1/5] Loading Dataset & Computing Unit 1 Statistical Summaries...")
    df_raw = load_raw_dataset()
    stats = compute_summary_statistics(df_raw)
    print(f"  -> Total Samples: {stats['num_samples']}")
    print(f"  -> Numeric Features: {len(stats['numeric_features'])}")
    print(f"  -> Categorical Features: {len(stats['categorical_features'])}")
    print(f"  -> Sample Feature Mean & Quantiles (Algorithms):")
    algo_stats = stats['feature_statistics']['percentage in Algorithms']
    print(f"     Mean: {algo_stats['mean']:.2f}, Var: {algo_stats['variance']:.2f}, Q25: {algo_stats['q25']}, Median: {algo_stats['median']}, Q75: {algo_stats['q75']}")

    # -------------------------------------------------------------
    # Step 2: Feature Engineering & Preprocessing
    # -------------------------------------------------------------
    print("\n[Step 2/5] Preparing Features for Regression & Classification...")
    df_processed, metadata = prepare_dataset_for_modeling()
    print(f"  -> Engineered Target 1 (Regression): 'Expected_Salary_k' & 'Career_Readiness_Score'")
    print(f"  -> Target 2 (Classification): 'Career_Track' with {len(metadata['career_tracks'])} Strategic Domains")

    # Save EDA plots
    print("  -> Generating Publication-Quality EDA Visualizations...")
    fig_paths = generate_eda_visualizations(df_processed)
    for k, v in fig_paths.items():
        print(f"     Saved {k}: {v}")

    # -------------------------------------------------------------
    # Step 3: Train Unit 2 Regression Models
    # -------------------------------------------------------------
    print("\n[Step 3/5] Executing Unit 2 Linear Models for Regression...")
    feature_cols = [
        'Acedamic percentage in Operating Systems',
        'percentage in Algorithms',
        'Percentage in Programming Concepts',
        'Percentage in Software Engineering',
        'Percentage in Computer Networks',
        'Percentage in Electronics Subjects',
        'Percentage in Computer Architecture',
        'Percentage in Mathematics',
        'Percentage in Communication skills',
        'Hours working per day',
        'Logical quotient rating',
        'hackathons',
        'coding skills rating',
        'public speaking points',
        'can work long time before system?_encoded',
        'self-learning capability?_encoded',
        'Extra-courses did_encoded',
        'reading_writing_encoded',
        'memory_capability_encoded'
    ]
    
    X = df_processed[feature_cols].values
    y_reg = df_processed['Expected_Salary_k'].values
    y_clf = df_processed['Career_Track'].values

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.20, random_state=42, stratify=y_clf
    )

    reg_suite = RegressionSuite()
    reg_results = reg_suite.train_and_evaluate(X_train, y_reg_train, X_test, y_reg_test)
    print("  -> Regression Model Evaluation Completed.")
    for name, m in reg_results.items():
        print(f"     * {name:38s} | RMSE: {m.get('RMSE'):.4f} | R²: {m.get('R2_Score'):.4f}")

    # -------------------------------------------------------------
    # Step 4: Train Unit 2 Classification Models
    # -------------------------------------------------------------
    print("\n[Step 4/5] Executing Unit 2 Linear Models for Classification...")
    clf_suite = ClassificationSuite()
    clf_results = clf_suite.train_and_evaluate(X_train, y_clf_train, X_test, y_clf_test)
    print("  -> Classification Model Evaluation Completed.")
    for name, m in clf_results.items():
        print(f"     * {name:35s} | Accuracy: {m.get('Accuracy')*100:.2f}% | F1-Macro: {m.get('F1_Macro'):.4f} | 5-Fold CV: {m.get('CV_Mean_Accuracy'):.4f}")

    # -------------------------------------------------------------
    # Step 5: Multi-Persona Debate Agent Simulation & Stress Testing
    # -------------------------------------------------------------
    print("\n[Step 5/5] Launching Multi-Persona Debate Agent Harness...")
    coordinator = MultiPersonaDebateCoordinator()
    synthetic_profiles = generate_synthetic_test_profiles()
    
    debate_summaries = []
    for prof in synthetic_profiles:
        # Construct feature vector for synthetic student
        # [OS, Algo, Prog, SE, Net, Elec, Arch, Math, Comm, Hours, LQ, Hack, Code, Speak, LongWork, SelfLearn, Extra, ReadWrite, Memory]
        acad = prof['academic_avg']
        sample_vec = np.array([[
            acad, acad, acad, acad, acad, acad, acad, acad, 80.0,
            prof['hours_worked'], prof['logical_quotient'], prof['hackathons'],
            prof['coding_skills'], prof['public_speaking'],
            1 if prof['can_work_long'] == 'yes' else 0, 1, 1, 3, 3
        ]])
        
        # ML Inference
        class_probs = clf_suite.predict_probabilities(sample_vec)
        salary_pred = reg_suite.predict_with_uncertainty(sample_vec)
        
        # Multi-Persona Debate
        debate_result = coordinator.run_debate(prof, class_probs, salary_pred)
        debate_summaries.append({
            "profile": prof['profile_name'],
            "archetype": prof['archetype'],
            "recommended_track": debate_result['recommended_career_track'],
            "confidence": debate_result['confidence_probability'],
            "uncertainty": debate_result['calibrated_uncertainty'],
            "salary_range": debate_result['expected_salary_range'],
            "override_recommended": debate_result['human_override_status']['override_recommended'],
            "override_reasons": debate_result['human_override_status']['reasons']
        })
        
        print(f"\n  [Debate Run] -> {prof['profile_name']}")
        print(f"    - Consensus Track: {debate_result['recommended_career_track']} (Confidence: {debate_result['confidence_probability']*100:.1f}%)")
        print(f"    - Calibrated Uncertainty: {debate_result['calibrated_uncertainty']*100:.1f}%")
        print(f"    - Expected Compensation: {debate_result['expected_salary_range']}")
        print(f"    - Human Override Required: {'YES ⚠️' if debate_result['human_override_status']['override_recommended'] else 'NO (Autonomous Approval)'}")
        if debate_result['human_override_status']['reasons']:
            for r in debate_result['human_override_status']['reasons']:
                print(f"      * Trigger: {r}")

    # Save comprehensive evaluation results
    full_eval = {
        "regression_benchmarks": reg_results,
        "classification_benchmarks": clf_results,
        "synthetic_profiles_evaluation": debate_summaries
    }
    with open("data/evaluation_results.json", "w") as f:
        json.dump(full_eval, f, indent=2)
    print("\nSaved evaluation results to data/evaluation_results.json")
    
    # Generate and print comparison table
    tables_md = format_markdown_comparison_tables(reg_results, clf_results)
    with open("data/benchmarks_summary.md", "w") as f:
        f.write(tables_md)
    print("Saved benchmark summary to data/benchmarks_summary.md")
    print("\n" + "=" * 80)
    print("  ALL 5 EVALUATION RUBRIC REQUIREMENTS EXECUTED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
