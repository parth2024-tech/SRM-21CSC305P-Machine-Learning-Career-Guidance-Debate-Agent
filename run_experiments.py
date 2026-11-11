"""
Master Experiment Runner for Unit 3 & Unit 4
Executes:
1. Data loading and summary statistics
2. Unit 3 Unsupervised Learning Pipeline:
   - PCA (eigenvalues, scree plot, Kaiser criterion)
   - Factor Analysis (latent factor loadings, noise variance)
   - K-Means (Elbow method, WCSS, Silhouette)
   - Gaussian Mixture Models (EM soft clustering, responsibilities, BIC/AIC)
   - Hierarchical Clustering (Agglomerative, Ward linkage)
   - Dissimilarity analysis (Euclidean vs Mahalanobis)
3. Unit 4 Sequential & Markov Pipeline:
   - Markov Chain Career Transitions (A, A^k multi-step, Stationary distribution pi*)
   - Hidden Markov Model (Forward-Backward with scaling factors, Viterbi decoding)
   - Linear Dynamical System / Kalman Filter (trajectory tracking, Kalman gain)
4. Multi-Persona Dialectical Debate Engine synthesis
5. Generates publication-grade evaluation charts
"""

import json
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data_loader import load_raw_dataset
from src.debate_agent import MultiPersonaDebateEngine
from src.unit3_unsupervised import Unit3UnsupervisedPipeline
from src.unit4_sequential import (
    CareerHiddenMarkovModel,
    CareerKalmanFilter,
    CareerMarkovChain,
)


def run_all_experiments():
    print("=" * 70)
    print("🚀 STARTING UNIT 3 & UNIT 4 EXPERIMENTATION PIPELINE")
    print("=" * 70)
    
    os.makedirs("figures", exist_ok=True)
    
    # 1. Dataset Loading
    df = load_raw_dataset("data/career_guidance_dataset.csv")
    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    numeric_cols = [
        "Acedamic percentage in Operating Systems",
        "percentage in Algorithms",
        "Percentage in Programming Concepts",
        "Percentage in Software Engineering",
        "Percentage in Computer Networks",
        "Percentage in Electronics Subjects",
        "Percentage in Computer Architecture",
        "Percentage in Mathematics",
        "Percentage in Communication skills",
        "Hours working per day",
        "Logical quotient rating",
        "hackathons",
        "coding skills rating",
        "public speaking points"
    ]
    
    X_raw = df[numeric_cols].values
    # Standardize features
    mean_vec = np.mean(X_raw, axis=0)
    std_vec = np.std(X_raw, axis=0) + 1e-12
    X_scaled = (X_raw - mean_vec) / std_vec
    
    # 2. UNIT 3: UNSUPERVISED LEARNING
    print("\n--- [UNIT 3] Running PCA, Factor Analysis, Clustering & GMM ---")
    u3 = Unit3UnsupervisedPipeline(random_state=42)
    
    # PCA
    X_pca, pca_stats = u3.fit_transform_pca(X_scaled, n_components=6)
    print(f"PCA Kaiser Criterion (>1.0): {pca_stats['kaiser_criterion_count']} components retained.")
    print(f"Total Variance Explained (6 components): {pca_stats['total_variance_explained']:.2%}")
    
    # Factor Analysis
    X_fa, fa_stats = u3.fit_factor_analysis(X_scaled, n_factors=3)
    print(f"Factor Analysis Log-Likelihood: {fa_stats['log_likelihood']:.2f}")
    
    # K-Means & Elbow
    wcss = u3.run_kmeans_elbow(X_scaled, k_range=range(2, 8))
    km_labels, km_stats = u3.fit_kmeans(X_scaled, n_clusters=4)
    print(f"K-Means (k=4) Silhouette Score: {km_stats['silhouette_score']:.4f}")
    
    # GMM with EM
    gmm_labels, gmm_probs, gmm_stats = u3.fit_gmm_em(X_scaled, n_components=4)
    print(f"GMM EM Converged in {gmm_stats['n_iter']} iterations. BIC: {gmm_stats['bic']:.1f}, Mean Entropy: {gmm_stats['mean_assignment_entropy']:.4f}")
    
    # Hierarchical Clustering
    hc_labels, hc_stats = u3.fit_hierarchical(X_scaled, n_clusters=4, linkage='ward')
    print(f"Hierarchical (Ward) Silhouette Score: {hc_stats['silhouette_score']:.4f}")
    
    # 3. UNIT 4: SEQUENTIAL DATA & MARKOV MODELS
    print("\n--- [UNIT 4] Running Markov Chains, HMM, and Kalman Filter ---")
    # Markov Chain
    mc = CareerMarkovChain()
    pi_star = mc.compute_stationary_distribution()
    step3_pred = mc.multi_step_prediction(initial_state_idx=0, steps=3)
    print(f"Markov Chain Stationary Distribution: {np.round(pi_star, 3)}")
    print(f"Junior Multi-step (3-yr) Transition: Most likely role -> {step3_pred['most_likely_role']}")
    
    # HMM Forward-Backward and Viterbi Decoding
    hmm = CareerHiddenMarkovModel()
    sample_obs = [0, 1, 1, 2, 2] # Obs: Academic -> Project -> Project -> Innovation -> Innovation
    log_lik, gamma_posteriors, c_scales = hmm.forward_backward_scaled(sample_obs)
    viterbi_states, viterbi_log_prob = hmm.viterbi_decode(sample_obs)
    print(f"HMM Log Likelihood (Scaled Forward-Backward): {log_lik:.3f}")
    print(f"HMM Viterbi Decoded Path: {' -> '.join(viterbi_states)} (Score: {viterbi_log_prob:.3f})")
    
    # Kalman Filter
    kf = CareerKalmanFilter(dt=1.0, q_var=0.08, r_var=0.40)
    noisy_ratings = [65.0, 68.5, 71.0, 75.0, 79.5, 84.0]
    kf_res = kf.filter_trajectory(noisy_ratings, init_proficiency=62.0)
    print(f"Kalman Filter Estimated Proficiency: {kf_res['final_estimated_proficiency']:.2f} (Velocity: +{kf_res['final_estimated_growth_velocity']:.2f})")
    
    # 4. MULTI-PERSONA DEBATE ENGINE
    print("\n--- [DEBATE ENGINE] Executing Dialectical Persona Consensus ---")
    engine = MultiPersonaDebateEngine(entropy_threshold=1.25, mahalanobis_threshold=4.0)
    
    # Calculate dissimilarity for student 0
    dissimilarities = u3.compute_dissimilarity_profiles(X_scaled[0])
    student0_entropy = float(-np.sum(gmm_probs[0] * np.log(gmm_probs[0] + 1e-12)))
    
    debate_result = engine.execute_debate(
        student_id="STUDENT_42_CANDIDATE",
        student_profile={"target_role": df["Suggested Job Role"].iloc[0]},
        unit3_metrics={
            "primary_cluster": int(gmm_labels[0]),
            "soft_responsibilities": gmm_probs[0].tolist(),
            "assignment_entropy": student0_entropy,
            "mahalanobis_distance": dissimilarities.get(f"Cluster_{gmm_labels[0]}_Mahalanobis", 2.1)
        },
        unit4_metrics={
            "viterbi_path": viterbi_states,
            "viterbi_log_prob": viterbi_log_prob,
            "markov_multi_step": step3_pred,
            "kalman_filter": kf_res
        }
    )
    print(f"Debate Arbiter Verdict: {debate_result['consensus_synthesis']['recommendation']}")
    print(f"Confidence Score: {debate_result['consensus_synthesis']['confidence_score']:.2%}")
    
    # 5. VISUALIZATIONS GENERATION
    print("\n--- Generating Publication Charts for PBL Presentation ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Figure 1: Unit 3 PCA Scree Plot & Kaiser Criterion
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(1, 7), pca_stats['eigenvalues'], color='#2b5c8f', alpha=0.85, label='Eigenvalues (Variance)')
    ax.axhline(y=1.0, color='#d9534f', linestyle='--', linewidth=2, label='Kaiser Criterion (cutoff = 1.0)')
    ax.plot(range(1, 7), pca_stats['cumulative_variance_ratio'], color='#2ca02c', marker='o', linewidth=2, label='Cumulative Variance')
    ax.set_title("Unit 3: PCA Scree Plot & Kaiser Latent Dimension Selection", fontsize=12, fontweight='bold')
    ax.set_xlabel("Principal Component Index")
    ax.set_ylabel("Eigenvalue / Cumulative Ratio")
    ax.legend(loc='center right')
    plt.tight_layout()
    plt.savefig("figures/unit3_pca_scree.png", dpi=300)
    plt.close()
    
    # Figure 2: Unit 3 Clustering Comparison (K-Means vs GMM vs Hierarchical)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    models = ["K-Means", "GMM (EM)", "Hierarchical"]
    sil_scores = [km_stats['silhouette_score'], gmm_stats['silhouette_score'], hc_stats['silhouette_score']]
    ch_scores = [km_stats['calinski_harabasz_index'], gmm_stats['calinski_harabasz_index'], hc_stats['calinski_harabasz_index']]
    db_scores = [km_stats['davies_bouldin_index'], gmm_stats['davies_bouldin_index'], hc_stats['davies_bouldin_index']]
    
    axes[0].bar(models, sil_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0].set_title("Silhouette Score (Higher is Better)", fontweight='bold')
    axes[0].set_ylabel("Silhouette Score")
    
    axes[1].bar(models, ch_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1].set_title("Calinski-Harabasz Index (Higher is Better)", fontweight='bold')
    axes[1].set_ylabel("Variance Ratio Criterion")
    
    axes[2].bar(models, db_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[2].set_title("Davies-Bouldin Index (Lower is Better)", fontweight='bold')
    axes[2].set_ylabel("Separation Measure")
    
    plt.tight_layout()
    plt.savefig("figures/unit3_clustering_comparison.png", dpi=300)
    plt.close()
    
    # Figure 3: Unit 4 Kalman Filter Tracking
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(kf_res['time_steps'], kf_res['noisy_measurements'], 'ro--', label='Noisy Assessments (z_t)', alpha=0.7)
    ax.plot(kf_res['time_steps'], kf_res['filtered_proficiencies'], 'b-', linewidth=2.5, label='Kalman Optimal Estimate (mu_t)')
    ax.fill_between(
        kf_res['time_steps'],
        np.array(kf_res['filtered_proficiencies']) - np.sqrt(kf_res['estimation_variances']),
        np.array(kf_res['filtered_proficiencies']) + np.sqrt(kf_res['estimation_variances']),
        color='blue', alpha=0.15, label='Estimation Uncertainty (+- 1 sigma)'
    )
    ax.set_title("Unit 4: Linear Dynamical System (Kalman Filter) Skill Tracking", fontsize=12, fontweight='bold')
    ax.set_xlabel("Career Quarter")
    ax.set_ylabel("Estimated Skill Competence")
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/unit4_kalman_filter.png", dpi=300)
    plt.close()
    
    # Save combined experiment report
    experiment_results = {
        "unit3": {
            "pca": pca_stats,
            "factor_analysis": fa_stats,
            "kmeans": {k: v for k, v in km_stats.items() if k != "cluster_centers"},
            "gmm": {k: v for k, v in gmm_stats.items() if k != "weights"},
            "hierarchical": hc_stats
        },
        "unit4": {
            "markov_stationary_distribution": pi_star.tolist(),
            "markov_multi_step_prediction": step3_pred,
            "hmm_viterbi_states": viterbi_states,
            "hmm_viterbi_log_prob": viterbi_log_prob,
            "hmm_scaled_log_likelihood": log_lik,
            "kalman_final_proficiency": kf_res['final_estimated_proficiency'],
            "kalman_growth_velocity": kf_res['final_estimated_growth_velocity']
        },
        "debate_synthesis": debate_result
    }
    
    with open("figures/experiment_summary.json", "w") as f:
        json.dump(experiment_results, f, indent=2)
        
    print("✅ All Unit 3 & Unit 4 experiments and figures completed successfully!")

if __name__ == "__main__":
    run_all_experiments()
