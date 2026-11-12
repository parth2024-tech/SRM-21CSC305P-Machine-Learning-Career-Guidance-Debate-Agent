"""
Generates high-resolution presentation figures specifically tailored for 
widescreen 16:9 PowerPoint slides (6.2" x 5.5" aspect ratio).
Completely eliminates awkward whitespace while maintaining 100% mathematical fidelity.
"""
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

os.makedirs("figures", exist_ok=True)

# ── 1. SLIDE 5: Unit 3 Clustering Comparison (Vertical 3-Panel) ─────────────
def gen_ppt_clustering_comparison():
    fig, axes = plt.subplots(3, 1, figsize=(6.2, 5.5), dpi=300)
    models = ["K-Means", "GMM (EM)", "Hierarchical"]
    colors = ['#1e40af', '#ea580c', '#16a34a']
    
    sil_scores = [0.0482, 0.0431, 0.0115]
    ch_scores = [874.2, 812.5, 645.1]
    db_scores = [2.84, 2.91, 3.42]
    
    # 1. Silhouette
    bars1 = axes[0].bar(models, sil_scores, color=colors, width=0.55, edgecolor='#1e293b', linewidth=1)
    axes[0].set_title("Silhouette Score (Higher is Better)", fontsize=10.5, fontweight='bold', pad=4)
    axes[0].set_ylabel("Score", fontsize=9)
    axes[0].set_ylim(0, 0.062)
    axes[0].bar_label(bars1, fmt='%.4f', padding=3, fontsize=8.5, fontweight='bold')
    axes[0].tick_params(labelsize=8.5)
    
    # 2. Calinski-Harabasz
    bars2 = axes[1].bar(models, ch_scores, color=colors, width=0.55, edgecolor='#1e293b', linewidth=1)
    axes[1].set_title("Calinski-Harabasz Index (Higher is Better)", fontsize=10.5, fontweight='bold', pad=4)
    axes[1].set_ylabel("Variance Ratio", fontsize=9)
    axes[1].set_ylim(0, 1050)
    axes[1].bar_label(bars2, fmt='%.1f', padding=3, fontsize=8.5, fontweight='bold')
    axes[1].tick_params(labelsize=8.5)
    
    # 3. Davies-Bouldin
    bars3 = axes[2].bar(models, db_scores, color=colors, width=0.55, edgecolor='#1e293b', linewidth=1)
    axes[2].set_title("Davies-Bouldin Index (Lower is Better)", fontsize=10.5, fontweight='bold', pad=4)
    axes[2].set_ylabel("Separation", fontsize=9)
    axes[2].set_ylim(0, 4.2)
    axes[2].bar_label(bars3, fmt='%.2f', padding=3, fontsize=8.5, fontweight='bold')
    axes[2].tick_params(labelsize=8.5)
    
    plt.subplots_adjust(hspace=0.42, top=0.94, bottom=0.08, left=0.15, right=0.95)
    out_path = "figures/ppt_unit3_clustering_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Created {out_path}")

# ── 2. SLIDE 6: Unit 4 Markov Transitions (Vertical 2-Panel) ────────────────
def gen_ppt_markov_transitions():
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.5), dpi=300)
    state_labels = ['Junior', 'Mid', 'Senior', 'Lead', 'Dir']
    
    A = np.array([
        [0.35, 0.45, 0.15, 0.05, 0.00],
        [0.05, 0.35, 0.45, 0.12, 0.03],
        [0.02, 0.08, 0.45, 0.35, 0.10],
        [0.01, 0.02, 0.10, 0.52, 0.35],
        [0.00, 0.01, 0.04, 0.15, 0.80]
    ])
    A3 = np.linalg.matrix_power(A, 3)
    
    sns.heatmap(A, annot=True, fmt='.2f', cmap='Oranges', ax=axes[0], cbar=True,
                cbar_kws={'shrink': 0.85},
                xticklabels=state_labels, yticklabels=state_labels, annot_kws={"size": 8.5, "weight": "bold"})
    axes[0].set_title('1-Step Career Transition Matrix (A)', fontsize=10.5, fontweight='bold', pad=4)
    axes[0].set_xlabel('Next State S_{t+1}', fontsize=8.5)
    axes[0].set_ylabel('Current State S_t', fontsize=8.5)
    axes[0].tick_params(labelsize=8)
    
    sns.heatmap(A3, annot=True, fmt='.2f', cmap='Purples', ax=axes[1], cbar=True,
                cbar_kws={'shrink': 0.85},
                xticklabels=state_labels, yticklabels=state_labels, annot_kws={"size": 8.5, "weight": "bold"})
    axes[1].set_title('3-Step Projection Matrix (A^3)', fontsize=10.5, fontweight='bold', pad=4)
    axes[1].set_xlabel('Projected State S_{t+3}', fontsize=8.5)
    axes[1].set_ylabel('Origin State S_t', fontsize=8.5)
    axes[1].tick_params(labelsize=8)
    
    plt.subplots_adjust(hspace=0.45, top=0.94, bottom=0.08, left=0.14, right=0.96)
    out_path = "figures/ppt_unit4_markov_transition_dynamics.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Created {out_path}")

# ── 3. SLIDE 7: Unit 4 Workflow (Vertical 5-Stage Architecture) ─────────────
def gen_ppt_unit4_workflow():
    fig, ax = plt.subplots(figsize=(6.2, 5.5), dpi=300)
    ax.axis('off')
    
    stages = [
        ("1. Longitudinal Candidate History", "Sequential academic & quarterly skill evaluations (t = 1...T)", "#f1f5f9", "#475569"),
        ("2. First-Order Markov Transition Chain", "Transition Matrix A, Multi-Step Powers A^k, Stationary pi*", "#e0f2fe", "#0284c7"),
        ("3. Hidden Markov Model (HMM)", "Scaled Forward-Backward evaluation & Viterbi MAP Decoding", "#ede9fe", "#7c3aed"),
        ("4. Linear Dynamical System (Kalman Filter)", "Optimal state tracking (mu_t, P_t) under noisy evaluation", "#fef3c7", "#d97706"),
        ("5. Multi-Horizon Career Roadmap", "Aspirational milestones, latent velocity & bottleneck alerts", "#dcfce7", "#16a34a")
    ]
    
    n_stages = len(stages)
    box_height = 0.72
    spacing = 0.28
    start_y = 5.2 - box_height
    
    for i, (title, subtitle, bg, border) in enumerate(stages):
        y = start_y - i * (box_height + spacing)
        rect = patches.FancyBboxPatch((0.2, y), 5.8, box_height, boxstyle="round,pad=0.08", fc=bg, ec=border, lw=1.5)
        ax.add_patch(rect)
        ax.text(3.1, y + box_height * 0.65, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color="#0f172a")
        ax.text(3.1, y + box_height * 0.28, subtitle, ha='center', va='center', fontsize=8.0, color="#334155")
        
        if i < n_stages - 1:
            arrow_top = y
            arrow_bottom = y - spacing
            ax.annotate("", xy=(3.1, arrow_bottom + 0.02), xytext=(3.1, arrow_top - 0.02),
                        arrowprops=dict(arrowstyle="->", lw=2.0, color="#64748b"))
            
    ax.set_xlim(0, 6.2)
    ax.set_ylim(0, 5.3)
    plt.tight_layout()
    out_path = "figures/ppt_unit4_workflow.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created {out_path}")

# ── 4. SLIDE 9: Unit 5 Workflow (Vertical 5-Stage Architecture) ─────────────
def gen_ppt_unit5_workflow():
    fig, ax = plt.subplots(figsize=(6.2, 5.5), dpi=300)
    ax.axis('off')
    
    stages = [
        ("1. Standardized Competency Vector x", "Standardized academic, logic & coding feature profile", "#f1f5f9", "#475569"),
        ("2. Single CART Decision Tree", "Greedy recursive binary partitioning on Gini Impurity", "#e0f2fe", "#0284c7"),
        ("3. Sequential Boosting (AdaBoost)", "Iterative sample reweighting exp(alpha_m) on hard cases", "#fef3c7", "#d97706"),
        ("4. Random Forest Bagging", "Bootstrap aggregating + sqrt(D) feature subspace decorrelation", "#ede9fe", "#7c3aed"),
        ("5. Bayesian Model Averaging (BMA)", "Marginal likelihood weighting P(y|x, D) = sum P(y|M_m)P(M_m)", "#dcfce7", "#16a34a")
    ]
    
    n_stages = len(stages)
    box_height = 0.72
    spacing = 0.28
    start_y = 5.2 - box_height
    
    for i, (title, subtitle, bg, border) in enumerate(stages):
        y = start_y - i * (box_height + spacing)
        rect = patches.FancyBboxPatch((0.2, y), 5.8, box_height, boxstyle="round,pad=0.08", fc=bg, ec=border, lw=1.5)
        ax.add_patch(rect)
        ax.text(3.1, y + box_height * 0.65, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color="#0f172a")
        ax.text(3.1, y + box_height * 0.28, subtitle, ha='center', va='center', fontsize=8.0, color="#334155")
        
        if i < n_stages - 1:
            arrow_top = y
            arrow_bottom = y - spacing
            ax.annotate("", xy=(3.1, arrow_bottom + 0.02), xytext=(3.1, arrow_top - 0.02),
                        arrowprops=dict(arrowstyle="->", lw=2.0, color="#64748b"))
            
    ax.set_xlim(0, 6.2)
    ax.set_ylim(0, 5.3)
    plt.tight_layout()
    out_path = "figures/ppt_unit5_workflow.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created {out_path}")

# ── 5. SLIDE 2: Capstone Debate Architecture (Vertical Flow) ────────────────
def gen_ppt_capstone_architecture():
    fig, ax = plt.subplots(figsize=(6.2, 5.5), dpi=300)
    ax.axis('off')
    
    # 1. Top Input
    rect_in = patches.FancyBboxPatch((0.3, 4.4), 5.6, 0.75, boxstyle="round,pad=0.08", fc="#f1f5f9", ec="#475569", lw=1.5)
    ax.add_patch(rect_in)
    ax.text(3.1, 4.88, "Student Candidate Profile Vector x", ha='center', va='center', fontsize=9.5, fontweight='bold', color="#0f172a")
    ax.text(3.1, 4.58, "Unit 2 Bayes Reg + Unit 3 PCA/GMM + Unit 4 LDS + Unit 5 Ensembles", ha='center', va='center', fontsize=7.5, color="#334155")
    
    # 2. Middle Personas (3 columns in middle)
    personas = [
        ("The Planner", "Aspirational\nViterbi Roadmap", 0.3, "#dbeafe", "#2563eb"),
        ("The Critic", "Mahalanobis Gap\n& Bottlenecks", 2.25, "#fef3c7", "#d97706"),
        ("Ethics Sentinel", "Assignment\nEntropy H(gamma)", 4.2, "#fee2e2", "#dc2626")
    ]
    for name, desc, x, bg, border in personas:
        rect = patches.FancyBboxPatch((x, 2.0), 1.7, 1.8, boxstyle="round,pad=0.08", fc=bg, ec=border, lw=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.85, 3.45, name, ha='center', va='center', fontsize=9.0, fontweight='bold', color="#0f172a")
        ax.text(x + 0.85, 2.7, desc, ha='center', va='center', fontsize=8.0, color="#334155")
        
        # Arrows from Input
        ax.annotate("", xy=(x + 0.85, 3.82), xytext=(x + 0.85, 4.38),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#64748b"))
        # Arrows to Arbiter
        ax.annotate("", xy=(x + 0.85, 1.25), xytext=(x + 0.85, 1.98),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#64748b"))
        
    # 3. Bottom Arbiter
    rect_arb = patches.FancyBboxPatch((0.3, 0.2), 5.6, 1.0, boxstyle="round,pad=0.08", fc="#dcfce7", ec="#16a34a", lw=1.5)
    ax.add_patch(rect_arb)
    ax.text(3.1, 0.85, "Dialectical Arbiter Consensus Engine", ha='center', va='center', fontsize=10.0, fontweight='bold', color="#0f172a")
    ax.text(3.1, 0.45, "Calibrated Confidence Score + Human Counselor Escalation Threshold", ha='center', va='center', fontsize=8.0, color="#334155")
    
    ax.set_xlim(0, 6.2)
    ax.set_ylim(0, 5.3)
    plt.tight_layout()
    out_path = "figures/ppt_capstone_debate_architecture.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created {out_path}")

# ── 6. SLIDE 3: PCA Scree Plot (Taller Aspect Ratio) ────────────────────────
def gen_ppt_pca_scree():
    fig, ax = plt.subplots(figsize=(6.2, 5.5), dpi=300)
    eigenvalues = [1.038, 1.030, 1.029, 1.019, 1.019, 1.010]
    cum_var = [7.4, 14.8, 22.1, 29.4, 36.7, 43.9]
    components = [f"PC{i}" for i in range(1, 7)]
    
    bars = ax.bar(components, eigenvalues, color='#2563eb', alpha=0.85, edgecolor='#1e293b', width=0.55, label='Eigenvalue (Variance)')
    ax.axhline(y=1.0, color='#dc2626', linestyle='--', linewidth=2.0, label='Kaiser Cutoff (cutoff = 1.0)')
    
    ax.set_title("Unit 3: PCA Scree Plot & Kaiser Latent Selection", fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel("Principal Component", fontsize=9.5, fontweight='bold')
    ax.set_ylabel("Eigenvalue Magnitude (lambda)", fontsize=9.5, fontweight='bold')
    ax.set_ylim(0, 1.25)
    ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=8.5, fontweight='bold')
    
    # Secondary axis for cumulative variance
    ax2 = ax.twinx()
    ax2.plot(components, cum_var, color='#16a34a', marker='o', linewidth=2.2, markersize=6, label='Cumulative Variance (%)')
    ax2.set_ylabel("Cumulative Variance (%)", fontsize=9.5, fontweight='bold', color='#16a34a')
    ax2.set_ylim(0, 60)
    for i, txt in enumerate(cum_var):
        ax2.annotate(f"{txt:.1f}%", (components[i], cum_var[i] + 2.0), fontsize=8.0, fontweight='bold', color='#15803d', ha='center')
    ax2.grid(False)
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8.5)
    
    plt.subplots_adjust(top=0.92, bottom=0.22, left=0.14, right=0.86)
    out_path = "figures/ppt_unit3_pca_scree.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Created {out_path}")

# ── 7. SLIDE 4: GMM Soft Responsibilities Heatmap ───────────────────────────
def gen_ppt_gmm_heatmap():
    fig, ax = plt.subplots(figsize=(6.2, 5.5), dpi=300)
    np.random.seed(42)
    # 16 representative candidates across 4 archetypes
    probs = np.array([
        [0.84, 0.08, 0.05, 0.03],
        [0.10, 0.72, 0.12, 0.06],
        [0.05, 0.08, 0.81, 0.06],
        [0.02, 0.04, 0.08, 0.86],
        [0.79, 0.11, 0.06, 0.04],
        [0.08, 0.76, 0.10, 0.06],
        [0.06, 0.09, 0.77, 0.08],
        [0.03, 0.05, 0.09, 0.83],
        [0.45, 0.35, 0.12, 0.08],
        [0.12, 0.42, 0.38, 0.08],
        [0.04, 0.14, 0.74, 0.08],
        [0.88, 0.05, 0.04, 0.03],
        [0.07, 0.82, 0.06, 0.05],
        [0.05, 0.07, 0.80, 0.08],
        [0.32, 0.34, 0.22, 0.12],
        [0.03, 0.06, 0.08, 0.83]
    ])
    xticklabels = [f"Cluster {k}\n(gamma_{k})" for k in range(4)]
    yticklabels = [f"Cand #{i+1}" for i in range(16)]
    
    sns.heatmap(probs, annot=True, fmt='.2f', cmap='Blues', ax=ax, cbar=True,
                cbar_kws={'shrink': 0.85},
                xticklabels=xticklabels, yticklabels=yticklabels, annot_kws={"size": 7.5, "weight": "bold"})
    ax.set_title("GMM Posterior Responsibilities (gamma_nk = P(Z_k=1 | x_n))", fontsize=10.5, fontweight='bold', pad=8)
    ax.set_xlabel("Latent Mixture Archetypes", fontsize=9.0, fontweight='bold')
    ax.set_ylabel("Candidate Index", fontsize=9.0, fontweight='bold')
    ax.tick_params(labelsize=8)
    
    plt.subplots_adjust(top=0.92, bottom=0.12, left=0.18, right=0.96)
    out_path = "figures/ppt_unit3_gmm_responsibilities_heatmap.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Created {out_path}")

# ── 8. SLIDE 8: Kalman Filter (LDS Tracking) ────────────────────────────────
def gen_ppt_kalman_filter():
    fig, ax = plt.subplots(figsize=(6.2, 5.5), dpi=300)
    quarters = list(range(1, 7))
    noisy = [65.0, 68.5, 71.0, 75.0, 79.5, 84.0]
    filtered = [63.2, 67.1, 70.4, 74.8, 79.1, 83.5]
    uncertainty = [3.2, 2.6, 2.2, 1.9, 1.7, 1.5]
    
    ax.plot(quarters, noisy, 'ro--', label='Noisy Semester Assessment (z_t)', alpha=0.75, linewidth=1.5, markersize=6)
    ax.plot(quarters, filtered, 'b-', linewidth=2.5, label='Kalman Optimal State (mu_t)', marker='s', markersize=5)
    ax.fill_between(
        quarters,
        np.array(filtered) - np.array(uncertainty),
        np.array(filtered) + np.array(uncertainty),
        color='blue', alpha=0.15, label='State Uncertainty Envelope (+- 1 sigma)'
    )
    
    ax.set_title("Unit 4: Discrete Kalman Filter Skill Tracking", fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel("Career Evaluation Quarter (t)", fontsize=9.5, fontweight='bold')
    ax.set_ylabel("Estimated Skill Competency", fontsize=9.5, fontweight='bold')
    ax.set_ylim(55, 92)
    ax.legend(loc='lower right', fontsize=8.5)
    ax.tick_params(labelsize=9)
    
    # Annotate final growth velocity
    ax.annotate("Final Proficiency: 83.5\nVelocity: +4.10 / Qtr",
                xy=(6, 83.5), xytext=(4.0, 86.5),
                arrowprops=dict(facecolor='black', shrink=0.08, width=1.2, headwidth=6),
                fontsize=8.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#dcfce7", ec="#16a34a"))
    
    plt.subplots_adjust(top=0.92, bottom=0.12, left=0.14, right=0.95)
    out_path = "figures/ppt_unit4_kalman_filter.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Created {out_path}")

# ── 9. SLIDE 11: Unit 5 Ensemble Benchmark ──────────────────────────────────
def gen_ppt_ensemble_comparison():
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 5.5), dpi=300)
    models = ["CART", "AdaBoost", "Random Forest", "GBDT"]
    colors = ['#94a3b8', '#38bdf8', '#2563eb', '#16a34a']
    
    # Accuracy, Precision, Recall, Macro F1
    acc = [0.278, 0.282, 0.287, 0.265]
    prec = [0.245, 0.124, 0.349, 0.204]
    rec = [0.278, 0.282, 0.287, 0.265]
    f1 = [0.118, 0.093, 0.091, 0.139]
    
    # Subplot (0,0): Accuracy
    b0 = axes[0, 0].bar(models, acc, color=colors, width=0.6, edgecolor='#1e293b')
    axes[0, 0].set_title("Classification Accuracy", fontsize=9.5, fontweight='bold')
    axes[0, 0].set_ylim(0, 0.38)
    axes[0, 0].bar_label(b0, fmt='%.3f', padding=2, fontsize=7.5, fontweight='bold')
    axes[0, 0].tick_params(labelsize=7.5)
    
    # Subplot (0,1): Precision
    b1 = axes[0, 1].bar(models, prec, color=colors, width=0.6, edgecolor='#1e293b')
    axes[0, 1].set_title("Weighted Precision", fontsize=9.5, fontweight='bold')
    axes[0, 1].set_ylim(0, 0.44)
    axes[0, 1].bar_label(b1, fmt='%.3f', padding=2, fontsize=7.5, fontweight='bold')
    axes[0, 1].tick_params(labelsize=7.5)
    
    # Subplot (1,0): Recall
    b2 = axes[1, 0].bar(models, rec, color=colors, width=0.6, edgecolor='#1e293b')
    axes[1, 0].set_title("Weighted Recall", fontsize=9.5, fontweight='bold')
    axes[1, 0].set_ylim(0, 0.38)
    axes[1, 0].bar_label(b2, fmt='%.3f', padding=2, fontsize=7.5, fontweight='bold')
    axes[1, 0].tick_params(labelsize=7.5)
    
    # Subplot (1,1): Macro F1
    b3 = axes[1, 1].bar(models, f1, color=colors, width=0.6, edgecolor='#1e293b')
    axes[1, 1].set_title("Macro F1-Score", fontsize=9.5, fontweight='bold')
    axes[1, 1].set_ylim(0, 0.19)
    axes[1, 1].bar_label(b3, fmt='%.3f', padding=2, fontsize=7.5, fontweight='bold')
    axes[1, 1].tick_params(labelsize=7.5)
    
    plt.subplots_adjust(hspace=0.45, wspace=0.35, top=0.92, bottom=0.08, left=0.12, right=0.96)
    out_path = "figures/ppt_unit5_ensemble_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Created {out_path}")

if __name__ == "__main__":
    print("Generating PPT-optimized figures (6.2\" x 5.5\" aspect ratio)...")
    gen_ppt_clustering_comparison()
    gen_ppt_markov_transitions()
    gen_ppt_unit4_workflow()
    gen_ppt_unit5_workflow()
    gen_ppt_capstone_architecture()
    gen_ppt_pca_scree()
    gen_ppt_gmm_heatmap()
    gen_ppt_kalman_filter()
    gen_ppt_ensemble_comparison()
    print("All 9 presentation figures generated successfully!")
