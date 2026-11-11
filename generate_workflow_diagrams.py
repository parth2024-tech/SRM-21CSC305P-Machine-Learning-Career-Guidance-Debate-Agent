"""
Generates publication-quality workflow and architecture diagrams for the SRM Project Report.
Matches Chapters 2, 3 (Unit 3 & Unit 4), 4, and 5 requirements.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_unit2_workflow():
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=200)
    ax.axis('off')
    
    boxes = [
        ("Raw Kaggle Dataset\n(20k Records, 39 Features)", "#e0f2fe", "#0284c7"),
        ("Data Cleaning &\nZ-Score Scaling\nz = (x - μ) / σ", "#f1f5f9", "#475569"),
        ("Parametric Moment\nEstimation & Quantiles\n(Mean, Var, Q25, Q50, Q75)", "#fef3c7", "#d97706"),
        ("Unit 2 Supervised\nModelling Suite", "#ede9fe", "#7c3aed"),
        ("Predictive Outputs\n• Bayesian Salary (95% CI)\n• Career Track (RBF SVM)", "#dcfce7", "#16a34a")
    ]
    
    for i, (text, bg, border) in enumerate(boxes):
        x = i * 2.05 + 0.1
        rect = patches.FancyBboxPatch((x, 0.4), 1.75, 1.2, boxstyle="round,pad=0.1", fc=bg, ec=border, lw=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.875, 1.0, text, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#0f172a')
        
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + 1.85, 1.0), xytext=(x + 2.05, 1.0),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color="#334155"))
            
    ax.set_xlim(0, 10.3)
    ax.set_ylim(0, 2.0)
    plt.tight_layout()
    plt.savefig("figures/unit2_workflow.png", bbox_inches='tight')
    plt.close()
    print("Created unit2_workflow.png")

def create_unit3_workflow():
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=200)
    ax.axis('off')
    
    boxes = [
        ("Standardized Features\n(14 Competencies)", "#f1f5f9", "#475569"),
        ("Principal Component\nAnalysis (PCA)\n(Kaiser Cutoff: λ > 1.0)", "#e0f2fe", "#0284c7"),
        ("6 Orthogonal Latent\nComponents\n(43.89% Variance)", "#fef3c7", "#d97706"),
        ("Gaussian Mixture\nModel (EM Algorithm)\nK = 4 Archetypes", "#ede9fe", "#7c3aed"),
        ("Soft Clustering &\nUncertainty Metrics\n• Responsibilities γ_nk\n• Entropy H(γ) & Mahal. D_M", "#dcfce7", "#16a34a")
    ]
    
    for i, (text, bg, border) in enumerate(boxes):
        x = i * 2.05 + 0.1
        rect = patches.FancyBboxPatch((x, 0.4), 1.75, 1.2, boxstyle="round,pad=0.1", fc=bg, ec=border, lw=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.875, 1.0, text, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#0f172a')
        
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + 1.85, 1.0), xytext=(x + 2.05, 1.0),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color="#334155"))
            
    ax.set_xlim(0, 10.3)
    ax.set_ylim(0, 2.0)
    plt.tight_layout()
    plt.savefig("figures/unit3_workflow.png", bbox_inches='tight')
    plt.close()
    print("Created unit3_workflow.png")

def create_unit4_workflow():
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=200)
    ax.axis('off')
    
    boxes = [
        ("Longitudinal Candidate\nSemester History", "#f1f5f9", "#475569"),
        ("First-Order Markov Chain\n• Transition Matrix A\n• Multi-Step Powers A^k\n• Stationary Eq. π*", "#e0f2fe", "#0284c7"),
        ("Hidden Markov Model\n• Scaled Fwd-Bkwd\n• Viterbi MAP Decoding:\nNovice → Mid → Expert", "#ede9fe", "#7c3aed"),
        ("Linear Dynamical System\nDiscrete Kalman Filter\n• Predict: μ_t^-, P_t^-\n• Update: K_t, μ_t, P_t", "#fef3c7", "#d97706"),
        ("Sequential Outputs\n• Trajectory Roadmap\n• Latent Velocity v_t\n• Bottleneck Alerts", "#dcfce7", "#16a34a")
    ]
    
    for i, (text, bg, border) in enumerate(boxes):
        x = i * 2.05 + 0.1
        rect = patches.FancyBboxPatch((x, 0.4), 1.75, 1.2, boxstyle="round,pad=0.1", fc=bg, ec=border, lw=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.875, 1.0, text, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#0f172a')
        
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + 1.85, 1.0), xytext=(x + 2.05, 1.0),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color="#334155"))
            
    ax.set_xlim(0, 10.3)
    ax.set_ylim(0, 2.0)
    plt.tight_layout()
    plt.savefig("figures/unit4_workflow.png", bbox_inches='tight')
    plt.close()
    print("Created unit4_workflow.png")

def create_unit5_workflow():
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=200)
    ax.axis('off')
    
    boxes = [
        ("Standardized Profile\nCompetency Vector x", "#f1f5f9", "#475569"),
        ("CART Decision Trees\nBinary Splitting on\nGini Impurity I_G(t)", "#e0f2fe", "#0284c7"),
        ("Sequential Boosting\nAdaBoost Sample\nReweighting exp(α_m)", "#fef3c7", "#d97706"),
        ("Random Forest\nBagging + √D Feature\nSubspace Decorrelation", "#ede9fe", "#7c3aed"),
        ("Bayesian Model Averaging\nP(y|x, D) = Σ P(y|M_m)P(M_m)\nConfidence Weights", "#dcfce7", "#16a34a")
    ]
    
    for i, (text, bg, border) in enumerate(boxes):
        x = i * 2.05 + 0.1
        rect = patches.FancyBboxPatch((x, 0.4), 1.75, 1.2, boxstyle="round,pad=0.1", fc=bg, ec=border, lw=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.875, 1.0, text, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#0f172a')
        
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + 1.85, 1.0), xytext=(x + 2.05, 1.0),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color="#334155"))
            
    ax.set_xlim(0, 10.3)
    ax.set_ylim(0, 2.0)
    plt.tight_layout()
    plt.savefig("figures/unit5_workflow.png", bbox_inches='tight')
    plt.close()
    print("Created unit5_workflow.png")

def create_capstone_architecture():
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=200)
    ax.axis('off')
    
    # Input
    rect_in = patches.FancyBboxPatch((0.1, 1.6), 2.0, 1.2, boxstyle="round,pad=0.1", fc="#f1f5f9", ec="#475569", lw=1.5)
    ax.add_patch(rect_in)
    ax.text(1.1, 2.2, "Candidate Vector x\n+ 5-Unit ML Outputs\n(Bayes Reg, PCA, GMM,\nHMM, Kalman, Ensembles)", ha='center', va='center', fontsize=8.5, fontweight='bold')
    
    # Personas
    personas = [
        ("Persona 1: The Planner\n• Unit 2: Salary Target μ_N\n• Unit 3: GMM Membership γ_nk\n• Unit 4: Viterbi MAP Trajectory\nStance: Optimistic Growth", 2.8, "#dbeafe", "#2563eb"),
        ("Persona 2: The Critic\n• Unit 3: Mahalanobis Distance D_M\n• Unit 4: Markov Bottlenecks\n• Unit 4: Kalman Covariance P_t\nStance: Feasibility & Risk Audit", 1.6, "#fef3c7", "#d97706"),
        ("Persona 3: Ethics Sentinel\n• Unit 3: Assignment Entropy H(γ)\n• Unit 4: Absorbing State Trap\n• Fairness & Demographic Parity\nStance: Calibrated Governance", 0.4, "#fee2e2", "#dc2626")
    ]
    
    for text, y, bg, border in personas:
        rect = patches.FancyBboxPatch((2.7, y), 3.2, 1.0, boxstyle="round,pad=0.1", fc=bg, ec=border, lw=1.5)
        ax.add_patch(rect)
        ax.text(4.3, y + 0.5, text, ha='center', va='center', fontsize=8, fontweight='bold', color="#0f172a")
        
        ax.annotate("", xy=(2.2, 2.2), xytext=(2.7, y + 0.5),
                    arrowprops=dict(arrowstyle="<-", lw=1.5, color="#64748b"))
        ax.annotate("", xy=(6.0, y + 0.5), xytext=(6.5, 2.2),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#64748b"))
        
    # Arbiter
    rect_arb = patches.FancyBboxPatch((6.6, 1.6), 3.2, 1.2, boxstyle="round,pad=0.1", fc="#dcfce7", ec="#16a34a", lw=1.5)
    ax.add_patch(rect_arb)
    ax.text(8.2, 2.2, "Dialectical Arbiter Consensus\n• Evaluates H(γ) vs Cutoff τ_entropy\n• Checks D_M vs Cutoff τ_mahal\n• Computes Calibrated Confidence\n• Triggers Counselor Escalation", ha='center', va='center', fontsize=8.5, fontweight='bold', color="#0f172a")
    
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0.2, 4.2)
    plt.tight_layout()
    plt.savefig("figures/capstone_debate_architecture.png", bbox_inches='tight')
    plt.close()
    print("Created capstone_debate_architecture.png")

if __name__ == "__main__":
    create_unit2_workflow()
    create_unit3_workflow()
    create_unit4_workflow()
    create_unit5_workflow()
    create_capstone_architecture()
