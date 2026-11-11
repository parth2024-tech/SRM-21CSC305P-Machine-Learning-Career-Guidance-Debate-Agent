"""
Exploratory Data Analysis & Insights Module (Rubric 2)
Generates publication-quality visualizations and quantitative statistical insights:
  - Correlation matrix of technical & cognitive metrics
  - Univariate distribution of continuous features (KDE, Mean, Variance, Quantiles)
  - Career Track discrete distributions & balance
  - Skills radar / boxplot profile by Career Domain
  - Regression trendlines for Expected Starting Compensation
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8


def generate_eda_visualizations(df_processed: pd.DataFrame, output_dir: str = "figures") -> dict[str, str]:
    """Generates and saves all 5 key EDA figures for academic presentation."""
    os.makedirs(output_dir, exist_ok=True)
    generated_figures = {}
    
    # 1. Pearson Correlation Heatmap (Unit 1: Covariance & Correlation)
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    core_features = [
        'Acedamic percentage in Operating Systems',
        'percentage in Algorithms',
        'Percentage in Programming Concepts',
        'Percentage in Computer Networks',
        'Percentage in Mathematics',
        'Hours working per day',
        'Logical quotient rating',
        'hackathons',
        'coding skills rating',
        'public speaking points',
        'Career_Readiness_Score',
        'Expected_Salary_k'
    ]
    # Clean display names
    clean_labels = [c.replace('Acedamic percentage in ', '').replace('Percentage in ', '').replace('percentage in ', '') for c in core_features]
    corr = df_processed[core_features].corr()
    
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar_kws={'label': 'Pearson Correlation (r)'},
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        ax=ax1,
        linewidths=0.5
    )
    ax1.set_title("Unit 1 & 2: Correlation Matrix of Academic & Technical Features", fontsize=13, pad=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    fig1_path = os.path.join(output_dir, "fig1_correlation_matrix.png")
    fig1.tight_layout()
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)
    generated_figures["correlation_matrix"] = fig1_path

    # 2. Continuous Variables Distribution & Quantiles
    fig2, axes = plt.subplots(2, 2, figsize=(12, 9))
    continuous_samples = [
        ('Career_Readiness_Score', 'Career Readiness Composite Score', axes[0, 0], '#2b5c8f'),
        ('Expected_Salary_k', 'Expected Starting Salary ($k)', axes[0, 1], '#2a9d8f'),
        ('coding skills rating', 'Coding Skills Rating (1-9)', axes[1, 0], '#e76f51'),
        ('Logical quotient rating', 'Logical Quotient Rating (1-9)', axes[1, 1], '#7209b7')
    ]
    for col, title, ax, color in continuous_samples:
        data = df_processed[col]
        sns.histplot(data, kde=True, ax=ax, color=color, stat="density", bins=20, alpha=0.6)
        mean_val = data.mean()
        q25, q75 = data.quantile(0.25), data.quantile(0.75)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
        ax.axvline(q25, color='orange', linestyle=':', linewidth=1.2, label=f'Q25: {q25:.2f}')
        ax.axvline(q75, color='orange', linestyle=':', linewidth=1.2, label=f'Q75: {q75:.2f}')
        ax.set_title(f"Distribution & Quantiles: {title}", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.set_ylabel("Probability Density")
    fig2_path = os.path.join(output_dir, "fig2_feature_distributions.png")
    fig2.tight_layout()
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)
    generated_figures["feature_distributions"] = fig2_path

    # 3. Discrete Target Distribution across Career Tracks
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    track_counts = df_processed['Career_Track'].value_counts()
    bars = ax3.barh(track_counts.index, track_counts.values, color=['#1d3557', '#457b9d', '#a8dadc', '#e63946', '#f4a261'], edgecolor='#333333')
    ax3.set_title("Discrete Probability Distribution P(Career Track)", fontsize=13, pad=12)
    ax3.set_xlabel("Sample Count")
    for bar in bars:
        w = bar.get_width()
        pct = (w / len(df_processed)) * 100
        ax3.text(w + 100, bar.get_y() + bar.get_height()/2, f"{w} ({pct:.1f}%)", va='center', fontsize=9)
    ax3.set_xlim(0, max(track_counts.values) * 1.15)
    fig3_path = os.path.join(output_dir, "fig3_career_track_distribution.png")
    fig3.tight_layout()
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)
    generated_figures["career_tracks"] = fig3_path

    # 4. Skill Profiles across Strategic Career Tracks
    fig4, ax4 = plt.subplots(figsize=(11, 6))
    sns.boxplot(
        data=df_processed,
        x='Career_Track',
        y='Career_Readiness_Score',
        hue='Career_Track',
        palette='Set2',
        legend=False,
        ax=ax4
    )
    ax4.set_title("Career Readiness Distribution across Specialized Career Tracks", fontsize=13, pad=12)
    ax4.tick_params(axis='x', rotation=15)
    ax4.set_ylabel("Readiness Score (Index)")
    fig4_path = os.path.join(output_dir, "fig4_skills_by_career.png")
    fig4.tight_layout()
    fig4.savefig(fig4_path, dpi=300)
    plt.close(fig4)
    generated_figures["skills_by_career"] = fig4_path

    # 5. Regression Relationship: Salary vs Readiness with Trendline
    fig5, ax5 = plt.subplots(figsize=(9, 6))
    sns.regplot(
        data=df_processed.sample(1500, random_state=42),
        x='Career_Readiness_Score',
        y='Expected_Salary_k',
        scatter_kws={'alpha': 0.3, 'color': '#1d3557', 's': 20},
        line_kws={'color': '#e63946', 'linewidth': 2, 'label': 'Linear Regression OLS Fit'},
        ax=ax5
    )
    ax5.set_title("Linear Model for Regression: Starting Salary vs. Career Readiness", fontsize=13, pad=12)
    ax5.set_xlabel("Career Readiness Composite Index")
    ax5.set_ylabel("Expected Starting Salary ($k / annum)")
    ax5.legend(loc='upper left')
    fig5_path = os.path.join(output_dir, "fig5_salary_vs_skills_regression.png")
    fig5.tight_layout()
    fig5.savefig(fig5_path, dpi=300)
    plt.close(fig5)
    generated_figures["salary_regression"] = fig5_path

    return generated_figures
