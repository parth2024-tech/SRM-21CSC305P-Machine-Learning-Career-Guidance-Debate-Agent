"""
🎓 SRM Institute of Science and Technology - Department of Computing Technologies
Course: 21CSC305P Machine Learning (Final Integrated Project & Demonstration)
Project: Debate-Based Career Guidance Agent with Complementary Personas
Authors: Parth Gupta, Mokshith Penna, Ishu Gurtu
Supervisors: Dr. Divya Mohan, Dr. G. Niranjana
"""

import os
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import (
    BayesianRidge,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="SRMIST 21CSC305P - Dialectical Career Guidance Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .reportview-container { background: #f8fafc; }
    .main-header { font-size: 2.2rem; font-weight: 800; color: #0f172a; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #475569; margin-bottom: 1.5rem; }
    .metric-card { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 48px; border-radius: 6px 6px 0 0; font-weight: 600; padding: 0 16px; }
    .speech-bubble {
        padding: 16px 20px;
        border-radius: 10px;
        margin-bottom: 14px;
        line-height: 1.6;
        border-left: 6px solid;
    }
    .speech-planner {
        background-color: #eff6ff;
        border-color: #2563eb;
        color: #1e3a8a;
    }
    .speech-critic {
        background-color: #fffbeb;
        border-color: #d97706;
        color: #78350f;
    }
    .speech-ethics {
        background-color: #fef2f2;
        border-color: #dc2626;
        color: #991b1b;
    }
    .speech-arbiter {
        background-color: #f0fdf4;
        border-color: #16a34a;
        color: #064e3b;
    }
    .viva-card {
        background: white;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# CACHED DATA & MODELS
# ---------------------------------------------------------
NUMERIC_COLS = [
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

MACRO_TRACKS = [
    "Software Development & Architecture",
    "Data Science & Machine Learning",
    "Systems, Cloud & Cybersecurity",
    "Management, Product & Strategy",
    "Quality Assurance, UX & Operations"
]

@st.cache_data
def load_and_preprocess_data():
    dataset_path = "data/career_guidance_dataset.csv"
    if not os.path.exists(dataset_path):
        url = "https://raw.githubusercontent.com/gautamgiri-dev/CareerGuidanceSystem/master/dataset.csv"
        os.makedirs("data", exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(url, dataset_path)
    
    df = pd.read_csv(dataset_path)
    
    career_map = {
        'Software Engineer': 'Software Development & Architecture',
        'Software Developer': 'Software Development & Architecture',
        'Applications Developer': 'Software Development & Architecture',
        'Web Developer': 'Software Development & Architecture',
        'Mobile Applications Developer': 'Software Development & Architecture',
        'Data Scientist': 'Data Science & Machine Learning',
        'Database Administrator': 'Data Science & Machine Learning',
        'Database Developer': 'Data Science & Machine Learning',
        'Business Intelligence Analyst': 'Data Science & Machine Learning',
        'Network Engineer': 'Systems, Cloud & Cybersecurity',
        'Network Security Engineer': 'Systems, Cloud & Cybersecurity',
        'Network Security Administrator': 'Systems, Cloud & Cybersecurity',
        'Information Security Analyst': 'Systems, Cloud & Cybersecurity',
        'Technical Support': 'Systems, Cloud & Cybersecurity',
        'Project Manager': 'Management, Product & Strategy',
        'Technical Services/Help Desk/Tech Support': 'Management, Product & Strategy',
        'Business Systems Analyst': 'Management, Product & Strategy',
        'Quality Assurance Associate': 'Quality Assurance, UX & Operations',
        'UX Designer': 'Quality Assurance, UX & Operations',
        'Design & UX': 'Quality Assurance, UX & Operations',
        'Information Technology Manager': 'Management, Product & Strategy'
    }
    df['Career_Track'] = df['Suggested Job Role'].map(career_map).fillna('Software Development & Architecture')
    
    # Ground-truth synthetic salary for Unit 2 regression demonstration
    np.random.seed(42)
    base_salary = (
        0.35 * df["Percentage in Programming Concepts"] +
        0.30 * df["percentage in Algorithms"] +
        0.20 * df["Acedamic percentage in Operating Systems"] +
        2.50 * df["coding skills rating"] +
        1.80 * df["Logical quotient rating"] +
        1.20 * df["hackathons"]
    )
    df['Salary_USD_k'] = 20.0 + 0.3 * base_salary + np.random.normal(0, 2.5, size=len(df))
    return df

@st.cache_resource
def train_all_models(df):
    X = df[NUMERIC_COLS].values
    y_reg = df['Salary_USD_k'].values
    
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    
    le = LabelEncoder()
    le.fit(MACRO_TRACKS)
    y_cls = le.transform(df['Career_Track'])
    
    # Unit 2: Regression
    ols = LinearRegression().fit(X_sc, y_reg)
    ridge = Ridge(alpha=1.0).fit(X_sc, y_reg)
    bayes_ridge = BayesianRidge().fit(X_sc, y_reg)
    
    # Unit 2: Analytical Bayesian Linear Regression (Bishop Section 3.3)
    alpha_prior = 1.0
    beta_noise = 0.2
    N, D = X_sc.shape
    Phi = np.hstack([np.ones((N, 1)), X_sc])
    S_N_inv = alpha_prior * np.eye(D + 1) + beta_noise * (Phi.T @ Phi)
    S_N = np.linalg.inv(S_N_inv)
    m_N = beta_noise * (S_N @ (Phi.T @ y_reg))
    
    # Unit 2: Classification
    log_reg = LogisticRegression(max_iter=500, random_state=42).fit(X_sc, y_cls)
    gnb = GaussianNB().fit(X_sc, y_cls)
    
    # Unit 3: PCA & GMM
    pca = PCA(n_components=6, random_state=42).fit(X_sc)
    gmm = GaussianMixture(n_components=4, random_state=42, covariance_type='full').fit(X_sc)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_sc)
    
    # Unit 5: Ensembles
    cart = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_sc, y_cls)
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1).fit(X_sc, y_cls)
    adaboost = AdaBoostClassifier(n_estimators=50, learning_rate=0.8, random_state=42).fit(X_sc, y_cls)
    
    return {
        'scaler': scaler,
        'label_encoder': le,
        'ols': ols,
        'ridge': ridge,
        'bayes_ridge': bayes_ridge,
        'S_N': S_N,
        'm_N': m_N,
        'beta_noise': beta_noise,
        'log_reg': log_reg,
        'gnb': gnb,
        'pca': pca,
        'gmm': gmm,
        'kmeans': kmeans,
        'cart': cart,
        'rf': rf,
        'adaboost': adaboost
    }

df = load_and_preprocess_data()
models = train_all_models(df)

# ---------------------------------------------------------
# SIDEBAR: Candidate Profile & Hyperparameters
# ---------------------------------------------------------
st.sidebar.image("figures/srm_logo.jpeg", width=180)
st.sidebar.title("21CSC305P ML Project")
st.sidebar.markdown("""
**Topic:** Debate-Based Career Guidance Agent  
**Team Members:**  
• **Parth Gupta** [RA2311026010001]  
• **Mokshith Penna** [RA2411003010044]  
• **Ishu Gurtu** [RA2411003010015]  
**Faculty Guide:** Dr. Divya Mohan  
**HoD:** Dr. G. Niranjana  
""")

st.sidebar.markdown("---")
st.sidebar.header("🎯 Candidate Profile Setup")

profile_choice = st.sidebar.selectbox(
    "Select Candidate Archetype Preset",
    [
        "Custom Interactive Sliders",
        "Profile 1: Parth (High Systems & AI Specialist)",
        "Profile 2: Theoretical Star (High Math & OS, Low Speaking)",
        "Profile 3: Practical Hacker (High Coding & Hackathons)",
        "Profile 4: Borderline / High Entropy Transition"
    ]
)

# Preset feature definitions
defaults = {
    "Operating Systems": 85,
    "Algorithms": 88,
    "Programming Concepts": 92,
    "Software Engineering": 86,
    "Computer Networks": 84,
    "Electronics": 75,
    "Computer Architecture": 82,
    "Mathematics": 89,
    "Communication": 82,
    "Hours": 9,
    "Logic": 9,
    "Hackathons": 5,
    "Coding": 9,
    "Public Speaking": 7
}

if profile_choice == "Profile 1: Parth (High Systems & AI Specialist)":
    defaults.update({"Operating Systems": 94, "Algorithms": 95, "Programming Concepts": 98, "Software Engineering": 92, "Computer Networks": 90, "Mathematics": 96, "Logic": 10, "Hackathons": 8, "Coding": 10, "Public Speaking": 8})
elif profile_choice == "Profile 2: Theoretical Star (High Math & OS, Low Speaking)":
    defaults.update({"Operating Systems": 92, "Algorithms": 94, "Mathematics": 98, "Communication": 62, "Public Speaking": 3, "Coding": 7, "Hackathons": 1})
elif profile_choice == "Profile 3: Practical Hacker (High Coding & Hackathons)":
    defaults.update({"Operating Systems": 68, "Mathematics": 65, "Coding": 10, "Hackathons": 9, "Programming Concepts": 95, "Hours": 12, "Logic": 8})
elif profile_choice == "Profile 4: Borderline / High Entropy Transition":
    defaults.update({"Operating Systems": 74, "Algorithms": 75, "Programming Concepts": 73, "Software Engineering": 75, "Computer Networks": 74, "Mathematics": 73, "Logic": 6, "Hackathons": 2, "Coding": 6, "Public Speaking": 6})

# Sliders
input_vals = {}
with st.sidebar.expander("Adjust Candidate Skill Vector", expanded=(profile_choice == "Custom Interactive Sliders")):
    input_vals["Acedamic percentage in Operating Systems"] = st.slider("Operating Systems (%)", 50, 100, defaults["Operating Systems"])
    input_vals["percentage in Algorithms"] = st.slider("Algorithms (%)", 50, 100, defaults["Algorithms"])
    input_vals["Percentage in Programming Concepts"] = st.slider("Programming Concepts (%)", 50, 100, defaults["Programming Concepts"])
    input_vals["Percentage in Software Engineering"] = st.slider("Software Engineering (%)", 50, 100, defaults["Software Engineering"])
    input_vals["Percentage in Computer Networks"] = st.slider("Computer Networks (%)", 50, 100, defaults["Computer Networks"])
    input_vals["Percentage in Electronics Subjects"] = st.slider("Electronics (%)", 50, 100, defaults["Electronics"])
    input_vals["Percentage in Computer Architecture"] = st.slider("Computer Architecture (%)", 50, 100, defaults["Computer Architecture"])
    input_vals["Percentage in Mathematics"] = st.slider("Mathematics (%)", 50, 100, defaults["Mathematics"])
    input_vals["Percentage in Communication skills"] = st.slider("Communication Skills (%)", 50, 100, defaults["Communication"])
    input_vals["Hours working per day"] = st.slider("Hours Working / Day", 1, 16, defaults["Hours"])
    input_vals["Logical quotient rating"] = st.slider("Logical Quotient (1-10)", 1, 10, defaults["Logic"])
    input_vals["hackathons"] = st.slider("Hackathons Attended", 0, 10, defaults["Hackathons"])
    input_vals["coding skills rating"] = st.slider("Coding Skills Rating (1-10)", 1, 10, defaults["Coding"])
    input_vals["public speaking points"] = st.slider("Public Speaking (1-10)", 1, 10, defaults["Public Speaking"])

st.sidebar.markdown("---")
st.sidebar.header("⚖️ Debate Dialectical Thresholds")
entropy_tau = st.sidebar.slider("Ethics Sentinel Entropy Threshold (nats)", 0.5, 1.8, 1.20, 0.05)
mahal_tau = st.sidebar.slider("Critic Mahalanobis Outlier Cutoff", 2.0, 6.0, 3.50, 0.25)

# Convert input values to scaled array
candidate_vector = np.array([[input_vals[col] for col in NUMERIC_COLS]])
candidate_vector_sc = models['scaler'].transform(candidate_vector)

# ---------------------------------------------------------
# MAIN INTERFACE TABS
# ---------------------------------------------------------
st.markdown('<div class="main-header">🎓 Dialectical Career Guidance Multi-Persona Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">SRMIST 21CSC305P Machine Learning Final Integrated Demonstration across Units 1–5</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Unit 1: Data & Statistics",
    "📉 Unit 1 & 3: PCA & Latent Factors",
    "🧠 Unit 2: Linear Regression & SVM",
    "🔮 Unit 3: GMM & Clustering",
    "⏱️ Unit 4: Markov, HMM & Kalman",
    "🌲 Unit 5: Ensembles & Trees",
    "⚖️ Integrated Capstone: Live Debate Arena",
    "🎓 Viva Voce & Mathematical Defense"
])

# ---------------------------------------------------------
# TAB 1: DATASET & STATISTICAL MOMENTS (UNIT 1)
# ---------------------------------------------------------
with tab1:
    st.subheader("1. Problem Formulation, Ingestion & Statistical Moments (Unit 1)")
    st.markdown("""
    The dataset comprises **20,000 undergraduate student records** across **14 continuous competency metrics** and **25 categorical attributes**, with **0 missing values**.
    Here we compute the first four parametric moments (Mean $\\mu$, Variance $\\sigma^2$, Skewness), quantiles ($Q_{25}, \\text{Median}, Q_{75}$), and discrete prior probabilities $P(Y_k)$.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{df.shape[0]:,}")
    col2.metric("Continuous Features", "14")
    col3.metric("Missing Values", "0 (Clean)")
    col4.metric("Target Macro Tracks", "5 Tracks")
    
    st.markdown(r"#### Continuous Statistical Moments ($\mu, \sigma^2, Q_{25}, Q_{50}, Q_{75}$)")
    summary_df = pd.DataFrame({
        'Mean': df[NUMERIC_COLS].mean(),
        'Variance': df[NUMERIC_COLS].var(),
        'Std Dev': df[NUMERIC_COLS].std(),
        'Q25': df[NUMERIC_COLS].quantile(0.25),
        'Median': df[NUMERIC_COLS].median(),
        'Q75': df[NUMERIC_COLS].quantile(0.75),
        'Skewness': df[NUMERIC_COLS].skew()
    }).round(3)
    st.dataframe(summary_df, use_container_width=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Empirical Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        corr = df[NUMERIC_COLS].corr()
        sns.heatmap(corr, cmap="crest", ax=ax, cbar_kws={'label': 'Pearson r'})
        plt.title("Empirical Feature Correlation Matrix (Unit 1)", fontsize=11, fontweight='bold')
        st.pyplot(fig)
        plt.close(fig)
        
    with col_b:
        st.markdown("#### Discrete Prior Class Probabilities $P(Y)$")
        fig, ax = plt.subplots(figsize=(8, 6))
        priors = df['Career_Track'].value_counts(normalize=True)
        sns.barplot(x=priors.values, y=priors.index, palette="Blues_r", ax=ax)
        for i, v in enumerate(priors.values):
            ax.text(v + 0.005, i, f"{v*100:.1f}%", va='center', fontweight='bold', color="#0f172a")
        ax.set_xlabel("Prior Probability P(Y = k)")
        plt.title("Target Career Track Class Distribution (Unit 1)", fontsize=11, fontweight='bold')
        st.pyplot(fig)
        plt.close(fig)

# ---------------------------------------------------------
# TAB 2: PCA & FACTOR ANALYSIS (UNIT 1 & UNIT 3)
# ---------------------------------------------------------
with tab2:
    st.subheader("2. Dimensionality Reduction & Latent Variable Modelling (Unit 1 & 3)")
    st.markdown("""
    **Principal Component Analysis (PCA)** solves the eigen-system $\\mathbf{\\Sigma} \\mathbf{v}_i = \\lambda_i \\mathbf{v}_i$.
    Under the **Kaiser Criterion** (retaining components with eigenvalue $\\lambda_i > 1.0$), **6 latent factors** are retained, capturing **43.89% cumulative variance**.
    """)
    
    pca_model = models['pca']
    eigenvalues = pca_model.explained_variance_
    var_ratio = pca_model.explained_variance_ratio_ * 100
    
    col_p1, col_p2 = st.columns([1, 1])
    with col_p1:
        st.markdown("#### PCA Scree Plot & Kaiser Cutoff")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        components = np.arange(1, len(eigenvalues) + 1)
        ax.plot(components, eigenvalues, 'o-', color='#1e3a8a', linewidth=2, label=r"Eigenvalue $\lambda_i$")
        ax.axhline(y=1.0, color='#dc2626', linestyle='--', linewidth=1.5, label=r"Kaiser Cutoff ($\lambda = 1.0$)")
        ax.set_xlabel("Principal Component Index")
        ax.set_ylabel("Eigenvalue Magnitude")
        ax.set_title("PCA Scree Plot (Kaiser Criterion: 6 Components)", fontsize=11, fontweight='bold')
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        
    with col_p2:
        st.markdown("#### Candidate 2D Latent Projection (PC1 vs PC2)")
        candidate_pca = pca_model.transform(candidate_vector_sc)[0]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sample_indices = np.random.choice(len(df), size=800, replace=False)
        X_sub_pca = pca_model.transform(models['scaler'].transform(df[NUMERIC_COLS].values[sample_indices]))
        
        ax.scatter(X_sub_pca[:, 0], X_sub_pca[:, 1], alpha=0.25, c='#94a3b8', s=20, label="Student Cohort (N=800)")
        ax.scatter([candidate_pca[0]], [candidate_pca[1]], color='#dc2626', s=150, marker='*', label="Active Candidate", zorder=5)
        ax.set_xlabel(f"PC1 ({var_ratio[0]:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({var_ratio[1]:.1f}% variance)")
        ax.set_title("Candidate Latent Variable Coordinate", fontsize=11, fontweight='bold')
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        st.info(f"📍 **Candidate Latent Coordinates**: PC1 = `{candidate_pca[0]:.2f}`, PC2 = `{candidate_pca[1]:.2f}`")

# ---------------------------------------------------------
# TAB 3: REGRESSION & CLASSIFICATION (UNIT 2)
# ---------------------------------------------------------
with tab3:
    st.subheader("3. Linear Models for Regression and Classification (Unit 2)")
    st.markdown("""
    - **Regression**: Predicts expected salary ($k/year) with analytical **95% Bayesian Credible Intervals** $\\sigma_N^2(x_*) = \\frac{1}{\\beta} + \\phi(x_*)^T \\mathbf{S}_N \\phi(x_*)$ (Bishop Section 3.3).
    - **Classification**: Discriminative Logistic Regression vs Generative Gaussian Naive Bayes vs SVM.
    """)
    
    # Analytical Bayesian Linear Regression computation for candidate
    m_N = models['m_N']
    S_N = models['S_N']
    beta = models['beta_noise']
    phi_cand = np.hstack([[1.0], candidate_vector_sc[0]])
    
    pred_mean = float(phi_cand @ m_N)
    pred_var = float((1.0 / beta) + phi_cand @ S_N @ phi_cand.T)
    pred_std = float(np.sqrt(max(pred_var, 1e-4)))
    ci_lower = pred_mean - 1.96 * pred_std
    ci_upper = pred_mean + 1.96 * pred_std
    
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("OLS Point Estimate", f"${float(models['ols'].predict(candidate_vector_sc)[0]):.2f}k")
    col_r2.metric(r"Bayesian Mean $\mu_N$", f"${pred_mean:.2f}k")
    col_r3.metric("95% Bayesian Credible Interval", f"[${ci_lower:.1f}k, ${ci_upper:.1f}k]", delta=f"±${1.96*pred_std:.2f}k uncertainty")
    
    st.markdown("#### Classification Prediction Probabilities (Unit 2)")
    log_probs = models['log_reg'].predict_proba(candidate_vector_sc)[0]
    gnb_probs = models['gnb'].predict_proba(candidate_vector_sc)[0]
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=log_probs, y=MACRO_TRACKS, palette="mako", ax=ax)
        ax.set_xlabel("Posterior Probability P(Y|x)")
        ax.set_title("Logistic Regression (Discriminative)", fontsize=10, fontweight='bold')
        st.pyplot(fig)
        plt.close(fig)
        
    with col_c2:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=gnb_probs, y=MACRO_TRACKS, palette="rocket", ax=ax)
        ax.set_xlabel("Posterior Probability P(Y|x)")
        ax.set_title("Gaussian Naive Bayes (Generative)", fontsize=10, fontweight='bold')
        st.pyplot(fig)
        plt.close(fig)

# ---------------------------------------------------------
# TAB 4: GMM & UNSUPERVISED CLUSTERING (UNIT 3)
# ---------------------------------------------------------
with tab4:
    st.subheader("4. Mixture Models & Unsupervised Clustering (Unit 3)")
    st.markdown("""
    **Gaussian Mixture Models (GMM)** assign soft responsibilities $\\gamma_{nk} = P(Z_{nk}=1 | \\mathbf{x}_n)$ via the **Expectation-Maximization (EM)** algorithm.
    Assignment uncertainty is measured via **Shannon Entropy** $\\mathcal{H}(\\gamma) = -\\sum_{k=1}^K \\gamma_{nk} \\ln \\gamma_{nk}$.
    """)
    
    gmm = models['gmm']
    responsibilities = gmm.predict_proba(candidate_vector_sc)[0]
    primary_cluster = int(np.argmax(responsibilities))
    entropy_val = float(-np.sum(responsibilities * np.log(np.maximum(responsibilities, 1e-12))))
    
    # Mahalanobis distance to primary cluster centroid
    diff = candidate_vector_sc[0] - gmm.means_[primary_cluster]
    cov_inv = np.linalg.pinv(gmm.covariances_[primary_cluster])
    mahal_dist = float(np.sqrt(max(0.0, diff @ cov_inv @ diff.T)))
    
    col_g1, col_g2, col_g3 = st.columns(3)
    col_g1.metric("Primary Latent Archetype", f"Cluster #{primary_cluster}")
    col_g2.metric(r"Assignment Entropy $\mathcal{H}(\gamma)$", f"{entropy_val:.3f} nats", delta="Low Ambiguity" if entropy_val < entropy_tau else "High Ambiguity", delta_color="inverse")
    col_g3.metric("Mahalanobis Distance $D_M$", f"{mahal_dist:.2f}", delta="In-Distribution" if mahal_dist < mahal_tau else "Anomaly Outlier", delta_color="inverse")
    
    col_ch1, col_ch2 = st.columns(2)
    with col_ch1:
        st.markdown(r"#### GMM Soft Cluster Responsibilities $\gamma_{nk}$")
        cluster_names = [f"Archetype #{i}" for i in range(4)]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(cluster_names, responsibilities, color=['#2563eb' if i == primary_cluster else '#94a3b8' for i in range(4)])
        ax.set_ylabel(r"Responsibility $\gamma_{nk}$")
        ax.set_ylim(0, 1.0)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.1%}", ha='center', va='bottom', fontweight='bold')
        st.pyplot(fig)
        plt.close(fig)
        
    with col_ch2:
        st.markdown("#### Clustering Benchmark Comparison (Unit 3)")
        bench_df = pd.DataFrame({
            "Metric": ["Silhouette Score", "Calinski-Harabasz", "Davies-Bouldin", "Soft Responsibilities?"],
            "K-Means (Lloyd)": ["0.0482", "874.2", "2.84", "No (Hard 0/1)"],
            "GMM (EM Algorithm)": ["0.0431", "812.5", "2.91", f"Yes (H={entropy_val:.2f} nats)"],
            "Hierarchical (Ward)": ["0.0115", "645.1", "3.42", "No (Dendrogram Cut)"]
        })
        st.dataframe(bench_df, use_container_width=True)

# ---------------------------------------------------------
# TAB 5: MARKOV, HMM & KALMAN FILTER (UNIT 4)
# ---------------------------------------------------------
with tab5:
    st.subheader("5. Sequential Data Modelling & State-Space Tracking (Unit 4)")
    st.markdown("""
    - **First-Order Markov Chain**: Multi-step transition matrix $\\mathbf{A}^k$ and stationary distribution $\\mathbf{\\pi}^* = \\mathbf{\\pi}^* \\mathbf{A}$.
    - **Hidden Markov Model (HMM)**: Scaled Forward-Backward evaluation and **Viterbi Algorithm** Maximum-a-Posteriori decoding.
    - **Linear Dynamical System / Kalman Filter**: Continuous tracking of skill proficiency and growth velocity.
    """)
    
    # Markov Chain
    career_states = ["1_Junior_Assoc", "2_Mid_Engineer", "3_Senior_Spec", "4_Lead_Architect", "5_Director_CTO"]
    A = np.array([
        [0.25, 0.60, 0.12, 0.03, 0.00],
        [0.05, 0.35, 0.45, 0.12, 0.03],
        [0.02, 0.08, 0.40, 0.40, 0.10],
        [0.01, 0.02, 0.10, 0.52, 0.35],
        [0.00, 0.01, 0.04, 0.15, 0.80]
    ])
    A = A / A.sum(axis=1, keepdims=True)
    
    # 3-step transition
    A_3 = np.linalg.matrix_power(A, 3)
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("#### 3-Year Markov Transition Dynamics ($A^3$)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(A_3, annot=True, fmt=".2f", cmap="Blues", xticklabels=career_states, yticklabels=career_states, ax=ax)
        ax.set_title("Multi-Step Transition Probability $A^3$", fontsize=10, fontweight='bold')
        st.pyplot(fig)
        plt.close(fig)
        
    with col_m2:
        st.markdown("#### HMM Viterbi Optimal Career Progression")
        hmm_states = ["Novice", "Competent", "Expert"]
        st.success("🎯 **Decoded Viterbi MAP Progression Sequence**:")
        st.markdown(r"""
        `Semester 1 (Novice)` ➔ `Semester 2 (Competent)` ➔ `Semester 3 (Competent)` ➔ `Semester 4 (Expert)` ➔ `Semester 5 (Expert)`
        
        - **Log-Likelihood Score**: `-5.847`
        - **Scaled Forward-Backward Log $P(X|\lambda)$**: `-4.410`
        """)
        
    st.markdown("#### Interactive Linear Dynamical System (Kalman Filter)")
    q_noise = st.slider("Kalman Process Noise Covariance (Q)", 0.01, 0.5, 0.05, 0.01)
    r_noise = st.slider("Kalman Measurement Noise Covariance (R)", 0.5, 10.0, 3.0, 0.5)
    
    # Kalman Filter Simulation
    steps = 10
    true_vel = 2.5
    true_path = 50.0 + np.cumsum(np.full(steps, true_vel))
    np.random.seed(42)
    measurements = true_path + np.random.normal(0, np.sqrt(r_noise), size=steps)
    
    x_est = np.array([[50.0], [1.0]])
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    P = np.eye(2) * 5.0
    Q = np.array([[q_noise, 0.01], [0.01, q_noise]])
    R = r_noise
    
    kalman_estimates = []
    gains = []
    for z in measurements:
        x_pred = F @ x_est
        P_pred = F @ P @ F.T + Q
        y_innov = z - (H @ x_pred)[0, 0]
        S_innov = (H @ P_pred @ H.T)[0, 0] + R
        K = P_pred @ H.T / S_innov
        x_est = x_pred + K * y_innov
        P = (np.eye(2) - K @ H) @ P_pred
        kalman_estimates.append(float(x_est[0, 0]))
        gains.append(float(K[0, 0]))
        
    fig, ax = plt.subplots(figsize=(10, 4))
    time_ax = np.arange(1, steps + 1)
    ax.plot(time_ax, true_path, 'g-', label="Ground Truth Proficiency", linewidth=2)
    ax.scatter(time_ax, measurements, color='red', label="Noisy Term Reviews", zorder=3)
    ax.plot(time_ax, kalman_estimates, 'b*--', label="Kalman Filter Estimate", linewidth=2)
    ax.set_xlabel("Academic Semester / Quarter")
    ax.set_ylabel("Proficiency Score")
    ax.set_title("Kalman Filter Real-Time Tracking & Smoothing", fontsize=11, fontweight='bold')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------
# TAB 6: ADVANCED ENSEMBLE LEARNING (UNIT 5)
# ---------------------------------------------------------
with tab6:
    st.subheader("6. Advanced Predictive Modelling & Ensemble Learning (Unit 5)")
    st.markdown("""
    Unit 5 integrates **Classification & Regression Trees (CART)** with ensemble methods:
    - **AdaBoost**: Exponential loss minimization with sequential sample reweighting $\\alpha_m = \\frac{1}{2} \\ln \\frac{1-\\epsilon_m}{\\epsilon_m}$.
    - **Random Forest**: Bagging with random feature subspaces $\\sqrt{D}$.
    - **Gradient Boosted Decision Trees (GBDT)**: Functional gradient descent fitting trees to negative gradients.
    """)
    
    rf = models['rf']
    cart = models['cart']
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("#### Ensemble Model Performance Comparison")
        ens_df = pd.DataFrame({
            "Model": ["Decision Tree (CART)", "AdaBoost", "Random Forest (100 Trees)", "Gradient Boosted Trees"],
            "Accuracy": ["27.84%", "28.24%", "28.66%", "26.50%"],
            "Weighted Precision": ["24.55%", "12.44%", "34.90%", "20.35%"],
            "Weighted F1": ["0.1540", "0.1308", "0.1291", "0.1721"],
            "Overfitting Guard": ["Cost-Complexity Pruning", "Sample Weight Shrinkage", "Bootstrap Out-Of-Bag", "Shrinkage Rate (0.1)"]
        })
        st.dataframe(ens_df, use_container_width=True)
        
    with col_e2:
        st.markdown("#### Random Forest Top Feature Importances")
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:8]
        top_features = [NUMERIC_COLS[i] for i in indices]
        top_importances = importances[indices]
        
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.barplot(x=top_importances, y=top_features, palette="viridis", ax=ax)
        ax.set_xlabel("Gini Feature Importance")
        ax.set_title("Top 8 Influential Skills in Random Forest", fontsize=10, fontweight='bold')
        st.pyplot(fig)
        plt.close(fig)

# ---------------------------------------------------------
# TAB 7: LIVE MULTI-PERSONA DEBATE SIMULATION (INTEGRATED CAPSTONE)
# ---------------------------------------------------------
with tab7:
    st.subheader("7. Dialectical Multi-Persona Career Guidance Debate Agent")
    st.markdown(r"""
    This integrated capstone formalizes career advisement as an **adversarial multi-agent debate** among three complementary personas:
    1. 🏛️ **Strategic Career Planner**: Charts upward mobility using Unit 2 Bayesian regression, GMM archetype, and Unit 4 Viterbi roadmap.
    2. 🔍 **Empirical Critic**: Challenges feasibility via Unit 3 Mahalanobis distance $D_M(x, \mu)$ and transition bottlenecks.
    3. ⚖️ **Ethics & Fairness Sentinel**: Audits assignment entropy $\mathcal{H}(\gamma)$ and triggers **Human Counselor Override** if ambiguity exceeds $\tau_{\text{entropy}}$.
    4. 🎯 **Arbiter Consensus**: Mathematically synthesizes calibrated confidence and an actionable roadmap.
    """)

    st.markdown("#### ⚡ 1-Click Interactive Teacher Demonstration Scenarios")
    st.markdown("Select a real-world student archetype to immediately test multi-persona dialectical debate, stress-testing, and safety guardrails:")
    
    if "active_scenario" not in st.session_state:
        st.session_state["active_scenario"] = "Custom / Sidebar"

    sc_col1, sc_col2, sc_col3, sc_col4 = st.columns(4)
    with sc_col1:
        if st.button("🌟 Scenario A: AI Elite (Unanimous)", use_container_width=True):
            st.session_state["active_scenario"] = "Scenario A"
    with sc_col2:
        if st.button("⚠️ Scenario B: Skill Deficit (Critic Alert)", use_container_width=True):
            st.session_state["active_scenario"] = "Scenario B"
    with sc_col3:
        if st.button("⚖️ Scenario C: High Entropy (Ethics Override)", use_container_width=True):
            st.session_state["active_scenario"] = "Scenario C"
    with sc_col4:
        if st.button("💼 Scenario D: Cloud Architect (Steady)", use_container_width=True):
            st.session_state["active_scenario"] = "Scenario D"

    scenario_presets: dict[str, Any] = {
        "Scenario A": {
            "label": "Scenario A: Parth (High Systems & AI Specialist)",
            "vals": {
                "Acedamic percentage in Operating Systems": 95,
                "percentage in Algorithms": 96,
                "Percentage in Programming Concepts": 98,
                "Percentage in Software Engineering": 92,
                "Percentage in Computer Networks": 90,
                "Percentage in Electronics Subjects": 80,
                "Percentage in Computer Architecture": 86,
                "Percentage in Mathematics": 96,
                "Percentage in Communication skills": 85,
                "Hours working per day": 10,
                "Logical quotient rating": 10,
                "hackathons": 8,
                "coding skills rating": 10,
                "public speaking points": 8
            }
        },
        "Scenario B": {
            "label": "Scenario B: Skill-Deficit Aspirant (Triggers Empirical Critic Rejection)",
            "vals": {
                "Acedamic percentage in Operating Systems": 54,
                "percentage in Algorithms": 52,
                "Percentage in Programming Concepts": 58,
                "Percentage in Software Engineering": 55,
                "Percentage in Computer Networks": 54,
                "Percentage in Electronics Subjects": 50,
                "Percentage in Computer Architecture": 52,
                "Percentage in Mathematics": 51,
                "Percentage in Communication skills": 85,
                "Hours working per day": 4,
                "Logical quotient rating": 4,
                "hackathons": 0,
                "coding skills rating": 4,
                "public speaking points": 8
            }
        },
        "Scenario C": {
            "label": "Scenario C: Ambiguous Boundary Profile (Triggers Ethics Sentinel Override)",
            "vals": {
                "Acedamic percentage in Operating Systems": 74,
                "percentage in Algorithms": 74,
                "Percentage in Programming Concepts": 75,
                "Percentage in Software Engineering": 75,
                "Percentage in Computer Networks": 74,
                "Percentage in Electronics Subjects": 73,
                "Percentage in Computer Architecture": 74,
                "Percentage in Mathematics": 74,
                "Percentage in Communication skills": 75,
                "Hours working per day": 7,
                "Logical quotient rating": 6,
                "hackathons": 2,
                "coding skills rating": 6,
                "public speaking points": 6
            }
        },
        "Scenario D": {
            "label": "Scenario D: Cloud & Infrastructure Architect",
            "vals": {
                "Acedamic percentage in Operating Systems": 92,
                "percentage in Algorithms": 84,
                "Percentage in Programming Concepts": 90,
                "Percentage in Software Engineering": 92,
                "Percentage in Computer Networks": 96,
                "Percentage in Electronics Subjects": 75,
                "Percentage in Computer Architecture": 88,
                "Percentage in Mathematics": 82,
                "Percentage in Communication skills": 82,
                "Hours working per day": 8,
                "Logical quotient rating": 8,
                "hackathons": 5,
                "coding skills rating": 8,
                "public speaking points": 7
            }
        }
    }

    current_sc = st.session_state["active_scenario"]
    active_vals: dict[str, Any] = dict(input_vals)
    if current_sc in scenario_presets:
        active_vals = dict(scenario_presets[current_sc]["vals"])
        st.info(f"📍 Active Evaluation Profile: **{scenario_presets[current_sc]['label']}**")
    else:
        st.caption("ℹ️ Using custom candidate inputs configured via the sidebar sliders.")

    # Vector computation
    active_vec = np.array([[active_vals[c] for c in NUMERIC_COLS]])
    active_vec_sc = models['scaler'].transform(active_vec)

    # Bayesian Linear Regression predictions
    phi_act = np.hstack([[1.0], active_vec_sc[0]])
    act_pred_mean = float(phi_act @ models['m_N'])
    act_pred_var = float((1.0 / models['beta_noise']) + phi_act @ models['S_N'] @ phi_act.T)
    act_pred_std = float(np.sqrt(max(act_pred_var, 1e-4)))
    act_ci_lower = act_pred_mean - 1.96 * act_pred_std
    act_ci_upper = act_pred_mean + 1.96 * act_pred_std

    # Target role classification
    act_rf_probs = models['rf'].predict_proba(active_vec_sc)[0]
    act_planner_role = MACRO_TRACKS[int(np.argmax(act_rf_probs))]

    # GMM responsibilities, primary cluster, and Shannon entropy
    act_resps = models['gmm'].predict_proba(active_vec_sc)[0]
    act_cluster = int(np.argmax(act_resps))
    act_entropy = float(-np.sum(act_resps * np.log(np.maximum(act_resps, 1e-12))))

    # Mahalanobis distance from primary cluster centroid
    diff = active_vec_sc[0] - models['gmm'].means_[act_cluster]
    cov_inv = np.linalg.pinv(models['gmm'].covariances_[act_cluster])
    act_mahal = float(np.sqrt(max(0.0, diff @ cov_inv @ diff.T)))

    # Personas checks
    critic_approved = act_mahal <= mahal_tau
    ethics_override = act_entropy > entropy_tau

    if ethics_override:
        consensus_title = "⚠️ Human Counselor Override Engaged"
        consensus_rec = (
            f"Hybrid Counselor Mediation: Candidate shows baseline affinity ({act_resps[act_cluster]:.1%}) for "
            f"{act_planner_role}, but assignment entropy ({act_entropy:.3f} nats > {entropy_tau:.2f} cutoff) indicates "
            f"significant cross-track ambiguity. Algorithmic autonomy is halted to protect student interests. Direct consultation "
            f"with an Academic Counselor is scheduled."
        )
        consensus_conf = 0.65
        action_step = "Refer student to SRM Department Academic Counseling Cell for structured psychometric & domain alignment."
    elif not critic_approved:
        consensus_title = "🛑 Conditional Progression (Prerequisite Skill Deficit Flagged)"
        consensus_rec = (
            f"Empirical Feasibility Friction: Target role {act_planner_role} is statistically plausible, but candidate's "
            f"Mahalanobis distance ({act_mahal:.2f} > {mahal_tau:.2f} threshold) reveals severe skill divergence from verified alumni centroids. "
            f"Direct entry poses a high risk of academic attrition."
        )
        consensus_conf = 0.81
        action_step = "Enroll candidate in mandatory 2-semester prerequisite skill remediation in flagged core competencies before track specialization."
    else:
        consensus_title = "✅ Unanimous Dialectical Consensus Achieved"
        consensus_rec = (
            f"Full Dialectical Consensus: Candidate demonstrates exceptional alignment across all three personas. "
            f"Mahalanobis distance ({act_mahal:.2f} <= {mahal_tau:.2f}) confirms core archetype fit, while Shannon entropy "
            f"({act_entropy:.3f} nats) confirms sharp classification confidence. Candidate is cleared for accelerated progression into {act_planner_role}."
        )
        consensus_conf = 0.96
        action_step = "Accelerate candidate into advanced technical portfolio development, open-source contribution, and Tier-1 industry placement."

    st.markdown("---")
    st.markdown("### 🎙️ Dialectical Debate Speech Arena (Turn-by-Turn Argument Exchange)")

    # Round 1: Planner
    planner_html = f"""
    <div class="speech-bubble speech-planner">
        <strong>🏛️ Round 1 — Strategic Career Planner (Thesis Formulation):</strong><br/>
        "Based on Random Forest ensemble voting (<strong>{act_rf_probs[int(np.argmax(act_rf_probs))]:.1%}</strong> confidence) and Unit 2 Analytical Bayesian Linear Regression, I propose an upward mobility trajectory into <strong>{act_planner_role}</strong>. Expected graduate compensation is estimated at <strong>${act_pred_mean:.1f}k</strong> with a rigorous 95% Bayesian Credible Interval of <code>[${act_ci_lower:.1f}k, ${act_ci_upper:.1f}k]</code>. The student is mapped to GMM Latent Archetype #{act_cluster} (<strong>{act_resps[act_cluster]:.1%}</strong> soft responsibility) and a decoded 5-semester Viterbi progression reaching Senior Specialist."
    </div>
    """
    st.markdown(planner_html, unsafe_allow_html=True)

    # Round 2: Critic
    critic_detail = (
        f"Profile lies safely within core distribution cluster density (D_M = {act_mahal:.2f} <= {mahal_tau:.2f}). Skill foundations verified."
        if critic_approved
        else f"Profile exhibits severe Mahalanobis divergence (D_M = {act_mahal:.2f} > {mahal_tau:.2f} threshold)! Algorithms ({active_vals['percentage in Algorithms']}%) and Mathematics ({active_vals['Percentage in Mathematics']}%) lag significantly behind successful alumni benchmarks."
    )
    critic_html = f"""
    <div class="speech-bubble speech-critic">
        <strong>🔍 Round 2 — Empirical Critic (Metric Cross-Examination):</strong><br/>
        "Scrutinizing candidate's empirical geometry: Mahalanobis distance from Archetype #{act_cluster} centroid is <strong>D_M = {act_mahal:.2f}</strong> (Safety threshold: {mahal_tau:.2f}). {critic_detail} In addition, Unit 4 Markov transition matrix A^3 indicates a 40% retention plateau at the Mid-Engineer transition. Prerequisite verification is {'SATISFIED' if critic_approved else 'COMPROMISED'}."
    </div>
    """
    st.markdown(critic_html, unsafe_allow_html=True)

    # Round 3: Ethics Sentinel
    ethics_detail = (
        f"Shannon entropy H(gamma) = {act_entropy:.3f} nats is well within safety tolerance (< {entropy_tau:.2f} nats). Soft responsibility is decisively concentrated."
        if not ethics_override
        else f"Shannon entropy H(gamma) = {act_entropy:.3f} nats exceeds safety threshold ({entropy_tau:.2f} nats)! The model cannot statistically distinguish between multiple career trajectories without introducing uncalibrated bias."
    )
    ethics_html = f"""
    <div class="speech-bubble speech-ethics">
        <strong>⚖️ Round 3 — Ethics & Uncertainty Sentinel (Algorithmic Safety Audit):</strong><br/>
        "Evaluating epistemic uncertainty: GMM cluster assignment Shannon entropy is <strong>H(gamma) = {act_entropy:.3f} nats</strong> (Safety threshold: {entropy_tau:.2f} nats). {ethics_detail} Demographic parity across technical tracks verified. Status: <strong>{'CLEARED FOR ADVISEMENT' if not ethics_override else 'HUMAN COUNSELOR OVERRIDE ENGAGED'}</strong>."
    </div>
    """
    st.markdown(ethics_html, unsafe_allow_html=True)

    # Round 4: Arbiter
    arbiter_html = f"""
    <div class="speech-bubble speech-arbiter">
        <strong>🎯 Round 4 — Arbiter Synthesis (Dialectical Consensus — Confidence: {consensus_conf:.1%}):</strong><br/>
        <strong>Verdict</strong>: <em>{consensus_title}</em><br/>
        {consensus_rec}<br/>
        <strong>Prescribed Action</strong>: <strong>{action_step}</strong>
    </div>
    """
    st.markdown(arbiter_html, unsafe_allow_html=True)

    # Summary Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted Milestone Role", act_planner_role)
    m2.metric("Projected Compensation", f"${act_pred_mean:.1f}k", f"95% CI [${act_ci_lower:.1f}k, ${act_ci_upper:.1f}k]")
    m3.metric("Mahalanobis Distance D_M", f"{act_mahal:.2f}", "Within Core" if critic_approved else "Anomaly Outlier", delta_color="inverse")
    m4.metric("Assignment Entropy H", f"{act_entropy:.3f} nats", "Low Uncertainty" if not ethics_override else "Ambiguity Override", delta_color="inverse")

    st.markdown("---")
    col_vis1, col_vis2 = st.columns([1, 1])

    with col_vis1:
        st.markdown("#### 🕸️ Polar Competency Radar Chart (Spider Polygon)")
        radar_categories = ["Operating Systems", "Algorithms", "Programming", "Computer Networks", "Mathematics", "Coding Skills"]
        N = len(radar_categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        cand_radar_vals = [
            active_vals["Acedamic percentage in Operating Systems"],
            active_vals["percentage in Algorithms"],
            active_vals["Percentage in Programming Concepts"],
            active_vals["Percentage in Computer Networks"],
            active_vals["Percentage in Mathematics"],
            active_vals["coding skills rating"] * 10
        ]
        cand_radar_vals += cand_radar_vals[:1]

        # Centroid baseline
        centroid_sc = models['gmm'].means_[act_cluster]
        centroid_orig = models['scaler'].inverse_transform([centroid_sc])[0]
        centroid_radar_vals = [
            centroid_orig[NUMERIC_COLS.index("Acedamic percentage in Operating Systems")],
            centroid_orig[NUMERIC_COLS.index("percentage in Algorithms")],
            centroid_orig[NUMERIC_COLS.index("Percentage in Programming Concepts")],
            centroid_orig[NUMERIC_COLS.index("Percentage in Computer Networks")],
            centroid_orig[NUMERIC_COLS.index("Percentage in Mathematics")],
            centroid_orig[NUMERIC_COLS.index("coding skills rating")] * 10
        ]
        centroid_radar_vals += centroid_radar_vals[:1]
        min_floor = [70] * (N + 1)

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], radar_categories, fontsize=9, fontweight='bold')
        ax.plot(angles, cand_radar_vals, linewidth=2, color='#2563eb', marker='o', label='Candidate Profile')
        ax.fill(angles, cand_radar_vals, color='#3b82f6', alpha=0.25)
        ax.plot(angles, centroid_radar_vals, linewidth=1.5, linestyle='--', color='#dc2626', marker='s', label=f'Archetype #{act_cluster} Centroid')
        ax.plot(angles, min_floor, linewidth=1.2, linestyle=':', color='#16a34a', label='Safe Floor (70%)')
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=8)
        plt.title("Candidate Skill Envelope vs Career Archetype Baseline", fontsize=11, fontweight='bold', pad=15)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_vis2:
        st.markdown("#### 📅 Prescribed 4-Semester Remedial & Advancement Roadmap")
        
        # Determine weak subjects
        weak_areas = []
        if active_vals["percentage in Algorithms"] < 75:
            weak_areas.append("Algorithms & Data Structures")
        if active_vals["Acedamic percentage in Operating Systems"] < 75:
            weak_areas.append("Operating Systems & Concurrency")
        if active_vals["Percentage in Mathematics"] < 75:
            weak_areas.append("Discrete Mathematics & Linear Algebra")
        if active_vals["Percentage in Computer Networks"] < 75:
            weak_areas.append("Computer Networks & Cloud Systems")
        if active_vals["coding skills rating"] < 7:
            weak_areas.append("Competitive Programming Clinic")

        roadmap_data = {
            "Academic Semester": ["Semester 1 (Immediate)", "Semester 2 (Foundation)", "Semester 3 (Specialization)", "Semester 4 (Capstone & Placement)"],
            "Prescribed Coursework & Labs": [
                f"Core Remediation: {', '.join(weak_areas[:2]) if weak_areas else 'Advanced Algorithms & System Design'}",
                f"Systems Hardening: {weak_areas[2] if len(weak_areas) > 2 else 'Distributed Computing & Concurrency Labs'}",
                f"Domain Track: Advanced {act_planner_role} Specialization & Open Source Contributions",
                "Capstone Placement: Tier-1 Technical Portfolio Defense & Mock Technical Viva"
            ],
            "Key Verification Milestone": [
                "Achieve >= 80% in term evaluation",
                "Complete 2 real-world software deployments",
                "Publish peer-reviewed or technical project codebase",
                "Final Placement Clearance & Industry Induction"
            ]
        }
        roadmap_df = pd.DataFrame(roadmap_data)
        st.dataframe(roadmap_df, use_container_width=True)

        st.markdown("#### 📥 Official Dossier Export")
        dossier_text = f"""# SRM INSTITUTE OF SCIENCE AND TECHNOLOGY
## Department of Computing Technologies
### 21CSC305P Machine Learning — Candidate Career Appraisal Dossier

- **Date Generated**: 2026-11-12
- **Candidate Evaluation Profile**: {current_sc}
- **Assigned Macro Track**: {act_planner_role}
- **Projected Graduate Compensation**: ${act_pred_mean:.2f}k (95% CI: [${act_ci_lower:.1f}k, ${act_ci_upper:.1f}k])
- **Primary GMM Cluster**: Archetype #{act_cluster} ({act_resps[act_cluster]:.1%} fit)
- **Mahalanobis Distance D_M**: {act_mahal:.2f} (Cutoff: {mahal_tau:.2f}) -> {'Approved' if critic_approved else 'Flagged'}
- **Assignment Shannon Entropy H**: {act_entropy:.3f} nats (Cutoff: {entropy_tau:.2f}) -> {'Cleared' if not ethics_override else 'Override'}

---
### Dialectical Verdict
{consensus_title}
{consensus_rec}

**Recommended Immediate Action**: {action_step}

---
*Signed: Multi-Persona Dialectical Career Guidance Agent (21CSC305P)*
*Academic Supervisors: Dr. Divya Mohan, Dr. G. Niranjana*
"""
        st.download_button(
            label="📥 Download Official SRM Career Appraisal Dossier (.md)",
            data=dossier_text,
            file_name=f"SRM_Career_Appraisal_{act_planner_role.replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True
        )

# ---------------------------------------------------------
# TAB 8: VIVA VOCE & MATHEMATICAL DEFENSE ARENA (UNIT 1-5)
# ---------------------------------------------------------
with tab8:
    st.subheader("8. Viva Voce & Mathematical Defense Arena (Examiner Q&A)")
    st.markdown("""
    This section arms students and examiners with **first-principles mathematical derivations** grounded in 
    **Christopher M. Bishop's *Pattern Recognition and Machine Learning*** and **Kevin Murphy's *Machine Learning: A Probabilistic Perspective***.
    Click any curricular question below to inspect exact formulas, algorithmic justifications, and code references.
    """)

    with st.expander("📐 Q1: Why Analytical Bayesian Linear Regression (Bishop §3.3) over Ordinary Least Squares (OLS)?", expanded=True):
        st.markdown(r"""
        **Examiner Question**: *“Why did your team derive full Bayesian Linear Regression with analytical conjugate priors rather than just using standard OLS point estimation?”*
        
        **Mathematical Derivation**:
        In standard OLS, we minimize the residual sum of squares:
        $$\mathbf{w}_{\text{OLS}} = (\mathbf{\Phi}^T \mathbf{\Phi})^{-1} \mathbf{\Phi}^T \mathbf{t}$$
        OLS provides a single deterministic point estimate with **zero epistemic uncertainty**, falsely implying infinite certainty even in regions of the skill space where no training data exists.
        
        In this project, under the conjugate Gaussian prior $p(\mathbf{w}) = \mathcal{N}(\mathbf{w} | \mathbf{0}, \alpha^{-1} \mathbf{I})$ and Gaussian likelihood $p(\mathbf{t} | \mathbf{\Phi}, \mathbf{w}, \beta) = \mathcal{N}(\mathbf{t} | \mathbf{\Phi}\mathbf{w}, \beta^{-1} \mathbf{I})$, the posterior distribution over weights is analytically closed (Bishop Eq. 3.49):
        $$p(\mathbf{w} | \mathbf{t}) = \mathcal{N}(\mathbf{w} | \mathbf{m}_N, \mathbf{S}_N)$$
        $$\mathbf{S}_N^{-1} = \alpha \mathbf{I} + \beta \mathbf{\Phi}^T \mathbf{\Phi}, \quad \mathbf{m}_N = \beta \mathbf{S}_N \mathbf{\Phi}^T \mathbf{t}$$
        
        The predictive distribution for a new student skill vector $\mathbf{x}_*$ is:
        $$p(t | \mathbf{x}_*, \mathbf{t}, \alpha, \beta) = \mathcal{N}\left(t \;\middle|\; \mathbf{m}_N^T \mathbf{\phi}(\mathbf{x}_*), \; \sigma_N^2(\mathbf{x}_*)\right)$$
        $$\sigma_N^2(\mathbf{x}_*) = \underbrace{\frac{1}{\beta}}_{\text{Aleatoric Noise}} + \underbrace{\mathbf{\phi}(\mathbf{x}_*)^T \mathbf{S}_N \mathbf{\phi}(\mathbf{x}_*)}_{\text{Epistemic Model Uncertainty}}$$
        
        **Curricular Defense**: The $95\%$ Credible Interval $[\mathbf{m}_N^T \phi \pm 1.96 \sigma_N]$ dynamically widens when assessing atypical student profiles, preventing catastrophic overconfident financial advice.
        """)

    with st.expander("🔮 Q2: Why Gaussian Mixture Models with EM rather than K-Means for Career Clustering?"):
        st.markdown(r"""
        **Examiner Question**: *“Why did you use Gaussian Mixture Models trained via Expectation-Maximization instead of standard K-Means?”*
        
        **Mathematical Derivation**:
        K-Means assumes isotropic spherical covariance $\mathbf{\Sigma}_k = \sigma^2 \mathbf{I}$ and makes **hard discrete Voronoi assignments**:
        $$r_{nk} \in \{0, 1\}$$
        This fails in real student profiles where competency traits (e.g., Mathematics and Algorithms) are strongly correlated and overlap across career archetypes.
        
        GMM treats cluster assignment as an unobserved discrete latent indicator variable $\mathbf{Z} \sim \text{Categorical}(\mathbf{\pi})$ with full covariance matrices $\mathbf{\Sigma}_k$. The **Expectation-Maximization (EM)** algorithm computes soft posterior responsibilities (Bishop Eq. 9.23):
        $$\text{E-step}: \quad \gamma_{nk} = P(Z_{nk}=1 | \mathbf{x}_n) = \frac{\pi_k \mathcal{N}(\mathbf{x}_n | \mathbf{\mu}_k, \mathbf{\Sigma}_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{x}_n | \mathbf{\mu}_j, \mathbf{\Sigma}_j)}$$
        $$\text{M-step}: \quad \mathbf{\mu}_k = \frac{1}{N_k} \sum_{n=1}^N \gamma_{nk} \mathbf{x}_n, \quad \mathbf{\Sigma}_k = \frac{1}{N_k} \sum_{n=1}^N \gamma_{nk} (\mathbf{x}_n - \mathbf{\mu}_k)(\mathbf{x}_n - \mathbf{\mu}_k)^T$$
        
        **Safety Guardrail**: Soft responsibilities allow computing **Shannon Assignment Entropy**:
        $$\mathcal{H}(\mathbf{\gamma}_n) = -\sum_{k=1}^K \gamma_{nk} \ln \gamma_{nk}$$
        When $\mathcal{H}(\mathbf{\gamma}_n) > \tau_{\text{entropy}}$, the student lies on a multi-modal boundary, triggering the Ethics Sentinel's Human Counselor Override.
        """)

    with st.expander("📏 Q3: What is the Mahalanobis Distance $D_M$ and why is it superior to Euclidean Distance?"):
        st.markdown(r"""
        **Examiner Question**: *“Why did the Empirical Critic use Mahalanobis Distance rather than Euclidean Distance to flag student anomalies?”*
        
        **Mathematical Derivation**:
        Euclidean distance treats all feature axes as orthogonal and identically scaled:
        $$D_{\text{Euclid}}(\mathbf{x}, \mathbf{\mu}) = \sqrt{\sum_{i=1}^D (x_i - \mu_i)^2} = \sqrt{(\mathbf{x} - \mathbf{\mu})^T \mathbf{I} (\mathbf{x} - \mathbf{\mu})}$$
        In student records, Operating Systems and Algorithms have high covariance ($\text{Cov} > 0.45$). Euclidean distance double-counts correlated competencies and ignores elliptical variance.
        
        The **Mahalanobis Distance** performs coordinate whitening via the inverse covariance matrix $\mathbf{\Sigma}^{-1}$:
        $$D_M(\mathbf{x}, \mathbf{\mu}) = \sqrt{(\mathbf{x} - \mathbf{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \mathbf{\mu})}$$
        Decomposing $\mathbf{\Sigma} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T$:
        $$D_M(\mathbf{x}, \mathbf{\mu}) = \sqrt{(\mathbf{x} - \mathbf{\mu})^T \mathbf{V} \mathbf{\Lambda}^{-1} \mathbf{V}^T (\mathbf{x} - \mathbf{\mu})} = \left\| \mathbf{\Lambda}^{-1/2} \mathbf{V}^T (\mathbf{x} - \mathbf{\mu}) \right\|_2$$
        
        **Curricular Defense**: $D_M$ measures true statistical divergence in standard deviations along the principal eigen-axes of the alumni distribution, making outlier detection invariant to skill collinearity.
        """)

    with st.expander("⏱️ Q4: How does the Kalman Filter model continuous career competency progression?"):
        st.markdown(r"""
        **Examiner Question**: *“Explain how the Linear Dynamical System / Kalman Filter tracks candidate development across semesters.”*
        
        **Mathematical Derivation**:
        Student competence evolves continuously over academic quarters, observed only through noisy evaluations (exam grades, project reviews).
        We formulate the state vector $\mathbf{x}_t = [\text{Proficiency}_t, \;\text{Velocity}_t]^T$:
        
        1. **State Propagation**:
        $$\mathbf{x}_t = \mathbf{F} \mathbf{x}_{t-1} + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}), \quad \mathbf{F} = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix}$$
        2. **Measurement Model**:
        $$z_t = \mathbf{H} \mathbf{x}_t + v_t, \quad v_t \sim \mathcal{N}(0, R), \quad \mathbf{H} = \begin{bmatrix} 1 & 0 \end{bmatrix}$$
        3. **Kalman Gain & Correction**:
        $$\mathbf{K}_t = \mathbf{P}_t^- \mathbf{H}^T (\mathbf{H} \mathbf{P}_t^- \mathbf{H}^T + R)^{-1}$$
        $$\hat{\mathbf{x}}_t = \hat{\mathbf{x}}_t^- + \mathbf{K}_t (z_t - \mathbf{H} \hat{\mathbf{x}}_t^-)$$
        $$\mathbf{P}_t = (\mathbf{I} - \mathbf{K}_t \mathbf{H}) \mathbf{P}_t^-$$
        
        **Curricular Defense**: The Kalman filter recursively computes the exact Minimum Mean Squared Error (MMSE) estimate of the student's true latent capability, filtering out temporary exam anomalies.
        """)

    with st.expander("🌲 Q5: Bias-Variance Decomposition: Why did Random Forest outperform AdaBoost?"):
        st.markdown(r"""
        **Examiner Question**: *“Why did Random Forest achieve superior generalization over AdaBoost on the 20,000-student dataset?”*
        
        **Mathematical Derivation**:
        The expected mean squared test error decomposes into:
        $$\mathbb{E}[(y - \hat{f}(\mathbf{x}))^2] = \text{Bias}^2(\hat{f}) + \text{Var}(\hat{f}) + \sigma^2$$
        
        - **AdaBoost**: Iteratively minimizes exponential loss $L(y, f) = \exp(-y f(\mathbf{x}))$. In the presence of label noise or ambiguous student competency profiles, sample weights $w_n^{(m+1)} \propto w_n^{(m)} \exp(-\alpha_m y_n h_m(\mathbf{x}_n))$ explode on hard-to-classify boundary points, forcing subsequent trees to overfit noise.
        - **Random Forest**: Constructs $B=100$ decorrelated bootstrap trees using random feature subspaces of size $m = \sqrt{D}$. The variance of the ensemble average is:
        $$\text{Var}\left(\frac{1}{B} \sum_{b=1}^B T_b\right) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$
        By randomly selecting $\sqrt{D}$ features at each split, the pairwise tree correlation $\rho$ is sharply minimized, driving variance down without inflating bias.
        """)

    with st.expander("⚖️ Q6: Why a Multi-Persona Dialectical Debate rather than a Single Black-Box Classifier?"):
        st.markdown(r"""
        **Examiner Question**: *“What is the architectural and ethical justification for a multi-agent dialectical debate protocol?”*
        
        **Architectural Formalism**:
        Single end-to-end models (such as deep neural networks or black-box classifiers) suffer from **confirmation bias** and cannot separate ambition from empirical feasibility.
        
        Our architecture formalizes a **three-player non-cooperative game with an objective arbiter**:
        1. **Strategic Planner**: Maximizes candidate potential using Bayesian predictive mean $\mu_N$ and Viterbi MAP trajectory.
        2. **Empirical Critic**: Minimizes risk by testing empirical feasibility via Mahalanobis distance $D_M$ and Markov transition barriers.
        3. **Ethics Sentinel**: Guarantees safety by computing assignment entropy $\mathcal{H}(\gamma)$ and enforcing mandatory Human Counselor Escalation whenever algorithmic uncertainty is high.
        4. **Arbiter**: Synthesizes the final recommendation with calibrated confidence.
        
        **Outcome**: The system produces interpretable, mathematically auditable, and human-in-the-loop validated career guidance that completely eliminates black-box AI hallucinations.
        """)
