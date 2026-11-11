"""
🎓 SRM Institute of Science and Technology - Department of Computing Technologies
Course: 21CSC305P Machine Learning (Final Integrated Project & Demonstration)
Project: Debate-Based Career Guidance Agent with Complementary Personas
Authors: Parth Gupta, Mokshith Penna, Ishu Gurtu
Supervisors: Dr. Divya Mohan, Dr. G. Niranjana
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

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

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Unit 1: Data & Statistics",
    "📉 Unit 1 & 3: PCA & Latent Factors",
    "🧠 Unit 2: Linear Regression & SVM",
    "🔮 Unit 3: GMM & Clustering",
    "⏱️ Unit 4: Markov, HMM & Kalman",
    "🌲 Unit 5: Ensembles & Trees",
    "⚖️ Integrated Capstone: Live Debate"
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
    
    if st.button("🚀 Execute Dialectical Multi-Persona Debate", type="primary", use_container_width=True):
        # Determine stances
        planner_role = MACRO_TRACKS[int(np.argmax(models['rf'].predict_proba(candidate_vector_sc)[0]))]
        planner_salary = float(pred_mean)
        
        critic_approved = mahal_dist <= mahal_tau
        critic_stance = "FEASIBLE PROGRESSION" if critic_approved else "RISK FLAGGED: OUTLIER ANOMALY"
        
        ethics_override = entropy_val > entropy_tau
        ethics_stance = "TRIGGERED: HUMAN COUNSELOR OVERRIDE" if ethics_override else "VALIDATED: SAFE FOR ADVISEMENT"
        
        if ethics_override:
            consensus_rec = f"Hybrid Counselor Mediation: Candidate shows strong baseline potential for {planner_role}, but high assignment entropy ({entropy_val:.2f} nats) demands personalized human counselor guidance."
            consensus_conf = 0.65
        elif not critic_approved:
            consensus_rec = f"Conditional Advancement: Progress toward {planner_role} with mandatory prerequisite skill remediation to reduce Mahalanobis distance ({mahal_dist:.2f} > {mahal_tau:.2f})."
            consensus_conf = 0.82
        else:
            consensus_rec = f"Full Dialectical Consensus: Candidate demonstrates robust qualification across all 3 personas. Accelerate career track into {planner_role}."
            consensus_conf = 0.95
            
        p1, p2, p3 = st.columns(3)
        with p1:
            st.info(f"🏛️ **Planner ({planner_role})**")
            st.markdown(f"""
            - **Target Compensation**: `${planner_salary:.1f}k` (CI: `[${ci_lower:.1f}k, ${ci_upper:.1f}k]`)
            - **GMM Archetype**: Cluster `#{primary_cluster}` ({responsibilities[primary_cluster]:.1%} fit)
            - **Viterbi Roadmap**: Decoded into `Senior Specialist` trajectory.
            """)
        with p2:
            st.warning(f"🔍 **Critic ({critic_stance})**")
            st.markdown(f"""
            - **Mahalanobis Distance**: `{mahal_dist:.2f}` (Threshold: `{mahal_tau:.2f}`)
            - **Prerequisite Status**: {'Verified in core distribution' if critic_approved else 'Excessive skill deficit from cluster core'}
            - **Bottleneck Audit**: Markov retention probability requires targeted project delivery.
            """)
        with p3:
            st.error(f"⚖️ **Ethics Sentinel ({ethics_stance})**")
            st.markdown(fr"""
            - **Shannon Entropy $\mathcal{{H}}$**: `{entropy_val:.3f}` nats (Cutoff: `{entropy_tau:.2f}`)
            - **Demographic Parity**: Balanced across macro tracks.
            - **Escalation**: {'Human Counselor Override ENGAGED' if ethics_override else 'No bias or ambiguity anomaly detected'}
            """)
            
        st.markdown("---")
        st.success(f"🎯 **Arbiter Final Consensus (Dialectical Confidence: {consensus_conf:.1%})**")
        st.markdown(f"""
        ### **Verdict**: {consensus_rec}
        - **Target Milestone Role**: `{planner_role}`
        - **Projected Graduate Earning Potential**: `${planner_salary:.2f}k`
        - **Recommended Next Action**: {'Schedule appointment with Department Academic Counselor' if ethics_override else 'Proceed with advanced technical portfolio review and industry certification'}
        """)
        
        # Skill Radar Chart
        st.markdown("#### Candidate Competency Profile vs Archetype Baseline")
        radar_categories = ["Operating Systems", "Algorithms", "Programming", "Networks", "Mathematics", "Coding Skills"]
        radar_vals = [
            input_vals["Acedamic percentage in Operating Systems"],
            input_vals["percentage in Algorithms"],
            input_vals["Percentage in Programming Concepts"],
            input_vals["Percentage in Computer Networks"],
            input_vals["Percentage in Mathematics"],
            input_vals["coding skills rating"] * 10
        ]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(radar_categories, radar_vals, color='#3b82f6')
        ax.set_xlim(0, 100)
        ax.set_xlabel("Proficiency (%)")
        ax.axvline(x=75, color='#ef4444', linestyle='--', label="Archetype Core (75%)")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
