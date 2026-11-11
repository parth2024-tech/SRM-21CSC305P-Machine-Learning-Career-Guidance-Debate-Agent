# 🎓 SRMIST 21CSC305P Machine Learning Final Integrated Project
## Debate-Based Career Guidance Agent with Complementary Personas

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://localhost:8501)
[![Typst](https://img.shields.io/badge/Compiled%20with-Typst-239dad.svg)](https://typst.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

### 📌 Project & Academic Identification
* **Institution:** SRM Institute of Science and Technology, Kattankulathur
* **Department:** Department of Computing Technologies, College of Engineering and Technology
* **Course Code:** `21CSC305P` — Machine Learning
* **Project Component:** Final Integrated Project (PBL-1, PBL-2, PBL-3 & Viva-Voce)
* **Student Team:**
  - **Parth Gupta** (`RA2311026010001`) — Lead Engineer & Mathematical Modelling
  - **Mokshith Penna** (`RA2411003010044`) — Unsupervised Clustering & Sequential Formulations
  - **Ishu Gurtu** (`RA2411003010015`) — Ensemble Architecture & Empirical Evaluation
* **Faculty Supervisor:** **Dr. Divya Mohan** (Assistant Professor, Department of Computing Technologies)
* **Head of Department:** **Dr. G. Niranjana** (Professor & Head, Department of Computing Technologies)

---

## 📖 Executive Abstract
Conventional algorithmic career counseling systems rely on deterministic, single-objective classifiers that output uncalibrated point estimates, overlook longitudinal skill evolution, and lack transparent governance. 

This project formulates an integrated **Dialectical Multi-Persona Career Guidance Debate Agent** developed across all five curricular units of the SRM `21CSC305P` syllabus. Grounded on an empirical benchmark of **20,000 undergraduate student records across 39 academic, psychological, and technical attributes**, the platform models career advisement as a structured adversarial debate among three complementary algorithmic personas:
1. **The Strategic Career Planner** (Optimistic growth, Viterbi MAP trajectory decoding, Bayesian salary forecasting).
2. **The Empirical Critic** (Stress-testing feasibility via Mahalanobis distance anomalies and transition bottlenecks).
3. **The Ethics & Fairness Sentinel** (Auditing assignment entropy $\mathcal{H}(\gamma)$ and triggering automated human counselor escalation when epistemic ambiguity is detected).
4. **The Arbiter Consensus Engine** (Synthesizing a calibrated, actionable career development roadmap).

---

## 🏛️ Comprehensive 5-Unit Curricular Architecture

| Unit | Technical Scope | Algorithmic Implementations | Mathematical Principles |
| :--- | :--- | :--- | :--- |
| **Unit 1** | **Statistical Moments & EDA** | Parametric moment estimation, Quantile analysis ($Q_{25}, \text{Median}, Q_{75}$), Covariance matrix, Prior class probabilities $P(Y_k)$. | $\mu = \mathbb{E}[X]$, $\sigma^2 = \text{Var}(X)$, $\text{Cov}(X, Y) = \mathbb{E}[(X-\mu_X)(Y-\mu_Y)]$ |
| **Unit 2** | **Supervised Linear Models** | OLS Least Squares (MLE), Ridge ($L_2$), Analytical Bayesian Linear Regression (Bishop PRML Sec 3.3), Logistic Regression, Gaussian Naive Bayes, Linear & RBF SVM. | $\mathbf{w}_{\text{ML}} = (\mathbf{\Phi}^T\mathbf{\Phi})^{-1}\mathbf{\Phi}^T\mathbf{t}$, $\mathbf{S}_N^{-1} = \alpha \mathbf{I} + \beta \mathbf{\Phi}^T\mathbf{\Phi}$, $\sigma_N^2(x_*) = \frac{1}{\beta} + \phi(x_*)^T \mathbf{S}_N \phi(x_*)$ |
| **Unit 3** | **Unsupervised & Mixture Models** | Principal Component Analysis (PCA with Kaiser criterion $\lambda > 1.0$), Factor Analysis, K-Means (Lloyd), Gaussian Mixture Models with EM, Hierarchical Ward Linkage. | $\mathbf{\Sigma} v_i = \lambda_i v_i$, $\gamma_{nk} = \frac{\pi_k \mathcal{N}(x_n\|\mu_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_n\|\mu_j, \Sigma_j)}$, $D_M(x, \mu) = \sqrt{(x-\mu)^T \Sigma^{-1} (x-\mu)}$ |
| **Unit 4** | **Sequential & State-Space Models** | First-Order Markov Transition Dynamics ($A^k$, $\pi^* = \pi^* A$), Hidden Markov Models (Scaled Forward-Backward, Viterbi MAP decoding), Linear Dynamical System (Kalman Filter). | $P(S_t\|S_{t-1})$, $\pi^* = \pi^* A$, $v_t(j) = \max_i [v_{t-1}(i) a_{ij}] b_j(o_t)$, $\mathbf{K}_t = \mathbf{P}_{t\|t-1} \mathbf{C}^T (\mathbf{C}\mathbf{P}_{t\|t-1}\mathbf{C}^T + R)^{-1}$ |
| **Unit 5** | **Advanced Ensemble Learning** | Classification and Regression Trees (CART with Gini impurity), AdaBoost (exponential loss), Random Forests (Bagging), Gradient Boosted Trees (GBDT). | $I_G(t) = 1 - \sum p_i^2$, $\alpha_m = \frac{1}{2}\ln \frac{1-\epsilon_m}{\epsilon_m}$, $w_i \leftarrow w_i \exp(\alpha_m \mathbb{I}(y_i \neq h_m(x_i)))$ |
| **Capstone** | **Dialectical Multi-Persona Agent** | 3-Persona adversarial debate with calibrated Arbiter consensus synthesis and automated human counselor override. | $\mathcal{H}(\gamma) = -\sum \gamma_{nk} \ln \gamma_{nk}$, Dialectical conflict resolution |

---

## 🗂️ Project Repository Structure

```
├── .ase-project.json                    # Second Brain project governance & conventions
├── app.py                               # Full-featured Streamlit interactive web demonstration
├── course_syllabus.txt                  # SRM 21CSC305P complete course syllabus & evaluation rubric
├── generate_master_notebook.py          # Script generating master end-to-end Jupyter Notebook
├── pbl_complete_end_to_end.ipynb        # Master fully executed Jupyter Notebook (Units 1-5 + Capstone)
├── pbl1_career_guidance_debate.ipynb    # Unit 1 & 2 Google Colab notebook
├── pbl2_career_guidance_unit3_unit4.ipynb # Unit 3 & 4 Google Colab notebook
├── run_experiments.py                   # Automated scientific experiment runner
├── report_srm_format.typ                # Institutional project report in Typst (Double institutional border)
├── report_srm_format.pdf                # Publication-grade compiled PDF project report (720+ lines)
├── gpt_researcher_career_guidance_report.typ # Deep technical investigation Typst paper
├── gpt_researcher_career_guidance_report.pdf # Compiled GPT Researcher PDF report
├── data/
│   ├── career_guidance_dataset.csv      # 20,000-record benchmark dataset
│   ├── benchmarks_summary.md            # Empirical performance benchmarks table
│   └── evaluation_results.json          # Machine-readable evaluation results
├── figures/                             # Generated publication-quality figures
│   ├── fig1_correlation_matrix.png
│   ├── fig2_feature_distributions.png
│   ├── fig3_career_track_distribution.png
│   ├── fig4_skills_by_career.png
│   ├── fig5_salary_vs_skills_regression.png
│   ├── unit3_pca_scree.png
│   ├── unit3_clustering_comparison.png
│   ├── unit3_gmm_responsibilities_heatmap.png
│   ├── unit4_markov_transition_dynamics.png
│   ├── unit4_kalman_filter.png
│   ├── unit5_ensemble_comparison.png
│   └── srm_logo.jpeg
└── src/
    ├── data_loader.py                   # Dataset ingestion and preprocessing pipeline
    ├── eda.py                           # Exploratory data analysis & statistical moments
    ├── models_regression.py             # OLS, Ridge, Analytical Bayesian Linear Regression
    ├── models_classification.py         # Logistic Regression, GaussianNB, SVM (Linear & RBF)
    ├── unit3_unsupervised.py            # PCA, Factor Analysis, K-Means, GMM/EM, Mahalanobis
    ├── unit4_sequential.py              # Markov Chains, HMM (Viterbi), Kalman Filter
    ├── unit5_ensemble.py                # CART, AdaBoost, Random Forest, GBDT
    ├── metrics_evaluator.py             # Evaluation metrics suite
    └── debate_agent.py                  # Dialectical Multi-Persona Debate Engine
```

---

## 🚀 Quickstart & Demonstration Guide

### 1. Prerequisites & Virtual Environment
Ensure Python 3.10+ is installed on your workstation:
```bash
# Clone the repository
git clone https://github.com/parth2024-tech/SRM-21CSC305P-Machine-Learning-Career-Guidance-Debate-Agent.git
cd SRM-21CSC305P-Machine-Learning-Career-Guidance-Debate-Agent

# (Optional) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install streamlit numpy pandas matplotlib seaborn scikit-learn nbformat
```

### 2. Launch the Streamlit Interactive Demonstration
To launch the live demonstration platform:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.
- **Explore Tab 1 to Tab 6**: Test individual unit implementations, adjust Kalman filter noise parameters ($Q, R$), view GMM soft responsibilities, and analyze Random Forest feature importance.
- **Explore Tab 7**: Select candidate archetypes (or customize skills with sliders) and click **"Execute Dialectical Multi-Persona Debate"** to witness live multi-agent career consensus generation!

### 3. Run Experiments Headlessly
To re-run the full evaluation pipeline and regenerate all figures:
```bash
python3 run_experiments.py
```

### 4. Compile Project Reports
The project reports are authored in [Typst](https://typst.app) according to official SRM formatting guidelines:
```bash
# Compile institutional SRM report
typst compile report_srm_format.typ report_srm_format.pdf

# Compile deep technical research report
typst compile gpt_researcher_career_guidance_report.typ gpt_researcher_career_guidance_report.pdf
```

---

## 📊 Experimental Results & Benchmarks Summary

### Unit 2: Supervised Regression Benchmark (Predicting Compensation $\hat{y}$)
| Model Architecture | MSE | RMSE | MAE | $R^2$ Score | Predictive Uncertainty |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ordinary Least Squares (MLE)** | 6.8381 | 2.6150 | 2.0876 | 0.8731 | Point estimate only |
| **Ridge Regression ($\alpha=1.0$)** | 6.8381 | 2.6150 | 2.0876 | 0.8731 | Point estimate only |
| **Analytical Bayesian Linear Regression** | **6.8405** | **2.6154** | **2.0878** | **0.8730** | **$\sigma_N = \pm \$2.24k$ (95% CI)** |
| **Polynomial Curve Fitting (Degree 2)** | 6.0458 | 2.4588 | 1.9671 | 0.8878 | Point estimate only |

### Unit 3: Unsupervised Clustering Benchmarks
| Algorithm | Silhouette Score | Calinski-Harabasz Index | Davies-Bouldin Index | Assignment Type |
| :--- | :---: | :---: | :---: | :---: |
| **K-Means (Lloyd)** | 0.0482 | 874.2 | 2.84 | Hard Binary Partition |
| **Gaussian Mixture Model (EM)** | **0.0431** | **812.5** | **2.91** | **Soft Responsibilities ($\gamma_{nk}$)** |
| **Hierarchical Clustering (Ward)** | 0.0115 | 645.1 | 3.42 | Dendrogram Tree Cut |

### Unit 5: Advanced Ensemble Learning Benchmark
| Model | Accuracy | Weighted Precision | Recall | Weighted F1 | Regularization Strategy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (CART)** | 27.84% | 24.55% | 27.84% | 0.1540 | Cost-Complexity Pruning |
| **AdaBoost** | 28.24% | 12.44% | 28.24% | 0.1308 | Sample Weight Shrinkage |
| **Random Forest (100 Trees)** | **28.66%** | **34.90%** | **28.66%** | **0.1291** | **Bootstrap Aggregation ($\sqrt{D}$)** |
| **Gradient Boosted Trees (GBDT)**| 26.50% | 20.35% | 26.50% | 0.1721 | Shrinkage Rate $\eta=0.1$ |

---

## ⚖️ Academic Integrity & Anti-Plagiarism Statement
This project has been engineered from primary mathematical principles referencing:
1. Christopher M. Bishop, *Pattern Recognition and Machine Learning* (PRML), Springer, 2006.
2. Kevin Patrick Murphy, *Machine Learning: A Probabilistic Perspective*, MIT Press, 2012.

All code, derivations, analytical Bayesian linear regression formulas, Markov matrix power routines, and dialectical debate consensus algorithms are original work crafted specifically for the SRMIST 21CSC305P curriculum.
