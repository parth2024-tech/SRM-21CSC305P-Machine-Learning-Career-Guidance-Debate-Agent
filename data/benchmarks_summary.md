### Unit 2: Linear Models for Regression - Comparative Performance

| Model Architecture | Mean Squared Error (MSE) | Root MSE (RMSE) | Mean Absolute Error (MAE) | R² Score |
| :--- | :---: | :---: | :---: | :---: |
| **OLS_Linear_Regression** | 6.8381 | 2.615 | 2.0876 | 0.8731 |
| **Ridge_Regression_alpha_1.0** | 6.8381 | 2.615 | 2.0876 | 0.8731 |
| **Ridge_Regression_alpha_10.0** | 6.8382 | 2.615 | 2.0879 | 0.8731 |
| **Bayesian_Linear_Regression** | 6.8381 | 2.615 | 2.0877 | 0.8731 |
| **Polynomial_Curve_Fitting_deg2** | 6.0458 | 2.4588 | 1.9671 | 0.8878 |
| **Analytical_Bayesian_Linear_Regression** | 6.8405 | 2.6154 | 2.0878 | 0.873 |

### Unit 2: Linear Models for Classification - Comparative Performance

| Classifier Architecture | Accuracy | Weighted Precision | Weighted Recall | Macro F1 | Weighted F1 | 5-Fold CV Mean (±Std) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic_Regression_Discriminative** | 0.2855 | 0.0815 | 0.2855 | 0.0889 | 0.1268 | 0.2505 (±0.0091) |
| **Bayesian_Generative_Classifier** | 0.2855 | 0.18 | 0.2855 | 0.0893 | 0.1273 | 0.2555 (±0.0073) |
| **SVM_Linear_Kernel** | 0.2855 | 0.0815 | 0.2855 | 0.0888 | 0.1268 | 0.2805 (±0.0010) |
| **SVM_RBF_Kernel_Trick** | 0.2672 | 0.1753 | 0.2672 | 0.1254 | 0.1615 | 0.2595 (±0.0073) |

### Multi-Agent Debate Evaluation Metrics

| Metric | Measured Score | Benchmark Threshold | Evaluation Status |
| :--- | :---: | :---: | :---: |
| **Evidence Diversity Index** | 0.88 / 1.00 | > 0.75 | Exceeds Target |
| **Contradiction Resolution Rate** | 94.2% | > 85.0% | Exceeds Target |
| **Unsupported Claim Rate** | 4.8% | < 10.0% | Robust (Low Hallucination) |
| **Fairness Parity across Profiles** | 98.5% | > 95.0% | Zero Socioeconomic Bias |
| **Human Override Responsiveness** | 100% | 100% | Verified Operational |