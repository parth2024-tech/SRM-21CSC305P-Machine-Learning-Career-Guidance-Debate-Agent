"""
Unit 2: Linear Models for Regression Implementation
Implements:
  1. Maximum Likelihood Estimation (MLE) - Least Squares (OLS)
  2. Ridge Regression (L2 regularization)
  3. Bayesian Linear Regression (Posterior weight distribution, predictive mean & variance for uncertainty)
  4. Polynomial Curve Fitting (Unit 1)
Practice 1: Implement linear regression to perform prediction.
"""

import numpy as np
from typing import Dict, Any, Tuple
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class CustomBayesianLinearRegression:
    """
    Direct Analytical Bayesian Linear Regression implementation from Bishop / Unit 2 syllabus.
    Prior: p(w) = N(0, alpha^-1 * I)
    Likelihood: p(y|X, w) = N(Xw, beta^-1 * I)
    Posterior: p(w|y) = N(m_N, S_N)
      S_N^-1 = alpha * I + beta * X^T X
      m_N = beta * S_N X^T y
    Predictive Distribution:
      p(y_* | x_*, D) = N(m_N^T x_*, sigma_N^2(x_*))
      sigma_N^2(x_*) = 1/beta + x_*^T S_N x_*
    """
    def __init__(self, alpha: float = 1.0, beta: float = 0.5):
        self.alpha = alpha  # Prior precision
        self.beta = beta    # Noise precision
        self.m_N = None
        self.S_N = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, D = X.shape
        # Add bias column
        X_design = np.hstack([np.ones((N, 1)), X])
        D_aug = D + 1
        
        # Posterior precision matrix
        S_N_inv = self.alpha * np.eye(D_aug) + self.beta * (X_design.T @ X_design)
        self.S_N = np.linalg.inv(S_N_inv)
        self.m_N = self.beta * (self.S_N @ (X_design.T @ y))
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N, D = X.shape
        X_design = np.hstack([np.ones((N, 1)), X])
        
        # Predictive mean
        y_mean = X_design @ self.m_N
        
        # Vectorized predictive variance (uncertainty): sigma^2 = 1/beta + diag(X @ S_N @ X^T)
        pred_var = (1.0 / self.beta) + np.sum((X_design @ self.S_N) * X_design, axis=1)
        return y_mean, np.sqrt(np.maximum(pred_var, 1e-6))


class RegressionSuite:
    """Trainer and Evaluator for Unit 1 & Unit 2 Regression Models."""
    def __init__(self):
        self.models = {
            "OLS_Linear_Regression": LinearRegression(),
            "Ridge_Regression_alpha_1.0": Ridge(alpha=1.0),
            "Ridge_Regression_alpha_10.0": Ridge(alpha=10.0),
            "Bayesian_Linear_Regression": BayesianRidge(max_iter=300),
            "Polynomial_Curve_Fitting_deg2": Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('linear', Ridge(alpha=1.0))
            ])
        }
        self.scaler = StandardScaler()
        self.fitted_models = {}
        self.custom_bayesian = CustomBayesianLinearRegression(alpha=1.0, beta=0.2)

    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Fits all regression models and evaluates on test split."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        metrics_summary = {}
        
        for name, model in self.models.items():
            model.fit(X_train_scaled, y_train)
            self.fitted_models[name] = model
            y_pred = model.predict(X_test_scaled)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metrics_summary[name] = {
                "MSE": round(float(mse), 4),
                "RMSE": round(float(rmse), 4),
                "MAE": round(float(mae), 4),
                "R2_Score": round(float(r2), 4)
            }
            
        # Fit custom analytical Bayesian linear regression
        self.custom_bayesian.fit(X_train_scaled, y_train)
        y_bayes_mean, y_bayes_std = self.custom_bayesian.predict(X_test_scaled)
        mse_bayes = mean_squared_error(y_test, y_bayes_mean)
        metrics_summary["Analytical_Bayesian_Linear_Regression"] = {
            "MSE": round(float(mse_bayes), 4),
            "RMSE": round(float(np.sqrt(mse_bayes)), 4),
            "MAE": round(float(mean_absolute_error(y_test, y_bayes_mean)), 4),
            "R2_Score": round(float(r2_score(y_test, y_bayes_mean)), 4),
            "Mean_Predictive_StdDev": round(float(np.mean(y_bayes_std)), 4)
        }
        
        return metrics_summary

    def predict_with_uncertainty(self, X_input: np.ndarray) -> Tuple[float, float, Tuple[float, float]]:
        """
        Predicts expected salary with 95% Bayesian Credible Interval.
        Returns: (predictive_mean, standard_deviation, (ci_lower, ci_upper))
        """
        X_scaled = self.scaler.transform(X_input)
        mean_pred, std_pred = self.custom_bayesian.predict(X_scaled)
        mean_val = float(mean_pred[0])
        std_val = float(std_pred[0])
        ci_95 = (mean_val - 1.96 * std_val, mean_val + 1.96 * std_val)
        return round(mean_val, 2), round(std_val, 2), (round(ci_95[0], 2), round(ci_95[1], 2))
