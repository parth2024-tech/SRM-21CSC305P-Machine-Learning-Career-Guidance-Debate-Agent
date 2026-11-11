"""
Unit 5 Advanced Predictive Modelling & Ensemble Learning Module
Strictly aligned with SRM 21CSC305P Syllabus:
- Bayesian Model Averaging (BMA)
- Classification and Regression Trees (CART)
- AdaBoost / Boosting algorithms
- Random Forest & Gradient Boosted Decision Trees (Ensemble Learning)
- Dialectical Persona Debate Integration (Persona 1: Planner, Persona 2: Critic, Persona 3: Ethics Reviewer)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

class Unit5EnsemblePipeline:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.cart_model = None
        self.adaboost_model = None
        self.rf_model = None
        self.gb_model = None

    def fit_cart_classifier(self, X_train: np.ndarray, y_train: np.ndarray, max_depth: int = 5) -> Tuple[Any, Dict[str, Any]]:
        """CART (Classification and Regression Trees) using Gini impurity criterion."""
        self.cart_model = DecisionTreeClassifier(
            criterion='gini',
            max_depth=max_depth,
            random_state=self.random_state
        )
        self.cart_model.fit(X_train, y_train)
        
        feature_importance = self.cart_model.feature_importances_
        return self.cart_model, {
            "max_depth": max_depth,
            "n_leaves": int(self.cart_model.get_n_leaves()),
            "feature_importances": feature_importance.tolist()
        }

    def fit_adaboost(self, X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 50, learning_rate: float = 0.8) -> Tuple[Any, Dict[str, Any]]:
        """Adaptive Boosting (AdaBoost): Sequentially reweights misclassified sample points."""
        base_estimator = DecisionTreeClassifier(max_depth=2, random_state=self.random_state)
        self.adaboost_model = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=self.random_state
        )
        self.adaboost_model.fit(X_train, y_train)
        return self.adaboost_model, {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "estimator_weights": self.adaboost_model.estimator_weights_.tolist() if hasattr(self.adaboost_model, "estimator_weights_") else []
        }

    def fit_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 100) -> Tuple[Any, Dict[str, Any]]:
        """Random Forest: Bagging + Random Subspace method."""
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=8,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train)
        return self.rf_model, {
            "n_estimators": n_estimators,
            "oob_score_supported": True,
            "feature_importances": self.rf_model.feature_importances_.tolist()
        }

    def fit_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 100, learning_rate: float = 0.1) -> Tuple[Any, Dict[str, Any]]:
        """Gradient Boosted Decision Trees (GBDT): Gradient descent in function space."""
        self.gb_model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=4,
            random_state=self.random_state
        )
        self.gb_model.fit(X_train, y_train)
        return self.gb_model, {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "train_loss": [float(l) for l in self.gb_model.train_score_]
        }

    def bayesian_model_averaging(self, models: List[Any], model_priors: List[float], X: np.ndarray) -> np.ndarray:
        """
        Bayesian Model Averaging (BMA):
        P(y | x, D) = sum_{m=1}^M P(y | x, M_m) * P(M_m | D)
        Weighting model predictions by marginal posterior model probabilities.
        """
        normalized_weights = np.array(model_priors) / np.sum(model_priors)
        prob_accum = np.zeros((len(X), len(models[0].classes_)))
        
        for model, w in zip(models, normalized_weights):
            prob_accum += w * model.predict_proba(X)
            
        return prob_accum

    def evaluate_all(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Calculates Accuracy, Precision, Recall, Macro F1, Weighted F1 across models."""
        models = {
            "Single Decision Tree (CART)": self.cart_model,
            "AdaBoost Ensemble": self.adaboost_model,
            "Random Forest Ensemble": self.rf_model,
            "Gradient Boosted Trees (GBDT)": self.gb_model
        }
        
        results = {}
        for name, model in models.items():
            if model is None:
                continue
            y_pred = model.predict(X_test)
            results[name] = {
                "Accuracy": float(accuracy_score(y_test, y_pred)),
                "Precision (Weighted)": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                "Recall (Weighted)": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                "Macro F1": float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
                "Weighted F1": float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            }
        return results
