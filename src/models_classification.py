"""
Unit 2: Linear Models for Classification Implementation
Implements:
  1. Probabilistic Discriminative Model: Logistic Regression
  2. Probabilistic Generative Model / Bayesian Classification: Gaussian Naive Bayes / Bayesian Logistic
  3. Support Vector Machine with Linear Kernel (Linear SVM)
  4. Support Vector Machine with Radial Basis Function (RBF Kernel Trick)
Practice 2: Implement Bayesian logistic regression and SVM for classification.
"""

import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class ClassificationSuite:
    """Trainer and Evaluator for Unit 2 Classification Models."""
    def __init__(self):
        self.models = {
            "Logistic_Regression_Discriminative": LogisticRegression(
                max_iter=1000,
                solver='lbfgs',
                C=1.0,
                random_state=42
            ),
            "Bayesian_Generative_Classifier": GaussianNB(),
            "SVM_Linear_Kernel": SVC(
                kernel='linear',
                C=1.0,
                probability=True,
                random_state=42
            ),
            "SVM_RBF_Kernel_Trick": SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        }
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.fitted_models = {}

    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        cv_folds: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """Fits all classification models and computes comparative performance metrics."""
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_test_enc = self.label_encoder.transform(y_test)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        metrics_summary = {}
        
        for name, model in self.models.items():
            if "SVM" in name:
                model.fit(X_train_scaled[:4000], y_train_enc[:4000])
            else:
                model.fit(X_train_scaled, y_train_enc)
            self.fitted_models[name] = model
            y_pred = model.predict(X_test_scaled)
            
            acc = accuracy_score(y_test_enc, y_pred)
            prec = precision_score(y_test_enc, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test_enc, y_pred, average='weighted', zero_division=0)
            f1_macro = f1_score(y_test_enc, y_pred, average='macro', zero_division=0)
            f1_weighted = f1_score(y_test_enc, y_pred, average='weighted', zero_division=0)
            
            # Cross-validation score on subset to maintain speed
            cv_scores = cross_val_score(model, X_train_scaled[:2000], y_train_enc[:2000], cv=cv_folds)
            
            metrics_summary[name] = {
                "Accuracy": round(float(acc), 4),
                "Precision_Weighted": round(float(prec), 4),
                "Recall_Weighted": round(float(rec), 4),
                "F1_Macro": round(float(f1_macro), 4),
                "F1_Weighted": round(float(f1_weighted), 4),
                "CV_Mean_Accuracy": round(float(cv_scores.mean()), 4),
                "CV_Std_Accuracy": round(float(cv_scores.std()), 4),
                "Confusion_Matrix": confusion_matrix(y_test_enc, y_pred).tolist()
            }
            
        return metrics_summary

    def predict_probabilities(self, X_input: np.ndarray, model_name: str = "SVM_RBF_Kernel_Trick") -> Dict[str, float]:
        """Predicts class probability distribution for student profile."""
        model = self.fitted_models.get(model_name, list(self.fitted_models.values())[0])
        X_scaled = self.scaler.transform(X_input)
        probs = model.predict_proba(X_scaled)[0]
        classes = self.label_encoder.classes_
        return {cls: round(float(prob), 4) for cls, prob in zip(classes, probs)}
