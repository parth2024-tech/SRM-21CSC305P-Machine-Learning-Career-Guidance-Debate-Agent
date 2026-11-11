"""
Unit 3 Unsupervised Learning & Dimensionality Reduction Module
Strictly aligned with Unit 3 Syllabus & Practice:
- K-Means Clustering (Lloyd's Algorithm, WCSS / Elbow Method)
- Gaussian Mixture Models (GMM) with EM Algorithm (E-step responsibilities, M-step parameter updates)
- Factor Analysis (FA) & Probabilistic PCA (PPCA) (Latent variable models, low-rank parameterization)
- Principal Component Analysis (PCA) (Eigenvalue decomposition, Scree plot, Kaiser criterion)
- Hierarchical Clustering (Agglomerative, Ward / Average linkage, Dendrograms)
- Clustering Evaluation & Dissimilarity Measures (Silhouette, Calinski-Harabasz, Davies-Bouldin, Euclidean vs Mahalanobis)
"""

from typing import Any

import matplotlib
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

matplotlib.use('Agg')

class Unit3UnsupervisedPipeline:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.pca_model = None
        self.fa_model = None
        self.kmeans_model = None
        self.gmm_model = None
        self.hierarchical_model = None
        self.scaler_mean = None
        self.scaler_std = None

    def fit_transform_pca(self, X: np.ndarray, n_components: int = 5) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Principal Component Analysis:
        - Maximum variance projection
        - Covariance matrix eigenvectors and eigenvalues
        - Scree plot data and explained variance ratio
        """
        self.pca_model = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = self.pca_model.fit_transform(X)
        
        explained_var = self.pca_model.explained_variance_
        explained_var_ratio = self.pca_model.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var_ratio)
        
        # Kaiser Criterion: Eigenvalues > 1.0 retain significant variance
        kaiser_components = int(np.sum(explained_var > 1.0))
        
        pca_summary = {
            "n_components": n_components,
            "eigenvalues": explained_var.tolist(),
            "explained_variance_ratio": explained_var_ratio.tolist(),
            "cumulative_variance_ratio": cumulative_var.tolist(),
            "kaiser_criterion_count": kaiser_components,
            "total_variance_explained": float(cumulative_var[-1])
        }
        return X_pca, pca_summary

    def fit_factor_analysis(self, X: np.ndarray, n_factors: int = 3) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Factor Analysis (Continuous Latent Variable Model):
        x = W * z + mu + epsilon, where z ~ N(0, I), epsilon ~ N(0, Psi)
        Decomposes variance into shared latent factors (W) and unique variances (Psi).
        """
        self.fa_model = FactorAnalysis(n_components=n_factors, random_state=self.random_state, max_iter=1000)
        X_fa = self.fa_model.fit_transform(X)
        
        loadings = self.fa_model.components_ # Shape: (n_factors, n_features)
        noise_variance = self.fa_model.noise_variance_ # Psi
        
        fa_summary = {
            "n_factors": n_factors,
            "loadings_shape": list(loadings.shape),
            "log_likelihood": float(self.fa_model.score(X)),
            "noise_variance_mean": float(np.mean(noise_variance)),
            "noise_variance_min": float(np.min(noise_variance)),
            "noise_variance_max": float(np.max(noise_variance))
        }
        return X_fa, fa_summary

    def run_kmeans_elbow(self, X: np.ndarray, k_range: range = range(2, 9)) -> dict[int, float]:
        """
        K-Means Clustering: WCSS (Within-Cluster Sum of Squares) / Elbow Curve
        Objective: min sum_{k} sum_{i in S_k} ||x_i - mu_k||^2
        """
        wcss_dict = {}
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            km.fit(X)
            wcss_dict[k] = float(km.inertia_)
        return wcss_dict

    def fit_kmeans(self, X: np.ndarray, n_clusters: int = 5) -> tuple[np.ndarray, dict[str, Any]]:
        """K-Means Hard Assignment Clustering."""
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = self.kmeans_model.fit_predict(X)
        inertia = float(self.kmeans_model.inertia_)
        
        sil = float(silhouette_score(X, labels))
        ch = float(calinski_harabasz_score(X, labels))
        db = float(davies_bouldin_score(X, labels))
        
        metrics = {
            "n_clusters": n_clusters,
            "wcss_inertia": inertia,
            "silhouette_score": sil,
            "calinski_harabasz_index": ch,
            "davies_bouldin_index": db,
            "cluster_centers": self.kmeans_model.cluster_centers_.tolist()
        }
        return labels, metrics

    def fit_gmm_em(self, X: np.ndarray, n_components: int = 5) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Gaussian Mixture Model (GMM) with Expectation-Maximization (EM):
        - E-step: Posterior responsibilities gamma(z_{nk}) = P(z_k = 1 | x_n)
        - M-step: Updating weights pi_k, means mu_k, covariances Sigma_k
        - Computes Soft Responsibilities and Log-Likelihood convergence
        """
        self.gmm_model = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            random_state=self.random_state,
            max_iter=200
        )
        self.gmm_model.fit(X)
        
        # Soft Responsibilities: gamma_nk
        responsibilities = self.gmm_model.predict_proba(X)
        hard_labels = self.gmm_model.predict(X)
        
        # Log-likelihood and Information Criteria
        log_likelihood = float(self.gmm_model.score(X) * len(X))
        bic = float(self.gmm_model.bic(X))
        aic = float(self.gmm_model.aic(X))
        
        # Entropy of responsibilities (Uncertainty measure for Debate Ethics Reviewer)
        # H(gamma) = - sum(gamma * log(gamma + 1e-12))
        entropies = -np.sum(responsibilities * np.log(responsibilities + 1e-12), axis=1)
        mean_entropy = float(np.mean(entropies))
        
        sil = float(silhouette_score(X, hard_labels))
        ch = float(calinski_harabasz_score(X, hard_labels))
        db = float(davies_bouldin_score(X, hard_labels))
        
        metrics = {
            "n_components": n_components,
            "converged": bool(self.gmm_model.converged_),
            "n_iter": int(self.gmm_model.n_iter_),
            "log_likelihood": log_likelihood,
            "bic": bic,
            "aic": aic,
            "weights": self.gmm_model.weights_.tolist(),
            "mean_assignment_entropy": mean_entropy,
            "silhouette_score": sil,
            "calinski_harabasz_index": ch,
            "davies_bouldin_index": db
        }
        return hard_labels, responsibilities, metrics

    def fit_hierarchical(self, X: np.ndarray, n_clusters: int = 5, linkage: str = 'ward') -> tuple[np.ndarray, dict[str, Any]]:
        """
        Hierarchical Agglomerative Clustering (Bottom-Up AGNES):
        Computes pairwise dissimilarity distance matrix and merges clusters
        via Ward's Minimum Variance Linkage.
        """
        # Subsample for scaling dissimilarity if X is very large
        sample_size = min(len(X), 2000)
        X_sub = X[:sample_size]
        
        self.hierarchical_model = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage=linkage)
        labels = self.hierarchical_model.fit_predict(X_sub)
        
        sil = float(silhouette_score(X_sub, labels))
        ch = float(calinski_harabasz_score(X_sub, labels))
        db = float(davies_bouldin_score(X_sub, labels))
        
        metrics = {
            "n_clusters": n_clusters,
            "linkage": linkage,
            "sample_size": sample_size,
            "silhouette_score": sil,
            "calinski_harabasz_index": ch,
            "davies_bouldin_index": db
        }
        return labels, metrics

    def compute_dissimilarity_profiles(self, sample_x: np.ndarray) -> dict[str, float]:
        """
        Unit 3 Dissimilarity Metrics:
        Calculates Euclidean vs Mahalanobis distance from cluster centroids.
        Used by the Persona Critic to detect anomalous/outlier career profiles.
        """
        if self.gmm_model is None or not hasattr(self.gmm_model, 'means_'):
            raise ValueError("GMM model must be fitted first.")
            
        distances = {}
        for k in range(self.gmm_model.n_components):
            mu_k = self.gmm_model.means_[k]
            sigma_k = self.gmm_model.covariances_[k]
            
            # Euclidean distance
            eucl = float(np.linalg.norm(sample_x - mu_k))
            
            # Mahalanobis distance: sqrt((x - mu)^T Sigma^{-1} (x - mu))
            try:
                inv_sigma = np.linalg.pinv(sigma_k)
                diff = sample_x - mu_k
                mahal = float(np.sqrt(np.maximum(0.0, np.dot(np.dot(diff, inv_sigma), diff))))
            except Exception:
                mahal = eucl
                
            distances[f"Cluster_{k}_Euclidean"] = eucl
            distances[f"Cluster_{k}_Mahalanobis"] = mahal
            
        return distances
