"""
Unit 4 Sequential Data, Markov Models, HMM & LDS Module
Strictly aligned with Unit 4 Syllabus:
- Sequential Data Properties & Transition Dynamics (Autocorrelation, Lag-k features)
- First-Order Discrete Markov Chains (Transition matrix A, Multi-step prediction A^k, Stationary distribution pi*)
- Hidden Markov Models (HMM) (Evaluation: Forward-Backward with scaling factors, Decoding: Viterbi Algorithm)
- Linear Dynamical Systems (LDS) / Kalman Filter (Continuous state evolution, Kalman Gain, Predict/Update steps)
"""

import numpy as np
from typing import Dict, Any, Tuple, List

class CareerMarkovChain:
    """
    First-Order Markov Chain Modeling for Career Progression:
    - First-order Markov property: P(S_t | S_{1:t-1}) = P(S_t | S_{t-1})
    - State space: S = {Junior/Graduate, Mid-Level Engineer, Senior Specialist, Lead Architect, Executive/Director}
    - Transition matrix A where sum_j A_{ij} = 1
    - Stationary distribution pi* satisfying pi* = pi* A
    """
    def __init__(self, states: List[str] = None):
        if states is None:
            self.states = [
                "1_Junior_Associate",
                "2_Mid_Engineer",
                "3_Senior_Specialist",
                "4_Lead_Architect",
                "5_Director_CTO"
            ]
        else:
            self.states = states
        self.n_states = len(self.states)
        self.transition_matrix = self._initialize_realistic_career_matrix()

    def _initialize_realistic_career_matrix(self) -> np.ndarray:
        # Realistic corporate career transition matrix with upward mobility and retention
        A = np.array([
            [0.25, 0.60, 0.12, 0.03, 0.00],  # From Junior
            [0.05, 0.35, 0.45, 0.12, 0.03],  # From Mid
            [0.02, 0.08, 0.40, 0.40, 0.10],  # From Senior
            [0.01, 0.02, 0.10, 0.52, 0.35],  # From Architect
            [0.00, 0.01, 0.04, 0.15, 0.80]   # From Director/CTO (absorbing-like)
        ])
        # Ensure row-normalization
        return A / A.sum(axis=1, keepdims=True)

    def multi_step_prediction(self, initial_state_idx: int, steps: int = 3) -> Dict[str, Any]:
        """Calculates A^k multi-step transition probability and distribution."""
        A_k = np.linalg.matrix_power(self.transition_matrix, steps)
        state_dist = A_k[initial_state_idx]
        return {
            "steps": steps,
            "starting_state": self.states[initial_state_idx],
            "transition_matrix_power_k": A_k.tolist(),
            "predicted_distribution": {self.states[i]: float(state_dist[i]) for i in range(self.n_states)},
            "most_likely_role": self.states[int(np.argmax(state_dist))]
        }

    def compute_stationary_distribution(self) -> np.ndarray:
        """
        Solves for stationary distribution pi* = pi* A using eigenvector decomposition:
        Left eigenvector corresponding to eigenvalue lambda = 1.
        """
        eigenvalues, left_eigenvectors = np.linalg.eig(self.transition_matrix.T)
        # Find index of eigenvalue closest to 1.0
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(left_eigenvectors[:, idx])
        stationary = stationary / np.sum(stationary)
        return stationary


class CareerHiddenMarkovModel:
    """
    Hidden Markov Model (HMM) for Latent Competency & Career Milestones:
    Hidden States Z: {Novice (Z0), Competent (Z1), Expert (Z2)}
    Observations X: Career Milestone Assessments {Academic Pass (0), Project Deliverable (1), High-Impact Innovation (2)}
    Solves:
      1. Decoding: Viterbi Algorithm (MAP path: z* = argmax P(Z, X | lambda))
      2. Evaluation: Forward-Backward Algorithm with scaling factors c_t to prevent underflow
    """
    def __init__(self):
        self.states = ["Novice", "Competent", "Expert"]
        self.observations = ["Academic_Pass", "Project_Deliverable", "Innovation_Award"]
        self.n_states = len(self.states)
        self.n_obs = len(self.observations)
        
        # Prior pi
        self.pi = np.array([0.70, 0.25, 0.05])
        
        # Transition matrix A
        self.A = np.array([
            [0.60, 0.35, 0.05],
            [0.10, 0.65, 0.25],
            [0.02, 0.18, 0.80]
        ])
        
        # Emission matrix B: P(Observation | Hidden State)
        self.B = np.array([
            [0.70, 0.25, 0.05],  # Novice: high academic pass, low innovation
            [0.20, 0.60, 0.20],  # Competent: high project deliverables
            [0.05, 0.35, 0.60]   # Expert: high innovation awards
        ])

    def forward_backward_scaled(self, obs_seq: List[int]) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Scaled Forward-Backward Algorithm:
        Computes forward variables alpha_t(i) and backward variables beta_t(i)
        with scaling factor c_t = 1 / sum_i alpha_t(i) to eliminate floating-point underflow.
        Returns:
          log_likelihood: ln P(X | lambda)
          gamma: Smoothed state posteriors P(Z_t = i | X)
        """
        T = len(obs_seq)
        alpha = np.zeros((T, self.n_states))
        beta = np.zeros((T, self.n_states))
        c = np.zeros(T)

        # 1. Forward Pass with scaling
        alpha[0] = self.pi * self.B[:, obs_seq[0]]
        c[0] = 1.0 / (np.sum(alpha[0]) + 1e-15)
        alpha[0] *= c[0]

        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, obs_seq[t]]
            c[t] = 1.0 / (np.sum(alpha[t]) + 1e-15)
            alpha[t] *= c[t]

        # Log Likelihood = - sum_{t=0}^{T-1} ln(c_t)
        log_likelihood = -float(np.sum(np.log(c + 1e-15)))

        # 2. Backward Pass with scaling
        beta[T-1] = c[T-1]
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.A[i, :] * self.B[:, obs_seq[t+1]] * beta[t+1]) * c[t]

        # 3. Smoothed Posterior Gamma: gamma_t(i) = P(Z_t = i | X)
        gamma = alpha * beta
        gamma = gamma / (np.sum(gamma, axis=1, keepdims=True) + 1e-15)

        return log_likelihood, gamma, c

    def viterbi_decode(self, obs_seq: List[int]) -> Tuple[List[str], float]:
        """
        Viterbi Algorithm (Decoding Problem):
        Finds the single most likely hidden career competency trajectory:
        z* = argmax_{z_1...z_T} P(z_1...z_T, x_1...x_T)
        Uses log-probabilities to ensure numerical stability.
        """
        T = len(obs_seq)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        log_pi = np.log(self.pi + 1e-15)
        log_A = np.log(self.A + 1e-15)
        log_B = np.log(self.B + 1e-15)

        # Initialization
        delta[0] = log_pi + log_B[:, obs_seq[0]]

        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                scores = delta[t-1] + log_A[:, j]
                psi[t, j] = int(np.argmax(scores))
                delta[t, j] = np.max(scores) + log_B[j, obs_seq[t]]

        # Termination & Backtracking
        best_path_idx = [0] * T
        best_path_idx[T-1] = int(np.argmax(delta[T-1]))
        max_log_prob = float(np.max(delta[T-1]))

        for t in range(T-2, -1, -1):
            best_path_idx[t] = psi[t+1, best_path_idx[t+1]]

        decoded_states = [self.states[idx] for idx in best_path_idx]
        return decoded_states, max_log_prob


class CareerKalmanFilter:
    """
    Linear Dynamical System (LDS) / Kalman Filter:
    Continuously models latent career skill velocity and proficiency under noisy quarterly metrics.
    State: z_t = [Skill_Proficiency, Skill_Growth_Rate]^T
    System Dynamics:
      State evolution:    z_t = A z_{t-1} + w_t,   w_t ~ N(0, Q)
      Observation model:  x_t = C z_t + v_t,       v_t ~ N(0, R)
    Algorithms:
      Predict Step:   mu_t^- = A mu_{t-1},   P_t^- = A P_{t-1} A^T + Q
      Update Step:    K_t = P_t^- C^T (C P_t^- C^T + R)^{-1}
                      mu_t = mu_t^- + K_t (x_t - C mu_t^-)
                      P_t = (I - K_t C) P_t^-
    """
    def __init__(self, dt: float = 1.0, q_var: float = 0.05, r_var: float = 0.5):
        self.dt = dt
        # State transition matrix A
        self.A = np.array([
            [1.0, dt],
            [0.0, 1.0]
        ])
        # Observation matrix C: measures skill proficiency directly
        self.C = np.array([[1.0, 0.0]])
        # Process noise covariance Q
        self.Q = np.array([
            [0.25 * (dt**4) * q_var, 0.5 * (dt**3) * q_var],
            [0.5 * (dt**3) * q_var,  (dt**2) * q_var]
        ])
        # Measurement noise covariance R
        self.R = np.array([[r_var]])
        self.I = np.eye(2)

    def filter_trajectory(self, noisy_measurements: List[float], init_proficiency: float = 50.0) -> Dict[str, Any]:
        """Runs the Kalman filter over a sequence of noisy performance ratings."""
        T = len(noisy_measurements)
        mu = np.array([[init_proficiency], [1.0]])  # Initial state [prof, velocity]
        P = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        
        filtered_states = []
        kalman_gains = []
        variances = []
        innovations = []

        for t in range(T):
            # 1. Predict Step
            mu_pred = self.A @ mu
            P_pred = self.A @ P @ self.A.T + self.Q

            # 2. Update Step
            z_obs = np.array([[noisy_measurements[t]]])
            y_tilde = z_obs - self.C @ mu_pred  # Innovation
            S = self.C @ P_pred @ self.C.T + self.R
            K = P_pred @ self.C.T @ np.linalg.inv(S) # Kalman Gain

            mu = mu_pred + K @ y_tilde
            P = (self.I - K @ self.C) @ P_pred

            filtered_states.append(float(mu[0, 0]))
            kalman_gains.append(float(K[0, 0]))
            variances.append(float(P[0, 0]))
            innovations.append(float(y_tilde[0, 0]))

        return {
            "time_steps": list(range(1, T + 1)),
            "noisy_measurements": [float(m) for m in noisy_measurements],
            "filtered_proficiencies": filtered_states,
            "kalman_gains": kalman_gains,
            "estimation_variances": variances,
            "innovations": innovations,
            "final_estimated_proficiency": float(filtered_states[-1]),
            "final_estimated_growth_velocity": float(mu[1, 0])
        }
