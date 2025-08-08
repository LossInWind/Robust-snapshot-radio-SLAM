from __future__ import annotations

import math
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Optional, Tuple, List, Dict

import numpy as np

from .linear_algebra import matrix_inverse


@dataclass
class SlamState:
    xn: Optional[np.ndarray] = None  # shape (4,)
    xl: Optional[np.ndarray] = None  # shape (2, n_landmarks)
    L: Optional[np.ndarray] = None   # cost grid (M, N)
    inliers: Optional[np.ndarray] = None  # shape (m_k,), bool mask of inliers
    los_candidate: Optional[int] = None
    dt: Optional[np.ndarray] = None  # shape (3,)


def rotation_matrix(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def compute_parameters(tx: np.ndarray,
                       y: np.ndarray,
                       theta: float,
                       los_candidate: Optional[int],
                       gamma: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    I = np.eye(2)
    m_k = y.shape[1]

    MU = np.zeros((2, m_k))
    ETA = np.zeros((2, m_k))
    ETA_bar = np.zeros((2, m_k))
    HH = np.zeros((2, 3, m_k))
    AA = np.zeros((3, 3, m_k))
    bb = np.zeros((3, m_k))
    UI = np.zeros((2, m_k))
    VI = np.zeros((2, m_k))

    rot_tx = rotation_matrix(float(tx[2]))
    rot_rx = rotation_matrix(theta)

    for j in range(m_k):
        tau = float(y[0, j])
        aod = float(y[1, j])
        aoa = float(y[2, j])

        ui = rot_tx @ np.array([math.cos(aod), math.sin(aod)], dtype=float)
        vi = rot_rx @ np.array([math.cos(aoa), math.sin(aoa)], dtype=float)

        mu = tx[:2] - tau * vi
        H = np.hstack([I, -vi.reshape(2, 1)])

        if los_candidate is not None and j == los_candidate:
            eta = np.zeros((2,), dtype=float)
            eta_bar = np.zeros((2,), dtype=float)

            A = gamma[j] * (H.T @ H)
            b = gamma[j] * (H.T @ mu)
        else:
            eta = ui + vi
            norm_eta = np.linalg.norm(eta)
            eta_bar = (eta / norm_eta) if norm_eta > 0 else np.zeros_like(eta)

            tmp = H.T @ (I - np.outer(eta_bar, eta_bar))
            A = gamma[j] * (tmp @ H)
            b = gamma[j] * (tmp @ mu)

        MU[:, j] = mu
        ETA[:, j] = eta
        ETA_bar[:, j] = eta_bar
        HH[:, :, j] = H
        bb[:, j] = b
        AA[:, :, j] = A
        UI[:, j] = ui
        VI[:, j] = vi

    return MU, ETA, ETA_bar, HH, AA, bb, UI, VI


def compute_model(AA: np.ndarray, bb: np.ndarray, idx: np.ndarray) -> np.ndarray:
    b = np.zeros((3,), dtype=float)
    A = np.zeros((3, 3), dtype=float)
    for j in idx:
        b += bb[:, j]
        A += AA[:, :, j]
    invA = matrix_inverse(A)
    x_hat = invA @ b
    return x_hat


def compute_cost(x_hat: np.ndarray, HH: np.ndarray, MU: np.ndarray, ETA_bar: np.ndarray) -> np.ndarray:
    m_k = HH.shape[2]
    nu2 = np.zeros((m_k,), dtype=float)
    for j in range(m_k):
        tmp = HH[:, :, j] @ x_hat - MU[:, j]
        proj = float(ETA_bar[:, j].T @ tmp)
        nu = tmp - proj * ETA_bar[:, j]
        nu2[j] = float(nu.T @ nu)
    return nu2


def is_feasible(x_hat: np.ndarray,
                y: np.ndarray,
                MU: np.ndarray,
                ETA: np.ndarray,
                HH: np.ndarray,
                idx: np.ndarray,
                eta2_threshold: float) -> bool:
    if np.any(y[0, :] - x_hat[2] < 0):
        return False

    los_idx = int(np.argmin(y[0, :]))

    for j in idx:
        eta = ETA[:, j]
        eta2 = float(eta.T @ eta)
        if j == los_idx and eta2 < eta2_threshold:
            gamma = 1.0
        else:
            denom = (y[0, j] - x_hat[2]) * eta2
            if denom == 0:
                return False
            gamma = float(eta.T @ (HH[:, :, j] @ x_hat - MU[:, j])) / denom
        if gamma < 0 or gamma > 1:
            return False
    return True


def determine_consensus_set(idx_vec: np.ndarray,
                            y: np.ndarray,
                            MU: np.ndarray,
                            ETA: np.ndarray,
                            ETA_bar: np.ndarray,
                            HH: np.ndarray,
                            AA: np.ndarray,
                            bb: np.ndarray,
                            epsilon: float,
                            eta2_threshold: float) -> Optional[np.ndarray]:
    # Compute model
    x_hat = compute_model(AA, bb, idx_vec)

    if is_feasible(x_hat, y, MU, ETA, HH, idx_vec, eta2_threshold):
        # Compute residuals
        nu2 = compute_cost(x_hat, HH, MU, ETA_bar)
        outliers = nu2 > epsilon
        # Require at least as many inliers as constraints
        if np.sum(~outliers) < idx_vec.shape[0]:
            return None
        return outliers
    else:
        return None


def fit_model(outliers: np.ndarray,
              y: np.ndarray,
              MU: np.ndarray,
              ETA: np.ndarray,
              ETA_bar: np.ndarray,
              HH: np.ndarray,
              AA: np.ndarray,
              bb: np.ndarray,
              gamma: np.ndarray,
              epsilon: float,
              eta2_threshold: float) -> Tuple[float, np.ndarray]:
    idx = np.flatnonzero(~outliers)
    x_hat = compute_model(AA, bb, idx)

    if is_feasible(x_hat, y, MU, ETA, HH, idx, eta2_threshold):
        nu2 = compute_cost(x_hat, HH, MU, ETA_bar)
        nu2 = nu2.copy()
        nu2[outliers] = epsilon
        L = float(np.dot(gamma, nu2))
        return L, x_hat
    else:
        return math.nan, x_hat


def localization(tx: np.ndarray,
                 y: np.ndarray,
                 power: np.ndarray,
                 los_candidate: Optional[int],
                 params: Dict) -> Tuple[SlamState, float]:
    R = rotation_matrix

    m_k = y.shape[1]

    gain = 10.0 ** (power / 10.0)
    epsilon = float(params['epsilon'])
    eta2_threshold = float(params['eta2_threshold'])

    # Determine all possible measurement combinations
    if los_candidate is None:
        comb_list = [np.array(c, dtype=int) for c in combinations(range(m_k), 4)]
    else:
        comb_list = [np.array([los_candidate, j], dtype=int) for j in range(m_k) if j != los_candidate]

    # Determine candidate orientations
    if los_candidate is None:
        theta_grid = np.linspace(-math.pi, math.pi, int(params['N']))
    else:
        v = -rotation_matrix(-y[2, los_candidate]) @ rotation_matrix(float(tx[2])) @ np.array([
            math.cos(y[1, los_candidate]), math.sin(y[1, los_candidate])
        ])
        theta_grid = np.array([math.atan2(float(v[1]), float(v[0]))])

    time_start = time.perf_counter()

    N = theta_grid.shape[0]
    M = len(comb_list)
    LL = np.full((M, N), math.nan, dtype=float)
    X_hat = np.zeros((3, M, N), dtype=float)
    OUTLIERS = np.ones((m_k, M, N), dtype=bool)

    for n in range(N):
        theta = float(theta_grid[n])
        MU, ETA, ETA_bar, HH, AA, bb, _, _ = compute_parameters(tx, y, theta, los_candidate, gain)
        for m in range(M):
            outliers = determine_consensus_set(comb_list[m], y, MU, ETA, ETA_bar, HH, AA, bb, epsilon, eta2_threshold)
            if outliers is not None:
                L, x_hat = fit_model(outliers, y, MU, ETA, ETA_bar, HH, AA, bb, gain, epsilon, eta2_threshold)
                if not math.isnan(L):
                    LL[m, n] = L
                    X_hat[:, m, n] = x_hat
                    OUTLIERS[:, m, n] = outliers

    dt = time.perf_counter() - time_start

    # Estimate UE state: choose minimum cost ignoring NaN
    if np.all(np.isnan(LL)):
        # no valid solution
        return SlamState(xn=None, xl=None, L=LL, inliers=None, los_candidate=los_candidate), dt

    j_flat = int(np.nanargmin(LL))
    i_m, j_n = divmod(j_flat, N)
    x_hat = np.array([X_hat[0, i_m, j_n], X_hat[1, i_m, j_n], theta_grid[j_n], X_hat[2, i_m, j_n]], dtype=float)

    obj = SlamState()
    obj.xn = x_hat
    obj.xl = None
    obj.L = LL
    obj.inliers = ~OUTLIERS[:, i_m, j_n]
    obj.los_candidate = los_candidate

    return obj, dt


def h_func(tx: np.ndarray, rx: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sp = p
    dx1 = tx[0] - sp[0]
    dy1 = tx[1] - sp[1]
    dx2 = sp[0] - rx[0]
    dy2 = sp[1] - rx[1]
    d12 = dx1 * dx1 + dy1 * dy1
    d22 = dx2 * dx2 + dy2 * dy2
    d1 = math.sqrt(d12)
    d2 = math.sqrt(d22)

    mu = np.array([
        d1 + d2 + rx[3],
        math.atan2(-dy1, -dx1) - tx[2],
        math.atan2(dy2, dx2) - rx[2],
    ], dtype=float)

    # Jacobians
    Hn = np.array([
        [-dx2 / d2, -dy2 / d2, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [dy2 / d22, -dx2 / d22, -1.0, 0.0],
    ], dtype=float)

    Hl = np.array([
        [dx2 / d2 - dx1 / d1, dy2 / d2 - dy1 / d1],
        [dy1 / d12, -dx1 / d12],
        [-dy2 / d22, dx2 / d22],
    ], dtype=float)

    return mu, Hn, Hl


def landmark_cost(tx: np.ndarray, y: np.ndarray, xn: np.ndarray, xl: np.ndarray, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    mu, _, Hl = h_func(tx, xn, xl)

    nu = y.copy() - mu
    # wrap angles to (-pi, pi)
    nu[1] = (nu[1] + math.pi) % (2 * math.pi) - math.pi
    nu[2] = (nu[2] + math.pi) % (2 * math.pi) - math.pi

    A = Hl.T @ W @ Hl
    b = Hl.T @ W @ nu
    cost = -0.5 * float(nu.T @ W @ nu)
    return A, b, cost


def mapping(obj: SlamState,
            tx: np.ndarray,
            y: np.ndarray,
            power: np.ndarray,
            params: Dict) -> Tuple[SlamState, float]:
    max_iterations = int(params['max_iterations'])
    convergence_threshold = float(params['convergence_threshold'])
    W = matrix_inverse(params['R'])

    time_start = time.perf_counter()

    gain = 10.0 ** (power / 10.0)
    los_candidate = obj.los_candidate
    inliers = obj.inliers.copy() if obj.inliers is not None else np.zeros((y.shape[1],), dtype=bool)
    xn = obj.xn

    if xn is None:
        return obj, time.perf_counter() - time_start

    if los_candidate is not None and 0 <= los_candidate < inliers.shape[0]:
        inliers[los_candidate] = False

    m_k = y.shape[1]
    n_k = int(np.sum(inliers))
    m = np.zeros((2, n_k), dtype=float)

    los_idx = int(np.argmin(y[0, :]))
    i = 0

    for j in range(m_k):
        if inliers[j]:
            MU, ETA, _, H, _, _, UI, VI = compute_parameters(tx, y[:, j:j + 1], float(xn[2]), None, gain[j:j + 1])
            mu = MU[:, 0]
            eta = ETA[:, 0]
            Hn = np.hstack([H[:, :, 0], np.zeros((2, 1))])  # not used directly
            tau = float(y[0, j] - xn[3])

            eta2 = float(eta.T @ eta)
            if j == los_idx and eta2 < float(params['eta2_threshold']):
                gamma = 0.5
            else:
                denom = tau * eta2
                if denom == 0:
                    gamma = 0.5
                else:
                    gamma = float(eta.T @ (H[:, :, 0] @ xn[[0, 1, 3]] - mu)) / denom

            p1 = tx[:2] + tau * gamma * UI[:, 0]
            p2 = xn[:2] + tau * (1.0 - gamma) * VI[:, 0]
            xl = (p1 + p2) / 2.0

            A, b, L = landmark_cost(tx, y[:, j].copy(), xn, xl, W)
            for n in range(1, max_iterations):
                delta = matrix_inverse(A) @ b
                A_up, b_up, L_up = landmark_cost(tx, y[:, j].copy(), xn, xl + delta, W)
                if L_up >= L:
                    xl = xl + delta
                else:
                    break
                if abs((L - L_up) / (L_up if L_up != 0 else 1.0)) < convergence_threshold:
                    break
                else:
                    A, b, L = A_up, b_up, L_up
            m[:, i] = xl
            i += 1

    obj.xl = m
    dt = time.perf_counter() - time_start
    return obj, dt