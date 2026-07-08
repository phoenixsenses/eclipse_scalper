"""Faz 6A modelleri — dusuk karmasiklik, tam deterministik (seed'li), saf numpy.

- cusum_changepoints : degisim-noktasi tespiti (model sirasi #1)
- seeded_kmeans      : time-series clustering (#3; temporal smoothing ile)
- hmm_fit            : diagonal-Gaussian HMM, EM (#2), kmeans-init, Viterbi decode
- ari                : Adjusted Rand Index (seed/perturbation stabilite olcumu)
- Standardizer       : YALNIZ exploration'da fit (normalization leakage engeli)
"""
from __future__ import annotations
import numpy as np


class Standardizer:
    """Fit yalniz exploration verisinde; validation'a AYNI param uygulanir."""
    def __init__(self):
        self.mu = None; self.sd = None; self.fit_range = None

    def fit(self, X: np.ndarray, fit_range: tuple[int, int]) -> "Standardizer":
        self.mu = np.nanmedian(X, axis=0)
        q75, q25 = np.nanpercentile(X, 75, axis=0), np.nanpercentile(X, 25, axis=0)
        self.sd = np.where((q75 - q25) > 0, (q75 - q25), 1.0)
        self.fit_range = fit_range
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mu) / self.sd
        # impute: exploration medyani = 0 (transform sonrasi)
        return np.nan_to_num(np.clip(Z, -6, 6), nan=0.0)


def cusum_changepoints(x: np.ndarray, k: float = 0.5, h: float = 8.0) -> list[int]:
    """Tek boyutlu CUSUM; degisim noktasi indeksleri."""
    x = np.nan_to_num(x, nan=np.nanmedian(x))
    mu = np.median(x); sd = np.std(x) or 1.0
    z = (x - mu) / sd
    sp = sn = 0.0; cps = []
    for i, v in enumerate(z):
        sp = max(0.0, sp + v - k); sn = max(0.0, sn - v - k)
        if sp > h or sn > h:
            cps.append(i); sp = sn = 0.0
    return cps


def seeded_kmeans(Z: np.ndarray, k: int, seed: int, iters: int = 60,
                  smooth: int = 3) -> tuple[np.ndarray, np.ndarray, float]:
    """Deterministik k-means (kmeans++ init) + temporal mod-smoothing.
    Doner: labels, centers, inertia."""
    rng = np.random.RandomState(seed)
    n = Z.shape[0]
    centers = [Z[rng.randint(n)]]
    for _ in range(k - 1):
        d2 = np.min([np.sum((Z - c) ** 2, axis=1) for c in centers], axis=0)
        p = d2 / d2.sum() if d2.sum() > 0 else None
        centers.append(Z[rng.choice(n, p=p)])
    C = np.array(centers)
    for _ in range(iters):
        d = ((Z[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        lab = d.argmin(axis=1)
        newC = np.array([Z[lab == j].mean(axis=0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(newC, C, atol=1e-9):
            C = newC; break
        C = newC
    d = ((Z[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    lab = d.argmin(axis=1)
    inertia = float(d[np.arange(n), lab].sum())
    if smooth > 1:  # temporal mod filtresi (sequence tutarliligi)
        out = lab.copy()
        for i in range(n):
            w = lab[max(0, i - smooth):i + smooth + 1]
            out[i] = np.bincount(w, minlength=k).argmax()
        lab = out
    return lab, C, inertia


def hmm_fit(Z: np.ndarray, k: int, seed: int, iters: int = 25):
    """Diagonal-Gaussian HMM (EM). kmeans init. Doner: labels(viterbi), A, means, ll."""
    n, d = Z.shape
    lab0, C, _ = seeded_kmeans(Z, k, seed, smooth=1)
    means = C.copy()
    var = np.array([np.maximum(Z[lab0 == j].var(axis=0), 1e-3) if (lab0 == j).any()
                    else np.ones(d) for j in range(k)])
    A = np.full((k, k), 1.0 / k)
    for i in range(k):
        A[i] = 0.1 / (k - 1) if k > 1 else 1.0
        A[i, i] = 0.9
    pi = np.full(k, 1.0 / k)

    def log_emis():
        le = np.zeros((n, k))
        for j in range(k):
            le[:, j] = -0.5 * (((Z - means[j]) ** 2 / var[j]).sum(axis=1)
                               + np.log(2 * np.pi * var[j]).sum())
        return le

    ll_prev = -np.inf
    for _ in range(iters):
        le = log_emis()
        # forward-backward (log-space, scaled)
        la = np.zeros((n, k)); lb = np.zeros((n, k))
        la[0] = np.log(pi + 1e-300) + le[0]
        for t in range(1, n):
            mx = la[t - 1].max()
            la[t] = le[t] + mx + np.log(np.exp(la[t - 1] - mx) @ A + 1e-300)
        for t in range(n - 2, -1, -1):
            v = lb[t + 1] + le[t + 1]
            mx = v.max()
            lb[t] = mx + np.log(A @ np.exp(v - mx) + 1e-300)
        lg = la + lb
        lg -= lg.max(axis=1, keepdims=True)
        g = np.exp(lg); g /= g.sum(axis=1, keepdims=True)
        # M-step
        pi = g[0] / g[0].sum()
        num = np.zeros((k, k))
        for t in range(n - 1):
            v = la[t][:, None] + np.log(A + 1e-300) + (le[t + 1] + lb[t + 1])[None, :]
            v -= v.max()
            xi = np.exp(v); xi /= xi.sum()
            num += xi
        A = num / np.maximum(num.sum(axis=1, keepdims=True), 1e-300)
        w = g.sum(axis=0)
        means = (g.T @ Z) / w[:, None]
        for j in range(k):
            var[j] = np.maximum((g[:, j:j + 1] * (Z - means[j]) ** 2).sum(axis=0) / w[j], 1e-3)
        mx = la[-1].max()
        ll = mx + np.log(np.exp(la[-1] - mx).sum())
        if abs(ll - ll_prev) < 1e-3:
            break
        ll_prev = ll
    # viterbi
    le = log_emis()
    dp = np.log(pi + 1e-300) + le[0]; back = np.zeros((n, k), dtype=int)
    for t in range(1, n):
        v = dp[:, None] + np.log(A + 1e-300)
        back[t] = v.argmax(axis=0)
        dp = v.max(axis=0) + le[t]
    lab = np.zeros(n, dtype=int); lab[-1] = dp.argmax()
    for t in range(n - 2, -1, -1):
        lab[t] = back[t + 1][lab[t + 1]]
    return lab, A, means, float(ll_prev)


def ari(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand Index (label-permutation'a dayanikli stabilite olcusu)."""
    a = np.asarray(a); b = np.asarray(b)
    n = len(a)
    ct = {}
    for x, y in zip(a, b):
        ct[(x, y)] = ct.get((x, y), 0) + 1
    sum_comb = sum(v * (v - 1) / 2 for v in ct.values())
    a_cnt = {}; b_cnt = {}
    for x in a: a_cnt[x] = a_cnt.get(x, 0) + 1
    for y in b: b_cnt[y] = b_cnt.get(y, 0) + 1
    sa = sum(v * (v - 1) / 2 for v in a_cnt.values())
    sb = sum(v * (v - 1) / 2 for v in b_cnt.values())
    tot = n * (n - 1) / 2
    exp = sa * sb / tot if tot else 0.0
    mx = (sa + sb) / 2
    return float((sum_comb - exp) / (mx - exp)) if mx != exp else 1.0


def transition_matrix(labels: np.ndarray, k: int) -> np.ndarray:
    M = np.zeros((k, k))
    for i in range(len(labels) - 1):
        M[labels[i], labels[i + 1]] += 1
    return M / np.maximum(M.sum(axis=1, keepdims=True), 1)


def transition_entropy(A: np.ndarray) -> float:
    p = A[A > 0]
    return float(-(p * np.log2(p)).sum() / A.shape[0])
