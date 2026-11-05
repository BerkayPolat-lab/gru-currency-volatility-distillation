"""Classical volatility baselines: HAR, GARCH(1,1), EWMA (RiskMetrics).

Drop into model/src/gru_vol/baselines.py.

Conventions (match the GRU pipeline):
  * prices: 1-D array of OBSERVED DEXUSEU prices (blank dates already dropped).
  * r[t] = log(P[t] / P[t-1]); r[0] is NaN.
  * An origin t is an index into r. A forecast at t uses r[..t] only and
    targets v[t, h] = mean(r[t+1 .. t+h] ** 2).
  * Every model is FIT ON TRAIN ORIGINS / TRAIN RETURNS ONLY, then frozen.
  * All forecasts are in original variance units (same scale as v_{t,5}).

Wire `origins` and split labels to the rows of examples.csv so every
baseline is scored on exactly the same examples as HC, T, S_dir and S_KD.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Common helpers
# --------------------------------------------------------------------------

def log_returns(prices: np.ndarray) -> np.ndarray:
    """r[t] = log(P[t]/P[t-1]); r[0] = NaN so indices line up with prices."""
    p = np.asarray(prices, dtype=float)
    r = np.full(p.shape, np.nan)
    r[1:] = np.diff(np.log(p))
    return r


def variance_target(r: np.ndarray, origins: np.ndarray, h: int = 5) -> np.ndarray:
    """v[t,h] = mean of r^2 over t+1..t+h (uncentered, as in Eq. 2)."""
    r2 = r ** 2
    return np.array([r2[t + 1:t + h + 1].mean() for t in origins])


def trailing_mean_sq(r: np.ndarray, origins: np.ndarray, k: int) -> np.ndarray:
    """Mean of r^2 over the k observations ending at t (inclusive)."""
    r2 = r ** 2
    out = np.array([r2[t - k + 1:t + 1].mean() for t in origins])
    if np.isnan(out).any():
        raise ValueError(f"origin too early for a {k}-observation window")
    return out


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((np.asarray(pred) - np.asarray(target)) ** 2))


# --------------------------------------------------------------------------
# HAR (Corsi 2009): v_{t,h} = b0 + b_d RV_d + b_w RV_w + b_m RV_m
# --------------------------------------------------------------------------

HAR_WINDOWS = (1, 5, 22)  # daily, weekly, monthly components


def har_features(r: np.ndarray, origins: np.ndarray,
                 windows: tuple[int, ...] = HAR_WINDOWS) -> np.ndarray:
    """Design matrix [1, RV_1, RV_5, RV_22] at each origin."""
    cols = [trailing_mean_sq(r, origins, k) for k in windows]
    return np.column_stack([np.ones(len(origins))] + cols)


def fit_har(r: np.ndarray, train_origins: np.ndarray, h: int = 5,
            log: bool = False) -> dict:
    """OLS on train origins only. log=True fits log v on log RV (log-HAR)."""
    X = har_features(r, train_origins)
    y = variance_target(r, train_origins, h)
    if log:
        X = np.column_stack([X[:, 0], np.log(X[:, 1:])])
        y = np.log(y)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    params = {"coef": coef, "log": log, "h": h}
    if log:
        # Lognormal bias correction: E[v] = exp(mu + s^2 / 2)
        resid = y - X @ coef
        params["resid_var"] = float(resid.var(ddof=X.shape[1]))
    return params


def predict_har(params: dict, r: np.ndarray, origins: np.ndarray,
                floor: float = 1e-12) -> np.ndarray:
    X = har_features(r, origins)
    if params["log"]:
        X = np.column_stack([X[:, 0], np.log(X[:, 1:])])
        return np.exp(X @ params["coef"] + 0.5 * params["resid_var"])
    # Linear HAR can (rarely) go negative; floor it so it stays a variance.
    return np.maximum(X @ params["coef"], floor)


# --------------------------------------------------------------------------
# GARCH(1,1), zero mean: s2[t+1] = w + a r[t]^2 + b s2[t]
# Parameters estimated once on train returns with `arch`, then the
# recursion is run forward by hand so causality is explicit.
# --------------------------------------------------------------------------

def fit_garch(r: np.ndarray, train_end: int) -> dict:
    """MLE on r[1..train_end] (inclusive) with `arch`. train_end = last train origin.

    Returns are scaled x100 for the optimiser (arch's recommendation);
    omega is rescaled back to original units.
    """
    from arch import arch_model

    r_train = r[1:train_end + 1]
    res = arch_model(100 * r_train, mean="Zero", vol="GARCH", p=1, q=1,
                     dist="normal", rescale=False).fit(disp="off")
    if res.convergence_flag != 0:
        raise RuntimeError(f"GARCH fit did not converge (flag {res.convergence_flag})")
    p = res.params
    omega, alpha, beta = p["omega"] / 1e4, p["alpha[1]"], p["beta[1]"]
    if alpha + beta >= 1:
        raise ValueError(f"non-stationary fit: alpha+beta={alpha + beta:.4f}")
    return {"omega": float(omega), "alpha": float(alpha), "beta": float(beta),
            "s2_init": float(np.mean(r_train ** 2)),  # train-only start value
            "loglik": float(res.loglikelihood)}


def garch_filter(params: dict, r: np.ndarray) -> np.ndarray:
    """s2_next[t] = one-step-ahead variance for day t+1, known at t."""
    w, a, b = params["omega"], params["alpha"], params["beta"]
    s2_next = np.full(r.shape, np.nan)
    s2 = params["s2_init"]                    # variance for day 1
    for t in range(1, len(r)):
        s2 = w + a * r[t] ** 2 + b * s2       # variance for day t+1
        s2_next[t] = s2
    return s2_next


def predict_garch(params: dict, r: np.ndarray, origins: np.ndarray,
                  h: int = 5) -> np.ndarray:
    """Mean of the 1..h-step-ahead variance forecasts made at t.

    E_t[s2_{t+k}] = s2_bar + (a+b)^(k-1) * (s2_{t+1} - s2_bar)
    """
    w, a, b = params["omega"], params["alpha"], params["beta"]
    persist = a + b
    s2_bar = w / (1 - persist)
    s2_1 = garch_filter(params, r)[origins]
    decay = np.mean([persist ** (k - 1) for k in range(1, h + 1)])
    return s2_bar + decay * (s2_1 - s2_bar)


# --------------------------------------------------------------------------
# EWMA / RiskMetrics: s2[t+1] = lam * s2[t] + (1 - lam) r[t]^2, flat forecast
# --------------------------------------------------------------------------

def predict_ewma(r: np.ndarray, origins: np.ndarray, lam: float = 0.94,
                 s2_init: float | None = None, init_obs: int = 60) -> np.ndarray:
    """No fitting. Initialised with the mean r^2 of the first `init_obs` returns."""
    if s2_init is None:
        s2_init = float(np.mean(r[1:init_obs + 1] ** 2))
    s2_next = np.full(r.shape, np.nan)
    s2 = s2_init
    for t in range(1, len(r)):
        s2 = lam * s2 + (1 - lam) * r[t] ** 2
        s2_next[t] = s2
    return s2_next[origins]


# --------------------------------------------------------------------------
# One call that fits everything on train and forecasts every origin
# --------------------------------------------------------------------------

def run_all(prices: np.ndarray, origins: np.ndarray, split: np.ndarray,
            h: int = 5) -> dict[str, np.ndarray]:
    """split: array of 'train'/'val'/'test' aligned with origins.

    Returns {model_name: forecasts at every origin}. Score val now; leave
    test scoring for the one-time final evaluation.
    """
    r = log_returns(prices)
    origins = np.asarray(origins)
    train = origins[np.asarray(split) == "train"]
    # Last train RETURN that is fully inside train: the last train origin's
    # own return. (Forward windows never cross splits by construction.)
    train_end = int(train.max())

    har = fit_har(r, train, h)
    loghar = fit_har(r, train, h, log=True)
    garch = fit_garch(r, train_end)

    return {
        "HC": trailing_mean_sq(r, origins, 60),
        "EWMA": predict_ewma(r, origins),
        "HAR": predict_har(har, r, origins),
        "logHAR": predict_har(loghar, r, origins),
        "GARCH": predict_garch(garch, r, origins, h),
        "_params": {"HAR": har, "logHAR": loghar, "GARCH": garch},
    }