"""Forecast MSE and high-volatility MSE in original variance-proxy units, plus each model's MSE gain over the historical control."""

import numpy as np


def mse(target, forecast):
    """Mean squared error in original variance-proxy units."""
    target, forecast = np.asarray(target, dtype=float), np.asarray(forecast, dtype=float)
    return float(np.mean((forecast - target) ** 2))


def high_vol_mse(target, forecast, cutoff):
    """MSE on examples whose realized target exceeds the training cutoff, plus how many qualify. Descriptive only."""
    target, forecast = np.asarray(target, dtype=float), np.asarray(forecast, dtype=float)
    mask = target > cutoff
    n = int(mask.sum())
    return (mse(target[mask], forecast[mask]) if n else float("nan")), n


def gain_over_control(mse_control, mse_model):
    """How much a model lowers MSE relative to the historical control (positive is better)."""
    return mse_control - mse_model


def retained_gain(mse_control, mse_student, mse_teacher):
    """G = (MSE_HC - MSE_student) / (MSE_HC - MSE_teacher). Research target: G >= 0.90."""
    return (mse_control - mse_student) / (mse_control - mse_teacher)