"""GRU teacher and student architectures with a shared nonnegative output head, plus the trailing 60 historical control."""

import torch
from torch import nn


def historical_control(squared_returns, window):
    """Trailing mean of squared returns, ending at and including the forecast origin.

    The forecast at origin t is (1/window) * sum of r(t-k)^2 for k = 0..window-1.
    Uses only data observed by t. Pass the full squared-return series so the
    earliest origins have their complete history.
    """
    return squared_returns.rolling(window).mean().rename(f"hc_{window}")


class GRUForecaster(nn.Module):
    """gru over 60 step window -> softplus head -> predicted ratio (always >= 0)"""

    def __init__(self, n_inputs=4, hidden_size=32, num_layers=1, dropout=0.0):
        super().__init__()
        # torch only applies dropout between stacked layers, so 1 layer student gets none
        self.gru = nn.GRU(n_inputs, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, 60, 4), only last time step hidden state used for forecast
        out, _ = self.gru(x)
        return nn.functional.softplus(self.head(out[:, -1])).squeeze(-1)


def build_model(cfg, kind):
    """kind = "teacher" or "student", sizes from config"""
    m = cfg["models"][kind]
    return GRUForecaster(len(cfg["features"]["inputs"]), m["hidden_size"], m["num_layers"], m["dropout"])