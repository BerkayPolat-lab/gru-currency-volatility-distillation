"""GRU teacher and student architectures with a shared nonnegative output head, plus the trailing-20 historical control."""


def historical_control(squared_returns, window):
    """Trailing mean of squared returns, ending at and including the forecast origin.

    The forecast at origin t is (1/window) * sum of r(t-k)^2 for k = 0..window-1.
    Uses only data observed by t. Pass the full squared-return series so the
    earliest origins have their complete history.
    """
    return squared_returns.rolling(window).mean().rename(f"hc_{window}")