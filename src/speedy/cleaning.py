"""Data-cleaning policy. It is always fitted on the training slice only.

The functions are called inside each backtest fold and inside `fit`, so
cleaning can never carry information from the forecast period into a model.

Policy for anomalies: a store-week is flagged when it is extreme after
removing the store's own level and the move that all stores made in that
week (robust z-score > `z`), and it is not a holiday week (stores really do
differ in how they react to holidays). Flagged values are winsorised to
+-`clip` robust sigmas. They are neither deleted nor imputed.

Returns (negative department rows) are kept net: reported revenue is net of
returns and the effect on total sales is about 0.001%.
"""

import numpy as np
import pandas as pd

from .features import date_features

HOLIDAY_COLUMNS = [
    "th-1", "th0", "th1",
    "xm-2", "xm-1", "xm0", "xm1", "xm2", "xm3",
    "superbowl", "laborday", "easter",
]  # fmt: skip
REPORT_COLUMNS = ["Date", "Store", "z", "original", "cleaned"]


def protected_weeks(dates) -> np.ndarray:
    """True for weeks that are holiday weeks (never treated as anomalies)."""
    holiday_flags = date_features(dates)[HOLIDAY_COLUMNS]
    return (holiday_flags.sum(axis=1) > 0).to_numpy()


def robust_residuals(history: pd.DataFrame):
    """Log sales without the store level and the common weekly movement.

    Returns (log_sales, residuals, z_scores, median, scale), where the
    z-score is the residual centred by store and divided by a robust sigma.
    """
    log_sales = np.log(history)
    centred = log_sales - log_sales.mean()
    residuals = centred.sub(centred.median(axis=1), axis=0)
    median = residuals.median()
    scale = 1.4826 * (residuals - median).abs().median()
    scale = scale.where(scale > 0)  # a zero spread means nothing to flag
    z_scores = (residuals - median) / scale
    return log_sales, residuals, z_scores, median, scale


def winsorize_anomalies(
    history: pd.DataFrame, z: float = 5.0, clip: float = 3.5
):
    """Clip non-holiday outliers of a weeks x stores sales table.

    Returns the cleaned table and a report of the cells that were changed.
    """
    log_sales, residuals, z_scores, median, scale = robust_residuals(history)
    protected = protected_weeks(history.index)
    flagged = (z_scores.abs() > z).to_numpy() & ~protected[:, None]
    if not flagged.any():
        return history.copy(), pd.DataFrame(columns=REPORT_COLUMNS)

    clipped = median + np.sign(z_scores) * clip * scale
    cleaned_log = log_sales.where(~flagged, log_sales - residuals + clipped)
    cleaned = np.exp(cleaned_log)

    rows, cols = np.nonzero(flagged)
    report = pd.DataFrame(
        {
            "Date": history.index[rows],
            "Store": history.columns[cols],
            "z": z_scores.to_numpy()[rows, cols],
            "original": history.to_numpy()[rows, cols],
            "cleaned": cleaned.to_numpy()[rows, cols],
        }
    )
    return cleaned, report
