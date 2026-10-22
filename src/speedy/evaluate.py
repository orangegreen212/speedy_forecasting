"""Walk-forward evaluation, blend selection and uncertainty estimates.

The last five 13-week blocks of the history are forecast one after the
other ("folds"). The blend used for a fold is chosen only from the folds
before it, so the result also measures the choosing itself.
"""

from dataclasses import dataclass, field
from itertools import product
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .cleaning import winsorize_anomalies
from .data import anomalies_to_records, store_types, to_wide
from .features import date_features
from .forecasters import (
    DRIFT_GRID,
    HOLT_GRID,
    holt_fc,
    naive_drift_fc,
    naive_fc,
    prophet_fc,
    ridge_fc,
)
from .metrics import (
    block_bootstrap,
    coverage,
    interval_multipliers,
    metrics,
    wmape,
)

ALPHAS = (0.1, 1.0, 10.0)
FOLD_WEEKS = 13
N_FOLDS = 5
YEAR_AHEAD_WEEKS = 58
TENTHS = 10
HOLIDAY_FEATURES = [
    "th-1", "th0", "th1", "xm-2", "xm-1", "xm0", "xm1", "xm2", "xm3",
]  # fmt: skip


class BlendConfig(NamedTuple):
    """Choice of base-model settings and blend weights.

    The weights are whole tenths and add up to 10 (3 means 30%).
    """

    ridge_alpha: float
    holt_key: str
    drift_key: str
    w_naive: int
    w_ridge: int
    w_prophet: int
    w_holt: int

    def to_list(self) -> list:
        """Plain list for JSON files."""
        return [float(self.ridge_alpha), self.holt_key, self.drift_key] + [
            int(w) for w in self[3:]
        ]


# Untuned starting blends, used where no earlier fold exists.
DEFAULT_CFG = BlendConfig(1.0, "h1", "d5", 3, 2, 3, 2)
NO_PROPHET_CFG = BlendConfig(1.0, "h1", "d5", 4, 3, 0, 3)

# Every (naive, ridge, prophet, holt) weight split in tenths.
WEIGHT_GRID = np.array(
    [
        (w_naive, w_ridge, w_prophet, TENTHS - w_naive - w_ridge - w_prophet)
        for w_naive in range(TENTHS + 1)
        for w_ridge in range(TENTHS + 1 - w_naive)
        for w_prophet in range(TENTHS + 1 - w_naive - w_ridge)
    ]
)


@dataclass
class Fold:
    """One backtest fold: the test weeks and every base-model forecast."""

    test: pd.DataFrame
    components: dict
    origin: int
    _scores: dict = field(default_factory=dict, repr=False)

    def scores(self, prophet: bool) -> dict:
        """WMAPE of every candidate config on this fold (cached).

        All weight splits are scored at once with one tensor product.
        """
        if prophet in self._scores:
            return self._scores[prophet]
        actual = self.test.to_numpy()
        zeros = np.zeros_like(actual)
        scores = {}
        for alpha, holt_key, drift_key in product(
            ALPHAS, HOLT_GRID, DRIFT_GRID
        ):
            stacked = np.stack(
                [
                    self.components[f"drift_{drift_key}"].to_numpy(),
                    self.components[f"ridge_{alpha}"].to_numpy(),
                    self.components["prophet"].to_numpy()
                    if prophet
                    else zeros,
                    self.components[f"holt_{holt_key}"].to_numpy(),
                ]
            )
            blends = np.tensordot(WEIGHT_GRID / TENTHS, stacked, axes=(1, 0))
            fold_wmape = np.abs(blends - actual).sum(axis=(1, 2))
            fold_wmape = fold_wmape / actual.sum()
            for weights, value in zip(WEIGHT_GRID, fold_wmape, strict=True):
                config = BlendConfig(
                    alpha, holt_key, drift_key, *map(int, weights)
                )
                scores[config] = float(value)
        self._scores[prophet] = scores
        return scores


def make_fold(
    wide,
    types,
    origin,
    horizon,
    prophet=True,
    alphas=ALPHAS,
    n_jobs=1,
    clean=True,
) -> Fold:
    """Forecast `horizon` weeks after week number `origin` with all models.

    Only the weeks before `origin` are used, also for cleaning; the test
    actuals stay raw.
    """
    train, test = wide.iloc[:origin], wide.iloc[origin : origin + horizon]
    if train.index.max() >= test.index.min():
        raise ValueError("training weeks must come before the test weeks")
    if clean:
        train, _ = winsorize_anomalies(train)
    x_train, x_future = date_features(train.index), date_features(test.index)

    components = {"naive": naive_fc(train, test.index)}
    for alpha in alphas:
        components[f"ridge_{alpha}"] = ridge_fc(
            train, types, x_train, x_future, alpha
        )
    for key in HOLT_GRID:
        components[f"holt_{key}"] = holt_fc(
            train, types, x_train, x_future, key
        )
    for key, shrink in DRIFT_GRID.items():
        components[f"drift_{key}"] = naive_drift_fc(
            train, types, test.index, shrink
        )
    if prophet:
        components["prophet"] = prophet_fc(train, test.index, n_jobs=n_jobs)
    return Fold(test=test, components=components, origin=origin)


def blend_from_cfg(components: dict, cfg: BlendConfig) -> pd.DataFrame:
    """Blend of the base forecasts of one fold."""
    blend = (cfg.w_naive / TENTHS) * components[f"drift_{cfg.drift_key}"] + (
        cfg.w_ridge / TENTHS
    ) * components[f"ridge_{cfg.ridge_alpha}"]
    if cfg.w_holt > 0:
        blend = (
            blend + (cfg.w_holt / TENTHS) * components[f"holt_{cfg.holt_key}"]
        )
    if cfg.w_prophet > 0:
        blend = blend + (cfg.w_prophet / TENTHS) * components["prophet"]
    return blend


def candidate_cfgs(prophet=True, allow_trend=True, alphas=ALPHAS) -> list:
    """All configs that may be chosen.

    Without `prophet` the Prophet weight is 0. Without `allow_trend` there
    is no Holt part and no drift (used to measure what the trend fixes add).
    """
    holt_keys = list(HOLT_GRID) if allow_trend else ["h1"]
    drift_keys = list(DRIFT_GRID) if allow_trend else ["d0"]
    configs = []
    for alpha, holt_key, drift_key in product(alphas, holt_keys, drift_keys):
        for weights in WEIGHT_GRID:
            w_naive, w_ridge, w_prophet, w_holt = map(int, weights)
            if w_prophet > 0 and not prophet:
                continue
            if w_holt > 0 and not allow_trend:
                continue
            configs.append(
                BlendConfig(
                    alpha,
                    holt_key,
                    drift_key,
                    w_naive,
                    w_ridge,
                    w_prophet,
                    w_holt,
                )
            )
    return configs


def select_cfg(folds, prophet=True, allow_trend=True) -> BlendConfig:
    """The config with the lowest mean WMAPE over the given folds."""
    tables = [fold.scores(prophet) for fold in folds]
    return min(
        candidate_cfgs(prophet, allow_trend),
        key=lambda cfg: np.mean([table[cfg] for table in tables]),
    )


def walk_forward(folds, prophet=True, allow_trend=True):
    """Forecast every fold with a config chosen from the earlier folds only.

    The first fold has no earlier fold and uses a fixed untuned default.
    Returns (actuals, forecasts, configs used per fold).
    """
    default = DEFAULT_CFG if prophet else NO_PROPHET_CFG
    if not allow_trend:
        default = default._replace(
            drift_key="d0",
            w_naive=default.w_naive + default.w_holt,
            w_holt=0,
        )
    configs, forecasts = [], []
    for j, fold in enumerate(folds):
        cfg = (
            default if j == 0 else select_cfg(folds[:j], prophet, allow_trend)
        )
        configs.append(cfg)
        forecasts.append(blend_from_cfg(fold.components, cfg))
    actual = pd.concat([fold.test for fold in folds])
    return actual, pd.concat(forecasts), configs


def bias_factor(actual, forecast, types, strength) -> pd.Series:
    """Per-store factor 1 + strength * (actual / forecast - 1).

    The ratio of total actual to total forecast is taken per store type.
    """
    ratio = {}
    for kind in types.unique():
        columns = [c for c in actual.columns if types[c] == kind]
        ratio[kind] = float(
            actual[columns].to_numpy().sum()
            / forecast[columns].to_numpy().sum()
        )
    return pd.Series(
        {c: 1 + strength * (ratio[types[c]] - 1) for c in actual.columns}
    )


def apply_factor(forecast, factor) -> pd.DataFrame:
    return forecast * factor.reindex(forecast.columns).to_numpy()


def holiday_week_mask(dates) -> np.ndarray:
    """True for weeks around Thanksgiving and Christmas."""
    flags = date_features(dates)[HOLIDAY_FEATURES]
    return (flags.sum(axis=1) > 0).to_numpy()


# ----------------------------------------------------------------------
# Steps of the full evaluation
# ----------------------------------------------------------------------
def _bias_corrected_walk_forward(folds, forecast, types, strength):
    """Apply a bias factor learned from the earlier folds only."""
    parts = []
    for j, fold in enumerate(folds):
        part = forecast.loc[fold.test.index]
        if j > 0 and strength > 0:
            past = pd.concat([f.test for f in folds[:j]])
            factor = bias_factor(
                past, forecast.loc[past.index], types, strength
            )
            part = apply_factor(part, factor)
        parts.append(part)
    return pd.concat(parts)


def _bias_correction_study(folds, actual, forecast, types, n_boot):
    """Test whether a type-level bias correction improves the forecast.

    Rule fixed in advance: adopt the best strength only if the 95%
    interval of its WMAPE gain is above zero. The first fold is never
    corrected, so the test starts at the second fold.
    Returns (report, adopted strength).
    """
    later = actual.index >= folds[1].test.index[0]
    candidates = {}
    for strength in (0.5, 1.0):
        corrected = _bias_corrected_walk_forward(
            folds, forecast, types, strength
        )
        gain = block_bootstrap(
            actual[later],
            {"c": corrected[later], "u": forecast[later]},
            "WMAPE",
            n=n_boot,
            diff_against="u",
        )["c"]
        candidates[str(strength)] = {
            "WMAPE_gain_vs_uncorrected": gain,
            "metrics": metrics(actual[later], corrected[later]),
        }
    best = max(
        candidates, key=lambda k: candidates[k]["WMAPE_gain_vs_uncorrected"][0]
    )
    lower_bound = candidates[best]["WMAPE_gain_vs_uncorrected"][1]
    adopted = float(best) if lower_bound > 0 else 0.0
    report = {
        "candidates": candidates,
        "adopted_lam": adopted,
        "uncorrected_metrics_same_weeks": metrics(
            actual[later], forecast[later]
        ),
        "rule": "adopt best lambda only if the 95% CI of the WMAPE gain "
        "excludes 0",
        "note": "lambda picked on the same weeks it is scored on: mild "
        "optimism; first fold is never corrected",
    }
    return report, adopted


def _holiday_summary(actual, forecast, naive, n_boot):
    """Blend against naive in the holiday weeks (None if too few weeks)."""
    mask = holiday_week_mask(actual.index)
    if mask.sum() < 3:
        return None
    difference = block_bootstrap(
        actual[mask],
        {"naive": naive[mask], "blend": forecast[mask]},
        "WMAPE",
        block=1,
        n=n_boot,
        diff_against="naive",
    )["blend"]
    return {
        "n_weeks": int(mask.sum()),
        "blend_WMAPE": float(
            wmape(actual[mask].to_numpy(), forecast[mask].to_numpy())
        ),
        "naive_WMAPE": float(
            wmape(actual[mask].to_numpy(), naive[mask].to_numpy())
        ),
        "naive_minus_blend": difference,
        "note": "one holiday season only: too few weeks to tune "
        "holiday-specific weights",
    }


def _annual_planning_range(quarter_errors):
    """80% predictive range of actual / forecast - 1 for one new quarter.

    A Student-t interval from only five quarters, so it is indicative.
    """
    mean = float(np.mean(quarter_errors))
    spread = float(np.std(quarter_errors, ddof=1))
    n = len(quarter_errors)
    half_width = student_t.ppf(0.9, n - 1) * spread * np.sqrt(1 + 1 / n)
    return [mean - half_width, mean + half_width]


def _band_coverage_check(actual, forecast):
    """Calibrate the 80% band on the first 3 folds, test on the rest."""
    split = 3 * FOLD_WEEKS
    multipliers = interval_multipliers(
        actual.iloc[:split], forecast.iloc[:split]
    )
    return {
        "calibration_weeks": [
            str(actual.index[0].date()),
            str(actual.index[split - 1].date()),
        ],
        "test_weeks": [
            str(actual.index[split].date()),
            str(actual.index[-1].date()),
        ],
        "multipliers": multipliers,
        "coverage_80": coverage(
            actual.iloc[split:], forecast.iloc[split:], multipliers
        ),
    }


def _year_ahead_check(wide, types, final_cfg, factor, prophet, n_jobs, clean):
    """Forecast the last 58 weeks in one go (the real forecast horizon)."""
    fold = make_fold(
        wide,
        types,
        len(wide) - YEAR_AHEAD_WEEKS,
        YEAR_AHEAD_WEEKS,
        prophet,
        n_jobs=n_jobs,
        clean=clean,
    )
    untuned = DEFAULT_CFG if prophet else NO_PROPHET_CFG
    selected = blend_from_cfg(fold.components, final_cfg)
    corrected = apply_factor(selected, factor)
    actual = fold.test
    report = {
        "default_weights_untuned": metrics(
            actual, blend_from_cfg(fold.components, untuned)
        ),
        "selected_cfg": metrics(actual, selected),
        "selected_cfg_bias_corrected": metrics(actual, corrected),
        "seasonal_naive": metrics(actual, fold.components["naive"]),
        "note": "single 58-week sample: one year, not a distribution. "
        "selected_cfg weights and bias factors saw overlapping "
        "weeks.",
    }
    return fold, corrected, report


def run_evaluation(
    panel, prophet=True, n_boot=1000, n_jobs=1, clean=True
) -> dict:
    """Full backtest. Returns a JSON-friendly dict plus `_arrays` (tables).

    `_arrays` holds the forecast and actual tables used for plots; remove
    it with `result.pop("_arrays")` before saving the dict as JSON.
    """
    wide = to_wide(panel)
    types = store_types(panel)
    n_weeks = len(wide)
    origins = range(n_weeks - N_FOLDS * FOLD_WEEKS, n_weeks, FOLD_WEEKS)
    folds = [
        make_fold(
            wide,
            types,
            origin,
            FOLD_WEEKS,
            prophet,
            n_jobs=n_jobs,
            clean=clean,
        )
        for origin in origins
    ]

    actual, blend_raw, cfgs = walk_forward(folds, prophet)
    _, blend_no_trend, _ = walk_forward(folds, prophet, allow_trend=False)
    bias_study, strength = _bias_correction_study(
        folds, actual, blend_raw, types, n_boot
    )
    blend = (
        _bias_corrected_walk_forward(folds, blend_raw, types, strength)
        if strength > 0
        else blend_raw
    )

    def stack(name):
        return pd.concat([fold.components[name] for fold in folds])

    benchmarks = {
        "seasonal_naive": stack("naive"),
        "naive_with_drift_d5": stack("drift_d5"),
        "ridge_alpha1": stack("ridge_1.0"),
        "holt_alone": stack("holt_h1"),
    }
    if prophet:
        benchmarks["prophet"] = stack("prophet")
    all_forecasts = {
        "blend_walk_forward": blend,
        "blend_no_bias_correction": blend_raw,
        "blend_without_trend_fixes": blend_no_trend,
        **benchmarks,
    }

    result = {
        "oof_weeks": [
            str(actual.index[0].date()),
            str(actual.index[-1].date()),
        ],
        "oof_cfgs": [cfg.to_list() for cfg in cfgs],
        "bias_correction": bias_study,
        "oof_metrics": {
            name: metrics(actual, forecast)
            for name, forecast in all_forecasts.items()
        },
        "oof_ci": {
            stat: block_bootstrap(actual, all_forecasts, stat, n=n_boot)
            for stat in (
                "MAE",
                "RMSE",
                "WMAPE",
                "Bias",
                "Weekly_total_MAPE",
            )
        },
    }
    skill = block_bootstrap(
        actual,
        {"naive": benchmarks["seasonal_naive"], "blend": blend},
        "WMAPE",
        n=n_boot,
        diff_against="naive",
    )
    result["oof_skill_vs_naive_WMAPE_diff"] = {
        "naive_minus_blend": skill["blend"]
    }
    holiday = _holiday_summary(
        actual, blend, benchmarks["seasonal_naive"], n_boot
    )
    if holiday:
        result["holiday_weeks"] = holiday

    quarter_errors = [
        float(
            fold.test.to_numpy().sum()
            / blend.loc[fold.test.index].to_numpy().sum()
            - 1
        )
        for fold in folds
    ]
    result["quarter_total_error"] = quarter_errors
    result["annual_planning_range"] = _annual_planning_range(quarter_errors)
    result["coverage_check"] = _band_coverage_check(actual, blend)

    final_cfg = select_cfg(folds, prophet)
    result["final_cfg"] = final_cfg.to_list()
    factor = (
        bias_factor(actual, blend_raw, types, strength)
        if strength > 0
        else pd.Series(1.0, index=wide.columns)
    )
    result["store_scale"] = {int(c): float(v) for c, v in factor.items()}

    year_fold, year_blend, result["year_ahead"] = _year_ahead_check(
        wide, types, final_cfg, factor, prophet, n_jobs, clean
    )
    weekly_total_error = (
        year_fold.test.sum(axis=1) / year_blend.sum(axis=1) - 1
    )
    result["calibration"] = {
        "store_week_mult": interval_multipliers(year_fold.test, year_blend),
        "weekly_total_err_q": [
            float(np.quantile(weekly_total_error, 0.1)),
            float(np.quantile(weekly_total_error, 0.9)),
        ],
        "annual_planning_range": result["annual_planning_range"],
    }
    _, anomalies = winsorize_anomalies(wide)
    result["anomalies_winsorised_full_history_descriptive"] = (
        anomalies_to_records(anomalies)
    )
    result["_arrays"] = {
        "oof_actual": actual,
        "oof_blend": blend,
        "oof_benchmarks": benchmarks,
        "ya_actual": year_fold.test,
        "ya_blend": year_blend,
    }
    return result
