"""Walk-forward evaluation, nested blend selection, block-bootstrap uncertainty, interval calibration."""
import numpy as np
import pandas as pd
from .data import to_wide
from .features import date_features
from .models import naive_fc, ridge_fc, prophet_fc

ALPHAS = (0.1, 1.0, 10.0)
DEFAULT_CFG = (1.0, 4, 3, 3)          # (ridge alpha, w_naive*10, w_ridge*10, w_prophet*10) ~ equal weights, untuned


# ------------------------------------------------------------------ metrics
def metrics(a: pd.DataFrame, f: pd.DataFrame) -> dict:
    e = f - a
    return dict(WMAPE=float(np.abs(e).sum().sum() / a.sum().sum()),
                MAPE=float((np.abs(e) / a).mean().mean()),
                Bias=float(e.sum().sum() / a.sum().sum()),                       # (forecast-actual)/actual; <0 = forecast too low
                Store_annual_APE=float((np.abs(e.sum()) / a.sum()).mean()),
                Weekly_total_MAPE=float((np.abs(e.sum(axis=1)) / a.sum(axis=1)).mean()))


def _wmape(A, F): return np.abs(F - A).sum() / A.sum()
def _bias(A, F): return (F - A).sum() / A.sum()
def _total_mape(A, F): return (np.abs(F.sum(1) - A.sum(1)) / A.sum(1)).mean()
STATS = {"WMAPE": _wmape, "Bias": _bias, "Weekly_total_MAPE": _total_mape}


def block_bootstrap(a, forecasts: dict, stat="WMAPE", block=6, n=1000, seed=0, diff_against=None):
    """Moving-block bootstrap over WEEKS (weeks are autocorrelated, stores within a week are kept together).
    Returns {name: (point, lo95, hi95)}. With diff_against=name_ref, returns the paired difference stat(ref)-stat(name)."""
    A = a.to_numpy(); T = len(A); fn = STATS[stat]
    rng = np.random.default_rng(seed); Fs = {k: v.to_numpy() for k, v in forecasts.items()}
    nb = int(np.ceil(T / block)); draws = {k: [] for k in Fs}
    for _ in range(n):
        starts = rng.integers(0, T - block + 1, size=nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:T]
        for k, F in Fs.items():
            if diff_against is None:
                draws[k].append(fn(A[idx], F[idx]))
            else:
                draws[k].append(fn(A[idx], Fs[diff_against][idx]) - fn(A[idx], F[idx]))
    out = {}
    for k, F in Fs.items():
        pt = fn(A, F) if diff_against is None else fn(A, Fs[diff_against]) - fn(A, F)
        out[k] = (float(pt), float(np.quantile(draws[k], .025)), float(np.quantile(draws[k], .975)))
    return out


# ------------------------------------------------------------------ folds and blending
def make_fold(wide, types, origin, h, prophet=True, alphas=ALPHAS, n_jobs=1):
    tr, te = wide.iloc[:origin], wide.iloc[origin:origin + h]
    assert tr.index.max() < te.index.min()
    Xtr, Xf = date_features(tr.index), date_features(te.index)
    comp = {"naive": naive_fc(tr, te.index)}
    for a in alphas:
        comp[f"ridge_{a}"] = ridge_fc(tr, types, Xtr, Xf, a)
    if prophet:
        comp["prophet"] = prophet_fc(tr, te.index, n_jobs=n_jobs)
    return {"test": te, "comp": comp, "origin": origin}


def blend_from_cfg(comp, cfg):
    a, wn, wr, wp = cfg
    out = (wn / 10) * comp["naive"] + (wr / 10) * comp[f"ridge_{a}"]
    return out + (wp / 10) * comp["prophet"] if wp > 0 else out


def candidate_cfgs(prophet=True, alphas=ALPHAS):
    return [(a, wn, wr, 10 - wn - wr) for a in alphas for wn in range(11) for wr in range(11 - wn)
            if prophet or (10 - wn - wr) == 0]


def select_cfg(folds, prophet=True):
    """Pick (alpha, weights) minimising mean WMAPE over the given folds."""
    best = min(candidate_cfgs(prophet), key=lambda c: np.mean([metrics(f["test"], blend_from_cfg(f["comp"], c))["WMAPE"] for f in folds]))
    return best


def walk_forward(folds, prophet=True):
    """Fold j is forecast with a config chosen ONLY from folds < j (fold 0 uses DEFAULT_CFG).
    Evaluates the whole procedure (selection included), so the result is not optimistic."""
    cfgs, fc = [], []
    for j, f in enumerate(folds):
        cfg = DEFAULT_CFG if j == 0 else select_cfg(folds[:j], prophet)
        if not prophet and cfg[3] > 0: cfg = (cfg[0], 5, 5, 0)
        cfgs.append(cfg); fc.append(blend_from_cfg(f["comp"], cfg))
    actual = pd.concat([f["test"] for f in folds]); return actual, pd.concat(fc), cfgs


def interval_multipliers(actual, forecast, lo=0.1, hi=0.9):
    e = np.log(actual.to_numpy() / forecast.to_numpy()).ravel()
    return float(np.exp(np.quantile(e, lo))), float(np.exp(np.quantile(e, hi)))


def coverage(actual, forecast, mult):
    return float(((actual >= forecast * mult[0]) & (actual <= forecast * mult[1])).mean().mean())


# ------------------------------------------------------------------ full evaluation
def run_evaluation(panel, prophet=True, n_boot=1000, n_jobs=1) -> dict:
    wide = to_wide(panel); types = panel.drop_duplicates("Store").set_index("Store")["Type"].reindex(wide.columns)
    n = len(wide); h = 13
    origins = list(range(n - 5 * h, n, h))                     # 5 contiguous 13-week folds ending at the last week
    folds = [make_fold(wide, types, o, h, prophet, n_jobs=n_jobs) for o in origins]
    actual, wf, cfgs = walk_forward(folds, prophet)
    benchmarks = {"seasonal_naive": pd.concat([f["comp"]["naive"] for f in folds]),
                  "ridge_alpha1": pd.concat([f["comp"]["ridge_1.0"] for f in folds])}
    if prophet: benchmarks["prophet"] = pd.concat([f["comp"]["prophet"] for f in folds])
    allf = {"blend_walk_forward": wf, **benchmarks}
    res = {"oof_weeks": [str(actual.index[0].date()), str(actual.index[-1].date())], "oof_cfgs": [list(map(float, c)) for c in cfgs],
           "oof_metrics": {k: metrics(actual, v) for k, v in allf.items()}}
    # uncertainty on the headline numbers (block bootstrap over weeks)
    res["oof_ci"] = {s: block_bootstrap(actual, allf, s, n=n_boot) for s in ("WMAPE", "Bias", "Weekly_total_MAPE")}
    d = block_bootstrap(actual, {"naive": benchmarks["seasonal_naive"], "blend": wf}, "WMAPE", n=n_boot, diff_against="naive")
    res["oof_skill_vs_naive_WMAPE_diff"] = {"naive_minus_blend": d["blend"]}   # >0 and CI above 0 => blend really better
    # quarter-level totals (what guidance is about): actual/forecast-1 per 13-week fold
    q = [float(f["test"].to_numpy().sum() / wf.loc[f["test"].index].to_numpy().sum() - 1) for f in folds]
    res["quarter_total_error"] = q
    m, s = float(np.mean(q)), float(np.std(q, ddof=1))
    from scipy.stats import t as _t                                  # 80% predictive interval for a new period's error, n small => t, not normal
    half = _t.ppf(0.9, len(q) - 1) * s * np.sqrt(1 + 1 / len(q))
    res["annual_planning_range"] = [m - half, m + half]               # actual/forecast-1; quarter errors share one sign, so treated as fully correlated
    # coverage of 80% store-week bands: calibrate on the first 3 folds, test on the last 2 (strictly later weeks)
    k = 3 * h; mult = interval_multipliers(actual.iloc[:k], wf.iloc[:k])
    res["coverage_check"] = {"calibration_weeks": [str(actual.index[0].date()), str(actual.index[k - 1].date())],
                             "test_weeks": [str(actual.index[k].date()), str(actual.index[-1].date())],
                             "multipliers": mult, "coverage_80": coverage(actual.iloc[k:], wf.iloc[k:], mult)}
    # production configuration: selected on all folds
    final_cfg = select_cfg(folds, prophet) if prophet else (select_cfg(folds, False)[0], 5, 5, 0)
    res["final_cfg"] = [float(final_cfg[0])] + [int(x) for x in final_cfg[1:]]
    # 1-year-ahead check: train to week n-58, forecast 58 weeks (same horizon as the real forecast)
    ya = make_fold(wide, types, n - 58, 58, prophet, n_jobs=n_jobs)
    res["year_ahead"] = {}
    for name, cfg in {"equal_weights_untuned": DEFAULT_CFG if prophet else (1.0, 5, 5, 0), "selected_cfg": final_cfg}.items():
        f = blend_from_cfg(ya["comp"], cfg); res["year_ahead"][name] = metrics(ya["test"], f)
    res["year_ahead"]["seasonal_naive"] = metrics(ya["test"], ya["comp"]["naive"])
    res["year_ahead"]["note"] = "single 58-week sample: one year, not a distribution. 'selected_cfg' weights saw overlapping weeks."
    f = blend_from_cfg(ya["comp"], final_cfg)
    tot = ya["test"].sum(axis=1) / f.sum(axis=1) - 1
    res["calibration"] = {"store_week_mult": interval_multipliers(ya["test"], f),
                          "weekly_total_err_q": [float(np.quantile(tot, .1)), float(np.quantile(tot, .9))],
                          "annual_planning_range": res["annual_planning_range"]}
    res["_arrays"] = dict(oof_actual=actual, oof_blend=wf, oof_benchmarks=benchmarks, ya_actual=ya["test"], ya_blend=f)
    return res
