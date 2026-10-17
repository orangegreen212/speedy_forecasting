"""Walk-forward evaluation, nested blend selection, block-bootstrap uncertainty, interval calibration."""
import numpy as np
import pandas as pd
from .data import to_wide
from .features import date_features
from .models import naive_fc, naive_drift_fc, ridge_fc, prophet_fc, holt_fc, HOLT_GRID, DRIFT_GRID
from .cleaning import winsorize_anomalies

ALPHAS = (0.1, 1.0, 10.0)
DEFAULT_CFG = (1.0, "h1", "d5", 3, 2, 3, 2)   # (ridge alpha, holt key, drift key, w_naive, w_ridge, w_prophet, w_holt in tenths): untuned
NO_PROPHET_CFG = (1.0, "h1", "d5", 4, 3, 0, 3)


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
def make_fold(wide, types, origin, h, prophet=True, alphas=ALPHAS, n_jobs=1, clean=True):
    tr, te = wide.iloc[:origin], wide.iloc[origin:origin + h]
    assert tr.index.max() < te.index.min()
    if clean:                                   # cleaning sees the TRAINING slice only; test actuals stay raw
        tr, _ = winsorize_anomalies(tr)
    Xtr, Xf = date_features(tr.index), date_features(te.index)
    comp = {"naive": naive_fc(tr, te.index)}
    for a in alphas:
        comp[f"ridge_{a}"] = ridge_fc(tr, types, Xtr, Xf, a)
    for k in HOLT_GRID:
        comp[f"holt_{k}"] = holt_fc(tr, types, Xtr, Xf, k)
    for k in DRIFT_GRID:
        comp[f"drift_{k}"] = naive_drift_fc(tr, types, te.index, DRIFT_GRID[k])
    if prophet:
        comp["prophet"] = prophet_fc(tr, te.index, n_jobs=n_jobs)
    return {"test": te, "comp": comp, "origin": origin}


def blend_from_cfg(comp, cfg):
    a, hk, dk, wn, wr, wp, wh = cfg
    out = (wn / 10) * comp[f"drift_{dk}"] + (wr / 10) * comp[f"ridge_{a}"]
    if wh > 0: out = out + (wh / 10) * comp[f"holt_{hk}"]
    return out + (wp / 10) * comp["prophet"] if wp > 0 else out


def candidate_cfgs(prophet=True, allow_trend=True, alphas=ALPHAS):
    out = []
    for a in alphas:
        for hk in (HOLT_GRID if allow_trend else ["h1"]):
            for dk in (DRIFT_GRID if allow_trend else ["d0"]):
                for wn in range(11):
                    for wr in range(11 - wn):
                        for wp in range(11 - wn - wr):
                            wh = 10 - wn - wr - wp
                            if (not prophet and wp > 0) or (not allow_trend and wh > 0): continue
                            out.append((a, hk, dk, wn, wr, wp, wh))
    return out


_WEIGHTS = np.array([(wn, wr, wp, 10 - wn - wr - wp) for wn in range(11) for wr in range(11 - wn) for wp in range(11 - wn - wr)])


def _fold_scores(f, prophet):
    """WMAPE of EVERY candidate config on this fold, computed once and cached (vectorised over the weight grid)."""
    if f.get("scores_for") == prophet:
        return f["scores"]
    te = f["test"].to_numpy(); den = te.sum(); scores = {}
    zero = np.zeros_like(te)
    for a in ALPHAS:
        for hk in HOLT_GRID:
            for dk in DRIFT_GRID:
                comps = np.stack([f["comp"][f"drift_{dk}"].to_numpy(), f["comp"][f"ridge_{a}"].to_numpy(),
                                  f["comp"]["prophet"].to_numpy() if prophet else zero, f["comp"][f"holt_{hk}"].to_numpy()])
                wm = np.abs(np.tensordot(_WEIGHTS / 10, comps, axes=(1, 0)) - te).sum(axis=(1, 2)) / den
                for w, v in zip(_WEIGHTS, wm):
                    scores[(a, hk, dk, int(w[0]), int(w[1]), int(w[2]), int(w[3]))] = float(v)
    f["scores"], f["scores_for"] = scores, prophet
    return scores


def select_cfg(folds, prophet=True, allow_trend=True):
    """Pick (alpha, holt key, weights) minimising mean WMAPE over the given folds."""
    tables = [_fold_scores(f, prophet) for f in folds]
    return min(candidate_cfgs(prophet, allow_trend), key=lambda c: np.mean([t[c] for t in tables]))


def walk_forward(folds, prophet=True, allow_trend=True):
    """Fold j is forecast with a config chosen ONLY from folds < j (fold 0 uses a fixed untuned default).
    Evaluates the whole procedure (selection included), so the result is not optimistic."""
    cfgs, fc = [], []
    default = DEFAULT_CFG if prophet else NO_PROPHET_CFG
    if not allow_trend: default = (default[0], default[1], "d0", default[3] + default[6], default[4], default[5], 0)
    for j, f in enumerate(folds):
        cfg = default if j == 0 else select_cfg(folds[:j], prophet, allow_trend)
        cfgs.append(cfg); fc.append(blend_from_cfg(f["comp"], cfg))
    actual = pd.concat([f["test"] for f in folds]); return actual, pd.concat(fc), cfgs


def interval_multipliers(actual, forecast, lo=0.1, hi=0.9):
    e = np.log(actual.to_numpy() / forecast.to_numpy()).ravel()
    return float(np.exp(np.quantile(e, lo))), float(np.exp(np.quantile(e, hi)))


def coverage(actual, forecast, mult):
    return float(((actual >= forecast * mult[0]) & (actual <= forecast * mult[1])).mean().mean())


# ------------------------------------------------------------------ bias correction and holiday weeks
def bias_factor(actual, forecast, types, lam):
    """Per-store multiplicative factor = 1 + lam * (actual/forecast - 1), ratio of sums per store TYPE, from the data given."""
    ratio = {}
    for t in types.unique():
        cols = [c for c in actual.columns if types[c] == t]
        ratio[t] = float(actual[cols].to_numpy().sum() / forecast[cols].to_numpy().sum())
    return pd.Series({c: 1 + lam * (ratio[types[c]] - 1) for c in actual.columns})


def apply_factor(fc, fac):
    return fc * fac.reindex(fc.columns).to_numpy()


def holiday_week_mask(dates):
    f = date_features(dates)
    return (f[["th-1", "th0", "th1", "xm-2", "xm-1", "xm0", "xm1", "xm2", "xm3"]].sum(axis=1) > 0).to_numpy()


# ------------------------------------------------------------------ full evaluation
def run_evaluation(panel, prophet=True, n_boot=1000, n_jobs=1, clean=True) -> dict:
    wide = to_wide(panel); types = panel.drop_duplicates("Store").set_index("Store")["Type"].reindex(wide.columns)
    n = len(wide); h = 13
    origins = list(range(n - 5 * h, n, h))                     # 5 contiguous 13-week folds ending at the last week
    folds = [make_fold(wide, types, o, h, prophet, n_jobs=n_jobs, clean=clean) for o in origins]
    actual, wf0, cfgs = walk_forward(folds, prophet)           # blend WITHOUT bias correction
    _, wf_noholt, _ = walk_forward(folds, prophet, allow_trend=False)   # ablation: no store-trend fixes (no holt, no drift)

    # ---- bias correction, evaluated walk-forward: factor for fold j uses only folds < j
    def corrected(lam):
        out = []
        for j, f in enumerate(folds):
            fj = wf0.loc[f["test"].index]
            if j == 0 or lam == 0:
                out.append(fj); continue
            prev = pd.concat([g["test"] for g in folds[:j]])
            out.append(apply_factor(fj, bias_factor(prev, wf0.loc[prev.index], types, lam)))
        return pd.concat(out)
    sub = (actual.index >= folds[1]["test"].index[0])
    bc = {"candidates": {}}
    for lam in (0.5, 1.0):
        wc = corrected(lam)
        d = block_bootstrap(actual[sub], {"c": wc[sub], "u": wf0[sub]}, "WMAPE", n=n_boot, diff_against="u")["c"]  # WMAPE(uncorrected)-WMAPE(corrected)
        bc["candidates"][str(lam)] = {"WMAPE_gain_vs_uncorrected": d, "metrics": metrics(actual[sub], wc[sub])}
    best = max(bc["candidates"], key=lambda k: bc["candidates"][k]["WMAPE_gain_vs_uncorrected"][0])
    lam_ad = float(best) if bc["candidates"][best]["WMAPE_gain_vs_uncorrected"][1] > 0 else 0.0   # rule fixed in advance: adopt only if CI excludes 0
    bc.update(adopted_lam=lam_ad, uncorrected_metrics_same_weeks=metrics(actual[sub], wf0[sub]),
              rule="adopt best lambda only if the 95% CI of the WMAPE gain excludes 0",
              note="lambda picked on the same weeks it is scored on: mild optimism; first fold is never corrected")
    wf = corrected(lam_ad) if lam_ad > 0 else wf0

    benchmarks = {"seasonal_naive": pd.concat([f["comp"]["naive"] for f in folds]),
                  "naive_with_drift_d5": pd.concat([f["comp"]["drift_d5"] for f in folds]),
                  "ridge_alpha1": pd.concat([f["comp"]["ridge_1.0"] for f in folds])}
    benchmarks["holt_alone"] = pd.concat([f["comp"]["holt_h1"] for f in folds])
    if prophet: benchmarks["prophet"] = pd.concat([f["comp"]["prophet"] for f in folds])
    allf = {"blend_walk_forward": wf, "blend_no_bias_correction": wf0, "blend_without_trend_fixes": wf_noholt, **benchmarks}
    res = {"oof_weeks": [str(actual.index[0].date()), str(actual.index[-1].date())], "oof_cfgs": [[float(c[0]), c[1], c[2]] + [int(x) for x in c[3:]] for c in cfgs],
           "bias_correction": bc, "oof_metrics": {k: metrics(actual, v) for k, v in allf.items()}}
    res["oof_ci"] = {s: block_bootstrap(actual, allf, s, n=n_boot) for s in ("WMAPE", "Bias", "Weekly_total_MAPE")}
    d = block_bootstrap(actual, {"naive": benchmarks["seasonal_naive"], "blend": wf}, "WMAPE", n=n_boot, diff_against="naive")
    res["oof_skill_vs_naive_WMAPE_diff"] = {"naive_minus_blend": d["blend"]}
    # holiday weeks: where most money is, and where the blend was weaker than naive
    mk = holiday_week_mask(actual.index)
    if mk.sum() >= 3:
        hd = block_bootstrap(actual[mk], {"naive": benchmarks["seasonal_naive"][mk], "blend": wf[mk]}, "WMAPE", block=1, n=n_boot, diff_against="naive")["blend"]
        res["holiday_weeks"] = {"n_weeks": int(mk.sum()), "blend_WMAPE": float(_wmape(actual[mk].to_numpy(), wf[mk].to_numpy())),
                                "naive_WMAPE": float(_wmape(actual[mk].to_numpy(), benchmarks["seasonal_naive"][mk].to_numpy())),
                                "naive_minus_blend": hd, "note": "one holiday season only: too few weeks to tune holiday-specific weights"}
    q = [float(f["test"].to_numpy().sum() / wf.loc[f["test"].index].to_numpy().sum() - 1) for f in folds]
    res["quarter_total_error"] = q
    m, s_ = float(np.mean(q)), float(np.std(q, ddof=1))
    from scipy.stats import t as _t
    half = _t.ppf(0.9, len(q) - 1) * s_ * np.sqrt(1 + 1 / len(q))
    res["annual_planning_range"] = [m - half, m + half]
    k = 3 * h; mult = interval_multipliers(actual.iloc[:k], wf.iloc[:k])
    res["coverage_check"] = {"calibration_weeks": [str(actual.index[0].date()), str(actual.index[k - 1].date())],
                             "test_weeks": [str(actual.index[k].date()), str(actual.index[-1].date())],
                             "multipliers": mult, "coverage_80": coverage(actual.iloc[k:], wf.iloc[k:], mult)}
    final_cfg = select_cfg(folds, prophet)
    res["final_cfg"] = [float(final_cfg[0]), final_cfg[1], final_cfg[2]] + [int(x) for x in final_cfg[3:]]   # alpha, holt key, drift key, w_naive, w_ridge, w_prophet, w_holt (x10)
    fac = bias_factor(actual, wf0, types, lam_ad) if lam_ad > 0 else pd.Series(1.0, index=wide.columns)
    res["store_scale"] = {int(c): float(v) for c, v in fac.items()}
    ya = make_fold(wide, types, n - 58, 58, prophet, n_jobs=n_jobs, clean=clean)
    res["year_ahead"] = {}
    for name, cfg in {"equal_weights_untuned": DEFAULT_CFG if prophet else NO_PROPHET_CFG, "selected_cfg": final_cfg}.items():
        res["year_ahead"][name] = metrics(ya["test"], blend_from_cfg(ya["comp"], cfg))
    f = apply_factor(blend_from_cfg(ya["comp"], final_cfg), fac)
    res["year_ahead"]["selected_cfg_bias_corrected"] = metrics(ya["test"], f)
    res["year_ahead"]["seasonal_naive"] = metrics(ya["test"], ya["comp"]["naive"])
    res["year_ahead"]["note"] = "single 58-week sample: one year, not a distribution. selected_cfg weights and bias factors saw overlapping weeks."
    tot = ya["test"].sum(axis=1) / f.sum(axis=1) - 1
    res["calibration"] = {"store_week_mult": interval_multipliers(ya["test"], f),
                          "weekly_total_err_q": [float(np.quantile(tot, .1)), float(np.quantile(tot, .9))],
                          "annual_planning_range": res["annual_planning_range"]}
    _, an = winsorize_anomalies(wide)
    res["anomalies_winsorised_full_history_descriptive"] = an.assign(Date=an.Date.astype(str)).round(3).to_dict("records") if len(an) else []
    res["_arrays"] = dict(oof_actual=actual, oof_blend=wf, oof_benchmarks=benchmarks, ya_actual=ya["test"], ya_blend=f)
    return res
