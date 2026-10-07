import numpy as np, pandas as pd
from speedy.evaluate import metrics, block_bootstrap, candidate_cfgs, interval_multipliers, coverage


def frames():
    a = pd.DataFrame({"s1": [100., 200., 100., 200.], "s2": [100., 100., 100., 100.]})
    return a, a * 1.1


def test_metrics_known_values():
    a, f = frames(); m = metrics(a, f)
    assert np.isclose(m["WMAPE"], 0.1) and np.isclose(m["Bias"], 0.1) and np.isclose(m["MAPE"], 0.1)


def test_bootstrap_deterministic_and_contains_point():
    rng = np.random.default_rng(1); a = pd.DataFrame(rng.uniform(90, 110, (40, 3))); f = a * rng.uniform(.9, 1.1, a.shape)
    r1 = block_bootstrap(a, {"m": f}, n=200, seed=3); r2 = block_bootstrap(a, {"m": f}, n=200, seed=3)
    assert r1 == r2; pt, lo, hi = r1["m"]; assert lo <= pt <= hi


def test_paired_difference_zero_for_identical_models():
    a, f = frames(); a = pd.concat([a] * 5, ignore_index=True); f = a * 1.1
    r = block_bootstrap(a, {"x": f}, n=50, diff_against="x")
    assert r["x"] == (0.0, 0.0, 0.0)


def test_candidate_weights_sum_to_ten_and_prophet_off():
    assert all(c[1] + c[2] + c[3] == 10 for c in candidate_cfgs())
    assert all(c[3] == 0 for c in candidate_cfgs(prophet=False))


def test_interval_multipliers_order_and_coverage():
    rng = np.random.default_rng(0); f = pd.DataFrame(rng.uniform(90, 110, (200, 5))); a = f * rng.lognormal(0, .05, f.shape)
    lo, hi = interval_multipliers(a, f); assert lo < 1 < hi
    assert abs(coverage(a, f, (lo, hi)) - 0.8) < 0.03
