import numpy as np
import pandas as pd
from math import erf, sqrt

import yfinance as yf

TRADING_DAYS = 252

# ===============================
# CONFIG (MATCH LRP_TS LOGIC)
# ===============================

PRICE_CSV   = "sp500_adjclose_2005_2023.csv"
EVENTS_CSV  = "legal_events_scored.csv"

LOOKBACK       = 252        # 12-month momentum
TOP_N          = 10         # number of names in the portfolio

BAN_LOOKBACK   = 30         # days a stock is banned from LONGS after bad legal hit
BAN_SEV_THRESH = 3          # severity threshold for bans
SURPRISE_WIN   = 252        # rolling window for expected severity

GAMMA_LEGAL    = 2.0        # weight on legal factor in combined signal

EXPO_HIGH  = 1.0            # no leverage
EXPO_NEUT  = 1.0
EXPO_LOW   = 0.7            # de-risk when many extreme events
EXTREME_SEV_THRESH = 4      # severity threshold for "extreme" events
EXTREME_COUNT      = 3      # if >= this many extreme hits in a day => de-risk

BENCH = "SPY"


# ===============================
# Basic helpers
# ===============================

def annualized_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    cum = (1 + returns).prod()
    years = len(returns) / TRADING_DAYS
    return cum ** (1 / years) - 1 if years > 0 else np.nan


def annualized_vol(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    return returns.std(ddof=1) * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    excess = returns - rf / TRADING_DAYS
    mu = excess.mean() * TRADING_DAYS
    sig = excess.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return np.nan if sig == 0 else mu / sig


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def t_test_excess_returns(strategy: pd.Series, benchmark: pd.Series):
    """
    One-sample t-test on daily excess returns (strategy - benchmark).
    H0: mean(excess) <= 0
    H1: mean(excess) > 0
    """
    strategy = strategy.dropna()
    benchmark = benchmark.dropna()
    common = strategy.index.intersection(benchmark.index)
    excess = (strategy.loc[common] - benchmark.loc[common]).dropna()

    n = len(excess)
    if n < 2:
        return np.nan, 1.0

    mean_ex = excess.mean()
    std_ex = excess.std(ddof=1)
    se = std_ex / np.sqrt(n) if std_ex > 0 else np.nan

    if not np.isfinite(se) or se == 0:
        return np.nan, 1.0

    t_stat = mean_ex / se
    p_one_sided = 1 - norm_cdf(t_stat)
    return t_stat, p_one_sided


def sharpe_diff_test(strategy: pd.Series, benchmark: pd.Series):
    """
    Test whether strategy Sharpe > benchmark Sharpe using
    t-test on excess returns as proxy.
    """
    strat_sh = sharpe_ratio(strategy)
    bench_sh = sharpe_ratio(benchmark)
    t_stat, p_val = t_test_excess_returns(strategy, benchmark)
    return strat_sh, bench_sh, t_stat, p_val


def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_boot: int = 2000,
    alpha: float = 0.05,
    block_size: int = 5,
):
    """
    Block bootstrap Sharpe ratio to get confidence interval.
    """
    returns = returns.dropna()
    r = returns.values
    n = len(r)
    if n == 0:
        return np.nan, np.nan, np.array([])

    n_blocks = int(np.ceil(n / block_size))
    sh_samples = []

    for _ in range(n_boot):
        idx_blocks = np.random.randint(0, n_blocks, size=n_blocks)
        boot = []
        for b in idx_blocks:
            start = b * block_size
            end = min(start + block_size, n)
            boot.extend(r[start:end])
        boot_series = pd.Series(boot)
        sh_samples.append(sharpe_ratio(boot_series))

    sh_samples = np.array(sh_samples)
    lower = np.percentile(sh_samples, 100 * alpha / 2)
    upper = np.percentile(sh_samples, 100 * (1 - alpha / 2))
    return lower, upper, sh_samples


def deflated_sharpe_ratio(
    returns: pd.Series,
    sr_threshold: float = 0.0,
    n_strats: int = 1,
):
    """
    Approximate Probabilistic Sharpe Ratio (PSR) with a simple
    multiple-testing correction via raising the Sharpe threshold.
    """
    r = returns.dropna()
    T = len(r)
    if T < 2:
        return np.nan

    sr = sharpe_ratio(r)
    skew = r.skew()
    ex_kurt = r.kurt()  # excess kurtosis

    if n_strats > 1:
        penalty = 0.5 * np.log(n_strats) / np.sqrt(max(T - 1, 1))
    else:
        penalty = 0.0

    sr_star = sr_threshold + penalty

    denom = 1 - skew * sr + ((ex_kurt - 1) / 4.0) * (sr ** 2)
    if denom <= 0:
        return np.nan

    z = (sr - sr_star) * np.sqrt(T - 1) / np.sqrt(denom)
    psr = norm_cdf(z)
    return psr


def yearly_stability_report(strategy: pd.Series, benchmark: pd.Series):
    """
    Compute CAGR, Sharpe, and t-test per calendar year.
    """
    strategy = strategy.dropna()
    benchmark = benchmark.dropna()
    common = strategy.index.intersection(benchmark.index)
    strat = strategy.loc[common]
    bench = benchmark.loc[common]

    years = sorted(set(d.year for d in strat.index))

    results = []
    for y in years:
        mask = strat.index.year == y
        r_s = strat[mask]
        r_b = bench[mask]
        if len(r_s) < 50:
            continue

        cagr_s = annualized_return(r_s)
        cagr_b = annualized_return(r_b)
        sh_s = sharpe_ratio(r_s)
        sh_b = sharpe_ratio(r_b)
        t_stat, p_val = t_test_excess_returns(r_s, r_b)

        results.append({
            "year": y,
            "cagr_strategy": cagr_s,
            "cagr_benchmark": cagr_b,
            "sharpe_strategy": sh_s,
            "sharpe_benchmark": sh_b,
            "t_stat_excess": t_stat,
            "p_val_excess": p_val,
        })

    return results


# ===============================
# Data loading + signal building
# ===============================

def load_price_data():
    prices = pd.read_csv(PRICE_CSV, parse_dates=[0])
    prices = prices.sort_values(prices.columns[0]).set_index(prices.columns[0])
    prices = prices.astype(float)
    idx = prices.index

    # Ensure SPY exists
    if "SPY" not in prices.columns:
        spy_df = yf.download(
            BENCH,
            start=idx[0],
            end=idx[-1] + pd.Timedelta(days=1),
            auto_adjust=False,
            progress=False,
        )
        if isinstance(spy_df.columns, pd.MultiIndex):
            lvl0 = spy_df.columns.get_level_values(0)
            if "Adj Close" in lvl0:
                spy = spy_df.xs("Adj Close", level=0, axis=1)
            elif "Close" in lvl0:
                spy = spy_df.xs("Close", level=0, axis=1)
            else:
                spy = spy_df.xs(spy_df.columns[0][0], level=0, axis=1)
        else:
            if "Adj Close" in spy_df.columns:
                spy = spy_df["Adj Close"]
            elif "Close" in spy_df.columns:
                spy = spy_df["Close"]
            else:
                spy = spy_df.iloc[:, 0]

        spy.name = "SPY"
        spy = spy.reindex(idx).ffill().bfill()
        prices["SPY"] = spy

    return prices


def build_signals_and_returns():
    """
    Rebuild Momentum, Legal-only, and Legal-Enhanced Momentum
    exactly like the trading script, and also return the
    cross-sectional signals for regressions.
    """
    prices = load_price_data()
    idx = prices.index

    # Load events
    events = pd.read_csv(EVENTS_CSV, parse_dates=["date"])
    events["ticker"] = events["ticker"].astype(str).str.upper()

    # Legal universe = tickers in events AND in price data, excluding SPY
    legal_universe = sorted(set(events["ticker"]).intersection(set(prices.columns)))
    if "SPY" in legal_universe:
        legal_universe.remove("SPY")

    if len(legal_universe) == 0:
        raise RuntimeError("No overlap between legal events tickers and price tickers.")

    keep_cols = legal_universe + ["SPY"]
    prices = prices[keep_cols]
    idx = prices.index

    events = events[events["ticker"].isin(legal_universe)].copy()
    events = events.sort_values("date")

    # Daily returns
    returns_all = prices.pct_change(fill_method=None).fillna(0.0)
    returns_legal = returns_all[legal_universe]
    ret_SPY = returns_all["SPY"]

    # Legal matrices
    legal_severity = pd.DataFrame(0,    index=idx, columns=legal_universe)
    legal_palpha   = pd.DataFrame(0.0,  index=idx, columns=legal_universe)
    legal_ban      = pd.DataFrame(False,index=idx, columns=legal_universe)
    legal_extreme_neg = pd.DataFrame(False, index=idx, columns=legal_universe)

    events_by_ticker = {t: events[events["ticker"] == t] for t in legal_universe}

    for t in legal_universe:
        evt = events_by_ticker[t]
        if evt.empty:
            continue

        for _, row in evt.iterrows():
            d = row["date"]
            if pd.isna(d):
                continue

            pos = idx.searchsorted(d)
            if pos >= len(idx):
                continue
            trade_date = idx[pos]

            sev = int(row.get("severity_hat_rounded", row.get("severity_true", 0)))
            pal = float(row.get("p_alpha_hat", 0.0))
            try:
                direction = int(row.get("direction_true", 0))
            except Exception:
                direction = 0

            # max per day
            legal_severity.loc[trade_date, t] = max(legal_severity.loc[trade_date, t], sev)
            legal_palpha.loc[trade_date, t]   = max(legal_palpha.loc[trade_date, t], pal)

            # bans
            if sev >= BAN_SEV_THRESH and direction < 0:
                ban_start_pos = pos + 1
                ban_end_date  = trade_date + pd.Timedelta(days=BAN_LOOKBACK)
                ban_end_pos   = idx.searchsorted(ban_end_date)
                ban_end_pos   = min(ban_end_pos, len(idx) - 1)
                if ban_start_pos <= ban_end_pos:
                    legal_ban.iloc[ban_start_pos:ban_end_pos + 1,
                                   legal_ban.columns.get_loc(t)] = True

            # extreme negatives for systemic risk
            if sev >= EXTREME_SEV_THRESH and direction < 0:
                legal_extreme_neg.loc[trade_date, t] = True

    # Expected severity, surprise, legal_alpha
    expected_severity = (
        legal_severity
        .rolling(window=SURPRISE_WIN, min_periods=20)
        .mean()
        .fillna(legal_severity.mean().clip(lower=0.5))
    )
    surprise = legal_severity - expected_severity
    legal_alpha = surprise * legal_palpha  # core legal signal

    # Momentum signal (12m, 1-day lag)
    momentum_raw = prices[legal_universe].pct_change(LOOKBACK, fill_method=None).shift(1)

    # Cross-sectional z-scores
    def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
        mean = df.mean(axis=1)
        std  = df.std(axis=1).replace(0, np.nan)
        z = (df.sub(mean, axis=0)).div(std, axis=0)
        return z.fillna(0.0)

    z_mom   = cs_zscore(momentum_raw)
    z_legal = cs_zscore(legal_alpha)
    signal_combined = z_mom + GAMMA_LEGAL * z_legal

    # Portfolio construction
    def build_topN_portfolio(signal_matrix: pd.DataFrame,
                             tickers: list[str],
                             top_n: int,
                             banned: pd.DataFrame | None = None) -> pd.DataFrame:
        port = pd.DataFrame(0.0, index=signal_matrix.index, columns=tickers)
        for d in signal_matrix.index:
            row = signal_matrix.loc[d, tickers].replace([np.inf, -np.inf], np.nan).dropna()
            if row.empty:
                continue
            if banned is not None:
                mask_ok = ~banned.loc[d, row.index]
                row = row[mask_ok]
                if row.empty:
                    continue
            n = min(top_n, len(row))
            if n <= 0:
                continue
            top = row.nlargest(n).index
            port.loc[d, top] = 1.0 / n
        return port

    # Pure momentum portfolio
    port_mom = build_topN_portfolio(momentum_raw, legal_universe, TOP_N, banned=None)

    # Legal-only portfolio: rank purely by z_legal, with bans & systemic exposure
    port_legal = build_topN_portfolio(z_legal, legal_universe, TOP_N, banned=legal_ban)

    # Legal-enhanced momentum: combined signal + bans
    port_LEM = build_topN_portfolio(signal_combined, legal_universe, TOP_N, banned=legal_ban)

    # Exposure from systemic legal risk (only de-risking)
    daily_extreme_count = legal_extreme_neg.sum(axis=1)
    exposure = pd.Series(EXPO_NEUT, index=idx, dtype=float)
    for d in idx:
        c = daily_extreme_count.loc[d]
        if c >= EXTREME_COUNT:
            exposure.loc[d] = EXPO_LOW
        else:
            exposure.loc[d] = EXPO_HIGH

    # Returns
    ret_mom = (port_mom.shift(1)   * returns_legal).sum(axis=1)
    ret_leg_raw = (port_legal.shift(1) * returns_legal).sum(axis=1)
    ret_LEM_raw = (port_LEM.shift(1)   * returns_legal).sum(axis=1)

    # Apply exposure to legal-based strategies
    ret_legal = exposure * ret_leg_raw
    ret_LEM   = exposure * ret_LEM_raw

    ret_SPY = ret_SPY

    ret_mom   = ret_mom.fillna(0.0)
    ret_legal = ret_legal.fillna(0.0)
    ret_LEM   = ret_LEM.fillna(0.0)
    ret_SPY   = ret_SPY.fillna(0.0)

    return (
        ret_SPY,
        ret_mom,
        ret_legal,
        ret_LEM,
        z_mom,
        z_legal,
        returns_legal,
    )


# ===============================
# Regression helpers
# ===============================

def ols_regression(y: pd.Series, X: pd.DataFrame):
    """
    Simple OLS: y = X * beta + eps
    X should already include a constant column if you want one.
    Returns (beta, t_stats, r2).
    """
    # align
    df = pd.concat([y, X], axis=1).dropna()
    if df.shape[0] < X.shape[1] + 5:
        return None, None, None

    y_al = df.iloc[:, 0].values
    X_al = df.iloc[:, 1:].values  # assume first col is y

    # add intercept if not present (check if first col is all ones)
    if not np.allclose(X_al[:, 0], 1.0):
        X_al = np.column_stack([np.ones(len(y_al)), X_al])
    k = X_al.shape[1]
    n = X_al.shape[0]

    beta, residuals, rank, s = np.linalg.lstsq(X_al, y_al, rcond=None)
    if residuals.size > 0:
        rss = residuals[0]
    else:
        # manual RSS
        rss = np.sum((y_al - X_al @ beta) ** 2)
    tss = np.sum((y_al - y_al.mean()) ** 2)
    sigma2 = rss / (n - k)
    cov_beta = sigma2 * np.linalg.inv(X_al.T @ X_al)
    se_beta = np.sqrt(np.diag(cov_beta))
    t_stats = beta / se_beta
    r2 = 1 - rss / tss if tss > 0 else np.nan

    return beta, t_stats, r2


def fama_macbeth_legal_alpha(z_mom: pd.DataFrame,
                             z_legal: pd.DataFrame,
                             returns_legal: pd.DataFrame):
    """
    Cross-sectional regression of next-day returns on
    momentum z-scores and legal z-scores.

    Fama–MacBeth-style:
      r_{i,t+1} = a_t + b_mom,t * MOM_{i,t} + b_legal,t * LEGAL_{i,t} + eps_{i,t+1}

    We collect b_legal,t over time and compute its time-series mean and t-stat.
    """
    # forward returns (next-day)
    fwd_ret = returns_legal.shift(-1)

    betas_legal = []

    common_dates = z_mom.index.intersection(z_legal.index).intersection(fwd_ret.index)

    for d in common_dates:
        mom_row = z_mom.loc[d]
        leg_row = z_legal.loc[d]
        ret_row = fwd_ret.loc[d]

        # Align and drop missing
        df = pd.concat(
            [
                ret_row.rename("ret"),
                mom_row.rename("mom"),
                leg_row.rename("legal"),
            ],
            axis=1,
        ).dropna()

        # Need a decent cross-section
        if df.shape[0] < 20:
            continue

        # If legal has no cross-sectional variation that day, we can't estimate a slope
        if df["legal"].std(ddof=1) == 0:
            continue

        # Build X and y
        y = df["ret"].values
        X = np.column_stack([
            np.ones(len(df)),      # intercept
            df["mom"].values,
            df["legal"].values,
        ])

        # Solve y = X beta + eps with least squares; skip if rank-deficient
        try:
            beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue

        # Need full rank: intercept, mom, legal all linearly independent
        if rank < 3:
            continue

        # beta = [alpha_t, beta_mom,t, beta_legal,t]
        betas_legal.append(beta[2])

    if len(betas_legal) == 0:
        return np.nan, np.nan, 0

    betas_legal = np.array(betas_legal)
    mean_beta = betas_legal.mean()
    std_beta = betas_legal.std(ddof=1)
    N = len(betas_legal)

    t_beta = mean_beta / (std_beta / np.sqrt(N)) if std_beta > 0 else np.nan

    return mean_beta, t_beta, N


# ===============================
# MAIN: run all tests
# ===============================

def run_all_tests():
    (
        ret_SPY,
        ret_mom,
        ret_legal,
        ret_LEM,
        z_mom,
        z_legal,
        returns_legal,
    ) = build_signals_and_returns()

    # Align
    common = ret_LEM.index.intersection(ret_SPY.index).intersection(ret_mom.index).intersection(ret_legal.index)
    ret_SPY   = ret_SPY.loc[common]
    ret_mom   = ret_mom.loc[common]
    ret_legal = ret_legal.loc[common]
    ret_LEM   = ret_LEM.loc[common]

    ret_incr = ret_LEM - ret_mom  # incremental legal alpha

    # ==================================================
    # 1) BASIC PERFORMANCE
    # ==================================================
    print("=== BASIC PERFORMANCE (FULL SAMPLE) ===")
    for name, r in [
        ("SPY (benchmark)", ret_SPY),
        ("Momentum (MOM)", ret_mom),
        ("Legal-only", ret_legal),
        ("Legal-Enhanced MOM (LEM)", ret_LEM),
        ("LEM – MOM (incremental)", ret_incr),
    ]:
        cagr = annualized_return(r)
        vol  = annualized_vol(r)
        sh   = sharpe_ratio(r)
        # max drawdown:
        eq   = (1 + r).cumprod()
        roll_max = eq.cummax()
        dd = eq / roll_max - 1.0
        mdd = dd.min() if len(dd) > 0 else np.nan
        print(f"{name:26} | CAGR={cagr:6.2%}, VOL={vol:6.2%}, SHARPE={sh:5.2f}, MDD={-mdd:6.2%}")
    print()

    # ==================================================
    # 2) t-TESTS ON EXCESS RETURNS
    # ==================================================
    print("=== t-TESTS ON EXCESS RETURNS (one-sided H0: alpha <= 0) ===")
    pairs = [
        ("Legal-only vs SPY", ret_legal, ret_SPY),
        ("LEM vs SPY",        ret_LEM,   ret_SPY),
        ("LEM vs MOM",        ret_LEM,   ret_mom),
        ("LEM–MOM vs 0",      ret_incr,  pd.Series(0.0, index=ret_incr.index)),
    ]
    for label, strat, bench in pairs:
        t_stat, p_val = t_test_excess_returns(strat, bench)
        print(f"{label:20} | t={t_stat:6.3f}, p(one-sided)={p_val:7.4f}")
    print()

    # ==================================================
    # 3) SHARPE DIFF TESTS (approx)
    # ==================================================
    print("=== SHARPE DIFF TESTS (via excess returns) ===")
    for label, strat, bench in [
        ("LEM vs SPY", ret_LEM, ret_SPY),
        ("LEM vs MOM", ret_LEM, ret_mom),
    ]:
        s_sh, b_sh, t_sh, p_sh = sharpe_diff_test(strat, bench)
        print(f"{label:12} | Sharpe_strat={s_sh:5.2f}, Sharpe_bench={b_sh:5.2f}, "
              f"t={t_sh:6.3f}, p(one-sided)={p_sh:7.4f}")
    print()

    # ==================================================
    # 4) BOOTSTRAPPED SHARPE CI (strategy & incremental)
    # ==================================================
    print("=== BOOTSTRAPPED SHARPE CI (LEM and LEM–MOM) ===")
    for label, series in [
        ("LEM", ret_LEM),
        ("LEM–MOM", ret_incr),
    ]:
        lower, upper, _ = bootstrap_sharpe_ci(series)
        print(f"{label:7} | 95% Sharpe CI: [{lower:5.3f}, {upper:5.3f}]")
    print()

    # ==================================================
    # 5) DEFLATED / PROBABILISTIC SHARPE (PSR)
    # ==================================================
    print("=== PROBABILISTIC SHARPE (PSR-style) ===")
    for label, series in [
        ("Legal-only", ret_legal),
        ("LEM",        ret_LEM),
        ("LEM–MOM",    ret_incr),
    ]:
        psr = deflated_sharpe_ratio(series, sr_threshold=0.0, n_strats=1)
        print(f"{label:10} | Prob(Sharpe > 0) ~ {psr:5.3f}")
    print()

    # ==================================================
    # 6) YEARLY STABILITY REPORT (LEM vs SPY and vs MOM)
    # ==================================================
    print("=== YEARLY STABILITY: LEM vs SPY ===")
    yearly_spy = yearly_stability_report(ret_LEM, ret_SPY)
    for row in yearly_spy:
        print(
            f"{row['year']}: "
            f"CAGR_LEM={row['cagr_strategy']:6.2%}, "
            f"CAGR_SPY={row['cagr_benchmark']:6.2%}, "
            f"Sharpe_LEM={row['sharpe_strategy']:5.2f}, "
            f"Sharpe_SPY={row['sharpe_benchmark']:5.2f}, "
            f"t_excess={row['t_stat_excess']:6.3f}, "
            f"p_excess={row['p_val_excess']:7.4f}"
        )
    print()

    print("=== YEARLY STABILITY: LEM vs MOM ===")
    yearly_mom = yearly_stability_report(ret_LEM, ret_mom)
    for row in yearly_mom:
        print(
            f"{row['year']}: "
            f"CAGR_LEM={row['cagr_strategy']:6.2%}, "
            f"CAGR_MOM={row['cagr_benchmark']:6.2%}, "
            f"Sharpe_LEM={row['sharpe_strategy']:5.2f}, "
            f"Sharpe_MOM={row['sharpe_benchmark']:5.2f}, "
            f"t_excess={row['t_stat_excess']:6.3f}, "
            f"p_excess={row['p_val_excess']:7.4f}"
        )
    print()

    # ==================================================
    # 7) REGRESSION ANALYSIS
    #    (Time-series + Fama–MacBeth cross-sectional)
    # ==================================================

    print("=== TIME-SERIES REGRESSIONS (daily, OLS) ===")

    # Legal-only vs SPY and MOM
    # Model 1: r_legal = alpha + b_mkt * r_SPY
    df1 = pd.DataFrame({
        "r_legal": ret_legal,
        "r_SPY":   ret_SPY,
    }).dropna()
    beta1, t1, r2_1 = ols_regression(
        df1["r_legal"],
        pd.DataFrame({"const": 1.0, "r_SPY": df1["r_SPY"]})
    )
    if beta1 is not None:
        print(f"Legal-only ~ SPY: alpha={beta1[0]:.6f} (t={t1[0]:.2f}), "
              f"beta_SPY={beta1[1]:.2f} (t={t1[1]:.2f}), R2={r2_1:.3f}")

    # Model 2: r_LEM = alpha + b_mkt * r_SPY + b_mom * r_MOM
    df2 = pd.DataFrame({
        "r_LEM":   ret_LEM,
        "r_SPY":   ret_SPY,
        "r_MOM":   ret_mom,
    }).dropna()
    beta2, t2, r2_2 = ols_regression(
        df2["r_LEM"],
        pd.DataFrame({"const": 1.0, "r_SPY": df2["r_SPY"], "r_MOM": df2["r_MOM"]})
    )
    if beta2 is not None:
        # order: alpha, beta_SPY, beta_MOM
        print(f"LEM ~ SPY + MOM: alpha={beta2[0]:.6f} (t={t2[0]:.2f}), "
              f"beta_SPY={beta2[1]:.2f} (t={t2[1]:.2f}), "
              f"beta_MOM={beta2[2]:.2f} (t={t2[2]:.2f}), R2={r2_2:.3f}")
    print()

    print("=== FAMA–MACBETH CROSS-SECTIONAL REGRESSION (next-day returns on MOM & LEGAL) ===")
    mean_beta_legal, t_beta_legal, N_cs = fama_macbeth_legal_alpha(
        z_mom, z_legal, returns_legal
    )
    print(f"Average legal beta (cross-sectional): {mean_beta_legal:.6f}")
    print(f"t-stat for legal beta: {t_beta_legal:.2f} over {N_cs} cross-sections")


if __name__ == "__main__":
    run_all_tests()
