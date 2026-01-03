import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except ImportError:
    yf = None

###############################################################################
# CONFIG
###############################################################################

PRICE_CSV   = "sp500_adjclose_2005_2023.csv"
EVENTS_CSV  = "legal_events_scored.csv"

LOOKBACK       = 252        # 12-month momentum
TOP_N          = 10         # number of names in the portfolio

# Ban logic (more realistic and actually active)
BAN_LOOKBACK   = 30         # days a stock is banned from LONGS after bad legal hit
BAN_SEV_THRESH = 3          # severity threshold for bans
SURPRISE_WIN   = 252        # rolling window for expected severity

# === LEGAL ALPHA SHAPING ===
# Exponential decay so legal events have a multi-day effect
LEGAL_DECAY_HALFLIFE = 10   # in trading days (roughly 2 weeks)
# Weight on legal factor in combined signal (static)
GAMMA_LEGAL_STATIC   = 2.0
# Optionally re-scale legal vs momentum dynamically based on cross-sectional vol
USE_DYNAMIC_GAMMA    = True
TARGET_LEGAL_WEIGHT  = 1.0  # relative strength vs momentum when dynamic scaling

# Systemic risk de-risking (no leverage up, only down)
EXPO_HIGH  = 1.0            # we never go above 1.0 now
EXPO_NEUT  = 1.0
EXPO_LOW   = 0.7            # de-risk when many extreme events
EXTREME_SEV_THRESH = 4      # severity threshold for "extreme" events
EXTREME_COUNT      = 3      # if >= this many extreme hits in a day => de-risk

###############################################################################
# 1. LOAD PRICES
###############################################################################

prices = pd.read_csv(PRICE_CSV, parse_dates=[0])
prices = prices.sort_values(prices.columns[0]).set_index(prices.columns[0])
prices = prices.astype(float)

idx = prices.index
print(f"[INFO] Price history: {idx[0].date()} -> {idx[-1].date()}")
print(f"[INFO] Raw price tickers (sample): {list(prices.columns)[:10]} ...")

###############################################################################
# 2. ENSURE SPY EXISTS
###############################################################################

if "SPY" not in prices.columns:
    if yf is None:
        raise RuntimeError("SPY not in price CSV and yfinance is not installed.")
    print("[INFO] SPY not found in CSV. Downloading from yfinance...")

    spy_df = yf.download(
        "SPY",
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

###############################################################################
# 3. LOAD LEGAL EVENTS & DEFINE LEGAL UNIVERSE
###############################################################################

events = pd.read_csv(EVENTS_CSV, parse_dates=["date"])
events["ticker"] = events["ticker"].astype(str).str.upper()

# Legal universe = tickers in events AND in price data
legal_universe = sorted(set(events["ticker"]).intersection(set(prices.columns)))
if "SPY" in legal_universe:
    legal_universe.remove("SPY")

if len(legal_universe) == 0:
    raise RuntimeError("No overlap between legal events tickers and price tickers.")

print(f"[INFO] LEGAL UNIVERSE SIZE: {len(legal_universe)}")
print(f"[INFO] Legal universe tickers (sample): {legal_universe[:10]}")

# Restrict prices to legal universe + SPY
keep_cols = legal_universe + ["SPY"]
prices = prices[keep_cols]
idx = prices.index

# Restrict events to the legal universe only
events = events[events["ticker"].isin(legal_universe)].copy()
events = events.sort_values("date")
print(f"[INFO] Events aligned to legal universe: {len(events)}")

# Daily returns (including SPY)
returns = prices.pct_change(fill_method=None).fillna(0.0)

###############################################################################
# 4. BUILD DAILY LEGAL MATRICES
###############################################################################

legal_severity = pd.DataFrame(0,    index=idx, columns=legal_universe)
legal_palpha   = pd.DataFrame(0.0,  index=idx, columns=legal_universe)
legal_ban      = pd.DataFrame(False,index=idx, columns=legal_universe)

# Track extreme negative hits for systemic de-risking
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

        # Keep max per day
        legal_severity.loc[trade_date, t] = max(legal_severity.loc[trade_date, t], sev)
        legal_palpha.loc[trade_date, t]   = max(legal_palpha.loc[trade_date, t], pal)

        # Bans: moderately bad-and-worse negative events
        if sev >= BAN_SEV_THRESH and direction < 0:
            ban_start_pos = pos + 1
            ban_end_date  = trade_date + pd.Timedelta(days=BAN_LOOKBACK)
            ban_end_pos   = idx.searchsorted(ban_end_date)
            ban_end_pos   = min(ban_end_pos, len(idx) - 1)
            if ban_start_pos <= ban_end_pos:
                legal_ban.iloc[ban_start_pos:ban_end_pos + 1,
                               legal_ban.columns.get_loc(t)] = True

        # Extreme negatives for systemic risk
        if sev >= EXTREME_SEV_THRESH and direction < 0:
            legal_extreme_neg.loc[trade_date, t] = True

print(f"[INFO] Any bans? {legal_ban.values.any()}")
if legal_ban.values.any():
    print(f"[INFO] Fraction of banned ticker-days: {legal_ban.values.mean():.4%}")

###############################################################################
# 5. EXPECTED SEVERITY, SURPRISE, LEGAL_ALPHA + DECAY
###############################################################################

expected_severity = (
    legal_severity
    .rolling(window=SURPRISE_WIN, min_periods=20)
    .mean()
    .fillna(legal_severity.mean().clip(lower=0.5))
)

surprise = legal_severity - expected_severity
legal_alpha_inst = surprise * legal_palpha  # instantaneous legal signal

# === NEW: Exponential decay so events matter for ~several days ===
legal_alpha = legal_alpha_inst.ewm(halflife=LEGAL_DECAY_HALFLIFE).mean()

###############################################################################
# 6. MOMENTUM SIGNAL (12M lookback, 1-day lag)
###############################################################################

momentum_raw = prices[legal_universe].pct_change(LOOKBACK, fill_method=None).shift(1)

###############################################################################
# 7. CROSS-SECTIONAL Z-SCORES & COMBINED SIGNAL
###############################################################################

def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean(axis=1)
    std  = df.std(axis=1).replace(0, np.nan)
    z = (df.sub(mean, axis=0)).div(std, axis=0)
    return z.fillna(0.0)

z_mom   = cs_zscore(momentum_raw)
z_legal = cs_zscore(legal_alpha)

# Optional dynamic scaling so legal & momentum have comparable "strength"
if USE_DYNAMIC_GAMMA:
    # Cross-sectional dispersion proxies for each day
    mom_disp   = z_mom.abs().mean(axis=1)
    legal_disp = z_legal.abs().mean(axis=1).replace(0, np.nan)

    # Ratio of dispersions; where legal_disp is tiny, fallback to static
    gamma_series = (mom_disp / legal_disp) * TARGET_LEGAL_WEIGHT
    gamma_series = gamma_series.replace([np.inf, -np.inf], np.nan).fillna(GAMMA_LEGAL_STATIC)
    gamma_series = gamma_series.clip(lower=0.5, upper=4.0)
    gamma_df = pd.DataFrame(np.repeat(gamma_series.values[:, None], len(legal_universe), axis=1),
                            index=z_mom.index, columns=z_mom.columns)
    signal_combined = z_mom + gamma_df * z_legal
else:
    signal_combined = z_mom + GAMMA_LEGAL_STATIC * z_legal

###############################################################################
# 8. PORTFOLIO CONSTRUCTION (LONG-ONLY)
###############################################################################

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

# Pure momentum portfolio (top-N by momentum only)
port_mom = build_topN_portfolio(momentum_raw, legal_universe, TOP_N, banned=None)

# Legal-only portfolio: top-N by legal signal only (with bans)
port_legal_only = build_topN_portfolio(legal_alpha, legal_universe, TOP_N, banned=legal_ban)

# Legal-enhanced momentum: top-N by combined signal, with bans
port_LEM = build_topN_portfolio(signal_combined, legal_universe, TOP_N, banned=legal_ban)

###############################################################################
# 9. SYSTEMIC LEGAL RISK → EXPOSURE (ONLY DOWN, NEVER UP)
###############################################################################

daily_extreme_count = legal_extreme_neg.sum(axis=1)

exposure = pd.Series(EXPO_NEUT, index=idx, dtype=float)
for d in idx:
    c = daily_extreme_count.loc[d]
    if c >= EXTREME_COUNT:
        exposure.loc[d] = EXPO_LOW   # de-risk
    else:
        exposure.loc[d] = EXPO_HIGH  # which equals 1.0 here

print(f"[INFO] Average exposure level: {exposure.mean():.3f}")
print(f"[INFO] Exposure distribution:")
print(exposure.value_counts().sort_index())

###############################################################################
# 10. DAILY RETURNS
###############################################################################

# Base momentum long-only
ret_mom = (port_mom.shift(1) * returns[legal_universe]).sum(axis=1)

# Legal-only long-only
ret_legal_only = (port_legal_only.shift(1) * returns[legal_universe]).sum(axis=1)

# Legal-enhanced momentum with de-risking
ret_LEM_raw = (port_LEM.shift(1) * returns[legal_universe]).sum(axis=1)
ret_LEM     = exposure * ret_LEM_raw

# SPY benchmark (for context)
ret_SPY = returns["SPY"]

###############################################################################
# 11. PERFORMANCE METRICS
###############################################################################

def perf_stats(r: pd.Series):
    r = r.dropna()
    if len(r) == 0:
        return np.nan, np.nan, np.nan, np.nan

    equity = (1.0 + r).cumprod()
    n_days = len(r)
    n_years = n_days / 252.0

    cagr = equity.iloc[-1] ** (1.0 / n_years) - 1.0 if n_years > 0 else np.nan
    vol = r.std() * np.sqrt(252)
    sharpe = cagr / vol if (vol is not None and vol > 0) else np.nan

    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    mdd = dd.min()

    return cagr, vol, sharpe, -mdd  # positive max drawdown

c_SPY        = perf_stats(ret_SPY)
c_mom        = perf_stats(ret_mom)
c_LEM        = perf_stats(ret_LEM)
c_legal_only = perf_stats(ret_legal_only)

# Incremental alpha series: LEM minus base momentum
ret_incremental = (ret_LEM - ret_mom)
c_incr          = perf_stats(ret_incremental)

print("\n===== PERFORMANCE (LEGAL-ENHANCED MOMENTUM) =====")
print(f"{'SPY (long-only)':16} | CAGR={c_SPY[0]:.3%}, VOL={c_SPY[1]:.3%}, SHARPE={c_SPY[2]:.2f}, MDD={c_SPY[3]:.2%}")
print(f"{'Momentum':16}       | CAGR={c_mom[0]:.3%}, VOL={c_mom[1]:.3%}, SHARPE={c_mom[2]:.2f}, MDD={c_mom[3]:.2%}")
print(f"{'Legal-only':16}     | CAGR={c_legal_only[0]:.3%}, VOL={c_legal_only[1]:.3%}, SHARPE={c_legal_only[2]:.2f}, MDD={c_legal_only[3]:.2%}")
print(f"{'Legal-Enh Mom':16}  | CAGR={c_LEM[0]:.3%}, VOL={c_LEM[1]:.3%}, SHARPE={c_LEM[2]:.2f}, MDD={c_LEM[3]:.2%}")
print(f"{'LEM – MOM (incr)':16} | CAGR={c_incr[0]:.3%}, VOL={c_incr[1]:.3%}, SHARPE={c_incr[2]:.2f}, MDD={c_incr[3]:.2%}")

###############################################################################
# 12. EQUITY CURVES PLOT
###############################################################################

equity_SPY        = (1.0 + ret_SPY).cumprod()
equity_mom        = (1.0 + ret_mom).cumprod()
equity_LEM        = (1.0 + ret_LEM).cumprod()
equity_legal_only = (1.0 + ret_legal_only).cumprod()
equity_incr       = (1.0 + ret_incremental).cumprod()

# Normalize to 1.0 at start
for eq in [equity_SPY, equity_mom, equity_LEM, equity_legal_only, equity_incr]:
    if len(eq) > 0 and eq.iloc[0] != 0:
        eq /= eq.iloc[0]

plt.figure(figsize=(11, 7))
equity_SPY.plot(label="SPY (long-only)", alpha=0.8)
equity_mom.plot(label="Momentum (Legal Universe)", alpha=0.8)
equity_LEM.plot(label="Legal-Enhanced Momentum", linewidth=2.0)
equity_legal_only.plot(label="Legal-only (Top-N Legal Alpha)", linestyle="--", alpha=0.9)
plt.title("Equity Curves: SPY vs Momentum vs Legal-Enhanced vs Legal-only")
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("equity_curves_legal_enhanced_momentum.png")
plt.show()

# Incremental equity (LEM – MOM) to show pure contribution of legal overlay
plt.figure(figsize=(10, 5))
equity_incr.plot(label="LEM – MOM Incremental Equity")
plt.title("Incremental Equity from Legal Overlay (LEM – MOM)")
plt.xlabel("Date")
plt.ylabel("Growth of $1 from Overlay")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("equity_incremental_legal_overlay.png")
plt.show()

print("[INFO] Saved equity curve plots to:")
print("  - equity_curves_legal_enhanced_momentum.png")
print("  - equity_incremental_legal_overlay.png")
