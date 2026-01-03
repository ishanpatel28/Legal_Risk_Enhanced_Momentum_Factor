import numpy as np
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------

PRICE_CSV = "sp500_adjclose_2005_2023.csv"   # your price file
EVENTS_CSV = "legal_events.csv"             # the 4,000-event file you generated
OUT_CSV    = "legal_events_aligned.csv"

# Quantile cutoffs for mapping events -> days
MAJOR_Q = 0.02      # top/bottom 2% for major events
MED_Q_LOW = 0.10    # 10% tails for medium events

RNG_SEED = 42


# -----------------------------
# LOAD DATA
# -----------------------------

def load_prices(path: str) -> pd.DataFrame:
    """
    Expect format:
      Date, AAPL, MSFT, ...
    """
    df = pd.read_csv(path, parse_dates=[0])
    df = df.sort_values(df.columns[0])
    df = df.set_index(df.columns[0])
    df = df.astype(float)
    return df


def load_events(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns or "ticker" not in df.columns:
        raise ValueError("Events CSV must have 'date' and 'ticker' columns.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "ticker"])
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    return df


# -----------------------------
# BUILD VOLATILITY BUCKETS PER TICKER/YEAR
# -----------------------------

def build_vol_buckets(prices: pd.DataFrame):
    """
    For each ticker and year, precompute candidate date buckets:
      - major_pos: big up days
      - major_neg: big down days
      - med_pos: medium positive days
      - med_neg: medium negative days
      - calm: low-volatility "boring" days
    """
    idx = prices.index
    tickers = prices.columns
    returns = prices.pct_change(fill_method=None).fillna(0.0)

    buckets = {}   # (ticker, year, bucket_name) -> np.array of dates

    for t in tickers:
        ser = returns[t].dropna()
        if ser.empty:
            continue

        df_t = pd.DataFrame({
            "date": ser.index,
            "ret": ser.values
        })
        df_t["abs_ret"] = df_t["ret"].abs()
        df_t["year"] = df_t["date"].dt.year

        for year, grp in df_t.groupby("year"):
            if len(grp) < 30:
                # too few days; skip year-level buckets
                continue

            # quantiles
            q_low_major  = grp["ret"].quantile(MAJOR_Q)
            q_high_major = grp["ret"].quantile(1 - MAJOR_Q)
            q_low_med    = grp["ret"].quantile(MED_Q_LOW)
            q_high_med   = grp["ret"].quantile(1 - MED_Q_LOW)
            q_calm       = grp["abs_ret"].quantile(0.30)  # calm = bottom 30% abs returns

            major_neg = grp.loc[grp["ret"] <= q_low_major, "date"].values
            major_pos = grp.loc[grp["ret"] >= q_high_major, "date"].values

            med_neg = grp.loc[(grp["ret"] > q_low_major) & (grp["ret"] <= q_low_med), "date"].values
            med_pos = grp.loc[(grp["ret"] < q_high_major) & (grp["ret"] >= q_high_med), "date"].values

            calm = grp.loc[grp["abs_ret"] <= q_calm, "date"].values

            buckets[(t, year, "major_neg")] = major_neg
            buckets[(t, year, "major_pos")] = major_pos
            buckets[(t, year, "med_neg")]   = med_neg
            buckets[(t, year, "med_pos")]   = med_pos
            buckets[(t, year, "calm")]      = calm

    return returns, buckets


# -----------------------------
# EVENT → SEVERITY BUCKET & SIGN
# -----------------------------

def severity_bucket(sev: float) -> str:
    sev = float(sev)
    if sev >= 4.0:
        return "major"
    if sev >= 2.0:
        return "medium"
    return "minor"


def choose_bucket_name(sev_bucket: str, direction: int) -> str:
    """
    Map severity + direction into a bucket key used in buckets dict.
    """
    if sev_bucket == "major":
        if direction > 0:
            return "major_pos"
        elif direction < 0:
            return "major_neg"
        else:
            return "calm"
    elif sev_bucket == "medium":
        if direction > 0:
            return "med_pos"
        elif direction < 0:
            return "med_neg"
        else:
            return "calm"
    else:  # minor
        # minor events mostly go to calm days; direction can bias if desired
        return "calm"


# -----------------------------
# ALIGN EVENTS TO PRICE DAYS
# -----------------------------

def align_events_to_prices(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    returns: pd.DataFrame,
    buckets: dict,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    For each event row, choose a new 'aligned' trading date that matches:
      - severity_true (major / medium / minor)
      - direction_true (+1 / -1 / 0)
      - original year (or fallback)
    """
    rng = np.random.default_rng(rng_seed)
    idx = prices.index
    tickers = prices.columns

    # Make sure we don't crash if columns are missing
    if "severity_true" not in events.columns:
        raise ValueError("Events CSV must have 'severity_true' column.")
    if "direction_true" not in events.columns:
        raise ValueError("Events CSV must have 'direction_true' column.")

    events = events.copy()
    events["original_date"] = events["date"]
    aligned_dates = []

    # Precompute generic fallback candidate sets per ticker (all years)
    fallback_buckets = {}
    for (t, year, name), dates in buckets.items():
        if len(dates) == 0:
            continue
        key_all = (t, "ALL", name)
        if key_all not in fallback_buckets:
            fallback_buckets[key_all] = list(dates)
        else:
            fallback_buckets[key_all].extend(list(dates))

    # Global full trading-day fallback
    all_trading_days = idx.values

    for i, row in events.iterrows():
        t = str(row["ticker"]).upper()
        if t not in tickers:
            # ticker not in price universe; keep original date
            aligned_dates.append(row["original_date"])
            continue

        sev = float(row["severity_true"])
        sev_bucket = severity_bucket(sev)

        # direction_true might be string or numeric
        try:
            direction = int(row["direction_true"])
        except Exception:
            direction = 0

        orig_date = row["original_date"]
        year = orig_date.year

        # Decide bucket name to pull from
        bucket_name = choose_bucket_name(sev_bucket, direction)

        # 1. Try ticker-year-bucket
        candidates = buckets.get((t, year, bucket_name), np.array([]))

        # 2. If empty, try ticker-ALL-bucket (across all years)
        if len(candidates) == 0:
            candidates = np.array(fallback_buckets.get((t, "ALL", bucket_name), []))

        # 3. If still empty, try calm days for ticker ALL
        if len(candidates) == 0 and bucket_name != "calm":
            candidates = np.array(fallback_buckets.get((t, "ALL", "calm"), []))

        # 4. If still empty, fall back to any trading day
        if len(candidates) == 0:
            chosen = rng.choice(all_trading_days)
        else:
            chosen = rng.choice(candidates)

        aligned_dates.append(chosen)

    events["date"] = pd.to_datetime(aligned_dates)
    return events


# -----------------------------
# MAIN
# -----------------------------

def main():
    print("[INFO] Loading prices...")
    prices = load_prices(PRICE_CSV)

    print("[INFO] Loading events...")
    events = load_events(EVENTS_CSV)

    # Align tickers universe
    common_tickers = sorted(set(prices.columns).intersection(set(events["ticker"].unique())))
    prices = prices[common_tickers]
    events = events[events["ticker"].isin(common_tickers)].copy()

    print(f"[INFO] Price universe tickers: {len(common_tickers)}")
    print(f"[INFO] Events after aligning to universe: {len(events)}")

    print("[INFO] Building volatility buckets...")
    returns, buckets = build_vol_buckets(prices)

    print("[INFO] Aligning events to real price days...")
    events_aligned = align_events_to_prices(prices, events, returns, buckets, rng_seed=RNG_SEED)

    # Diagnostics: compare severity vs realized future move as a sanity check
    idx = prices.index

    def fwd_ret(ticker, ev_date, h=5):
        if ticker not in prices.columns:
            return np.nan
        col = prices[ticker]
        pos = idx.searchsorted(ev_date)
        if pos <= 0 or pos + h >= len(idx):
            return np.nan
        p0 = col.iloc[pos]
        pT = col.iloc[pos + h]
        return (pT / p0) - 1.0

    events_aligned["ret_fwd_5d"] = events_aligned.apply(
        lambda r: fwd_ret(r["ticker"], r["date"], h=5), axis=1
    )

    print("\n[INFO] Basic sanity check on aligned data:")
    print(events_aligned[["ticker", "original_date", "date", "severity_true", "direction_true", "ret_fwd_5d"]].head())

    print(f"\n[INFO] Saving aligned events to: {OUT_CSV}")
    events_aligned.to_csv(OUT_CSV, index=False)


if __name__ == "__main__":
    main()
