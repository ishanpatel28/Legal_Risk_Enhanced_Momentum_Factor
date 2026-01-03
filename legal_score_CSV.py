import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)

# -----------------------------
# Coarse event-type mapping
# -----------------------------
# Core alpha-driving legal events (we keep these distinct)
ALPHA_CORE_TYPES = {
    "SEC_ENFORCEMENT",
    "FDA_ACTION",
    "FDA_APPROVAL",
    "ANTITRUST",
    "PATENT_IP",
    "PRODUCT_LIABILITY",
    "ACCOUNTING_FRAUD",
    "DATA_PRIVACY",
    "ENVIRONMENTAL",
    "LABOR_LAWSUIT",
    "EARNINGS_DISCLOSURE_ISSUE",
}

# Other regulatory / admin-y events
OTHER_REG_TYPES = {
    "REGULATORY_UPDATE",
    "FINRA_ACTION",
    "SUPPLY_CHAIN",
    "OTHER",
    "MINOR_REG_NOTICE",
    "FOLLOWUP_FILING",
}


def map_event_type(raw: str) -> str:
    """
    Map raw event_type into coarse classes used by the NLP reader.
    - Core alpha types stay as-is
    - Other regulatory types collapse into OTHER_REGULATORY
    - Everything else -> GENERIC_NOISE
    """
    raw = str(raw).upper()
    if raw in ALPHA_CORE_TYPES:
        return raw
    if raw in OTHER_REG_TYPES:
        return "OTHER_REGULATORY"
    return "GENERIC_NOISE"


def add_coarse_type_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event_type_coarse"] = df["event_type"].astype(str).apply(map_event_type)
    return df


def make_text_column(df: pd.DataFrame) -> pd.Series:
    """
    Build the raw text the NLP model will see: industry tag (if present) + headline + body.
    If no industry column exists, we use [UNKNOWN].
    """
    headline = df["headline"].fillna("")
    body = df["body"].fillna("")
    industry = df.get("industry", pd.Series(["UNKNOWN"] * len(df)))

    return "[" + industry.astype(str) + "] " + headline.astype(str) + " " + body.astype(str)


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        strip_accents="unicode",
        lowercase=True,
    )


def collapse_rare_classes(y: pd.Series, min_count: int = 2) -> pd.Series:
    """
    Any event_type_coarse class with < min_count samples gets collapsed into GENERIC_NOISE.
    This avoids train_test_split(stratify=...) crashes when a class has only 1 sample.
    """
    vc = y.value_counts()
    rare_labels = vc[vc < min_count].index.tolist()
    if rare_labels:
        print(f"[INFO] Collapsing rare classes (count < {min_count}) into GENERIC_NOISE: {rare_labels}")
        y = y.copy()
        y[y.isin(rare_labels)] = "GENERIC_NOISE"
    return y


def train_models(
    X_text: pd.Series,
    y_type: pd.Series,
    y_severity: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[TfidfVectorizer, LogisticRegression, Ridge]:
    """
    Train:
      - a TF-IDF vectorizer
      - a multinomial logistic regression for event type
      - a ridge regression model for severity

    Returns the fitted (vectorizer, clf_type, reg_severity).
    """

    # Collapse ultra-rare classes before splitting/stratifying
    y_type = collapse_rare_classes(y_type, min_count=2)

    # If after collapsing we still somehow have only 1 class, bail on classification
    unique_classes = y_type.unique()
    if len(unique_classes) < 2:
        raise ValueError(
            f"Not enough distinct classes to train classifier. "
            f"Got classes: {list(unique_classes)}"
        )

    # Split before vectorizing so we don't leak test data into vocab fitting
    X_train_text, X_test_text, y_train_type, y_test_type, y_train_sev, y_test_sev = train_test_split(
        X_text,
        y_type,
        y_severity,
        test_size=test_size,
        random_state=random_state,
        stratify=y_type,
    )

    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # Event type classifier
    clf_type = LogisticRegression(
        multi_class="multinomial",
        max_iter=1000,
        n_jobs=-1,
    )
    clf_type.fit(X_train, y_train_type)

    # Severity regressor
    reg_sev = Ridge(alpha=1.0)
    reg_sev.fit(X_train, y_train_sev)

    # ---- Evaluation ----
    print("===== Event Type Classification (Validation) =====")
    y_pred_type = clf_type.predict(X_test)
    print(classification_report(y_test_type, y_pred_type))
    print("Confusion matrix:")
    print(confusion_matrix(y_test_type, y_pred_type))

    print("\n===== Severity Regression (Validation) =====")
    y_pred_sev = reg_sev.predict(X_test)
    mae = mean_absolute_error(y_test_sev, y_pred_sev)
    r2 = r2_score(y_test_sev, y_pred_sev)
    print(f"MAE (severity): {mae:.3f}")
    print(f"R^2 (severity): {r2:.3f}")

    return vectorizer, clf_type, reg_sev


def logistic_alpha_prob(severity_hat: np.ndarray, k: float = 1.2, center: float = 2.5) -> np.ndarray:
    """
    Smooth mapping from severity_hat to probability of being alpha.
    p = 1 / (1 + exp(-k*(severity - center)))
    """
    sev = np.asarray(severity_hat, dtype=float)
    return 1.0 / (1.0 + np.exp(-k * (sev - center)))


def main():
    # 1. Load events CSV (aligned legal events)
    csv_path = "legal_events_aligned.csv"
    df = pd.read_csv(csv_path)

    # 2. Parse dates (not strictly needed for NLP, but good hygiene)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 3. Ensure required columns exist
    required_cols = ["headline", "body", "event_type", "severity_true"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in {csv_path}, but it was missing.")

    # Drop rows without any text at all
    df = df[df["headline"].notna() | df["body"].notna()]
    if df.empty:
        raise ValueError("No rows with usable text (headline/body) in the dataset.")

    # 4. Add coarse event type labels
    df = add_coarse_type_column(df)

    # 5. Build text column
    df["text"] = make_text_column(df)

    # 6. Prepare targets
    y_type = df["event_type_coarse"]
    y_severity = pd.to_numeric(df["severity_true"], errors="coerce")

    # drop rows where severity_true is NaN
    mask_valid_sev = ~y_severity.isna()
    if not mask_valid_sev.all():
        dropped = (~mask_valid_sev).sum()
        print(f"[INFO] Dropping {dropped} rows with invalid severity_true.")
        df = df[mask_valid_sev].copy()
        y_type = df["event_type_coarse"]
        y_severity = pd.to_numeric(df["severity_true"], errors="coerce")

    # 7. Train models (and print validation stats)
    vectorizer, clf_type, reg_sev = train_models(df["text"], y_type, y_severity)

    # 8. Fit on all data before generating final predictions
    X_all = vectorizer.fit_transform(df["text"])
    clf_type.fit(X_all, y_type)
    reg_sev.fit(X_all, y_severity)

    # 9. Generate predictions for entire dataset
    event_type_hat = clf_type.predict(X_all)
    severity_hat = reg_sev.predict(X_all)
    severity_hat_rounded = np.clip(np.rint(severity_hat), 0, 5).astype(int)
    p_alpha_hat = logistic_alpha_prob(severity_hat)

    df["event_type_hat"] = event_type_hat
    df["severity_hat"] = severity_hat
    df["severity_hat_rounded"] = severity_hat_rounded
    df["p_alpha_hat"] = p_alpha_hat

    # 10. Save scored CSV (keep all original columns like original_date, ret_fwd_5d, etc.)
    out_path = "legal_events_scored.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved scored events to: {out_path}")


if __name__ == "__main__":
    main()
