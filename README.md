# Legal Risk–Enhanced Momentum Factor

## Overview
This project develops and evaluates a legal-risk–enhanced momentum (LEM) strategy that integrates legal event signals with traditional price momentum in U.S. equities. The goal is to test whether legal risk information improves risk-adjusted performance and downside behavior relative to standard momentum and the market benchmark.

---

## Motivation
Price momentum is a well-documented anomaly, but it can perform poorly during periods of elevated uncertainty and regime shifts. Legal events such as lawsuits, regulatory actions, and enforcement announcements introduce asymmetric downside risk that is not captured by price signals alone. This project studies whether conditioning momentum exposure on legal risk improves performance and robustness.

---

## Strategy Construction
Three strategies are evaluated:
- **Momentum (MOM):** baseline price momentum strategy  
- **Legal-only:** strategy driven solely by legal risk signals  
- **Legal-Enhanced Momentum (LEM):** momentum adjusted using legal risk information  

The LEM strategy is designed to preserve momentum exposure during normal conditions while adapting positioning when legal risk is elevated.

---

## Empirical Evaluation
The strategy is evaluated using daily U.S. equity data with the following analyses:
- Full-sample performance metrics (CAGR, volatility, Sharpe, max drawdown)
- Incremental performance versus momentum (LEM − MOM)
- One-sided t-tests on excess returns
- Sharpe ratio difference tests
- Bootstrapped Sharpe confidence intervals
- Probabilistic Sharpe Ratio (PSR-style)
- Year-by-year stability analysis
- Time-series regressions controlling for market and momentum factors
- Fama–MacBeth cross-sectional regressions

---

## Key Results
- The LEM strategy achieves a higher Sharpe ratio and lower drawdown than both SPY and standard momentum.
- Incremental performance over momentum is positive, with improved downside protection.
- Statistical tests show significant outperformance versus the market and marginal but consistent improvement over momentum.
- Performance is particularly strong during volatile and crisis periods.
- Results suggest legal risk acts as a conditioning signal rather than a standalone cross-sectional predictor.

---

## Project Structure

### Data
- `sp500_adjclose_2005_2023.csv`  
  Daily adjusted close prices for S&P 500 constituents used to construct momentum signals and benchmark returns.

- `legal_events.csv`  
  Raw legal and regulatory event data.

- `legal_events_aligned.csv`  
  Legal events aligned to trading dates and matched to equity returns.

- `legal_events_scored.csv`  
  Processed legal risk scores used as inputs to the strategy.

---

### Strategy Construction
- `LRP_TS.py`  
  Implements the legal-risk-enhanced momentum strategy, combining legal risk signals with traditional price momentum.

---

### Legal Risk Processing
- `generate_legal_events.py`  
  Generates or simulates legal event data for empirical testing.

- `legal_score_CSV.py`  
  Transforms raw legal events into quantitative legal risk scores.

---

### Statistical Evaluation
- `Stats_Tests.py`  
  Comprehensive statistical analysis including performance metrics, hypothesis tests, Sharpe ratio comparisons, bootstrap confidence intervals, time-series regressions, and stability analysis.

---

### Outputs
- `equity_curves_legal_enhanced_momentum.png`  
  Equity curve comparison between SPY, momentum, and legal-risk-enhanced momentum strategies.

- `equity_incremental_legal_overlay.png`  
  Incremental performance visualization showing the contribution of legal risk over momentum.

---

## Notes
This project focuses on empirical evaluation and interpretability rather than production deployment. Results are presented under the physical measure and do not account for transaction costs or implementation constraints.

---

## Disclaimer
This project is for educational and research purposes only and does not constitute financial advice or a trading recommendation.
