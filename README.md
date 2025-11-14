# Prediction Market Arbitrage Analysis
This project analyzes prediction markets (Kalshi, Polymarket, and PredictIt), collecting and cleaning data from APIs using scikit-learn to match similar markets on each site and identify market efficiencies, especially through arbitrage opportunities.

---

## Project Overview
- Automated scrapers using API clients for Kalshi, Polymarket, and PredictIt
- Data processing using machine learning to combine and standardize markets across platforms
- Analysis of price movements and implied probabilities
- Example datasets included for quick reproducability
- Jupyter Notebook Python files provided

---

## Technologies Used
- Python: pandas, NumPy, requests, JSON, time, re, datetime, os
- Machine Learning / Pattern Matching: scikit-learn (TfidfVectorizer, cosine_similarity),  rapidfuzz (fuzz)
- Jupyter Notebook

---

## Analysis / Conclusions
- Located 75 arbitrage opportunities of at least 3% ROI across the three sites from 7/16/25 to 8/16/25
- Mean ROI of 9.41%
- 15 of 75 (20%) were able to be sold for a profit within the one month period, averaging a sell date 12 days after purchase
- 46 of 75 (61.3%) featured end dates in 2026 or later
- While the model is very profitable, most profitable markets featured low liquidity, preventing large positions from being filled at favorable prices.
