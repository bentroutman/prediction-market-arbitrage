# Prediction Market Arbitrage Analysis
This project analyzes prediction markets (Kalshi, Polymarket, and PredictIt), collecting and cleaning data from APIs using scikit-learn to match similar markets on each site and identify market efficiencies, especially through arbitrage opportunities.

---

## Project Overview
- Automated scrapers using API clients for Kalshi, Polymarket, and PredictIt
- Data processing using machine learning to combine and standardize markets across platforms
- Identify arbitrage opportunities and send a notification to the user
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
- While the model is very profitable, most profitable markets featured low liquidity, preventing large positions from being filled at favorable prices
- Long-duration markets lock up capital, resulting in significant opportunity loss

---

## Key Takeaway
Models proved very effective at identifying arbitrage opportunities, but profitability depends heavily on liquidity, market depth, and execution constraints. Capital lockup significantly reduces effective ROI, since long-dated markets tie up funds for months or years. While the text similarity and market-matching algorithms (TF-IDF, RapidFuzz) I used can somewhat successfully map related markets across platforms, human observance is needed to prevent major errors. Despite these factors, under the right circumstances, arbitrage opportunities are frequent and highly lucrative across all three prediction markets analyzed.
