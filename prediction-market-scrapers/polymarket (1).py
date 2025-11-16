#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Scrapes polymarket api for open markets

def prediction_markets():
    import requests
    import pandas as pd
    import json
    import time
    import os
    import re
    from datetime import datetime
    from tqdm.notebook import tqdm
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    from rapidfuzz import fuzz

    rows = []
    offset = 0
    limit = 1000
    
    while True:
        url = f"https://gamma-api.polymarket.com/events?closed=false&limit={limit}&offset={offset}"
        r = requests.get(url)
        events = r.json()
    
        if not events:
            break 
    
        for event in events:
            event_id = event.get("id")
            category = event.get("category", "")
            end_date = event.get("endDate")
            created_at = event.get("createdAt")
            status = "closed" if event.get("closed") else "open"
            event_liquidity = event.get("liquidity")
            event_volume_usd = event.get("volume")
    
            for market in event.get("markets", []):
                title = market.get("question", event.get("title"))
                market_liquidity = float(market.get("liquidity", event_liquidity) or 0)
                market_volume_usd = float(market.get("volume", event_volume_usd) or 0)
    
                outcome_names = json.loads(market.get("outcomes", "[]"))
                outcome_prices = json.loads(market.get("outcomePrices", "[]"))
    
                for name, price in zip(outcome_names, outcome_prices):
                    rows.append({
                        "event_id": event_id,
                        "title": title,
                        "outcome": name,
                        "price": float(price),
                        "end_date": end_date,
                        "liquidity": market_liquidity,
                        "volume_usd": market_volume_usd
                    })
    
        offset += limit
        time.sleep(0.2)
    
    df = pd.DataFrame(rows)
    df.to_csv(r"Polymarket.csv", index=False) # replace with desired file location

while True:
    import time
    prediction_markets()
    time.sleep(3600)

