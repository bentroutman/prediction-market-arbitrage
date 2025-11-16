#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# Scrapes Kalshi api for open markets

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
    import csv
    
    output_path = r"Kalshi.csv" # replace with desired file location
    
    base_url = "https://api.elections.kalshi.com/trade-api/v2/events"
    params = {
        "with_nested_markets": "true",
        "status": "open",
        "limit": 200
    }
    
    rows = []
    cursor = None
    
    while True:
        if cursor:
            params["cursor"] = cursor
        else:
            params.pop("cursor", None)
    
        response = requests.get(base_url, params=params)
        data = response.json()
    
        events = data.get("events", [])
        for event in events:
            event_id = event.get("event_ticker")
            event_name = event.get("title") + " " + event.get("sub_title")
            for market in event.get("markets", []):
                market_id = market.get("ticker")
                market_subtitle = market.get("yes_sub_title") or ""
                end_date = market.get("close_time")
    
                full_name = f"{event_name} - {market_subtitle}" if market_subtitle else event_name
    
                rows.append({
                    "event_id": event_id,
                    "market_id": market_id,
                    "full_name": full_name,
                    "end_date": end_date,
                    "side": "YES",
                    "bid": market.get("yes_bid"),
                    "ask": market.get("yes_ask")
                })
                rows.append({
                    "event_id": event_id,
                    "market_id": market_id,
                    "full_name": full_name,
                    "end_date": end_date,
                    "side": "NO",
                    "bid": market.get("no_bid"),
                    "ask": market.get("no_ask")
                })
    
        cursor = data.get("cursor")
        if not cursor:
            break
    
    if rows:
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = rows[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        print("No data found to write.")
    
while True:
    import time
    prediction_markets()
    time.sleep(3600)

