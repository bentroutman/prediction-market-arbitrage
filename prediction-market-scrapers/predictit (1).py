#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Scrapes PredictIt api for open markets

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
    
    output_path = r"PredictIt.csv" # replace with desired file location
    
    url = "https://www.predictit.org/api/marketdata/all/"
    response = requests.get(url)
    data = response.json()
    
    rows = []
    
    for market in data.get("markets", []):
        market_id = market.get("id")
        market_name = market.get("name")
        end_date = market.get("timeStamp")
    
        for contract in market.get("contracts", []):
            contract_name = contract.get("name")
            contract_id = contract.get("id")
            yes_price = contract.get("bestBuyYesCost")
            no_price = contract.get("bestBuyNoCost")
    
            rows.append({
                "market_id": market_id,
                "contract_id": contract_id,
                "market_name": market_name + " - " + contract_name,
                "end_date": end_date,
                "side": "YES",
                "price": yes_price
            })
    
            rows.append({
                "market_id": market_id,
                "contract_id": contract_id,
                "market_name": market_name + " - " + contract_name,
                "end_date": end_date,
                "side": "NO",
                "price": no_price
            })
    
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

