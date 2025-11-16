#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Locates arbitrage opportunities on Polymarket, Kalshi, and PredictIt using data scraped from apis
# Sends a Discord notification when profitable arbitrage opportunities are available
# Run after prediction-market-scrapers and getting csv files on each prediction market

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

    # Replace with actual file locations
    match_files = [
        r"Kalshi_PredictIt_Matches.csv",
        r"Kalshi_Polymarket_Matches.csv",
        r"PredictIt_Polymarket_Matches.csv"
    ]
    
    for match_file in match_files:
        old_match_file = match_file.replace(".csv", "_Old.csv")
        if os.path.exists(match_file):
            try:
                if os.path.exists(old_match_file):
                    os.remove(old_match_file)  # Remove existing file
                os.rename(match_file, old_match_file)
            except FileExistsError:
                print(f"Could not rename {match_file} to {old_match_file} as it still exists")
            except Exception as e:
                print(f"Error renaming {match_file}: {str(e)}")
    
    def preprocess_name(name):
        if pd.isna(name):
            return ""
        name = str(name).lower()
        # Synonym mapping
        synonyms = {
            'gubernatorial': 'governorship',
            'presidential': 'president',
            'senatorial': 'senate'
        }
        # State abbreviation mapping for all 50 states and DC
        state_abbrevs = [
            'al-', 'ak-', 'az-', 'ar-', 'ca-', 'co-', 'ct-', 'de-', 'fl-', 'ga-',
            'hi-', 'id-', 'il-', 'in-', 'ia-', 'ks-', 'ky-', 'la-', 'me-', 'md-',
            'ma-', 'mi-', 'mn-', 'ms-', 'mo-', 'mt-', 'ne-', 'nv-', 'nh-', 'nj-',
            'nm-', 'ny-', 'nc-', 'nd-', 'oh-', 'ok-', 'or-', 'pa-', 'ri-', 'sc-',
            'sd-', 'tn-', 'tx-', 'ut-', 'vt-', 'va-', 'wa-', 'wv-', 'wi-', 'wy-',
            'dc-'
        ]
        state_names = [
            'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado', 'connecticut', 'delaware', 'florida', 'georgia',
            'hawaii', 'idaho', 'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana', 'maine', 'maryland',
            'massachusetts', 'michigan', 'minnesota', 'mississippi', 'missouri', 'montana', 'nebraska', 'nevada', 'new hampshire', 'new jersey',
            'new mexico', 'new york', 'north carolina', 'north dakota', 'ohio', 'oklahoma', 'oregon', 'pennsylvania', 'rhode island', 'south carolina',
            'south dakota', 'tennessee', 'texas', 'utah', 'vermont', 'virginia', 'washington', 'west virginia', 'wisconsin', 'wyoming',
            'district of columbia'
        ]
        state_dict = {abbrev[:-1]: name for abbrev, name in zip(state_abbrevs, state_names)}
        for abbrev, full in state_dict.items():
            name = re.sub(r'\b' + re.escape(abbrev) + r'-(\d+)\b',
                          lambda m: f"{full} {m.group(1)} district", name)
        name = re.sub(r'\b(\d+)(st|nd|rd|th)\b', r'\1', name)
        for old, new in synonyms.items():
            name = re.sub(r'\b' + old + r'\b', new, name)
        name = re.sub(r'[^\w\s]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        name = re.sub(r'district', 'district ', name)
        return name
    
    def extract_timeframe(text, end_date):
        timeframes = []
        text = str(text).lower()
        # Month mapping for normalization
        month_map = {
            'jan': 'january', 'feb': 'february', 'mar': 'march', 'apr': 'april',
            'may': 'may', 'jun': 'june', 'jul': 'july', 'aug': 'august',
            'sep': 'september', 'oct': 'october', 'nov': 'november', 'dec': 'december'
        }
        # Extract years
        years = re.findall(r'\b(20\d{2})\b', text)
        timeframes.extend(years)
        specific_dates = re.findall(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}\s*,\s*(20\d{2})', text)
        for month, year in specific_dates:
            timeframes.append(f"{month_map[month.lower()]} {year}")
        standalone_months = re.findall(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', text)
        for month in standalone_months:
            timeframes.append(month_map[month.lower()])
        before_years = re.findall(r'\bbefore\s+(20\d{2})\b', text)
        for year in before_years:
            timeframes.append(year)
            try:
                prev_year = str(int(year) - 1)
                timeframes.append(prev_year)
            except ValueError:
                pass
        relative_terms = ["this month", "this year", "before august"] if "before august" in text else []
        timeframes.extend(relative_terms)
        # Extract year from end_date
        if isinstance(end_date, str) and not pd.isna(end_date):
            try:
                end_date_year = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%SZ").year
                timeframes.append(str(end_date_year))
            except (ValueError, TypeError):
                pass
        return list(set(timeframes))
    
    def extract_entities(text):
        text = str(text)
        # State abbreviation mapping
        state_abbrevs = {
            'al': 'alabama', 'ak': 'alaska', 'az': 'arizona', 'ar': 'arkansas', 'ca': 'california',
            'co': 'colorado', 'ct': 'connecticut', 'de': 'delaware', 'fl': 'florida', 'ga': 'georgia',
            'hi': 'hawaii', 'id': 'idaho', 'il': 'illinois', 'in': 'indiana', 'ia': 'iowa',
            'ks': 'kansas', 'ky': 'kentucky', 'la': 'louisiana', 'me': 'maine', 'md': 'maryland',
            'ma': 'massachusetts', 'mi': 'michigan', 'mn': 'minnesota', 'ms': 'mississippi', 'mo': 'missouri',
            'mt': 'montana', 'ne': 'nebraska', 'nv': 'nevada', 'nh': 'new hampshire', 'nj': 'new jersey',
            'nm': 'new mexico', 'ny': 'new york', 'nc': 'north carolina', 'nd': 'north dakota', 'oh': 'ohio',
            'ok': 'oklahoma', 'or': 'oregon', 'pa': 'pennsylvania', 'ri': 'rhode island', 'sc': 'south carolina',
            'sd': 'south dakota', 'tn': 'tennessee', 'tx': 'texas', 'ut': 'utah', 'vt': 'vermont',
            'va': 'virginia', 'wa': 'washington', 'wv': 'west virginia', 'wi': 'wisconsin', 'wy': 'wyoming',
            'dc': 'district of columbia'
        }
        pattern = r'\b(?:[A-Z][a-z]*\s+)+[A-Z][a-z]*\b'
        entities = re.findall(pattern, text)
        # Normalize state-district pairs
        district_pattern = r'\b([A-Z][a-z\s]*(?:\'s)? \d+(?:st|nd|rd|th)? District\b|\b[A-Z]{2}-\d+\b)'
        districts = re.findall(district_pattern, text)
        normalized_districts = []
        for district in districts:
            for abbrev, full in state_abbrevs.items():
                if abbrev.upper() in district:
                    district = district.replace(abbrev.upper(), full.title())
                elif district.startswith(full.title()):
                    district = district.replace(full.title(), full.title())
            district = re.sub(r'\b(\d+)(st|nd|rd|th)\b', r'\1', district)
            normalized_districts.append(district)
        entities.extend(normalized_districts)
        # Include specific single-word entities and all state names
        known_entities = {
            'US', 'Trump', 'Ripple', 'Truth Social', 'The Witcher', 'Rotten Tomatoes',
            'Republican', 'Democratic', 'Run', 'Run for', 'Win', 'Winner', 'Independent', 'Governor',
            'Gubernatorial', 'Governorship', 'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
            'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois',
            'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts',
            'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
            'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota',
            'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
            'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin',
            'Wyoming', 'District of Columbia', 'Germany', 'France', 'Japan', 'Canada', 'Italy', 'Palestine',
            'Netherlands', 'Adrienne Adams', 'MVP', 'Breanna Stewart', 'Philadelphia', 'NL East', 'NL', 'Tariff',
            'Court'
        }
        entities.extend([word for word in text.split() if word in known_entities])
        numerical_patterns = [
            r'\b\d+\b(?!.*\b20\d{2}\b)',  
            r'\b\d+-\d+\b', 
            r'\b\d+k\b',
            r'\babove \d+\b', 
            r'\bbelow \d+\b' 
        ]
        for pattern in numerical_patterns:
            matches = re.findall(pattern, text.lower())
            entities.extend(matches)
        entities = [e.replace('1k', '1000') if '1k' in e else e for e in entities]
        stop_words = {
            'Will', 'Who', 'Before', 'The', 'This', 'By', 'Be', 'On', 'At', 'Season',
            'Cards', 'Day', 'How', 'Gold', 'Year', 'Above', 'Below', 'Score', 'Party'
        }
        entities = [e.strip() for e in entities if e.strip() not in stop_words and len(e.strip()) > 2 and ':' not in e]
        for entity in known_entities:
            if entity.lower() in text.lower():
                entities.append(entity)
        return list(set(entities))
    
    def extract_district(text):
        text = str(text).lower()
        match = re.search(r'(?:district|dist\.?)\s*(\d+)|(\d+)(?:st|nd|rd|th)?\s*district|\b[a-z]{2}-(\d+)\b', text)
        if match:
            return match.group(1) or match.group(2) or match.group(3)
        return None
    
    def same_district(name1, name2):
        d1 = extract_district(name1)
        d2 = extract_district(name2)
        return d1 == d2  # True if same district number or both None
    
    def are_timeframes_compatible(tf1, tf2):
        if not tf1 or not tf2:
            return True
        for t1 in tf1:
            for t2 in tf2:
                if t1 == t2:
                    return True
                if t1.startswith("20") and t2.split()[-1] == t1:
                    return True
                if t2.startswith("20") and t1.split()[-1] == t2:
                    return True
                if (t1 == "this year" or t1 == "before august") and (t2.startswith("20") or t2.split()[-1].startswith("20")):
                    return True
                if (t2 == "this year" or t2 == "before august") and (t1.startswith("20") or t1.split()[-1].startswith("20")):
                    return True
        return False
    
    # Check entity overlap
    def entities_overlap(entities1, entities2):
        if not entities1 or not entities2:
            return True
        # Define critical action words that must match or be absent
        action_words = {'run', 'win', 'nominate', 'elect', 'appointed', 'confirmed'}
        # Define party entities that must match or be absent
        party_words = {'democratic', 'republican', 'independent'}
        # Define district entities
        district_words = {e for e in entities1 if 'District' in e or '-' in e} | {e for e in entities2 if 'District' in e or '-' in e}
        entities1_actions = [e.lower() for e in entities1 if e.lower() in action_words]
        entities2_actions = [e.lower() for e in entities2 if e.lower() in action_words]
        entities1_parties = [e.lower() for e in entities1 if e.lower() in party_words]
        entities2_parties = [e.lower() for e in entities2 if e.lower() in party_words]
        entities1_districts = [e for e in entities1 if 'District' in e or '-' in e]
        entities2_districts = [e for e in entities2 if 'District' in e or '-' in e]
        # If either has an action word, they must match exactly
        if entities1_actions or entities2_actions:
            if set(entities1_actions) != set(entities2_actions):
                return False
        # If either has a party, they must match exactly
        if entities1_parties or entities2_parties:
            if set(entities1_parties) != set(entities2_parties):
                return False
        # If either has a district, they must match exactly
        if entities1_districts or entities2_districts:
            if set(entities1_districts) != set(entities2_districts):
                return False
        # Require matching state if both have one
        us_states = {
            'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado', 'connecticut',
            'delaware', 'florida', 'georgia', 'hawaii', 'idaho', 'illinois', 'indiana', 'iowa',
            'kansas', 'kentucky', 'louisiana', 'maine', 'maryland', 'massachusetts', 'michigan',
            'minnesota', 'mississippi', 'missouri', 'montana', 'nebraska', 'nevada',
            'new hampshire', 'new jersey', 'new mexico', 'new york', 'north carolina',
            'north dakota', 'ohio', 'oklahoma', 'oregon', 'pennsylvania', 'rhode island',
            'south carolina', 'south dakota', 'tennessee', 'texas', 'utah', 'vermont',
            'virginia', 'washington', 'west virginia', 'wisconsin', 'wyoming', 'district of columbia'
        }
        states1 = {e.lower() for e in entities1 if e.lower() in us_states}
        states2 = {e.lower() for e in entities2 if e.lower() in us_states}
        if states1 and states2:
            if states1 != states2:
                return False
        # Require matching country if both have one
        countries = {
            'germany', 'france', 'japan', 'canada', 'italy', 'israel', 'uk', 'spain', 'us', 'u.s.', 'palestine', 'netherlands'
        }
        countries1 = {e.lower() for e in entities1 if e.lower() in countries}
        countries2 = {e.lower() for e in entities2 if e.lower() in countries}
        if countries1 and countries2:
            if countries1 != countries2:
                return False
        # Require at least one shared non-action, non-party, non-district entity
        non_action_party_district_entities1 = [e for e in entities1 if e.lower() not in action_words and e.lower() not in party_words and e not in district_words]
        non_action_party_district_entities2 = [e for e in entities2 if e.lower() not in action_words and e.lower() not in party_words and e not in district_words]
        if not non_action_party_district_entities1 or not non_action_party_district_entities2:
            return True
        return any(e1.lower() in [e2.lower() for e2 in non_action_party_district_entities2] for e1 in non_action_party_district_entities1)
    
    kalshi_data = pd.read_csv(r"Kalshi.csv") # replace with actual file location
    polymarket_data = pd.read_csv(r"Polymarket.csv") # replace with actual file location
    
    kalshi_data.loc[kalshi_data['full_name'].str.contains("Above", na=False), 'full_name'] = \
        kalshi_data['full_name'].str.replace(r'\bAbove', 'Greater than', regex=True)
    polymarket_data.loc[polymarket_data['title'].str.contains("WNBA MVP", na=False), 'title'] = \
        polymarket_data['title'].str.replace(r'\bWNBA MVP', 'MVP', regex=True)
    polymarket_data.loc[polymarket_data['title'].str.contains("National League", na=False), 'title'] = \
        polymarket_data['title'].str.replace(r'\bNational League', 'NL', regex=True)
    
    # Pivot data for prices
    kalshi_pivot = kalshi_data.pivot_table(
        index=["full_name", "market_id", "end_date"],
        columns="side",
        values="ask",
        aggfunc="first"
    ).reset_index()
    kalshi_pivot.columns.name = None
    kalshi_pivot.rename(columns={"YES": "kalshi_yes", "NO": "kalshi_no"}, inplace=True)
    
    polymarket_pivot = polymarket_data.pivot_table(
        index=["title", "event_id", "end_date"],
        columns="outcome",
        values="price",
        aggfunc="first"
    ).reset_index()
    polymarket_pivot.columns.name = None
    polymarket_pivot.rename(columns={"Yes": "polymarket_yes", "No": "polymarket_no"}, inplace=True)
    
    # Preprocess event names and extract timeframes
    kalshi_pivot['processed_name'] = kalshi_pivot['full_name'].apply(preprocess_name)
    polymarket_pivot['processed_name'] = polymarket_pivot['title'].apply(preprocess_name)
    kalshi_pivot['timeframes'] = kalshi_pivot.apply(lambda x: extract_timeframe(x['full_name'], x['end_date']), axis=1)
    polymarket_pivot['timeframes'] = polymarket_pivot.apply(lambda x: extract_timeframe(x['title'], x['end_date']), axis=1)
    kalshi_pivot['entities'] = kalshi_pivot['full_name'].apply(extract_entities)
    polymarket_pivot['entities'] = polymarket_pivot['title'].apply(extract_entities)
    
    # Filter by year
    kalshi_years = kalshi_pivot['timeframes'].apply(lambda x: set(x).intersection({'2025', 'this year', 'before august'}))
    polymarket_years = polymarket_pivot['timeframes'].apply(lambda x: set(x).intersection({'2025', 'this year', 'before august'}))
    kalshi_subset = kalshi_pivot[kalshi_years.apply(len) > 0]
    polymarket_subset = polymarket_pivot[polymarket_years.apply(len) > 0]
    
    vectorizer = TfidfVectorizer(stop_words='english')
    kalshi_vectors = vectorizer.fit_transform(kalshi_subset['processed_name'])
    polymarket_vectors = vectorizer.transform(polymarket_subset['processed_name'])
    cosine_sim = cosine_similarity(kalshi_vectors, polymarket_vectors)
    
    # Filter pairs with high similarity
    cosine_threshold = 0.8
    similar_pairs = np.where(cosine_sim >= cosine_threshold)
    matches = []
    
    # Perform precise matching with rapidfuzz
    for idx in tqdm(range(len(similar_pairs[0])), desc="Matching similar pairs"):
        i, j = similar_pairs[0][idx], similar_pairs[1][idx]
        name1 = kalshi_subset['processed_name'].iloc[i]
        name2 = polymarket_subset['processed_name'].iloc[j]
        tf1 = kalshi_subset['timeframes'].iloc[i]
        tf2 = polymarket_subset['timeframes'].iloc[j]
        ent1 = kalshi_subset['entities'].iloc[i]
        ent2 = polymarket_subset['entities'].iloc[j]
        kalshi_event = kalshi_subset['full_name'].iloc[i]
        polymarket_event = polymarket_subset['title'].iloc[j]
        if name1 and name2 and same_district(name1, name2):
            similarity = fuzz.token_sort_ratio(name1, name2)
            timeframe_ok = are_timeframes_compatible(tf1, tf2)
            entity_ok = entities_overlap(ent1, ent2)
            threshold = 81 if any(x in kalshi_event.lower() or x in polymarket_event.lower() for x in ["how many", "less than", "between", ">", "<"]) else 70
            if similarity >= threshold and timeframe_ok and entity_ok:
                kalshi_yes = kalshi_subset['kalshi_yes'].iloc[i]
                kalshi_no = kalshi_subset['kalshi_no'].iloc[i]
                polymarket_yes = polymarket_subset['polymarket_yes'].iloc[j] * 100 if pd.notna(polymarket_subset['polymarket_yes'].iloc[j]) else None
                polymarket_no = polymarket_subset['polymarket_no'].iloc[j] * 100 if pd.notna(polymarket_subset['polymarket_no'].iloc[j]) else None
                # Calculate arbitrage
                calc_kalshi_yes = kalshi_yes if pd.notna(kalshi_yes) else (100 - kalshi_no if pd.notna(kalshi_no) else 0)
                calc_kalshi_no = kalshi_no if pd.notna(kalshi_no) else (100 - kalshi_yes if pd.notna(kalshi_yes) else 0)
                calc_polymarket_yes = polymarket_yes if pd.notna(polymarket_yes) else (100 - polymarket_no if pd.notna(polymarket_no) else 0)
                calc_polymarket_no = polymarket_no if pd.notna(polymarket_no) else (100 - polymarket_yes if pd.notna(polymarket_yes) else 0)
                matches.append({
                    'Kalshi_Event': kalshi_event,
                    'Polymarket_Event': polymarket_event,
                    'Similarity_Score': similarity,
                    'Timeframes_Match': timeframe_ok,
                    'Entities_Match': entity_ok,
                    'Kalshi_Timeframes': tf1,
                    'Polymarket_Timeframes': tf2,
                    'Kalshi_Entities': ent1,
                    'Polymarket_Entities': ent2,
                    'kalshi_yes': kalshi_yes,
                    'kalshi_no': kalshi_no,
                    'polymarket_yes': polymarket_yes,
                    'polymarket_no': polymarket_no,
                    'arb1': 100 - calc_polymarket_yes - calc_kalshi_no,
                    'arb2': 100 - calc_polymarket_no - calc_kalshi_yes
                })
    
    matches_df = pd.DataFrame(matches)
    if not matches_df.empty:
        matches_df = matches_df.sort_values(by='Similarity_Score', ascending=False)
        matches_df = matches_df.drop_duplicates(subset=['Kalshi_Event', 'Polymarket_Event'], keep='first')
    
    # Filter for significant arbitrage and non-zero prices
    matches_df = matches_df[
        ((matches_df["arb1"] >= 10) | (matches_df["arb2"] >= 10)) &
        (matches_df["kalshi_yes"].notna() & (matches_df["kalshi_yes"] != 0)) &
        (matches_df["kalshi_no"].notna() & (matches_df["kalshi_no"] != 0)) &
        (matches_df["polymarket_yes"].notna() & (matches_df["polymarket_yes"] != 0)) &
        (matches_df["polymarket_no"].notna() & (matches_df["polymarket_no"] != 0))
    ]
    
    # Save to CSV
    output_path = r"Kalshi_Polymarket_Matches.csv" # replace with desired file location
    matches_df.to_csv(output_path, index=False)
    
    print(f"Saved {len(matches_df)} matched markets")
    def preprocess_name(name):
        if pd.isna(name):
            return ""
        name = str(name).lower()
        name = re.sub(r'[^\w\s]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        return name
    
    def extract_timeframe(text, end_date):
        timeframes = []
        text = str(text).lower()
        years = re.findall(r'\b(20\d{2})\b', text)
        specific_dates = re.findall(r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}\s*,\s*(20\d{2})', text)
        relative_terms = ["this month", "this year", "before august"] if "before august" in text else []
        timeframes.extend(years)
        timeframes.extend([date[0] for date in specific_dates])
        timeframes.extend(relative_terms)
        if isinstance(end_date, str) and not pd.isna(end_date):
            try:
                end_date_year = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%SZ").year
                timeframes.append(str(end_date_year))
            except (ValueError, TypeError):
                pass
        return list(set(timeframes))
    
    def extract_entities(text):
        text = str(text)
        pattern = r'\b(?:[A-Z][a-z]*\s+)+[A-Z][a-z]*\b'
        entities = re.findall(pattern, text)
        # Include specific single-word (easily expandable to improve accuracy)
        known_entities = {'US', 'Trump', 'Ripple', 'Truth Social', 'The Witcher', 'Rotten Tomatoes'}
        entities.extend([word for word in text.split() if word in known_entities])
        # Extract numerical outcomes (exclude years)
        numerical_patterns = [
            r'\b\d+\b(?!.*\b20\d{2}\b)',
            r'\b\d+-\d+\b', 
            r'\b\d+k\b', 
            r'\babove \d+\b',  
            r'\bbelow \d+\b' 
        ]
        for pattern in numerical_patterns:
            matches = re.findall(pattern, text.lower())
            entities.extend(matches)
        entities = [e.replace('1k', '1000') if '1k' in e else e for e in entities]
        stop_words = {'Will', 'Who', 'Before', 'The', 'This', 'By', 'Be', 'In', 'On', 'At', 'Season', 'Cards', 'Day', 'How', 'Gold', 'Year', 'Above', 'Below', 'Score'}
        entities = [e.strip() for e in entities if e.strip() not in stop_words and len(e.strip()) > 2 and ':' not in e]
        # Add known entities
        for entity in known_entities:
            if entity.lower() in text.lower():
                entities.append(entity)
        return list(set(entities))
    
    # Check timeframe compatibility
    def are_timeframes_compatible(tf1, tf2):
        if not tf1 or not tf2:
            return True
        for t1 in tf1:
            for t2 in tf2:
                if t1 == t2:
                    return True
                if (t1 == "this year" or t1 == "before august") and t2 == "2025":
                    return True
                if (t2 == "this year" or t2 == "before august") and t1 == "2025":
                    return True
                if t1.startswith("20") and t2.startswith("20") and t1 == t2:
                    return True
        return False
    
    # Check entity overlap
    def entities_overlap(entities1, entities2):
        if not entities1 or not entities2:
            return True
        return any(e1.lower() in [e2.lower() for e2 in entities2] for e1 in entities1)
    
    predictit_data = pd.read_csv(r"PredictIt.csv") # replace with actual file location
    polymarket_data = pd.read_csv(r"Polymarket.csv") # replace with actual file location
    predictit_data["price"] = predictit_data["price"] * 100
    
    # Pivot data for prices
    polymarket_pivot = polymarket_data.pivot_table(
        index=["title", "event_id", "end_date"],
        columns="outcome",
        values="price",
        aggfunc="first"
    ).reset_index()
    polymarket_pivot.columns.name = None
    polymarket_pivot.rename(columns={"Yes": "polymarket_yes", "No": "polymarket_no"}, inplace=True)
    
    predictit_pivot = predictit_data.pivot_table(
        index=["market_name", "market_id", "end_date"],
        columns="side",
        values="price",
        aggfunc="first"
    ).reset_index()
    predictit_pivot.columns.name = None
    predictit_pivot.rename(columns={"YES": "predictit_yes", "NO": "predictit_no"}, inplace=True)
    
    # Preprocess event names and extract timeframes
    polymarket_pivot['processed_name'] = polymarket_pivot['title'].apply(preprocess_name)
    predictit_pivot['processed_name'] = predictit_pivot['market_name'].apply(preprocess_name)
    polymarket_pivot['timeframes'] = polymarket_pivot.apply(lambda x: extract_timeframe(x['title'], x['end_date']), axis=1)
    predictit_pivot['timeframes'] = predictit_pivot.apply(lambda x: extract_timeframe(x['market_name'], x['end_date']), axis=1)
    polymarket_pivot['entities'] = polymarket_pivot['title'].apply(extract_entities)
    predictit_pivot['entities'] = predictit_pivot['market_name'].apply(extract_entities)
    
    # Filter by year
    predictit_years = predictit_pivot['timeframes'].apply(lambda x: set(x).intersection({'2025', 'this year', 'before august'}))
    polymarket_years = polymarket_pivot['timeframes'].apply(lambda x: set(x).intersection({'2025', 'this year', 'before august'}))
    predictit_subset = predictit_pivot[predictit_years.apply(len) > 0]
    polymarket_subset = polymarket_pivot[polymarket_years.apply(len) > 0]
    
    vectorizer = TfidfVectorizer(stop_words='english')
    predictit_vectors = vectorizer.fit_transform(predictit_subset['processed_name'])
    polymarket_vectors = vectorizer.transform(polymarket_subset['processed_name'])
    cosine_sim = cosine_similarity(predictit_vectors, polymarket_vectors)
    
    # Filter pairs with high similarity
    cosine_threshold = 0.8
    similar_pairs = np.where(cosine_sim >= cosine_threshold)
    matches = []
    
    # Perform precise matching with rapidfuzz
    for idx in tqdm(range(len(similar_pairs[0])), desc="Matching similar pairs"):
        i, j = similar_pairs[0][idx], similar_pairs[1][idx]
        name1 = predictit_subset['processed_name'].iloc[i]
        name2 = polymarket_subset['processed_name'].iloc[j]
        tf1 = predictit_subset['timeframes'].iloc[i]
        tf2 = polymarket_subset['timeframes'].iloc[j]
        ent1 = predictit_subset['entities'].iloc[i]
        ent2 = polymarket_subset['entities'].iloc[j]
        predictit_event = predictit_subset['market_name'].iloc[i]
        polymarket_event = polymarket_subset['title'].iloc[j]
        if name1 and name2:
            similarity = fuzz.token_sort_ratio(name1, name2)
            timeframe_ok = are_timeframes_compatible(tf1, tf2)
            entity_ok = entities_overlap(ent1, ent2)
            threshold = 75 if "how many" in predictit_event.lower() or "how many" in polymarket_event.lower() or "less than" in predictit_event.lower() or "less than" in polymarket_event.lower() or "between" in predictit_event.lower() or "between" in polymarket_event.lower() or ">" in predictit_event.lower() or ">" in polymarket_event.lower() or "<" in predictit_event.lower() or "<" in polymarket_event.lower() else 75
            if similarity >= threshold and timeframe_ok and entity_ok:
                # Handle missing prices for arbitrage calculations
                predictit_yes = predictit_subset['predictit_yes'].iloc[i]
                predictit_no = predictit_subset['predictit_no'].iloc[i]
                polymarket_yes = polymarket_subset['polymarket_yes'].iloc[j] * 100 if pd.notna(polymarket_subset['polymarket_yes'].iloc[j]) else None
                polymarket_no = polymarket_subset['polymarket_no'].iloc[j] * 100 if pd.notna(polymarket_subset['polymarket_no'].iloc[j]) else None
                
                # Assume 100 - complementary price if missing
                calc_predictit_yes = predictit_yes if pd.notna(predictit_yes) else (100 - predictit_no if pd.notna(predictit_no) else 0)
                calc_predictit_no = predictit_no if pd.notna(predictit_no) else (100 - predictit_yes if pd.notna(predictit_yes) else 0)
                calc_polymarket_yes = polymarket_yes if pd.notna(polymarket_yes) else (100 - polymarket_no if pd.notna(polymarket_no) else 0)
                calc_polymarket_no = polymarket_no if pd.notna(polymarket_no) else (100 - polymarket_yes if pd.notna(polymarket_yes) else 0)
                
                matches.append({
                    'PredictIt_Event': predictit_event,
                    'Polymarket_Event': polymarket_event,
                    'Similarity_Score': similarity,
                    'Timeframes_Match': timeframe_ok,
                    'Entities_Match': entity_ok,
                    'PredictIt_Timeframes': tf1,
                    'Polymarket_Timeframes': tf2,
                    'PredictIt_Entities': ent1,
                    'Polymarket_Entities': ent2,
                    'predictit_yes': predictit_yes,
                    'predictit_no': predictit_no,
                    'polymarket_yes': polymarket_yes,
                    'polymarket_no': polymarket_no,
                    'arb1': 100 - calc_polymarket_yes - calc_predictit_no,
                    'arb2': 100 - calc_polymarket_no - calc_predictit_yes
                })
    
    matches_df = pd.DataFrame(matches)
    if not matches_df.empty:
        matches_df = matches_df.sort_values(by='Similarity_Score', ascending=False)
        matches_df = matches_df.drop_duplicates(subset=['PredictIt_Event', 'Polymarket_Event'], keep='first')
    
    # Filter for significant arbitrage and non-zero prices
    matches_df = matches_df[
        ((matches_df["arb1"] >= 10) | (matches_df["arb2"] >= 10)) &
        (matches_df["predictit_yes"].notna() & (matches_df["predictit_yes"] != 0)) &
        (matches_df["predictit_no"].notna() & (matches_df["predictit_no"] != 0)) &
        (matches_df["polymarket_yes"].notna() & (matches_df["polymarket_yes"] != 0)) &
        (matches_df["polymarket_no"].notna() & (matches_df["polymarket_no"] != 0))
    ]
    
    # Save to CSV
    output_path = r"PredictIt_Polymarket_Matches.csv" # replace with actual file location
    matches_df.to_csv(output_path, index=False)
    
    print(f"Saved {len(matches_df)} matched markets")
    
    import pandas as pd
    from rapidfuzz import fuzz
    import re
    from datetime import datetime
    from tqdm.notebook import tqdm
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # Preprocess event names
    def preprocess_name(name):
        if pd.isna(name):
            return ""
        name = str(name).lower()
        
        # Synonym mapping
        synonyms = {
            'gubernatorial': 'governorship',
            'presidential': 'president',
            'senatorial': 'senate'
        }
        # State abbreviation mapping for all 50 states and DC
        state_abbrevs = [
            'al-', 'ak-', 'az-', 'ar-', 'ca-', 'co-', 'ct-', 'de-', 'fl-', 'ga-',
            'hi-', 'id-', 'il-', 'in-', 'ia-', 'ks-', 'ky-', 'la-', 'me-', 'md-',
            'ma-', 'mi-', 'mn-', 'ms-', 'mo-', 'mt-', 'ne-', 'nv-', 'nh-', 'nj-',
            'nm-', 'ny-', 'nc-', 'nd-', 'oh-', 'ok-', 'or-', 'pa-', 'ri-', 'sc-',
            'sd-', 'tn-', 'tx-', 'ut-', 'vt-', 'va-', 'wa-', 'wv-', 'wi-', 'wy-',
            'dc-'
        ]
        state_names = [
            'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado', 'connecticut', 'delaware', 'florida', 'georgia',
            'hawaii', 'idaho', 'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana', 'maine', 'maryland',
            'massachusetts', 'michigan', 'minnesota', 'mississippi', 'missouri', 'montana', 'nebraska', 'nevada', 'new hampshire', 'new jersey',
            'new mexico', 'new york', 'north carolina', 'north dakota', 'ohio', 'oklahoma', 'oregon', 'pennsylvania', 'rhode island', 'south carolina',
            'south dakota', 'tennessee', 'texas', 'utah', 'vermont', 'virginia', 'washington', 'west virginia', 'wisconsin', 'wyoming',
            'district of columbia'
        ]
        state_dict = {abbrev[:-1]: name for abbrev, name in zip(state_abbrevs, state_names)}
        for abbrev, full in state_dict.items():
            name = re.sub(r'\b' + re.escape(abbrev) + r'-(\d+)\b',
                  lambda m: f"{full} {m.group(1)} district",
                  name)
        name = re.sub(r'\b(\d+)(st|nd|rd|th)\b', r'\1', name)
        for old, new in synonyms.items():
            name = re.sub(r'\b' + old + r'\b', new, name)
        name = re.sub(r'[^\w\s]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        name = re.sub(r'district', 'district ', name)
        return name
    
    # Extract timeframes
    def extract_timeframe(text, end_date):
        timeframes = []
        text = str(text).lower()
        years = re.findall(r'\b(20\d{2})\b', text)
        specific_dates = re.findall(r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}\s*,\s*(20\d{2})', text)
        relative_terms = ["this month", "this year", "before august"] if "before august" in text else []
        timeframes.extend(years)
        timeframes.extend([date[0] for date in specific_dates])
        timeframes.extend(relative_terms)
        if isinstance(end_date, str) and not pd.isna(end_date):
            try:
                end_date_year = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%SZ").year
                timeframes.append(str(end_date_year))
            except (ValueError, TypeError):
                pass
        return list(set(timeframes))
    
    def extract_entities(text):
        text = str(text)
        # State abbreviation mapping for district normalization
        pattern = r'\b(?:[A-Z][a-z]*\s+)+[A-Z][a-z]*\b'
        entities = re.findall(pattern, text)
        district_pattern = r'\b([A-Z][a-z\s]*(?:\'s)? \d+(?:st|nd|rd|th)? District\b|\b[A-Z]{2}-\d+\b)'
        districts = re.findall(district_pattern, text)
        normalized_districts = []
        # Include specific single-word entities and all state names (easily expandable to improve precision)
        known_entities = {'US', 'Trump', 'Truth Social', 'The Witcher', 'Rotten Tomatoes', 'Republican', 'Democratic', 'Run', 'Run for', 'Win', 'Winner', 'Independent', 'Governor', 'Gubernatorial', 'Governorship', 'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming', 'District of Columbia'}
        entities.extend([word for word in text.split() if word in known_entities])
        numerical_patterns = [
            r'\b\d+\b(?!.*\b20\d{2}\b)',
            r'\b\d+-\d+\b',
            r'\b\d+k\b', 
            r'\babove \d+\b', 
            r'\bbelow \d+\b'
        ]
        for pattern in numerical_patterns:
            matches = re.findall(pattern, text.lower())
            entities.extend(matches)
        # Normalize "1k" to "1000"
        entities = [e.replace('1k', '1000') if '1k' in e else e for e in entities]
        stop_words = {'Will', 'Who', 'Before', 'The', 'This', 'By', 'Be', 'On', 'At', 'Season', 'Cards', 'Day', 'How', 'Gold', 'Year', 'Above', 'Below', 'Score', 'Party'}
        entities = [e.strip() for e in entities if e.strip() not in stop_words and len(e.strip()) > 2 and ':' not in e]
        for entity in known_entities:
            if entity.lower() in text.lower():
                entities.append(entity)
        return list(set(entities))
    
    # Check timeframe compatibility
    def are_timeframes_compatible(tf1, tf2):
        if not tf1 or not tf2:
            return True
        for t1 in tf1:
            for t2 in tf2:
                if t1 == t2:
                    return True
                if (t1 == "this year" or t1 == "before august") and t2.startswith("20"):
                    return True
                if (t2 == "this year" or t2 == "before august") and t1.startswith("20"):
                    return True
                if t1.startswith("20") and t2.startswith("20") and t1 == t2:
                    return True
        return False
    
    def extract_district(text):
        text = text.lower()
        match = re.search(r'(?:district|dist\.?)\s*(\d+)|(\d+)(?:st|nd|rd|th)?\s*district', text)
        if match:
            return match.group(1) or match.group(2)
        return None
    
    def same_district(name1, name2):
        d1 = extract_district(name1)
        d2 = extract_district(name2)
        return d1 == d2  # True only if same district number or both None
    
    def entities_overlap(entities1, entities2):
        if not entities1 or not entities2:
            return True
        # Define critical action words that must match or be absent
        action_words = {'run', 'win', 'nominate', 'elect', 'appointed', 'confirmed'}
        # Define party entities that must match or be absent
        party_words = {'democratic', 'republican', 'independent'}
    
        
        # Require matching state if both have one
        us_states = {
            'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado', 'connecticut',
            'delaware', 'florida', 'georgia', 'hawaii', 'idaho', 'illinois', 'indiana', 'iowa',
            'kansas', 'kentucky', 'louisiana', 'maine', 'maryland', 'massachusetts', 'michigan',
            'minnesota', 'mississippi', 'missouri', 'montana', 'nebraska', 'nevada',
            'new hampshire', 'new jersey', 'new mexico', 'new york', 'north carolina',
            'north dakota', 'ohio', 'oklahoma', 'oregon', 'pennsylvania', 'rhode island',
            'south carolina', 'south dakota', 'tennessee', 'texas', 'utah', 'vermont',
            'virginia', 'washington', 'west virginia', 'wisconsin', 'wyoming', 'district of columbia'
        }
        
        states1 = {e.lower() for e in entities1 if e.lower() in us_states}
        states2 = {e.lower() for e in entities2 if e.lower() in us_states}
    
    
        if len(states1) > 0 and len(states2) > 0:
            for state1 in states1:
                for state2 in states2:
                    if state1 in state2:
                        return True
            for state2 in states2:
                for state1 in states1:
                    if state2 in state1:
                        return True
            return False
    
        
        district_words = {e for e in entities1 if 'District' in e or '-' in e} | {e for e in entities2 if 'District' in e or '-' in e}
        entities1_actions = [e.lower() for e in entities1 if e.lower() in action_words]
        entities2_actions = [e.lower() for e in entities2 if e.lower() in action_words]
        entities1_parties = [e.lower() for e in entities1 if e.lower() in party_words]
        entities2_parties = [e.lower() for e in entities2 if e.lower() in party_words]
        
        # If either has an action word, they must match exactly
        if entities1_actions or entities2_actions:
            if set(entities1_actions) != set(entities2_actions):
                return False
    
        # If either has a party, they must match exactly
        if entities1_parties or entities2_parties:
            if set(entities1_parties) != set(entities2_parties):
                return False
        
        # Require at least one shared non-action, non-party, non-district entity
        non_action_party_district_entities1 = [e for e in entities1 if e.lower() not in action_words and e.lower() not in party_words and e not in district_words]
        non_action_party_district_entities2 = [e for e in entities2 if e.lower() not in action_words and e.lower() not in party_words and e not in district_words]
        if not non_action_party_district_entities1 or not non_action_party_district_entities2:
            return True
        return any(e1.lower() in [e2.lower() for e2 in non_action_party_district_entities2] for e1 in non_action_party_district_entities1)
    
    kalshi_data = pd.read_csv(r"Kalshi.csv") # replace with actual file location
        
    kalshi_data.loc[kalshi_data['full_name'].str.contains("AZ-", na=False), 'full_name'] = \
        kalshi_data['full_name'].str.replace(r'\bAZ-', 'Arizona District', regex=True)
    kalshi_data.loc[kalshi_data['full_name'].str.contains("CA-", na=False), 'full_name'] = \
        kalshi_data['full_name'].str.replace(r'\bCA-', 'California District', regex=True)
    kalshi_data.loc[kalshi_data['full_name'].str.contains("CT-", na=False), 'full_name'] = \
        kalshi_data['full_name'].str.replace(r'\bCT-', 'Connecticut District', regex=True)
    kalshi_data.loc[kalshi_data['full_name'].str.contains("CO-", na=False), 'full_name'] = \
        kalshi_data['full_name'].str.replace(r'\bCO-', 'Colorado District', regex=True)
    kalshi_data.loc[kalshi_data['full_name'].str.contains("IL-", na=False), 'full_name'] = \
        kalshi_data['full_name'].str.replace(r'\bIL-', 'Illinois District', regex=True)
    kalshi_data.loc[kalshi_data['full_name'].str.contains("IN-", na=False), 'full_name'] = \
        kalshi_data['full_name'].str.replace(r'\bIN-', 'Indiana District', regex=True)
    kalshi_data.loc[kalshi_data['full_name'].str.contains("IA-", na=False), 'full_name'] = \
        kalshi_data['full_name'].str.replace(r'\bIA-', 'Iowa District', regex=True)
    kalshi_data.loc[kalshi_data['full_name'].str.contains("ME-", na=False), 'full_name'] = \
        kalshi_data['full_name'].str.replace(r'\bME-', 'Maine District', regex=True)
    kalshi_data.loc[kalshi_data['full_name'].str.contains("MI-", na=False), 'full_name'] = \
        kalshi_data['full_name'].str.replace(r'\bMI-', 'Michigan District', regex=True)
    kalshi_data.loc[kalshi_data['full_name'].str.contains("MN-", na=False), 'full_name'] = \
        kalshi_data['full_name'].str.replace(r'\bMN-', 'Minnesota District', regex=True)
    kalshi_data.loc[kalshi_data['full_name'].str.contains("MT-", na=False), 'full_name'] = \
        kalshi_data['full_name'].str.replace(r'\bMT-', 'Montana District', regex=True)
    kalshi_data.loc[kalshi_data['full_name'].str.contains("NC-", na=False), 'full_name'] = \
        kalshi_data['full_name'].str.replace(r'\bNC-', 'North Carolina District', regex=True)
    kalshi_data.loc[kalshi_data['full_name'].str.contains("NE-", na=False), 'full_name'] = \
        kalshi_data['full_name'].str.replace(r'\bNE-', 'Nebraska District', regex=True)
    kalshi_data.loc[kalshi_data['full_name'].str.contains("NH-", na=False), 'full_name'] = \
        kalshi_data['full_name'].str.replace(r'\bNH-', 'New Hampshire District', regex=True)
    kalshi_data.loc[kalshi_data['full_name'].str.contains("NJ-", na=False), 'full_name'] = \
        kalshi_data['full_name'].str.replace(r'\bNJ-', 'New Jersey District', regex=True)
    kalshi_data.loc[kalshi_data['full_name'].str.contains("NM-", na=False), 'full_name'] = \
        kalshi_data['full_name'].str.replace(r'\bNM-', 'New Mexico District', regex=True)
    kalshi_data.loc[kalshi_data['full_name'].str.contains("NV-", na=False), 'full_name'] = \
        kalshi_data['full_name'].str.replace(r'\bNV-', 'Nevada District', regex=True)
    kalshi_data.loc[kalshi_data['full_name'].str.contains("NY-", na=False), 'full_name'] = \
        kalshi_data['full_name'].str.replace(r'\bNY-', 'New York District', regex=True)
    kalshi_data.loc[kalshi_data['full_name'].str.contains("OH-", na=False), 'full_name'] = \
        kalshi_data['full_name'].str.replace(r'\bOH-', 'Ohio District', regex=True)
    kalshi_data.loc[kalshi_data['full_name'].str.contains("OR-", na=False), 'full_name'] = \
        kalshi_data['full_name'].str.replace(r'\bOR-', 'Oregon District', regex=True)
    kalshi_data.loc[kalshi_data['full_name'].str.contains("PA-", na=False), 'full_name'] = \
        kalshi_data['full_name'].str.replace(r'\bPA-', 'Pennsylvania District', regex=True)
    kalshi_data.loc[kalshi_data['full_name'].str.contains("TN-", na=False), 'full_name'] = \
        kalshi_data['full_name'].str.replace(r'\bTN-', 'Tennessee District', regex=True)
    kalshi_data.loc[kalshi_data['full_name'].str.contains("VA-", na=False), 'full_name'] = \
        kalshi_data['full_name'].str.replace(r'\bVA-', 'Virginia District', regex=True)
    kalshi_data.loc[kalshi_data['full_name'].str.contains("WA-", na=False), 'full_name'] = \
        kalshi_data['full_name'].str.replace(r'\bWA-', 'Washington District', regex=True)
    kalshi_data.loc[kalshi_data['full_name'].str.contains("WI-", na=False), 'full_name'] = \
        kalshi_data['full_name'].str.replace(r'\bWI-', 'Wisconsin District', regex=True)

    predictit_data = pd.read_csv(r"C:\Users\johnt\Downloads\Prediction Markets\PredictIt.csv")
    predictit_data['price'] = predictit_data['price'] * 100
    
    # Pivot data for prices
    kalshi_pivot = kalshi_data.pivot_table(
        index=["full_name", "market_id", "end_date"],
        columns="side",
        values="ask",
        aggfunc="first"
    ).reset_index()
    kalshi_pivot.columns.name = None
    kalshi_pivot.rename(columns={"YES": "kalshi_yes", "NO": "kalshi_no"}, inplace=True)
    
    predictit_pivot = predictit_data.pivot_table(
        index=["market_name", "market_id", "end_date"],
        columns="side",
        values="price",
        aggfunc="first"
    ).reset_index()
    predictit_pivot.columns.name = None
    predictit_pivot.rename(columns={"YES": "predictit_yes", "NO": "predictit_no"}, inplace=True)
    
    # Preprocess event names and extract timeframes
    kalshi_pivot['processed_name'] = kalshi_pivot['full_name'].apply(preprocess_name)
    predictit_pivot['processed_name'] = predictit_pivot['market_name'].apply(preprocess_name)
    kalshi_pivot['timeframes'] = kalshi_pivot.apply(lambda x: extract_timeframe(x['full_name'], x['end_date']), axis=1)
    predictit_pivot['timeframes'] = predictit_pivot.apply(lambda x: extract_timeframe(x['market_name'], x['end_date']), axis=1)
    kalshi_pivot['entities'] = kalshi_pivot['full_name'].apply(extract_entities)
    predictit_pivot['entities'] = predictit_pivot['market_name'].apply(extract_entities)
    kalshi_subset = kalshi_pivot
    predictit_subset = predictit_pivot
    
    vectorizer = TfidfVectorizer(stop_words='english')
    kalshi_vectors = vectorizer.fit_transform(kalshi_subset['processed_name'])
    predictit_vectors = vectorizer.transform(predictit_subset['processed_name'])
    cosine_sim = cosine_similarity(kalshi_vectors, predictit_vectors)
    
    # Filter pairs with high similarity
    cosine_threshold = 0.72
    similar_pairs = np.where(cosine_sim >= cosine_threshold)
    matches = []
    
    # Higher precision matching
    for idx in tqdm(range(len(similar_pairs[0])), desc="Matching similar pairs"):
        i, j = similar_pairs[0][idx], similar_pairs[1][idx]
        name1 = kalshi_subset['processed_name'].iloc[i]
        name2 = predictit_subset['processed_name'].iloc[j]
        tf1 = kalshi_subset['timeframes'].iloc[i]
        tf2 = predictit_subset['timeframes'].iloc[j]
        ent1 = kalshi_subset['entities'].iloc[i]
        ent2 = predictit_subset['entities'].iloc[j]
        kalshi_event = kalshi_subset['full_name'].iloc[i]
        predictit_event = predictit_subset['market_name'].iloc[j]
        if name1 and name2:
            similarity = fuzz.token_sort_ratio(name1, name2)
            timeframe_ok = are_timeframes_compatible(tf1, tf2)
            entity_ok = entities_overlap(ent1, ent2)
            kalshi_event = re.sub(r'District', 'District ', kalshi_event)
            threshold = 80 if "how many" in kalshi_event.lower() or "how many" in predictit_event.lower() or "less than" in kalshi_event.lower() or "less than" in predictit_event.lower() or "between" in kalshi_event.lower() or "between" in predictit_event.lower() or ">" in kalshi_event.lower() or ">" in predictit_event.lower() or "<" in kalshi_event.lower() or "<" in predictit_event.lower() else 80
            if similarity >= threshold and timeframe_ok and entity_ok and same_district(name1, name2):
                # Handle missing prices for arbitrage calculations
                kalshi_yes = kalshi_subset['kalshi_yes'].iloc[i]
                kalshi_no = kalshi_subset['kalshi_no'].iloc[i]
                predictit_yes = predictit_subset['predictit_yes'].iloc[j]
                predictit_no = predictit_subset['predictit_no'].iloc[j]
                
                # Assume 100 - complementary price if missing
                calc_kalshi_yes = kalshi_yes if pd.notna(kalshi_yes) else (100 - kalshi_no if pd.notna(kalshi_no) else 0)
                calc_kalshi_no = kalshi_no if pd.notna(kalshi_no) else (100 - kalshi_yes if pd.notna(kalshi_yes) else 0)
                calc_predictit_yes = predictit_yes if pd.notna(predictit_yes) else (100 - predictit_no if pd.notna(predictit_no) else 0)
                calc_predictit_no = predictit_no if pd.notna(predictit_no) else (100 - predictit_yes if pd.notna(predictit_yes) else 0)
                
                matches.append({
                    'Kalshi_Event': kalshi_event,
                    'PredictIt_Event': predictit_event,
                    'Similarity_Score': similarity,
                    'Timeframes_Match': timeframe_ok,
                    'Entities_Match': entity_ok,
                    'Kalshi_Timeframes': tf1,
                    'PredictIt_Timeframes': tf2,
                    'Kalshi_Entities': ent1,
                    'PredictIt_Entities': ent2,
                    'kalshi_yes': kalshi_yes,
                    'kalshi_no': kalshi_no,
                    'predictit_yes': predictit_yes,
                    'predictit_no': predictit_no,
                    'arb1': 100 - calc_predictit_yes - calc_kalshi_no,
                    'arb2': 100 - calc_predictit_no - calc_kalshi_yes
                })
    
    matches_df = pd.DataFrame(matches)
    if not matches_df.empty:
        matches_df = matches_df.sort_values(by='Similarity_Score', ascending=False)
        matches_df = matches_df.drop_duplicates(subset=['Kalshi_Event', 'PredictIt_Event'], keep='first')
    
    
    # Filter for significant arbitrage and non-zero prices
    matches_df = matches_df[
        ((matches_df["arb1"] >= 5) | (matches_df["arb2"] >= 5)) &
        (matches_df["kalshi_yes"].notna() & (matches_df["kalshi_yes"] != 0)) &
        (matches_df["kalshi_no"].notna() & (matches_df["kalshi_no"] != 0)) &
        (matches_df["predictit_yes"].notna() & (matches_df["predictit_yes"] != 0)) &
        (matches_df["predictit_no"].notna() & (matches_df["predictit_no"] != 0))
    ]
    
    # Filter out mismatched "west virginia" presence (common occurrence)
    matches_df = matches_df[
        matches_df.apply(
            lambda row: ("west virginia" in row["Kalshi_Event"].lower()) == ("west virginia" in row["PredictIt_Event"].lower()),
            axis=1
        )
    ]
    
    
    output_path = r"Kalshi_PredictIt_Matches.csv" # replace with desired file location
    matches_df.to_csv(output_path, index=False)
    print(f"Saved {len(matches_df)} matched markets")
        
    POST_WEBHOOK_URL = "https://discord.com/api/webhooks/"  # Replace with 'post' channel webhook
    SEND_WEBHOOK_URL = "https://discord.com/api/webhooks/"  # Replace with 'send' channel webhook
    
    # Webhook helper function
    def send_webhook_message(webhook_url, content, username=None, avatar_url=None):
        # Sends a message to a Discord channel using a webhook.
        payload = {
            "content": content,
            "username": username,
            "avatar_url": avatar_url
        }
        try:
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Failed to send message: {e}")
    
    def post(msg, fileName):
        # Sends a low-priority notification to the designated 'post' channel using a webhook.
        if "Kalshi_PredictIt_Matches_New_Or_Changed" in fileName:
            POST_WEBHOOK_URL = "https://discord.com/api/webhooks/" # replace with actual post webhook url
            path = pd.read_csv(r"Kalshi_PredictIt_Matches_New_Or_Changed.csv") # replace with desired path
            for i in range(len(path)):
                if path.shape[1] > 15:
                    row = path.iloc[i, [0, 1, 13, 14, 15, 16]]
                else:
                    row = path.iloc[i, [0, 1, 13, 14]]
                if pd.notna(row[3]) and isinstance(row[3], (int, float)):
                    pass
                else:
                    row[2] = row[4]
                    row[3] = row[5]
                msg = (
                    f"Kalshi Event: {row[0]}\n"
                    f"PredictIt Event: {row[1]}\n"
                    f"Arb1: {row[2]:.2f}\n"
                    f"Arb2: {row[3]:.2f}"
                )
                title = ""
                content = f"{title}\n{msg}"
                send_webhook_message(POST_WEBHOOK_URL, content, username="Projections Bot")
                
        elif "PredictIt_Polymarket_Matches_New_Or_Changed" in fileName:
            POST_WEBHOOK_URL = 'https://discord.com/api/webhooks/' # replace with actual post webhook url
            path = pd.read_csv(r"PredictIt_Polymarket_Matches_New_Or_Changed.csv") # replace with desired path
            for i in range(len(path)):
                if path.shape[1] > 15:
                    row = path.iloc[i, [0, 1, 13, 14, 15, 16]]
                else:
                    row = path.iloc[i, [0, 1, 13, 14]]
                if pd.notna(row[3]) and isinstance(row[3], (int, float)):
                    pass
                else:
                    row[2] = row[4]
                    row[3] = row[5]
                msg = (
                    f"PredictIt Event: {row[0]}\n"
                    f"Polymarket Event: {row[1]}\n"
                    f"Arb1: {row[2]:.2f}\n"
                    f"Arb2: {row[3]:.2f}"
                )
                title = ""
                content = f"{title}\n{msg}"
                send_webhook_message(POST_WEBHOOK_URL, content, username="Projections Bot")
                
        elif "Kalshi_Polymarket_Matches_New_Or_Changed" in fileName:
            POST_WEBHOOK_URL = 'https://discord.com/api/webhooks/' # replace with actual post webhook url
            path = pd.read_csv(r"Kalshi_Polymarket_Matches_New_Or_Changed.csv") # replace with desired path
            for i in range(len(path)):
                if path.shape[1] > 15:
                    row = path.iloc[i, [0, 1, 13, 14, 15, 16]]
                else:
                    row = path.iloc[i, [0, 1, 13, 14]]
                if pd.notna(row[3]) and isinstance(row[3], (int, float)):
                    pass
                else:
                    row[2] = row[4]
                    row[3] = row[5]
                msg = (
                    f"Kalshi Event: {row[0]}\n"
                    f"Polymarket Event: {row[1]}\n"
                    f"Arb1: {row[2]:.2f}\n"
                    f"Arb2: {row[3]:.2f}"
                )
                title = ""
                content = f"{title}\n{msg}"
                send_webhook_message(POST_WEBHOOK_URL, content, username="Projections Bot")
    
    # Compare matches and identify new or changed events
    def compare_matches(new_matches_df, old_file_path, output_path, platform1_name, platform2_name, new_name):
        new_matches_df['event_pair'] = new_matches_df[f'{platform1_name}_Event'] + '|' + new_matches_df[f'{platform2_name}_Event']        
        if os.path.exists(old_file_path):
            old_matches_df = pd.read_csv(old_file_path)
            old_matches_df['event_pair'] = old_matches_df[f'{platform1_name}_Event'] + '|' + old_matches_df[f'{platform2_name}_Event']
            
            # Find new events
            new_events = new_matches_df[~new_matches_df['event_pair'].isin(old_matches_df['event_pair'])]
            
            # Find events with increased arbitrage value (>= 5)
            common_events = new_matches_df[new_matches_df['event_pair'].isin(old_matches_df['event_pair'])]
            if not common_events.empty:
                common_events = common_events.merge(
                    old_matches_df[['event_pair', 'arb1', 'arb2']],
                    on='event_pair',
                    suffixes=('_new', '_old')
                )
                changed_events = common_events[
                    (common_events['arb1_new'] >= common_events['arb1_old'] + 15) |
                    (common_events['arb2_new'] >= common_events['arb2_old'] + 15)
                ]
            else:
                changed_events = pd.DataFrame()
            
            # Combine new and changed events
            new_or_changed_df = pd.concat([new_events, changed_events], ignore_index=True)
            
            if not new_or_changed_df.empty:
                new_or_changed_df = new_or_changed_df.drop(columns=['event_pair', 'arb1_old', 'arb2_old'], errors='ignore')
                new_or_changed_df.to_csv(output_path, index=False)
                print(f"Saved {len(new_or_changed_df)} new or significantly changed events to {output_path}")
                post("test1", new_name)
            else:
                print(f"No new or significantly changed events found for {platform1_name}-{platform2_name}")
        else:
            # If no old file exists, all events are new
            new_matches_df.drop(columns=['event_pair'], errors='ignore').to_csv(output_path, index=False)
            print(f"Saved {len(new_matches_df)} new events to {output_path} (no old file found)")
            post("test1", new_name)
    
    # Kalshi-PredictIt comparison
    matches_df = pd.read_csv(r"Kalshi_PredictIt_Matches.csv") # replace with actual file location
    old_kalshi_predictit_file = r"Kalshi_PredictIt_Matches_Old.csv" # replace with actual file location
    new_or_changed_kalshi_predictit_path = r"Kalshi_PredictIt_Matches_New_Or_Changed.csv" # replace with actual file location
    compare_matches(matches_df, old_kalshi_predictit_file, new_or_changed_kalshi_predictit_path, 'Kalshi', 'PredictIt', "Kalshi_PredictIt_Matches_New_Or_Changed")
    
    
    # Compare Kalshi-Polymarket matches
    matches_kp_df = pd.read_csv(r"Kalshi_Polymarket_Matches.csv") # replace with actual file location
    old_kalshi_polymarket_file = r"Kalshi_Polymarket_Matches_Old.csv" # replace with actual file location
    new_or_changed_kalshi_polymarket_path = r"Kalshi_Polymarket_Matches_New_Or_Changed.csv" # replace with actual file location
    compare_matches(matches_kp_df, old_kalshi_polymarket_file, new_or_changed_kalshi_polymarket_path, 'Kalshi', 'Polymarket', "Kalshi_Polymarket_Matches_New_Or_Changed.csv")
    
    
    # Compare PredictIt-Polymarket matches
    matches_pp_df = pd.read_csv(r"PredictIt_Polymarket_Matches.csv") # replace with actual file location
    old_predictit_polymarket_file = r"PredictIt_Polymarket_Matches_Old.csv" # replace with actual file location
    new_or_changed_predictit_polymarket_path = r"PredictIt_Polymarket_Matches_New_Or_Changed.csv" # replace with actual file location
    compare_matches(matches_pp_df, old_predictit_polymarket_file, new_or_changed_predictit_polymarket_path, 'PredictIt', 'Polymarket', "PredictIt_Polymarket_Matches_New_Or_Changed.csv")

while True:
    import time
    prediction_markets()
    time.sleep(3600)

