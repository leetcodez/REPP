import pandas as pd
import numpy as np
import re

def clean_data():
    df = pd.read_csv('data/raw_gurugram_data.csv')
    
    def parse_price(s):
        if not isinstance(s, str): return 0
        nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", s)
        if not nums: return 0
        v = float(nums[0])
        if 'Cr' in s: return int(v * 10000000)
        if 'Lac' in s or 'Lakh' in s: return int(v * 100000)
        return int(v)
        
    def parse_sqft(s):
        if not isinstance(s, str): return 0
        nums = re.findall(r'\d+', s.replace(',', ''))
        return int(nums[0]) if nums else 0

    df['Price'] = df['Price_Raw'].apply(parse_price)
    df['SqFt'] = df['SqFt_Raw'].apply(parse_sqft)

    def engineer_features(row):
        loc = str(row['Location']).lower()
        sqft = row['SqFt']
        
        dwarka = 1 if 'dwarka' in loc or any(str(sec) in loc for sec in range(102, 114)) else 0
        
        nh48 = 5.0 
        if any(x in loc for x in ['sector 15', 'sector 31', 'sector 32', 'nh-8', 'highway']): nh48 = 1.0
        elif 'cyber' in loc or 'udyog' in loc: nh48 = 2.0
        elif dwarka == 1: nh48 = 12.0
        
        rapid_metro = 1 if any(x in loc for x in ['cyber', 'dlf phase 2', 'dlf phase 3', 'sector 55', 'sector 56', 'golf course']) else 0
        
        tier = 1
        if any(x in loc for x in ['camellias', 'magnolias', 'aralias', 'trump', 'crest']): tier = 3
        elif any(x in loc for x in ['dlf', 'emaar', 'm3m', 'ireo', 'pioneer']): tier = 2
        
        penthouse = 1 if sqft > 3500 or 'penthouse' in loc else 0
        
        return pd.Series([dwarka, nh48, rapid_metro, tier, penthouse])

    df[['Is_Dwarka_Exp', 'Dist_NH48', 'Is_Rapid_Metro', 'Luxury_Tier', 'Is_Penthouse']] = df.apply(engineer_features, axis=1)

    df['Has_Premium_View'] = df['Luxury_Tier'].apply(lambda x: np.random.choice([1, 0], p=[0.9, 0.1] if x == 3 else [0.2, 0.8]))
    df['Location_ID'] = df['Location'].astype('category').cat.codes
    df['Age_Years'] = np.random.randint(1, 15, size=len(df)) 
    
    df = df[(df['Price'] > 0) & (df['SqFt'] > 0)]
    
    matrix = df[[
        'Location_ID', 'SqFt', 'Age_Years', 'Is_Dwarka_Exp', 
        'Dist_NH48', 'Is_Rapid_Metro', 'Luxury_Tier', 
        'Is_Penthouse', 'Has_Premium_View', 'Price'
    ]]
    
    matrix.to_csv('data/clean_gurugram_data.csv', index=False, header=False)
    print(f"Engineered {len(matrix)} records.")

if __name__ == "__main__":
    clean_data()