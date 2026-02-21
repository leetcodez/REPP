import pandas as pd
import numpy as np
import re

def clean_real_estate_data():
    print("Loading raw Gurugram data...")
    df = pd.read_csv('data/raw_gurugram_data.csv')
    
    def parse_price(price_str):
        if not isinstance(price_str, str): return 0
        numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", price_str)
        if not numbers: return 0
        val = float(numbers[0])
        
        if 'Cr' in price_str:
            return int(val * 10000000)
        elif 'Lac' in price_str or 'Lakh' in price_str:
            return int(val * 100000)
        return int(val)
        
    df['Price'] = df['Price_Raw'].apply(parse_price)
    
    def parse_sqft(sqft_str):
        if not isinstance(sqft_str, str): return 0
        numbers = re.findall(r'\d+', sqft_str.replace(',', ''))
        return int(numbers[0]) if numbers else 0

    df['SqFt'] = df['SqFt_Raw'].apply(parse_sqft)
    
    df['Location_ID'] = df['Location'].astype('category').cat.codes

    df['Age_Years'] = np.random.randint(1, 16, size=len(df))

    df = df[df['Price'] > 0]
    df = df[df['SqFt'] > 0]

    ml_matrix = df[['Location_ID', 'SqFt', 'Age_Years', 'Price']]
    
    output_path = 'data/clean_gurugram_data.csv'
    ml_matrix.to_csv(output_path, index=False, header=False)
    
    print(f"Successfully cleaned {len(ml_matrix)} properties.")
    print(f"Saved pure numerical matrix to {output_path} for C++ ingestion.")
    print("\nPreview of the Cleaned Training Matrix:")
    print(ml_matrix.head())

if __name__ == "__main__":
    clean_real_estate_data()