from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import os

def setup_headless_chrome():
    print("Setting up headless Chrome for WSL...")
    options = Options()
    options.add_argument('--headless') 
    options.add_argument('--no-sandbox') 
    options.add_argument('--disable-dev-shm-usage') 
    options.add_argument('--window-size=1920,1080')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)

def scrape_gurugram_properties():
    driver = setup_headless_chrome()
    properties_data = []
    
    target_url = "https://www.magicbricks.com/property-for-sale/residential-real-estate?bedroom=2,3&proptype=Multistorey-Apartment,Builder-Floor-Apartment,Penthouse,Studio-Apartment&cityName=Gurgaon"
    
    try:
        print(f"Loading website: {target_url}")
        driver.get(target_url)
        
        time.sleep(5) 
        
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
        print("Extracting property cards...")
        
        property_cards = driver.find_elements(By.CLASS_NAME, 'mb-srp__card') 
        
        for card in property_cards:
            try:
                price = card.find_element(By.CLASS_NAME, 'mb-srp__card__price--amount').text
                location = card.find_element(By.CLASS_NAME, 'mb-srp__card--title').text
                sqft = card.find_element(By.CLASS_NAME, 'mb-srp__card__summary--value').text
                
                properties_data.append({
                    "Location": location,
                    "Price_Raw": price,
                    "SqFt_Raw": sqft,
                    "Age_Years": 5 
                })
            except Exception as e:
                continue
                
    finally:
        print("Closing browser...")
        driver.quit()

    df = pd.DataFrame(properties_data)
    
    os.makedirs('data', exist_ok=True)
    
    output_path = 'data/raw_gurugram_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Success! Scraped {len(df)} properties and saved to {output_path}")

if __name__ == "__main__":
    scrape_gurugram_properties()