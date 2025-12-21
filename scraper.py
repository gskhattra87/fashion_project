from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from sqlalchemy import text
import time
import random

# Update scrape_pinterest definition to accept default_category AND default_gender
def scrape_pinterest(hashtag, limit, db_engine, default_category="Unknown", default_gender="Unisex"):
    """
    Scrapes image URLs from Pinterest with enhanced scrolling resilience and saves category and gender,
    while filtering out full outfits or non-clothing items.
    """
    search_url = f"https://www.pinterest.com/search/pins/?q={hashtag}"
    
    # --- Advanced Filtering Keywords ---
    # Skip if the image likely contains a full outfit or is a generic scene
    OUTFIT_EXCLUSION_TAGS = ['ootd', 'outfit', 'style board', 'lookbook', 'street style', 'fashion inspo', 'entire look']
    # If the category is Accessory, we are more strict to avoid generic images
    ACCESSORY_STRICT_TAGS = ['home decor', 'flat lay', 'pattern', 'design']

    # Setup Selenium (Headless Chrome)
    chrome_options = Options()
    chrome_options.add_argument("--headless") 
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    # Vary User Agent for resilience
    chrome_options.add_argument(f"user-agent={random.choice(['Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:100.0) Gecko/20100101 Firefox/100.0'])}")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.implicitly_wait(5)
    
    print(f"Scraping started for #{hashtag} (Category: {default_category}, Gender: {default_gender}, Target: {limit})...")
    
    try:
        driver.get(search_url)
        
        processed_urls = set()
        scroll_count = 0
        MAX_SCROLLS = 75
        
        while len(processed_urls) < limit and scroll_count < MAX_SCROLLS:
            last_height = driver.execute_script("return document.body.scrollHeight")
            
            # CRITICAL FIX 1: AGGRESSIVE RANDOM SCROLLING
            random_scroll_amount = random.randint(1500, 3000)
            driver.execute_script(f"window.scrollBy(0, {random_scroll_amount});")
            time.sleep(random.uniform(1.5, 3.0))
            
            try:
                WebDriverWait(driver, 15).until(
                    lambda d: d.execute_script("return document.body.scrollHeight > arguments[0]", last_height)
                )
            except:
                break

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            images = soup.find_all('img')
            
            new_urls_found = 0
            
            with db_engine.connect() as conn:
                for img in images:
                    src = img.get('src')
                    
                    if src and src.startswith("http") and ("pinimg.com" in src or "pinterest.com" in src):
                        
                        high_res_url = src
                        # High-resolution fix
                        if "236x" in src or "474x" in src or "564x" in src:
                             high_res_url = src.replace('236x', '736x').replace('474x', '736x').replace('564x', '736x')

                        if src not in processed_urls:
                            
                            # --- CRITICAL FIX 2: ADVANCED FILTERING ---
                            # Use the input hashtag (which is passed as the item's tag) for filtering
                            tag_lower = hashtag.lower()
                            
                            # Filter 1: Exclude obvious full outfits
                            if any(ex_tag in tag_lower for ex_tag in OUTFIT_EXCLUSION_TAGS):
                                # print(f"Skipped {hashtag}: Detected as multi-item outfit.")
                                continue

                            # Filter 2: Be strict with accessories/non-clothing items
                            if default_category in ['Accessories', 'Unknown'] and any(strict_tag in tag_lower for strict_tag in ACCESSORY_STRICT_TAGS):
                                # print(f"Skipped {hashtag}: Generic scene or accessory strict skip.")
                                continue
                            
                            # Filter 3: Prevent saving generic Pinterest UI elements
                            if "236x" not in src and "474x" not in src and "564x" not in src and not ("736x" in src):
                                continue
                            # --- END ADVANCED FILTERING ---


                            # Insert into Database
                            query = text("INSERT INTO fashion_items (image_url, hashtags, category, gender) VALUES (:url, :tag, :cat, :gen)")
                            conn.execute(query, {"url": high_res_url, "tag": hashtag, "cat": default_category, "gen": default_gender})
                            processed_urls.add(src)
                            new_urls_found += 1
                        
                conn.commit()
            
            scroll_count += 1
            if new_urls_found == 0 and driver.execute_script("return document.body.scrollHeight") == last_height:
                 break

            print(f"-> Batch {scroll_count}: Added {new_urls_found} new. Total collected: {len(processed_urls)}")
            time.sleep(random.uniform(2, 4))

        total_collected = len(processed_urls)
        print(f"Scraping finished. Added {total_collected} images for #{hashtag}.")
        return total_collected

    except Exception as e:
        print(f"Critical Scraping Error: {e}")
        return 0
    finally:
        driver.quit()