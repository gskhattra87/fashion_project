import requests
from sqlalchemy import text
import time

def scrape_reddit(subreddit_name, limit, db_engine):
    """
    Scrapes fashion images from a specific Subreddit using Reddit's JSON API.
    """
    # Clean the input (remove 'r/' if present)
    subreddit = subreddit_name.replace("r/", "").strip()
    
    print(f"\n--- Starting Reddit Scraping for r/{subreddit} ---")
    
    # Reddit JSON endpoint (no API key needed for basic access)
    url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
    
    # User-Agent is REQUIRED by Reddit to prevent blocking
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) FashionAI/1.0'}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Error accessing Reddit: {response.status_code}")
            return 0
            
        data = response.json()
        posts = data['data']['children']
        
        count = 0
        with db_engine.connect() as conn:
            for post in posts:
                post_data = post['data']
                
                # We only want posts that link directly to images
                image_url = post_data.get('url_overridden_by_dest', '')
                title = post_data.get('title', '')
                
                # Filter for valid image extensions
                if any(ext in image_url for ext in ['.jpg', '.png', '.jpeg', 'i.redd.it', 'imgur.com']):
                    
                    # Construct hashtags from the title for metadata
                    # (Simple logic: use words from title as tags)
                    hashtags = title.replace(",", " ").replace("'", "")
                    
                    try:
                        # Insert into Database
                        query = text("""
                            INSERT INTO fashion_items (image_url, source, hashtags, gender)
                            VALUES (:url, 'Reddit', :tag, 'Unisex')
                        """)
                        conn.execute(query, {"url": image_url, "tag": hashtags})
                        conn.commit()
                        count += 1
                    except Exception as e:
                        # Duplicate entry or DB error, skip
                        pass
                        
        print(f"✅ Reddit scraping complete. Added {count} images from r/{subreddit}.")
        return count

    except Exception as e:
        print(f"Critical Error in Reddit scraper: {e}")
        return 0