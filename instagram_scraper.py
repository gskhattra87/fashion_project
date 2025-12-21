import instaloader
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from tqdm import tqdm
import time

# --- CONFIGURATION ---
DATABASE_URL = "mysql+pymysql://root:@localhost:9090/fashion_trend_db"
ENGINE = create_engine(DATABASE_URL)
TABLE_NAME = "fashion_items"


# --- INSTAGRAM CONFIGURATION ---
INSTAGRAM_USERNAME = "khatt_rags"  
INSTAGRAM_PASSWORD = "Khattra@007"  
SESSION_FILE_PATH = INSTAGRAM_USERNAME # Instaloader will save session as 'YOUR_INSTAGRAM_USERNAME.session'

# Initialize Instaloader
# L is a global object that handles connections and downloads
L = instaloader.Instaloader(
    # Keep only the standard argument for file handling.
    compress_json=False, 
)


try:
    # Try to load a saved session first
    L.load_session_from_file(INSTAGRAM_USERNAME, SESSION_FILE_PATH)
    print(f"Loaded session for {INSTAGRAM_USERNAME}. No need to log in.")
except FileNotFoundError:
    print(f"Session file not found. Logging in as {INSTAGRAM_USERNAME}...")
    try:
        L.login(INSTAGRAM_USERNAME, INSTAGRAM_PASSWORD)
        L.save_session_to_file(INSTAGRAM_USERNAME)
        print("Login successful and session saved.")
    except Exception as e:
        print(f"Login failed! Check username/password. Error: {e}")

 

def scrape_instagram_by_hashtag(hashtag: str, limit: int):
    """
    Scrapes images and metadata from Instagram for a given hashtag.
    
    Args:
        hashtag: The fashion hashtag to search (e.g., 'streetstyle').
        limit: The maximum number of posts to process.
    """
    print(f"\n--- Starting Instagram Scraping for #{hashtag} (Limit: {limit}) ---")
    
    # Get the top posts for the hashtag
    posts = L.get_hashtag_posts(hashtag)
    
    count = 0
    with ENGINE.connect() as conn:
        for post in tqdm(posts):
            if count >= limit:
                break
            
            # Filter for images (skip videos and carousels for simplicity)
            if post.is_video:
                continue

            # Extract data
            image_url = post.url
            # Join all tags into a single string for your database
            hashtags = ",".join(post.caption_hashtags) if post.caption_hashtags else ""
            
            if image_url:
                try:
                    # Insert data into the fashion_items table
                    insert_query = text(f"""
                        INSERT INTO {TABLE_NAME} (image_url, source, hashtags, gender)
                        VALUES (:image_url, 'Instagram', :hashtags, 'Unisex')
                        ON DUPLICATE KEY UPDATE hashtags = :hashtags 
                    """)
                    
                    conn.execute(insert_query, {
                        "image_url": image_url,
                        "hashtags": hashtags
                    })
                    conn.commit()
                    count += 1
                    
                except SQLAlchemyError as e:
                    # Often catches duplicate key errors or connection issues
                    print(f"\n[Error] Database insertion failed for {image_url}: {e}")
                    conn.rollback() 
            
            # Rate limiting: Wait a moment between posts to avoid being blocked
            time.sleep(1) 

    print(f"\n✅ Instagram scraping complete. Added {count} new items.")

if __name__ == '__main__':
    # You can call this from your main program or run directly for a test batch
    # Replace 'fashionnova' with a hashtag relevant to your project
    scrape_instagram_by_hashtag(hashtag='mensfashion', limit=50)