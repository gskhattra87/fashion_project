from sqlalchemy import create_engine, text
from tqdm import tqdm

# --- CONFIGURATION ---
DATABASE_URL = "mysql+pymysql://root:@localhost:9090/fashion_trend_db"
ENGINE = create_engine(DATABASE_URL)
TABLE_NAME = "fashion_items"

def get_correct_category(hashtags):
    """Determines the correct category based on keywords."""
    tag_lower = str(hashtags).lower()
    
    # Footwear (High priority to catch sneakers labeled as tops)
    if any(x in tag_lower for x in ['shoe', 'sneaker', 'boot', 'sandal', 'heel', 'flat', 'loafer', 'slipper', 'flip flop']):
        return 'Footwear'
        
    # Bottomwear
    if any(x in tag_lower for x in ['jean', 'pant', 'trouser', 'short', 'skirt', 'legging', 'jogger', 'pyjama', 'track']):
        return 'Bottomwear'
        
    # Topwear
    if any(x in tag_lower for x in ['shirt', 'top', 'blouse', 'jacket', 'sweater', 'hoodie', 'kurta', 'tee', 'vest', 'coat', 'blazer']):
        return 'Topwear'
        
    # Accessories
    if any(x in tag_lower for x in ['watch', 'bag', 'purse', 'hat', 'cap', 'jewel', 'belt', 'sunglass', 'tie', 'scarf', 'wallet', 'necklace', 'earring']):
        return 'Accessories'
        
    return 'Unknown'

def get_correct_gender(hashtags):
    tag_lower = str(hashtags).lower()
    if 'men' in tag_lower and 'women' not in tag_lower: return 'Men'
    if 'women' in tag_lower and 'men' not in tag_lower: return 'Women'
    if 'boy' in tag_lower: return 'Men'
    if 'girl' in tag_lower: return 'Women'
    return 'Unisex'

def fix_database_categories():
    print("--- Starting Database Category Cleanup ---")
    
    # Using 'begin()' context manager handles transaction commit/rollback automatically
    with ENGINE.begin() as conn:
        # Fetch all items
        items = conn.execute(text(f"SELECT id, hashtags FROM {TABLE_NAME}")).fetchall()
        print(f"Analyzing {len(items)} items...")
        
        updates = 0
        
        for item in tqdm(items):
            item_id = item[0]
            hashtags = item[1]
            
            new_cat = get_correct_category(hashtags)
            new_gen = get_correct_gender(hashtags)
            
            if new_cat != 'Unknown':
                conn.execute(text(f"""
                    UPDATE {TABLE_NAME} 
                    SET category = :cat, gender = :gen 
                    WHERE id = :id
                """), {"cat": new_cat, "gen": new_gen, "id": item_id})
                updates += 1
        
        print(f"✅ Successfully updated {updates} items with correct categories/genders.")
        print("Please run `python process_data.py` next if you need to update embeddings (optional).")

if __name__ == "__main__":
    fix_database_categories()