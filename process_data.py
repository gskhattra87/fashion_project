import numpy as np
import pickle
from sqlalchemy import create_engine, text
from tqdm import tqdm
import time
import requests
from io import BytesIO
from PIL import Image
import torch
from torchvision import models, transforms

# Import feature extraction logic and classification logic
from feature_extractor import get_embedding 
from category_model import get_predicted_category 

# --- CONFIGURATION ---
DATABASE_URL = "mysql+pymysql://root:@localhost:9090/fashion_trend_db"
TABLE_NAME = "fashion_items"
ENGINE = create_engine(DATABASE_URL)

# Batch size
BATCH_SIZE = 50 

MULTI_ITEM_KEYWORDS = ['ootd', 'outfit', 'full look', 'ensemble', 'collection', 'styleboard', 'lookbook']

# --- HELPER FUNCTIONS ---

def simulate_attribute_extraction(embedding: np.ndarray, item_id: int) -> dict:
    colors = ['Navy Blue', 'Forest Green', 'Burgundy', 'Charcoal Gray', 'Cream', 'Dusty Rose', 'Black', 'White']
    materials = ['Cotton', 'Denim', 'Polyester', 'Leather', 'Wool', 'Silk']
    patterns = ['Solid', 'Striped', 'Floral', 'Geometric', 'Tartan', 'Polka Dot']
    hash_val = sum(int(x) for x in str(item_id))
    return {
        "main_color": colors[hash_val % len(colors)],
        "material": materials[(hash_val + 1) % len(materials)],
        "pattern": patterns[(hash_val + 2) % len(patterns)]
    }

def check_single_item_heuristic(hashtags: str) -> str:
    if not hashtags: return 'Confirmed'
    tags_lower = hashtags.lower()
    if any(keyword in tags_lower for keyword in MULTI_ITEM_KEYWORDS):
        return 'Rejected'
    return 'Confirmed'

# --- PHASE 1: FETCH ---
def fetch_batch():
    print("   [DB] Fetching batch of items where embedding IS NULL...")
    try:
        with ENGINE.connect() as conn:
            query = text(f"""
                SELECT id, image_url, category, hashtags 
                FROM {TABLE_NAME} 
                WHERE embedding IS NULL 
                AND image_url IS NOT NULL 
                AND image_url != '' 
                LIMIT {BATCH_SIZE}
            """)
            results = conn.execute(query).fetchall()
            print(f"   [DB] Fetched {len(results)} items.")
            return results
    except Exception as e:
        print(f"   [ERROR] Database fetch failed: {e}")
        return []

# --- PHASE 2: PROCESS ---
def process_batch_in_memory(items):
    updates = []
    deletes = []
    
    print(f"   [CPU] Processing {len(items)} items...")
    
    for row in tqdm(items):
        item_id = row[0]
        url = row[1]
        existing_category = row[2] 
        hashtags = row[3] 
        
        try:
            # Verbose download check
            # print(f"      Downloading ID {item_id}: {url[:30]}...")
            embedding = get_embedding(url) 
            
            if embedding is not None:
                # 1. Classification
                predicted_cat = get_predicted_category(embedding)
                final_category = predicted_cat
                if final_category == 'Unknown' and existing_category and existing_category != 'Unknown':
                    final_category = existing_category
                
                # 2. Heuristics
                segmentation_status = check_single_item_heuristic(hashtags)
                attributes = simulate_attribute_extraction(embedding, item_id)
                serialized_embedding = pickle.dumps(embedding)
                
                updates.append({
                    "id": item_id,
                    "embedding": serialized_embedding,
                    "category": final_category,
                    "seg_check": segmentation_status,
                    "color": attributes['main_color'],
                    "material": attributes['material'],
                    "pattern": attributes['pattern']
                })
            else:
                print(f"      [WARN] Failed to get embedding for ID {item_id}. Marking for DELETE.")
                deletes.append(item_id)
                
        except Exception as e:
            print(f"      [ERROR] Exception for ID {item_id}: {e}")
            deletes.append(item_id)
            
    return updates, deletes

# --- PHASE 3: SAVE ---
def commit_batch_results(updates, deletes):
    if not updates and not deletes:
        print("   [DB] No updates or deletes to commit.")
        return

    print(f"   [DB] Committing Transaction: {len(updates)} Updates, {len(deletes)} Deletes...")
    
    try:
        with ENGINE.begin() as conn: # Transaction starts here
            
            # 1. Perform Updates
            if updates:
                update_query = text(f"""
                    UPDATE {TABLE_NAME} 
                    SET embedding = :embedding, category = :category, 
                    segmentation_check = :seg_check, main_color = :color, 
                    material = :material, pattern = :pattern
                    WHERE id = :id
                """)
                conn.execute(update_query, updates)
                print(f"      -> Updated {len(updates)} rows.")
            
            # 2. Perform Deletes
            if deletes:
                for del_id in deletes:
                    conn.execute(text(f"DELETE FROM {TABLE_NAME} WHERE id = :id"), {"id": del_id})
                print(f"      -> Deleted {len(deletes)} rows.")
                    
        print("   [SUCCESS] Transaction committed.")

    except Exception as e:
        print(f"   [CRITICAL DB ERROR] Commit failed: {e}")

# --- MAIN LOOP ---

def process_all_images():
    print(f"\n--- 🚀 STARTING DIAGNOSTIC PIPELINE ---")
    try:
        with ENGINE.connect() as conn:
            total = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE embedding IS NULL")).scalar()
        print(f"Total items pending in DB: {total}")
    except Exception as e:
        print(f"Could not connect to DB: {e}")
        return

    while True:
        # 1. Fetch
        items = fetch_batch()
        if not items:
            print("\n✅ Pipeline Finished. No more items found.")
            break
        
        # 2. Process
        updates, deletes = process_batch_in_memory(items)
        
        # 3. Save
        commit_batch_results(updates, deletes)
        
        time.sleep(0.5)

if __name__ == "__main__":
    process_all_images()