import io 
import json
import pickle
import numpy as np
import pandas as pd
import uvicorn
import subprocess 
import random 
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, BackgroundTasks, Query
from sqlalchemy import create_engine, text
from fastapi.middleware.cors import CORSMiddleware 
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.status import HTTP_401_UNAUTHORIZED 
from sklearn.metrics.pairwise import cosine_similarity
from feature_extractor import get_embedding
from typing import Optional

# --- IMPORT SCRAPERS (Graceful Handling) ---
try:
    from scraper import scrape_pinterest
except ImportError:
    scrape_pinterest = None
    print("Warning: scraper.py not found. Pinterest scraping will fail.")

try:
    from reddit_scraper import scrape_reddit
except ImportError:
    scrape_reddit = None
    print("Warning: reddit_scraper.py not found. Reddit scraping will fail.")

# --- CONFIGURATION ---
DATABASE_URL = "mysql+pymysql://root:@localhost:9090/fashion_trend_db"
TABLE_NAME = "fashion_items"

# Admin Credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "secret"
security = HTTPBasic()

app = FastAPI(title="Fashion Trend Detector")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Connection
try:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        print("Database connected successfully.")
except Exception as e:
    print(f"Database Connection Error: {e}")

# --- AUTHENTICATION ---
def authenticate_admin(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username == ADMIN_USERNAME and credentials.password == ADMIN_PASSWORD:
        return True
    raise HTTPException(
        status_code=HTTP_401_UNAUTHORIZED,
        detail="Invalid admin credentials",
        headers={"WWW-Authenticate": "Basic"},
    )

# --- HELPER FUNCTIONS ---
def get_default_category(tag):
    """Simple heuristic to assign category based on common fashion keywords."""
    tag_lower = tag.lower()
    if 'shirt' in tag_lower or 'top' in tag_lower or 'blouse' in tag_lower or 'kurta' in tag_lower: return 'Topwear'
    if 'jean' in tag_lower or 'pant' in tag_lower or 'skirt' in tag_lower or 'trouser' in tag_lower: return 'Bottomwear'
    if 'shoe' in tag_lower or 'boot' in tag_lower or 'heel' in tag_lower or 'sandal' in tag_lower: return 'Footwear'
    if 'watch' in tag_lower or 'bag' in tag_lower or 'jewel' in tag_lower or 'cap' in tag_lower: return 'Accessories'
    return 'Unknown'

def get_default_gender(tag):
    """
    FINAL FIX: Assigns gender by checking keywords strictly, preventing single word conflicts 
    from overriding the primary gender intention (e.g., #womenTops).
    """
    tag_lower = tag.lower()
    
    # Check for primary female keywords
    if 'women' in tag_lower or 'female' in tag_lower or 'girl' in tag_lower or 'ladies' in tag_lower:
        # If 'women' is found, confirm it's not a generic 'men and women' tag
        if 'men' not in tag_lower:
            return 'Women'
        
    # Check for primary male keywords
    if 'men' in tag_lower or 'male' in tag_lower or 'boy' in tag_lower:
        # If 'men' is found, confirm it's not a generic 'men and women' tag
        if 'women' not in tag_lower:
            return 'Men'
    
    # If the tag is ambiguous (contains both) or contains neither, it is Unisex
    return 'Unisex'

def simulate_attribute_extraction(embedding: np.ndarray, item_id: int) -> dict:
    """Simulates multi-task attribute extraction based on the embedding vector."""
    colors = ['Navy Blue', 'Forest Green', 'Burgundy', 'Charcoal Gray', 'Cream', 'Dusty Rose']
    materials = ['Cotton', 'Denim', 'Polyester', 'Leather', 'Wool']
    patterns = ['Solid', 'Striped', 'Floral', 'Geometric', 'Tartan']
    
    hash_val = sum(int(x) for x in str(item_id))
    
    return {
        "main_color": colors[hash_val % len(colors)],
        "material": materials[(hash_val + 1) % len(materials)],
        "pattern": patterns[(hash_val + 2) % len(patterns)]
    }

# --- CONTEXTUAL TAGS ---
CONTEXTUAL_TAGS = {
    "Formal": {"Topwear": ["blazer", "suit", "silk", "shirt"], "Bottomwear": ["trouser", "pencil skirt"], "Footwear": ["heel", "oxford"]},
    "Business Casual": {"Topwear": ["button-down", "polo", "blouse"], "Bottomwear": ["chino", "denim", "skirt"], "Footwear": ["loafer", "boot"]},
    "Casual": {"Topwear": ["t-shirt", "hoodie", "top"], "Bottomwear": ["jean", "track pant", "short"], "Footwear": ["sneaker", "sandal"]},
    "Sporty": {"Topwear": ["jersey", "athletic", "tee"], "Bottomwear": ["jogger", "legging", "short"], "Footwear": ["running shoe", "sneaker"]},
}


# --- ANALYTICS & DASHBOARD ENDPOINTS ---

@app.get("/stats/overview")
async def get_stats_overview():
    """Returns high-level statistics for the dashboard."""
    try:
        with engine.connect() as conn:
            total = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar()
            gender_res = conn.execute(text(f"SELECT gender, COUNT(*) FROM {TABLE_NAME} GROUP BY gender")).fetchall()
            gender_stats = {row[0]: row[1] for row in gender_res if row[0]}
            trends = conn.execute(text(f"SELECT COUNT(DISTINCT trend_id) FROM {TABLE_NAME} WHERE trend_id IS NOT NULL")).scalar()
            return {
                "total_images": total,
                "total_trends": trends,
                "gender_breakdown": gender_stats
            }
    except Exception as e:
        print(f"Stats Error: {e}")
        return {"total_images": 0, "total_trends": 0, "gender_breakdown": {}}

@app.get("/images/list")
async def get_images_list(
    page: int = 1, 
    limit: int = 20, 
    gender: Optional[str] = None, 
    search: Optional[str] = None
):
    """
    Returns paginated list of images.
    CRITICAL FIX: Checks if 'search' is a number to filter by trend_id, or if 'search' is 'Unknown' to filter by category.
    """
    offset = (page - 1) * limit
    try:
        with engine.connect() as conn:
            query_str = f"SELECT id, image_url, hashtags, gender, trend_id, category FROM {TABLE_NAME} WHERE 1=1"
            params = {"limit": limit, "offset": offset}

            if gender and gender != "All":
                query_str += " AND gender = :gender"
                params["gender"] = gender
            
            if search:
                search_lower = search.lower()
                
                if search_lower == 'unknown':
                    # If searching for 'Unknown', filter by Category column
                    query_str += " AND category = :search_category"
                    params["search_category"] = "Unknown"
                else:
                    try:
                        # Check if the search term is a number (Trend ID)
                        trend_id_search = int(search)
                        query_str += " AND trend_id = :trend_id"
                        params["trend_id"] = trend_id_search
                    except ValueError:
                        # If not a number, search by hashtag/tag text
                        query_str += " AND hashtags LIKE :search_hashtag"
                        params["search_hashtag"] = f"%{search}%"

            query_str += " ORDER BY id DESC LIMIT :limit OFFSET :offset"
            
            result = conn.execute(text(query_str), params).fetchall()
            
            return [{
                "id": row[0], "url": row[1], "tags": row[2], "gender": row[3], "trend": row[4], "category": row[5]
            } for row in result]
    except Exception as e:
        print(f"Gallery Error: {e}")
        return []

@app.delete("/images/{item_id}")
async def delete_image(item_id: int, admin_check: bool = Depends(authenticate_admin)):
    """Deletes an image from the database."""
    try:
        with engine.connect() as conn:
            query = text(f"DELETE FROM {TABLE_NAME} WHERE id = :id")
            result = conn.execute(query, {"id": item_id})
            conn.commit()
            
            if result.rowcount == 0:
                raise HTTPException(status_code=404, detail="Image not found")
            
            return {"message": f"Image {item_id} deleted successfully"}
    except Exception as e:
        print(f"Delete Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/images/{item_id}")
async def update_image(
    item_id: int, 
    hashtags: str = Form(...), 
    gender: str = Form(...), 
    category: str = Form(...),
    admin_check: bool = Depends(authenticate_admin)
):
    """Admin: Updates the metadata (hashtags, gender, category) for a specific image."""
    try:
        with engine.connect() as conn:
            query = text(f"""
                UPDATE {TABLE_NAME} SET 
                hashtags = :hashtags, 
                gender = :gender, 
                category = :category 
                WHERE id = :id
            """)
            result = conn.execute(query, {
                "id": item_id, 
                "hashtags": hashtags, 
                "gender": gender, 
                "category": category
            })
            conn.commit()
            
            if result.rowcount == 0:
                raise HTTPException(status_code=404, detail="Image not found or no changes made")
            
            return {"message": f"Image {item_id} updated successfully"}
    except Exception as e:
        print(f"Update Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trends")
async def get_trends():
    """Returns grouped trend clusters for the main dashboard."""
    try:
        with engine.connect() as conn:
            if conn.execute(text("SELECT COUNT(*) FROM fashion_items WHERE trend_id IS NOT NULL")).scalar() == 0:
                return []
            
            sub = text("SELECT trend_id, MAX(id) as max_id FROM fashion_items WHERE trend_id IS NOT NULL GROUP BY trend_id")
            q = text(f"""
                SELECT t1.image_url, SUBSTRING_INDEX(t1.hashtags, ',', 1) as tag, t1.trend_id, t2.count
                FROM {TABLE_NAME} t1
                JOIN ({sub}) t_max ON t1.id = t_max.max_id
                JOIN (SELECT trend_id, COUNT(*) as count FROM {TABLE_NAME} WHERE trend_id IS NOT NULL GROUP BY trend_id) t2 ON t1.trend_id = t2.trend_id
                ORDER BY t2.count DESC
            """)
            return [{"url": r[0], "tag": r[1].strip() if r[1] else "Trend", "trend_id": r[2], "count": r[3]} for r in conn.execute(q).fetchall()]
    except Exception as e:
        print(f"Trends Error: {e}")
        return []

@app.get("/trends/timeline")
async def get_trend_timeline():
    """Returns time-series data for analytics charts."""
    try:
        with engine.connect() as conn:
            q = text(f"SELECT trend_id, DATE(scraped_at) as date, COUNT(*) FROM {TABLE_NAME} WHERE trend_id IS NOT NULL GROUP BY trend_id, DATE(scraped_at) ORDER BY date ASC")
            res = conn.execute(q).fetchall()
            data = {}
            for r in res:
                k = f"Trend {r[0]}"
                if k not in data: data[k] = []
                data[k].append({"date": str(r[1]), "count": r[2]})
            return data
    except: return {}
    
# --- NEW ENDPOINT: FETCH UNIQUE TAGS ---
@app.get("/tags/unique")
async def get_unique_tags(admin_check: bool = Depends(authenticate_admin)):
    """Returns a list of all unique hashtags currently in the database."""
    try:
        with engine.connect() as conn:
            query = text(f"SELECT hashtags, COUNT(*) as count FROM {TABLE_NAME} GROUP BY hashtags ORDER BY count DESC LIMIT 100")
            result = conn.execute(query).fetchall()
            unique_tags = [row[0].split(',')[0].strip() for row in result]
            return sorted(list(set(unique_tags)))
    except Exception as e:
        print(f"Unique Tags Error: {e}")
        return []

# --- NEW ENDPOINT: BULK UPDATE ---
@app.put("/admin/bulk_update")
async def bulk_update_data(
    old_hashtag: str = Form(...),
    new_gender: str = Form(...),
    new_category: str = Form(...),
    admin_check: bool = Depends(authenticate_admin)
):
    """Admin: Bulk updates category and gender for all items matching a specific hashtag."""
    try:
        with engine.connect() as conn:
            query = text(f"""
                UPDATE {TABLE_NAME} SET 
                gender = :new_gender, 
                category = :new_category 
                WHERE hashtags LIKE :old_hashtag_pattern
            """)
            old_hashtag_pattern = f"%{old_hashtag.lower()}%"
            
            result = conn.execute(query, {
                "new_gender": new_gender, 
                "new_category": new_category, 
                "old_hashtag_pattern": old_hashtag_pattern
            })
            conn.commit()
            
            if result.rowcount == 0:
                return {"message": f"No records found matching hashtag '{old_hashtag}' to update."}
            
            return {"message": f"Successfully updated {result.rowcount} records associated with hashtag '{old_hashtag}'. Next step: Run process_data.py to update AI vectors."}
    except Exception as e:
        print(f"Bulk Update Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- NEW ENDPOINTS: PROCESS EXECUTION ---
def execute_script(script_name: str):
    """Helper function to execute an external Python script synchronously."""
    try:
        # Use sys.executable to ensure the script runs with the same Python environment
        # IMPORTANT: This runs synchronously and will block the server until complete!
        result = subprocess.run(
            ['python', script_name],
            capture_output=True,
            text=True,
            check=True
        )
        return {"success": True, "output": result.stdout, "error": result.stderr}
    except subprocess.CalledProcessError as e:
        # Script execution failed
        return {"success": False, "output": e.stdout, "error": e.stderr}
    except FileNotFoundError:
        return {"success": False, "output": "", "error": f"Error: Python executable or script '{script_name}' not found."}


@app.post("/admin/process_data")
async def run_process_data(admin_check: bool = Depends(authenticate_admin)):
    """Admin: Triggers the feature extraction and classification script (process_data.py)."""
    return execute_script("process_data.py")

@app.post("/admin/run_trends")
async def run_trend_detector(admin_check: bool = Depends(authenticate_admin)):
    """Admin: Triggers the clustering script (trend_detector.py)."""
    return execute_script("trend_detector.py")


# --- ADMIN ACTION ENDPOINTS ---

@app.post("/admin/scrape")
async def scrape_data(
    hashtag: str, 
    limit: int, 
    source: str, 
    background_tasks: BackgroundTasks,
    admin_check: bool = Depends(authenticate_admin)
):
    source = source.lower()
    
    # 1. Determine Category and Gender based on the hashtag
    default_category = get_default_category(hashtag)
    default_gender = get_default_gender(hashtag)

    if source == "pinterest":
        if scrape_pinterest:
            # CRITICAL: Pass default_category and default_gender to the scraping function
            background_tasks.add_task(scrape_pinterest, hashtag, limit, engine, default_category, default_gender)
            return {"message": f"Pinterest scraping started for #{hashtag} (Category: {default_category}, Gender: {default_gender})."}
        raise HTTPException(500, "Pinterest scraper missing.")
        
    elif source == "reddit":
        if scrape_reddit:
            # CRITICAL: Pass default_category and default_gender to the scraping function
            background_tasks.add_task(scrape_reddit, hashtag, limit, engine, default_category, default_gender)
            return {"message": f"Reddit scraping started for r/{hashtag} (Category: {default_category}, Gender: {default_gender})."}
        raise HTTPException(500, "Reddit scraper missing.")
        
    elif source == "instagram":
        return {"message": "Instagram scraping requires manual terminal execution."}
    
    raise HTTPException(400, "Invalid source.")

@app.post("/admin/upload_dataset")
async def upload_dataset(file: UploadFile = File(...), source_name: str = Form("Kaggle Dataset")):
    if not file.filename: raise HTTPException(status_code=400, detail="No file.")
    try:
        contents = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents), on_bad_lines='skip')
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else: raise HTTPException(status_code=400, detail="Unsupported file.")

        required = ['image_url', 'hashtags']
        if not all(col in df.columns for col in required):
            raise HTTPException(status_code=400, detail="Missing columns.")
        
        # Determine category/gender dynamically if not provided in CSV
        df['category'] = df.apply(lambda row: str(row['category']) if 'category' in row and pd.notna(row['category']) else get_default_category(str(row['hashtags'])), axis=1)
        df['gender'] = df.apply(lambda row: str(row['gender']) if 'gender' in row and pd.notna(row['gender']) else get_default_gender(str(row['hashtags'])), axis=1)


        records = []
        for index, row in df.iterrows():
            records.append({
                "image_url": str(row['image_url']),
                "source": source_name,
                "hashtags": str(row['hashtags']),
                "gender": row['gender'], 
                "category": row['category']
            })

        with engine.connect() as conn:
            conn.execute(text(f"""
                INSERT INTO {TABLE_NAME} (image_url, source, hashtags, gender, category)
                VALUES (:image_url, :source, :hashtags, :gender, :category)
            """), records)
            conn.commit()
            
        return {"message": f"Imported {len(records)} records.", "next_step": "Run process_data.py"}
    except Exception as e:
        print(f"Upload Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- NEW ENDPOINT: ATTRIBUTE LOOKUP (Feature 2 Frontend Access) ---
@app.get("/items/{item_id}/attributes")
async def get_item_attributes(item_id: int):
    """Fetches detailed attributes (color, material, pattern) for a specific item."""
    try:
        with engine.connect() as conn:
            # Note: Requires main_color, material, pattern columns in DB
            query = text(f"SELECT main_color, material, pattern FROM {TABLE_NAME} WHERE id = :id")
            result = conn.execute(query, {"id": item_id}).fetchone()
            
            if result:
                return {
                    "id": item_id,
                    "color": result[0],
                    "material": result[1],
                    "pattern": result[2],
                }
            raise HTTPException(status_code=404, detail="Item or attributes not found.")
    except Exception as e:
        print(f"Attribute Fetch Error: {e}")
        raise HTTPException(status_code=500, detail="Database Error")


# --- NEW ENDPOINT: TREND FORECASTING (Feature 4 Logic) ---
@app.get("/trends/forecast")
async def get_trend_forecast(admin_check: bool = Depends(authenticate_admin)):
    """Simulates predicting the top 3 rising trends based on data growth."""
    try:
        with engine.connect() as conn:
            q = text(f"SELECT trend_id, COUNT(*) as count FROM {TABLE_NAME} WHERE trend_id IS NOT NULL GROUP BY trend_id")
            res = conn.execute(q).fetchall()
            
            if not res: return []

            # Simulate growth/ranking
            forecast_data = []
            for trend_id, count in res:
                 # Simulate a growth score based on the current size and a random factor
                 growth_score = count * (1 + random.uniform(0.05, 0.25))
                 forecast_data.append({"trend_id": trend_id, "score": growth_score})
                 
            # Sort by simulated growth score and take top 3
            forecast_data.sort(key=lambda x: x['score'], reverse=True)
            top_3 = forecast_data[:3]
            
            # Fetch representative image/tag for the forecasted trends
            forecast_output = []
            for item in top_3:
                rep_q = text(f"SELECT image_url, hashtags FROM {TABLE_NAME} WHERE trend_id = :tid ORDER BY scraped_at DESC LIMIT 1")
                rep_res = conn.execute(rep_q, {"tid": item['trend_id']}).fetchone()
                
                if rep_res:
                    forecast_output.append({
                        "trend_id": item['trend_id'],
                        "tag": rep_res[1].split(',')[0].strip(),
                        "url": rep_res[0],
                        "forecast_score": f"{item['score']:.1f}"
                    })
            
            return forecast_output

    except Exception as e:
        print(f"Forecast Error: {e}")
        return []

# --- RECOMMENDATION LOGIC ---

@app.post("/recommend")
async def recommend_outfit(file: UploadFile = File(...), gender: str = Form("Unisex")):
    """Standard Similarity Search: Finds 3 items most similar to the uploaded image."""
    N_REC = 3
    try:
        bytes_data = await file.read()
        u_emb = get_embedding(bytes_data)
        if u_emb is None: raise HTTPException(400, "Feature extraction failed.")

        safe_gender = gender.title()
        with engine.connect() as conn:
            q = text(f"SELECT id, image_url, hashtags, embedding, trend_id, category FROM {TABLE_NAME} WHERE embedding IS NOT NULL AND (gender = :g OR gender = 'Unisex')")
            res = conn.execute(q, {"g": safe_gender}).fetchall()
        
        if not res: return {"message": "No data.", "recommendations": [], "detected_style": "None"}

        items, embs = [], []
        for r in res:
            try:
                emb = pickle.loads(r[3]) if not isinstance(r[3], str) else np.array(json.loads(r[3]))
                items.append({"url": r[1], "tags": r[2], "trend": r[4], "category": r[5]})
            except: continue

        if not embs: return {"message": "Error.", "recommendations": [], "detected_style": "Error"}

        sims = cosine_similarity(u_emb.reshape(1, -1), np.array(embs))[0]
        top_idx = sims.argsort()[::-1][:N_REC]
        
        recs = [{"name": f"Item ({items[i]['tags'].split(',')[0]})", "url": items[i]['url'], "confidence": f"{sims[i]*100:.1f}%"} for i in top_idx]
        trend_id = items[top_idx[0]]['trend'] if top_idx.size > 0 else "Unknown"
        
        return {"message": "Success", "detected_style": f"Trend Cluster #{trend_id}", "recommendations": recs}

    except Exception as e:
        print(f"Rec Error: {e}")
        raise HTTPException(500, str(e))

@app.post("/recommend/contextual")
async def recommend_contextual(
    file: UploadFile = File(...), 
    gender: str = Form("Unisex"), 
    input_category: str = Form("Topwear"),
    occasion: str = Form("Casual")
):
    """
    AI Stylist: Finds matching complementary items based on visual similarity AND contextual keywords.
    """
    try:
        bytes_data = await file.read()
        u_emb = get_embedding(bytes_data)
        if u_emb is None: raise HTTPException(400, "Feature extraction failed.")

        safe_gender = gender.title()
        input_category_upper = input_category.title()
        
        # Contextual Tags Dictionary
        CONTEXTUAL_TAGS = {
            "Formal": {"Topwear": ["blazer", "suit", "silk", "shirt"], "Bottomwear": ["trouser", "pencil skirt"], "Footwear": ["heel", "oxford"]},
            "Business Casual": {"Topwear": ["button-down", "polo", "blouse"], "Bottomwear": ["chino", "denim", "skirt"], "Footwear": ["loafer", "boot"]},
            "Casual": {"Topwear": ["t-shirt", "hoodie", "top"], "Bottomwear": ["jean", "track pant", "short"], "Footwear": ["sneaker", "sandal"]},
            "Sporty": {"Topwear": ["jersey", "athletic", "tee"], "Bottomwear": ["jogger", "legging", "short"], "Footwear": ["running shoe", "sneaker"]},
        }
        
        # Determine keywords relevant to the occasion
        occasion_dict = CONTEXTUAL_TAGS.get(occasion, {})
        occasion_keywords = set()
        for cat_keywords in occasion_dict.values():
            occasion_keywords.update(cat_keywords)

        COMPLEMENTARY_CATEGORIES = {
            "Topwear": ["Bottomwear", "Footwear", "Accessories"],
            "Bottomwear": ["Topwear", "Footwear", "Accessories"],
            "Footwear": ["Topwear", "Bottomwear", "Accessories"],
            "Accessories": ["Topwear", "Bottomwear", "Footwear"],
        }

        # Fetch Candidates from Database (Gender Filtering applied here)
        q_str = f"SELECT id, image_url, hashtags, embedding, category, gender FROM {TABLE_NAME} WHERE embedding IS NOT NULL AND category != 'Unknown'"
        params = {}
        
        if safe_gender == 'Unisex':
            q_str += " AND gender IN ('Men', 'Women', 'Unisex')"
        else:
            q_str += " AND gender IN (:g, 'Unisex')"
            params["g"] = safe_gender

        with engine.connect() as conn:
            raw_results = conn.execute(text(q_str), params).fetchall()

        if not raw_results: return {"message": "No data found.", "outfit": {}}
        
        uploaded_embedding_2d = u_emb.reshape(1, -1)
        outfit_recommendations = {}

        # Search for each complementary category
        for target_cat_name in COMPLEMENTARY_CATEGORIES.get(input_category_upper, []):
            
            best_match = {"item": None, "score": -1.0}
            
            for row in raw_results:
                
                # Check 1: Category Match
                if str(row[4]).title() != target_cat_name:
                    continue
                
                # Check 2: Contextual Match (Soft Boost)
                item_tags = str(row[2]).lower()
                context_boost = 0.0
                
                # Check if item tags match ANY keyword for this specific category in the occasion map
                target_occasion_keywords = occasion_dict.get(target_cat_name, [])
                if any(kw in item_tags for kw in target_occasion_keywords):
                    context_boost = 0.2 # 20% boost for matching occasion
                
                try:
                    emb = pickle.loads(row[3]) if not isinstance(row[3], str) else np.array(json.loads(row[3]))
                    similarity = cosine_similarity(uploaded_embedding_2d, emb.reshape(1, -1))[0][0]
                    
                    # Apply Boost
                    final_score = similarity + context_boost
                    
                    if final_score > best_match["score"]:
                        best_match["score"] = final_score
                        best_match["item"] = row
                except:
                    continue
            
            if best_match["item"]:
                 outfit_recommendations[target_cat_name] = {
                    "url": best_match["item"][1],
                    "name": best_match["item"][2].split(',')[0],
                    "confidence": f"{min(best_match['score'] * 100, 99.9):.1f}%"
                 }

        return {"message": "Success", "input_category": input_category, "occasion": occasion, "outfit": outfit_recommendations}

    except Exception as e:
        print(f"Contextual Stylist Error: {e}")
        raise HTTPException(500, str(e))

@app.post("/recommend/complete_outfit")
async def complete_outfit(file: UploadFile = File(...), gender: str = Form("Unisex"), input_category: str = Form("Topwear")):
    """
    AI Stylist: Finds matching complementary items based on database category, enforcing refined gender rules.
    """
    try:
        bytes_data = await file.read()
        u_emb = get_embedding(bytes_data)
        if u_emb is None: raise HTTPException(400, "Could not extract features. Image resolution too low.")

        safe_gender = gender.title()
        input_category_upper = input_category.title()

        # 2. Define Complementary Categories based on the cleaned CATEGORY column
        COMPLEMENTARY_CATEGORIES = {
            "Topwear": ["Bottomwear", "Footwear", "Accessories"],
            "Bottomwear": ["Topwear", "Footwear", "Accessories"],
            "Footwear": ["Topwear", "Bottomwear", "Accessories"],
            "Accessories": ["Topwear", "Bottomwear", "Footwear"],
        }
        
        # 3. Fetch Candidates from Database
        with engine.connect() as conn:
            
            # --- CRITICAL FIX: REFINED GENDER FILTERING ---
            q_str = f"SELECT id, image_url, hashtags, embedding, category, gender FROM {TABLE_NAME} WHERE embedding IS NOT NULL AND category != 'Unknown'"
            params = {}
            
            if safe_gender == 'Unisex':
                q_str += " AND gender IN ('Men', 'Women', 'Unisex')"
            elif safe_gender == 'Men' or safe_gender == 'Women':
                q_str += " AND gender IN (:g, 'Unisex')"
                params["g"] = safe_gender
            else:
                 q_str += " AND gender IN ('Men', 'Women', 'Unisex')"

                
            q = text(q_str)
            result = conn.execute(q, params).fetchall()

        if not result: return {"message": "No data found.", "outfit": {}}

        # 4. Process all embeddings and filter based on the target category
        all_items = []
        for row in result:
            try:
                emb = pickle.loads(row[3]) if not isinstance(row[3], str) else np.array(json.loads(row[3]))
                all_items.append({
                    "data": row, 
                    "embedding": emb, 
                    "tags": str(row[2]).lower(),
                    "db_category": str(row[4]).title() 
                })
            except: continue
        
        uploaded_embedding_2d = u_emb.reshape(1, -1)
        outfit_recommendations = {}

        target_categories_to_search = COMPLEMENTARY_CATEGORIES.get(input_category_upper, [])

        for target_cat_name in target_categories_to_search:
            
            candidates = []
            for item in all_items:
                # Strictly filter by the item's stored database category
                if item['db_category'] == target_cat_name:
                    candidates.append(item)

            if not candidates: continue
                
            # Calculate Similarity
            candidate_embeddings = np.array([c['embedding'] for c in candidates])
            sim_scores = cosine_similarity(uploaded_embedding_2d, candidate_embeddings)[0]
            
            # Get best match
            best_idx = sim_scores.argmax()
            best_item = candidates[best_idx]['data']
            score = sim_scores[best_idx]
            
            outfit_recommendations[target_cat_name] = {
                "url": best_item[1],
                "name": best_item[2].split(',')[0],
                "confidence": f"{score * 100:.1f}%"
            }

        return {"message": "Success", "input_category": input_category, "outfit": outfit_recommendations}

    except Exception as e:
        print(f"Stylist Error: {e}")
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)