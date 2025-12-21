import numpy as np
import pickle
from sqlalchemy import create_engine, text
from sklearn.cluster import KMeans
from tqdm import tqdm

# --- CONFIGURATION ---
DATABASE_URL = "mysql+pymysql://root:@localhost:9090/fashion_trend_db"
ENGINE = create_engine(DATABASE_URL)

# INCREASED CLUSTERS for larger dataset (40k images)
# 10 was good for testing; 50 is better for finding diverse styles in a large dataset.
NUM_CLUSTERS = 50 
EMBEDDING_DIMENSION = 2048

def fetch_all_embeddings():
    """
    Fetches all fashion item IDs and their embeddings from the database.
    """
    print("Fetching all embeddings from the database...")
    with ENGINE.connect() as conn:
        # Fetch only items that have an embedding
        query = text("SELECT id, embedding FROM fashion_items WHERE embedding IS NOT NULL")
        result = conn.execute(query).fetchall()
        
        items = []
        embeddings = []
        
        for item_id, pickled_embedding in tqdm(result):
            try:
                if isinstance(pickled_embedding, str):
                     # Handle legacy JSON format if any
                     import json
                     embedding_vector = np.array(json.loads(pickled_embedding))
                else:
                     # Handle standard pickle format
                     embedding_vector = pickle.loads(pickled_embedding)
                
                if embedding_vector.shape[0] == EMBEDDING_DIMENSION:
                    items.append(item_id)
                    embeddings.append(embedding_vector)
            except Exception as e:
                # Silently skip bad embeddings to ensure process completes
                continue
                    
        print(f"Successfully loaded {len(embeddings)} embeddings.")
        return items, np.array(embeddings)

def run_clustering(items: list, embeddings: np.ndarray):
    """
    Applies K-Means clustering to the image embeddings.
    """
    if len(embeddings) < NUM_CLUSTERS:
        print(f"Not enough data ({len(embeddings)}) to form {NUM_CLUSTERS} clusters. Using k={len(embeddings)} instead.")
        k = len(embeddings)
    else:
        k = NUM_CLUSTERS

    print(f"Starting K-Means clustering with K={k}...")
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, verbose=0)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    return cluster_labels

def update_trend_ids(items: list, cluster_labels: np.ndarray):
    """
    Saves the determined trend_id back into the fashion_items table.
    """
    print("Updating database with trend IDs...")
    
    # We update in batches for speed
    with ENGINE.connect() as conn:
        # Create a transaction
        trans = conn.begin()
        try:
            for i in tqdm(range(len(items))):
                item_id = items[i]
                trend_id = int(cluster_labels[i])
                conn.execute(
                    text("UPDATE fashion_items SET trend_id = :trend_id WHERE id = :id"),
                    {"trend_id": trend_id, "id": item_id}
                )
            trans.commit()
        except Exception as e:
            trans.rollback()
            print(f"Error updating trends: {e}")
            
    print("Database update complete.")

if __name__ == '__main__':
    print("--- Starting Trend Detection (Clustering) Pipeline ---")
    
    # 1. Fetch data
    item_ids, all_embeddings = fetch_all_embeddings()
    
    if len(item_ids) > 0:
        # 2. Run Clustering
        trend_ids = run_clustering(item_ids, all_embeddings)
        
        if trend_ids is not None:
            # 3. Save Results
            update_trend_ids(item_ids, trend_ids)
            print(f"\nTrend Detection Complete. You have {NUM_CLUSTERS} identified fashion trends.")
            
    else:
        print("\nNo embeddings found. Please run process_data.py first.")