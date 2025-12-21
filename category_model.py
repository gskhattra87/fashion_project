import numpy as np
import pickle
from sqlalchemy import create_engine, text
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Tuple

# --- CONFIGURATION ---
DATABASE_URL = "mysql+pymysql://root:@localhost:9090/fashion_trend_db"
TABLE_NAME = "fashion_items"
ENGINE = create_engine(DATABASE_URL)
EMBEDDING_DIMENSION = 2048

# Target Categories for the AI Stylist
TARGET_CATEGORIES = ['Topwear', 'Bottomwear', 'Footwear', 'Accessories']

def calculate_category_centroids() -> Dict[str, np.ndarray]:
    """
    Calculates the average embedding vector (centroid) for each target category 
    using the small, manually verified dataset in the database.
    This acts as the AI's 'knowledge' base.
    """
    print("Calculating category centroids for prediction...")
    centroids = {}
    
    with ENGINE.connect() as conn:
        for cat in TARGET_CATEGORIES:
            # Fetch all embeddings belonging to this category
            query = text(f"SELECT embedding FROM {TABLE_NAME} WHERE category = :cat AND embedding IS NOT NULL")
            result = conn.execute(query, {"cat": cat}).fetchall()
            
            embeddings = []
            for row in result:
                try:
                    # Load the binary embedding data
                    emb = pickle.loads(row[0])
                    embeddings.append(emb)
                except:
                    continue
            
            if embeddings:
                # Calculate the average vector for the category
                avg_centroid = np.mean(embeddings, axis=0)
                centroids[cat] = avg_centroid
                print(f"   -> Centroid found for {cat} ({len(embeddings)} items)")
            
    return centroids

# Calculate centroids once when the model is loaded
CATEGORY_CENTROIDS = calculate_category_centroids()

def get_predicted_category(new_embedding: np.ndarray) -> str:
    """
    Predicts the category of a new item by finding which category centroid 
    its embedding is closest to (using Cosine Similarity).
    """
    if not new_embedding.size:
        return "Unknown"
        
    best_match = {"category": "Unknown", "score": -1}
    
    # Reshape the new embedding for similarity calculation
    new_embedding_2d = new_embedding.reshape(1, -1)

    for cat, centroid_vector in CATEGORY_CENTROIDS.items():
        # Calculate similarity between the new item and the category average
        centroid_vector_2d = centroid_vector.reshape(1, -1)
        similarity = cosine_similarity(new_embedding_2d, centroid_vector_2d)[0][0]
        
        if similarity > best_match["score"]:
            best_match["score"] = similarity
            best_match["category"] = cat
            
    # Set a minimum confidence threshold to avoid random guesses
    if best_match["score"] > 0.4: # 40% similarity confidence required
        return best_match["category"]
    else:
        return "Unknown"

if __name__ == '__main__':
    # Test function: check which category the average 'Topwear' item belongs to
    if 'Topwear' in CATEGORY_CENTROIDS:
        test_emb = CATEGORY_CENTROIDS['Topwear']
        predicted = get_predicted_category(test_emb)
        print(f"\nSelf-test: Topwear centroid predicts as: {predicted}")
    else:
        print("\nRun process_data.py on clean data first to generate embeddings for centroids.")