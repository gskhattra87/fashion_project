import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
import numpy as np

# --- CONFIGURATION ---
EMBEDDING_DIMENSION = 2048
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_RESOLUTION = (300, 300) # Minimum required width and height
print(f"Using device: {DEVICE}")

# 1. Define Image Preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 2. Load Model
def load_feature_extractor():
    print("Loading ResNet-50 model...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    return model.to(DEVICE)

FEATURE_EXTRACTOR = load_feature_extractor()

# 3. Main Embedding Function (UPDATED for Resolution Check)
def get_embedding(image_source):
    """
    Extracts feature embedding from an image, enforcing minimum resolution.
    
    Args:
        image_source: Can be a URL string (http...) OR raw image bytes.
    """
    try:
        image = None
        
        # Determine image source and load
        if isinstance(image_source, str):
            # Case A: URL (Used by process_data.py)
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) FashionAI/1.0'}
            response = requests.get(image_source, headers=headers, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
        elif isinstance(image_source, bytes):
            # Case B: Bytes (Used by /recommend endpoint)
            image = Image.open(BytesIO(image_source)).convert('RGB')
            
        else:
            raise ValueError(f"Unsupported input type {type(image_source)}")

        # --- HIGH RESOLUTION CHECK ---
        width, height = image.size
        if width < MIN_RESOLUTION[0] or height < MIN_RESOLUTION[1]:
            print(f"Image rejected: Resolution {width}x{height} is below {MIN_RESOLUTION[0]}x{MIN_RESOLUTION[1]}.")
            return None
        
        # Process the image
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            embedding = FEATURE_EXTRACTOR(img_tensor)
        
        # Flatten and convert to numpy
        return embedding.flatten().cpu().numpy()
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None