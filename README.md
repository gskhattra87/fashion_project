
👗 # Fashion Trend Detection & AI Stylist System

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Python Version](https://img.shields.io/badge/Python-3.x-blue)
![Library](https://img.shields.io/badge/Library-pandas%20%7C%20fastapi%20%7C%20sqlalchemy%20%7C%20sklearn--Learn-orange)


Project Trend Setters is an end-to-end AI application designed to detect real-time fashion trends from social media and act as a personal AI Stylist. It leverages Computer Vision (ResNet-50) to understand clothing aesthetics and Machine Learning (K-Means) to identify emerging styles.

✨ # Key Features

🧠 # AI & Computer Vision

Visual Similarity Search: Upload an image to find visually similar items in the database.

AI Stylist ("Complete the Look"): Intelligent outfit generation that suggests matching Bottoms, Shoes, and Accessories for a given Top (and vice-versa).

Contextual Recommendations: Filter suggestions based on occasions (Casual, Formal, Sporty).

Trend Clustering: Automatically groups thousands of images into distinct "Trend Clusters" based on visual features.

Hybrid Classification: Automatically categorizes items (Topwear, Footwear, etc.) using a combination of visual embeddings and hashtag heuristics.

💻 # User Interface (PWA)

Progressive Web App (PWA): Installable on mobile/desktop with offline capability.

Virtual Wardrobe: Save favorite items to a local wishlist.

Trend Deep Dive: Explore specific trend clusters interactively.

Full Image Viewer: High-resolution zoom for details.

🛡️ # Admin Dashboard

Social Media Scraper: Automated scraping for Pinterest to build the dataset.

Bulk Data Management: Tools to bulk-edit categories or genders for thousands of items at once.

AI Pipeline Control: Trigger feature extraction and trend detection scripts directly from the UI.

Analytics: Charts visualizing category distribution and trend growth over time.

🛠️ # Tech Stack

Backend: Python, FastAPI, Uvicorn, SQLAlchemy

Database: MySQL

AI/ML: PyTorch (ResNet-50), Scikit-Learn (K-Means, Cosine Similarity), Pillow

Scraping: Selenium, BeautifulSoup4

Frontend: HTML5, Bootstrap 5, Chart.js, Vanilla JS (Service Workers for PWA)

🚀 # Installation & Setup

1. Clone the Repository

git clone [https://github.com/gskhattra87/fashion_project.git](https://github.com/gskhattra87/fashion_project.git)
cd fashion-trend-system


2. Install Dependencies

Ensure you have Python 3.9+ installed.

pip install -r requirements.txt


3. Database Configuration

Ensure MySQL is running (Default port: 3306 or custom 9090).

Create a database named fashion_trend_db.

Update the DATABASE_URL in main.py, process_data.py, trend_detector.py if your credentials differ from root:@localhost:9090.

4. Run the Application

Start the backend server (which also serves the frontend):

uvicorn main:app --reload


Access the app at: http://localhost:8000

📖 # Usage Guide

A. Admin Setup (First Run)

Go to http://localhost:8000/admin.html.

Login with default credentials: admin / secret.

Ingest Data: Use the Scraper tab to fetch data from Pinterest (e.g., #streetwear).

Process Data: Go to the AI Pipeline tab and click "Run process_data.py" to generate embeddings and classify items.

Detect Trends: Click "Run trend_detector.py" to cluster the items.

B. User Experience

Go to the Home Page (index.html).

Explore Trends: Click on any Trend Card to see all items in that style cluster.

Get Recommendations: Click "Complete the Look", upload a photo of a shirt, and see the AI suggest matching pants and shoes.

Install App: Click the "Install App" button in the nav bar to install as a PWA.

📂 # Project Structure

├── main.py                 # Core FastAPI backend & API endpoints

├── process_data.py         # AI Pipeline: Feature extraction & Classification

├── trend_detector.py       # ML Pipeline: K-Means Clustering

├── feature_extractor.py    # ResNet-50 Model definition

├── scraper.py              # Selenium Scraper logic

├── category_model.py       # Category prediction logic

├── fix_categories.py       # Database utility script

├── index.html              # User Frontend (PWA)

├── admin.html              # Admin Dashboard

├── dashboard.html          # Analytics View

├── gallery.html            # Image Management View

├── sw.js                   # Service Worker for PWA

├── manifest.json           # PWA Manifest

└── requirements.txt        # Python dependencies


🔮 # Future Roadmap

Integration with Object Detection (YOLO) for precise multi-item cropping.

Virtual Try-On (VTON) integration using GANs.

Real-time Price Tracking and "Shop the Look" affiliate links.

Developed by [Gurpreet Singh, Ritabrata Chakraborty]
