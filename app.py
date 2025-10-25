import flask
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import re
import os
import requests  # We need this
import io
import nltk

# --- 1. Initialize App ---
app = Flask(__name__)

# --- 2. Define Global Variables ---
recipe_vectors = None
ingredient_map = None
df_info = None
ingredient_columns = None

# --- 3. Downloader Function ---
# This simple downloader works for Hugging Face
def download_file(url):
    try:
        session = requests.Session()
        response = session.get(url, stream=True)
        response.raise_for_status() # Will error if download fails
        return io.BytesIO(response.content)
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

# --- 4. Model Loading Function ---
def load_models():
    # Tell this function to modify the GLOBAL variables
    global recipe_vectors, ingredient_map, df_info, ingredient_columns
    
    # --- !! PASTE YOUR 3 HUGGING FACE URLS HERE !! ---
    VECTORS_URL = "https://huggingface.co/Bhaveshrajput/Recipe-recommender/resolve/main/recipe_vectors.pkl"
    COLUMNS_URL = "https://huggingface.co/Bhaveshrajput/Recipe-recommender/resolve/main/ingredient_columns.pkl"
    INFO_URL    = "https://huggingface.co/Bhaveshrajput/Recipe-recommender/resolve/main/recipes_info.csv"

    try:
        # --- NLTK Fix for Render ---
        print("Downloading NLTK data to /tmp/nltk_data...")
        nltk.download('wordnet', download_dir='/tmp/nltk_data')
        nltk.download('omw-1.4', download_dir='/tmp/nltk_data')
        nltk.data.path.append('/tmp/nltk_data')
        print("NLTK data downloaded.")
        # ---------------------------
        
        # --- Download all 3 files from Hugging Face ---
        print("Downloading all 3 data files from Hugging Face...")
        vectors_file = download_file(VECTORS_URL)
        columns_file = download_file(COLUMNS_URL)
        info_file    = download_file(INFO_URL)

        if not all([vectors_file, columns_file, info_file]):
            raise FileNotFoundError("Failed to download one or more files from Hugging Face.")
        
        print("All files downloaded. Loading models...")
        recipe_vectors     = joblib.load(vectors_file)
        ingredient_columns = joblib.load(columns_file)
        df_info            = pd.read_csv(info_file).fillna('N/A')
        
        # Create the quick-lookup map
        ingredient_map = {name: i for i, name in enumerate(ingredient_columns)}
        
        print("All ML/data files loaded successfully!")
        print(f"Recipe vectors shape: {recipe_vectors.shape}")
        print(f"Recipe info shape: {df_info.shape}")
        
    except Exception as e:
        print(f"--- FATAL ERROR: Could not load ML models ---")
        print(e)
            
# --- 5. Input Cleaning Function ---
lemmatizer = WordNetLemmatizer()
stop_words = set([
    'cup', 'cups', 'oz', 'ounce', 'ounces', 'tbsp', 'tablespoon', 'tablespoons',
    'tsp', 'teaspoon', 'teaspoons', 'g', 'kg', 'ml', 'l', 'lb', 'lbs',
    'chopped', 'diced', 'sliced', 'minced', 'fresh', 'large', 'medium', 'small'
])

def clean_user_input(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s,]', '', text) # Remove punctuation/numbers
    words = re.split(r'[\s,]+', text)
    cleaned_words = []
    for word in words:
        if word and word not in stop_words:
            lemmatized_word = lemmatizer.lemmatize(word) # 'tomatoes' -> 'tomato'
            if lemmatized_word not in stop_words and len(lemmatized_word) > 2:
                cleaned_words.append(lemmatized_word)
    return cleaned_words

# --- 6. CALL THE MODEL LOADER ---
# Gunicorn runs this code when it imports the file.
print("Starting app, calling load_models()...")
load_models()

# --- 7. Define App Routes (Webpages) ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_recipes', methods=['POST'])
def get_recipes_api():
    if recipe_vectors is None or ingredient_map is None:
        return jsonify({"error": "Models are not loaded yet. Please try again in a moment."}), 503
    try:
        data = request.json
        user_ingredients_str = data.get('ingredients', '')
        user_words = clean_user_input(user_ingredients_str)
        user_vector = np.zeros((1, len(ingredient_map)))
        found_ingredients = []
        for word in user_words:
            if word in ingredient_map:
                index = ingredient_map[word]
                user_vector[0, index] = 1.0
                found_ingredients.append(word)
        if not found_ingredients:
            return jsonify([]) 
        similarity_scores = cosine_similarity(user_vector, recipe_vectors)
        scores = similarity_scores[0]
        top_matches_indices = scores.argsort()[-5:][::-1]
        results = []
        for index in top_matches_indices:
            if scores[index] > 0.05: 
                recipe_info = df_info.iloc[index].to_dict()
                recipe_info['score'] = float(scores[index])
                results.append(recipe_info)
        return jsonify(results)
    except Exception as e:
        print(f"Error in get_recipes_api: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

# --- 8. Run the App (for local testing only) ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
