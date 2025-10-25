import flask
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize # Still needed for the user vector
from nltk.stem import WordNetLemmatizer
import re
import os
import requests
import io
import nltk
import logging 

# --- 1. Initialize App ---
app = Flask(__name__)
gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(logging.INFO)

# --- 2. Define Global Variables ---
recipe_vectors_norm = None # Renamed
ingredient_map = None
df_info = None
ingredient_columns = None

# --- 3. Downloader Function ---
def download_file(url):
    try:
        session = requests.Session()
        response = session.get(url, stream=True)
        response.raise_for_status() 
        return io.BytesIO(response.content)
    except Exception as e:
        app.logger.error(f"Error downloading {url}: {e}")
        return None

# --- 4. Model Loading Function ---
def load_models():
    global recipe_vectors_norm, ingredient_map, df_info, ingredient_columns
    
    # --- !! PASTE YOUR 3 HUGGING FACE URLS HERE !! ---
    # --- !! VECTORS_URL must be the *new* file !! ---
    VECTORS_URL = "https://huggingface.co/Bhaveshrajput/Recipe-recommender/resolve/main/recipe_vectors_norm.pkl"
    COLUMNS_URL = "https://huggingface.co/Bhaveshrajput/Recipe-recommender/resolve/main/ingredient_columns.pkl"
    INFO_URL    = "https://huggingface.co/Bhaveshrajput/Recipe-recommender/resolve/main/recipes_info.csv"

    try:
        # --- NLTK Fix for Render ---
        app.logger.info("Downloading NLTK data to /tmp/nltk_data...")
        nltk.download('wordnet', download_dir='/tmp/nltk_data')
        nltk.download('omw-1.4', download_dir='/tmp/nltk_data')
        nltk.data.path.append('/tmp/nltk_data')
        app.logger.info("NLTK data downloaded.")
        
        app.logger.info("Downloading all 3 data files from Hugging Face...")
        vectors_file = download_file(VECTORS_URL)
        columns_file = download_file(COLUMNS_URL)
        info_file    = download_file(INFO_URL)

        if not all([vectors_file, columns_file, info_file]):
            raise FileNotFoundError("Failed to download one or more files from Hugging Face.")
        
        app.logger.info("All files downloaded. Loading models...")
        # Load the new pre-normalized file
        recipe_vectors_norm = joblib.load(vectors_file) 
        ingredient_columns  = joblib.load(columns_file)
        df_info             = pd.read_csv(info_file).fillna('N/A')
        
        ingredient_map = {name: i for i, name in enumerate(ingredient_columns)}
        
        app.logger.info("All ML/data files loaded successfully!")
        app.logger.info(f"Recipe vectors shape: {recipe_vectors_norm.shape}")
        app.logger.info(f"Recipe info shape: {df_info.shape}")
        
    except Exception as e:
        app.logger.error(f"--- FATAL ERROR: Could not load ML models --- {e}")
            
# --- 5. Input Cleaning Function ---
stop_words = set([
    'cup', 'cups', 'oz', 'ounce', 'ounces', 'tbsp', 'tablespoon', 'tablespoons',
    'tsp', 'teaspoon', 'teaspoons', 'g', 'kg', 'ml', 'l', 'lb', 'lbs',
    'chopped', 'diced', 'sliced', 'minced', 'fresh', 'large', 'medium', 'small'
])

def clean_user_input(text):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(r'[^a-z\s,]', '', text) 
    words = re.split(r'[\s,]+', text)
    cleaned_words = []
    for word in words:
        if word and word not in stop_words:
            lemmatized_word = lemmatizer.lemmatize(word) 
            if lemmatized_word not in stop_words and len(lemmatized_word) > 2:
                cleaned_words.append(lemmatized_word)
    return cleaned_words

# --- 6. CALL THE MODEL LOADER ---
app.logger.info("Starting app, calling load_models()...")
load_models()

# --- 7. Define App Routes (Webpages) ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_recipes', methods=['POST'])
def get_recipes_api():
    app.logger.info("Received request for /get_recipes")
    if recipe_vectors_norm is None or ingredient_map is None:
        app.logger.error("Models are not loaded, returning 503 error.")
        return jsonify({"error": "Models are not loaded yet. Please try again in a moment."}), 503
    try:
        data = request.json
        user_ingredients_str = data.get('ingredients', '')
        app.logger.info(f"Input ingredients: {user_ingredients_str}") 
        
        user_words = clean_user_input(user_ingredients_str)
        app.logger.info(f"Cleaned words: {user_words}") 
        
        user_vector = np.zeros((1, len(ingredient_map)))
        found_ingredients = []
        for word in user_words:
            if word in ingredient_map:
                index = ingredient_map[word]
                user_vector[0, index] = 1.0
                found_ingredients.append(word)
        
        app.logger.info(f"Found ingredients in database: {found_ingredients}") 
        if not found_ingredients:
            return jsonify([]) 

        # --- FINAL, MEMORY-EFFICIENT CALCULATION ---
        app.logger.info("--- Normalizing user_vector... ---")
        user_vector_norm = normalize(user_vector)
        
        app.logger.info("--- Calculating dot product... ---")
        # No more crashing! We just multiply the user vector by the pre-normalized vectors
        similarity_scores = np.dot(user_vector_norm, recipe_vectors_norm.T) 
        
        app.logger.info("--- Calculation FINISHED. ---")
        # --- END CALCULATION ---

        scores = similarity_scores[0]
        top_matches_indices = scores.argsort()[-5:][::-1]

        results = []
        for index in top_matches_indices:
            if scores[index] > 0.05: 
                recipe_info = df_info.iloc[index].to_dict()
                recipe_info['score'] = float(scores[index])
                results.append(recipe_info)
        
        app.logger.info(f"Returning {len(results)} recipes.") 
        return jsonify(results)
        
    except Exception as e:
        app.logger.error(f"--- CRASH IN get_recipes_api ---: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

# --- 8. Run the App (for local testing only) ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
