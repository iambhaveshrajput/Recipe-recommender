import flask
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import re
import os
import requests
import io
import nltk 

app = Flask(__name__)

recipe_vectors = None
ingredient_map = None
df_info = None
ingredient_columns = None

def download_gdrive_file(url):
    try:
        file_id = url.split('/d/')[1].split('/')[0]
        download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
        session = requests.Session()
        response = session.get(download_url, stream=True)
        response.raise_for_status() 
        return io.BytesIO(response.content)
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

def load_models():
    global recipe_vectors, ingredient_map, df_info, ingredient_columns
    
    VECTORS_URL = "https://drive.google.com/file/d/1DKHgVycWj9KMsKfBpMf-rzu_hEtS41fV/view?usp=drivesdk"
    COLUMNS_URL = "https://drive.google.com/file/d/1HtQai-oIxKkSDLJx6C853Ww5ZPcSXGIp/view?usp=drivesdk"
    INFO_URL = "https://drive.google.com/file/d/1UB_mcEPOY79KcLughKTo5EJEjuFPcrB4/view?usp=drivesdk"

    try:
        print("Downloading NLTK data to /tmp/nltk_data...")
        nltk.download('wordnet', download_dir='/tmp/nltk_data')
        nltk.download('omw-1.4', download_dir='/tmp/nltk_data')
        nltk.data.path.append('/tmp/nltk_data')
        print("NLTK data downloaded.")

        print("Downloading ML files from Google Drive...")
        
        vectors_file = download_gdrive_file(VECTORS_URL)
        columns_file = download_gdrive_file(COLUMNS_URL)
        info_file = download_gdrive_file(INFO_URL)
        
        if not all([vectors_file, columns_file, info_file]):
            raise FileNotFoundError("Failed to download one or more files from GDrive.")

        print("Files downloaded. Loading models...")
        recipe_vectors = joblib.load(vectors_file)
        ingredient_columns = joblib.load(columns_file)
        df_info = pd.read_csv(info_file).fillna('N/A')
        
        ingredient_map = {name: i for i, name in enumerate(ingredient_columns)}
        
        print("All ML/data files loaded successfully!")
        print(f"Recipe vectors shape: {recipe_vectors.shape}")
        print(f"Recipe info shape: {df_info.shape}")
        
    except Exception as e:
        print(f"--- FATAL ERROR: Could not load ML models ---")
        print(e)
            
lemmatizer = WordNetLemmatizer()
stop_words = set([
    'cup', 'cups', 'oz', 'ounce', 'ounces', 'tbsp', 'tablespoon', 'tablespoons',
    'tsp', 'teaspoon', 'teaspoons', 'g', 'kg', 'ml', 'l', 'lb', 'lbs',
    'chopped', 'diced', 'sliced', 'minced', 'fresh', 'large', 'medium', 'small'
])

def clean_user_input(text):
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

if __name__ == '__main__':
    load_models()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
