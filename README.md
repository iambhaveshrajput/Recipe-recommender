# Smart Recipe Recommender 🍲

A web application that recommends recipes based on the ingredients you have on hand. Built with a Python/Flask backend, a Scikit-learn ML model, and deployed on Render.

[**Visit the Live Demo!**](https://recipe-recommender-0uoq.onrender.com)
*(Note: The app is hosted on a free tier and may take 30-60 seconds to "wake up" if it's been inactive.)*

## 📖 Description

This project uses a machine learning model to solve a common problem: "What can I cook with the ingredients I have?" A user enters a comma-separated list of ingredients, and the backend returns a list of the best-matching recipes from a large dataset, ranked by a similarity score.

To add more value, the app then dynamically fetches the full recipe instructions and an image from an external API (TheMealDB) when the user clicks on a recipe title.

## ✨ Features

* **Ingredient-Based Recommendations:** Uses a cosine similarity model to find the best recipe matches.
* **Dynamic Procedure Fetching:** Clicking a recipe title makes a live API call to TheMealDB to get the full instructions and an image.
* **Clean, Responsive UI:** A simple HTML/CSS/JavaScript frontend that's easy to use on any device.
* **Decoupled Architecture:** Built with a separate frontend and a Flask API backend.

## 🛠️ Technology Stack

* **Backend:**
    * **Python 3**
    * **Flask:** For the web server and API endpoints.
    * **Gunicorn:** As the production web server on Render.
* **Machine Learning:**
    * **Scikit-learn:** Used to build the TF-IDF vectorizer and calculate cosine similarity (pre-normalized for memory efficiency).
    * **Pandas & NumPy:** For data manipulation and vector math.
    * **NLTK:** For cleaning and lemmatizing user-inputted ingredients.
* **Frontend:**
    * **HTML5**
    * **CSS3**
    * **JavaScript (ES6+):** Uses the `fetch` API to communicate with the backend.
* **External Services:**
    * **Hugging Face Hub:** Hosts the large model files (`.pkl`, `.csv`) that are too big for GitHub.
    * **TheMealDB API:** Used to fetch recipe instructions and images.
    * **Render:** For cloud hosting and continuous deployment.

---

## 🏗️ How It Works: Architecture

The project is split into three main parts:

1.  **Machine Learning Model (Offline):**
    * The original "Epicurious" dataset from Kaggle (20,000+ recipes) was processed in a Google Colab notebook.
    * The ingredient lists were vectorized (one-hot encoded) into a large NumPy matrix.
    * This matrix was pre-normalized (L2 normalization) and saved as `recipe_vectors_norm.pkl` to prevent memory crashes on the server.
    * The data files are hosted on Hugging Face.

2.  **Flask Backend (The "Brain"):**
    * When the server starts, it downloads the model files from Hugging Face.
    * **`/get_recipes` (POST):** This endpoint receives the user's ingredients. It cleans them (using NLTK), converts them into a normalized vector, and calculates the dot product against the pre-normalized recipe matrix. This is a very fast way to get the cosine similarity. It returns the top 5 matching recipe titles.
    * **`/get_procedure` (POST):** This endpoint receives a recipe title. It calls TheMealDB API, searches for that title, and returns the instructions and image URL.

3.  **HTML/JavaScript Frontend (The "Face"):**
    * The user's ingredients are sent to the `/get_recipes` endpoint.
    * The results are dynamically rendered as a list.
    * When a user clicks a recipe title, a new JavaScript function calls the `/get_procedure` endpoint and displays the returned instructions in a drop-down section.

---

## 🚀 Deployment

This app is deployed on [Render](https://render.com/) from this GitHub repository. The server runs the `gunicorn app:app` command to start. On startup, the server downloads the ML model files from Hugging Face before it's ready to accept requests.
