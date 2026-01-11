from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from PIL import Image
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model
import os
import pickle
import sys 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# --- 1. LOCAL DATA LOADING ---

def load_local_pickle(local_path: str, name: str):
    """Loads a pickle file from the local filesystem."""
    try:
        with open(local_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ {name} loaded from {local_path}")
        return data
    except FileNotFoundError:
        print(f"FATAL ERROR: Required file '{local_path}' not found.", file=sys.stderr)
        print("Please run 'python bootstrap.py' first.")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL ERROR: Failed to load {name}: {e}", file=sys.stderr)
        sys.exit(1)

# MODIFIED: Paths point to current directory
features = load_local_pickle(
    os.path.join(MODELS_DIR, "embedding.pkl"),
    "Embeddings"
)

filenames = load_local_pickle(
    os.path.join(MODELS_DIR, "filenames.pkl"),
    "Filenames"
)


try:
    # Assuming cleaned_style.csv is also in the current directory
    df = pd.read_csv(
    os.path.join(MODELS_DIR, "cleaned_style.csv")
    )
except FileNotFoundError:
    print("FATAL ERROR: cleaned_style.csv not found in the current directory.", file=sys.stderr)
    sys.exit(1)

# --- 2. MODEL SETUP ---

app = Flask(__name__)
try:
    model = load_model(
    os.path.join(MODELS_DIR, "ResNet50_feature_extractor2.keras"),
    compile=False
    )
    print("Model Found")
except Exception as e:
    print(f"Error: {e}")

knn = NearestNeighbors(
    n_neighbors=6,
    metric='cosine',
    algorithm='brute'
)

knn.fit(features)

# --- 3. HELPER FUNCTIONS ---

def extract_features_from_file(file):
    file.stream.seek(0)
    img = Image.open(file.stream).convert('RGB')
    img = img.resize((224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    feature = model.predict(img, verbose=0).flatten()
    feature = feature / norm(feature)
    return feature


def get_product_info(img_path):
    img_id = int(os.path.basename(img_path).split('.')[0])
    row = df[df['id'] == img_id]
    
    if not row.empty:
        return row.iloc[0]
    return None

# --- 4. FLASK ROUTE ---

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    error_message = None

    try:
        if request.method == 'POST':
            file = request.files.get('image')

            if not file or file.filename == '':
                error_message = "Please select an image file."
                return render_template(
                    'fashionrecommender.html',
                    recommendations=recommendations,
                    error_message=error_message
                )

            query_feature = extract_features_from_file(file)
            distances, indices = knn.kneighbors([query_feature])

            seen = set()
            for idx in indices[0]:
                img_name = os.path.basename(str(filenames[idx])).strip()

                if img_name in seen:
                    continue
                seen.add(img_name)

                product = get_product_info(img_name)
                if product is None:
                    continue

                recommendations.append({
                    'image': f"https://myfashion-images-recommender.s3.ap-south-1.amazonaws.com/images/{img_name}",
                    'name': product['productDisplayName'],
                    'articletype': product['articleType'],
                    'baseColour': product['baseColour'],
                    'season': product['season'],
                    'usage': product['usage'],
                    'year': product['year'],
                })

                if len(recommendations) == 6:
                    break

    except Exception as e:
        import traceback
        traceback.print_exc()   # 🔥 shows full error in terminal
        error_message = f"Internal error: {str(e)}"


    return render_template(
        'fashionrecommender.html',
        recommendations=recommendations,
        error_message=error_message
    )


if __name__ == '__main__':
    app.run(debug=True)