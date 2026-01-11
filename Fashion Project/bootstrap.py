import os
import requests
import sys

# -------------------------------------------------
# Base directory (where this script lives)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Existing models folder inside project
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Ensure models folder exists
os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------------------------------------
# Configuration
# -------------------------------------------------
RESOURCES = {
    'Embeddings': {
        'url': 'https://embeddingmodelfile.s3.ap-south-1.amazonaws.com/embedding.pkl',
        'local_path': os.path.join(MODELS_DIR, 'embedding.pkl')
    },
    'Filenames': {
        'url': 'https://embeddingmodelfile.s3.ap-south-1.amazonaws.com/filenames.pkl',
        'local_path': os.path.join(MODELS_DIR, 'filenames.pkl')
    },
    'ResNet50':{
        'url':'',
        'local_path':os.path.join(MODELS_DIR, 'ResNet50_feature_extractor2.keras')
    }
}

# -------------------------------------------------
# Download function
# -------------------------------------------------
def download_and_save_resource(url: str, local_path: str, name: str) -> None:
    print(f"\n--- Processing {name} ---")

    if os.path.exists(local_path):
        print(f"✅ {name} already exists at {local_path}. Skipping download.")
        return

    print(f"⬇️ Downloading {name}...")
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ Download failed for {name}: {e}", file=sys.stderr)
        return

    try:
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"💾 Saved {name} → {local_path}")
    except IOError as e:
        print(f"❌ File save error for {name}: {e}", file=sys.stderr)

# -------------------------------------------------
# Entry point
# -------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting resource download...")
    print("📂 Models directory:", MODELS_DIR)

    for name, info in RESOURCES.items():
        download_and_save_resource(info['url'], info['local_path'], name)

    print("\n✅ Download process complete.")