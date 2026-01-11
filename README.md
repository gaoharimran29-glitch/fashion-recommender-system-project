# Fashion Product Recommendation System  
### Deep Learning + Similarity Search + Cloud Storage

---

## Project Site link
- https://imranx.dpdns.org/
- https://www.imranx.dpdns.org/

## Overview
This project is an **AI-powered Fashion Product Recommendation System** that suggests visually similar fashion products based on an uploaded image.

The system leverages:
- **Deep Learning (ResNet50)** for image feature extraction  
- **Vector similarity search (KNN / Annoy-ready)** for recommendations  
- **AWS S3** for scalable image storage  
- **Flask** for backend deployment  

It is designed to **scale to large image datasets (23GB+)** while remaining efficient and maintainable.

---

## Problem Statement
Traditional fashion recommendation systems depend heavily on:
- Manual tagging  
- Text-based filters  
- User metadata  

These approaches fail when:
- Metadata is missing or inconsistent  
- Users want *visual similarity*, not just category similarity  

**Solution:** Recommend products using **visual features extracted by a deep learning model**.

---

## System Architecture

User Upload Image
↓
ResNet50 Feature Extraction
↓
Feature Normalization
↓
Similarity Search (KNN / Annoy)
↓
Top-N Similar Fashion Items
↓
Images Served from AWS S3


---

### Download Dataset
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-dataset")

print("Path to dataset files:", path)
```

### First train images on Resnet50

```python
import tensorflow as tf
import numpy as np
import os
import pickle
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm

# ==============================
# 1. BUILD MODEL
# ==============================

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# ==============================
# 2. LOAD FILENAMES (FIXED)
# ==============================

IMG_DIR = 'C:/Users/hp/.cache/kagglehub/datasets/paramaggarwal/fashion-product-images-small/versions/1/images/'

image_files = os.listdir(IMG_DIR)

# what we PROCESS (full paths)
image_paths = [os.path.join(IMG_DIR, fname) for fname in image_files]

# what we STORE (portable)
filenames = image_files

print("Total images:", len(filenames))
print("Sample filenames:", filenames[:5])

# ==============================
# 3. FAST tf.data PIPELINE
# ==============================

BATCH_SIZE = 128  # You can try 64 if GPU memory is low

def load_and_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224,224))
    img = preprocess_input(img)
    return img

dataset = tf.data.Dataset.from_tensor_slices(image_paths)
dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# ==============================
# 4. FEATURE EXTRACTION (FAST)
# ==============================

print("Extracting features...")
features = model.predict(dataset, verbose=1)

# Normalize embeddings
features = features / norm(features, axis=1, keepdims=True)

print("Feature shape:", features.shape)

# ==============================
# 5. SAVE OUTPUTS
# ==============================

pickle.dump(features, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))

print("✅ Embeddings and filenames saved successfully")
```
- filenames.pkl contains filename and embeddings.pkl contain vector reprentation of images

### Cleaning of styles.csv file containing product info

```python
import pandas as pd

df = pd.read_csv(r"styles.csv" , on_bad_lines='skip')
df['baseColour'].fillna(df['baseColour'].mode()[0] , inplace=True)
df['season'].fillna(df['season'].mode()[0] , inplace=True)
df['year'].fillna(df['year'].mode()[0] , inplace=True)
df['usage'].fillna(df['usage'].mode()[0] , inplace=True)
df.dropna(subset=['productDisplayName'] , inplace=True)

print(df.head())
print(df.isnull().sum())
print(df.shape)

print(df.columns)
```
- Then create a new folder for project.
- Create a folder named model, templates, and a file app.py
- In model folder add cleaned styles.csv
- I hosted embedding.pkl , filenames.pkl , Resnet50.keras on s3 and then downloading it using code.
- In template folder create a html file.
- You can get codes in above Fashion Project Folder

- So the basic architecture is that we already calculated vectors represntation for each image using Resnet 50.
- For new image we again calculate vectors representation using Resnet50
- Then we use this vector and predict nearest 6 vectors from whole using K nearest neighbour model usng cosine similarity.
- Then we find most nearets 6 vectors filenames and display them.
- I also used clous AWS S3 to host images.
- Used docker for ease in deployment on AWS EC2
- Then Route 53 for domain mapping with EC2 public ip and nginx certbot to make it secure.

# Project image
<img width="1913" height="944" alt="fashion" src="https://github.com/user-attachments/assets/5d288f48-8384-4f42-b848-f6baa6f2b219" />

## Tech Stack

### Deep Learning
- TensorFlow / Keras  
- ResNet50 (ImageNet pretrained)  
- Global Max Pooling  

### Machine Learning
- KNN (baseline similarity search)  
- Annoy (for optimized large-scale retrieval)  

### Backend
- Flask  
- NumPy, Pandas, Pickle  

### Cloud
- AWS S3  
- IAM (secure access control)  

---


---

## Feature Extraction Pipeline

1. Input image resized to **224×224**
2. Preprocessed using `preprocess_input`
3. Passed through **ResNet50 (without top layers)**
4. Global Max Pooling applied
5. Feature vector normalized using **L2 normalization**

## Recommendation Engine

### Baseline: KNN
- Metric: **Cosine Similarity**
- Brute-force search
- Suitable for **small to medium-sized datasets**

### ptimization: Annoy
- Approximate Nearest Neighbor (ANN) search
- Faster retrieval for **large-scale datasets**
- Memory-efficient indexing

Enables **near real-time recommendations at scale**.

---

## Cloud Storage with AWS S3

### Why AWS S3?
- Local hosting of **23GB+ images** is impractical
- GitHub has strict storage limits
- AWS S3 provides:
  - Highly durable storage
  - Public / private access control
  - Cost-efficient scalability

### Image Access Strategy
- Images are uploaded **once** to S3
- Model returns **image filenames**
- Frontend loads images via **S3 public URLs**


Example:
```text
https://myfashion-images-recommender.s3.ap-south-1.amazonaws.com/12345.jpg
```
### Dataset used

- Fashion product images

- Metadata: category, color, season, usage

- Dataset link :- https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset

## Contact

Gaohar Imran
📍 India
💼 Aspiring ML / Deep Learning Engineer
🔗 LinkedIn: https://www.linkedin.com/in/gaohar-imran-5a4063379/
