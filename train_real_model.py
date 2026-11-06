"""
LeafGuard AI - Real Model Training & OOD Calibration Engine
Uses MobileNetV2 transfer learning on the PlantVillage Potato dataset
and calibrates cosine-distance-based OOD rejection using real OOD datasets.
"""

import os
import glob
import json
import random
import time
import cv2
import numpy as np

# Ensure root directory is on sys.path
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

CLASSES = ["Potato Healthy", "Potato Early Blight", "Potato Late Blight"]
FOLDER_MAPPING = {
    "Potato Healthy": "Healthy",
    "Potato Early Blight": "Early_Blight",
    "Potato Late Blight": "Late_Blight"
}

BASE_DIR = config.BASE_DIR
POTATO_DATASET_DIR = os.path.join(BASE_DIR, "datasets", "Potato", "PLD_3_Classes_256")
OOD_DATASET_DIR = os.path.join(BASE_DIR, "datasets", "ODD")


def load_and_preprocess_images(file_paths, target_size=(224, 224), max_samples=None):
    """Loads images, resizes, converts to RGB and applies MobileNetV2 preprocessing."""
    if max_samples and len(file_paths) > max_samples:
        file_paths = random.sample(file_paths, max_samples)

    images = []
    valid_paths = []
    for p in file_paths:
        try:
            img = cv2.imread(p)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            images.append(img.astype(np.float32))
            valid_paths.append(p)
        except Exception:
            continue

    if not images:
        return np.empty((0, target_size[0], target_size[1], 3), dtype=np.float32), []

    batch = preprocess_input(np.array(images, dtype=np.float32))
    return batch, valid_paths


def extract_embeddings_in_batches(model, images, batch_size=32):
    """Extracts features in manageable batches and returns L2-normalized embeddings."""
    embeddings = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        emb = model.predict(batch, verbose=0)
        embeddings.append(emb)
    
    if not embeddings:
        return np.empty((0, 1280), dtype=np.float32)
        
    all_emb = np.vstack(embeddings)
    # L2 normalize
    norms = np.linalg.norm(all_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return all_emb / norms


def main():
    print("=" * 70)
    print("LeafGuard AI — Real Model Training & OOD Calibration Pipeline")
    print("=" * 70)

    # 1. Initialize MobileNetV2 backbone
    print("\n[1/5] Loading pretrained MobileNetV2 backbone...")
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=config.INPUT_SHAPE
    )
    base_model.trainable = False
    print("  [OK] MobileNetV2 backbone loaded.")

    # 2. Extract training features from real potato dataset
    print("\n[2/5] Extracting training features from Potato Leaf Dataset...")
    train_dir = os.path.join(POTATO_DATASET_DIR, "Training")
    
    X_train_emb_list = []
    y_train_list = []
    class_train_embeddings = {cls: [] for cls in CLASSES}

    # Sample 400 images per class for balanced, rapid training
    SAMPLES_PER_CLASS = 400
    for class_idx, class_name in enumerate(CLASSES):
        folder_name = FOLDER_MAPPING[class_name]
        folder_path = os.path.join(train_dir, folder_name)
        file_paths = glob.glob(os.path.join(folder_path, "*.*"))
        file_paths = [f for f in file_paths if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  Loading '{class_name}' (available: {len(file_paths)})...")
        
        batch, valid_paths = load_and_preprocess_images(file_paths, max_samples=SAMPLES_PER_CLASS)
        embs = extract_embeddings_in_batches(base_model, batch)
        
        X_train_emb_list.append(embs)
        y_train_list.append(np.full(len(embs), class_idx, dtype=np.int32))
        class_train_embeddings[class_name] = embs
        print(f"  [OK] Extracted {len(embs)} embeddings for '{class_name}'.")

    X_train_emb = np.vstack(X_train_emb_list)
    y_train = np.concatenate(y_train_list)

    # 3. Compute Normalized Class Centroids for OOD Detection
    print("\n[3/5] Calculating in-distribution class centroids...")
    centroids = {}
    for class_name, embs in class_train_embeddings.items():
        centroid = np.mean(embs, axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm > 0:
            centroid = centroid / c_norm
        centroids[class_name] = centroid.tolist()
        print(f"  [OK] Centroid computed for '{class_name}' (L2 norm: {np.linalg.norm(centroid):.4f})")

    # 4. Train Classification Head on Top of MobileNetV2 Features
    print("\n[4/5] Training classification head with transfer learning...")
    head_input = Input(shape=(1280,), name="feature_input")
    x = Dense(256, activation='relu', name="dense_features_1")(head_input)
    x = Dropout(0.3, name="dropout_1")(x)
    x = Dense(64, activation='relu', name="dense_features_2")(x)
    head_output = Dense(3, activation='softmax', name="prediction_probabilities")(x)

    head_model = Model(inputs=head_input, outputs=head_output, name="classification_head")
    head_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train head model
    head_model.fit(
        X_train_emb, y_train,
        epochs=15,
        batch_size=32,
        validation_split=0.15,
        verbose=1
    )

    # Build end-to-end full model
    full_input = Input(shape=config.INPUT_SHAPE, name="leaf_image_input")
    full_features = base_model(full_input)
    full_output = head_model(full_features)
    full_model = Model(inputs=full_input, outputs=full_output, name="LeafGuard_MobileNetV2_Full")
    full_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Save trained model
    os.makedirs(config.AI_MODULE_DIR, exist_ok=True)
    full_model.save(config.MODEL_PATH)
    print(f"  [OK] Saved end-to-end model to {config.MODEL_PATH}")

    # Save class_names.json aligned with indices 0, 1, 2
    with open(config.CLASS_NAMES_PATH, 'w') as f:
        json.dump(CLASSES, f, indent=2)
    print(f"  [OK] Aligned class names written to {config.CLASS_NAMES_PATH}")

    # 5. Calibrate OOD Cosine Distance Threshold Using Validation & OOD Datasets
    print("\n[5/5] Calibrating OOD distance threshold using validation and real OOD data...")
    val_dir = os.path.join(POTATO_DATASET_DIR, "Validation")
    
    in_dist_distances = []
    for class_name in CLASSES:
        folder_name = FOLDER_MAPPING[class_name]
        v_paths = glob.glob(os.path.join(val_dir, folder_name, "*.*"))[:100]
        batch, _ = load_and_preprocess_images(v_paths)
        embs = extract_embeddings_in_batches(base_model, batch)
        
        for e in embs:
            # Distance to nearest centroid
            dists = [1.0 - float(np.dot(e, np.array(c))) for c in centroids.values()]
            in_dist_distances.append(min(dists))

    in_dist_distances = np.array(in_dist_distances)
    print(f"  In-Distribution Validation Distances (N={len(in_dist_distances)}):")
    print(f"    Min: {in_dist_distances.min():.4f}, Mean: {in_dist_distances.mean():.4f}, "
          f"95th percentile: {np.percentile(in_dist_distances, 95):.4f}, Max: {in_dist_distances.max():.4f}")

    # Load OOD calibration samples (cars, animals, cups, random objects)
    ood_paths = []
    for category in ["car", "animals", "cup", "random_objects"]:
        cat_dir = os.path.join(OOD_DATASET_DIR, category)
        if os.path.exists(cat_dir):
            c_files = glob.glob(os.path.join(cat_dir, "**", "*.*"), recursive=True)
            c_imgs = [f for f in c_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            sampled = random.sample(c_imgs, min(75, len(c_imgs)))
            ood_paths.extend(sampled)
            print(f"    Loaded {len(sampled)} calibration samples from OOD '{category}'.")

    ood_batch, _ = load_and_preprocess_images(ood_paths)
    ood_embs = extract_embeddings_in_batches(base_model, ood_batch)

    ood_distances = []
    for e in ood_embs:
        dists = [1.0 - float(np.dot(e, np.array(c))) for c in centroids.values()]
        ood_distances.append(min(dists))

    ood_distances = np.array(ood_distances)
    print(f"  OOD Calibration Distances (N={len(ood_distances)}):")
    print(f"    Min: {ood_distances.min():.4f}, Mean: {ood_distances.mean():.4f}, "
          f"5th percentile: {np.percentile(ood_distances, 5):.4f}, Max: {ood_distances.max():.4f}")

    # Calibrate threshold:
    # We want a threshold that covers ~96-98% of in-distribution leaves while rejecting OOD objects.
    # Set threshold between the 97th percentile of in-distribution and 5th percentile of OOD
    p97_in = float(np.percentile(in_dist_distances, 97))
    p5_ood = float(np.percentile(ood_distances, 5))
    
    # Balanced calibrated threshold
    calibrated_threshold = round((p97_in + p5_ood) / 2.0, 3)
    if calibrated_threshold < p97_in:
        calibrated_threshold = round(p97_in + 0.02, 3)

    print(f"\n  ==> Calibrated OOD Distance Threshold: {calibrated_threshold}")
    print(f"      In-Distribution Acceptance Rate (Val): {np.mean(in_dist_distances <= calibrated_threshold)*100:.1f}%")
    print(f"      OOD Rejection Rate (Calibration): {np.mean(ood_distances > calibrated_threshold)*100:.1f}%")

    # Save OOD calibration
    calibration_data = {
        "calibrated": True,
        "model_architecture": "MobileNetV2",
        "feature_dim": 1280,
        "distance_metric": "cosine",
        "threshold": calibrated_threshold,
        "classes": centroids,
        "metrics": {
            "val_in_dist_mean": round(float(in_dist_distances.mean()), 4),
            "val_in_dist_p95": round(float(np.percentile(in_dist_distances, 95)), 4),
            "ood_mean": round(float(ood_distances.mean()), 4),
            "ood_p5": round(float(np.percentile(ood_distances, 5)), 4)
        }
    }

    with open(config.OOD_THRESHOLDS_PATH, 'w') as f:
        json.dump(calibration_data, f, indent=2)
    print(f"  [OK] Saved OOD thresholds & centroids to {config.OOD_THRESHOLDS_PATH}")

    # Update config.py with the calibrated threshold
    config_path = os.path.join(BASE_DIR, "config.py")
    with open(config_path, 'r') as f:
        config_text = f.read()
    
    import re
    config_text = re.sub(
        r'OOD_DISTANCE_THRESHOLD\s*=\s*[0-9\.]+',
        f'OOD_DISTANCE_THRESHOLD = {calibrated_threshold}',
        config_text
    )
    with open(config_path, 'w') as f:
        f.write(config_text)
    print(f"  [OK] Updated config.py with OOD_DISTANCE_THRESHOLD = {calibrated_threshold}")

    print("\n" + "=" * 70)
    print("[SUCCESS] Training & OOD Calibration Completed Successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
