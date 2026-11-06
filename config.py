import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# File Upload Configuration
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

# AI Module File Paths
AI_MODULE_DIR = os.path.join(BASE_DIR, 'AI_Module')
MODEL_PATH = os.path.join(AI_MODULE_DIR, 'model.keras')
CLASS_NAMES_PATH = os.path.join(AI_MODULE_DIR, 'class_names.json')
OOD_THRESHOLDS_PATH = os.path.join(AI_MODULE_DIR, 'ood_thresholds.json')

# Input shape expected by MobileNetV2
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)

# Image Quality Thresholds
MIN_IMAGE_DIMENSION = 64
LAPLACIAN_VAR_MIN = 60.0        # Variance below this indicates severe blur
BRIGHTNESS_MIN = 35.0           # Mean pixel value below this is too dark
BRIGHTNESS_MAX = 235.0          # Mean pixel value above this is washed out / overexposed

# Out-of-Distribution (OOD) Safety Thresholds
# Feature cosine distance: 0.0 means identical, 1.0 means orthogonal
OOD_DISTANCE_THRESHOLD = 0.39   # Distances above this are rejected as UNKNOWN OBJECT

# Prediction Confidence Threshold
CONFIDENCE_THRESHOLD = 0.70     # Below this, accepted leaf images are marked UNCERTAIN
