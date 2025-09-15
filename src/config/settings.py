"""
Configuration settings for BISINDO Alphabet Recognition.
All configuration constants and settings are defined here.
"""

from pathlib import Path
from typing import Dict, Any

# =============================================================================
# DATASET PATHS
# =============================================================================

DATASET_BASE_PATH = Path("dataset")
RAW_DATA_PATH = DATASET_BASE_PATH / "raw"
PROCESSED_DATA_PATH = DATASET_BASE_PATH / "processed"

# Organized processed subdirectories
AUGMENTED_DATA_PATH = PROCESSED_DATA_PATH / "01_augmented"
LANDMARKS_DATA_PATH = PROCESSED_DATA_PATH / "02_landmarks"
FEATURES_DATA_PATH = PROCESSED_DATA_PATH / "03_features"
MODELS_DATA_PATH = PROCESSED_DATA_PATH / "04_models"
METADATA_PATH = PROCESSED_DATA_PATH / "05_metadata"

# =============================================================================
# IMAGE PROCESSING SETTINGS
# =============================================================================

IMAGE_SIZE = (224, 224, 3)  # (height, width, channels)
TARGET_IMAGES_PER_CLASS = 50
RANDOM_SEED = 42

# =============================================================================
# MEDIAPIPE SETTINGS
# =============================================================================

MEDIAPIPE_CONFIG = {
    "static_image_mode": True,
    "max_num_hands": 2,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "model_complexity": 1
}

# =============================================================================
# AUGMENTATION SETTINGS
# =============================================================================

AUGMENTATION_CONFIG = {
    "target_images_per_class": TARGET_IMAGES_PER_CLASS,
    "random_seed": RANDOM_SEED,
    "use_imgaug": True,  # Fallback to OpenCV if imgaug fails
    "augmentation_techniques": {
        "rotation": {"min_angle": -15, "max_angle": 15},
        "translation": {"x_percent": (-0.1, 0.1), "y_percent": (-0.1, 0.1)},
        "scaling": {"min_scale": 0.9, "max_scale": 1.1},
        "brightness": {"min_multiply": 0.8, "max_multiply": 1.2},
        "contrast": {"min_alpha": 0.8, "max_alpha": 1.2},
        "noise": {"scale": (0, 0.05)},
        "blur": {"sigma": (0, 0.5)}
    }
}

# =============================================================================
# FEATURE EXTRACTION SETTINGS
# =============================================================================

FEATURE_EXTRACTION_CONFIG = {
    "normalize_features": True,
    "feature_selection": True,
    "n_features": 50,
    "use_pca": False,
    "pca_components": 30,
    "random_state": RANDOM_SEED,
    "geometric_features": {
        "distances": 10,  # Number of distance features per hand
        "angles": 15,     # Number of angle features per hand
        "area_perimeter": 2,  # Area and perimeter features per hand
        "finger_lengths": 5,   # Finger length features per hand
        "orientation": 3       # Orientation features per hand
    },
    "statistical_features": {
        "per_coordinate": 4,  # mean, std, min, max per coordinate (x, y, z)
        "per_hand": 12        # 4 stats × 3 coordinates per hand
    }
}

# =============================================================================
# MACHINE LEARNING SETTINGS
# =============================================================================

ML_TRAINING_CONFIG = {
    "test_size": 0.2,
    "random_state": RANDOM_SEED,
    "cv_folds": 5,
    "n_jobs": -1,  # Use all available cores
    "use_hyperparameter_tuning": True,
    "quick_training": False,  # Set to True for faster training
    "models": {
        "SVM": {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
            "kernel": ["rbf", "poly", "sigmoid"]
        },
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },
        "KNN": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "minkowski"]
        },
        "Logistic Regression": {
            "C": [0.1, 1, 10, 100],
            "penalty": ["l1", "l2", "elasticnet"],
            "solver": ["liblinear", "saga"]
        },
        "Naive Bayes": {
            "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
        },
        "Decision Tree": {
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy"]
        },
        "Gradient Boosting": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7]
        },
        "MLP": {
            "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001, 0.01]
        },
        "LDA": {
            "solver": ["svd", "lsqr", "eigen"],
            "shrinkage": [None, "auto", 0.1, 0.5, 0.9]
        },
        "QDA": {
            "reg_param": [0.0, 0.1, 0.5, 1.0]
        }
    }
}

# =============================================================================
# FILE NAMING CONVENTIONS
# =============================================================================

FILE_NAMING = {
    "images": "{type}_images.npy",
    "labels": "{type}_labels.npy", 
    "class_names": "{type}_class_names.json",
    "stats": "{type}_stats.json",
    "extraction_stats": "{type}_extraction_stats.json",
    "metadata": "{type}_metadata.json",
    "model": "{model_name}.pkl",
    "transformer": "feature_{transformer_name}.pkl"
}

# =============================================================================
# LOGGING SETTINGS
# =============================================================================

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "logs/bisindo.log"
}

# =============================================================================
# VALIDATION SETTINGS
# =============================================================================

VALIDATION_CONFIG = {
    "min_images_per_class": 1,
    "max_images_per_class": 1000,
    "min_landmark_confidence": 0.5,
    "min_feature_quality": 0.7
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_file_path(data_type: str, file_type: str, **kwargs) -> Path:
    """
    Get standardized file path based on data type and file type.
    
    Args:
        data_type: Type of data (augmented, landmarks, features, models)
        file_type: Type of file (images, labels, class_names, etc.)
        **kwargs: Additional parameters for file naming
        
    Returns:
        Path: Standardized file path
    """
    path_mapping = {
        "augmented": AUGMENTED_DATA_PATH,
        "landmarks": LANDMARKS_DATA_PATH,
        "features": FEATURES_DATA_PATH,
        "models": MODELS_DATA_PATH,
        "metadata": METADATA_PATH
    }
    
    base_path = path_mapping.get(data_type, PROCESSED_DATA_PATH)
    
    if file_type in FILE_NAMING:
        filename = FILE_NAMING[file_type].format(**kwargs)
        return base_path / filename
    else:
        return base_path / f"{data_type}_{file_type}"

def ensure_directories() -> None:
    """Ensure all required directories exist."""
    directories = [
        DATASET_BASE_PATH,
        RAW_DATA_PATH,
        PROCESSED_DATA_PATH,
        AUGMENTED_DATA_PATH,
        LANDMARKS_DATA_PATH,
        FEATURES_DATA_PATH,
        MODELS_DATA_PATH,
        METADATA_PATH
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
