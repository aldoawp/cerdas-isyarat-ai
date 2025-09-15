"""
Configuration module for BISINDO Alphabet Recognition.
This module contains all configuration settings and constants.
"""

from .settings import (
    # Dataset paths
    DATASET_BASE_PATH,
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    AUGMENTED_DATA_PATH,
    LANDMARKS_DATA_PATH,
    FEATURES_DATA_PATH,
    MODELS_DATA_PATH,
    METADATA_PATH,
    
    # Image processing settings
    IMAGE_SIZE,
    TARGET_IMAGES_PER_CLASS,
    RANDOM_SEED,
    
    # MediaPipe settings
    MEDIAPIPE_CONFIG,
    
    # Feature extraction settings
    FEATURE_EXTRACTION_CONFIG,
    
    # ML training settings
    ML_TRAINING_CONFIG,
    
    # Augmentation settings
    AUGMENTATION_CONFIG,
    
    # Validation settings
    VALIDATION_CONFIG,
    
    # File naming
    FILE_NAMING,
    
    # Logging settings
    LOGGING_CONFIG,
    
    # Utility functions
    ensure_directories,
    get_file_path,
)

__all__ = [
    'DATASET_BASE_PATH',
    'RAW_DATA_PATH', 
    'PROCESSED_DATA_PATH',
    'AUGMENTED_DATA_PATH',
    'LANDMARKS_DATA_PATH',
    'FEATURES_DATA_PATH',
    'MODELS_DATA_PATH',
    'METADATA_PATH',
    'IMAGE_SIZE',
    'TARGET_IMAGES_PER_CLASS',
    'RANDOM_SEED',
    'MEDIAPIPE_CONFIG',
    'FEATURE_EXTRACTION_CONFIG',
    'ML_TRAINING_CONFIG',
    'AUGMENTATION_CONFIG',
    'VALIDATION_CONFIG',
    'FILE_NAMING',
    'LOGGING_CONFIG',
    'ensure_directories',
    'get_file_path',
]
