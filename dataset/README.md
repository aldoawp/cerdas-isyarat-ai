# BISINDO Dataset Structure

This document describes the organized structure of the BISINDO (Bahasa Isyarat Indonesia) alphabet recognition dataset.

## 📁 Directory Structure

```
dataset/
├── raw/                          # Original raw images
│   ├── A/                        # Class A images
│   │   ├── body dot (1).jpg
│   │   ├── body dot (2).jpg
│   │   ├── body white (1).jpg
│   │   └── ...
│   ├── B/                        # Class B images
│   └── ...                       # Classes C-Z
│
└── processed/                    # Processed data (organized by pipeline stage)
    ├── 01_augmented/             # Image augmentation results
    │   ├── augmented_images.npy
    │   ├── augmented_labels.npy
    │   ├── class_names.json
    │   └── augmentation_stats.json
    │
    ├── 02_landmarks/             # Hand landmarks extraction results
    │   ├── hand_landmarks.npy
    │   ├── landmark_labels.npy
    │   ├── landmark_class_names.json
    │   ├── landmarks_metadata.json
    │   └── landmark_extraction_stats.json
    │
    ├── 03_features/              # Feature extraction results
    │   ├── extracted_features.npy
    │   ├── feature_labels.npy
    │   ├── feature_class_names.json
    │   ├── feature_names.json
    │   ├── feature_extraction_stats.json
    │   ├── feature_scaler.pkl
    │   └── feature_selector.pkl
    │
    ├── 04_models/                # Trained models and results
    │   ├── best_model.pkl
    │   ├── model_comparison.csv
    │   ├── training_results.json
    │   └── [other model files]
    │
    └── 05_metadata/              # Additional metadata and logs
        └── [metadata files]
```

## 🔄 Data Pipeline Stages

### 1. Raw Data (`raw/`)
- **Purpose**: Original BISINDO alphabet images
- **Format**: JPG images organized by class (A-Z)
- **Content**: 312 images across 26 classes
- **Structure**: Each class folder contains images with different backgrounds and hand positions

### 2. Augmented Data (`01_augmented/`)
- **Purpose**: Augmented images to increase dataset size
- **Files**:
  - `augmented_images.npy`: Augmented images array (1300, 224, 224, 3)
  - `augmented_labels.npy`: Corresponding labels array (1300,)
  - `class_names.json`: List of class names
  - `augmentation_stats.json`: Augmentation statistics
- **Content**: 1300 images (50 per class)

### 3. Landmarks Data (`02_landmarks/`)
- **Purpose**: Extracted hand landmarks using MediaPipe
- **Files**:
  - `hand_landmarks.npy`: Landmarks array (930, 126)
  - `landmark_labels.npy`: Corresponding labels array (930,)
  - `landmark_class_names.json`: List of class names
  - `landmarks_metadata.json`: Landmarks metadata
  - `landmark_extraction_stats.json`: Extraction statistics
- **Content**: 930 samples with 126 features per sample (21 landmarks × 3 coordinates × 2 hands)

### 4. Features Data (`03_features/`)
- **Purpose**: Engineered features from landmarks
- **Files**:
  - `extracted_features.npy`: Features array (930, 50)
  - `feature_labels.npy`: Corresponding labels array (930,)
  - `feature_class_names.json`: List of class names
  - `feature_names.json`: List of feature names
  - `feature_extraction_stats.json`: Feature extraction statistics
  - `feature_scaler.pkl`: Fitted StandardScaler
  - `feature_selector.pkl`: Fitted SelectKBest
- **Content**: 930 samples with 50 selected features per sample

### 5. Models Data (`04_models/`)
- **Purpose**: Trained machine learning models and results
- **Files**:
  - `best_model.pkl`: Best performing model
  - `model_comparison.csv`: Model comparison results
  - `training_results.json`: Detailed training results
  - `[model_name].pkl`: Individual trained models
- **Content**: Trained models and evaluation results

### 6. Metadata (`05_metadata/`)
- **Purpose**: Additional metadata, logs, and configuration files
- **Content**: Pipeline logs, configuration files, and other metadata

## 🛠️ Data Management

### Using the Data Manager

The `BISINDODataManager` class provides utilities for managing the organized dataset:

```python
from src.data_manager import BISINDODataManager

# Initialize data manager
dm = BISINDODataManager()

# Get dataset summary
dm.print_dataset_summary()

# Load specific data
images, labels, class_names = dm.load_augmented_data()
landmarks, labels, class_names = dm.load_landmarks_data()
features, labels, class_names = dm.load_features_data()

# Get pipeline status
status = dm.get_data_pipeline_status()
```

### File Naming Conventions

- **Images**: `[type]_images.npy` (e.g., `augmented_images.npy`)
- **Labels**: `[type]_labels.npy` (e.g., `feature_labels.npy`)
- **Class Names**: `[type]_class_names.json` (e.g., `landmark_class_names.json`)
- **Statistics**: `[type]_stats.json` or `[type]_extraction_stats.json`
- **Models**: `[model_name].pkl`
- **Transformers**: `feature_[transformer_name].pkl`

## 📊 Data Statistics

### Current Dataset Statistics
- **Raw Images**: 312 images, 26 classes
- **Augmented Images**: 1300 images, 50 per class
- **Landmarks**: 930 samples, 126 features per sample
- **Features**: 930 samples, 50 features per sample
- **Success Rate**: 71.5% landmark extraction success rate

### Data Flow
```
Raw Images (312) 
    ↓ [Augmentation]
Augmented Images (1300)
    ↓ [Landmark Extraction]
Landmarks (930 samples, 126 features)
    ↓ [Feature Engineering]
Features (930 samples, 50 features)
    ↓ [Model Training]
Trained Models
```

## 🔧 Maintenance

### Adding New Data
1. Place raw images in appropriate class folders under `raw/`
2. Run the augmentation pipeline to generate `01_augmented/` data
3. Run landmark extraction to generate `02_landmarks/` data
4. Run feature extraction to generate `03_features/` data
5. Train models to generate `04_models/` data

### Cleaning Up
- Use the data manager to check pipeline status
- Remove intermediate files if needed
- Keep backups of important model files

### Backup Strategy
- Regular backups of `04_models/` (trained models)
- Backup of `raw/` data (original images)
- Consider versioning for `processed/` data

## 🚀 Usage Examples

### Quick Dataset Overview
```bash
python src/data_manager.py
```

### Load Data for Training
```python
from src.data_manager import BISINDODataManager

dm = BISINDODataManager()
features, labels, class_names = dm.load_features_data()
print(f"Loaded {len(features)} samples with {features.shape[1]} features")
```

### Check Pipeline Status
```python
dm = BISINDODataManager()
status = dm.get_data_pipeline_status()
for step, stat in status.items():
    print(f"{step}: {stat}")
```

This organized structure makes the dataset much more maintainable and easier to work with!
