# BISINDO Source Code Structure

This document describes the clean, organized structure of the BISINDO Alphabet Recognition source code.

## 📁 Directory Structure

```
src/
├── config/                     # Configuration and settings
│   ├── __init__.py
│   └── settings.py            # All configuration constants
│
├── data/                      # Data processing modules
│   ├── __init__.py
│   ├── manager.py            # Data management utilities
│   ├── loader.py             # Dataset loading and preprocessing
│   └── validator.py          # Data validation utilities
│
├── core/                      # Core processing modules
│   ├── __init__.py
│   ├── augmentation.py       # Image augmentation (renamed from BISINDOImageAugmenter)
│   ├── landmarks.py          # Hand landmarks extraction (renamed from BISINDOHandLandmarksExtractor)
│   ├── features.py           # Feature extraction (renamed from BISINDOFeatureExtractor)
│   └── training.py           # Model training (renamed from BISINDOModelTrainer)
│
├── models/                    # Model definitions and utilities
│   └── __init__.py
│
├── utils/                     # Utility functions
│   ├── __init__.py
│   └── data_utils.py         # Data utility functions
│
├── tests/                     # Test modules
│   ├── test_augmentation.py
│   ├── test_features.py
│   ├── test_landmarks.py
│   └── test_ml_training.py
│
├── pipeline.py               # Main pipeline orchestrator
├── main.py                   # Entry point
└── README.md                 # This file
```

## 🏗️ Architecture Overview

### Clean Code Principles Applied

1. **Single Responsibility Principle**: Each module has a single, well-defined purpose
2. **Dependency Inversion**: High-level modules don't depend on low-level modules
3. **Configuration Management**: All settings centralized in `config/`
4. **Separation of Concerns**: Data, processing, and utilities are clearly separated
5. **Consistent Naming**: Clear, descriptive names throughout

### Module Responsibilities

#### 📋 `config/` - Configuration Management
- **`settings.py`**: Centralized configuration for all pipeline components
- **`__init__.py`**: Exports configuration constants
- **Benefits**: Easy to modify settings, no hardcoded values, consistent configuration

#### 📊 `data/` - Data Processing
- **`manager.py`**: High-level data management and organization
- **`loader.py`**: Raw dataset loading and preprocessing
- **`validator.py`**: Data quality validation and integrity checks
- **Benefits**: Clean data handling, validation, and management

#### ⚙️ `core/` - Core Processing
- **`augmentation.py`**: Image augmentation using imgaug/OpenCV
- **`landmarks.py`**: Hand landmarks extraction using MediaPipe
- **`features.py`**: Feature engineering from landmarks
- **`training.py`**: Machine learning model training and evaluation
- **Benefits**: Modular processing, easy to test and maintain

#### 🛠️ `utils/` - Utility Functions
- **`data_utils.py`**: Helper functions for data processing
- **Benefits**: Reusable utilities, clean separation of concerns

#### 🧪 `tests/` - Testing
- **Test modules**: Comprehensive testing for each component
- **Benefits**: Quality assurance, regression testing

#### 🚀 `pipeline.py` - Main Orchestrator
- **`BISINDOPipeline`**: Orchestrates the complete ML pipeline
- **Benefits**: Clean workflow, error handling, logging

## 🔄 Pipeline Flow

```
Raw Images → Data Loader → Augmenter → Landmarks Extractor → Feature Extractor → Model Trainer
     ↓              ↓           ↓              ↓                    ↓                ↓
  Validation    Validation  Validation    Validation          Validation      Validation
```

## 📝 Usage Examples

### Basic Usage
```python
from src.pipeline import BISINDOPipeline

# Initialize pipeline
pipeline = BISINDOPipeline()

# Run complete pipeline
results = pipeline.run_complete_pipeline()

# Check status
pipeline.print_pipeline_summary()
```

### Individual Components
```python
from src.core import ImageAugmenter, LandmarksExtractor
from src.data import DatasetLoader, DataValidator

# Load data
loader = DatasetLoader()
images, labels, classes = loader.load_dataset()

# Augment images
augmenter = ImageAugmenter()
augmented_images, augmented_labels = augmenter.augment_images(images, labels, classes)

# Extract landmarks
extractor = LandmarksExtractor()
landmarks, valid_labels = extractor.extract_landmarks_from_dataset(images, labels, classes)
```

### Configuration
```python
from src.config import AUGMENTATION_CONFIG, ML_TRAINING_CONFIG

# Custom configuration
custom_config = {
    'target_images_per_class': 100,
    'use_hyperparameter_tuning': True
}

# Use custom config
augmenter = ImageAugmenter(config=custom_config)
```

## 🎯 Key Improvements

### Before (Spaghetti Code)
- ❌ All files in one directory
- ❌ Hardcoded paths and settings
- ❌ Inconsistent naming
- ❌ No clear separation of concerns
- ❌ Difficult to maintain and test

### After (Clean Architecture)
- ✅ Organized directory structure
- ✅ Centralized configuration
- ✅ Consistent naming conventions
- ✅ Clear separation of concerns
- ✅ Easy to maintain and test
- ✅ Modular and extensible

## 🔧 Configuration Management

All configuration is centralized in `src/config/settings.py`:

```python
# Dataset paths
DATASET_BASE_PATH = Path("dataset")
AUGMENTED_DATA_PATH = PROCESSED_DATA_PATH / "01_augmented"

# Processing settings
IMAGE_SIZE = (224, 224, 3)
TARGET_IMAGES_PER_CLASS = 50

# ML settings
ML_TRAINING_CONFIG = {
    'test_size': 0.2,
    'cv_folds': 5,
    'models': {...}
}
```

## 🧪 Testing

Each module has corresponding test files:

```bash
# Run individual tests
python src/tests/test_augmentation.py
python src/tests/test_landmarks.py
python src/tests/test_features.py
python src/tests/test_ml_training.py

# Run all tests
python -m pytest src/tests/
```

## 📈 Benefits of Clean Architecture

1. **Maintainability**: Easy to understand and modify
2. **Testability**: Each component can be tested independently
3. **Extensibility**: Easy to add new features or modules
4. **Reusability**: Components can be reused in different contexts
5. **Debugging**: Easier to identify and fix issues
6. **Collaboration**: Multiple developers can work on different modules
7. **Documentation**: Clear structure makes documentation easier

## 🚀 Getting Started

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the pipeline**:
   ```bash
   python src/main.py
   ```

3. **Check pipeline status**:
   ```python
   from src.pipeline import BISINDOPipeline
   pipeline = BISINDOPipeline()
   pipeline.print_pipeline_summary()
   ```

## 🔮 Future Enhancements

- **Real-time inference**: Add real-time video processing
- **Web interface**: Create a web-based interface
- **Model serving**: Add model serving capabilities
- **Advanced features**: Add more sophisticated augmentation techniques
- **Performance optimization**: Optimize for speed and memory usage

This clean architecture makes the BISINDO system much more professional, maintainable, and ready for production use! 🎉
