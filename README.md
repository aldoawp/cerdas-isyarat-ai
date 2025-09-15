# 🤟 BISINDO Alphabet Recognition System

An AI-powered computer vision system for recognizing Indonesian Sign Language (BISINDO) alphabet gestures using machine learning and real-time video processing.

## 📋 Overview

This project implements a complete machine learning pipeline to recognize BISINDO (Bahasa Isyarat Indonesia) alphabet signs from images and video streams. The system uses hand landmark detection, feature extraction, and multiple machine learning algorithms to classify sign language gestures.

## 🎯 Features

- **Image Augmentation**: Increases dataset size using imgaug and OpenCV
- **Hand Landmark Detection**: Extracts hand landmarks using MediaPipe
- **Feature Engineering**: Creates meaningful features from hand landmarks
- **Multiple ML Models**: Trains and compares various algorithms (SVM, Random Forest, KNN, etc.)
- **Real-time Processing**: Ready for web browser integration with ONNX
- **Clean Architecture**: Modular, maintainable, and extensible codebase

## 🏗️ Architecture

```
src/
├── config/           # Configuration management
├── data/            # Data processing (loader, manager, validator)
├── core/            # Core processing (augmentation, landmarks, features, training)
├── models/          # Model definitions
├── utils/           # Utility functions
├── tests/           # Test modules
├── pipeline.py      # Main orchestrator
└── main.py          # Entry point
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Required packages (see `requirements.txt`)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd cerdas-isyarat-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your dataset**
   - Place BISINDO alphabet images in `dataset/raw/`
   - Organize by class folders (A, B, C, ..., Z)

### Usage

**Run the complete pipeline:**
```bash
python run_pipeline.py
```

**Or use individual components:**
```python
from src.pipeline import BISINDOPipeline

# Initialize and run pipeline
pipeline = BISINDOPipeline()
results = pipeline.run_complete_pipeline()
```

## 📊 Pipeline Workflow

1. **Data Loading** → Load raw BISINDO alphabet images
2. **Image Augmentation** → Generate 50 images per class using various techniques
3. **Landmark Extraction** → Extract hand landmarks using MediaPipe
4. **Feature Engineering** → Create meaningful features from landmarks
5. **Model Training** → Train and compare multiple ML algorithms
6. **Model Selection** → Choose the best performing model

## 🎛️ Configuration

All settings are centralized in `src/config/settings.py`:

- **Dataset paths**: Organized folder structure
- **Image processing**: Size, augmentation parameters
- **ML training**: Models, hyperparameters, cross-validation
- **Feature extraction**: Geometric and statistical features

## 📁 Dataset Structure

```
dataset/
├── raw/                    # Original images
│   ├── A/                 # Class A images
│   ├── B/                 # Class B images
│   └── ...                # Classes C-Z
│
└── processed/             # Processed data (auto-generated)
    ├── 01_augmented/      # Augmented images
    ├── 02_landmarks/      # Hand landmarks
    ├── 03_features/       # Extracted features
    ├── 04_models/         # Trained models
    └── 05_metadata/       # Additional metadata
```

## 🤖 Supported ML Models

- **Support Vector Machine (SVM)**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**
- **Naive Bayes**
- **Decision Tree**
- **Gradient Boosting**
- **Multi-layer Perceptron (MLP)**
- **Linear Discriminant Analysis (LDA)**
- **Quadratic Discriminant Analysis (QDA)**

## 📈 Performance

The system automatically:
- Trains all models with hyperparameter tuning
- Evaluates using cross-validation
- Compares performance metrics
- Selects the best model
- Saves results and visualizations

## 🧪 Testing

Run individual tests:
```bash
python src/tests/test_augmentation.py
python src/tests/test_landmarks.py
python src/tests/test_features.py
python src/tests/test_ml_training.py
```

## 🔧 Development

### Adding New Features

1. **New augmentation techniques**: Extend `src/core/augmentation.py`
2. **New ML models**: Add to `src/core/training.py`
3. **New features**: Extend `src/core/features.py`
4. **Configuration**: Update `src/config/settings.py`

### Code Quality

- **Clean Architecture**: Modular design with separation of concerns
- **Type Hints**: Full type annotation support
- **Documentation**: Comprehensive docstrings
- **Error Handling**: Robust error handling and fallbacks
- **Testing**: Unit tests for all components

## 📚 Key Technologies

- **Python 3.8+**: Core programming language
- **scikit-learn**: Machine learning algorithms
- **MediaPipe**: Hand landmark detection
- **OpenCV**: Image processing
- **imgaug**: Image augmentation
- **NumPy/Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization

## 🌐 Web Integration

The system is designed for web browser integration:
- **ONNX Export**: Models can be exported for browser inference
- **Real-time Processing**: Ready for video stream processing
- **API Ready**: Modular design supports REST API development

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For questions or issues:
- Create an issue in the repository
- Check the documentation in `src/README.md`
- Review the refactoring summary in `REFACTORING_SUMMARY.md`

## 🎉 Acknowledgments

- **BISINDO Community**: For Indonesian Sign Language resources
- **MediaPipe Team**: For hand landmark detection
- **scikit-learn Community**: For machine learning tools

---

**Ready to recognize BISINDO alphabet signs with AI!** 🤟✨