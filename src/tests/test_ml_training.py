"""
Test script for machine learning training functionality.
This script demonstrates the ML training pipeline with sample data.
"""

import numpy as np
import pandas as pd
from ml_training import BISINDOModelTrainer


def create_sample_data():
    """
    Create sample training data for testing ML training functionality.
    """
    np.random.seed(42)
    
    # Create sample features and labels
    n_samples = 200
    n_features = 20
    n_classes = 5
    
    # Generate features with some class separation
    X = np.random.randn(n_samples, n_features)
    
    # Add class-specific patterns
    for i in range(n_classes):
        class_indices = np.arange(i * 40, (i + 1) * 40)
        # Add class-specific bias to some features
        X[class_indices, i % n_features] += 2.0
        X[class_indices, (i + 1) % n_features] += 1.5
    
    # Generate labels
    y = np.repeat(np.arange(n_classes), 40)
    
    # Shuffle data
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    class_names = [f'Class_{i}' for i in range(n_classes)]
    
    return X, y, class_names


def test_ml_training():
    """
    Test the machine learning training functionality with sample data.
    """
    print("=== Testing Machine Learning Training ===")
    
    # Create sample data
    print("Creating sample training data...")
    X, y, class_names = create_sample_data()
    print(f"Sample data: {X.shape}, {len(class_names)} classes")
    
    # Initialize trainer
    print("\nInitializing ML trainer...")
    trainer = BISINDOModelTrainer(
        test_size=0.2,
        random_state=42,
        cv_folds=3,  # Reduced for faster testing
        n_jobs=1     # Reduced for testing
    )
    
    # Manually set data (simulating load_data)
    from sklearn.model_selection import train_test_split
    trainer.X_train, trainer.X_test, trainer.y_train, trainer.y_test = train_test_split(
        X, y, test_size=trainer.test_size, random_state=trainer.random_state, stratify=y
    )
    trainer.class_names = class_names
    
    # Update statistics
    trainer.training_stats.update({
        'total_samples': len(X),
        'train_samples': len(trainer.X_train),
        'test_samples': len(trainer.X_test),
        'n_features': X.shape[1],
        'n_classes': len(class_names)
    })
    
    print(f"Data split: {len(trainer.X_train)} train, {len(trainer.X_test)} test")
    
    # Train models with quick training
    print("\nTraining models (quick mode)...")
    results = trainer.train_models(
        use_hyperparameter_tuning=True,
        quick_training=True  # Use simplified parameter grids
    )
    
    print(f"Trained {len(results)} models")
    
    # Evaluate models
    print("\nEvaluating models...")
    df_results = trainer.evaluate_models()
    print("\nModel Comparison:")
    print(df_results)
    
    # Show best model
    print(f"\nBest model: {trainer.best_model_name}")
    print(f"Best accuracy: {trainer.best_score:.4f}")
    
    # Test individual model methods
    print("\nTesting individual model methods...")
    
    # Test model comparison plot
    try:
        trainer.plot_model_comparison()
        print("✓ Model comparison plot generated")
    except Exception as e:
        print(f"✗ Model comparison plot failed: {e}")
    
    # Test confusion matrix plot
    try:
        trainer.plot_confusion_matrix()
        print("✓ Confusion matrix plot generated")
    except Exception as e:
        print(f"✗ Confusion matrix plot failed: {e}")
    
    # Test classification report plot
    try:
        trainer.plot_classification_report()
        print("✓ Classification report plot generated")
    except Exception as e:
        print(f"✗ Classification report plot failed: {e}")
    
    # Test training summary
    summary = trainer.get_training_summary()
    print("\nTraining Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Test model saving (simulate)
    print("\nTesting model saving...")
    try:
        # Create a temporary directory for testing
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer.save_results(temp_dir)
            
            # Check if files were created
            files_created = os.listdir(temp_dir)
            print(f"Files created: {files_created}")
            
            if 'best_model.pkl' in files_created:
                print("✓ Best model saved successfully")
            else:
                print("✗ Best model not saved")
            
            if 'model_comparison.csv' in files_created:
                print("✓ Model comparison saved successfully")
            else:
                print("✗ Model comparison not saved")
            
            if 'training_results.json' in files_created:
                print("✓ Training results saved successfully")
            else:
                print("✗ Training results not saved")
                
    except Exception as e:
        print(f"✗ Model saving failed: {e}")
    
    # Test model loading
    print("\nTesting model loading...")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer.save_results(temp_dir)
            loaded_model = trainer.load_best_model(os.path.join(temp_dir, 'best_model.pkl'))
            print("✓ Model loaded successfully")
            print(f"Loaded model type: {type(loaded_model)}")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
    
    print("\nML training test completed successfully!")


def test_quick_training():
    """
    Test quick training mode with minimal models.
    """
    print("\n=== Testing Quick Training Mode ===")
    
    # Create sample data
    X, y, class_names = create_sample_data()
    
    # Initialize trainer with minimal models
    trainer = BISINDOModelTrainer(
        test_size=0.2,
        random_state=42,
        cv_folds=3,
        n_jobs=1
    )
    
    # Keep only a few models for quick testing
    trainer.models = {
        'SVM': trainer.models['SVM'],
        'Random Forest': trainer.models['Random Forest'],
        'KNN': trainer.models['KNN']
    }
    
    # Set data
    from sklearn.model_selection import train_test_split
    trainer.X_train, trainer.X_test, trainer.y_train, trainer.y_test = train_test_split(
        X, y, test_size=trainer.test_size, random_state=trainer.random_state, stratify=y
    )
    trainer.class_names = class_names
    
    # Train models
    print("Training 3 models in quick mode...")
    results = trainer.train_models(
        use_hyperparameter_tuning=False,  # Skip hyperparameter tuning for speed
        quick_training=True
    )
    
    # Evaluate
    df_results = trainer.evaluate_models()
    print("\nQuick Training Results:")
    print(df_results)
    
    print("Quick training test completed!")


if __name__ == "__main__":
    test_ml_training()
    test_quick_training()
