"""
Machine Learning Training and Evaluation Module for BISINDO Alphabet Recognition
This module handles training, evaluation, and comparison of multiple ML models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any
import matplotlib.pyplot as plt
from tqdm import tqdm

# Optional import for seaborn
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
import json
import pickle
import joblib
from datetime import datetime

# Import configuration with fallback
try:
    from ..config import (
        ML_TRAINING_CONFIG, MODELS_DATA_PATH, RANDOM_SEED
    )
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import (
        ML_TRAINING_CONFIG, MODELS_DATA_PATH, RANDOM_SEED
    )

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Optional imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class ModelTrainer:
    """
    A class to handle training, evaluation, and comparison of multiple ML models for BISINDO alphabet recognition.
    """
    
    def __init__(self, 
                 test_size: float = 0.2,
                 random_state: int = 42,
                 cv_folds: int = 5,
                 n_jobs: int = -1):
        """
        Initialize the model trainer.
        
        Args:
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            cv_folds (int): Number of cross-validation folds
            n_jobs (int): Number of parallel jobs (-1 for all cores)
        """
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        
        # Initialize models and their parameter grids
        self.models = self._initialize_models()
        self.param_grids = self._initialize_param_grids()
        
        # Results storage
        self.training_results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0.0
        
        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.class_names = None
        self.label_encoder = None
        
        # Statistics
        self.training_stats = {
            'total_samples': 0,
            'train_samples': 0,
            'test_samples': 0,
            'n_features': 0,
            'n_classes': 0,
            'training_time': 0,
            'models_trained': 0
        }
    
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize the machine learning models."""
        models = {
            'SVM': SVC(random_state=self.random_state, probability=True),
            'Random Forest': RandomForestClassifier(random_state=self.random_state, n_jobs=self.n_jobs),
            'KNN': KNeighborsClassifier(n_jobs=self.n_jobs),
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000, n_jobs=self.n_jobs),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(random_state=self.random_state),
            'Gradient Boosting': GradientBoostingClassifier(random_state=self.random_state),
            'MLP': MLPClassifier(random_state=self.random_state, max_iter=1000),
            'LDA': LinearDiscriminantAnalysis(),
            'QDA': QuadraticDiscriminantAnalysis()
        }
        
        # Add optional models if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
        
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = lgb.LGBMClassifier(random_state=self.random_state, n_jobs=self.n_jobs, verbose=-1)
        
        return models
    
    def _initialize_param_grids(self) -> Dict[str, Dict]:
        """Initialize parameter grids for hyperparameter tuning."""
        param_grids = {
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga']
            },
            'Naive Bayes': {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
            },
            'Decision Tree': {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'MLP': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01]
            },
            'LDA': {
                'solver': ['svd', 'lsqr', 'eigen'],
                'shrinkage': [None, 'auto', 0.1, 0.5, 0.9]
            },
            'QDA': {
                'reg_param': [0.0, 0.1, 0.5, 1.0]
            }
        }
        
        # Add optional model parameter grids
        if XGBOOST_AVAILABLE:
            param_grids['XGBoost'] = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        
        if LIGHTGBM_AVAILABLE:
            param_grids['LightGBM'] = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100]
            }
        
        return param_grids
    
    def load_data(self, 
                  features_path: str = "dataset/processed/03_features",
                  landmarks_path: str = "dataset/processed/02_landmarks",
                  use_features: bool = True) -> None:
        """
        Load training data from processed dataset.
        
        Args:
            features_path (str): Path to features dataset
            landmarks_path (str): Path to landmarks dataset
            use_features (bool): Whether to use extracted features or landmarks
        """
        features_path = Path(features_path)
        
        if use_features:
            # Load extracted features
            X = np.load(features_path / "extracted_features.npy")
            y = np.load(features_path / "feature_labels.npy")
            
            with open(features_path / "feature_class_names.json", 'r') as f:
                self.class_names = json.load(f)
            
            print(f"Loaded extracted features: {X.shape}, {len(self.class_names)} classes")
        else:
            # Load landmarks
            landmarks_path = Path(landmarks_path)
            X = np.load(landmarks_path / "hand_landmarks.npy")
            y = np.load(landmarks_path / "landmark_labels.npy")
            
            with open(landmarks_path / "landmark_class_names.json", 'r') as f:
                self.class_names = json.load(f)
            
            print(f"Loaded landmarks: {X.shape}, {len(self.class_names)} classes")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Update statistics
        self.training_stats.update({
            'total_samples': len(X),
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'n_features': X.shape[1],
            'n_classes': len(self.class_names)
        })
        
        print(f"Data split: {len(self.X_train)} train, {len(self.X_test)} test")
        print(f"Features: {X.shape[1]}, Classes: {len(self.class_names)}")
    
    def train_models(self, 
                    use_hyperparameter_tuning: bool = True,
                    quick_training: bool = False) -> Dict[str, Dict]:
        """
        Train all models with optional hyperparameter tuning.
        
        Args:
            use_hyperparameter_tuning (bool): Whether to perform hyperparameter tuning
            quick_training (bool): Whether to use simplified parameter grids for faster training
            
        Returns:
            Dict[str, Dict]: Training results for all models
        """
        print(f"\n=== Training {len(self.models)} Models ===")
        print(f"Hyperparameter tuning: {'Yes' if use_hyperparameter_tuning else 'No'}")
        print(f"Quick training: {'Yes' if quick_training else 'No'}")
        
        start_time = datetime.now()
        results = {}
        
        # Use simplified parameter grids for quick training
        if quick_training:
            param_grids = self._get_quick_param_grids()
        else:
            param_grids = self.param_grids
        
        for model_name, model in tqdm(self.models.items(), desc="Training models"):
            print(f"\nTraining {model_name}...")
            
            try:
                if use_hyperparameter_tuning and model_name in param_grids:
                    # Hyperparameter tuning with cross-validation
                    cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                    grid_search = GridSearchCV(
                        model, param_grids[model_name], 
                        cv=cv, scoring='accuracy', n_jobs=self.n_jobs, verbose=0
                    )
                    grid_search.fit(self.X_train, self.y_train)
                    
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    best_score = grid_search.best_score_
                    
                    print(f"  Best CV score: {best_score:.4f}")
                    print(f"  Best params: {best_params}")
                else:
                    # Train without hyperparameter tuning
                    best_model = model
                    best_model.fit(self.X_train, self.y_train)
                    best_params = {}
                    best_score = np.mean(cross_val_score(
                        best_model, self.X_train, self.y_train, 
                        cv=self.cv_folds, scoring='accuracy'
                    ))
                    
                    print(f"  CV score: {best_score:.4f}")
                
                # Evaluate on test set
                y_pred = best_model.predict(self.X_test)
                test_accuracy = accuracy_score(self.y_test, y_pred)
                
                # Calculate additional metrics
                precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
                
                # Store results
                results[model_name] = {
                    'model': best_model,
                    'best_params': best_params,
                    'cv_score': best_score,
                    'test_accuracy': test_accuracy,
                    'test_precision': precision,
                    'test_recall': recall,
                    'test_f1': f1,
                    'y_pred': y_pred
                }
                
                print(f"  Test accuracy: {test_accuracy:.4f}")
                print(f"  Test F1-score: {f1:.4f}")
                
                # Update best model
                if test_accuracy > self.best_score:
                    self.best_score = test_accuracy
                    self.best_model = best_model
                    self.best_model_name = model_name
                
            except Exception as e:
                print(f"  Error training {model_name}: {str(e)}")
                results[model_name] = {
                    'error': str(e),
                    'cv_score': 0.0,
                    'test_accuracy': 0.0,
                    'test_precision': 0.0,
                    'test_recall': 0.0,
                    'test_f1': 0.0
                }
        
        # Update training statistics
        end_time = datetime.now()
        self.training_stats['training_time'] = (end_time - start_time).total_seconds()
        self.training_stats['models_trained'] = len([r for r in results.values() if 'error' not in r])
        
        self.training_results = results
        return results
    
    def _get_quick_param_grids(self) -> Dict[str, Dict]:
        """Get simplified parameter grids for quick training."""
        return {
            'SVM': {'C': [1, 10], 'gamma': ['scale', 'auto']},
            'Random Forest': {'n_estimators': [50, 100], 'max_depth': [None, 10]},
            'KNN': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
            'Logistic Regression': {'C': [1, 10], 'penalty': ['l2']},
            'Naive Bayes': {'var_smoothing': [1e-9, 1e-7]},
            'Decision Tree': {'max_depth': [None, 10], 'criterion': ['gini', 'entropy']},
            'Gradient Boosting': {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.2]},
            'MLP': {'hidden_layer_sizes': [(50,), (100,)], 'activation': ['relu']},
            'LDA': {'solver': ['svd', 'lsqr']},
            'QDA': {'reg_param': [0.0, 0.1]}
        }
    
    def evaluate_models(self) -> pd.DataFrame:
        """
        Evaluate all trained models and return comparison DataFrame.
        
        Returns:
            pd.DataFrame: Model comparison results
        """
        if not self.training_results:
            raise ValueError("No models have been trained yet. Call train_models() first.")
        
        results_data = []
        for model_name, results in self.training_results.items():
            if 'error' not in results:
                results_data.append({
                    'Model': model_name,
                    'CV Score': results['cv_score'],
                    'Test Accuracy': results['test_accuracy'],
                    'Test Precision': results['test_precision'],
                    'Test Recall': results['test_recall'],
                    'Test F1-Score': results['test_f1']
                })
        
        df = pd.DataFrame(results_data)
        df = df.sort_values('Test Accuracy', ascending=False)
        
        return df
    
    def plot_model_comparison(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """Plot model comparison charts."""
        if not self.training_results:
            raise ValueError("No models have been trained yet. Call train_models() first.")
        
        df = self.evaluate_models()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Accuracy comparison
        axes[0, 0].bar(df['Model'], df['Test Accuracy'])
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. CV Score vs Test Accuracy
        axes[0, 1].scatter(df['CV Score'], df['Test Accuracy'], s=100)
        for i, model in enumerate(df['Model']):
            axes[0, 1].annotate(model, (df['CV Score'].iloc[i], df['Test Accuracy'].iloc[i]))
        axes[0, 1].set_xlabel('CV Score')
        axes[0, 1].set_ylabel('Test Accuracy')
        axes[0, 1].set_title('CV Score vs Test Accuracy')
        
        # 3. Precision, Recall, F1 comparison
        metrics = ['Test Precision', 'Test Recall', 'Test F1-Score']
        x = np.arange(len(df))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            axes[1, 0].bar(x + i*width, df[metric], width, label=metric.replace('Test ', ''))
        
        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Precision, Recall, F1-Score Comparison')
        axes[1, 0].set_xticks(x + width)
        axes[1, 0].set_xticklabels(df['Model'], rotation=45)
        axes[1, 0].legend()
        
        # 4. Model ranking
        ranking = df[['Model', 'Test Accuracy']].copy()
        ranking['Rank'] = range(1, len(ranking) + 1)
        
        axes[1, 1].barh(ranking['Model'], ranking['Test Accuracy'])
        axes[1, 1].set_xlabel('Test Accuracy')
        axes[1, 1].set_title('Model Ranking by Accuracy')
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, model_name: str = None, figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Plot confusion matrix for the best model or specified model.
        
        Args:
            model_name (str): Name of model to plot. If None, uses best model.
            figsize (Tuple[int, int]): Figure size
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.training_results:
            raise ValueError(f"Model '{model_name}' not found in training results.")
        
        results = self.training_results[model_name]
        if 'error' in results:
            raise ValueError(f"Model '{model_name}' had training errors.")
        
        y_pred = results['y_pred']
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=figsize)
        if SEABORN_AVAILABLE:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.class_names, yticklabels=self.class_names)
        else:
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.colorbar()
            plt.xticks(range(len(self.class_names)), self.class_names, rotation=45)
            plt.yticks(range(len(self.class_names)), self.class_names)
            # Add text annotations
            for i in range(len(self.class_names)):
                for j in range(len(self.class_names)):
                    plt.text(j, i, str(cm[i, j]), ha='center', va='center')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def plot_classification_report(self, model_name: str = None) -> None:
        """
        Plot classification report for the best model or specified model.
        
        Args:
            model_name (str): Name of model to plot. If None, uses best model.
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.training_results:
            raise ValueError(f"Model '{model_name}' not found in training results.")
        
        results = self.training_results[model_name]
        if 'error' in results:
            raise ValueError(f"Model '{model_name}' had training errors.")
        
        y_pred = results['y_pred']
        report = classification_report(self.y_test, y_pred, target_names=self.class_names, output_dict=True)
        
        # Convert to DataFrame for better visualization
        df_report = pd.DataFrame(report).transpose()
        df_report = df_report.drop(['accuracy', 'macro avg', 'weighted avg'])
        
        plt.figure(figsize=(12, 8))
        if SEABORN_AVAILABLE:
            sns.heatmap(df_report[['precision', 'recall', 'f1-score']], 
                       annot=True, fmt='.3f', cmap='YlOrRd')
        else:
            # Fallback to matplotlib
            im = plt.imshow(df_report[['precision', 'recall', 'f1-score']].values, 
                           cmap='YlOrRd', aspect='auto')
            plt.colorbar(im)
            plt.xticks(range(3), ['precision', 'recall', 'f1-score'])
            plt.yticks(range(len(df_report)), df_report.index)
            # Add text annotations
            for i in range(len(df_report)):
                for j in range(3):
                    plt.text(j, i, f"{df_report.iloc[i, j]:.3f}", 
                            ha='center', va='center')
        plt.title(f'Classification Report - {model_name}')
        plt.show()
    
    def save_results(self, save_path: str = "dataset/processed/04_models") -> None:
        """
        Save training results and best model.
        
        Args:
            save_path (str): Path to save results
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save best model
        if self.best_model is not None:
            joblib.dump(self.best_model, save_path / "best_model.pkl")
            print(f"Best model ({self.best_model_name}) saved to {save_path / 'best_model.pkl'}")
        
        # Save training results summary
        if self.training_results:
            df_results = self.evaluate_models()
            df_results.to_csv(save_path / "model_comparison.csv", index=False)
            
            # Save detailed results
            results_summary = {
                'best_model': self.best_model_name,
                'best_score': self.best_score,
                'training_stats': self.training_stats,
                'model_results': {}
            }
            
            for model_name, results in self.training_results.items():
                if 'error' not in results:
                    results_summary['model_results'][model_name] = {
                        'cv_score': results['cv_score'],
                        'test_accuracy': results['test_accuracy'],
                        'test_precision': results['test_precision'],
                        'test_recall': results['test_recall'],
                        'test_f1': results['test_f1'],
                        'best_params': results['best_params']
                    }
            
            with open(save_path / "training_results.json", 'w') as f:
                json.dump(results_summary, f, indent=2)
            
            print(f"Training results saved to {save_path}")
            print(f"Model comparison saved to {save_path / 'model_comparison.csv'}")
    
    def load_best_model(self, model_path: str = "dataset/processed/04_models/best_model.pkl") -> Any:
        """
        Load the best trained model.
        
        Args:
            model_path (str): Path to saved model
            
        Returns:
            Any: Loaded model
        """
        model = joblib.load(model_path)
        print(f"Best model loaded from {model_path}")
        return model
    
    def get_training_summary(self) -> Dict:
        """
        Get summary of training process and results.
        
        Returns:
            Dict: Training summary
        """
        summary = {
            "training_stats": self.training_stats,
            "best_model": self.best_model_name,
            "best_score": self.best_score,
            "total_models": len(self.models),
            "successful_models": len([r for r in self.training_results.values() if 'error' not in r]),
            "failed_models": len([r for r in self.training_results.values() if 'error' in r])
        }
        
        if self.training_results:
            df_results = self.evaluate_models()
            summary["top_3_models"] = df_results.head(3)[['Model', 'Test Accuracy']].to_dict('records')
        
        return summary


def main():
    """
    Main function to demonstrate ML training and evaluation.
    """
    print("=== Cerdas Isyarat - Machine Learning Training ===")
    
    # Initialize trainer
    trainer = BISINDOModelTrainer(
        test_size=0.2,
        random_state=42,
        cv_folds=5,
        n_jobs=-1
    )
    
    # Load data
    try:
        trainer.load_data(use_features=True)
        
        # Train models
        results = trainer.train_models(
            use_hyperparameter_tuning=True,
            quick_training=False  # Set to True for faster training
        )
        
        # Evaluate and compare models
        df_results = trainer.evaluate_models()
        print("\n=== Model Comparison ===")
        print(df_results)
        
        # Plot results
        trainer.plot_model_comparison()
        trainer.plot_confusion_matrix()
        trainer.plot_classification_report()
        
        # Save results
        trainer.save_results()
        
        # Show summary
        summary = trainer.get_training_summary()
        print("\n=== Training Summary ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        print("\nMachine learning training completed successfully!")
        print("Ready for the next step: Model comparison and selection")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the feature extraction step has been completed.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
