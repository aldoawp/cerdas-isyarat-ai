"""
Data Management Utility for BISINDO Alphabet Recognition
This module provides utilities for managing organized dataset structure.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pickle
import joblib

# Import configuration with fallback
try:
    from ..config import (
        DATASET_BASE_PATH, RAW_DATA_PATH, PROCESSED_DATA_PATH,
        AUGMENTED_DATA_PATH, LANDMARKS_DATA_PATH, FEATURES_DATA_PATH,
        MODELS_DATA_PATH, METADATA_PATH
    )
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import (
        DATASET_BASE_PATH, RAW_DATA_PATH, PROCESSED_DATA_PATH,
        AUGMENTED_DATA_PATH, LANDMARKS_DATA_PATH, FEATURES_DATA_PATH,
        MODELS_DATA_PATH, METADATA_PATH
    )


class BISINDODataManager:
    """
    A utility class for managing organized dataset structure and file operations.
    """
    
    def __init__(self, base_path: str = "dataset"):
        """
        Initialize the data manager.
        
        Args:
            base_path (str): Base path to the dataset directory
        """
        self.base_path = Path(base_path)
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"
        
        # Organized processed subdirectories
        self.augmented_path = self.processed_path / "01_augmented"
        self.landmarks_path = self.processed_path / "02_landmarks"
        self.features_path = self.processed_path / "03_features"
        self.models_path = self.processed_path / "04_models"
        self.metadata_path = self.processed_path / "05_metadata"
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.raw_path,
            self.processed_path,
            self.augmented_path,
            self.landmarks_path,
            self.features_path,
            self.models_path,
            self.metadata_path
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the dataset structure.
        
        Returns:
            Dict[str, Any]: Dataset information
        """
        info = {
            "raw_data": self._get_raw_data_info(),
            "augmented_data": self._get_augmented_data_info(),
            "landmarks_data": self._get_landmarks_data_info(),
            "features_data": self._get_features_data_info(),
            "models_data": self._get_models_data_info(),
            "metadata": self._get_metadata_info()
        }
        
        return info
    
    def _get_raw_data_info(self) -> Dict[str, Any]:
        """Get information about raw data."""
        if not self.raw_path.exists():
            return {"status": "not_found"}
        
        class_dirs = [d for d in self.raw_path.iterdir() if d.is_dir()]
        class_names = [d.name for d in class_dirs]
        
        total_images = 0
        class_counts = {}
        
        for class_dir in class_dirs:
            images = list(class_dir.glob("*.jpg"))
            class_counts[class_dir.name] = len(images)
            total_images += len(images)
        
        return {
            "status": "available",
            "total_classes": len(class_names),
            "total_images": total_images,
            "class_names": sorted(class_names),
            "class_counts": class_counts
        }
    
    def _get_augmented_data_info(self) -> Dict[str, Any]:
        """Get information about augmented data."""
        images_file = self.augmented_path / "augmented_images.npy"
        labels_file = self.augmented_path / "augmented_labels.npy"
        stats_file = self.augmented_path / "augmentation_stats.json"
        
        if not images_file.exists() or not labels_file.exists():
            return {"status": "not_found"}
        
        try:
            images = np.load(images_file)
            labels = np.load(labels_file)
            
            info = {
                "status": "available",
                "images_shape": images.shape,
                "labels_shape": labels.shape,
                "total_images": len(images),
                "image_size": images.shape[1:] if len(images.shape) > 1 else None
            }
            
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                info["augmentation_stats"] = stats
            
            return info
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _get_landmarks_data_info(self) -> Dict[str, Any]:
        """Get information about landmarks data."""
        landmarks_file = self.landmarks_path / "hand_landmarks.npy"
        labels_file = self.landmarks_path / "landmark_labels.npy"
        stats_file = self.landmarks_path / "landmark_extraction_stats.json"
        
        if not landmarks_file.exists() or not labels_file.exists():
            return {"status": "not_found"}
        
        try:
            landmarks = np.load(landmarks_file)
            labels = np.load(labels_file)
            
            info = {
                "status": "available",
                "landmarks_shape": landmarks.shape,
                "labels_shape": labels.shape,
                "total_samples": len(landmarks),
                "features_per_sample": landmarks.shape[1] if len(landmarks.shape) > 1 else None
            }
            
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                info["extraction_stats"] = stats
            
            return info
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _get_features_data_info(self) -> Dict[str, Any]:
        """Get information about features data."""
        features_file = self.features_path / "extracted_features.npy"
        labels_file = self.features_path / "feature_labels.npy"
        stats_file = self.features_path / "feature_extraction_stats.json"
        
        if not features_file.exists() or not labels_file.exists():
            return {"status": "not_found"}
        
        try:
            features = np.load(features_file)
            labels = np.load(labels_file)
            
            info = {
                "status": "available",
                "features_shape": features.shape,
                "labels_shape": labels.shape,
                "total_samples": len(features),
                "features_per_sample": features.shape[1] if len(features.shape) > 1 else None
            }
            
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                info["extraction_stats"] = stats
            
            return info
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _get_models_data_info(self) -> Dict[str, Any]:
        """Get information about trained models."""
        if not self.models_path.exists():
            return {"status": "not_found"}
        
        model_files = list(self.models_path.glob("*.pkl"))
        csv_files = list(self.models_path.glob("*.csv"))
        json_files = list(self.models_path.glob("*.json"))
        
        return {
            "status": "available" if model_files or csv_files or json_files else "empty",
            "model_files": [f.name for f in model_files],
            "csv_files": [f.name for f in csv_files],
            "json_files": [f.name for f in json_files],
            "total_files": len(model_files) + len(csv_files) + len(json_files)
        }
    
    def _get_metadata_info(self) -> Dict[str, Any]:
        """Get information about metadata files."""
        if not self.metadata_path.exists():
            return {"status": "not_found"}
        
        metadata_files = list(self.metadata_path.glob("*"))
        
        return {
            "status": "available" if metadata_files else "empty",
            "files": [f.name for f in metadata_files],
            "total_files": len(metadata_files)
        }
    
    def load_augmented_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load augmented dataset.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: Images, labels, class names
        """
        images = np.load(self.augmented_path / "augmented_images.npy")
        labels = np.load(self.augmented_path / "augmented_labels.npy")
        
        with open(self.augmented_path / "class_names.json", 'r') as f:
            class_names = json.load(f)
        
        return images, labels, class_names
    
    def load_landmarks_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load landmarks dataset.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: Landmarks, labels, class names
        """
        landmarks = np.load(self.landmarks_path / "hand_landmarks.npy")
        labels = np.load(self.landmarks_path / "landmark_labels.npy")
        
        with open(self.landmarks_path / "landmark_class_names.json", 'r') as f:
            class_names = json.load(f)
        
        return landmarks, labels, class_names
    
    def load_features_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load features dataset.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: Features, labels, class names
        """
        features = np.load(self.features_path / "extracted_features.npy")
        labels = np.load(self.features_path / "feature_labels.npy")
        
        with open(self.features_path / "feature_class_names.json", 'r') as f:
            class_names = json.load(f)
        
        return features, labels, class_names
    
    def save_augmented_data(self, images: np.ndarray, labels: np.ndarray, class_names: List[str], 
                          stats: Optional[Dict] = None) -> None:
        """
        Save augmented dataset.
        
        Args:
            images (np.ndarray): Augmented images
            labels (np.ndarray): Corresponding labels
            class_names (List[str]): Class names
            stats (Optional[Dict]): Augmentation statistics
        """
        np.save(self.augmented_path / "augmented_images.npy", images)
        np.save(self.augmented_path / "augmented_labels.npy", labels)
        
        with open(self.augmented_path / "class_names.json", 'w') as f:
            json.dump(class_names, f)
        
        if stats:
            with open(self.augmented_path / "augmentation_stats.json", 'w') as f:
                json.dump(stats, f, indent=2)
    
    def save_landmarks_data(self, landmarks: np.ndarray, labels: np.ndarray, class_names: List[str],
                          metadata: Optional[Dict] = None, stats: Optional[Dict] = None) -> None:
        """
        Save landmarks dataset.
        
        Args:
            landmarks (np.ndarray): Hand landmarks
            labels (np.ndarray): Corresponding labels
            class_names (List[str]): Class names
            metadata (Optional[Dict]): Landmarks metadata
            stats (Optional[Dict]): Extraction statistics
        """
        np.save(self.landmarks_path / "hand_landmarks.npy", landmarks)
        np.save(self.landmarks_path / "landmark_labels.npy", labels)
        
        with open(self.landmarks_path / "landmark_class_names.json", 'w') as f:
            json.dump(class_names, f)
        
        if metadata:
            with open(self.landmarks_path / "landmarks_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
        
        if stats:
            with open(self.landmarks_path / "landmark_extraction_stats.json", 'w') as f:
                json.dump(stats, f, indent=2)
    
    def save_features_data(self, features: np.ndarray, labels: np.ndarray, class_names: List[str],
                         feature_names: Optional[List[str]] = None, stats: Optional[Dict] = None,
                         transformers: Optional[Dict] = None) -> None:
        """
        Save features dataset.
        
        Args:
            features (np.ndarray): Extracted features
            labels (np.ndarray): Corresponding labels
            class_names (List[str]): Class names
            feature_names (Optional[List[str]]): Feature names
            stats (Optional[Dict]): Extraction statistics
            transformers (Optional[Dict]): Fitted transformers
        """
        np.save(self.features_path / "extracted_features.npy", features)
        np.save(self.features_path / "feature_labels.npy", labels)
        
        with open(self.features_path / "feature_class_names.json", 'w') as f:
            json.dump(class_names, f)
        
        if feature_names:
            with open(self.features_path / "feature_names.json", 'w') as f:
                json.dump(feature_names, f)
        
        if stats:
            with open(self.features_path / "feature_extraction_stats.json", 'w') as f:
                json.dump(stats, f, indent=2)
        
        if transformers:
            for name, transformer in transformers.items():
                joblib.dump(transformer, self.features_path / f"feature_{name}.pkl")
    
    def save_model(self, model: Any, model_name: str, results: Optional[Dict] = None) -> None:
        """
        Save trained model and results.
        
        Args:
            model (Any): Trained model
            model_name (str): Name of the model
            results (Optional[Dict]): Training results
        """
        joblib.dump(model, self.models_path / f"{model_name}.pkl")
        
        if results:
            with open(self.models_path / f"{model_name}_results.json", 'w') as f:
                json.dump(results, f, indent=2)
    
    def load_model(self, model_name: str) -> Any:
        """
        Load trained model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Any: Loaded model
        """
        return joblib.load(self.models_path / f"{model_name}.pkl")
    
    def get_data_pipeline_status(self) -> Dict[str, str]:
        """
        Get the status of each step in the data pipeline.
        
        Returns:
            Dict[str, str]: Status of each pipeline step
        """
        status = {}
        
        # Check raw data
        raw_info = self._get_raw_data_info()
        status["raw_data"] = raw_info["status"]
        
        # Check augmented data
        aug_info = self._get_augmented_data_info()
        status["augmented_data"] = aug_info["status"]
        
        # Check landmarks data
        landmarks_info = self._get_landmarks_data_info()
        status["landmarks_data"] = landmarks_info["status"]
        
        # Check features data
        features_info = self._get_features_data_info()
        status["features_data"] = features_info["status"]
        
        # Check models data
        models_info = self._get_models_data_info()
        status["models_data"] = models_info["status"]
        
        return status
    
    def print_dataset_summary(self) -> None:
        """Print a comprehensive summary of the dataset."""
        print("=" * 60)
        print("BISINDO Dataset Summary")
        print("=" * 60)
        
        info = self.get_dataset_info()
        
        # Raw data
        print("\n📁 Raw Data:")
        raw_info = info["raw_data"]
        if raw_info["status"] == "available":
            print(f"  ✅ Available: {raw_info['total_classes']} classes, {raw_info['total_images']} images")
            print(f"  📊 Classes: {', '.join(raw_info['class_names'][:5])}{'...' if len(raw_info['class_names']) > 5 else ''}")
        else:
            print(f"  ❌ {raw_info['status']}")
        
        # Augmented data
        print("\n🔄 Augmented Data:")
        aug_info = info["augmented_data"]
        if aug_info["status"] == "available":
            print(f"  ✅ Available: {aug_info['total_images']} images")
            if "augmentation_stats" in aug_info:
                stats = aug_info["augmentation_stats"]
                print(f"  📊 Original: {stats.get('total_original_images', 'N/A')} images")
                print(f"  📊 Augmented: {stats.get('total_augmented_images', 'N/A')} images")
        else:
            print(f"  ❌ {aug_info['status']}")
        
        # Landmarks data
        print("\n✋ Landmarks Data:")
        landmarks_info = info["landmarks_data"]
        if landmarks_info["status"] == "available":
            print(f"  ✅ Available: {landmarks_info['total_samples']} samples")
            print(f"  📊 Features per sample: {landmarks_info['features_per_sample']}")
        else:
            print(f"  ❌ {landmarks_info['status']}")
        
        # Features data
        print("\n🔧 Features Data:")
        features_info = info["features_data"]
        if features_info["status"] == "available":
            print(f"  ✅ Available: {features_info['total_samples']} samples")
            print(f"  📊 Features per sample: {features_info['features_per_sample']}")
        else:
            print(f"  ❌ {features_info['status']}")
        
        # Models data
        print("\n🤖 Models Data:")
        models_info = info["models_data"]
        if models_info["status"] == "available":
            print(f"  ✅ Available: {models_info['total_files']} files")
            if models_info["model_files"]:
                print(f"  📊 Models: {', '.join(models_info['model_files'])}")
        else:
            print(f"  ❌ {models_info['status']}")
        
        # Pipeline status
        print("\n🔄 Pipeline Status:")
        pipeline_status = self.get_data_pipeline_status()
        for step, status in pipeline_status.items():
            emoji = "✅" if status == "available" else "❌"
            print(f"  {emoji} {step.replace('_', ' ').title()}: {status}")
        
        print("\n" + "=" * 60)


def main():
    """Main function to demonstrate data manager functionality."""
    print("=== BISINDO Data Manager ===")
    
    # Initialize data manager
    dm = BISINDODataManager()
    
    # Print dataset summary
    dm.print_dataset_summary()
    
    # Get detailed info
    info = dm.get_dataset_info()
    print(f"\nDetailed info available for {len(info)} categories")


if __name__ == "__main__":
    main()
