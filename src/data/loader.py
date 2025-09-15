"""
Dataset Loader for BISINDO Alphabet Recognition.
This module handles loading and preprocessing of raw image datasets.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

try:
    from ..config import (
        RAW_DATA_PATH, IMAGE_SIZE, RANDOM_SEED
    )
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import (
        RAW_DATA_PATH, IMAGE_SIZE, RANDOM_SEED
    )


class DatasetLoader:
    """
    A class to handle loading and preprocessing of raw BISINDO alphabet dataset.
    """
    
    def __init__(self, 
                 data_path: Optional[Path] = None,
                 image_size: Tuple[int, int, int] = IMAGE_SIZE,
                 random_seed: int = RANDOM_SEED):
        """
        Initialize the dataset loader.
        
        Args:
            data_path (Optional[Path]): Path to raw dataset
            image_size (Tuple[int, int, int]): Target image size (height, width, channels)
            random_seed (int): Random seed for reproducibility
        """
        self.data_path = data_path or RAW_DATA_PATH
        self.image_size = image_size
        self.random_seed = random_seed
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        
        # Data storage
        self.images = None
        self.labels = None
        self.labels_encoded = None
        self.class_names = None
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'total_classes': 0,
            'images_per_class': {},
            'class_names': []
        }
    
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load the complete dataset from raw images.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: Images, labels, class names
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.data_path}")
        
        print(f"Loading dataset from: {self.data_path}")
        
        images = []
        labels = []
        class_names = []
        
        # Get all class directories
        class_dirs = [d for d in self.data_path.iterdir() if d.is_dir()]
        class_dirs.sort()  # Sort for consistent ordering
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            class_names.append(class_name)
            
            # Get all image files in the class directory
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
            
            print(f"Loading class '{class_name}': {len(image_files)} images")
            
            for image_file in image_files:
                try:
                    # Load and preprocess image
                    image = self._load_and_preprocess_image(image_file)
                    images.append(image)
                    labels.append(class_name)
                    
                except Exception as e:
                    print(f"Warning: Could not load image {image_file}: {e}")
                    continue
        
        # Convert to numpy arrays
        self.images = np.array(images)
        self.labels = np.array(labels)
        self.class_names = class_names
        
        # Encode labels
        self.labels_encoded = self.label_encoder.fit_transform(self.labels)
        
        # Update statistics
        self._update_statistics()
        
        print(f"Dataset loaded successfully!")
        print(f"  Total images: {len(self.images)}")
        print(f"  Total classes: {len(self.class_names)}")
        print(f"  Image shape: {self.images.shape}")
        print(f"  Classes: {', '.join(self.class_names[:5])}{'...' if len(self.class_names) > 5 else ''}")
        
        return self.images, self.labels_encoded, self.class_names
    
    def _load_and_preprocess_image(self, image_path: Path) -> np.ndarray:
        """
        Load and preprocess a single image.
        
        Args:
            image_path (Path): Path to the image file
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        
        # Normalize to [0, 1] range
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def _update_statistics(self) -> None:
        """Update dataset statistics."""
        if self.labels is not None:
            self.stats['total_images'] = len(self.labels)
            self.stats['total_classes'] = len(self.class_names)
            self.stats['class_names'] = self.class_names.copy()
            
            # Count images per class
            for class_name in self.class_names:
                count = np.sum(self.labels == class_name)
                self.stats['images_per_class'][class_name] = count
    
    def split_dataset(self, 
                     test_size: float = 0.2,
                     stratify: bool = True,
                     random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split dataset into training and testing sets.
        
        Args:
            test_size (float): Proportion of data for testing
            stratify (bool): Whether to stratify the split
            random_state (Optional[int]): Random state for reproducibility
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
        """
        if self.images is None or self.labels_encoded is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        random_state = random_state or self.random_seed
        
        stratify_labels = self.labels_encoded if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.images,
            self.labels_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_labels
        )
        
        print(f"Dataset split:")
        print(f"  Training set: {len(X_train)} images")
        print(f"  Testing set: {len(X_test)} images")
        print(f"  Test ratio: {test_size:.1%}")
        
        return X_train, X_test, y_train, y_test
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of images per class.
        
        Returns:
            Dict[str, int]: Class name to count mapping
        """
        return self.stats['images_per_class'].copy()
    
    def get_statistics(self) -> Dict:
        """
        Get comprehensive dataset statistics.
        
        Returns:
            Dict: Dataset statistics
        """
        stats = self.stats.copy()
        
        if self.images is not None:
            stats.update({
                'image_shape': self.images.shape,
                'data_type': str(self.images.dtype),
                'memory_usage_mb': self.images.nbytes / (1024 * 1024),
                'min_pixel_value': float(self.images.min()),
                'max_pixel_value': float(self.images.max()),
                'mean_pixel_value': float(self.images.mean()),
                'std_pixel_value': float(self.images.std())
            })
        
        return stats
    
    def save_preprocessed_data(self, save_path: Path) -> None:
        """
        Save preprocessed dataset to disk.
        
        Args:
            save_path (Path): Path to save the data
        """
        if self.images is None or self.labels_encoded is None:
            raise ValueError("No data to save. Call load_dataset() first.")
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save images and labels
        np.save(save_path / "images.npy", self.images)
        np.save(save_path / "labels.npy", self.labels_encoded)
        
        # Save class names
        import json
        with open(save_path / "class_names.json", 'w') as f:
            json.dump(self.class_names, f)
        
        # Save statistics
        with open(save_path / "dataset_stats.json", 'w') as f:
            json.dump(self.get_statistics(), f, indent=2)
        
        print(f"Preprocessed data saved to: {save_path}")
    
    def load_preprocessed_data(self, load_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load preprocessed dataset from disk.
        
        Args:
            load_path (Path): Path to load the data from
            
        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: Images, labels, class names
        """
        # Load images and labels
        self.images = np.load(load_path / "images.npy")
        self.labels_encoded = np.load(load_path / "labels.npy")
        
        # Load class names
        import json
        with open(load_path / "class_names.json", 'r') as f:
            self.class_names = json.load(f)
        
        # Fit label encoder
        self.label_encoder.fit(self.class_names)
        
        # Update statistics
        self._update_statistics()
        
        print(f"Preprocessed data loaded from: {load_path}")
        print(f"  Images: {self.images.shape}")
        print(f"  Labels: {self.labels_encoded.shape}")
        print(f"  Classes: {len(self.class_names)}")
        
        return self.images, self.labels_encoded, self.class_names
