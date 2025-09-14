"""
Cerdas Isyarat - BISINDO Alphabet Recognition
Dataset Loading and Preprocessing Module

This module handles loading and preprocessing of the BISINDO alphabet dataset
for training machine learning models to recognize sign language gestures.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class BISINDODatasetLoader:
    """
    A class to handle loading and preprocessing of BISINDO alphabet dataset.
    """
    
    def __init__(self, dataset_path: str = "dataset/raw", image_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_path (str): Path to the raw dataset directory
            image_size (Tuple[int, int]): Target size for resizing images (width, height)
        """
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size
        self.images = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        
        # Validate dataset path
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all images from the dataset and preprocess them.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Preprocessed images and their labels
        """
        print("Loading BISINDO alphabet dataset...")
        
        # Get all alphabet folders (A-Z)
        alphabet_folders = sorted([f for f in self.dataset_path.iterdir() 
                                 if f.is_dir() and f.name.isalpha() and len(f.name) == 1])
        
        print(f"Found {len(alphabet_folders)} alphabet folders: {[f.name for f in alphabet_folders]}")
        
        total_images = 0
        
        for folder in alphabet_folders:
            letter = folder.name
            print(f"Processing letter: {letter}")
            
            # Get all image files in the folder
            image_files = list(folder.glob("*.jpg"))
            print(f"  Found {len(image_files)} images")
            
            for image_file in image_files:
                try:
                    # Load and preprocess image
                    image = self._preprocess_image(image_file)
                    if image is not None:
                        self.images.append(image)
                        self.labels.append(letter)
                        total_images += 1
                except Exception as e:
                    print(f"  Error processing {image_file}: {e}")
        
        print(f"Successfully loaded {total_images} images")
        
        # Convert to numpy arrays
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        
        # Encode labels
        self.labels_encoded = self.label_encoder.fit_transform(self.labels)
        
        return self.images, self.labels_encoded
    
    def _preprocess_image(self, image_path: Path) -> np.ndarray:
        """
        Preprocess a single image.
        
        Args:
            image_path (Path): Path to the image file
            
        Returns:
            np.ndarray: Preprocessed image array
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, self.image_size)
        
        # Normalize pixel values to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def get_dataset_info(self) -> Dict:
        """
        Get information about the loaded dataset.
        
        Returns:
            Dict: Dataset information including counts and statistics
        """
        if len(self.images) == 0:
            return {"error": "Dataset not loaded yet. Call load_dataset() first."}
        
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        
        info = {
            "total_images": len(self.images),
            "image_shape": self.images[0].shape,
            "target_size": self.image_size,
            "num_classes": len(unique_labels),
            "classes": unique_labels.tolist(),
            "images_per_class": dict(zip(unique_labels, counts)),
            "label_encoder_classes": self.label_encoder.classes_.tolist()
        }
        
        return info
    
    def visualize_samples(self, num_samples: int = 8, figsize: Tuple[int, int] = (12, 8)):
        """
        Visualize sample images from the dataset.
        
        Args:
            num_samples (int): Number of samples to display
            figsize (Tuple[int, int]): Figure size for matplotlib
        """
        if len(self.images) == 0:
            print("Dataset not loaded yet. Call load_dataset() first.")
            return
        
        # Get random samples
        indices = np.random.choice(len(self.images), min(num_samples, len(self.images)), replace=False)
        
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            if i < len(axes):
                axes[i].imshow(self.images[idx])
                axes[i].set_title(f"Label: {self.labels[idx]}")
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(indices), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def split_dataset(self, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Split the dataset into training and testing sets.
        
        Args:
            test_size (float): Proportion of dataset to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        if len(self.images) == 0:
            raise ValueError("Dataset not loaded yet. Call load_dataset() first.")
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.images, self.labels_encoded, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.labels_encoded
        )
        
        print(f"Dataset split:")
        print(f"  Training set: {len(X_train)} images")
        print(f"  Testing set: {len(X_test)} images")
        
        return X_train, X_test, y_train, y_test


def main():
    """
    Main function to demonstrate dataset loading and preprocessing.
    """
    print("=== Cerdas Isyarat - BISINDO Dataset Loading ===")
    
    # Initialize dataset loader
    dataset_loader = BISINDODatasetLoader()
    
    # Load dataset
    images, labels = dataset_loader.load_dataset()
    
    # Get dataset information
    info = dataset_loader.get_dataset_info()
    print("\n=== Dataset Information ===")
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Split dataset
    X_train, X_test, y_train, y_test = dataset_loader.split_dataset()
    
    # Visualize samples
    print("\nVisualizing sample images...")
    dataset_loader.visualize_samples()
    
    print("\nDataset loading and preprocessing completed successfully!")
    print("Ready for the next step: Image augmentation with imgaug")


if __name__ == "__main__":
    main()
