"""
Data utilities for BISINDO alphabet recognition.
This module contains helper functions for data preprocessing and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import cv2


def normalize_images(images: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize images using different methods.
    
    Args:
        images (np.ndarray): Input images array
        method (str): Normalization method ('minmax', 'zscore', 'unit')
        
    Returns:
        np.ndarray: Normalized images
    """
    if method == 'minmax':
        # Min-max normalization to [0, 1]
        return (images - images.min()) / (images.max() - images.min())
    
    elif method == 'zscore':
        # Z-score normalization
        mean = images.mean()
        std = images.std()
        return (images - mean) / std
    
    elif method == 'unit':
        # Unit vector normalization
        norms = np.linalg.norm(images.reshape(images.shape[0], -1), axis=1)
        return images / norms.reshape(-1, 1, 1, 1)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def augment_image_brightness(image: np.ndarray, factor: float = 0.2) -> np.ndarray:
    """
    Augment image brightness.
    
    Args:
        image (np.ndarray): Input image
        factor (float): Brightness factor (0.0 = no change, 0.2 = ±20% change)
        
    Returns:
        np.ndarray: Brightness augmented image
    """
    # Random brightness adjustment
    brightness_factor = 1.0 + np.random.uniform(-factor, factor)
    augmented = image * brightness_factor
    
    # Clip values to valid range
    return np.clip(augmented, 0.0, 1.0)


def augment_image_contrast(image: np.ndarray, factor: float = 0.2) -> np.ndarray:
    """
    Augment image contrast.
    
    Args:
        image (np.ndarray): Input image
        factor (float): Contrast factor
        
    Returns:
        np.ndarray: Contrast augmented image
    """
    # Random contrast adjustment
    contrast_factor = 1.0 + np.random.uniform(-factor, factor)
    mean = image.mean()
    augmented = (image - mean) * contrast_factor + mean
    
    # Clip values to valid range
    return np.clip(augmented, 0.0, 1.0)


def plot_class_distribution(labels: np.ndarray, title: str = "Class Distribution"):
    """
    Plot the distribution of classes in the dataset.
    
    Args:
        labels (np.ndarray): Array of labels
        title (str): Plot title
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(12, 6))
    plt.bar(unique_labels, counts)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    print(f"Class distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count} images")


def save_preprocessed_data(images: np.ndarray, labels: np.ndarray, 
                          save_path: str = "dataset/processed"):
    """
    Save preprocessed data to disk.
    
    Args:
        images (np.ndarray): Preprocessed images
        labels (np.ndarray): Corresponding labels
        save_path (str): Path to save the data
    """
    import os
    from pathlib import Path
    
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save images and labels
    np.save(save_path / "images.npy", images)
    np.save(save_path / "labels.npy", labels)
    
    print(f"Preprocessed data saved to {save_path}")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")


def load_preprocessed_data(load_path: str = "dataset/processed") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load preprocessed data from disk.
    
    Args:
        load_path (str): Path to load the data from
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Loaded images and labels
    """
    from pathlib import Path
    
    load_path = Path(load_path)
    
    images = np.load(load_path / "images.npy")
    labels = np.load(load_path / "labels.npy")
    
    print(f"Preprocessed data loaded from {load_path}")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    
    return images, labels


def get_image_statistics(images: np.ndarray) -> Dict:
    """
    Get statistical information about the images.
    
    Args:
        images (np.ndarray): Input images array
        
    Returns:
        Dict: Statistical information
    """
    stats = {
        'shape': images.shape,
        'dtype': images.dtype,
        'min_value': images.min(),
        'max_value': images.max(),
        'mean_value': images.mean(),
        'std_value': images.std(),
        'memory_usage_mb': images.nbytes / (1024 * 1024)
    }
    
    return stats
