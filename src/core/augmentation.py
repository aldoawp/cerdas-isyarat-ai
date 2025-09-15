"""
Image Augmentation Module for BISINDO Alphabet Recognition
This module handles image augmentation using imgaug library to increase dataset size
and improve model generalization by creating 50 augmented images per original image.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import random

try:
    from ..config import (
        AUGMENTATION_CONFIG, AUGMENTED_DATA_PATH, 
        TARGET_IMAGES_PER_CLASS, RANDOM_SEED
    )
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import (
        AUGMENTATION_CONFIG, AUGMENTED_DATA_PATH, 
        TARGET_IMAGES_PER_CLASS, RANDOM_SEED
    )

# Try to import imgaug, fallback to OpenCV if not available
try:
    import imgaug as ia
    from imgaug import augmenters as iaa
    IMGAUG_AVAILABLE = True
except ImportError:
    IMGAUG_AVAILABLE = False
    print("Warning: imgaug not available, using OpenCV-based augmentation")


class ImageAugmenter:
    """
    A class to handle image augmentation for BISINDO alphabet dataset using imgaug.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the image augmenter.
        
        Args:
            config (Optional[Dict]): Augmentation configuration
        """
        self.config = config or AUGMENTATION_CONFIG
        self.target_images_per_class = self.config['target_images_per_class']
        self.random_seed = self.config['random_seed']
        
        # Set random seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Define augmentation pipeline
        if IMGAUG_AVAILABLE:
            try:
                # Set random seed for imgaug
                ia.seed(self.random_seed)
                self.augmentation_pipeline = self._create_augmentation_pipeline()
                self.use_imgaug = True
            except Exception as e:
                print(f"Warning: imgaug initialization failed: {e}")
                print("Falling back to OpenCV-based augmentation")
                self.use_imgaug = False
        else:
            self.use_imgaug = False
        
        # Statistics tracking
        self.augmentation_stats = {}
    
    def _create_augmentation_pipeline(self) -> iaa.Sequential:
        """
        Create the augmentation pipeline with various transformations.
        
        Returns:
            iaa.Sequential: Augmentation pipeline
        """
        # Define augmentation techniques suitable for hand gesture recognition
        # Using simpler, more reliable augmentations to avoid compatibility issues
        pipeline = iaa.Sequential([
            # Geometric transformations
            iaa.Sometimes(0.7, iaa.Affine(
                rotate=(-15, 15),  # Small rotation to simulate different hand positions
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # Small translation
                scale=(0.9, 1.1),  # Slight scaling
                shear=(-5, 5)  # Small shear transformation
            )),
            
            # Brightness and contrast adjustments
            iaa.Sometimes(0.6, iaa.MultiplyBrightness((0.8, 1.2))),  # Brightness variation
            iaa.Sometimes(0.6, iaa.LinearContrast((0.8, 1.2))),  # Contrast variation
            
            # Color adjustments
            iaa.Sometimes(0.4, iaa.MultiplyHueAndSaturation((0.8, 1.2))),  # Hue/saturation
            iaa.Sometimes(0.3, iaa.Grayscale(alpha=(0.0, 0.3))),  # Slight grayscale overlay
            
            # Noise and blur
            iaa.Sometimes(0.3, iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))),  # Gaussian noise
            iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 0.5))),  # Slight blur
            
            # Perspective transformation (removed elastic and weather effects to avoid compatibility issues)
            iaa.Sometimes(0.3, iaa.PerspectiveTransform(scale=(0.01, 0.1))),  # Perspective
        ], random_order=True)
        
        return pipeline
    
    def _opencv_augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Augment a single image using OpenCV-based methods.
        
        Args:
            image (np.ndarray): Input image (0-1 range)
            
        Returns:
            np.ndarray: Augmented image
        """
        # Convert to 0-255 range for OpenCV operations
        img_uint8 = (image * 255).astype(np.uint8)
        h, w = img_uint8.shape[:2]
        
        # Random rotation
        if random.random() < 0.7:
            angle = random.uniform(-15, 15)
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_uint8 = cv2.warpAffine(img_uint8, matrix, (w, h))
        
        # Random translation
        if random.random() < 0.5:
            tx = random.uniform(-0.1, 0.1) * w
            ty = random.uniform(-0.1, 0.1) * h
            matrix = np.float32([[1, 0, tx], [0, 1, ty]])
            img_uint8 = cv2.warpAffine(img_uint8, matrix, (w, h))
        
        # Random brightness adjustment
        if random.random() < 0.6:
            brightness = random.uniform(0.8, 1.2)
            img_uint8 = cv2.convertScaleAbs(img_uint8, alpha=brightness, beta=0)
        
        # Random contrast adjustment
        if random.random() < 0.6:
            contrast = random.uniform(0.8, 1.2)
            img_uint8 = cv2.convertScaleAbs(img_uint8, alpha=contrast, beta=0)
        
        # Random noise
        if random.random() < 0.3:
            noise = np.random.normal(0, 10, img_uint8.shape).astype(np.uint8)
            img_uint8 = cv2.add(img_uint8, noise)
        
        # Random blur
        if random.random() < 0.2:
            kernel_size = random.choice([3, 5])
            img_uint8 = cv2.GaussianBlur(img_uint8, (kernel_size, kernel_size), 0)
        
        # Convert back to 0-1 range
        return np.clip(img_uint8.astype(np.float32) / 255.0, 0.0, 1.0)
    
    def augment_images(self, images: np.ndarray, labels: np.ndarray, 
                      class_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment images to reach target number per class.
        
        Args:
            images (np.ndarray): Original images array
            labels (np.ndarray): Original labels array
            class_names (List[str]): List of class names
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Augmented images and labels
        """
        print(f"Starting image augmentation to reach {self.target_images_per_class} images per class...")
        
        augmented_images = []
        augmented_labels = []
        
        # Process each class
        for class_idx, class_name in enumerate(class_names):
            print(f"\nProcessing class: {class_name}")
            
            # Get images for this class
            class_mask = labels == class_idx
            class_images = images[class_mask]
            current_count = len(class_images)
            
            print(f"  Original images: {current_count}")
            
            if current_count >= self.target_images_per_class:
                print(f"  Class already has enough images, skipping augmentation")
                augmented_images.extend(class_images)
                augmented_labels.extend([class_idx] * current_count)
                self.augmentation_stats[class_name] = {
                    'original': current_count,
                    'augmented': 0,
                    'total': current_count
                }
                continue
            
            # Calculate how many augmentations we need
            needed_augmentations = self.target_images_per_class - current_count
            print(f"  Need {needed_augmentations} more images")
            
            # Add original images
            augmented_images.extend(class_images)
            augmented_labels.extend([class_idx] * current_count)
            
            # Generate augmented images
            if needed_augmentations > 0:
                # Calculate how many times we need to augment each original image
                augmentations_per_image = needed_augmentations // current_count
                remaining_augmentations = needed_augmentations % current_count
                
                print(f"  Generating {augmentations_per_image} augmentations per image")
                if remaining_augmentations > 0:
                    print(f"  Plus {remaining_augmentations} additional augmentations")
                
                # Augment each original image
                for i, original_image in enumerate(class_images):
                    # Calculate how many augmentations for this specific image
                    num_augmentations = augmentations_per_image
                    if i < remaining_augmentations:
                        num_augmentations += 1
                    
                    if num_augmentations > 0:
                        # Create augmented versions
                        if self.use_imgaug:
                            try:
                                augmented_versions = self.augmentation_pipeline.augment_images(
                                    [original_image] * num_augmentations
                                )
                                # Ensure values are in valid range [0, 1]
                                augmented_versions = np.clip(augmented_versions, 0.0, 1.0)
                            except Exception as e:
                                print(f"Warning: imgaug failed, using OpenCV fallback: {e}")
                                # Fallback to OpenCV
                                augmented_versions = []
                                for _ in range(num_augmentations):
                                    aug_img = self._opencv_augment_image(original_image)
                                    augmented_versions.append(aug_img)
                                augmented_versions = np.array(augmented_versions)
                        else:
                            # Use OpenCV-based augmentation
                            augmented_versions = []
                            for _ in range(num_augmentations):
                                aug_img = self._opencv_augment_image(original_image)
                                augmented_versions.append(aug_img)
                            augmented_versions = np.array(augmented_versions)
                        
                        # Add to results
                        augmented_images.extend(augmented_versions)
                        augmented_labels.extend([class_idx] * num_augmentations)
                
                self.augmentation_stats[class_name] = {
                    'original': current_count,
                    'augmented': needed_augmentations,
                    'total': self.target_images_per_class
                }
            
            print(f"  Final count: {len([l for l in augmented_labels if l == class_idx])} images")
        
        # Convert to numpy arrays
        augmented_images = np.array(augmented_images)
        augmented_labels = np.array(augmented_labels)
        
        print(f"\nAugmentation completed!")
        print(f"Original dataset: {len(images)} images")
        print(f"Augmented dataset: {len(augmented_images)} images")
        print(f"Total increase: {len(augmented_images) - len(images)} images")
        
        return augmented_images, augmented_labels
    
    def visualize_augmentations(self, original_image: np.ndarray, 
                              num_augmentations: int = 8, 
                              figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Visualize original image and its augmentations.
        
        Args:
            original_image (np.ndarray): Original image to augment
            num_augmentations (int): Number of augmented versions to show
            figsize (Tuple[int, int]): Figure size for matplotlib
        """
        # Generate augmented images
        if self.use_imgaug:
            try:
                augmented_images = self.augmentation_pipeline.augment_images(
                    [original_image] * num_augmentations
                )
                # Ensure values are in valid range [0, 1]
                augmented_images = np.clip(augmented_images, 0.0, 1.0)
            except Exception as e:
                print(f"Warning: imgaug failed in visualization, using OpenCV fallback: {e}")
                # Fallback to OpenCV
                augmented_images = []
                for _ in range(num_augmentations):
                    aug_img = self._opencv_augment_image(original_image)
                    augmented_images.append(aug_img)
                augmented_images = np.array(augmented_images)
        else:
            # Use OpenCV-based augmentation
            augmented_images = []
            for _ in range(num_augmentations):
                aug_img = self._opencv_augment_image(original_image)
                augmented_images.append(aug_img)
            augmented_images = np.array(augmented_images)
        
        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        axes = axes.ravel()
        
        # Show original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        # Show augmented images
        for i in range(1, min(num_augmentations + 1, len(axes))):
            axes[i].imshow(augmented_images[i-1])
            axes[i].set_title(f"Augmented {i}")
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(num_augmentations + 1, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def save_augmented_dataset(self, images: np.ndarray, labels: np.ndarray,
                             class_names: List[str], 
                             save_path: Optional[Path] = None) -> None:
        """
        Save augmented dataset to disk.
        
        Args:
            images (np.ndarray): Augmented images array
            labels (np.ndarray): Augmented labels array
            class_names (List[str]): List of class names
            save_path (Optional[Path]): Path to save the augmented dataset
        """
        save_path = save_path or AUGMENTED_DATA_PATH
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save images and labels
        np.save(save_path / "augmented_images.npy", images)
        np.save(save_path / "augmented_labels.npy", labels)
        
        # Save class names
        with open(save_path / "class_names.json", 'w') as f:
            json.dump(class_names, f)
        
        # Save augmentation statistics
        with open(save_path / "augmentation_stats.json", 'w') as f:
            json.dump(self.augmentation_stats, f, indent=2)
        
        print(f"Augmented dataset saved to {save_path}")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Classes: {len(class_names)}")
    
    def load_augmented_dataset(self, load_path: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load augmented dataset from disk.
        
        Args:
            load_path (Optional[Path]): Path to load the augmented dataset from
            
        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: Loaded images, labels, and class names
        """
        load_path = load_path or AUGMENTED_DATA_PATH
        
        images = np.load(load_path / "augmented_images.npy")
        labels = np.load(load_path / "augmented_labels.npy")
        
        with open(load_path / "class_names.json", 'r') as f:
            class_names = json.load(f)
        
        # Load augmentation statistics if available
        stats_file = load_path / "augmentation_stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                self.augmentation_stats = json.load(f)
        
        print(f"Augmented dataset loaded from {load_path}")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Classes: {len(class_names)}")
        
        return images, labels, class_names
    
    def get_augmentation_summary(self) -> Dict:
        """
        Get summary of augmentation statistics.
        
        Returns:
            Dict: Augmentation summary statistics
        """
        if not self.augmentation_stats:
            return {"error": "No augmentation statistics available"}
        
        total_original = sum(stats['original'] for stats in self.augmentation_stats.values())
        total_augmented = sum(stats['augmented'] for stats in self.augmentation_stats.values())
        total_final = sum(stats['total'] for stats in self.augmentation_stats.values())
        
        summary = {
            "total_classes": len(self.augmentation_stats),
            "total_original_images": total_original,
            "total_augmented_images": total_augmented,
            "total_final_images": total_final,
            "augmentation_ratio": total_augmented / total_original if total_original > 0 else 0,
            "per_class_stats": self.augmentation_stats
        }
        
        return summary
    
    def plot_augmentation_stats(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot augmentation statistics.
        
        Args:
            figsize (Tuple[int, int]): Figure size for matplotlib
        """
        if not self.augmentation_stats:
            print("No augmentation statistics available")
            return
        
        classes = list(self.augmentation_stats.keys())
        original_counts = [self.augmentation_stats[cls]['original'] for cls in classes]
        augmented_counts = [self.augmentation_stats[cls]['augmented'] for cls in classes]
        total_counts = [self.augmentation_stats[cls]['total'] for cls in classes]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Original vs Augmented
        x = np.arange(len(classes))
        width = 0.35
        
        ax1.bar(x - width/2, original_counts, width, label='Original', alpha=0.8)
        ax1.bar(x + width/2, augmented_counts, width, label='Augmented', alpha=0.8)
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Number of Images')
        ax1.set_title('Original vs Augmented Images per Class')
        ax1.set_xticks(x)
        ax1.set_xticklabels(classes, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Total images per class
        ax2.bar(classes, total_counts, alpha=0.8, color='green')
        ax2.set_xlabel('Classes')
        ax2.set_ylabel('Total Images')
        ax2.set_title('Total Images per Class (After Augmentation)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def main():
    """
    Main function to demonstrate image augmentation.
    """
    print("=== Cerdas Isyarat - Image Augmentation ===")
    
    # Initialize augmenter
    augmenter = BISINDOImageAugmenter(target_images_per_class=50)
    
    # Load original dataset (assuming it's already preprocessed)
    from main import BISINDODatasetLoader
    
    dataset_loader = BISINDODatasetLoader()
    images, labels = dataset_loader.load_dataset()
    class_names = dataset_loader.label_encoder.classes_.tolist()
    
    print(f"Loaded original dataset: {len(images)} images, {len(class_names)} classes")
    
    # Visualize some augmentations
    print("\nVisualizing augmentations for a sample image...")
    sample_image = images[0]
    augmenter.visualize_augmentations(sample_image)
    
    # Perform augmentation
    augmented_images, augmented_labels = augmenter.augment_images(
        images, labels, class_names
    )
    
    # Save augmented dataset
    augmenter.save_augmented_dataset(augmented_images, augmented_labels, class_names)
    
    # Show statistics
    summary = augmenter.get_augmentation_summary()
    print("\n=== Augmentation Summary ===")
    for key, value in summary.items():
        if key != 'per_class_stats':
            print(f"{key}: {value}")
    
    # Plot statistics
    augmenter.plot_augmentation_stats()
    
    print("\nImage augmentation completed successfully!")
    print("Ready for the next step: Hand landmarks extraction with MediaPipe")


if __name__ == "__main__":
    main()
