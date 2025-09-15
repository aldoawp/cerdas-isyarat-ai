"""
Test script for image augmentation functionality.
This script demonstrates the augmentation pipeline without requiring the full dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from augmentation import BISINDOImageAugmenter


def create_sample_dataset():
    """
    Create a small sample dataset for testing augmentation.
    """
    # Create sample images (simulating hand gesture images)
    np.random.seed(42)
    
    # Create 3 sample images for 2 classes (A and B)
    images = []
    labels = []
    
    # Class A - 2 images
    for i in range(2):
        # Create a simple image with some pattern
        img = np.random.rand(224, 224, 3) * 0.3
        # Add some structure to make it look more like a hand gesture
        img[50:150, 50:150] += 0.4  # Square pattern
        img[100:120, 100:120] += 0.3  # Inner square
        images.append(np.clip(img, 0, 1))
        labels.append(0)  # Class A
    
    # Class B - 2 images  
    for i in range(2):
        # Create a different pattern
        img = np.random.rand(224, 224, 3) * 0.3
        # Add circular pattern
        y, x = np.ogrid[:224, :224]
        center_x, center_y = 112, 112
        mask = (x - center_x)**2 + (y - center_y)**2 <= 50**2
        img[mask] += 0.4
        images.append(np.clip(img, 0, 1))
        labels.append(1)  # Class B
    
    return np.array(images), np.array(labels), ['A', 'B']


def test_augmentation():
    """
    Test the augmentation functionality with sample data.
    """
    print("=== Testing Image Augmentation ===")
    
    # Create sample dataset
    print("Creating sample dataset...")
    images, labels, class_names = create_sample_dataset()
    print(f"Sample dataset: {len(images)} images, {len(class_names)} classes")
    print(f"Images per class: {[np.sum(labels == i) for i in range(len(class_names))]}")
    
    # Initialize augmenter
    print("\nInitializing augmenter...")
    augmenter = BISINDOImageAugmenter(target_images_per_class=10)  # Small target for testing
    
    # Visualize original images
    print("\nVisualizing original images...")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i in range(min(4, len(images))):
        axes[i].imshow(images[i])
        axes[i].set_title(f"Original {class_names[labels[i]]}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
    
    # Visualize augmentations
    print("\nVisualizing augmentations for first image...")
    augmenter.visualize_augmentations(images[0], num_augmentations=8)
    
    # Perform augmentation
    print("\nPerforming augmentation...")
    augmented_images, augmented_labels = augmenter.augment_images(
        images, labels, class_names
    )
    
    # Show results
    print(f"\nAugmentation results:")
    print(f"Original: {len(images)} images")
    print(f"Augmented: {len(augmented_images)} images")
    print(f"Images per class after augmentation:")
    for i, class_name in enumerate(class_names):
        count = np.sum(augmented_labels == i)
        print(f"  {class_name}: {count} images")
    
    # Show statistics
    summary = augmenter.get_augmentation_summary()
    print(f"\nAugmentation summary:")
    for key, value in summary.items():
        if key != 'per_class_stats':
            print(f"  {key}: {value}")
    
    # Visualize some augmented images
    print("\nVisualizing augmented images...")
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    # Show first 10 augmented images
    for i in range(min(10, len(augmented_images))):
        axes[i].imshow(augmented_images[i])
        axes[i].set_title(f"Aug {class_names[augmented_labels[i]]}")
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(10, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\nAugmentation test completed successfully!")


if __name__ == "__main__":
    test_augmentation()
