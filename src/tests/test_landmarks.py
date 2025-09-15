"""
Test script for hand landmarks extraction functionality.
This script demonstrates the landmarks extraction pipeline with sample data.
"""

import numpy as np
import matplotlib.pyplot as plt
from hand_landmarks import BISINDOHandLandmarksExtractor


def create_sample_hand_images():
    """
    Create sample images that simulate hand gestures for testing landmarks extraction.
    """
    np.random.seed(42)
    
    # Create 3 sample images for 2 classes (A and B)
    images = []
    labels = []
    
    # Class A - 2 images with hand-like patterns
    for i in range(2):
        # Create a simple image with hand-like pattern
        img = np.random.rand(224, 224, 3) * 0.3
        
        # Add hand-like structure (simplified)
        # Palm area
        img[80:140, 80:140] += 0.4
        
        # Fingers (vertical lines)
        img[60:100, 90:100] += 0.5  # Thumb
        img[50:120, 110:120] += 0.5  # Index finger
        img[50:120, 130:140] += 0.5  # Middle finger
        img[50:120, 150:160] += 0.5  # Ring finger
        img[50:120, 170:180] += 0.5  # Pinky
        
        images.append(np.clip(img, 0, 1))
        labels.append(0)  # Class A
    
    # Class B - 2 images with different hand pattern
    for i in range(2):
        # Create a different hand pattern
        img = np.random.rand(224, 224, 3) * 0.3
        
        # Different hand pose
        # Palm area (more centered)
        img[100:160, 100:160] += 0.4
        
        # Fingers in different positions
        img[90:150, 110:120] += 0.5  # Thumb
        img[80:140, 120:130] += 0.5  # Index finger
        img[80:140, 140:150] += 0.5  # Middle finger
        img[80:140, 160:170] += 0.5  # Ring finger
        img[80:140, 180:190] += 0.5  # Pinky
        
        images.append(np.clip(img, 0, 1))
        labels.append(1)  # Class B
    
    return np.array(images), np.array(labels), ['A', 'B']


def test_landmarks_extraction():
    """
    Test the hand landmarks extraction functionality with sample data.
    """
    print("=== Testing Hand Landmarks Extraction ===")
    
    # Create sample dataset
    print("Creating sample hand gesture images...")
    images, labels, class_names = create_sample_hand_images()
    print(f"Sample dataset: {len(images)} images, {len(class_names)} classes")
    
    # Initialize landmarks extractor
    print("\nInitializing landmarks extractor...")
    extractor = BISINDOHandLandmarksExtractor()
    
    # Visualize original images
    print("\nVisualizing original images...")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i in range(min(4, len(images))):
        axes[i].imshow(images[i])
        axes[i].set_title(f"Original {class_names[labels[i]]}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
    
    # Test landmark extraction on individual images
    print("\nTesting landmark extraction on individual images...")
    for i, image in enumerate(images):
        print(f"\nImage {i+1} ({class_names[labels[i]]}):")
        landmarks = extractor.extract_landmarks_from_image(image)
        
        if landmarks is not None:
            print(f"  Landmarks extracted: {len(landmarks)} features")
            print(f"  Landmark shape: {landmarks.shape}")
            print(f"  First few landmarks: {landmarks[:10]}")
        else:
            print("  No landmarks detected")
    
    # Visualize landmarks on images
    print("\nVisualizing landmarks on images...")
    for i in range(min(2, len(images))):
        print(f"\nLandmarks visualization for image {i+1}:")
        extractor.visualize_landmarks(images[i])
    
    # Extract landmarks from entire dataset
    print("\nExtracting landmarks from entire dataset...")
    landmarks, valid_labels = extractor.extract_landmarks_from_dataset(
        images, labels, class_names
    )
    
    # Show results
    print(f"\nLandmark extraction results:")
    print(f"Original images: {len(images)}")
    print(f"Successful extractions: {len(landmarks)}")
    print(f"Failed extractions: {len(images) - len(landmarks)}")
    
    if len(landmarks) > 0:
        print(f"Landmarks shape: {landmarks.shape}")
        print(f"Features per sample: {landmarks.shape[1] if len(landmarks.shape) > 1 else 0}")
        
        # Show statistics
        summary = extractor.get_extraction_summary()
        print(f"\nExtraction summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Plot statistics
        extractor.plot_landmark_statistics()
        
        # Test normalization
        print("\nTesting landmark normalization...")
        normalized_landmarks = extractor.normalize_landmarks(landmarks, method='minmax')
        print(f"Normalized landmarks shape: {normalized_landmarks.shape}")
        print(f"Normalized range: [{normalized_landmarks.min():.3f}, {normalized_landmarks.max():.3f}]")
        
        # Test feature extraction
        print("\nTesting feature extraction...")
        features = extractor.get_landmark_features(landmarks)
        print(f"Feature extraction results:")
        for key, value in features.items():
            if key not in ['mean_landmarks', 'std_landmarks', 'min_values', 'max_values', 'zero_landmarks_count']:
                print(f"  {key}: {value}")
    
    print("\nHand landmarks extraction test completed successfully!")


if __name__ == "__main__":
    test_landmarks_extraction()
