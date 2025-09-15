"""
Test script for feature extraction functionality.
This script demonstrates the feature extraction pipeline with sample landmarks data.
"""

import numpy as np
import matplotlib.pyplot as plt
from feature_extraction import BISINDOFeatureExtractor


def create_sample_landmarks():
    """
    Create sample landmarks data for testing feature extraction.
    """
    np.random.seed(42)
    
    # Create 3 sample landmark sets for 2 classes (A and B)
    landmarks = []
    labels = []
    
    # Class A - 2 samples with hand-like landmark patterns
    for i in range(2):
        # Create landmarks for two hands (126 features total: 2 hands * 21 landmarks * 3 coordinates)
        sample_landmarks = []
        
        # Hand 1 - more spread out (like letter A)
        for hand_idx in range(2):
            if hand_idx == 0:  # First hand
                # Create a more spread out hand pattern
                hand_points = np.random.rand(21, 3) * 0.3 + 0.2
                # Make fingers more extended
                hand_points[4] = hand_points[0] + [0.1, 0.2, 0.05]  # Thumb tip
                hand_points[8] = hand_points[0] + [0.15, 0.25, 0.05]  # Index tip
                hand_points[12] = hand_points[0] + [0.2, 0.3, 0.05]  # Middle tip
                hand_points[16] = hand_points[0] + [0.18, 0.28, 0.05]  # Ring tip
                hand_points[20] = hand_points[0] + [0.12, 0.22, 0.05]  # Pinky tip
            else:  # Second hand (not detected - all zeros)
                hand_points = np.zeros((21, 3))
            
            # Flatten to 63 features per hand
            sample_landmarks.extend(hand_points.flatten())
        
        landmarks.append(sample_landmarks)
        labels.append(0)  # Class A
    
    # Class B - 2 samples with different hand pattern
    for i in range(2):
        # Create landmarks for two hands
        sample_landmarks = []
        
        # Hand 1 - more closed (like letter B)
        for hand_idx in range(2):
            if hand_idx == 0:  # First hand
                # Create a more closed hand pattern
                hand_points = np.random.rand(21, 3) * 0.2 + 0.3
                # Make fingers more closed
                hand_points[4] = hand_points[0] + [0.05, 0.1, 0.02]  # Thumb tip
                hand_points[8] = hand_points[0] + [0.08, 0.12, 0.02]  # Index tip
                hand_points[12] = hand_points[0] + [0.1, 0.15, 0.02]  # Middle tip
                hand_points[16] = hand_points[0] + [0.09, 0.13, 0.02]  # Ring tip
                hand_points[20] = hand_points[0] + [0.06, 0.11, 0.02]  # Pinky tip
            else:  # Second hand (not detected - all zeros)
                hand_points = np.zeros((21, 3))
            
            # Flatten to 63 features per hand
            sample_landmarks.extend(hand_points.flatten())
        
        landmarks.append(sample_landmarks)
        labels.append(1)  # Class B
    
    return np.array(landmarks), np.array(labels), ['A', 'B']


def test_feature_extraction():
    """
    Test the feature extraction functionality with sample landmarks data.
    """
    print("=== Testing Feature Extraction ===")
    
    # Create sample landmarks dataset
    print("Creating sample landmarks dataset...")
    landmarks, labels, class_names = create_sample_landmarks()
    print(f"Sample landmarks dataset: {len(landmarks)} samples, {len(class_names)} classes")
    print(f"Landmarks shape: {landmarks.shape}")
    
    # Initialize feature extractor
    print("\nInitializing feature extractor...")
    feature_extractor = BISINDOFeatureExtractor(
        normalize_features=True,
        feature_selection=True,
        n_features=20,  # Small number for testing
        use_pca=False,
        pca_components=10
    )
    
    # Extract features from landmarks
    print("\nExtracting features from landmarks...")
    features, valid_labels = feature_extractor.extract_features_from_dataset(
        landmarks, labels
    )
    
    print(f"Extracted features shape: {features.shape}")
    print(f"Feature names count: {len(feature_extractor.feature_names)}")
    
    # Show some feature names
    print(f"\nFirst 10 feature names:")
    for i, name in enumerate(feature_extractor.feature_names[:10]):
        print(f"  {i+1}. {name}")
    
    # Apply transformations
    print("\nApplying feature transformations...")
    transformed_features = feature_extractor.fit_transform_features(features, valid_labels)
    
    print(f"Transformed features shape: {transformed_features.shape}")
    
    # Show statistics
    summary = feature_extractor.get_feature_summary()
    print(f"\nFeature extraction summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Test individual feature extraction methods
    print("\nTesting individual feature extraction methods...")
    
    # Test geometric features
    geometric_features = feature_extractor.extract_geometric_features(landmarks)
    print(f"Geometric features shape: {geometric_features.shape}")
    
    # Test statistical features
    statistical_features = feature_extractor.extract_statistical_features(landmarks)
    print(f"Statistical features shape: {statistical_features.shape}")
    
    # Visualize features
    print("\nVisualizing feature analysis...")
    try:
        feature_extractor.plot_feature_importance(transformed_features, valid_labels)
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    # Test feature analysis
    print("\nFeature analysis:")
    print(f"Original landmarks: {landmarks.shape}")
    print(f"Extracted features: {features.shape}")
    print(f"Transformed features: {transformed_features.shape}")
    print(f"Feature reduction ratio: {transformed_features.shape[1] / landmarks.shape[1]:.3f}")
    
    # Show feature statistics
    print(f"\nFeature statistics:")
    print(f"  Mean: {transformed_features.mean():.3f}")
    print(f"  Std: {transformed_features.std():.3f}")
    print(f"  Min: {transformed_features.min():.3f}")
    print(f"  Max: {transformed_features.max():.3f}")
    
    print("\nFeature extraction test completed successfully!")


if __name__ == "__main__":
    test_feature_extraction()
