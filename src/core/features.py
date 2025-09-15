"""
Feature Extraction Module for BISINDO Alphabet Recognition
This module extracts meaningful features from hand landmarks for machine learning.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Optional import for seaborn
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Import configuration with fallback
try:
    from ..config import (
        FEATURE_EXTRACTION_CONFIG, FEATURES_DATA_PATH, RANDOM_SEED
    )
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import (
        FEATURE_EXTRACTION_CONFIG, FEATURES_DATA_PATH, RANDOM_SEED
    )


class FeatureExtractor:
    """
    A class to extract meaningful features from hand landmarks for BISINDO alphabet recognition.
    """
    
    def __init__(self, 
                 normalize_features: bool = True,
                 feature_selection: bool = True,
                 n_features: int = 50,
                 use_pca: bool = False,
                 pca_components: int = 30,
                 random_state: int = 42):
        """
        Initialize the feature extractor.
        
        Args:
            normalize_features (bool): Whether to normalize features
            feature_selection (bool): Whether to perform feature selection
            n_features (int): Number of features to select
            use_pca (bool): Whether to use PCA for dimensionality reduction
            pca_components (int): Number of PCA components
            random_state (int): Random seed for reproducibility
        """
        self.normalize_features = normalize_features
        self.feature_selection = feature_selection
        self.n_features = n_features
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.random_state = random_state
        
        # Initialize transformers
        self.scaler = StandardScaler() if normalize_features else None
        self.feature_selector = None
        self.pca = PCA(n_components=pca_components, random_state=random_state) if use_pca else None
        
        # Statistics tracking
        self.extraction_stats = {
            'total_samples': 0,
            'original_features': 0,
            'extracted_features': 0,
            'selected_features': 0,
            'final_features': 0
        }
        
        # Feature names and importance
        self.feature_names = []
        self.feature_importance = None
        self.is_fitted = False
    
    def extract_geometric_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract geometric features from hand landmarks.
        
        Args:
            landmarks (np.ndarray): Hand landmarks array (N, 126)
            
        Returns:
            np.ndarray: Geometric features array
        """
        features = []
        feature_names = []
        
        for i in range(len(landmarks)):
            sample_features = []
            
            # Reshape landmarks to (2, 21, 3) for two hands
            hand_landmarks = landmarks[i].reshape(2, 21, 3)
            
            for hand_idx in range(2):
                hand_points = hand_landmarks[hand_idx]  # (21, 3)
                
                # Skip if hand is not detected (all zeros)
                if np.all(hand_points == 0):
                    # Add zero features for missing hand
                    sample_features.extend([0] * 50)  # 50 features per hand
                    continue
                
                # 1. Distances between key points
                distances = self._calculate_distances(hand_points)
                sample_features.extend(distances)
                
                # 2. Angles between key points
                angles = self._calculate_angles(hand_points)
                sample_features.extend(angles)
                
                # 3. Hand area and perimeter
                area_perimeter = self._calculate_area_perimeter(hand_points)
                sample_features.extend(area_perimeter)
                
                # 4. Finger lengths
                finger_lengths = self._calculate_finger_lengths(hand_points)
                sample_features.extend(finger_lengths)
                
                # 5. Hand orientation
                orientation = self._calculate_hand_orientation(hand_points)
                sample_features.extend(orientation)
            
            features.append(sample_features)
        
        # Create feature names
        if not self.feature_names:
            for hand_idx in range(2):
                hand_name = f"hand_{hand_idx + 1}"
                feature_names.extend([f"{hand_name}_distance_{i}" for i in range(10)])  # 10 distances
                feature_names.extend([f"{hand_name}_angle_{i}" for i in range(15)])     # 15 angles
                feature_names.extend([f"{hand_name}_area", f"{hand_name}_perimeter"])   # 2 area/perimeter
                feature_names.extend([f"{hand_name}_finger_{i}" for i in range(5)])     # 5 finger lengths
                feature_names.extend([f"{hand_name}_orientation_{i}" for i in range(3)]) # 3 orientation
            self.feature_names = feature_names
        
        # Ensure all samples have the same number of features
        expected_features = len(self.feature_names)
        for i, sample_features in enumerate(features):
            if len(sample_features) != expected_features:
                # Pad with zeros if too short, truncate if too long
                if len(sample_features) < expected_features:
                    sample_features.extend([0.0] * (expected_features - len(sample_features)))
                else:
                    sample_features = sample_features[:expected_features]
                features[i] = sample_features
        
        return np.array(features)
    
    def _calculate_distances(self, hand_points: np.ndarray) -> List[float]:
        """Calculate distances between key hand landmarks."""
        distances = []
        
        # Key landmark indices for distance calculations
        key_points = [
            (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),  # Wrist to finger bases
            (4, 8), (8, 12), (12, 16), (16, 20),       # Fingertips
            (1, 2), (5, 6), (9, 10), (13, 14), (17, 18) # Finger joints
        ]
        
        for i, j in key_points:
            if i < len(hand_points) and j < len(hand_points):
                dist = np.linalg.norm(hand_points[i] - hand_points[j])
                distances.append(dist)
            else:
                distances.append(0.0)
        
        return distances[:10]  # Return first 10 distances
    
    def _calculate_angles(self, hand_points: np.ndarray) -> List[float]:
        """Calculate angles between key hand landmarks."""
        angles = []
        
        # Key landmark triplets for angle calculations
        angle_points = [
            (0, 1, 2), (1, 2, 3), (2, 3, 4),           # Thumb angles
            (0, 5, 6), (5, 6, 7), (6, 7, 8),           # Index finger angles
            (0, 9, 10), (9, 10, 11), (10, 11, 12),     # Middle finger angles
            (0, 13, 14), (13, 14, 15), (14, 15, 16),   # Ring finger angles
            (0, 17, 18), (17, 18, 19), (18, 19, 20)    # Pinky angles
        ]
        
        for i, j, k in angle_points:
            if i < len(hand_points) and j < len(hand_points) and k < len(hand_points):
                # Calculate angle between vectors
                v1 = hand_points[j] - hand_points[i]
                v2 = hand_points[k] - hand_points[j]
                
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical errors
                    angle = np.arccos(cos_angle)
                    angles.append(angle)
                else:
                    angles.append(0.0)
            else:
                angles.append(0.0)
        
        return angles[:15]  # Return first 15 angles
    
    def _calculate_area_perimeter(self, hand_points: np.ndarray) -> List[float]:
        """Calculate hand area and perimeter."""
        # Use convex hull for area calculation
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(hand_points[:, :2])  # Use only x, y coordinates
            area = hull.volume  # In 2D, volume is area
            perimeter = hull.area  # In 2D, area is perimeter
        except ImportError:
            # Fallback: simple approximation
            area = 0.0
            perimeter = 0.0
            for i in range(len(hand_points) - 1):
                perimeter += np.linalg.norm(hand_points[i+1] - hand_points[i])
        
        return [area, perimeter]
    
    def _calculate_finger_lengths(self, hand_points: np.ndarray) -> List[float]:
        """Calculate individual finger lengths."""
        finger_lengths = []
        
        # Finger landmark sequences
        finger_sequences = [
            [0, 1, 2, 3, 4],      # Thumb
            [0, 5, 6, 7, 8],      # Index
            [0, 9, 10, 11, 12],   # Middle
            [0, 13, 14, 15, 16],  # Ring
            [0, 17, 18, 19, 20]   # Pinky
        ]
        
        for finger_seq in finger_sequences:
            length = 0.0
            for i in range(len(finger_seq) - 1):
                if finger_seq[i] < len(hand_points) and finger_seq[i+1] < len(hand_points):
                    length += np.linalg.norm(hand_points[finger_seq[i+1]] - hand_points[finger_seq[i]])
            finger_lengths.append(length)
        
        return finger_lengths
    
    def _calculate_hand_orientation(self, hand_points: np.ndarray) -> List[float]:
        """Calculate hand orientation features."""
        # Use wrist and middle finger base for orientation
        wrist = hand_points[0]
        middle_base = hand_points[9]
        
        # Calculate orientation vector
        orientation_vector = middle_base - wrist
        
        # Calculate angles
        angle_x = np.arctan2(orientation_vector[1], orientation_vector[0])
        angle_y = np.arctan2(orientation_vector[2], orientation_vector[0])
        angle_z = np.arctan2(orientation_vector[2], orientation_vector[1])
        
        return [angle_x, angle_y, angle_z]
    
    def extract_statistical_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from landmarks.
        
        Args:
            landmarks (np.ndarray): Hand landmarks array (N, 126)
            
        Returns:
            np.ndarray: Statistical features array
        """
        features = []
        
        for i in range(len(landmarks)):
            sample_features = []
            
            # Reshape landmarks to (2, 21, 3) for two hands
            hand_landmarks = landmarks[i].reshape(2, 21, 3)
            
            for hand_idx in range(2):
                hand_points = hand_landmarks[hand_idx]  # (21, 3)
                
                # Skip if hand is not detected
                if np.all(hand_points == 0):
                    sample_features.extend([0] * 12)  # 12 statistical features per hand
                    continue
                
                # Statistical features for each coordinate
                for coord in range(3):  # x, y, z
                    coord_values = hand_points[:, coord]
                    
                    # Basic statistics
                    sample_features.append(np.mean(coord_values))
                    sample_features.append(np.std(coord_values))
                    sample_features.append(np.min(coord_values))
                    sample_features.append(np.max(coord_values))
            
            features.append(sample_features)
        
        # Ensure all samples have the same number of features (24 total: 12 per hand * 2 hands)
        expected_features = 24
        for i, sample_features in enumerate(features):
            if len(sample_features) != expected_features:
                # Pad with zeros if too short, truncate if too long
                if len(sample_features) < expected_features:
                    sample_features.extend([0.0] * (expected_features - len(sample_features)))
                else:
                    sample_features = sample_features[:expected_features]
                features[i] = sample_features
        
        return np.array(features)
    
    def extract_features_from_dataset(self, 
                                    landmarks: np.ndarray, 
                                    labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from landmarks dataset.
        
        Args:
            landmarks (np.ndarray): Landmarks array
            labels (np.ndarray): Labels array
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Extracted features and labels
        """
        print(f"Extracting features from {len(landmarks)} landmark samples...")
        
        self.extraction_stats['total_samples'] = len(landmarks)
        self.extraction_stats['original_features'] = landmarks.shape[1]
        
        # Extract geometric features
        print("Extracting geometric features...")
        geometric_features = self.extract_geometric_features(landmarks)
        
        # Extract statistical features
        print("Extracting statistical features...")
        statistical_features = self.extract_statistical_features(landmarks)
        
        # Combine features
        all_features = np.hstack([geometric_features, statistical_features])
        
        self.extraction_stats['extracted_features'] = all_features.shape[1]
        
        print(f"Feature extraction completed!")
        print(f"Original landmarks: {landmarks.shape}")
        print(f"Extracted features: {all_features.shape}")
        print(f"Geometric features: {geometric_features.shape[1]}")
        print(f"Statistical features: {statistical_features.shape[1]}")
        
        return all_features, labels
    
    def fit_transform_features(self, 
                             features: np.ndarray, 
                             labels: np.ndarray) -> np.ndarray:
        """
        Apply feature transformations (normalization, selection, PCA).
        
        Args:
            features (np.ndarray): Input features
            labels (np.ndarray): Labels for feature selection
            
        Returns:
            np.ndarray: Transformed features
        """
        print("Applying feature transformations...")
        
        transformed_features = features.copy()
        
        # 1. Normalization
        if self.normalize_features and self.scaler is not None:
            print("Applying feature normalization...")
            transformed_features = self.scaler.fit_transform(transformed_features)
        
        # 2. Feature selection
        if self.feature_selection:
            print(f"Applying feature selection (selecting {self.n_features} features)...")
            self.feature_selector = SelectKBest(
                score_func=f_classif, 
                k=min(self.n_features, transformed_features.shape[1])
            )
            transformed_features = self.feature_selector.fit_transform(transformed_features, labels)
            self.extraction_stats['selected_features'] = transformed_features.shape[1]
        
        # 3. PCA
        if self.use_pca and self.pca is not None:
            print(f"Applying PCA (reducing to {self.pca_components} components)...")
            transformed_features = self.pca.fit_transform(transformed_features)
            self.extraction_stats['final_features'] = transformed_features.shape[1]
        
        if not self.use_pca:
            self.extraction_stats['final_features'] = transformed_features.shape[1]
        
        self.is_fitted = True
        
        print(f"Feature transformation completed!")
        print(f"Final features shape: {transformed_features.shape}")
        
        return transformed_features
    
    def plot_feature_importance(self, 
                              features: np.ndarray, 
                              labels: np.ndarray,
                              figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot feature importance and analysis.
        
        Args:
            features (np.ndarray): Features array
            labels (np.ndarray): Labels array
            figsize (Tuple[int, int]): Figure size
        """
        if not self.is_fitted:
            print("Features not fitted yet. Call fit_transform_features first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Feature importance (if feature selection was used)
        if self.feature_selector is not None:
            scores = self.feature_selector.scores_
            selected_indices = self.feature_selector.get_support(indices=True)
            
            # Plot top features
            top_features = sorted(zip(selected_indices, scores[selected_indices]), 
                                key=lambda x: x[1], reverse=True)[:20]
            
            indices, scores = zip(*top_features)
            feature_names = [self.feature_names[i] for i in indices]
            
            axes[0, 0].barh(range(len(feature_names)), scores)
            axes[0, 0].set_yticks(range(len(feature_names)))
            axes[0, 0].set_yticklabels(feature_names, fontsize=8)
            axes[0, 0].set_xlabel('F-Score')
            axes[0, 0].set_title('Top 20 Feature Importance')
        
        # 2. Feature distribution
        axes[0, 1].hist(features.flatten(), bins=50, alpha=0.7)
        axes[0, 1].set_xlabel('Feature Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Feature Value Distribution')
        
        # 3. PCA explained variance (if PCA was used)
        if self.pca is not None:
            explained_variance = self.pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            axes[1, 0].plot(range(1, len(explained_variance) + 1), explained_variance, 'bo-')
            axes[1, 0].set_xlabel('Principal Component')
            axes[1, 0].set_ylabel('Explained Variance Ratio')
            axes[1, 0].set_title('PCA Explained Variance')
            
            axes[1, 1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
            axes[1, 1].set_xlabel('Principal Component')
            axes[1, 1].set_ylabel('Cumulative Explained Variance')
            axes[1, 1].set_title('PCA Cumulative Explained Variance')
            axes[1, 1].axhline(y=0.95, color='g', linestyle='--', label='95%')
            axes[1, 1].legend()
        else:
            # Feature correlation heatmap (sample)
            if features.shape[1] <= 20:
                corr_matrix = np.corrcoef(features.T)
                if SEABORN_AVAILABLE:
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
                else:
                    im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
                    axes[1, 0].set_title('Feature Correlation Matrix')
                    plt.colorbar(im, ax=axes[1, 0])
            
            # Class distribution
            unique_labels, counts = np.unique(labels, return_counts=True)
            axes[1, 1].bar(unique_labels, counts)
            axes[1, 1].set_xlabel('Class')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Class Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def save_features(self, 
                     features: np.ndarray, 
                     labels: np.ndarray,
                     class_names: List[str],
                     save_path: str = "dataset/processed/03_features") -> None:
        """
        Save extracted features to disk.
        
        Args:
            features (np.ndarray): Extracted features array
            labels (np.ndarray): Corresponding labels
            class_names (List[str]): List of class names
            save_path (str): Path to save the features
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save features and labels
        np.save(save_path / "extracted_features.npy", features)
        np.save(save_path / "feature_labels.npy", labels)
        
        # Save class names
        with open(save_path / "feature_class_names.json", 'w') as f:
            json.dump(class_names, f)
        
        # Save extraction statistics
        with open(save_path / "feature_extraction_stats.json", 'w') as f:
            json.dump(self.extraction_stats, f, indent=2)
        
        # Save feature names
        with open(save_path / "feature_names.json", 'w') as f:
            json.dump(self.feature_names, f)
        
        # Save transformers (if fitted)
        if self.is_fitted:
            import pickle
            
            if self.scaler is not None:
                with open(save_path / "feature_scaler.pkl", 'wb') as f:
                    pickle.dump(self.scaler, f)
            
            if self.feature_selector is not None:
                with open(save_path / "feature_selector.pkl", 'wb') as f:
                    pickle.dump(self.feature_selector, f)
            
            if self.pca is not None:
                with open(save_path / "feature_pca.pkl", 'wb') as f:
                    pickle.dump(self.pca, f)
        
        print(f"Extracted features saved to {save_path}")
        print(f"  Features shape: {features.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Classes: {len(class_names)}")
    
    def load_features(self, load_path: str = "dataset/processed/03_features") -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load extracted features from disk.
        
        Args:
            load_path (str): Path to load the features from
            
        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: Loaded features, labels, and class names
        """
        load_path = Path(load_path)
        
        features = np.load(load_path / "extracted_features.npy")
        labels = np.load(load_path / "feature_labels.npy")
        
        with open(load_path / "feature_class_names.json", 'r') as f:
            class_names = json.load(f)
        
        # Load extraction statistics
        stats_file = load_path / "feature_extraction_stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                self.extraction_stats = json.load(f)
        
        # Load feature names
        names_file = load_path / "feature_names.json"
        if names_file.exists():
            with open(names_file, 'r') as f:
                self.feature_names = json.load(f)
        
        print(f"Extracted features loaded from {load_path}")
        print(f"  Features shape: {features.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Classes: {len(class_names)}")
        
        return features, labels, class_names
    
    def get_feature_summary(self) -> Dict:
        """
        Get summary of feature extraction statistics.
        
        Returns:
            Dict: Feature extraction summary
        """
        summary = {
            "total_samples": self.extraction_stats['total_samples'],
            "original_landmark_features": self.extraction_stats['original_features'],
            "extracted_features": self.extraction_stats['extracted_features'],
            "selected_features": self.extraction_stats.get('selected_features', 0),
            "final_features": self.extraction_stats['final_features'],
            "normalization_applied": self.normalize_features,
            "feature_selection_applied": self.feature_selection,
            "pca_applied": self.use_pca,
            "feature_reduction_ratio": self.extraction_stats['final_features'] / self.extraction_stats['original_features'] if self.extraction_stats['original_features'] > 0 else 0
        }
        
        return summary


def main():
    """
    Main function to demonstrate feature extraction.
    """
    print("=== Cerdas Isyarat - Feature Extraction ===")
    
    # Initialize feature extractor
    feature_extractor = BISINDOFeatureExtractor(
        normalize_features=True,
        feature_selection=True,
        n_features=50,
        use_pca=False,
        pca_components=30
    )
    
    # Load landmarks dataset
    try:
        from hand_landmarks import BISINDOHandLandmarksExtractor
        landmarks_extractor = BISINDOHandLandmarksExtractor()
        landmarks, labels, class_names = landmarks_extractor.load_landmarks()
        
        print(f"Loaded landmarks dataset: {len(landmarks)} samples, {len(class_names)} classes")
        
        # Extract features from landmarks
        features, valid_labels = feature_extractor.extract_features_from_dataset(
            landmarks, labels
        )
        
        # Apply transformations
        transformed_features = feature_extractor.fit_transform_features(features, valid_labels)
        
        # Save features
        feature_extractor.save_features(transformed_features, valid_labels, class_names)
        
        # Show statistics
        summary = feature_extractor.get_feature_summary()
        print("\n=== Feature Extraction Summary ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # Plot feature analysis
        feature_extractor.plot_feature_importance(transformed_features, valid_labels)
        
        print("\nFeature extraction completed successfully!")
        print("Ready for the next step: Train and evaluate machine learning models")
        
    except FileNotFoundError:
        print("Landmarks dataset not found. Please run landmarks extraction first.")
    except ImportError:
        print("Could not import landmarks module. Please ensure it's available.")


if __name__ == "__main__":
    main()
