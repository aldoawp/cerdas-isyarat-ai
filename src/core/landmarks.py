"""
Hand Landmarks Extraction Module for BISINDO Alphabet Recognition
This module handles hand landmark extraction using MediaPipe for sign language recognition.
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import pickle

try:
    from ..config import (
        MEDIAPIPE_CONFIG, LANDMARKS_DATA_PATH, RANDOM_SEED
    )
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import (
        MEDIAPIPE_CONFIG, LANDMARKS_DATA_PATH, RANDOM_SEED
    )


class LandmarksExtractor:
    """
    A class to handle hand landmark extraction for BISINDO alphabet dataset using MediaPipe.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the hand landmarks extractor.
        
        Args:
            config (Optional[Dict]): MediaPipe configuration
        """
        self.config = config or MEDIAPIPE_CONFIG
        self.static_image_mode = self.config['static_image_mode']
        self.max_num_hands = self.config['max_num_hands']
        self.min_detection_confidence = self.config['min_detection_confidence']
        self.min_tracking_confidence = self.config['min_tracking_confidence']
        
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize hands model
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_image_mode,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        # Statistics tracking
        self.extraction_stats = {
            'total_images': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'single_hand_detections': 0,
            'dual_hand_detections': 0,
            'no_hand_detections': 0
        }
    
    def extract_landmarks_from_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract hand landmarks from a single image.
        
        Args:
            image (np.ndarray): Input image (RGB format, 0-1 range)
            
        Returns:
            Optional[np.ndarray]: Extracted landmarks array or None if no hands detected
        """
        # Convert image to uint8 format for MediaPipe
        if image.dtype == np.float32 or image.dtype == np.float64:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        # Process image with MediaPipe
        results = self.hands.process(image_uint8)
        
        # Extract landmarks
        landmarks = []
        
        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            
            # Update statistics
            if num_hands == 1:
                self.extraction_stats['single_hand_detections'] += 1
            elif num_hands == 2:
                self.extraction_stats['dual_hand_detections'] += 1
            
            # Extract landmarks for each detected hand
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract 21 landmarks per hand (x, y, z coordinates)
                hand_points = []
                for landmark in hand_landmarks.landmark:
                    hand_points.extend([landmark.x, landmark.y, landmark.z])
                landmarks.extend(hand_points)
            
            # Pad with zeros if only one hand detected (to maintain consistent feature size)
            if num_hands == 1:
                # Add 21 landmarks * 3 coordinates = 63 zeros for the second hand
                landmarks.extend([0.0] * 63)
            
            return np.array(landmarks)
        else:
            self.extraction_stats['no_hand_detections'] += 1
            return None
    
    def extract_landmarks_from_dataset(self, 
                                     images: np.ndarray, 
                                     labels: np.ndarray,
                                     class_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract hand landmarks from a dataset of images.
        
        Args:
            images (np.ndarray): Array of images
            labels (np.ndarray): Array of labels
            class_names (List[str]): List of class names
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Extracted landmarks and corresponding labels
        """
        print(f"Extracting hand landmarks from {len(images)} images...")
        
        extracted_landmarks = []
        valid_labels = []
        failed_indices = []
        
        self.extraction_stats['total_images'] = len(images)
        
        # Process each image
        for i, image in enumerate(tqdm(images, desc="Extracting landmarks")):
            landmarks = self.extract_landmarks_from_image(image)
            
            if landmarks is not None:
                extracted_landmarks.append(landmarks)
                valid_labels.append(labels[i])
                self.extraction_stats['successful_extractions'] += 1
            else:
                failed_indices.append(i)
                self.extraction_stats['failed_extractions'] += 1
        
        # Convert to numpy arrays
        if extracted_landmarks:
            extracted_landmarks = np.array(extracted_landmarks)
            valid_labels = np.array(valid_labels)
        else:
            extracted_landmarks = np.array([])
            valid_labels = np.array([])
        
        print(f"\nLandmark extraction completed!")
        print(f"Successfully extracted: {len(extracted_landmarks)} landmarks")
        print(f"Failed extractions: {len(failed_indices)} images")
        print(f"Success rate: {len(extracted_landmarks)/len(images)*100:.1f}%")
        
        if failed_indices:
            print(f"Failed image indices: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''}")
        
        return extracted_landmarks, valid_labels
    
    def visualize_landmarks(self, 
                          image: np.ndarray, 
                          landmarks: Optional[np.ndarray] = None,
                          figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Visualize hand landmarks on an image.
        
        Args:
            image (np.ndarray): Input image
            landmarks (Optional[np.ndarray]): Pre-extracted landmarks (optional)
            figsize (Tuple[int, int]): Figure size for matplotlib
        """
        # Convert image to uint8 format
        if image.dtype == np.float32 or image.dtype == np.float64:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        # Extract landmarks if not provided
        if landmarks is None:
            landmarks = self.extract_landmarks_from_image(image)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Original image
        axes[0].imshow(image_uint8)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Image with landmarks
        if landmarks is not None:
            # Process image with MediaPipe for visualization
            results = self.hands.process(image_uint8)
            
            # Create annotated image
            annotated_image = image_uint8.copy()
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
            
            axes[1].imshow(annotated_image)
            axes[1].set_title(f"Hand Landmarks Detected")
        else:
            axes[1].imshow(image_uint8)
            axes[1].set_title("No Hand Detected")
        
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()
    
    def plot_landmark_statistics(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot statistics about landmark extraction.
        
        Args:
            figsize (Tuple[int, int]): Figure size for matplotlib
        """
        if self.extraction_stats['total_images'] == 0:
            print("No extraction statistics available")
            return
        
        # Prepare data for plotting
        categories = ['Successful', 'Failed', 'Single Hand', 'Dual Hand', 'No Hand']
        values = [
            self.extraction_stats['successful_extractions'],
            self.extraction_stats['failed_extractions'],
            self.extraction_stats['single_hand_detections'],
            self.extraction_stats['dual_hand_detections'],
            self.extraction_stats['no_hand_detections']
        ]
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Bar chart for extraction results
        colors = ['green', 'red', 'blue', 'orange', 'gray']
        bars = ax1.bar(categories, values, color=colors, alpha=0.7)
        ax1.set_title('Hand Landmark Extraction Statistics')
        ax1.set_ylabel('Number of Images')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value}', ha='center', va='bottom')
        
        # Pie chart for success/failure ratio
        success_rate = self.extraction_stats['successful_extractions'] / self.extraction_stats['total_images'] * 100
        failure_rate = 100 - success_rate
        
        ax2.pie([success_rate, failure_rate], 
                labels=['Successful', 'Failed'], 
                colors=['green', 'red'], 
                autopct='%1.1f%%',
                startangle=90)
        ax2.set_title(f'Extraction Success Rate\n({self.extraction_stats["successful_extractions"]}/{self.extraction_stats["total_images"]} images)')
        
        plt.tight_layout()
        plt.show()
    
    def save_landmarks(self, 
                      landmarks: np.ndarray, 
                      labels: np.ndarray,
                      class_names: List[str],
                      save_path: Optional[Path] = None) -> None:
        """
        Save extracted landmarks to disk.
        
        Args:
            landmarks (np.ndarray): Extracted landmarks array
            labels (np.ndarray): Corresponding labels
            class_names (List[str]): List of class names
            save_path (Optional[Path]): Path to save the landmarks
        """
        save_path = save_path or LANDMARKS_DATA_PATH
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save landmarks and labels
        np.save(save_path / "hand_landmarks.npy", landmarks)
        np.save(save_path / "landmark_labels.npy", labels)
        
        # Save class names
        with open(save_path / "landmark_class_names.json", 'w') as f:
            json.dump(class_names, f)
        
        # Save extraction statistics
        with open(save_path / "landmark_extraction_stats.json", 'w') as f:
            json.dump(self.extraction_stats, f, indent=2)
        
        # Save landmarks metadata
        metadata = {
            'landmarks_shape': landmarks.shape,
            'labels_shape': labels.shape,
            'num_classes': len(class_names),
            'landmarks_per_hand': 21,
            'coordinates_per_landmark': 3,
            'total_features_per_hand': 63,
            'total_features': 126  # 2 hands * 63 features
        }
        
        with open(save_path / "landmarks_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Hand landmarks saved to {save_path}")
        print(f"  Landmarks shape: {landmarks.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Classes: {len(class_names)}")
        print(f"  Features per sample: {landmarks.shape[1] if len(landmarks.shape) > 1 else 0}")
    
    def load_landmarks(self, load_path: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load extracted landmarks from disk.
        
        Args:
            load_path (Optional[Path]): Path to load the landmarks from
            
        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: Loaded landmarks, labels, and class names
        """
        load_path = load_path or LANDMARKS_DATA_PATH
        
        landmarks = np.load(load_path / "hand_landmarks.npy")
        labels = np.load(load_path / "landmark_labels.npy")
        
        with open(load_path / "landmark_class_names.json", 'r') as f:
            class_names = json.load(f)
        
        # Load extraction statistics if available
        stats_file = load_path / "landmark_extraction_stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                self.extraction_stats = json.load(f)
        
        print(f"Hand landmarks loaded from {load_path}")
        print(f"  Landmarks shape: {landmarks.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Classes: {len(class_names)}")
        
        return landmarks, labels, class_names
    
    def get_landmark_features(self, landmarks: np.ndarray) -> Dict:
        """
        Get statistical features from landmarks.
        
        Args:
            landmarks (np.ndarray): Extracted landmarks array
            
        Returns:
            Dict: Statistical features of the landmarks
        """
        if len(landmarks) == 0:
            return {"error": "No landmarks available"}
        
        features = {
            'total_samples': len(landmarks),
            'features_per_sample': landmarks.shape[1] if len(landmarks.shape) > 1 else 0,
            'mean_landmarks': landmarks.mean(axis=0).tolist() if len(landmarks.shape) > 1 else [],
            'std_landmarks': landmarks.std(axis=0).tolist() if len(landmarks.shape) > 1 else [],
            'min_values': landmarks.min(axis=0).tolist() if len(landmarks.shape) > 1 else [],
            'max_values': landmarks.max(axis=0).tolist() if len(landmarks.shape) > 1 else [],
            'zero_landmarks_count': np.sum(landmarks == 0, axis=0).tolist() if len(landmarks.shape) > 1 else []
        }
        
        return features
    
    def normalize_landmarks(self, landmarks: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize landmarks using different methods.
        
        Args:
            landmarks (np.ndarray): Input landmarks array
            method (str): Normalization method ('minmax', 'zscore', 'unit')
            
        Returns:
            np.ndarray: Normalized landmarks
        """
        if method == 'minmax':
            # Min-max normalization to [0, 1]
            min_vals = landmarks.min(axis=0, keepdims=True)
            max_vals = landmarks.max(axis=0, keepdims=True)
            return (landmarks - min_vals) / (max_vals - min_vals + 1e-8)
        
        elif method == 'zscore':
            # Z-score normalization
            mean_vals = landmarks.mean(axis=0, keepdims=True)
            std_vals = landmarks.std(axis=0, keepdims=True)
            return (landmarks - mean_vals) / (std_vals + 1e-8)
        
        elif method == 'unit':
            # Unit vector normalization
            norms = np.linalg.norm(landmarks, axis=1, keepdims=True)
            return landmarks / (norms + 1e-8)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def get_extraction_summary(self) -> Dict:
        """
        Get summary of landmark extraction statistics.
        
        Returns:
            Dict: Extraction summary statistics
        """
        if self.extraction_stats['total_images'] == 0:
            return {"error": "No extraction statistics available"}
        
        success_rate = self.extraction_stats['successful_extractions'] / self.extraction_stats['total_images'] * 100
        
        summary = {
            "total_images_processed": self.extraction_stats['total_images'],
            "successful_extractions": self.extraction_stats['successful_extractions'],
            "failed_extractions": self.extraction_stats['failed_extractions'],
            "success_rate_percent": success_rate,
            "single_hand_detections": self.extraction_stats['single_hand_detections'],
            "dual_hand_detections": self.extraction_stats['dual_hand_detections'],
            "no_hand_detections": self.extraction_stats['no_hand_detections']
        }
        
        return summary


def main():
    """
    Main function to demonstrate hand landmark extraction.
    """
    print("=== Cerdas Isyarat - Hand Landmarks Extraction ===")
    
    # Initialize landmarks extractor
    extractor = BISINDOHandLandmarksExtractor()
    
    # Load augmented dataset
    try:
        from augmentation import BISINDOImageAugmenter
        augmenter = BISINDOImageAugmenter()
        images, labels, class_names = augmenter.load_augmented_dataset()
        
        print(f"Loaded augmented dataset: {len(images)} images, {len(class_names)} classes")
        
        # Visualize some landmark extractions
        print("\nVisualizing landmark extraction for sample images...")
        for i in range(min(3, len(images))):
            print(f"\nSample {i+1}:")
            extractor.visualize_landmarks(images[i])
        
        # Extract landmarks from dataset
        landmarks, valid_labels = extractor.extract_landmarks_from_dataset(
            images, labels, class_names
        )
        
        # Save landmarks
        extractor.save_landmarks(landmarks, valid_labels, class_names)
        
        # Show statistics
        summary = extractor.get_extraction_summary()
        print("\n=== Landmark Extraction Summary ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # Plot statistics
        extractor.plot_landmark_statistics()
        
        print("\nHand landmark extraction completed successfully!")
        print("Ready for the next step: Feature extraction from landmarks")
        
    except FileNotFoundError:
        print("Augmented dataset not found. Please run augmentation first.")
    except ImportError:
        print("Could not import augmentation module. Please ensure it's available.")


if __name__ == "__main__":
    main()
