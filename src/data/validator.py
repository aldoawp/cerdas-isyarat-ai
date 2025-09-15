"""
Data Validation module for BISINDO Alphabet Recognition.
This module provides validation utilities for dataset quality and integrity.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import cv2

try:
    from ..config import VALIDATION_CONFIG, IMAGE_SIZE
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import VALIDATION_CONFIG, IMAGE_SIZE


class DataValidator:
    """
    A class to validate dataset quality and integrity.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the data validator.
        
        Args:
            config (Optional[Dict]): Validation configuration
        """
        self.config = config or VALIDATION_CONFIG
        self.validation_results = {}
    
    def validate_images(self, images: np.ndarray, labels: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
        """
        Validate image dataset quality.
        
        Args:
            images (np.ndarray): Image array
            labels (np.ndarray): Label array
            class_names (List[str]): List of class names
            
        Returns:
            Dict[str, Any]: Validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check basic shape and type
        if not isinstance(images, np.ndarray):
            results['errors'].append("Images must be a numpy array")
            results['valid'] = False
        
        if not isinstance(labels, np.ndarray):
            results['errors'].append("Labels must be a numpy array")
            results['valid'] = False
        
        if len(images) != len(labels):
            results['errors'].append(f"Images and labels length mismatch: {len(images)} vs {len(labels)}")
            results['valid'] = False
        
        # Check image dimensions
        if len(images.shape) != 4:
            results['errors'].append(f"Images must be 4D array (samples, height, width, channels), got {len(images.shape)}D")
            results['valid'] = False
        
        if images.shape[1:] != IMAGE_SIZE:
            results['warnings'].append(f"Image size mismatch: expected {IMAGE_SIZE}, got {images.shape[1:]}")
        
        # Check data types
        if images.dtype != np.float32:
            results['warnings'].append(f"Images should be float32, got {images.dtype}")
        
        # Check pixel value range
        if images.min() < 0 or images.max() > 1:
            results['warnings'].append(f"Pixel values should be in [0, 1] range, got [{images.min():.3f}, {images.max():.3f}]")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(images)):
            results['errors'].append("Images contain NaN values")
            results['valid'] = False
        
        if np.any(np.isinf(images)):
            results['errors'].append("Images contain infinite values")
            results['valid'] = False
        
        # Check labels
        unique_labels = np.unique(labels)
        if len(unique_labels) != len(class_names):
            results['errors'].append(f"Label count mismatch: {len(unique_labels)} unique labels vs {len(class_names)} class names")
            results['valid'] = False
        
        if not np.array_equal(unique_labels, np.arange(len(class_names))):
            results['errors'].append("Labels must be consecutive integers starting from 0")
            results['valid'] = False
        
        # Check class distribution
        class_counts = np.bincount(labels)
        min_count = class_counts.min()
        max_count = class_counts.max()
        
        if min_count < self.config['min_images_per_class']:
            results['warnings'].append(f"Some classes have fewer than {self.config['min_images_per_class']} images (min: {min_count})")
        
        if max_count > self.config['max_images_per_class']:
            results['warnings'].append(f"Some classes have more than {self.config['max_images_per_class']} images (max: {max_count})")
        
        # Calculate statistics
        results['statistics'] = {
            'total_images': len(images),
            'total_classes': len(class_names),
            'image_shape': images.shape,
            'class_distribution': dict(zip(class_names, class_counts)),
            'min_class_count': int(min_count),
            'max_class_count': int(max_count),
            'mean_class_count': float(np.mean(class_counts)),
            'std_class_count': float(np.std(class_counts))
        }
        
        self.validation_results['images'] = results
        return results
    
    def validate_landmarks(self, landmarks: np.ndarray, labels: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
        """
        Validate landmarks dataset quality.
        
        Args:
            landmarks (np.ndarray): Landmarks array
            labels (np.ndarray): Label array
            class_names (List[str]): List of class names
            
        Returns:
            Dict[str, Any]: Validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check basic shape and type
        if not isinstance(landmarks, np.ndarray):
            results['errors'].append("Landmarks must be a numpy array")
            results['valid'] = False
        
        if len(landmarks) != len(labels):
            results['errors'].append(f"Landmarks and labels length mismatch: {len(landmarks)} vs {len(labels)}")
            results['valid'] = False
        
        # Check landmarks dimensions (should be 2D: samples x features)
        if len(landmarks.shape) != 2:
            results['errors'].append(f"Landmarks must be 2D array (samples, features), got {len(landmarks.shape)}D")
            results['valid'] = False
        
        # Check expected feature count (126 for 2 hands: 21 landmarks × 3 coordinates × 2 hands)
        expected_features = 126
        if landmarks.shape[1] != expected_features:
            results['warnings'].append(f"Expected {expected_features} features, got {landmarks.shape[1]}")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(landmarks)):
            results['errors'].append("Landmarks contain NaN values")
            results['valid'] = False
        
        if np.any(np.isinf(landmarks)):
            results['errors'].append("Landmarks contain infinite values")
            results['valid'] = False
        
        # Check for zero landmarks (failed detections)
        zero_landmarks = np.all(landmarks == 0, axis=1)
        zero_count = np.sum(zero_landmarks)
        
        if zero_count > 0:
            results['warnings'].append(f"{zero_count} samples have zero landmarks (failed detections)")
        
        # Check landmark value ranges (should be normalized coordinates)
        if landmarks.min() < 0 or landmarks.max() > 1:
            results['warnings'].append(f"Landmark values should be in [0, 1] range, got [{landmarks.min():.3f}, {landmarks.max():.3f}]")
        
        # Calculate statistics
        results['statistics'] = {
            'total_samples': len(landmarks),
            'total_features': landmarks.shape[1],
            'failed_detections': int(zero_count),
            'success_rate': float(1 - zero_count / len(landmarks)),
            'min_value': float(landmarks.min()),
            'max_value': float(landmarks.max()),
            'mean_value': float(landmarks.mean()),
            'std_value': float(landmarks.std())
        }
        
        self.validation_results['landmarks'] = results
        return results
    
    def validate_features(self, features: np.ndarray, labels: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
        """
        Validate features dataset quality.
        
        Args:
            features (np.ndarray): Features array
            labels (np.ndarray): Label array
            class_names (List[str]): List of class names
            
        Returns:
            Dict[str, Any]: Validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check basic shape and type
        if not isinstance(features, np.ndarray):
            results['errors'].append("Features must be a numpy array")
            results['valid'] = False
        
        if len(features) != len(labels):
            results['errors'].append(f"Features and labels length mismatch: {len(features)} vs {len(labels)}")
            results['valid'] = False
        
        # Check features dimensions (should be 2D: samples x features)
        if len(features.shape) != 2:
            results['errors'].append(f"Features must be 2D array (samples, features), got {len(features.shape)}D")
            results['valid'] = False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(features)):
            results['errors'].append("Features contain NaN values")
            results['valid'] = False
        
        if np.any(np.isinf(features)):
            results['errors'].append("Features contain infinite values")
            results['valid'] = False
        
        # Check for constant features (zero variance)
        feature_vars = np.var(features, axis=0)
        constant_features = np.sum(feature_vars == 0)
        
        if constant_features > 0:
            results['warnings'].append(f"{constant_features} features have zero variance (constant)")
        
        # Check feature correlation (high correlation might indicate redundancy)
        if features.shape[1] > 1:
            corr_matrix = np.corrcoef(features.T)
            high_corr_pairs = np.sum((np.abs(corr_matrix) > 0.95) & (corr_matrix != 1.0))
            
            if high_corr_pairs > 0:
                results['warnings'].append(f"{high_corr_pairs} feature pairs have high correlation (>0.95)")
        
        # Calculate statistics
        results['statistics'] = {
            'total_samples': len(features),
            'total_features': features.shape[1],
            'constant_features': int(constant_features),
            'min_value': float(features.min()),
            'max_value': float(features.max()),
            'mean_value': float(features.mean()),
            'std_value': float(features.std()),
            'feature_variance_range': [float(feature_vars.min()), float(feature_vars.max())]
        }
        
        self.validation_results['features'] = results
        return results
    
    def validate_file_integrity(self, file_path: Path) -> Dict[str, Any]:
        """
        Validate file integrity and accessibility.
        
        Args:
            file_path (Path): Path to the file to validate
            
        Returns:
            Dict[str, Any]: Validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check if file exists
        if not file_path.exists():
            results['errors'].append(f"File does not exist: {file_path}")
            results['valid'] = False
            return results
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size == 0:
            results['errors'].append("File is empty")
            results['valid'] = False
        
        # Check file extension and try to load
        try:
            if file_path.suffix == '.npy':
                data = np.load(file_path)
                results['statistics']['shape'] = data.shape
                results['statistics']['dtype'] = str(data.dtype)
                results['statistics']['size_mb'] = file_size / (1024 * 1024)
            elif file_path.suffix == '.json':
                import json
                with open(file_path, 'r') as f:
                    data = json.load(f)
                results['statistics']['type'] = type(data).__name__
                results['statistics']['size_mb'] = file_size / (1024 * 1024)
            elif file_path.suffix == '.pkl':
                import pickle
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                results['statistics']['type'] = type(data).__name__
                results['statistics']['size_mb'] = file_size / (1024 * 1024)
            else:
                results['warnings'].append(f"Unknown file type: {file_path.suffix}")
                
        except Exception as e:
            results['errors'].append(f"Could not load file: {str(e)}")
            results['valid'] = False
        
        return results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get summary of all validation results.
        
        Returns:
            Dict[str, Any]: Validation summary
        """
        summary = {
            'overall_valid': True,
            'total_validations': len(self.validation_results),
            'valid_validations': 0,
            'invalid_validations': 0,
            'total_errors': 0,
            'total_warnings': 0,
            'results': self.validation_results
        }
        
        for validation_type, result in self.validation_results.items():
            if result['valid']:
                summary['valid_validations'] += 1
            else:
                summary['invalid_validations'] += 1
                summary['overall_valid'] = False
            
            summary['total_errors'] += len(result['errors'])
            summary['total_warnings'] += len(result['warnings'])
        
        return summary
    
    def print_validation_report(self) -> None:
        """Print a comprehensive validation report."""
        summary = self.get_validation_summary()
        
        print("=" * 60)
        print("DATA VALIDATION REPORT")
        print("=" * 60)
        
        print(f"Overall Status: {'✅ VALID' if summary['overall_valid'] else '❌ INVALID'}")
        print(f"Total Validations: {summary['total_validations']}")
        print(f"Valid: {summary['valid_validations']}")
        print(f"Invalid: {summary['invalid_validations']}")
        print(f"Total Errors: {summary['total_errors']}")
        print(f"Total Warnings: {summary['total_warnings']}")
        
        for validation_type, result in self.validation_results.items():
            print(f"\n📊 {validation_type.upper()} VALIDATION:")
            print(f"  Status: {'✅ VALID' if result['valid'] else '❌ INVALID'}")
            
            if result['errors']:
                print("  ❌ Errors:")
                for error in result['errors']:
                    print(f"    - {error}")
            
            if result['warnings']:
                print("  ⚠️  Warnings:")
                for warning in result['warnings']:
                    print(f"    - {warning}")
            
            if result['statistics']:
                print("  📈 Statistics:")
                for key, value in result['statistics'].items():
                    print(f"    {key}: {value}")
        
        print("\n" + "=" * 60)
