"""
Main Pipeline for BISINDO Alphabet Recognition.
This module orchestrates the complete machine learning pipeline.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import logging

try:
    from .config import ensure_directories
    from .data import BISINDODataManager, DatasetLoader, DataValidator
    from .core import ImageAugmenter, LandmarksExtractor, FeatureExtractor, ModelTrainer
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from config import ensure_directories
    from data import BISINDODataManager, DatasetLoader, DataValidator
    from core import ImageAugmenter, LandmarksExtractor, FeatureExtractor, ModelTrainer


class BISINDOPipeline:
    """
    Main pipeline class for BISINDO Alphabet Recognition.
    Orchestrates the complete ML pipeline from raw images to trained models.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the BISINDO pipeline.
        
        Args:
            config (Optional[Dict]): Pipeline configuration
        """
        self.config = config or {}
        
        # Initialize components
        self.data_manager = BISINDODataManager()
        self.data_loader = DatasetLoader()
        self.validator = DataValidator()
        
        # Initialize processing modules
        self.augmenter = ImageAugmenter()
        self.landmarks_extractor = LandmarksExtractor()
        self.feature_extractor = FeatureExtractor()
        self.model_trainer = ModelTrainer()
        
        # Pipeline state
        self.pipeline_state = {
            'raw_data_loaded': False,
            'augmented': False,
            'landmarks_extracted': False,
            'features_extracted': False,
            'models_trained': False
        }
        
        # Ensure directories exist
        ensure_directories()
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def run_complete_pipeline(self, 
                            skip_steps: Optional[list] = None,
                            quick_mode: bool = False) -> Dict[str, Any]:
        """
        Run the complete BISINDO pipeline.
        
        Args:
            skip_steps (Optional[list]): List of steps to skip
            quick_mode (bool): Whether to run in quick mode (faster, less thorough)
            
        Returns:
            Dict[str, Any]: Pipeline results
        """
        skip_steps = skip_steps or []
        results = {}
        
        self.logger.info("Starting BISINDO Pipeline")
        
        try:
            # Step 1: Load raw data
            if 'load_data' not in skip_steps:
                self.logger.info("Step 1: Loading raw dataset")
                results['raw_data'] = self._load_raw_data()
                self.pipeline_state['raw_data_loaded'] = True
            
            # Step 2: Augment images
            if 'augmentation' not in skip_steps:
                self.logger.info("Step 2: Image augmentation")
                results['augmentation'] = self._run_augmentation()
                self.pipeline_state['augmented'] = True
            
            # Step 3: Extract landmarks
            if 'landmarks' not in skip_steps:
                self.logger.info("Step 3: Hand landmarks extraction")
                results['landmarks'] = self._run_landmarks_extraction()
                self.pipeline_state['landmarks_extracted'] = True
            
            # Step 4: Extract features
            if 'features' not in skip_steps:
                self.logger.info("Step 4: Feature extraction")
                results['features'] = self._run_feature_extraction()
                self.pipeline_state['features_extracted'] = True
            
            # Step 5: Train models
            if 'training' not in skip_steps:
                self.logger.info("Step 5: Model training")
                results['training'] = self._run_model_training(quick_mode=quick_mode)
                self.pipeline_state['models_trained'] = True
            
            # Final validation
            self.logger.info("Final validation")
            results['validation'] = self._run_final_validation()
            
            self.logger.info("Pipeline completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
        
        return results
    
    def _load_raw_data(self) -> Dict[str, Any]:
        """Load raw dataset."""
        try:
            images, labels, class_names = self.data_loader.load_dataset()
            
            # Validate data
            validation_result = self.validator.validate_images(images, labels, class_names)
            
            return {
                'images': images,
                'labels': labels,
                'class_names': class_names,
                'validation': validation_result,
                'statistics': self.data_loader.get_statistics()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load raw data: {str(e)}")
            raise
    
    def _run_augmentation(self) -> Dict[str, Any]:
        """Run image augmentation."""
        try:
            # Load raw data if not already loaded
            if not self.pipeline_state['raw_data_loaded']:
                images, labels, class_names = self.data_loader.load_dataset()
            else:
                # Use previously loaded data
                images, labels, class_names = self.data_loader.images, self.data_loader.labels_encoded, self.data_loader.class_names
            
            # Run augmentation
            augmented_images, augmented_labels = self.augmenter.augment_images(
                images, labels, class_names
            )
            
            # Save augmented data
            self.augmenter.save_augmented_dataset(
                augmented_images, augmented_labels, class_names
            )
            
            # Get augmentation summary
            summary = self.augmenter.get_augmentation_summary()
            
            return {
                'augmented_images': augmented_images,
                'augmented_labels': augmented_labels,
                'class_names': class_names,
                'summary': summary
            }
            
        except Exception as e:
            self.logger.error(f"Failed to run augmentation: {str(e)}")
            raise
    
    def _run_landmarks_extraction(self) -> Dict[str, Any]:
        """Run hand landmarks extraction."""
        try:
            # Load augmented data
            images, labels, class_names = self.augmenter.load_augmented_dataset()
            
            # Extract landmarks
            landmarks, valid_labels = self.landmarks_extractor.extract_landmarks_from_dataset(
                images, labels, class_names
            )
            
            # Save landmarks
            self.landmarks_extractor.save_landmarks(
                landmarks, valid_labels, class_names
            )
            
            # Get extraction summary
            summary = self.landmarks_extractor.get_extraction_summary()
            
            return {
                'landmarks': landmarks,
                'labels': valid_labels,
                'class_names': class_names,
                'summary': summary
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract landmarks: {str(e)}")
            raise
    
    def _run_feature_extraction(self) -> Dict[str, Any]:
        """Run feature extraction."""
        try:
            # Load landmarks data
            landmarks, labels, class_names = self.landmarks_extractor.load_landmarks()
            
            # Extract features
            features, valid_labels = self.feature_extractor.extract_features_from_dataset(
                landmarks, labels
            )
            
            # Apply transformations
            transformed_features = self.feature_extractor.fit_transform_features(
                features, valid_labels
            )
            
            # Save features
            self.feature_extractor.save_features(
                transformed_features, valid_labels, class_names
            )
            
            # Get feature summary
            summary = self.feature_extractor.get_feature_summary()
            
            return {
                'features': transformed_features,
                'labels': valid_labels,
                'class_names': class_names,
                'summary': summary
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract features: {str(e)}")
            raise
    
    def _run_model_training(self, quick_mode: bool = False) -> Dict[str, Any]:
        """Run model training."""
        try:
            # Load features data
            features, labels, class_names = self.feature_extractor.load_features()
            
            # Load data into trainer
            self.model_trainer.load_data(use_features=True)
            
            # Train models
            results = self.model_trainer.train_models(
                use_hyperparameter_tuning=not quick_mode,
                quick_training=quick_mode
            )
            
            # Evaluate models
            df_results = self.model_trainer.evaluate_models()
            
            # Save results
            self.model_trainer.save_results()
            
            # Get training summary
            summary = self.model_trainer.get_training_summary()
            
            return {
                'results': results,
                'comparison': df_results,
                'summary': summary,
                'best_model': self.model_trainer.best_model_name,
                'best_score': self.model_trainer.best_score
            }
            
        except Exception as e:
            self.logger.error(f"Failed to train models: {str(e)}")
            raise
    
    def _run_final_validation(self) -> Dict[str, Any]:
        """Run final validation of the complete pipeline."""
        try:
            # Get pipeline status
            pipeline_status = self.data_manager.get_data_pipeline_status()
            
            # Get dataset summary
            dataset_info = self.data_manager.get_dataset_info()
            
            return {
                'pipeline_status': pipeline_status,
                'dataset_info': dataset_info,
                'pipeline_state': self.pipeline_state
            }
            
        except Exception as e:
            self.logger.error(f"Failed to run final validation: {str(e)}")
            raise
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status.
        
        Returns:
            Dict[str, Any]: Pipeline status information
        """
        return {
            'pipeline_state': self.pipeline_state,
            'data_pipeline_status': self.data_manager.get_data_pipeline_status(),
            'dataset_summary': self.data_manager.get_dataset_info()
        }
    
    def print_pipeline_summary(self) -> None:
        """Print a comprehensive pipeline summary."""
        print("=" * 80)
        print("BISINDO PIPELINE SUMMARY")
        print("=" * 80)
        
        # Pipeline state
        print("\n🔄 Pipeline State:")
        for step, status in self.pipeline_state.items():
            emoji = "✅" if status else "❌"
            print(f"  {emoji} {step.replace('_', ' ').title()}: {'Completed' if status else 'Pending'}")
        
        # Data pipeline status
        print("\n📊 Data Pipeline Status:")
        data_status = self.data_manager.get_data_pipeline_status()
        for step, status in data_status.items():
            emoji = "✅" if status == "available" else "❌"
            print(f"  {emoji} {step.replace('_', ' ').title()}: {status}")
        
        # Dataset summary
        print("\n📁 Dataset Summary:")
        self.data_manager.print_dataset_summary()
        
        print("\n" + "=" * 80)


def main():
    """Main function to run the BISINDO pipeline."""
    print("=== BISINDO Alphabet Recognition Pipeline ===")
    
    # Initialize pipeline
    pipeline = BISINDOPipeline()
    
    # Print current status
    pipeline.print_pipeline_summary()
    
    # Ask user if they want to run the pipeline
    response = input("\nDo you want to run the complete pipeline? (y/n): ").lower().strip()
    
    if response == 'y':
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(quick_mode=False)
        
        # Print final summary
        pipeline.print_pipeline_summary()
        
        print("\n🎉 Pipeline completed successfully!")
        print("Ready for real-time BISINDO alphabet recognition!")
        
    else:
        print("Pipeline not executed. You can run it later by calling:")
        print("  pipeline.run_complete_pipeline()")


if __name__ == "__main__":
    main()
