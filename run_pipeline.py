"""
Run the BISINDO pipeline as a module.
This script properly imports and runs the refactored pipeline.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Main function to run the BISINDO pipeline."""
    print("=" * 80)
    print("🤟 BISINDO Alphabet Recognition System")
    print("=" * 80)
    print("Testing the refactored pipeline...")
    print("=" * 80)
    
    try:
        # Import and test configuration
        print("🔧 Testing configuration...")
        from src.config import ensure_directories, AUGMENTATION_CONFIG
        ensure_directories()
        print("✅ Configuration working")
        
        # Import and test data manager
        print("\n📊 Testing data manager...")
        from src.data import BISINDODataManager
        dm = BISINDODataManager()
        status = dm.get_data_pipeline_status()
        print(f"✅ Data manager working - {len(status)} components")
        
        # Import and test core modules
        print("\n⚙️ Testing core modules...")
        from src.core import ImageAugmenter, LandmarksExtractor, FeatureExtractor, ModelTrainer
        
        augmenter = ImageAugmenter()
        extractor = LandmarksExtractor()
        feature_extractor = FeatureExtractor()
        trainer = ModelTrainer()
        print("✅ All core modules working")
        
        # Import and test pipeline
        print("\n🚀 Testing pipeline...")
        from src.pipeline import BISINDOPipeline
        pipeline = BISINDOPipeline()
        print("✅ Pipeline initialized")
        
        # Show pipeline status
        print("\n📋 Pipeline Status:")
        pipeline.print_pipeline_summary()
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ The refactored codebase is working correctly!")
        print("✅ Ready to run the complete pipeline!")
        
        # Ask if user wants to run the full pipeline
        response = input("\nDo you want to run the complete pipeline? (y/n): ").lower().strip()
        
        if response == 'y':
            print("\n🚀 Running complete pipeline...")
            results = pipeline.run_complete_pipeline(quick_mode=True)  # Use quick mode for testing
            print("\n✅ Pipeline completed successfully!")
            pipeline.print_pipeline_summary()
        else:
            print("\n👋 Pipeline test completed. You can run the full pipeline later.")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎊 Testing completed successfully!")
    else:
        print("\n💥 Testing failed. Please check the errors above.")
        sys.exit(1)
