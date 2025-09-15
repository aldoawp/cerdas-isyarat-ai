"""
Test script for the refactored BISINDO pipeline.
This script tests the complete pipeline to ensure everything works after refactoring.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test all module imports."""
    print("🧪 Testing module imports...")
    
    try:
        # Test config imports
        from config import ensure_directories, AUGMENTATION_CONFIG, ML_TRAINING_CONFIG
        print("✅ Config modules imported successfully")
        
        # Test data imports
        from data import BISINDODataManager, DatasetLoader, DataValidator
        print("✅ Data modules imported successfully")
        
        # Test core imports
        from core import ImageAugmenter, LandmarksExtractor, FeatureExtractor, ModelTrainer
        print("✅ Core modules imported successfully")
        
        # Test pipeline import
        from pipeline import BISINDOPipeline
        print("✅ Pipeline module imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {str(e)}")
        return False

def test_configuration():
    """Test configuration system."""
    print("\n🔧 Testing configuration system...")
    
    try:
        from config import ensure_directories, AUGMENTATION_CONFIG, ML_TRAINING_CONFIG
        
        # Test directory creation
        ensure_directories()
        print("✅ Directories created successfully")
        
        # Test config access
        print(f"✅ Augmentation config: {AUGMENTATION_CONFIG['target_images_per_class']} images per class")
        print(f"✅ ML config: {ML_TRAINING_CONFIG['test_size']} test size")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {str(e)}")
        return False

def test_data_manager():
    """Test data manager functionality."""
    print("\n📊 Testing data manager...")
    
    try:
        from data import BISINDODataManager
        
        # Initialize data manager
        dm = BISINDODataManager()
        print("✅ Data manager initialized")
        
        # Test directory structure
        status = dm.get_data_pipeline_status()
        print(f"✅ Data pipeline status: {len(status)} components checked")
        
        # Test dataset info
        info = dm.get_dataset_info()
        print(f"✅ Dataset info: {len(info)} categories")
        
        return True
        
    except Exception as e:
        print(f"❌ Data manager error: {str(e)}")
        return False

def test_core_modules():
    """Test core processing modules."""
    print("\n⚙️ Testing core modules...")
    
    try:
        from core import ImageAugmenter, LandmarksExtractor, FeatureExtractor, ModelTrainer
        
        # Test augmentation module
        augmenter = ImageAugmenter()
        print("✅ ImageAugmenter initialized")
        
        # Test landmarks module
        extractor = LandmarksExtractor()
        print("✅ LandmarksExtractor initialized")
        
        # Test features module
        feature_extractor = FeatureExtractor()
        print("✅ FeatureExtractor initialized")
        
        # Test training module
        trainer = ModelTrainer()
        print("✅ ModelTrainer initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Core modules error: {str(e)}")
        return False

def test_pipeline():
    """Test pipeline initialization."""
    print("\n🚀 Testing pipeline initialization...")
    
    try:
        from pipeline import BISINDOPipeline
        
        # Initialize pipeline
        pipeline = BISINDOPipeline()
        print("✅ BISINDOPipeline initialized")
        
        # Test pipeline status
        status = pipeline.get_pipeline_status()
        print(f"✅ Pipeline status: {len(status)} status items")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline error: {str(e)}")
        return False

def main():
    """Main test function."""
    print("=" * 80)
    print("🧪 BISINDO PIPELINE REFACTORING TEST")
    print("=" * 80)
    print("Testing the refactored codebase for any errors...")
    print("=" * 80)
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration System", test_configuration),
        ("Data Manager", test_data_manager),
        ("Core Modules", test_core_modules),
        ("Pipeline", test_pipeline),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 80)
    print("📋 TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The refactoring was successful!")
        print("✅ The codebase is ready for use.")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the errors above.")
        print("❌ The codebase needs fixes before it can be used.")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
