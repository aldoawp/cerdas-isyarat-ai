"""
Main entry point for BISINDO Alphabet Recognition.
This module provides a clean interface to run the complete pipeline.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from .pipeline import BISINDOPipeline
from .config import ensure_directories


def main():
    """Main function to run the BISINDO pipeline."""
    print("=" * 80)
    print("🤟 BISINDO Alphabet Recognition System")
    print("=" * 80)
    print("An AI-powered system for recognizing Indonesian Sign Language (BISINDO) alphabet")
    print("=" * 80)
    
    try:
        # Ensure all directories exist
        ensure_directories()
        
        # Initialize and run pipeline
        pipeline = BISINDOPipeline()
        
        # Show current status
        pipeline.print_pipeline_summary()
        
        # Run complete pipeline
        print("\n🚀 Running complete pipeline...")
        results = pipeline.run_complete_pipeline(quick_mode=False)
        print("\n✅ Complete pipeline finished!")
        pipeline.print_pipeline_summary()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user.")
        print("👋 Goodbye!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please check the error and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
