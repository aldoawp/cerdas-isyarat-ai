"""
Script to update file paths in existing modules to use the new organized structure.
"""

import os
import re
from pathlib import Path


def update_file_paths(file_path: str, old_path: str, new_path: str) -> None:
    """
    Update file paths in a given file.
    
    Args:
        file_path (str): Path to the file to update
        old_path (str): Old path pattern to replace
        new_path (str): New path to replace with
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the old path with new path
    updated_content = content.replace(old_path, new_path)
    
    if content != updated_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        print(f"Updated paths in: {file_path}")
    else:
        print(f"No changes needed in: {file_path}")


def main():
    """Update file paths in all relevant modules."""
    
    # Define path mappings
    path_mappings = [
        # (old_path, new_path)
        ("dataset/processed", "dataset/processed/01_augmented"),  # For augmentation
        ("dataset/processed", "dataset/processed/02_landmarks"),  # For landmarks
        ("dataset/processed", "dataset/processed/03_features"),   # For features
        ("dataset/processed", "dataset/processed/04_models"),     # For models
    ]
    
    # Files to update
    files_to_update = [
        "src/augmentation.py",
        "src/hand_landmarks.py", 
        "src/feature_extraction.py",
        "src/ml_training.py",
        "src/main.py"
    ]
    
    print("Updating file paths in modules...")
    
    for file_path in files_to_update:
        if os.path.exists(file_path):
            print(f"\nUpdating {file_path}:")
            
            # Read the file to see what paths it uses
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check which paths are used in this file
            if "augmented" in content and "save_augmented_dataset" in content:
                update_file_paths(file_path, "dataset/processed", "dataset/processed/01_augmented")
            
            if "landmark" in content and ("save_landmarks" in content or "load_landmarks" in content):
                update_file_paths(file_path, "dataset/processed", "dataset/processed/02_landmarks")
            
            if "feature" in content and ("save_features" in content or "load_features" in content):
                update_file_paths(file_path, "dataset/processed", "dataset/processed/03_features")
            
            if "model" in content and ("save_results" in content or "save_model" in content):
                update_file_paths(file_path, "dataset/processed", "dataset/processed/04_models")
        else:
            print(f"File not found: {file_path}")
    
    print("\nPath updates completed!")


if __name__ == "__main__":
    main()
