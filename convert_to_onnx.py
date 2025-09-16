#!/usr/bin/env python3
"""
Convert trained BISINDO model to ONNX format for web browser compatibility.

This script loads the trained Random Forest model and converts it to ONNX format
for use in web browsers with ONNX.js.
"""

import pickle
import joblib
import numpy as np
import json
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
from onnxruntime import InferenceSession


def load_model_and_metadata():
    """Load the trained model and related metadata."""
    print("Loading model and metadata...")
    
    # Load the trained model (saved with joblib)
    model_path = "dataset/processed/04_models/best_model.pkl"
    model = joblib.load(model_path)
    
    # Load feature names and class names
    # Note: The model was trained with 50 features, but feature_names.json has 70
    # We need to use the actual number of features the model expects
    with open("dataset/processed/03_features/feature_names.json", 'r') as f:
        all_feature_names = json.load(f)
    
    # The model expects 50 features, so we'll use the first 50 feature names
    # or create generic names if needed
    if len(all_feature_names) >= model.n_features_in_:
        feature_names = all_feature_names[:model.n_features_in_]
    else:
        # Create generic feature names if we don't have enough
        feature_names = [f"feature_{i}" for i in range(model.n_features_in_)]
    
    with open("dataset/processed/03_features/feature_class_names.json", 'r') as f:
        class_names = json.load(f)
    
    # Load feature scaler (if used)
    scaler_path = "dataset/processed/03_features/feature_scaler.pkl"
    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    
    print(f"Model type: {type(model).__name__}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    
    return model, feature_names, class_names, scaler


def convert_to_onnx(model, feature_names, class_names, scaler=None):
    """Convert the sklearn model to ONNX format."""
    print("\nConverting model to ONNX format...")
    
    # Define input type for ONNX conversion
    # Input shape: (batch_size, n_features)
    initial_type = [('float_input', FloatTensorType([None, len(feature_names)]))]
    
    # Convert the model to ONNX
    try:
        onnx_model = convert_sklearn(
            model,
            initial_types=initial_type,
            target_opset=11,  # Use opset 11 for better browser compatibility
            options={id(model): {'zipmap': False}}  # Disable zipmap for better performance
        )
        
        # Save the ONNX model
        onnx_path = "dataset/processed/04_models/bisindo_model.onnx"
        with open(onnx_path, 'wb') as f:
            f.write(onnx_model.SerializeToString())
        
        print(f"ONNX model saved to: {onnx_path}")
        print(f"Model size: {os.path.getsize(onnx_path) / 1024:.2f} KB")
        
        return onnx_model, onnx_path
        
    except Exception as e:
        print(f"Error converting to ONNX: {e}")
        return None, None


def validate_onnx_model(onnx_path, model, feature_names, class_names, scaler=None):
    """Validate the ONNX model by comparing predictions with the original model."""
    print("\nValidating ONNX model...")
    
    try:
        # Load the ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Create ONNX Runtime session
        session = InferenceSession(onnx_path)
        
        # Generate some test data (random features for validation)
        np.random.seed(42)
        test_data = np.random.rand(5, len(feature_names)).astype(np.float32)
        
        # Get predictions from original model
        original_predictions = model.predict(test_data)
        original_probabilities = model.predict_proba(test_data)
        
        # Get predictions from ONNX model
        onnx_inputs = {session.get_inputs()[0].name: test_data}
        onnx_outputs = session.run(None, onnx_inputs)
        
        # ONNX model outputs - check the output format
        print(f"ONNX outputs: {len(onnx_outputs)} outputs")
        for i, output in enumerate(onnx_outputs):
            print(f"Output {i}: shape {output.shape}, type {type(output)}")
        
        # Handle different ONNX output formats
        if len(onnx_outputs) == 1:
            # Single output - could be probabilities or predictions
            onnx_output = onnx_outputs[0]
            if onnx_output.ndim == 2 and onnx_output.shape[1] > 1:
                # Probabilities
                onnx_probabilities = onnx_output
                onnx_predictions = np.argmax(onnx_probabilities, axis=1)
            else:
                # Predictions
                onnx_predictions = onnx_output.flatten()
                # Create dummy probabilities for comparison
                onnx_probabilities = np.zeros((len(onnx_predictions), len(class_names)))
                for i, pred in enumerate(onnx_predictions):
                    onnx_probabilities[i, pred] = 1.0
        else:
            # Multiple outputs - first is predictions, second is probabilities
            onnx_predictions = onnx_outputs[0].flatten().astype(int)
            onnx_probabilities = onnx_outputs[1]
        
        # Map predictions to class names
        original_class_names = [class_names[i] for i in original_predictions]
        onnx_class_names = [class_names[i] for i in onnx_predictions]
        
        print("Validation Results:")
        print("=" * 50)
        for i in range(len(test_data)):
            print(f"Sample {i+1}:")
            print(f"  Original: {original_class_names[i]} (prob: {original_probabilities[i].max():.4f})")
            print(f"  ONNX:     {onnx_class_names[i]} (prob: {onnx_probabilities[i].max():.4f})")
            print(f"  Match:    {original_predictions[i] == onnx_predictions[i]}")
            print()
        
        # Check if all predictions match
        matches = np.array_equal(original_predictions, onnx_predictions)
        print(f"All predictions match: {matches}")
        
        return matches
        
    except Exception as e:
        print(f"Error validating ONNX model: {e}")
        return False


def create_web_integration_files(onnx_path, feature_names, class_names, scaler=None):
    """Create additional files needed for web integration."""
    print("\nCreating web integration files...")
    
    # Create metadata file for web usage
    metadata = {
        "model_info": {
            "name": "BISINDO Alphabet Recognition",
            "version": "1.0",
            "description": "Random Forest model for BISINDO alphabet recognition",
            "input_features": len(feature_names),
            "output_classes": len(class_names),
            "classes": class_names,
            "feature_names": feature_names
        },
        "preprocessing": {
            "scaler_used": scaler is not None,
            "input_normalization": "Required if scaler is used"
        },
        "usage": {
            "input_shape": [1, len(feature_names)],
            "input_type": "float32",
            "output_type": "float32"
        }
    }
    
    # Save metadata
    metadata_path = "dataset/processed/04_models/model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {metadata_path}")
    
    # Create a simple JavaScript example for web usage
    js_example = f"""
// Example usage of BISINDO ONNX model in web browser
// This is a template - you'll need to implement the actual feature extraction

const modelPath = './bisindo_model.onnx';
const featureNames = {json.dumps(feature_names, indent=2)};
const classNames = {json.dumps(class_names, indent=2)};

// Load the ONNX model
async function loadModel() {{
    const session = new ort.InferenceSession();
    await session.loadModel(modelPath);
    return session;
}}

// Predict BISINDO alphabet from hand landmarks
async function predictBisindo(handLandmarks, session) {{
    // Extract features from hand landmarks (implement this based on your feature extraction logic)
    const features = extractFeatures(handLandmarks);
    
    // Normalize features if scaler was used during training
    // const normalizedFeatures = normalizeFeatures(features);
    
    // Prepare input tensor
    const inputTensor = new ort.Tensor('float32', features, [1, {len(feature_names)}]);
    
    // Run inference
    const results = await session.run({{input: inputTensor}});
    const probabilities = results.output.data;
    
    // Get prediction
    const predictionIndex = probabilities.indexOf(Math.max(...probabilities));
    const predictedClass = classNames[predictionIndex];
    const confidence = probabilities[predictionIndex];
    
    return {{
        prediction: predictedClass,
        confidence: confidence,
        probabilities: probabilities
    }};
}}

// Example feature extraction function (implement based on your training pipeline)
function extractFeatures(handLandmarks) {{
    // This is a placeholder - implement the actual feature extraction
    // based on your training pipeline (distances, angles, areas, etc.)
    const features = new Array({len(feature_names)}).fill(0);
    
    // TODO: Implement feature extraction logic here
    // - Calculate distances between landmarks
    // - Calculate angles between landmarks  
    // - Calculate areas and perimeters
    // - Extract finger information
    // - Calculate orientations
    
    return features;
}}
"""
    
    js_path = "dataset/processed/04_models/web_integration_example.js"
    with open(js_path, 'w') as f:
        f.write(js_example)
    
    print(f"JavaScript example saved to: {js_path}")


def main():
    """Main function to convert model to ONNX."""
    print("BISINDO Model to ONNX Converter")
    print("=" * 40)
    
    try:
        # Load model and metadata
        model, feature_names, class_names, scaler = load_model_and_metadata()
        
        # Convert to ONNX
        onnx_model, onnx_path = convert_to_onnx(model, feature_names, class_names, scaler)
        
        if onnx_model is None:
            print("Failed to convert model to ONNX")
            return
        
        # Validate the conversion
        is_valid = validate_onnx_model(onnx_path, model, feature_names, class_names, scaler)
        
        if is_valid:
            print("\n✅ ONNX conversion successful and validated!")
            
            # Create web integration files
            create_web_integration_files(onnx_path, feature_names, class_names, scaler)
            
            print("\n🎉 Model conversion completed successfully!")
            print(f"ONNX model ready for web usage: {onnx_path}")
            print("\nNext steps:")
            print("1. Copy the ONNX model to your web server")
            print("2. Include ONNX.js in your web application")
            print("3. Implement hand landmark detection (MediaPipe)")
            print("4. Extract features from landmarks using the same pipeline")
            print("5. Use the ONNX model for real-time predictions")
            
        else:
            print("\n❌ ONNX conversion validation failed!")
            
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
