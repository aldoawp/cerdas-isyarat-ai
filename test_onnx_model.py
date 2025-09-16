#!/usr/bin/env python3
"""
Test script to demonstrate how to use the converted ONNX model.
This shows how the model can be used for inference.
"""

import numpy as np
import json
from onnxruntime import InferenceSession

def test_onnx_inference():
    """Test the ONNX model with sample data."""
    print("Testing ONNX Model Inference")
    print("=" * 40)
    
    # Load the ONNX model
    model_path = "dataset/processed/04_models/bisindo_model.onnx"
    session = InferenceSession(model_path)
    
    # Load metadata
    with open("dataset/processed/04_models/model_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    print(f"Model: {metadata['model_info']['name']}")
    print(f"Input features: {metadata['model_info']['input_features']}")
    print(f"Output classes: {metadata['model_info']['output_classes']}")
    print(f"Classes: {metadata['model_info']['classes']}")
    
    # Create sample input data (50 features)
    # In real usage, these would be extracted from hand landmarks
    np.random.seed(42)
    sample_input = np.random.rand(1, 50).astype(np.float32)
    
    print(f"\nSample input shape: {sample_input.shape}")
    print(f"Sample input (first 5 features): {sample_input[0, :5]}")
    
    # Run inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: sample_input})
    
    # Get predictions and probabilities
    predictions = outputs[0]  # Shape: (1,)
    probabilities = outputs[1]  # Shape: (1, 26)
    
    predicted_class_idx = predictions[0]
    predicted_class = metadata['model_info']['classes'][predicted_class_idx]
    confidence = probabilities[0, predicted_class_idx]
    
    print(f"\nPrediction Results:")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    
    # Show top 5 predictions
    top5_indices = np.argsort(probabilities[0])[-5:][::-1]
    print(f"\nTop 5 predictions:")
    for i, idx in enumerate(top5_indices):
        class_name = metadata['model_info']['classes'][idx]
        prob = probabilities[0, idx]
        print(f"  {i+1}. {class_name}: {prob:.4f}")
    
    print(f"\n✅ ONNX model inference successful!")
    print(f"Model is ready for web integration!")

if __name__ == "__main__":
    test_onnx_inference()
