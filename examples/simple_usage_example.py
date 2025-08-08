#!/usr/bin/env python3
"""
Simple Usage Example - Gaudi 3 Scale Generation 1

This example demonstrates basic usage of the Gaudi 3 Scale library
for simple training scenarios.
"""

import sys
import logging
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Basic imports that should always work
from gaudi3_scale import __version__
from gaudi3_scale.exceptions import Gaudi3ScaleError, HPUNotAvailableError


def simple_training_example():
    """Demonstrate simple training workflow."""
    print(f"🚀 Gaudi 3 Scale v{__version__} - Simple Example")
    print("=" * 50)
    
    # 1. Basic configuration
    config = {
        "model_name": "simple-model",
        "batch_size": 16,
        "learning_rate": 0.001,
        "max_epochs": 5,
        "precision": "float32"
    }
    
    print("📋 Training Configuration:")
    for key, value in config.items():
        print(f"  • {key}: {value}")
    
    # 2. Check for HPU availability (mock for now)
    print("\n🔍 Checking HPU Availability...")
    try:
        # Try to import HPU-specific modules
        try:
            from gaudi3_scale import GaudiAccelerator
            accelerator = GaudiAccelerator()
            print("✅ Gaudi Accelerator initialized")
            
            # Mock HPU check since we don't have real hardware
            hpu_available = False  # Set to True when real HPUs are available
            if hpu_available:
                device_count = 8  # Mock device count
                print(f"✅ Found {device_count} HPU devices")
            else:
                print("⚠️  No HPU devices found - using simulation mode")
                
        except ImportError as e:
            print(f"⚠️  HPU libraries not available: {e}")
            print("   Running in CPU simulation mode")
            
    except Exception as e:
        print(f"❌ Error checking HPU availability: {e}")
        return False
    
    # 3. Basic model setup (simulation)
    print("\n🧠 Model Setup...")
    try:
        # Mock model creation
        model_config = {
            "model_type": "transformer",
            "hidden_size": 768,
            "num_layers": 12,
            "num_attention_heads": 12
        }
        
        print("✅ Model configuration created:")
        for key, value in model_config.items():
            print(f"  • {key}: {value}")
            
    except Exception as e:
        print(f"❌ Error setting up model: {e}")
        return False
    
    # 4. Basic training loop (simulation)
    print("\n🎯 Training Simulation...")
    try:
        epochs = config["max_epochs"]
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            # Simulate training step
            loss = 1.0 / epoch  # Mock decreasing loss
            accuracy = min(0.95, 0.5 + (epoch * 0.1))  # Mock improving accuracy
            
            print(f"  Epoch {epoch}/{epochs} - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}")
            
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return False
    
    # 5. Save results (simulation)
    print("\n💾 Saving Results...")
    try:
        output_dir = Path("./simple_output")
        output_dir.mkdir(exist_ok=True)
        
        # Mock saving model checkpoint
        checkpoint_path = output_dir / "model_checkpoint.pt"
        print(f"✅ Model saved to: {checkpoint_path}")
        
        # Mock saving training metrics
        metrics_path = output_dir / "training_metrics.json"
        print(f"✅ Metrics saved to: {metrics_path}")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return False
    
    print("\n🎉 Simple training example completed successfully!")
    return True


def simple_inference_example():
    """Demonstrate simple inference workflow."""
    print("\n🔮 Simple Inference Example")
    print("=" * 30)
    
    try:
        # Mock model loading
        print("📂 Loading trained model...")
        model_path = "./simple_output/model_checkpoint.pt"
        print(f"✅ Model loaded from: {model_path}")
        
        # Mock inference
        print("🧮 Running inference...")
        sample_inputs = ["Hello world", "Test input", "Another example"]
        
        for i, input_text in enumerate(sample_inputs, 1):
            # Mock prediction
            confidence = 0.85 + (i * 0.03)  # Mock confidence scores
            prediction = f"Prediction_{i}"
            
            print(f"  Input {i}: '{input_text}' → {prediction} (confidence: {confidence:.2f})")
        
        print("✅ Inference completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return False


def main():
    """Main example function."""
    print("🎯 Gaudi 3 Scale - Simple Usage Examples")
    print("=" * 60)
    
    try:
        # Run training example
        training_success = simple_training_example()
        
        if training_success:
            # Run inference example
            simple_inference_example()
        
        print("\n✨ All examples completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Examples interrupted by user")
    except Gaudi3ScaleError as e:
        print(f"\n❌ Gaudi 3 Scale Error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()