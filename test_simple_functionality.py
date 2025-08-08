#!/usr/bin/env python3
"""Test script to verify simple functionality works."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_import():
    """Test that basic package imports work."""
    print("🔍 Testing basic package import...")
    try:
        import gaudi3_scale
        print(f"✅ Package imported successfully - version {gaudi3_scale.__version__}")
        return True
    except Exception as e:
        print(f"❌ Package import failed: {e}")
        return False

def test_simple_trainer():
    """Test the simple trainer functionality."""
    print("\n🎯 Testing simple trainer...")
    try:
        from gaudi3_scale.simple_trainer import SimpleTrainer, SimpleTrainingConfig, quick_train
        
        # Test quick training function
        print("  Testing quick_train function...")
        results = quick_train(
            model_name="test-model",
            epochs=2,
            batch_size=4,
            verbose=False
        )
        
        assert results["success"] == True
        assert results["total_epochs"] == 2
        assert len(results["metrics_history"]) == 2
        print("  ✅ Quick train function works!")
        
        # Test detailed trainer
        print("  Testing detailed trainer...")
        config = SimpleTrainingConfig(
            model_name="detailed-test",
            max_epochs=3,
            batch_size=8
        )
        
        trainer = SimpleTrainer(config)
        detailed_results = trainer.train(verbose=False)
        
        assert detailed_results["success"] == True
        assert detailed_results["total_epochs"] == 3
        print("  ✅ Detailed trainer works!")
        
        # Test training summary
        summary = trainer.get_training_summary()
        assert summary["current_epoch"] == 3
        assert summary["model_name"] == "detailed-test"
        print("  ✅ Training summary works!")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Simple trainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_examples():
    """Test that examples run without errors."""
    print("\n📝 Testing examples...")
    try:
        # Test simple example
        print("  Testing simple usage example...")
        
        # Import and run the example
        sys.path.insert(0, str(Path(__file__).parent / "examples"))
        
        # We'll just import it to check for syntax errors
        import simple_usage_example
        print("  ✅ Simple usage example imports successfully!")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Examples test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Gaudi 3 Scale Simple Functionality")
    print("=" * 50)
    
    tests = [
        ("Basic Import", test_basic_import),
        ("Simple Trainer", test_simple_trainer), 
        ("Examples", test_examples)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name} test...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test FAILED with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Generation 1 functionality is working.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)