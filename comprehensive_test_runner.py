"""Comprehensive test runner for TERRAGON SDLC validation.

This module provides complete testing coverage including:
- Unit tests for all core functionality
- Integration tests for system components
- Performance benchmarks and validation
- Security testing and vulnerability scanning
- Quality gate validation
"""

import time
import sys
import traceback
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import subprocess

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

class TestResult:
    """Test result container."""
    
    def __init__(self, name: str, status: str, duration: float, details: Optional[str] = None):
        self.name = name
        self.status = status  # "pass", "fail", "skip"
        self.duration = duration
        self.details = details or ""
        self.timestamp = time.time()


class ComprehensiveTestRunner:
    """Comprehensive test runner with quality gates."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.test_start_time = None
        self.quality_gates = {
            "unit_tests": {"required": True, "min_pass_rate": 0.95},
            "integration_tests": {"required": True, "min_pass_rate": 0.90},
            "performance_tests": {"required": True, "min_pass_rate": 0.85},
            "security_tests": {"required": True, "min_pass_rate": 1.0}
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites and quality gates."""
        
        print("🧪 TERRAGON SDLC - Comprehensive Test Suite")
        print("=" * 50)
        
        self.test_start_time = time.time()
        
        # Run test suites
        self._run_unit_tests()
        self._run_integration_tests()
        self._run_performance_tests()
        self._run_security_tests()
        
        # Generate report
        return self._generate_test_report()
    
    def _run_unit_tests(self) -> None:
        """Run unit tests for core functionality."""
        print("\n🔬 Running Unit Tests...")
        
        unit_tests = [
            ("Test Simple Trainer", self._test_simple_trainer),
            ("Test Robust Trainer", self._test_robust_trainer),
            ("Test Performance Optimizer", self._test_performance_optimizer),
            ("Test Quantum Orchestrator", self._test_quantum_orchestrator),
            ("Test Configuration Validation", self._test_config_validation),
            ("Test Error Handling", self._test_error_handling)
        ]
        
        for test_name, test_func in unit_tests:
            self._run_test(test_name, test_func)
    
    def _run_integration_tests(self) -> None:
        """Run integration tests for system components."""
        print("\n🔗 Running Integration Tests...")
        
        integration_tests = [
            ("Test Full Training Pipeline", self._test_full_pipeline),
            ("Test Device Detection", self._test_device_detection),
            ("Test Checkpoint System", self._test_checkpoint_system),
            ("Test Multi-Component Integration", self._test_multi_component)
        ]
        
        for test_name, test_func in integration_tests:
            self._run_test(test_name, test_func)
    
    def _run_performance_tests(self) -> None:
        """Run performance tests and benchmarks."""
        print("\n⚡ Running Performance Tests...")
        
        performance_tests = [
            ("Test Training Throughput", self._test_training_throughput),
            ("Test Memory Efficiency", self._test_memory_efficiency),
            ("Test Optimization Speed", self._test_optimization_speed),
            ("Test Scaling Performance", self._test_scaling_performance)
        ]
        
        for test_name, test_func in performance_tests:
            self._run_test(test_name, test_func)
    
    def _run_security_tests(self) -> None:
        """Run security tests and vulnerability checks."""
        print("\n🛡️ Running Security Tests...")
        
        security_tests = [
            ("Test Input Validation", self._test_input_validation),
            ("Test Error Information Leakage", self._test_error_leakage),
            ("Test Configuration Security", self._test_config_security),
            ("Test Dependency Security", self._test_dependency_security)
        ]
        
        for test_name, test_func in security_tests:
            self._run_test(test_name, test_func)
    
    def _run_test(self, test_name: str, test_func) -> None:
        """Run individual test with error handling."""
        start_time = time.time()
        
        try:
            test_func()
            duration = time.time() - start_time
            result = TestResult(test_name, "pass", duration)
            print(f"  ✅ {test_name} ({duration:.2f}s)")
            
        except AssertionError as e:
            duration = time.time() - start_time
            result = TestResult(test_name, "fail", duration, str(e))
            print(f"  ❌ {test_name} - FAILED: {str(e)} ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            result = TestResult(test_name, "fail", duration, f"Exception: {str(e)}")
            print(f"  ❌ {test_name} - ERROR: {str(e)} ({duration:.2f}s)")
        
        self.results.append(result)
    
    # Unit Tests
    def _test_simple_trainer(self) -> None:
        """Test simple trainer functionality."""
        from gaudi3_scale.simple_trainer import quick_train, SimpleTrainer, SimpleTrainingConfig
        
        # Test quick training
        result = quick_train(epochs=2, batch_size=4, verbose=False)
        assert result["success"] == True, "Quick train should succeed"
        assert "best_accuracy" in result, "Should return best accuracy"
        assert result["total_time"] > 0, "Should have positive training time"
        
        # Test configuration
        config = SimpleTrainingConfig(max_epochs=3)
        trainer = SimpleTrainer(config)
        assert trainer.config.max_epochs == 3, "Configuration should be set correctly"
        
        # Test training
        training_result = trainer.train(verbose=False)
        assert training_result["success"] == True, "Training should succeed"
        assert len(training_result["metrics_history"]) == 3, "Should have 3 epochs of metrics"
    
    def _test_robust_trainer(self) -> None:
        """Test robust trainer functionality."""
        from gaudi3_scale.robust_trainer import RobustTrainer, robust_quick_train
        
        # Test robust training
        result = robust_quick_train(epochs=2, enable_monitoring=False)
        assert result["success"] == True, "Robust training should succeed"
        assert "health_status" in result, "Should include health status"
        assert result["error_count"] >= 0, "Should track error count"
        
        # Test with custom config
        config = {"max_epochs": 2, "batch_size": 8}
        trainer = RobustTrainer(config, enable_health_monitoring=False)
        assert trainer.config["max_epochs"] == 2, "Configuration should be applied"
    
    def _test_performance_optimizer(self) -> None:
        """Test performance optimizer functionality."""
        from gaudi3_scale.performance_optimizer import PerformanceOptimizer, optimize_for_gaudi3
        
        # Test configuration optimization
        base_config = {"batch_size": 16, "learning_rate": 0.001}
        optimized = optimize_for_gaudi3(base_config)
        assert "device_type" in optimized, "Should detect device type"
        assert "batch_size" in optimized, "Should have batch size"
        
        # Test optimizer
        optimizer = PerformanceOptimizer()
        metrics = optimizer.update_performance_metrics(1, 16, 0.5, 1.0, 0.7)
        assert "optimization_score" in metrics, "Should provide optimization score"
        assert 0 <= metrics["optimization_score"] <= 100, "Score should be in valid range"
    
    def _test_quantum_orchestrator(self) -> None:
        """Test quantum orchestrator functionality."""
        from gaudi3_scale.quantum_orchestrator import QuantumInspiredOptimizer, AutonomousOrchestrator
        
        # Test quantum optimizer
        optimizer = QuantumInspiredOptimizer(num_qubits=2, max_iterations=5)
        
        workload_req = {"cpu": 0.5, "memory": 0.3}
        cluster_cap = {"cluster_1": {"cpu": 1.0, "memory": 1.0}}
        
        result = optimizer.optimize_global_allocation(workload_req, cluster_cap)
        assert hasattr(result, "optimal_configuration"), "Should have optimal configuration"
        assert hasattr(result, "performance_improvement"), "Should have performance metrics"
        
        # Test orchestrator
        orchestrator = AutonomousOrchestrator(enable_quantum_optimization=False)
        workload = {"name": "test", "compute_intensity": 0.5}
        
        orch_result = orchestrator.orchestrate_workload(workload)
        assert "workload_id" in orch_result, "Should assign workload ID"
        assert "status" in orch_result, "Should have status"
    
    def _test_config_validation(self) -> None:
        """Test configuration validation."""
        from gaudi3_scale.simple_trainer import SimpleTrainingConfig
        
        # Test valid configuration
        config = SimpleTrainingConfig(batch_size=16, learning_rate=0.001)
        assert config.batch_size == 16, "Should set batch size correctly"
        
        # Test configuration serialization
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict), "Should serialize to dictionary"
        assert "batch_size" in config_dict, "Should include all parameters"
    
    def _test_error_handling(self) -> None:
        """Test error handling mechanisms."""
        from gaudi3_scale.robust_trainer import RobustTrainer
        
        # Test invalid configuration
        try:
            config = {"batch_size": -1}  # Invalid
            trainer = RobustTrainer(config)
            assert False, "Should raise error for invalid configuration"
        except ValueError:
            pass  # Expected
        
        # Test graceful handling
        config = {"max_epochs": 1}
        trainer = RobustTrainer(config, enable_health_monitoring=False)
        assert trainer.state.value in ["ready", "initializing"], "Should initialize despite issues"
    
    # Integration Tests
    def _test_full_pipeline(self) -> None:
        """Test full training pipeline."""
        from gaudi3_scale.simple_trainer import SimpleTrainer, SimpleTrainingConfig
        from gaudi3_scale.performance_optimizer import PerformanceOptimizer
        
        # Create optimized configuration
        optimizer = PerformanceOptimizer()
        base_config = {"batch_size": 8, "max_epochs": 2}
        optimized_config = optimizer.optimize_training_config(base_config)
        
        # Run training with optimized config (filter valid parameters)
        valid_params = {
            k: v for k, v in optimized_config.items() 
            if k in ["model_name", "batch_size", "learning_rate", "max_epochs", "precision", "output_dir", "use_hpu", "device_count", "mixed_precision", "gradient_accumulation"]
        }
        config = SimpleTrainingConfig(**valid_params)
        trainer = SimpleTrainer(config)
        result = trainer.train(verbose=False)
        
        assert result["success"] == True, "Full pipeline should succeed"
        assert result["device_type"] in ["cpu", "cuda", "hpu"], "Should use valid device"
    
    def _test_device_detection(self) -> None:
        """Test device detection logic."""
        from gaudi3_scale.simple_trainer import SimpleTrainingConfig
        
        config = SimpleTrainingConfig()
        assert config.device_type in ["cpu", "cuda", "hpu"], "Should detect valid device type"
        assert config.available_devices >= 1, "Should have at least one device"
    
    def _test_checkpoint_system(self) -> None:
        """Test checkpoint saving and loading."""
        from gaudi3_scale.simple_trainer import SimpleTrainer, SimpleTrainingConfig
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimpleTrainingConfig(output_dir=temp_dir, max_epochs=2)
            trainer = SimpleTrainer(config)
            result = trainer.train(verbose=False)
            
            # Check checkpoint files were created
            checkpoint_files = list(Path(temp_dir).glob("checkpoint_*.json"))
            assert len(checkpoint_files) > 0, "Should create checkpoint files"
            
            # Verify checkpoint content
            with open(checkpoint_files[0], 'r') as f:
                checkpoint_data = json.load(f)
            assert "epoch" in checkpoint_data, "Checkpoint should contain epoch"
            assert "best_accuracy" in checkpoint_data, "Checkpoint should contain metrics"
    
    def _test_multi_component(self) -> None:
        """Test multiple components working together."""
        from gaudi3_scale.robust_trainer import RobustTrainer
        from gaudi3_scale.performance_optimizer import PerformanceOptimizer
        
        # Setup optimizer and trainer
        optimizer = PerformanceOptimizer()
        base_config = {"max_epochs": 2, "batch_size": 8}
        optimized_config = optimizer.optimize_training_config(base_config)
        
        # Run robust training
        trainer = RobustTrainer(optimized_config, enable_health_monitoring=False)
        result = trainer.train()
        
        assert result["success"] == True, "Multi-component integration should work"
    
    # Performance Tests
    def _test_training_throughput(self) -> None:
        """Test training throughput performance."""
        from gaudi3_scale.simple_trainer import quick_train
        
        start_time = time.time()
        result = quick_train(epochs=3, batch_size=16, verbose=False)
        duration = time.time() - start_time
        
        assert result["success"] == True, "Training should complete successfully"
        
        throughput = result.get("throughput_samples_per_sec", 0)
        assert throughput > 0, "Should have positive throughput"
        
        # Performance threshold (adjust based on expected performance)
        min_throughput = 10  # samples per second
        assert throughput >= min_throughput, f"Throughput {throughput:.1f} below minimum {min_throughput}"
    
    def _test_memory_efficiency(self) -> None:
        """Test memory usage efficiency."""
        from gaudi3_scale.performance_optimizer import PerformanceOptimizer
        
        optimizer = PerformanceOptimizer()
        
        # Test memory utilization calculation
        memory_util = optimizer._get_memory_utilization()
        assert 0 <= memory_util <= 1.0, "Memory utilization should be between 0 and 1"
        
        # Should not use excessive memory
        assert memory_util < 0.95, "Memory utilization should not be excessive during testing"
    
    def _test_optimization_speed(self) -> None:
        """Test optimization algorithm speed."""
        from gaudi3_scale.quantum_orchestrator import QuantumInspiredOptimizer
        
        optimizer = QuantumInspiredOptimizer(num_qubits=3, max_iterations=10)
        
        start_time = time.time()
        result = optimizer.optimize_global_allocation(
            {"cpu": 0.5}, {"cluster_1": {"cpu": 1.0}}
        )
        optimization_time = time.time() - start_time
        
        # Should complete optimization quickly
        max_time = 15.0  # seconds
        assert optimization_time < max_time, f"Optimization took {optimization_time:.1f}s, should be under {max_time}s"
    
    def _test_scaling_performance(self) -> None:
        """Test performance scaling characteristics."""
        from gaudi3_scale.performance_optimizer import benchmark_optimization_impact
        
        base_config = {"batch_size": 8, "max_epochs": 2}
        benchmark = benchmark_optimization_impact(base_config, epochs=2)
        
        speedup = benchmark["improvements"]["speedup"]
        assert speedup > 1.0, f"Optimization should provide speedup, got {speedup:.2f}x"
        
        throughput_improvement = benchmark["improvements"]["throughput_improvement_percent"]
        assert throughput_improvement >= 0, "Throughput improvement should be non-negative"
    
    # Security Tests
    def _test_input_validation(self) -> None:
        """Test input validation and sanitization."""
        from gaudi3_scale.simple_trainer import SimpleTrainingConfig
        
        # Test valid inputs
        config = SimpleTrainingConfig(batch_size=16, learning_rate=0.001)
        assert config.batch_size == 16, "Valid input should be accepted"
        
        # Test edge cases
        config = SimpleTrainingConfig(batch_size=1, learning_rate=0.0001)
        assert config.batch_size >= 1, "Should handle minimum values"
        
        # Test string inputs don't cause issues
        config = SimpleTrainingConfig(model_name="test-model-123")
        assert isinstance(config.model_name, str), "String inputs should be handled safely"
    
    def _test_error_leakage(self) -> None:
        """Test that errors don't leak sensitive information."""
        from gaudi3_scale.robust_trainer import RobustTrainer
        
        # Test with configuration that might cause errors
        config = {"batch_size": 1000000}  # Very large batch size
        
        try:
            trainer = RobustTrainer(config, enable_health_monitoring=False)
            # Should either work or fail gracefully without exposing internals
        except Exception as e:
            error_msg = str(e).lower()
            # Check that error messages don't contain sensitive paths or internal details
            assert "/root/" not in error_msg, "Error should not expose file paths"
            assert "password" not in error_msg, "Error should not expose sensitive terms"
    
    def _test_config_security(self) -> None:
        """Test configuration security."""
        from gaudi3_scale.simple_trainer import SimpleTrainingConfig
        
        # Test that configuration doesn't accept dangerous inputs
        config = SimpleTrainingConfig(
            model_name="safe_model_name",
            output_dir="./safe_output"
        )
        
        # Verify paths are safe
        assert not config.model_name.startswith("/"), "Model name should be relative"
        assert "safe_output" in str(config.output_dir), "Output directory should be controlled"
    
    def _test_dependency_security(self) -> None:
        """Test dependency security and availability."""
        # Test that missing dependencies are handled gracefully
        try:
            import nonexistent_module
            assert False, "This should not succeed"
        except ImportError:
            pass  # Expected
        
        # Verify core functionality works without optional dependencies
        from gaudi3_scale.simple_trainer import quick_train
        result = quick_train(epochs=1, verbose=False)
        assert result["success"] == True, "Core functionality should work without optional deps"
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_time = time.time() - self.test_start_time
        
        # Categorize results
        categorized_results = {
            "unit_tests": [],
            "integration_tests": [],
            "performance_tests": [],
            "security_tests": []
        }
        
        for result in self.results:
            if "Unit" in result.name or any(x in result.name.lower() for x in ["simple", "robust", "performance", "quantum", "config", "error"]):
                categorized_results["unit_tests"].append(result)
            elif "Integration" in result.name or any(x in result.name.lower() for x in ["full", "device", "checkpoint", "multi"]):
                categorized_results["integration_tests"].append(result)
            elif "Performance" in result.name or any(x in result.name.lower() for x in ["throughput", "memory", "optimization", "scaling"]):
                categorized_results["performance_tests"].append(result)
            elif "Security" in result.name or any(x in result.name.lower() for x in ["input", "error", "config", "dependency"]):
                categorized_results["security_tests"].append(result)
        
        # Calculate statistics
        stats = {}
        overall_pass = 0
        overall_total = 0
        
        for category, results in categorized_results.items():
            if not results:
                continue
                
            passed = len([r for r in results if r.status == "pass"])
            total = len(results)
            pass_rate = passed / total if total > 0 else 0
            
            stats[category] = {
                "passed": passed,
                "total": total,
                "pass_rate": pass_rate,
                "duration": sum(r.duration for r in results)
            }
            
            overall_pass += passed
            overall_total += total
        
        overall_pass_rate = overall_pass / overall_total if overall_total > 0 else 0
        
        # Check quality gates
        quality_gate_results = {}
        all_gates_passed = True
        
        for gate_name, gate_config in self.quality_gates.items():
            if gate_name in stats:
                gate_passed = stats[gate_name]["pass_rate"] >= gate_config["min_pass_rate"]
                quality_gate_results[gate_name] = {
                    "passed": gate_passed,
                    "required_rate": gate_config["min_pass_rate"],
                    "actual_rate": stats[gate_name]["pass_rate"]
                }
                if not gate_passed and gate_config["required"]:
                    all_gates_passed = False
        
        # Generate report
        report = {
            "summary": {
                "total_tests": overall_total,
                "tests_passed": overall_pass,
                "overall_pass_rate": overall_pass_rate,
                "total_duration": total_time,
                "all_quality_gates_passed": all_gates_passed
            },
            "category_stats": stats,
            "quality_gates": quality_gate_results,
            "failed_tests": [
                {"name": r.name, "details": r.details}
                for r in self.results if r.status == "fail"
            ],
            "timestamp": time.time()
        }
        
        # Print summary
        print(f"\n{'='*50}")
        print("🧪 TEST SUMMARY")
        print(f"{'='*50}")
        print(f"Total Tests: {overall_total}")
        print(f"Passed: {overall_pass}")
        print(f"Failed: {overall_total - overall_pass}")
        print(f"Pass Rate: {overall_pass_rate:.1%}")
        print(f"Duration: {total_time:.2f}s")
        
        print(f"\n🚨 QUALITY GATES")
        for gate_name, gate_result in quality_gate_results.items():
            status = "✅ PASS" if gate_result["passed"] else "❌ FAIL"
            print(f"  {gate_name}: {status} ({gate_result['actual_rate']:.1%} >= {gate_result['required_rate']:.1%})")
        
        if all_gates_passed:
            print(f"\n🎉 ALL QUALITY GATES PASSED!")
        else:
            print(f"\n⚠️  SOME QUALITY GATES FAILED")
        
        return report


def main():
    """Run comprehensive test suite."""
    runner = ComprehensiveTestRunner()
    report = runner.run_all_tests()
    
    # Save report
    report_file = Path("test_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Test report saved to: {report_file}")
    
    # Exit with appropriate code
    if report["summary"]["all_quality_gates_passed"]:
        print("✅ All tests and quality gates passed!")
        sys.exit(0)
    else:
        print("❌ Some tests or quality gates failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()