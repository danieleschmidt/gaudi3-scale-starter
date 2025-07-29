"""Performance benchmark tests for Gaudi 3 Scale."""

import time
import pytest
from unittest.mock import Mock, patch
from gaudi3_scale.trainer import GaudiTrainer
from gaudi3_scale.accelerator import GaudiAccelerator


@pytest.fixture
def benchmark_model():
    """Create a mock model for benchmarking."""
    model = Mock()
    model.forward.return_value = Mock(loss=0.5)
    return model


@pytest.fixture
def benchmark_data():
    """Create mock training data."""
    batch = {
        'input_ids': Mock(),
        'attention_mask': Mock(),
        'labels': Mock()
    }
    return [batch] * 100  # 100 batches


class TestPerformanceBenchmarks:
    """Performance benchmark test suite."""
    
    @pytest.mark.benchmark
    def test_training_throughput(self, benchmark_model, benchmark_data):
        """Benchmark training throughput."""
        trainer = GaudiTrainer(model=benchmark_model)
        
        start_time = time.time()
        
        # Simulate training loop
        for batch in benchmark_data[:10]:  # Test with 10 batches
            with patch('torch.autograd.backward'):
                loss = trainer.training_step(batch, 0)
                assert loss is not None
        
        end_time = time.time()
        throughput = 10 / (end_time - start_time)  # batches per second
        
        # Assert minimum throughput (adjust based on requirements)
        assert throughput > 1.0, f"Throughput too low: {throughput} batches/sec"
    
    @pytest.mark.benchmark
    def test_memory_usage(self, benchmark_model):
        """Benchmark memory usage during training."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        trainer = GaudiTrainer(model=benchmark_model)
        
        # Simulate memory-intensive operations
        for _ in range(5):
            trainer.training_step({'input_ids': Mock()}, 0)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Assert reasonable memory usage (adjust threshold as needed)
        assert memory_increase < 500, f"Memory usage too high: {memory_increase}MB"
    
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_large_batch_processing(self, benchmark_model):
        """Benchmark processing of large batches."""
        trainer = GaudiTrainer(model=benchmark_model)
        
        # Simulate large batch
        large_batch = {
            'input_ids': Mock(),
            'attention_mask': Mock(), 
            'labels': Mock()
        }
        
        start_time = time.time()
        
        # Process large batch multiple times
        for _ in range(3):
            loss = trainer.training_step(large_batch, 0)
            assert loss is not None
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Assert reasonable processing time
        assert processing_time < 10.0, f"Large batch processing too slow: {processing_time}s"
    
    @pytest.mark.benchmark
    def test_accelerator_initialization_time(self):
        """Benchmark accelerator initialization time."""
        start_time = time.time()
        
        with patch('habana_frameworks.torch.hpu.is_available', return_value=True):
            accelerator = GaudiAccelerator()
            assert accelerator is not None
        
        end_time = time.time()
        init_time = end_time - start_time
        
        # Assert fast initialization
        assert init_time < 1.0, f"Accelerator initialization too slow: {init_time}s"