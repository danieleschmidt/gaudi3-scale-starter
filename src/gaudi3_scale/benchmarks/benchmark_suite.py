"""Comprehensive performance benchmarking and testing framework."""

import asyncio
import time
import statistics
import logging
import sys
import platform
import psutil
import json
import csv
from typing import Any, Dict, List, Optional, Union, Callable, AsyncIterator, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import tracemalloc
import gc
import threading

from ..monitoring.performance import get_performance_monitor
from ..cache.distributed_cache import get_distributed_cache
from ..services.async_service import get_async_service_manager
from ..algorithms.autoscaler import get_autoscaler
from ..optimization.memory_optimizer import get_memory_optimizer

logger = logging.getLogger(__name__)

class BenchmarkType(Enum):
    """Types of benchmarks."""
    CPU_INTENSIVE = "cpu_intensive"
    IO_INTENSIVE = "io_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    NETWORK_INTENSIVE = "network_intensive"
    CACHE_PERFORMANCE = "cache_performance"
    SCALING_PERFORMANCE = "scaling_performance"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    STRESS_TEST = "stress_test"
    LOAD_TEST = "load_test"

class BenchmarkStatus(Enum):
    """Benchmark execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    duration_seconds: float = 60.0
    warmup_seconds: float = 10.0
    iterations: int = 1
    concurrent_workers: int = 1
    sample_rate: float = 1.0  # seconds
    collect_detailed_metrics: bool = True
    enable_profiling: bool = False
    output_format: str = "json"  # json, csv, html
    output_path: Optional[Path] = None
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class BenchmarkResult:
    """Results from benchmark execution."""
    name: str
    benchmark_type: BenchmarkType
    status: BenchmarkStatus
    start_time: float
    end_time: float
    duration: float
    iterations: int
    
    # Performance metrics
    throughput: float = 0.0  # operations/second
    avg_latency: float = 0.0  # milliseconds
    min_latency: float = 0.0
    max_latency: float = 0.0
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    
    # Resource usage
    avg_cpu_percent: float = 0.0
    peak_cpu_percent: float = 0.0
    avg_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    
    # Error metrics
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    error_rate: float = 0.0
    
    # Detailed metrics
    latency_samples: List[float] = field(default_factory=list)
    resource_samples: List[Dict[str, float]] = field(default_factory=list)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Environment info
    system_info: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)
    
    def calculate_statistics(self):
        """Calculate statistical metrics from samples."""
        if self.latency_samples:
            sorted_latencies = sorted(self.latency_samples)
            self.min_latency = sorted_latencies[0]
            self.max_latency = sorted_latencies[-1]
            self.avg_latency = statistics.mean(sorted_latencies)
            
            n = len(sorted_latencies)
            if n > 0:
                self.p50_latency = sorted_latencies[int(n * 0.5)]
                self.p95_latency = sorted_latencies[int(n * 0.95)]
                self.p99_latency = sorted_latencies[int(n * 0.99)]
        
        if self.resource_samples:
            cpu_values = [s.get('cpu', 0) for s in self.resource_samples]
            memory_values = [s.get('memory_mb', 0) for s in self.resource_samples]
            
            if cpu_values:
                self.avg_cpu_percent = statistics.mean(cpu_values)
                self.peak_cpu_percent = max(cpu_values)
            
            if memory_values:
                self.avg_memory_mb = statistics.mean(memory_values)
                self.peak_memory_mb = max(memory_values)
        
        # Calculate throughput
        if self.duration > 0:
            self.throughput = self.successful_operations / self.duration
        
        # Calculate error rate
        if self.total_operations > 0:
            self.error_rate = (self.failed_operations / self.total_operations) * 100

class BenchmarkRunner:
    """Base class for benchmark runners."""
    
    def __init__(self, name: str, benchmark_type: BenchmarkType, config: BenchmarkConfig):
        self.name = name
        self.benchmark_type = benchmark_type
        self.config = config
        self.logger = logger.getChild(f"Benchmark.{name}")
        
        # State
        self._start_time = 0.0
        self._end_time = 0.0
        self._cancelled = False
        
        # Metrics collection
        self._latency_samples = deque()
        self._resource_samples = deque()
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Performance monitor
        self._perf_monitor = get_performance_monitor()
        self._timer = self._perf_monitor.timer(f"benchmark.{name}")
    
    async def run(self) -> BenchmarkResult:
        """Run the benchmark and return results."""
        self.logger.info(f"Starting benchmark '{self.name}'")
        
        result = BenchmarkResult(
            name=self.name,
            benchmark_type=self.benchmark_type,
            status=BenchmarkStatus.RUNNING,
            start_time=time.time(),
            end_time=0.0,
            duration=0.0,
            iterations=self.config.iterations
        )
        
        try:
            # System info collection
            result.system_info = self._collect_system_info()
            result.config = asdict(self.config)
            
            # Start resource monitoring
            if self.config.collect_detailed_metrics:
                self._monitor_task = asyncio.create_task(self._monitor_resources())
            
            # Warmup phase
            if self.config.warmup_seconds > 0:
                self.logger.info(f"Warming up for {self.config.warmup_seconds}s")
                await self._warmup()
            
            self._start_time = time.time()
            
            # Run benchmark
            await self._execute_benchmark(result)
            
            self._end_time = time.time()
            result.end_time = self._end_time
            result.duration = self._end_time - self._start_time
            
            # Stop monitoring
            if self._monitor_task:
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Collect final metrics
            result.latency_samples = list(self._latency_samples)
            result.resource_samples = list(self._resource_samples)
            result.calculate_statistics()
            
            result.status = BenchmarkStatus.COMPLETED
            self.logger.info(f"Benchmark '{self.name}' completed successfully")
            
        except Exception as e:
            result.status = BenchmarkStatus.FAILED
            result.custom_metrics['error'] = str(e)
            self.logger.error(f"Benchmark '{self.name}' failed: {e}")
            
        return result
    
    async def _execute_benchmark(self, result: BenchmarkResult):
        """Execute the actual benchmark logic. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _execute_benchmark")
    
    async def _warmup(self):
        """Perform warmup operations. Override in subclasses if needed."""
        await asyncio.sleep(self.config.warmup_seconds)
    
    async def _monitor_resources(self):
        """Monitor system resources during benchmark."""
        process = psutil.Process()
        
        while not self._cancelled:
            try:
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                
                sample = {
                    'timestamp': time.time(),
                    'cpu': cpu_percent,
                    'memory_mb': memory_mb,
                    'threads': process.num_threads(),
                }
                
                self._resource_samples.append(sample)
                
                await asyncio.sleep(self.config.sample_rate)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.warning(f"Error collecting resource metrics: {e}")
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information."""
        return {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': sys.version,
            'cpu_count': psutil.cpu_count(),
            'memory_total_mb': psutil.virtual_memory().total / (1024 * 1024),
            'timestamp': datetime.now().isoformat()
        }
    
    def record_operation(self, latency: float, success: bool = True):
        """Record operation result."""
        if self.config.collect_detailed_metrics:
            self._latency_samples.append(latency * 1000)  # Convert to milliseconds
    
    def cancel(self):
        """Cancel benchmark execution."""
        self._cancelled = True
        if self._monitor_task:
            self._monitor_task.cancel()

class CPUBenchmark(BenchmarkRunner):
    """CPU-intensive benchmark."""
    
    def __init__(self, config: BenchmarkConfig):
        super().__init__("CPU Intensive", BenchmarkType.CPU_INTENSIVE, config)
    
    async def _execute_benchmark(self, result: BenchmarkResult):
        """Execute CPU-intensive benchmark."""
        def cpu_intensive_work(n: int) -> int:
            """CPU-intensive calculation."""
            total = 0
            for i in range(n):
                total += i * i
                if i % 1000 == 0:
                    # Yield to allow cancellation
                    if self._cancelled:
                        break
            return total
        
        operations = 0
        failures = 0
        
        with ThreadPoolExecutor(max_workers=self.config.concurrent_workers) as executor:
            end_time = time.time() + self.config.duration_seconds
            
            while time.time() < end_time and not self._cancelled:
                start = time.time()
                
                try:
                    # Submit CPU work
                    futures = [
                        executor.submit(cpu_intensive_work, 100000)
                        for _ in range(self.config.concurrent_workers)
                    ]
                    
                    # Wait for completion
                    for future in futures:
                        future.result(timeout=1.0)
                    
                    latency = time.time() - start
                    self.record_operation(latency, True)
                    operations += self.config.concurrent_workers
                    
                except Exception as e:
                    failures += 1
                    self.logger.warning(f"CPU operation failed: {e}")
        
        result.total_operations = operations
        result.successful_operations = operations - failures
        result.failed_operations = failures

class IOBenchmark(BenchmarkRunner):
    """I/O intensive benchmark."""
    
    def __init__(self, config: BenchmarkConfig):
        super().__init__("I/O Intensive", BenchmarkType.IO_INTENSIVE, config)
    
    async def _execute_benchmark(self, result: BenchmarkResult):
        """Execute I/O intensive benchmark."""
        service_manager = await get_async_service_manager()
        file_service = await service_manager.get_file_service()
        
        operations = 0
        failures = 0
        
        async def io_operation():
            """Perform I/O operation."""
            import tempfile
            
            with tempfile.NamedTemporaryFile(mode='w+', delete=True) as tmp_file:
                test_data = "x" * 1024  # 1KB test data
                
                # Write operation
                await file_service.write_file(tmp_file.name, test_data)
                
                # Read operation
                data = await file_service.read_file(tmp_file.name)
                
                return len(data)
        
        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(self.config.concurrent_workers)
        
        async def limited_io_operation():
            async with semaphore:
                return await io_operation()
        
        end_time = time.time() + self.config.duration_seconds
        
        while time.time() < end_time and not self._cancelled:
            start = time.time()
            
            try:
                # Run concurrent I/O operations
                tasks = [
                    limited_io_operation()
                    for _ in range(self.config.concurrent_workers)
                ]
                
                await asyncio.gather(*tasks)
                
                latency = time.time() - start
                self.record_operation(latency, True)
                operations += self.config.concurrent_workers
                
            except Exception as e:
                failures += 1
                self.logger.warning(f"I/O operation failed: {e}")
        
        result.total_operations = operations
        result.successful_operations = operations - failures
        result.failed_operations = failures

class MemoryBenchmark(BenchmarkRunner):
    """Memory intensive benchmark."""
    
    def __init__(self, config: BenchmarkConfig):
        super().__init__("Memory Intensive", BenchmarkType.MEMORY_INTENSIVE, config)
    
    async def _execute_benchmark(self, result: BenchmarkResult):
        """Execute memory intensive benchmark."""
        operations = 0
        failures = 0
        allocated_objects = []
        
        # Start memory tracking
        tracemalloc.start()
        
        end_time = time.time() + self.config.duration_seconds
        
        while time.time() < end_time and not self._cancelled:
            start = time.time()
            
            try:
                # Allocate and manipulate memory
                for _ in range(self.config.concurrent_workers):
                    # Allocate large list
                    data = list(range(100000))
                    
                    # Perform operations
                    data.sort()
                    data.reverse()
                    
                    # Keep reference to prevent immediate GC
                    allocated_objects.append(data)
                    
                    # Periodically clear to test GC
                    if len(allocated_objects) > 100:
                        allocated_objects.clear()
                        gc.collect()
                
                latency = time.time() - start
                self.record_operation(latency, True)
                operations += self.config.concurrent_workers
                
            except MemoryError as e:
                failures += 1
                self.logger.warning(f"Memory operation failed: {e}")
                # Emergency cleanup
                allocated_objects.clear()
                gc.collect()
                
        # Final cleanup
        allocated_objects.clear()
        
        # Get memory statistics
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        result.custom_metrics['peak_memory_traced_mb'] = peak / (1024 * 1024)
        result.custom_metrics['current_memory_traced_mb'] = current / (1024 * 1024)
        
        result.total_operations = operations
        result.successful_operations = operations - failures
        result.failed_operations = failures

class CacheBenchmark(BenchmarkRunner):
    """Cache performance benchmark."""
    
    def __init__(self, config: BenchmarkConfig):
        super().__init__("Cache Performance", BenchmarkType.CACHE_PERFORMANCE, config)
    
    async def _execute_benchmark(self, result: BenchmarkResult):
        """Execute cache performance benchmark."""
        cache = get_distributed_cache()
        
        operations = 0
        failures = 0
        cache_hits = 0
        cache_misses = 0
        
        # Pre-populate cache with test data
        for i in range(1000):
            await cache.set(f"test_key_{i}", f"test_value_{i}", ttl=300)
        
        end_time = time.time() + self.config.duration_seconds
        
        while time.time() < end_time and not self._cancelled:
            start = time.time()
            
            try:
                # Mix of cache operations
                tasks = []
                
                for _ in range(self.config.concurrent_workers):
                    key = f"test_key_{operations % 1000}"
                    
                    if operations % 3 == 0:
                        # GET operation
                        task = cache.get(key)
                    elif operations % 3 == 1:
                        # SET operation
                        task = cache.set(f"new_key_{operations}", f"value_{operations}", ttl=60)
                    else:
                        # DELETE operation
                        task = cache.delete(f"old_key_{operations - 100}")
                    
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Count hits/misses
                for res in results:
                    if res is not None and not isinstance(res, Exception):
                        cache_hits += 1
                    else:
                        cache_misses += 1
                
                latency = time.time() - start
                self.record_operation(latency, True)
                operations += self.config.concurrent_workers
                
            except Exception as e:
                failures += 1
                self.logger.warning(f"Cache operation failed: {e}")
        
        result.custom_metrics['cache_hits'] = cache_hits
        result.custom_metrics['cache_misses'] = cache_misses
        result.custom_metrics['cache_hit_rate'] = cache_hits / (cache_hits + cache_misses) * 100 if (cache_hits + cache_misses) > 0 else 0
        
        result.total_operations = operations
        result.successful_operations = operations - failures
        result.failed_operations = failures

class ThroughputBenchmark(BenchmarkRunner):
    """Throughput benchmark."""
    
    def __init__(self, config: BenchmarkConfig, target_function: Callable):
        super().__init__("Throughput", BenchmarkType.THROUGHPUT, config)
        self.target_function = target_function
    
    async def _execute_benchmark(self, result: BenchmarkResult):
        """Execute throughput benchmark."""
        operations = 0
        failures = 0
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.concurrent_workers)
        
        async def run_operation():
            async with semaphore:
                if asyncio.iscoroutinefunction(self.target_function):
                    return await self.target_function()
                else:
                    return self.target_function()
        
        end_time = time.time() + self.config.duration_seconds
        
        while time.time() < end_time and not self._cancelled:
            batch_start = time.time()
            
            try:
                # Run batch of operations
                batch_size = min(self.config.concurrent_workers, 100)
                tasks = [run_operation() for _ in range(batch_size)]
                
                await asyncio.gather(*tasks, return_exceptions=True)
                
                batch_latency = time.time() - batch_start
                self.record_operation(batch_latency / batch_size, True)
                operations += batch_size
                
            except Exception as e:
                failures += 1
                self.logger.warning(f"Throughput operation failed: {e}")
        
        result.total_operations = operations
        result.successful_operations = operations - failures
        result.failed_operations = failures

class BenchmarkSuite:
    """Main benchmark suite coordinator."""
    
    def __init__(self, output_path: Optional[Path] = None):
        self.output_path = output_path or Path("benchmark_results")
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Results storage
        self.results: List[BenchmarkResult] = []
        self.running_benchmarks: Dict[str, BenchmarkRunner] = {}
        
        # Default benchmarks
        self._default_benchmarks = [
            (CPUBenchmark, BenchmarkConfig(duration_seconds=30, concurrent_workers=4)),
            (IOBenchmark, BenchmarkConfig(duration_seconds=30, concurrent_workers=10)),
            (MemoryBenchmark, BenchmarkConfig(duration_seconds=30, concurrent_workers=2)),
            (CacheBenchmark, BenchmarkConfig(duration_seconds=30, concurrent_workers=8)),
        ]
    
    async def run_all_benchmarks(self, config: Optional[BenchmarkConfig] = None) -> List[BenchmarkResult]:
        """Run all default benchmarks."""
        self.logger.info("Starting comprehensive benchmark suite")
        
        results = []
        
        for benchmark_class, default_config in self._default_benchmarks:
            final_config = config or default_config
            benchmark = benchmark_class(final_config)
            
            try:
                result = await benchmark.run()
                results.append(result)
                self.results.append(result)
            except Exception as e:
                self.logger.error(f"Benchmark {benchmark.name} failed: {e}")
        
        # Save results
        await self._save_results(results)
        
        self.logger.info(f"Benchmark suite completed. {len(results)} benchmarks executed.")
        return results
    
    async def run_custom_benchmark(self, benchmark: BenchmarkRunner) -> BenchmarkResult:
        """Run a custom benchmark."""
        self.running_benchmarks[benchmark.name] = benchmark
        
        try:
            result = await benchmark.run()
            self.results.append(result)
            return result
        finally:
            self.running_benchmarks.pop(benchmark.name, None)
    
    async def run_throughput_benchmark(self, 
                                     target_function: Callable,
                                     config: Optional[BenchmarkConfig] = None) -> BenchmarkResult:
        """Run throughput benchmark for specific function."""
        benchmark_config = config or BenchmarkConfig(duration_seconds=60, concurrent_workers=10)
        benchmark = ThroughputBenchmark(benchmark_config, target_function)
        
        return await self.run_custom_benchmark(benchmark)
    
    async def run_stress_test(self, duration_minutes: int = 10) -> List[BenchmarkResult]:
        """Run stress test with extended duration."""
        stress_config = BenchmarkConfig(
            duration_seconds=duration_minutes * 60,
            concurrent_workers=20,
            collect_detailed_metrics=True,
            sample_rate=5.0
        )
        
        # Run all benchmarks with stress configuration
        return await self.run_all_benchmarks(stress_config)
    
    async def _save_results(self, results: List[BenchmarkResult]):
        """Save benchmark results to files."""
        if not results:
            return
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_file = self.output_path / f"benchmark_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump([result.to_dict() for result in results], f, indent=2, default=str)
        
        # Save CSV summary
        csv_file = self.output_path / f"benchmark_summary_{timestamp}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Name', 'Type', 'Status', 'Duration', 'Throughput',
                'Avg Latency', 'P95 Latency', 'P99 Latency',
                'Success Rate', 'Peak CPU', 'Peak Memory'
            ])
            
            # Data
            for result in results:
                writer.writerow([
                    result.name,
                    result.benchmark_type.value,
                    result.status.value,
                    f"{result.duration:.2f}s",
                    f"{result.throughput:.2f} ops/s",
                    f"{result.avg_latency:.2f}ms",
                    f"{result.p95_latency:.2f}ms",
                    f"{result.p99_latency:.2f}ms",
                    f"{100 - result.error_rate:.2f}%",
                    f"{result.peak_cpu_percent:.1f}%",
                    f"{result.peak_memory_mb:.1f}MB"
                ])
        
        self.logger.info(f"Benchmark results saved to {json_file} and {csv_file}")
    
    def generate_report(self, results: Optional[List[BenchmarkResult]] = None) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        target_results = results or self.results
        
        if not target_results:
            return {}
        
        # Calculate aggregate statistics
        total_operations = sum(r.total_operations for r in target_results)
        total_duration = sum(r.duration for r in target_results)
        avg_throughput = statistics.mean([r.throughput for r in target_results if r.throughput > 0])
        avg_latency = statistics.mean([r.avg_latency for r in target_results if r.avg_latency > 0])
        
        # Group by benchmark type
        by_type = defaultdict(list)
        for result in target_results:
            by_type[result.benchmark_type.value].append(result)
        
        return {
            'summary': {
                'total_benchmarks': len(target_results),
                'successful_benchmarks': len([r for r in target_results if r.status == BenchmarkStatus.COMPLETED]),
                'total_operations': total_operations,
                'total_duration': total_duration,
                'avg_throughput': avg_throughput,
                'avg_latency': avg_latency,
                'timestamp': datetime.now().isoformat()
            },
            'by_type': dict(by_type),
            'detailed_results': [result.to_dict() for result in target_results],
            'system_info': target_results[0].system_info if target_results else {}
        }
    
    def cancel_all_benchmarks(self):
        """Cancel all running benchmarks."""
        for benchmark in self.running_benchmarks.values():
            benchmark.cancel()
        
        self.logger.info("All running benchmarks cancelled")
    
    def get_running_benchmarks(self) -> List[str]:
        """Get list of currently running benchmark names."""
        return list(self.running_benchmarks.keys())


# Global benchmark suite
_benchmark_suite: Optional[BenchmarkSuite] = None

def get_benchmark_suite() -> BenchmarkSuite:
    """Get global benchmark suite instance."""
    global _benchmark_suite
    if _benchmark_suite is None:
        _benchmark_suite = BenchmarkSuite()
    return _benchmark_suite


# Convenience functions
async def quick_benchmark(duration_seconds: int = 30) -> Dict[str, Any]:
    """Run quick benchmark suite."""
    suite = get_benchmark_suite()
    config = BenchmarkConfig(duration_seconds=duration_seconds, concurrent_workers=5)
    results = await suite.run_all_benchmarks(config)
    return suite.generate_report(results)


async def benchmark_function(func: Callable, 
                           duration_seconds: int = 60,
                           concurrent_workers: int = 10) -> BenchmarkResult:
    """Benchmark specific function."""
    suite = get_benchmark_suite()
    config = BenchmarkConfig(
        duration_seconds=duration_seconds,
        concurrent_workers=concurrent_workers,
        collect_detailed_metrics=True
    )
    
    return await suite.run_throughput_benchmark(func, config)


# Decorator for easy benchmarking
def benchmark(duration_seconds: int = 30, concurrent_workers: int = 5):
    """Decorator to benchmark function performance."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Run original function
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            # Run benchmark
            benchmark_result = await benchmark_function(
                lambda: func(*args, **kwargs), 
                duration_seconds, 
                concurrent_workers
            )
            
            logger.info(f"Benchmark for {func.__name__}: {benchmark_result.throughput:.2f} ops/s")
            return result
        return wrapper
    return decorator