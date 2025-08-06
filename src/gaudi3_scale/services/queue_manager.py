"""Advanced batch processing and queue management system with priority support."""

import asyncio
import time
import threading
import logging
import pickle
import json
import heapq
from typing import Any, Dict, List, Optional, Union, Callable, AsyncIterator, Iterator, Type
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
import weakref
from abc import ABC, abstractmethod
import uuid

from ..cache.distributed_cache import get_distributed_cache
from ..monitoring.performance import get_performance_monitor

logger = logging.getLogger(__name__)

class TaskPriority(IntEnum):
    """Task priority levels (lower number = higher priority)."""
    CRITICAL = 0
    HIGH = 1  
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class QueueType(Enum):
    """Queue types."""
    FIFO = "fifo"
    LIFO = "lifo"
    PRIORITY = "priority"
    DELAY = "delay"
    RATE_LIMITED = "rate_limited"

@dataclass
class Task:
    """Task representation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    func: Callable = None
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    scheduled_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    result: Any = None
    error: Optional[Exception] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 300.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """For priority queue ordering."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.scheduled_at < other.scheduled_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization."""
        return {
            'id': self.id,
            'priority': self.priority.value,
            'status': self.status.value,
            'created_at': self.created_at,
            'scheduled_at': self.scheduled_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'timeout': self.timeout,
            'metadata': self.metadata,
            'error': str(self.error) if self.error else None
        }

@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int = 50
    batch_timeout: float = 5.0  # seconds
    max_concurrent_batches: int = 5
    enable_dynamic_batching: bool = True
    min_batch_size: int = 1
    batch_flush_interval: float = 1.0
    retry_failed_items: bool = True

@dataclass
class QueueConfig:
    """Configuration for queue management."""
    max_size: int = 10000
    consumer_count: int = 5
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff_factor: float = 2.0
    max_retry_delay: float = 60.0
    dead_letter_queue: bool = True
    task_timeout: float = 300.0
    rate_limit: Optional[float] = None  # tasks per second
    enable_persistence: bool = False
    persistence_interval: float = 10.0

class TaskQueue(ABC):
    """Abstract base class for task queues."""
    
    @abstractmethod
    async def put(self, task: Task) -> None:
        """Add task to queue."""
        pass
    
    @abstractmethod
    async def get(self) -> Task:
        """Get task from queue."""
        pass
    
    @abstractmethod
    async def task_done(self, task: Task) -> None:
        """Mark task as completed."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get queue size."""
        pass
    
    @abstractmethod
    def empty(self) -> bool:
        """Check if queue is empty."""
        pass

class PriorityTaskQueue(TaskQueue):
    """Priority-based task queue."""
    
    def __init__(self, config: QueueConfig):
        self.config = config
        self._queue: List[Task] = []
        self._queue_lock = asyncio.Lock()
        self._not_empty = asyncio.Condition()
        self._unfinished_tasks = 0
        
    async def put(self, task: Task) -> None:
        """Add task to priority queue."""
        async with self._queue_lock:
            if len(self._queue) >= self.config.max_size:
                raise RuntimeError("Queue is full")
            
            heapq.heappush(self._queue, task)
            self._unfinished_tasks += 1
            
        async with self._not_empty:
            self._not_empty.notify()
    
    async def get(self) -> Task:
        """Get highest priority task from queue."""
        async with self._not_empty:
            while self.empty():
                await self._not_empty.wait()
        
        async with self._queue_lock:
            return heapq.heappop(self._queue)
    
    async def task_done(self, task: Task) -> None:
        """Mark task as completed."""
        self._unfinished_tasks = max(0, self._unfinished_tasks - 1)
    
    def size(self) -> int:
        """Get queue size."""
        return len(self._queue)
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._queue) == 0

class DelayQueue(TaskQueue):
    """Delay queue for scheduled tasks."""
    
    def __init__(self, config: QueueConfig):
        self.config = config
        self._queue: List[Task] = []
        self._queue_lock = asyncio.Lock()
        self._not_empty = asyncio.Condition()
        self._unfinished_tasks = 0
        
    async def put(self, task: Task) -> None:
        """Add task to delay queue."""
        if task.scheduled_at <= 0:
            task.scheduled_at = time.time()
            
        async with self._queue_lock:
            if len(self._queue) >= self.config.max_size:
                raise RuntimeError("Queue is full")
            
            heapq.heappush(self._queue, task)
            self._unfinished_tasks += 1
            
        async with self._not_empty:
            self._not_empty.notify()
    
    async def get(self) -> Task:
        """Get task that is ready for execution."""
        while True:
            async with self._not_empty:
                while self.empty():
                    await self._not_empty.wait()
            
            async with self._queue_lock:
                if self._queue and self._queue[0].scheduled_at <= time.time():
                    return heapq.heappop(self._queue)
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)
    
    async def task_done(self, task: Task) -> None:
        """Mark task as completed."""
        self._unfinished_tasks = max(0, self._unfinished_tasks - 1)
    
    def size(self) -> int:
        """Get queue size."""
        return len(self._queue)
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._queue) == 0

class RateLimitedQueue(TaskQueue):
    """Rate-limited task queue."""
    
    def __init__(self, config: QueueConfig):
        self.config = config
        self._queue: asyncio.Queue[Task] = asyncio.Queue(maxsize=config.max_size)
        self._last_get = 0.0
        self._rate_limiter = asyncio.Semaphore(1)
        
    async def put(self, task: Task) -> None:
        """Add task to rate-limited queue."""
        await self._queue.put(task)
    
    async def get(self) -> Task:
        """Get task with rate limiting."""
        async with self._rate_limiter:
            if self.config.rate_limit:
                min_interval = 1.0 / self.config.rate_limit
                elapsed = time.time() - self._last_get
                if elapsed < min_interval:
                    await asyncio.sleep(min_interval - elapsed)
            
            task = await self._queue.get()
            self._last_get = time.time()
            return task
    
    async def task_done(self, task: Task) -> None:
        """Mark task as completed."""
        self._queue.task_done()
    
    def size(self) -> int:
        """Get queue size."""
        return self._queue.qsize()
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()

class BatchProcessor:
    """High-performance batch processor."""
    
    def __init__(self, config: BatchConfig, processor_func: Callable):
        self.config = config
        self.processor_func = processor_func
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Batch management
        self._current_batch: List[Any] = []
        self._batch_lock = asyncio.Lock()
        self._last_flush = time.time()
        self._processing_semaphore = asyncio.Semaphore(config.max_concurrent_batches)
        
        # Metrics
        self._batches_processed = 0
        self._items_processed = 0
        self._failed_items = 0
        self._avg_batch_size = 0.0
        
        # Background flush task
        self._flush_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
    async def start(self) -> None:
        """Start batch processor."""
        if self.config.batch_flush_interval > 0:
            self._flush_task = asyncio.create_task(self._flush_loop())
        self.logger.info("Batch processor started")
    
    async def stop(self) -> None:
        """Stop batch processor."""
        self._shutdown_event.set()
        
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Process remaining items
        await self._flush_batch()
        self.logger.info("Batch processor stopped")
    
    async def add_item(self, item: Any) -> None:
        """Add item to batch."""
        async with self._batch_lock:
            self._current_batch.append(item)
            
            # Check if we should flush
            should_flush = (
                len(self._current_batch) >= self.config.batch_size or
                (self.config.enable_dynamic_batching and 
                 time.time() - self._last_flush >= self.config.batch_timeout)
            )
            
            if should_flush:
                await self._flush_batch()
    
    async def _flush_batch(self) -> None:
        """Flush current batch for processing."""
        async with self._batch_lock:
            if not self._current_batch:
                return
            
            batch = self._current_batch.copy()
            self._current_batch.clear()
            self._last_flush = time.time()
        
        # Process batch asynchronously
        asyncio.create_task(self._process_batch(batch))
    
    async def _process_batch(self, batch: List[Any]) -> None:
        """Process a batch of items."""
        async with self._processing_semaphore:
            try:
                start_time = time.time()
                
                if asyncio.iscoroutinefunction(self.processor_func):
                    results = await self.processor_func(batch)
                else:
                    # Run in thread pool for CPU-bound operations
                    loop = asyncio.get_event_loop()
                    with ThreadPoolExecutor() as executor:
                        results = await loop.run_in_executor(executor, self.processor_func, batch)
                
                # Update metrics
                processing_time = time.time() - start_time
                self._update_metrics(len(batch), processing_time, success=True)
                
                self.logger.debug(f"Processed batch of {len(batch)} items in {processing_time:.3f}s")
                
            except Exception as e:
                self.logger.error(f"Error processing batch of {len(batch)} items: {e}")
                self._update_metrics(len(batch), 0, success=False)
                
                # Handle failed items if retry is enabled
                if self.config.retry_failed_items:
                    for item in batch:
                        asyncio.create_task(self._retry_item(item))
    
    async def _retry_item(self, item: Any) -> None:
        """Retry processing a failed item individually."""
        try:
            if asyncio.iscoroutinefunction(self.processor_func):
                await self.processor_func([item])
            else:
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    await loop.run_in_executor(executor, self.processor_func, [item])
            
            self.logger.debug(f"Successfully retried item: {item}")
            
        except Exception as e:
            self.logger.error(f"Failed to retry item {item}: {e}")
            self._failed_items += 1
    
    async def _flush_loop(self) -> None:
        """Background loop for flushing batches."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.batch_flush_interval)
                
                async with self._batch_lock:
                    elapsed = time.time() - self._last_flush
                    has_items = len(self._current_batch) >= self.config.min_batch_size
                    timeout_exceeded = elapsed >= self.config.batch_timeout
                    
                    if has_items and (timeout_exceeded or len(self._current_batch) >= self.config.batch_size):
                        await self._flush_batch()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in flush loop: {e}")
    
    def _update_metrics(self, batch_size: int, processing_time: float, success: bool) -> None:
        """Update processing metrics."""
        self._batches_processed += 1
        self._items_processed += batch_size
        
        # Update average batch size with exponential moving average
        self._avg_batch_size = (self._avg_batch_size * 0.9) + (batch_size * 0.1)
        
        if not success:
            self._failed_items += batch_size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'batches_processed': self._batches_processed,
            'items_processed': self._items_processed,
            'failed_items': self._failed_items,
            'avg_batch_size': self._avg_batch_size,
            'current_batch_size': len(self._current_batch),
            'success_rate': (self._items_processed - self._failed_items) / max(self._items_processed, 1) * 100,
            'config': {
                'batch_size': self.config.batch_size,
                'batch_timeout': self.config.batch_timeout,
                'max_concurrent_batches': self.config.max_concurrent_batches
            }
        }

class QueueManager:
    """Advanced queue management system."""
    
    def __init__(self, config: Optional[QueueConfig] = None):
        self.config = config or QueueConfig()
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Queue storage
        self._queues: Dict[str, TaskQueue] = {}
        self._consumers: Dict[str, List[asyncio.Task]] = {}
        self._dead_letter_queue: Optional[TaskQueue] = None
        
        # Batch processors
        self._batch_processors: Dict[str, BatchProcessor] = {}
        
        # Persistence
        self._cache = get_distributed_cache() if self.config.enable_persistence else None
        self._persistence_task: Optional[asyncio.Task] = None
        
        # Metrics
        self._performance_monitor = get_performance_monitor()
        self._task_counter = self._performance_monitor.counter("queue.tasks_processed")
        self._error_counter = self._performance_monitor.counter("queue.errors")
        self._processing_timer = self._performance_monitor.timer("queue.processing_time")
        
        # Shutdown management
        self._shutdown_event = asyncio.Event()
        
        # Initialize dead letter queue
        if self.config.dead_letter_queue:
            self._dead_letter_queue = PriorityTaskQueue(self.config)
    
    async def create_queue(self, name: str, queue_type: QueueType = QueueType.PRIORITY, 
                          config: Optional[QueueConfig] = None) -> TaskQueue:
        """Create a new task queue."""
        queue_config = config or self.config
        
        if queue_type == QueueType.PRIORITY:
            queue = PriorityTaskQueue(queue_config)
        elif queue_type == QueueType.DELAY:
            queue = DelayQueue(queue_config)
        elif queue_type == QueueType.RATE_LIMITED:
            queue = RateLimitedQueue(queue_config)
        else:
            # Default to priority queue
            queue = PriorityTaskQueue(queue_config)
        
        self._queues[name] = queue
        
        # Start consumers
        await self._start_consumers(name, queue_config.consumer_count)
        
        self.logger.info(f"Created {queue_type.value} queue '{name}' with {queue_config.consumer_count} consumers")
        return queue
    
    async def get_queue(self, name: str) -> Optional[TaskQueue]:
        """Get queue by name."""
        return self._queues.get(name)
    
    async def submit_task(self, queue_name: str, func: Callable, *args, 
                         priority: TaskPriority = TaskPriority.NORMAL,
                         delay: float = 0.0, timeout: float = 300.0,
                         max_retries: int = 3, **kwargs) -> str:
        """Submit task to queue."""
        task = Task(
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            scheduled_at=time.time() + delay,
            timeout=timeout,
            max_retries=max_retries
        )
        
        queue = self._queues.get(queue_name)
        if not queue:
            raise ValueError(f"Queue '{queue_name}' not found")
        
        await queue.put(task)
        
        # Cache task if persistence is enabled
        if self._cache:
            await self._cache_task(task)
        
        self.logger.debug(f"Submitted task {task.id} to queue '{queue_name}'")
        return task.id
    
    async def submit_batch_task(self, queue_name: str, items: List[Any], 
                               processor_func: Callable,
                               batch_config: Optional[BatchConfig] = None) -> str:
        """Submit batch processing task."""
        config = batch_config or BatchConfig()
        
        # Create or get batch processor
        processor_key = f"{queue_name}_batch_{hash(processor_func.__name__)}"
        if processor_key not in self._batch_processors:
            self._batch_processors[processor_key] = BatchProcessor(config, processor_func)
            await self._batch_processors[processor_key].start()
        
        processor = self._batch_processors[processor_key]
        
        # Add items to processor
        for item in items:
            await processor.add_item(item)
        
        batch_id = str(uuid.uuid4())
        self.logger.info(f"Submitted batch task {batch_id} with {len(items)} items")
        return batch_id
    
    async def _start_consumers(self, queue_name: str, consumer_count: int) -> None:
        """Start consumer tasks for queue."""
        consumers = []
        
        for i in range(consumer_count):
            consumer_task = asyncio.create_task(
                self._consumer_worker(queue_name, f"{queue_name}_consumer_{i}")
            )
            consumers.append(consumer_task)
        
        self._consumers[queue_name] = consumers
    
    async def _consumer_worker(self, queue_name: str, consumer_id: str) -> None:
        """Consumer worker that processes tasks from queue."""
        queue = self._queues[queue_name]
        
        while not self._shutdown_event.is_set():
            try:
                # Get task from queue
                task = await queue.get()
                task.status = TaskStatus.RUNNING
                task.started_at = time.time()
                
                # Process task
                await self._process_task(task, consumer_id)
                
                # Mark task as done
                await queue.task_done(task)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in consumer {consumer_id}: {e}")
                await asyncio.sleep(1)  # Brief pause before retrying
    
    async def _process_task(self, task: Task, consumer_id: str) -> None:
        """Process individual task."""
        with self._processing_timer.measure():
            try:
                # Set timeout
                if task.timeout > 0:
                    result = await asyncio.wait_for(
                        self._execute_task(task),
                        timeout=task.timeout
                    )
                else:
                    result = await self._execute_task(task)
                
                # Task completed successfully
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
                
                self._task_counter.increment()
                self.logger.debug(f"Task {task.id} completed by {consumer_id}")
                
            except asyncio.TimeoutError:
                await self._handle_task_failure(task, TimeoutError("Task timeout"))
            except Exception as e:
                await self._handle_task_failure(task, e)
    
    async def _execute_task(self, task: Task) -> Any:
        """Execute task function."""
        if asyncio.iscoroutinefunction(task.func):
            return await task.func(*task.args, **task.kwargs)
        else:
            # Run in thread pool for CPU-bound tasks
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                return await loop.run_in_executor(executor, task.func, *task.args, **task.kwargs)
    
    async def _handle_task_failure(self, task: Task, error: Exception) -> None:
        """Handle task failure with retry logic."""
        task.error = error
        task.retry_count += 1
        
        self._error_counter.increment()
        self.logger.warning(f"Task {task.id} failed (attempt {task.retry_count}): {error}")
        
        if task.retry_count <= task.max_retries:
            # Retry task with exponential backoff
            delay = self.config.retry_delay * (self.config.retry_backoff_factor ** (task.retry_count - 1))
            delay = min(delay, self.config.max_retry_delay)
            
            task.status = TaskStatus.RETRYING
            task.scheduled_at = time.time() + delay
            
            # Re-queue task
            queue = self._queues.get(task.metadata.get('queue_name'))
            if queue:
                await queue.put(task)
            
            self.logger.info(f"Task {task.id} scheduled for retry in {delay:.1f}s")
        else:
            # Task failed permanently
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            
            # Send to dead letter queue if enabled
            if self._dead_letter_queue:
                await self._dead_letter_queue.put(task)
                self.logger.warning(f"Task {task.id} sent to dead letter queue after {task.retry_count} failures")
    
    async def _cache_task(self, task: Task) -> None:
        """Cache task for persistence."""
        if self._cache:
            try:
                cache_key = f"task:{task.id}"
                task_data = task.to_dict()
                await self._cache.set(cache_key, task_data, ttl=86400)  # 24 hours
            except Exception as e:
                self.logger.warning(f"Failed to cache task {task.id}: {e}")
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status by ID."""
        if self._cache:
            cache_key = f"task:{task_id}"
            task_data = await self._cache.get(cache_key)
            return task_data
        return None
    
    def get_queue_stats(self, queue_name: str) -> Dict[str, Any]:
        """Get queue statistics."""
        queue = self._queues.get(queue_name)
        if not queue:
            return {}
        
        consumers = self._consumers.get(queue_name, [])
        
        return {
            'name': queue_name,
            'size': queue.size(),
            'empty': queue.empty(),
            'consumers': len(consumers),
            'active_consumers': len([c for c in consumers if not c.done()])
        }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all queues."""
        stats = {
            'queues': {},
            'batch_processors': {},
            'dead_letter_queue_size': self._dead_letter_queue.size() if self._dead_letter_queue else 0,
            'total_queues': len(self._queues)
        }
        
        # Queue stats
        for queue_name in self._queues:
            stats['queues'][queue_name] = self.get_queue_stats(queue_name)
        
        # Batch processor stats
        for processor_name, processor in self._batch_processors.items():
            stats['batch_processors'][processor_name] = processor.get_stats()
        
        return stats
    
    async def shutdown(self) -> None:
        """Shutdown queue manager."""
        self.logger.info("Shutting down queue manager...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Stop batch processors
        for processor in self._batch_processors.values():
            await processor.stop()
        
        # Cancel consumer tasks
        for consumers in self._consumers.values():
            for consumer in consumers:
                consumer.cancel()
        
        # Wait for consumers to finish
        for consumers in self._consumers.values():
            await asyncio.gather(*consumers, return_exceptions=True)
        
        # Stop persistence task
        if self._persistence_task:
            self._persistence_task.cancel()
            try:
                await self._persistence_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Queue manager shutdown complete")


# Global queue manager
_queue_manager: Optional[QueueManager] = None

async def get_queue_manager() -> QueueManager:
    """Get global queue manager instance."""
    global _queue_manager
    if _queue_manager is None:
        _queue_manager = QueueManager()
    return _queue_manager


# Convenience decorators
def queue_task(queue_name: str, priority: TaskPriority = TaskPriority.NORMAL):
    """Decorator to automatically queue function calls."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            manager = await get_queue_manager()
            return await manager.submit_task(queue_name, func, *args, priority=priority, **kwargs)
        return wrapper
    return decorator


def batch_process(queue_name: str, batch_config: Optional[BatchConfig] = None):
    """Decorator for batch processing functions."""
    def decorator(func):
        async def wrapper(items: List[Any]):
            manager = await get_queue_manager()
            return await manager.submit_batch_task(queue_name, items, func, batch_config)
        return wrapper
    return decorator