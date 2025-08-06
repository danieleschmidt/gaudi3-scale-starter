"""Retry logic and resilience utilities for Gaudi 3 Scale.

This module provides comprehensive retry mechanisms, backoff strategies,
and resilience patterns for handling transient failures in distributed
training and infrastructure operations.
"""

import asyncio
import functools
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
from enum import Enum
import logging

from .exceptions import (
    Gaudi3ScaleError, NetworkError, HPUError, TrainingError, 
    ResourceError, ServiceUnavailableError, TimeoutError
)
from .logging_utils import get_logger

logger = get_logger('retry_utils')


class RetryStrategy(Enum):
    """Available retry strategies."""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"
    CUSTOM = "custom"


class BackoffStrategy(ABC):
    """Abstract base class for backoff strategies."""
    
    @abstractmethod
    def calculate_delay(self, attempt: int, base_delay: float) -> float:
        """Calculate delay for a given attempt.
        
        Args:
            attempt: Current attempt number (0-indexed)
            base_delay: Base delay in seconds
            
        Returns:
            Delay in seconds for this attempt
        """
        pass


class FixedDelayStrategy(BackoffStrategy):
    """Fixed delay between retry attempts."""
    
    def calculate_delay(self, attempt: int, base_delay: float) -> float:
        """Calculate fixed delay."""
        return base_delay


class ExponentialBackoffStrategy(BackoffStrategy):
    """Exponential backoff with jitter."""
    
    def __init__(self, multiplier: float = 2.0, max_delay: float = 300.0, jitter: bool = True):
        """Initialize exponential backoff strategy.
        
        Args:
            multiplier: Exponential multiplier
            max_delay: Maximum delay in seconds
            jitter: Whether to add random jitter
        """
        self.multiplier = multiplier
        self.max_delay = max_delay
        self.jitter = jitter
    
    def calculate_delay(self, attempt: int, base_delay: float) -> float:
        """Calculate exponential backoff delay with optional jitter."""
        delay = min(base_delay * (self.multiplier ** attempt), self.max_delay)
        
        if self.jitter:
            # Add ±25% jitter
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)


class LinearBackoffStrategy(BackoffStrategy):
    """Linear backoff strategy."""
    
    def __init__(self, increment: float = 1.0, max_delay: float = 60.0):
        """Initialize linear backoff strategy.
        
        Args:
            increment: Linear increment per attempt
            max_delay: Maximum delay in seconds
        """
        self.increment = increment
        self.max_delay = max_delay
    
    def calculate_delay(self, attempt: int, base_delay: float) -> float:
        """Calculate linear backoff delay."""
        return min(base_delay + (attempt * self.increment), self.max_delay)


class FibonacciBackoffStrategy(BackoffStrategy):
    """Fibonacci sequence backoff strategy."""
    
    def __init__(self, max_delay: float = 300.0):
        """Initialize fibonacci backoff strategy.
        
        Args:
            max_delay: Maximum delay in seconds
        """
        self.max_delay = max_delay
        self._fib_cache = {0: 1, 1: 1}
    
    def _fibonacci(self, n: int) -> int:
        """Calculate fibonacci number with caching."""
        if n not in self._fib_cache:
            self._fib_cache[n] = self._fibonacci(n-1) + self._fibonacci(n-2)
        return self._fib_cache[n]
    
    def calculate_delay(self, attempt: int, base_delay: float) -> float:
        """Calculate fibonacci backoff delay."""
        fib_multiplier = self._fibonacci(attempt)
        delay = base_delay * fib_multiplier
        return min(delay, self.max_delay)


class CustomBackoffStrategy(BackoffStrategy):
    """Custom backoff strategy using a user-provided function."""
    
    def __init__(self, delay_function: Callable[[int, float], float]):
        """Initialize custom backoff strategy.
        
        Args:
            delay_function: Function that takes (attempt, base_delay) and returns delay
        """
        self.delay_function = delay_function
    
    def calculate_delay(self, attempt: int, base_delay: float) -> float:
        """Calculate delay using custom function."""
        return self.delay_function(attempt, base_delay)


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
        backoff_strategy: Optional[BackoffStrategy] = None,
        retryable_exceptions: Optional[List[Type[Exception]]] = None,
        non_retryable_exceptions: Optional[List[Type[Exception]]] = None,
        timeout: Optional[float] = None,
        should_retry_predicate: Optional[Callable[[Exception], bool]] = None,
        on_retry_callback: Optional[Callable[[int, Exception, float], None]] = None,
        on_final_failure_callback: Optional[Callable[[Exception], None]] = None
    ):
        """Initialize retry configuration.
        
        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            strategy: Retry strategy to use
            backoff_strategy: Custom backoff strategy (overrides strategy)
            retryable_exceptions: List of exception types that should trigger retries
            non_retryable_exceptions: List of exception types that should never be retried
            timeout: Total timeout for all retry attempts
            should_retry_predicate: Custom function to determine if exception should be retried
            on_retry_callback: Callback called before each retry
            on_final_failure_callback: Callback called when all retries are exhausted
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.strategy = strategy
        self.timeout = timeout
        self.should_retry_predicate = should_retry_predicate
        self.on_retry_callback = on_retry_callback
        self.on_final_failure_callback = on_final_failure_callback
        
        # Set up backoff strategy
        if backoff_strategy:
            self.backoff_strategy = backoff_strategy
        else:
            self.backoff_strategy = self._create_backoff_strategy(strategy)
        
        # Set up retryable exceptions
        if retryable_exceptions is not None:
            self.retryable_exceptions = tuple(retryable_exceptions)
        else:
            self.retryable_exceptions = self._default_retryable_exceptions()
        
        # Set up non-retryable exceptions
        if non_retryable_exceptions is not None:
            self.non_retryable_exceptions = tuple(non_retryable_exceptions)
        else:
            self.non_retryable_exceptions = self._default_non_retryable_exceptions()
    
    def _create_backoff_strategy(self, strategy: RetryStrategy) -> BackoffStrategy:
        """Create backoff strategy based on strategy enum."""
        if strategy == RetryStrategy.FIXED_DELAY:
            return FixedDelayStrategy()
        elif strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            return ExponentialBackoffStrategy()
        elif strategy == RetryStrategy.LINEAR_BACKOFF:
            return LinearBackoffStrategy()
        elif strategy == RetryStrategy.FIBONACCI_BACKOFF:
            return FibonacciBackoffStrategy()
        else:
            return FixedDelayStrategy()  # Fallback
    
    def _default_retryable_exceptions(self) -> Tuple[Type[Exception], ...]:
        """Default list of retryable exceptions."""
        return (
            NetworkError,
            ServiceUnavailableError,
            TimeoutError,
            ConnectionError,
            OSError,
            # HPU-specific transient errors
            HPUError,  # Only if it's a transient error
        )
    
    def _default_non_retryable_exceptions(self) -> Tuple[Type[Exception], ...]:
        """Default list of non-retryable exceptions."""
        return (
            KeyboardInterrupt,
            SystemExit,
            MemoryError,
            # Configuration and validation errors typically shouldn't be retried
            ValueError,
            TypeError,
            AttributeError,
        )
    
    def should_retry(self, exception: Exception) -> bool:
        """Determine if an exception should trigger a retry.
        
        Args:
            exception: Exception that was raised
            
        Returns:
            True if the exception should be retried
        """
        # Check custom predicate first
        if self.should_retry_predicate:
            return self.should_retry_predicate(exception)
        
        # Check non-retryable exceptions
        if isinstance(exception, self.non_retryable_exceptions):
            return False
        
        # Check retryable exceptions
        if isinstance(exception, self.retryable_exceptions):
            return True
        
        # For Gaudi3ScaleError, check if it's a transient error
        if isinstance(exception, Gaudi3ScaleError):
            return self._is_transient_gaudi_error(exception)
        
        # Default: don't retry unknown exceptions
        return False
    
    def _is_transient_gaudi_error(self, error: Gaudi3ScaleError) -> bool:
        """Check if a Gaudi3ScaleError is transient and should be retried."""
        transient_error_codes = {
            # Network and infrastructure errors
            'NETWORK_CONNECTION_FAILED',
            'SERVICE_UNAVAILABLE',
            'TIMEOUT_ERROR',
            'CLUSTER_DEPLOYMENT_FAILED',  # Might be transient
            
            # Resource errors (might be temporary)
            'RESOURCE_ALLOCATION_FAILED',
            
            # HPU errors (some might be transient)
            'HPU_COMMUNICATION_ERROR',
            'HPU_DEVICE_ERROR',  # Depends on the specific error
            
            # Training errors (some might be transient)
            'TRAINING_STEP_FAILED',  # Might be due to transient issues
            'DATA_LOADING_ERROR',   # Could be network-related
            
            # Monitoring errors
            'METRICS_COLLECTION_FAILED',
        }
        
        return error.error_code.name in transient_error_codes


class RetryResult:
    """Result of a retry operation."""
    
    def __init__(
        self,
        success: bool,
        result: Any = None,
        exception: Optional[Exception] = None,
        attempts: int = 0,
        total_time: float = 0.0
    ):
        """Initialize retry result.
        
        Args:
            success: Whether the operation eventually succeeded
            result: The result if successful
            exception: The final exception if failed
            attempts: Total number of attempts made
            total_time: Total time spent on all attempts
        """
        self.success = success
        self.result = result
        self.exception = exception
        self.attempts = attempts
        self.total_time = total_time


class RetryExecutor:
    """Executor for retry operations."""
    
    def __init__(self, config: RetryConfig):
        """Initialize retry executor.
        
        Args:
            config: Retry configuration
        """
        self.config = config
    
    def execute(self, func: Callable, *args, **kwargs) -> RetryResult:
        """Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function positional arguments
            **kwargs: Function keyword arguments
            
        Returns:
            RetryResult containing the outcome
        """
        start_time = time.time()
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                # Check timeout
                if self.config.timeout and (time.time() - start_time) > self.config.timeout:
                    logger.warning(f"Retry timeout exceeded after {attempt} attempts")
                    break
                
                # Execute the function
                result = func(*args, **kwargs)
                
                # Success!
                total_time = time.time() - start_time
                logger.info(
                    f"Operation succeeded after {attempt + 1} attempt(s)",
                    extra={
                        'function': func.__name__ if hasattr(func, '__name__') else str(func),
                        'attempts': attempt + 1,
                        'total_time': total_time,
                        'retry_success': True
                    }
                )
                
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempt + 1,
                    total_time=total_time
                )
                
            except Exception as e:
                last_exception = e
                
                # Check if we should retry this exception
                if not self.config.should_retry(e):
                    logger.info(
                        f"Exception not retryable: {type(e).__name__}",
                        extra={'exception_type': type(e).__name__, 'exception_message': str(e)}
                    )
                    break
                
                # Check if we have more attempts
                if attempt >= self.config.max_attempts - 1:
                    logger.warning(
                        f"Maximum retry attempts ({self.config.max_attempts}) exceeded",
                        extra={
                            'function': func.__name__ if hasattr(func, '__name__') else str(func),
                            'max_attempts': self.config.max_attempts,
                            'final_exception': str(e)
                        }
                    )
                    break
                
                # Calculate delay for next attempt
                delay = self.config.backoff_strategy.calculate_delay(attempt, self.config.base_delay)
                
                # Call retry callback if provided
                if self.config.on_retry_callback:
                    self.config.on_retry_callback(attempt + 1, e, delay)
                
                logger.warning(
                    f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s",
                    extra={
                        'function': func.__name__ if hasattr(func, '__name__') else str(func),
                        'attempt': attempt + 1,
                        'exception_type': type(e).__name__,
                        'exception_message': str(e),
                        'retry_delay': delay
                    }
                )
                
                # Wait before next attempt
                time.sleep(delay)
        
        # All attempts failed
        total_time = time.time() - start_time
        
        # Call final failure callback if provided
        if self.config.on_final_failure_callback and last_exception:
            self.config.on_final_failure_callback(last_exception)
        
        logger.error(
            f"All retry attempts failed",
            extra={
                'function': func.__name__ if hasattr(func, '__name__') else str(func),
                'attempts': self.config.max_attempts,
                'total_time': total_time,
                'final_exception': str(last_exception) if last_exception else 'Unknown'
            }
        )
        
        return RetryResult(
            success=False,
            exception=last_exception,
            attempts=self.config.max_attempts,
            total_time=total_time
        )
    
    async def execute_async(self, func: Callable, *args, **kwargs) -> RetryResult:
        """Execute async function with retry logic.
        
        Args:
            func: Async function to execute
            *args: Function positional arguments
            **kwargs: Function keyword arguments
            
        Returns:
            RetryResult containing the outcome
        """
        start_time = time.time()
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                # Check timeout
                if self.config.timeout and (time.time() - start_time) > self.config.timeout:
                    logger.warning(f"Retry timeout exceeded after {attempt} attempts")
                    break
                
                # Execute the async function
                result = await func(*args, **kwargs)
                
                # Success!
                total_time = time.time() - start_time
                logger.info(
                    f"Async operation succeeded after {attempt + 1} attempt(s)",
                    extra={
                        'function': func.__name__ if hasattr(func, '__name__') else str(func),
                        'attempts': attempt + 1,
                        'total_time': total_time,
                        'retry_success': True
                    }
                )
                
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempt + 1,
                    total_time=total_time
                )
                
            except Exception as e:
                last_exception = e
                
                # Check if we should retry this exception
                if not self.config.should_retry(e):
                    logger.info(
                        f"Exception not retryable: {type(e).__name__}",
                        extra={'exception_type': type(e).__name__, 'exception_message': str(e)}
                    )
                    break
                
                # Check if we have more attempts
                if attempt >= self.config.max_attempts - 1:
                    logger.warning(
                        f"Maximum retry attempts ({self.config.max_attempts}) exceeded",
                        extra={
                            'function': func.__name__ if hasattr(func, '__name__') else str(func),
                            'max_attempts': self.config.max_attempts,
                            'final_exception': str(e)
                        }
                    )
                    break
                
                # Calculate delay for next attempt
                delay = self.config.backoff_strategy.calculate_delay(attempt, self.config.base_delay)
                
                # Call retry callback if provided
                if self.config.on_retry_callback:
                    self.config.on_retry_callback(attempt + 1, e, delay)
                
                logger.warning(
                    f"Async attempt {attempt + 1} failed, retrying in {delay:.2f}s",
                    extra={
                        'function': func.__name__ if hasattr(func, '__name__') else str(func),
                        'attempt': attempt + 1,
                        'exception_type': type(e).__name__,
                        'exception_message': str(e),
                        'retry_delay': delay
                    }
                )
                
                # Wait before next attempt
                await asyncio.sleep(delay)
        
        # All attempts failed
        total_time = time.time() - start_time
        
        # Call final failure callback if provided
        if self.config.on_final_failure_callback and last_exception:
            self.config.on_final_failure_callback(last_exception)
        
        logger.error(
            f"All async retry attempts failed",
            extra={
                'function': func.__name__ if hasattr(func, '__name__') else str(func),
                'attempts': self.config.max_attempts,
                'total_time': total_time,
                'final_exception': str(last_exception) if last_exception else 'Unknown'
            }
        )
        
        return RetryResult(
            success=False,
            exception=last_exception,
            attempts=self.config.max_attempts,
            total_time=total_time
        )


# Convenience functions and decorators

def retry_on_failure(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
    retryable_exceptions: Optional[List[Type[Exception]]] = None,
    **kwargs
) -> Callable:
    """Decorator to add retry logic to a function.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries
        strategy: Retry strategy to use
        retryable_exceptions: List of retryable exception types
        **kwargs: Additional RetryConfig parameters
        
    Returns:
        Decorated function with retry logic
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        strategy=strategy,
        retryable_exceptions=retryable_exceptions,
        **kwargs
    )
    executor = RetryExecutor(config)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = executor.execute(func, *args, **kwargs)
            if result.success:
                return result.result
            else:
                raise result.exception or RuntimeError("Retry failed with unknown error")
        
        return wrapper
    
    return decorator


def async_retry_on_failure(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
    retryable_exceptions: Optional[List[Type[Exception]]] = None,
    **kwargs
) -> Callable:
    """Decorator to add retry logic to an async function.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries
        strategy: Retry strategy to use
        retryable_exceptions: List of retryable exception types
        **kwargs: Additional RetryConfig parameters
        
    Returns:
        Decorated async function with retry logic
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        strategy=strategy,
        retryable_exceptions=retryable_exceptions,
        **kwargs
    )
    executor = RetryExecutor(config)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            result = await executor.execute_async(func, *args, **kwargs)
            if result.success:
                return result.result
            else:
                raise result.exception or RuntimeError("Async retry failed with unknown error")
        
        return wrapper
    
    return decorator


def execute_with_retry(
    func: Callable,
    *args,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
    **kwargs
):
    """Execute a function with retry logic.
    
    Args:
        func: Function to execute
        *args: Function arguments
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries
        strategy: Retry strategy to use
        **kwargs: Additional function keyword arguments and RetryConfig parameters
        
    Returns:
        Function result if successful
        
    Raises:
        Exception: The final exception if all retries failed
    """
    # Separate function kwargs from config kwargs
    func_kwargs = {}
    config_kwargs = {}
    
    config_keys = {
        'retryable_exceptions', 'non_retryable_exceptions', 'timeout',
        'should_retry_predicate', 'on_retry_callback', 'on_final_failure_callback'
    }
    
    for key, value in kwargs.items():
        if key in config_keys:
            config_kwargs[key] = value
        else:
            func_kwargs[key] = value
    
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        strategy=strategy,
        **config_kwargs
    )
    executor = RetryExecutor(config)
    
    result = executor.execute(func, *args, **func_kwargs)
    if result.success:
        return result.result
    else:
        raise result.exception or RuntimeError("Retry failed with unknown error")


# Circuit breaker for failing services

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying to recover
            expected_exception: Exception type that counts as failure
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
    
    def call(self, func: Callable, *args, **kwargs):
        """Call function through circuit breaker.
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise ServiceUnavailableError(
                    f"Circuit breaker is open (failures: {self.failure_count})"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: Type[Exception] = Exception
) -> Callable:
    """Decorator to add circuit breaker pattern to a function.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before trying to recover
        expected_exception: Exception type that counts as failure
        
    Returns:
        Decorated function with circuit breaker
    """
    breaker = CircuitBreaker(failure_threshold, recovery_timeout, expected_exception)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        return wrapper
    
    return decorator


# Predefined retry configurations for common scenarios

def get_hpu_retry_config() -> RetryConfig:
    """Get retry configuration for HPU operations."""
    return RetryConfig(
        max_attempts=5,
        base_delay=2.0,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        retryable_exceptions=[HPUError, NetworkError, TimeoutError],
        timeout=300.0  # 5 minutes total timeout
    )


def get_training_retry_config() -> RetryConfig:
    """Get retry configuration for training operations."""
    return RetryConfig(
        max_attempts=3,
        base_delay=5.0,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        retryable_exceptions=[TrainingError, NetworkError, ResourceError],
        timeout=600.0  # 10 minutes total timeout
    )


def get_network_retry_config() -> RetryConfig:
    """Get retry configuration for network operations."""
    return RetryConfig(
        max_attempts=5,
        base_delay=1.0,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        retryable_exceptions=[NetworkError, ServiceUnavailableError, TimeoutError, ConnectionError],
        timeout=120.0  # 2 minutes total timeout
    )