"""Structured logging utilities for Gaudi 3 Scale.

This module provides comprehensive logging capabilities with structured
logging, security features, and integration with monitoring systems.
"""

import json
import logging
import logging.handlers
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from threading import Lock
import uuid

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

try:
    from pythonjsonlogger import jsonlogger
    JSON_LOGGER_AVAILABLE = True
except ImportError:
    JSON_LOGGER_AVAILABLE = False

from .exceptions import Gaudi3ScaleError, ErrorCode
from .validation import InputSanitizer


class LogLevel:
    """Log level constants."""
    TRACE = 5
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class LogContext:
    """Thread-safe log context manager for maintaining context across operations."""
    
    def __init__(self):
        """Initialize log context."""
        self._context = {}
        self._lock = Lock()
    
    def set(self, key: str, value: Any) -> None:
        """Set a context value.
        
        Args:
            key: Context key
            value: Context value
        """
        with self._lock:
            self._context[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a context value.
        
        Args:
            key: Context key
            default: Default value if key not found
            
        Returns:
            Context value or default
        """
        with self._lock:
            return self._context.get(key, default)
    
    def update(self, context: Dict[str, Any]) -> None:
        """Update context with multiple values.
        
        Args:
            context: Dictionary of context values
        """
        with self._lock:
            self._context.update(context)
    
    def clear(self) -> None:
        """Clear all context."""
        with self._lock:
            self._context.clear()
    
    def copy(self) -> Dict[str, Any]:
        """Get a copy of the current context.
        
        Returns:
            Copy of current context
        """
        with self._lock:
            return self._context.copy()


class SecurityLogFilter(logging.Filter):
    """Filter to sanitize log records for security."""
    
    SENSITIVE_FIELDS = {
        'password', 'passwd', 'token', 'secret', 'key', 'auth',
        'authorization', 'credential', 'private', 'api_key'
    }
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter and sanitize log record.
        
        Args:
            record: Log record to filter
            
        Returns:
            True to allow the record
        """
        # Sanitize the message
        if hasattr(record, 'msg'):
            record.msg = InputSanitizer.sanitize_log_message(str(record.msg))
        
        # Sanitize additional fields
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key.lower() in self.SENSITIVE_FIELDS:
                    record.__dict__[key] = '[REDACTED]'
                elif isinstance(value, str):
                    record.__dict__[key] = InputSanitizer.sanitize_log_message(value)
        
        return True


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def __init__(
        self,
        include_extra_fields: bool = True,
        include_stack_trace: bool = True,
        max_message_length: int = 10000
    ):
        """Initialize structured formatter.
        
        Args:
            include_extra_fields: Whether to include extra fields
            include_stack_trace: Whether to include stack traces for exceptions
            max_message_length: Maximum message length
        """
        super().__init__()
        self.include_extra_fields = include_extra_fields
        self.include_stack_trace = include_stack_trace
        self.max_message_length = max_message_length
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted log message as JSON string
        """
        # Base log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage()[:self.max_message_length],
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add process/thread information
        log_entry.update({
            'process_id': record.process,
            'thread_id': record.thread,
            'thread_name': record.threadName,
        })
        
        # Add exception information if present
        if record.exc_info and self.include_stack_trace:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'stack_trace': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        if self.include_extra_fields:
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'lineno', 'funcName', 'created',
                    'msecs', 'relativeCreated', 'thread', 'threadName',
                    'processName', 'process', 'exc_info', 'exc_text', 'stack_info'
                }:
                    extra_fields[key] = value
            
            if extra_fields:
                log_entry['extra'] = extra_fields
        
        try:
            return json.dumps(log_entry, default=str, ensure_ascii=False)
        except Exception as e:
            # Fallback to simple format if JSON serialization fails
            return f"LOG_FORMAT_ERROR: {str(e)} - {record.getMessage()}"


class PerformanceLogger:
    """Logger for performance metrics and timing."""
    
    def __init__(self, logger: logging.Logger):
        """Initialize performance logger.
        
        Args:
            logger: Base logger to use
        """
        self.logger = logger
        self._timers = {}
    
    def start_timer(self, timer_name: str) -> None:
        """Start a performance timer.
        
        Args:
            timer_name: Name of the timer
        """
        self._timers[timer_name] = time.perf_counter()
    
    def end_timer(self, timer_name: str, log_level: int = LogLevel.INFO) -> float:
        """End a performance timer and log the duration.
        
        Args:
            timer_name: Name of the timer
            log_level: Log level for the timing message
            
        Returns:
            Duration in seconds
        """
        if timer_name not in self._timers:
            self.logger.warning(f"Timer '{timer_name}' was not started")
            return 0.0
        
        start_time = self._timers.pop(timer_name)
        duration = time.perf_counter() - start_time
        
        self.logger.log(
            log_level,
            f"Performance timer '{timer_name}' completed",
            extra={
                'timer_name': timer_name,
                'duration_seconds': duration,
                'duration_ms': duration * 1000,
                'performance_metric': True
            }
        )
        
        return duration
    
    def log_memory_usage(self, component: str, log_level: int = LogLevel.INFO) -> None:
        """Log memory usage information.
        
        Args:
            component: Component name for logging
            log_level: Log level for the message
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            self.logger.log(
                log_level,
                f"Memory usage for {component}",
                extra={
                    'component': component,
                    'memory_rss_mb': memory_info.rss / (1024 * 1024),
                    'memory_vms_mb': memory_info.vms / (1024 * 1024),
                    'memory_percent': process.memory_percent(),
                    'performance_metric': True
                }
            )
        except ImportError:
            self.logger.warning("psutil not available for memory logging")
        except Exception as e:
            self.logger.warning(f"Failed to log memory usage: {e}")


class AuditLogger:
    """Logger for security and operational audit events."""
    
    def __init__(self, logger: logging.Logger):
        """Initialize audit logger.
        
        Args:
            logger: Base logger to use
        """
        self.logger = logger
    
    def log_authentication_event(
        self,
        username: str,
        success: bool,
        source_ip: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log authentication events.
        
        Args:
            username: Username for authentication
            success: Whether authentication was successful
            source_ip: Source IP address
            additional_context: Additional context information
        """
        event = {
            'event_type': 'authentication',
            'username': username,
            'success': success,
            'source_ip': source_ip,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'audit': True
        }
        
        if additional_context:
            event.update(additional_context)
        
        level = LogLevel.INFO if success else LogLevel.WARNING
        self.logger.log(
            level,
            f"Authentication {'succeeded' if success else 'failed'} for user {username}",
            extra=event
        )
    
    def log_resource_access(
        self,
        resource_type: str,
        resource_id: str,
        action: str,
        username: Optional[str] = None,
        success: bool = True,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log resource access events.
        
        Args:
            resource_type: Type of resource accessed
            resource_id: ID of the resource
            action: Action performed
            username: Username performing the action
            success: Whether the action was successful
            additional_context: Additional context information
        """
        event = {
            'event_type': 'resource_access',
            'resource_type': resource_type,
            'resource_id': resource_id,
            'action': action,
            'username': username,
            'success': success,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'audit': True
        }
        
        if additional_context:
            event.update(additional_context)
        
        level = LogLevel.INFO if success else LogLevel.ERROR
        self.logger.log(
            level,
            f"Resource access: {action} {resource_type} {resource_id}",
            extra=event
        )
    
    def log_configuration_change(
        self,
        component: str,
        change_type: str,
        old_value: Optional[Any] = None,
        new_value: Optional[Any] = None,
        username: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log configuration changes.
        
        Args:
            component: Component that was changed
            change_type: Type of change made
            old_value: Previous value
            new_value: New value
            username: Username making the change
            additional_context: Additional context information
        """
        event = {
            'event_type': 'configuration_change',
            'component': component,
            'change_type': change_type,
            'old_value': str(old_value) if old_value is not None else None,
            'new_value': str(new_value) if new_value is not None else None,
            'username': username,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'audit': True
        }
        
        if additional_context:
            event.update(additional_context)
        
        self.logger.info(
            f"Configuration change: {change_type} in {component}",
            extra=event
        )


class Gaudi3Logger:
    """Main logger class for Gaudi 3 Scale with comprehensive logging features."""
    
    def __init__(
        self,
        name: str,
        level: int = LogLevel.INFO,
        log_file: Optional[str] = None,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        backup_count: int = 5,
        use_json_format: bool = True,
        enable_console: bool = True,
        enable_security_filter: bool = True
    ):
        """Initialize Gaudi3Logger.
        
        Args:
            name: Logger name
            level: Logging level
            log_file: Log file path (optional)
            max_file_size: Maximum file size before rotation
            backup_count: Number of backup files to keep
            use_json_format: Whether to use JSON formatting
            enable_console: Whether to enable console logging
            enable_security_filter: Whether to enable security filtering
        """
        self.name = name
        self.context = LogContext()
        
        # Create main logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()  # Remove any existing handlers
        
        # Create formatters
        if use_json_format and JSON_LOGGER_AVAILABLE:
            formatter = jsonlogger.JsonFormatter(
                '%(timestamp)s %(level)s %(name)s %(message)s',
                timestamp=True
            )
        elif use_json_format:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            if enable_security_filter:
                console_handler.addFilter(SecurityLogFilter())
            self.logger.addHandler(console_handler)
        
        # File handler with rotation
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            if enable_security_filter:
                file_handler.addFilter(SecurityLogFilter())
            self.logger.addHandler(file_handler)
        
        # Specialized loggers
        self.performance = PerformanceLogger(self.logger)
        self.audit = AuditLogger(self.logger)
        
        # Add trace level
        logging.addLevelName(LogLevel.TRACE, 'TRACE')
    
    def trace(self, msg: str, *args, **kwargs) -> None:
        """Log trace message."""
        self._log_with_context(LogLevel.TRACE, msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log debug message."""
        self._log_with_context(LogLevel.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs) -> None:
        """Log info message."""
        self._log_with_context(LogLevel.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log warning message."""
        self._log_with_context(LogLevel.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs) -> None:
        """Log error message."""
        self._log_with_context(LogLevel.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs) -> None:
        """Log critical message."""
        self._log_with_context(LogLevel.CRITICAL, msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs) -> None:
        """Log exception with stack trace."""
        kwargs['exc_info'] = True
        self._log_with_context(LogLevel.ERROR, msg, *args, **kwargs)
    
    def log_exception(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """Log an exception with structured information.
        
        Args:
            exception: Exception to log
            context: Additional context information
        """
        extra = {
            'exception_type': type(exception).__name__,
            'exception_message': str(exception),
            'exception_context': context or {}
        }
        
        # Add Gaudi3ScaleError specific information
        if isinstance(exception, Gaudi3ScaleError):
            extra.update({
                'error_code': exception.error_code.name,
                'error_code_value': exception.error_code.value,
                'error_context': exception.context,
                'recovery_suggestions': exception.recovery_suggestions
            })
        
        self.error(
            f"Exception occurred: {str(exception)}",
            extra=extra,
            exc_info=True
        )
    
    def log_operation_start(
        self,
        operation: str,
        operation_id: Optional[str] = None,
        **context
    ) -> str:
        """Log the start of an operation.
        
        Args:
            operation: Operation name
            operation_id: Optional operation ID (will generate if not provided)
            **context: Additional context information
            
        Returns:
            Operation ID for tracking
        """
        if operation_id is None:
            operation_id = str(uuid.uuid4())
        
        extra = {
            'operation': operation,
            'operation_id': operation_id,
            'operation_phase': 'start',
            **context
        }
        
        self.info(f"Starting operation: {operation}", extra=extra)
        return operation_id
    
    def log_operation_end(
        self,
        operation: str,
        operation_id: str,
        success: bool = True,
        duration: Optional[float] = None,
        **context
    ) -> None:
        """Log the end of an operation.
        
        Args:
            operation: Operation name
            operation_id: Operation ID from start
            success: Whether the operation was successful
            duration: Operation duration in seconds
            **context: Additional context information
        """
        extra = {
            'operation': operation,
            'operation_id': operation_id,
            'operation_phase': 'end',
            'success': success,
            **context
        }
        
        if duration is not None:
            extra['duration_seconds'] = duration
        
        level = LogLevel.INFO if success else LogLevel.ERROR
        status = 'completed' if success else 'failed'
        self.log(level, f"Operation {status}: {operation}", extra=extra)
    
    def _log_with_context(self, level: int, msg: str, *args, **kwargs) -> None:
        """Log message with current context.
        
        Args:
            level: Log level
            msg: Log message
            *args: Message arguments
            **kwargs: Additional keyword arguments
        """
        # Merge context into extra
        extra = kwargs.get('extra', {})
        context = self.context.copy()
        context.update(extra)
        kwargs['extra'] = context
        
        self.logger.log(level, msg, *args, **kwargs)
    
    def log(self, level: int, msg: str, *args, **kwargs) -> None:
        """Log message at specified level."""
        self._log_with_context(level, msg, *args, **kwargs)
    
    def set_level(self, level: int) -> None:
        """Set logging level."""
        self.logger.setLevel(level)
    
    def add_context(self, **context) -> None:
        """Add context to all log messages."""
        self.context.update(context)
    
    def clear_context(self) -> None:
        """Clear all context."""
        self.context.clear()
    
    def context_manager(self, **context):
        """Context manager for temporary context."""
        return LogContextManager(self, context)


class LogContextManager:
    """Context manager for temporary logging context."""
    
    def __init__(self, logger: Gaudi3Logger, context: Dict[str, Any]):
        """Initialize context manager.
        
        Args:
            logger: Logger instance
            context: Context to apply temporarily
        """
        self.logger = logger
        self.context = context
        self.original_context = None
    
    def __enter__(self):
        """Enter context manager."""
        self.original_context = self.logger.context.copy()
        self.logger.add_context(**self.context)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.logger.context.clear()
        if self.original_context:
            self.logger.context.update(self.original_context)


class LoggerFactory:
    """Factory for creating configured loggers."""
    
    _loggers = {}
    _default_config = {
        'level': LogLevel.INFO,
        'use_json_format': True,
        'enable_console': True,
        'enable_security_filter': True,
        'max_file_size': 100 * 1024 * 1024,
        'backup_count': 5
    }
    
    @classmethod
    def get_logger(
        self,
        name: str,
        log_file: Optional[str] = None,
        **kwargs
    ) -> Gaudi3Logger:
        """Get or create a logger with the specified configuration.
        
        Args:
            name: Logger name
            log_file: Log file path
            **kwargs: Additional logger configuration
            
        Returns:
            Configured Gaudi3Logger instance
        """
        # Create a unique key for this logger configuration
        config = {**self._default_config, **kwargs}
        logger_key = f"{name}:{log_file}:{hash(tuple(sorted(config.items())))}"
        
        if logger_key not in self._loggers:
            self._loggers[logger_key] = Gaudi3Logger(
                name=name,
                log_file=log_file,
                **config
            )
        
        return self._loggers[logger_key]
    
    @classmethod
    def configure_defaults(cls, **config) -> None:
        """Configure default logger settings.
        
        Args:
            **config: Default configuration values
        """
        cls._default_config.update(config)
    
    @classmethod
    def get_component_logger(cls, component: str) -> Gaudi3Logger:
        """Get a logger for a specific component.
        
        Args:
            component: Component name
            
        Returns:
            Configured logger for the component
        """
        log_dir = os.getenv('GAUDI3_LOG_DIR', './logs')
        log_file = os.path.join(log_dir, f'{component}.log')
        
        return cls.get_logger(
            name=f'gaudi3_scale.{component}',
            log_file=log_file
        )


# Create default component loggers
def get_logger(component: str = 'main') -> Gaudi3Logger:
    """Get a logger for a component.
    
    Args:
        component: Component name
        
    Returns:
        Configured logger
    """
    return LoggerFactory.get_component_logger(component)


# Decorators for automatic logging

def log_function_call(
    logger: Optional[Gaudi3Logger] = None,
    level: int = LogLevel.DEBUG,
    include_args: bool = False,
    include_result: bool = False
) -> Callable:
    """Decorator to automatically log function calls.
    
    Args:
        logger: Logger to use (will create if None)
        level: Log level
        include_args: Whether to include function arguments
        include_result: Whether to include return value
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = get_logger('function_calls')
        
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"
            
            extra = {
                'function_name': func_name,
                'function_call': True
            }
            
            if include_args:
                extra['arguments'] = {
                    'args': [str(arg) for arg in args],
                    'kwargs': {k: str(v) for k, v in kwargs.items()}
                }
            
            logger.log(level, f"Calling function {func_name}", extra=extra)
            
            try:
                result = func(*args, **kwargs)
                
                if include_result:
                    extra['result'] = str(result)
                
                logger.log(level, f"Function {func_name} completed successfully", extra=extra)
                return result
                
            except Exception as e:
                logger.log_exception(e, context={'function_name': func_name})
                raise
        
        return wrapper
    return decorator


def log_performance(logger: Optional[Gaudi3Logger] = None) -> Callable:
    """Decorator to automatically log function performance.
    
    Args:
        logger: Logger to use (will create if None)
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = get_logger('performance')
        
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"
            timer_name = f"function_{func_name}"
            
            logger.performance.start_timer(timer_name)
            try:
                return func(*args, **kwargs)
            finally:
                logger.performance.end_timer(timer_name)
        
        return wrapper
    return decorator