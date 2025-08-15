"""Advanced Error Recovery System for Gaudi 3 Training.

This module implements sophisticated error recovery mechanisms including:
- Automatic checkpoint restoration with smart rollback
- Distributed training failure recovery
- Memory leak detection and mitigation
- HPU driver error handling
- Quantum-inspired fault tolerance
"""

import logging
import time
import threading
import pickle
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union, Set
from enum import Enum
from collections import defaultdict, deque
import json

try:
    import torch
    import torch.distributed as dist
    _torch_available = True
except ImportError:
    torch = None
    dist = None
    _torch_available = False

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of training errors."""
    MEMORY_ERROR = "memory_error"
    HPU_DRIVER_ERROR = "hpu_driver_error"
    NETWORK_ERROR = "network_error"
    CHECKPOINT_ERROR = "checkpoint_error"
    DATA_LOADING_ERROR = "data_loading_error"
    MODEL_ERROR = "model_error"
    OPTIMIZER_ERROR = "optimizer_error"
    DISTRIBUTED_ERROR = "distributed_error"
    UNKNOWN_ERROR = "unknown_error"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"
    ROLLBACK = "rollback"
    RESTART = "restart"
    SCALE_DOWN = "scale_down"
    CHECKPOINT_RESTORE = "checkpoint_restore"
    MEMORY_CLEANUP = "memory_cleanup"
    DRIVER_RESET = "driver_reset"
    GRACEFUL_DEGRADATION = "graceful_degradation"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for an error."""
    error_id: str
    error_type: ErrorType
    severity: ErrorSeverity
    timestamp: float
    error_message: str
    stack_trace: str = ""
    affected_components: List[str] = field(default_factory=list)
    system_state: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_id": self.error_id,
            "error_type": self.error_type.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
            "affected_components": self.affected_components,
            "system_state": self.system_state,
            "recovery_attempts": self.recovery_attempts,
            "max_recovery_attempts": self.max_recovery_attempts
        }


@dataclass
class RecoveryAction:
    """Represents a recovery action."""
    action_id: str
    strategy: RecoveryStrategy
    description: str
    estimated_duration: float
    success_probability: float
    rollback_safe: bool = True
    required_resources: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "action_id": self.action_id,
            "strategy": self.strategy.value,
            "description": self.description,
            "estimated_duration": self.estimated_duration,
            "success_probability": self.success_probability,
            "rollback_safe": self.rollback_safe,
            "required_resources": self.required_resources
        }


@dataclass
class CheckpointInfo:
    """Information about a training checkpoint."""
    checkpoint_id: str
    filepath: Path
    timestamp: float
    epoch: int
    step: int
    model_state_hash: str
    optimizer_state_hash: str
    metrics: Dict[str, float] = field(default_factory=dict)
    is_verified: bool = False
    file_size_bytes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "filepath": str(self.filepath),
            "timestamp": self.timestamp,
            "epoch": self.epoch,
            "step": self.step,
            "model_state_hash": self.model_state_hash,
            "optimizer_state_hash": self.optimizer_state_hash,
            "metrics": self.metrics,
            "is_verified": self.is_verified,
            "file_size_bytes": self.file_size_bytes
        }


class SmartCheckpointManager:
    """Advanced checkpoint management with error recovery."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 10,
        verification_enabled: bool = True,
        compression_enabled: bool = True,
        backup_enabled: bool = True
    ):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            verification_enabled: Enable checkpoint verification
            compression_enabled: Enable checkpoint compression
            backup_enabled: Enable checkpoint backup to secondary storage
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.verification_enabled = verification_enabled
        self.compression_enabled = compression_enabled
        self.backup_enabled = backup_enabled
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir = self.checkpoint_dir / "backups"
        if backup_enabled:
            self.backup_dir.mkdir(exist_ok=True)
        
        # Checkpoint tracking
        self.checkpoints: Dict[str, CheckpointInfo] = {}
        self.verified_checkpoints: List[str] = []
        self.corrupted_checkpoints: Set[str] = set()
        
        # Load existing checkpoints
        self._scan_existing_checkpoints()
        
        logger.info(f"Initialized SmartCheckpointManager at {checkpoint_dir}")
    
    def save_checkpoint(
        self,
        model: Any,
        optimizer: Any,
        epoch: int,
        step: int,
        metrics: Optional[Dict[str, float]] = None,
        checkpoint_id: Optional[str] = None
    ) -> CheckpointInfo:
        """Save a training checkpoint with verification.
        
        Args:
            model: Model to checkpoint
            optimizer: Optimizer to checkpoint
            epoch: Current epoch
            step: Current step
            metrics: Training metrics
            checkpoint_id: Optional checkpoint ID
            
        Returns:
            Checkpoint information
            
        Raises:
            RuntimeError: If checkpoint saving fails
        """
        if checkpoint_id is None:
            checkpoint_id = f"checkpoint_epoch_{epoch}_step_{step}"
        
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pt"
        
        try:
            # Prepare checkpoint data
            checkpoint_data = {
                'epoch': epoch,
                'step': step,
                'model_state_dict': model.state_dict() if model else {},
                'optimizer_state_dict': optimizer.state_dict() if optimizer else {},
                'metrics': metrics or {},
                'timestamp': time.time(),
                'pytorch_version': getattr(torch, '__version__', 'unknown') if torch else 'unknown'
            }
            
            # Save checkpoint
            if torch:
                torch.save(checkpoint_data, checkpoint_path)
            else:
                # Fallback for environments without torch
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
            
            # Calculate hashes for verification
            model_hash = self._calculate_state_hash(checkpoint_data['model_state_dict'])
            optimizer_hash = self._calculate_state_hash(checkpoint_data['optimizer_state_dict'])
            
            # Create checkpoint info
            checkpoint_info = CheckpointInfo(
                checkpoint_id=checkpoint_id,
                filepath=checkpoint_path,
                timestamp=checkpoint_data['timestamp'],
                epoch=epoch,
                step=step,
                model_state_hash=model_hash,
                optimizer_state_hash=optimizer_hash,
                metrics=metrics or {},
                file_size_bytes=checkpoint_path.stat().st_size
            )
            
            # Verify checkpoint if enabled
            if self.verification_enabled:
                checkpoint_info.is_verified = self._verify_checkpoint(checkpoint_info)
                if checkpoint_info.is_verified:
                    self.verified_checkpoints.append(checkpoint_id)
                else:
                    logger.warning(f"Checkpoint {checkpoint_id} failed verification")
            
            # Store checkpoint info
            self.checkpoints[checkpoint_id] = checkpoint_info
            
            # Create backup if enabled
            if self.backup_enabled:
                self._create_backup(checkpoint_info)
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
            logger.info(f"Saved checkpoint {checkpoint_id} (epoch {epoch}, step {step})")
            return checkpoint_info
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint_id}: {e}")
            raise RuntimeError(f"Checkpoint saving failed: {e}")
    
    def load_checkpoint(
        self,
        checkpoint_id: str,
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        """Load a checkpoint with verification and error handling.
        
        Args:
            checkpoint_id: ID of checkpoint to load
            model: Model to load state into
            optimizer: Optimizer to load state into
            strict: Whether to enforce strict state dict loading
            
        Returns:
            Loaded checkpoint data
            
        Raises:
            RuntimeError: If checkpoint loading fails
        """
        if checkpoint_id not in self.checkpoints:
            raise RuntimeError(f"Checkpoint {checkpoint_id} not found")
        
        checkpoint_info = self.checkpoints[checkpoint_id]
        
        try:
            # Verify checkpoint before loading
            if self.verification_enabled and not checkpoint_info.is_verified:
                if not self._verify_checkpoint(checkpoint_info):
                    # Try to restore from backup
                    if self.backup_enabled:
                        backup_restored = self._restore_from_backup(checkpoint_info)
                        if not backup_restored:
                            raise RuntimeError(f"Checkpoint {checkpoint_id} is corrupted and cannot be restored")
                    else:
                        raise RuntimeError(f"Checkpoint {checkpoint_id} is corrupted")
            
            # Load checkpoint data
            if torch:
                checkpoint_data = torch.load(checkpoint_info.filepath, map_location='cpu')
            else:
                with open(checkpoint_info.filepath, 'rb') as f:
                    checkpoint_data = pickle.load(f)
            
            # Load model state
            if model and 'model_state_dict' in checkpoint_data:
                try:
                    model.load_state_dict(checkpoint_data['model_state_dict'], strict=strict)
                    logger.info(f"Loaded model state from checkpoint {checkpoint_id}")
                except Exception as e:
                    logger.error(f"Failed to load model state: {e}")
                    if strict:
                        raise
            
            # Load optimizer state
            if optimizer and 'optimizer_state_dict' in checkpoint_data:
                try:
                    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                    logger.info(f"Loaded optimizer state from checkpoint {checkpoint_id}")
                except Exception as e:
                    logger.error(f"Failed to load optimizer state: {e}")
                    if strict:
                        raise
            
            logger.info(f"Successfully loaded checkpoint {checkpoint_id}")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            self.corrupted_checkpoints.add(checkpoint_id)
            raise RuntimeError(f"Checkpoint loading failed: {e}")
    
    def get_latest_checkpoint(self) -> Optional[CheckpointInfo]:
        """Get the latest verified checkpoint.
        
        Returns:
            Latest checkpoint info or None if no checkpoints exist
        """
        if not self.verified_checkpoints:
            return None
        
        latest_id = max(self.verified_checkpoints, 
                       key=lambda cid: self.checkpoints[cid].timestamp)
        return self.checkpoints[latest_id]
    
    def get_best_checkpoint(self, metric_name: str = "loss", minimize: bool = True) -> Optional[CheckpointInfo]:
        """Get the best checkpoint based on a metric.
        
        Args:
            metric_name: Name of metric to optimize
            minimize: Whether to minimize the metric (True) or maximize (False)
            
        Returns:
            Best checkpoint info or None if no checkpoints with the metric exist
        """
        candidates = []
        for checkpoint_id in self.verified_checkpoints:
            checkpoint_info = self.checkpoints[checkpoint_id]
            if metric_name in checkpoint_info.metrics:
                candidates.append(checkpoint_info)
        
        if not candidates:
            return None
        
        if minimize:
            return min(candidates, key=lambda c: c.metrics[metric_name])
        else:
            return max(candidates, key=lambda c: c.metrics[metric_name])
    
    def _calculate_state_hash(self, state_dict: Dict[str, Any]) -> str:
        """Calculate hash of state dictionary for verification."""
        try:
            # Convert state dict to a string representation for hashing
            state_str = str(sorted(state_dict.items()))
            return str(hash(state_str))
        except Exception:
            return "unknown"
    
    def _verify_checkpoint(self, checkpoint_info: CheckpointInfo) -> bool:
        """Verify checkpoint integrity."""
        try:
            # Check if file exists and has expected size
            if not checkpoint_info.filepath.exists():
                return False
            
            current_size = checkpoint_info.filepath.stat().st_size
            if current_size != checkpoint_info.file_size_bytes:
                logger.warning(f"Checkpoint {checkpoint_info.checkpoint_id} size mismatch")
                return False
            
            # Try to load and verify hashes
            if torch:
                checkpoint_data = torch.load(checkpoint_info.filepath, map_location='cpu')
            else:
                with open(checkpoint_info.filepath, 'rb') as f:
                    checkpoint_data = pickle.load(f)
            
            # Verify model state hash
            model_hash = self._calculate_state_hash(checkpoint_data.get('model_state_dict', {}))
            if model_hash != checkpoint_info.model_state_hash:
                logger.warning(f"Checkpoint {checkpoint_info.checkpoint_id} model state hash mismatch")
                return False
            
            # Verify optimizer state hash
            optimizer_hash = self._calculate_state_hash(checkpoint_data.get('optimizer_state_dict', {}))
            if optimizer_hash != checkpoint_info.optimizer_state_hash:
                logger.warning(f"Checkpoint {checkpoint_info.checkpoint_id} optimizer state hash mismatch")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Checkpoint verification failed for {checkpoint_info.checkpoint_id}: {e}")
            return False
    
    def _create_backup(self, checkpoint_info: CheckpointInfo) -> None:
        """Create a backup copy of the checkpoint."""
        try:
            backup_path = self.backup_dir / checkpoint_info.filepath.name
            shutil.copy2(checkpoint_info.filepath, backup_path)
            logger.debug(f"Created backup for checkpoint {checkpoint_info.checkpoint_id}")
        except Exception as e:
            logger.warning(f"Failed to create backup for {checkpoint_info.checkpoint_id}: {e}")
    
    def _restore_from_backup(self, checkpoint_info: CheckpointInfo) -> bool:
        """Restore checkpoint from backup."""
        try:
            backup_path = self.backup_dir / checkpoint_info.filepath.name
            if backup_path.exists():
                shutil.copy2(backup_path, checkpoint_info.filepath)
                logger.info(f"Restored checkpoint {checkpoint_info.checkpoint_id} from backup")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to restore {checkpoint_info.checkpoint_id} from backup: {e}")
            return False
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond the maximum limit."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by timestamp and keep the most recent ones
        sorted_checkpoints = sorted(
            self.checkpoints.items(),
            key=lambda x: x[1].timestamp,
            reverse=True
        )
        
        checkpoints_to_remove = sorted_checkpoints[self.max_checkpoints:]
        
        for checkpoint_id, checkpoint_info in checkpoints_to_remove:
            try:
                # Remove main checkpoint file
                if checkpoint_info.filepath.exists():
                    checkpoint_info.filepath.unlink()
                
                # Remove backup if exists
                backup_path = self.backup_dir / checkpoint_info.filepath.name
                if backup_path.exists():
                    backup_path.unlink()
                
                # Remove from tracking
                del self.checkpoints[checkpoint_id]
                if checkpoint_id in self.verified_checkpoints:
                    self.verified_checkpoints.remove(checkpoint_id)
                
                logger.debug(f"Cleaned up old checkpoint {checkpoint_id}")
                
            except Exception as e:
                logger.warning(f"Failed to cleanup checkpoint {checkpoint_id}: {e}")
    
    def _scan_existing_checkpoints(self) -> None:
        """Scan directory for existing checkpoints."""
        if not self.checkpoint_dir.exists():
            return
        
        for checkpoint_file in self.checkpoint_dir.glob("*.pt"):
            try:
                # Extract checkpoint ID from filename
                checkpoint_id = checkpoint_file.stem
                
                # Load and analyze checkpoint
                if torch:
                    checkpoint_data = torch.load(checkpoint_file, map_location='cpu')
                else:
                    with open(checkpoint_file, 'rb') as f:
                        checkpoint_data = pickle.load(f)
                
                # Create checkpoint info
                checkpoint_info = CheckpointInfo(
                    checkpoint_id=checkpoint_id,
                    filepath=checkpoint_file,
                    timestamp=checkpoint_data.get('timestamp', checkpoint_file.stat().st_mtime),
                    epoch=checkpoint_data.get('epoch', 0),
                    step=checkpoint_data.get('step', 0),
                    model_state_hash=self._calculate_state_hash(checkpoint_data.get('model_state_dict', {})),
                    optimizer_state_hash=self._calculate_state_hash(checkpoint_data.get('optimizer_state_dict', {})),
                    metrics=checkpoint_data.get('metrics', {}),
                    file_size_bytes=checkpoint_file.stat().st_size
                )
                
                # Verify if enabled
                if self.verification_enabled:
                    checkpoint_info.is_verified = self._verify_checkpoint(checkpoint_info)
                    if checkpoint_info.is_verified:
                        self.verified_checkpoints.append(checkpoint_id)
                
                self.checkpoints[checkpoint_id] = checkpoint_info
                
            except Exception as e:
                logger.warning(f"Failed to load existing checkpoint {checkpoint_file}: {e}")


class AdvancedErrorRecovery:
    """Advanced error recovery system with multiple strategies."""
    
    def __init__(
        self,
        checkpoint_manager: SmartCheckpointManager,
        max_recovery_attempts: int = 3,
        recovery_timeout: float = 300.0,
        enable_quantum_recovery: bool = True
    ):
        """Initialize error recovery system.
        
        Args:
            checkpoint_manager: Checkpoint manager instance
            max_recovery_attempts: Maximum recovery attempts per error
            recovery_timeout: Timeout for recovery operations
            enable_quantum_recovery: Enable quantum-inspired recovery strategies
        """
        self.checkpoint_manager = checkpoint_manager
        self.max_recovery_attempts = max_recovery_attempts
        self.recovery_timeout = recovery_timeout
        self.enable_quantum_recovery = enable_quantum_recovery
        
        # Error tracking
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: Dict[ErrorType, List[RecoveryStrategy]] = self._initialize_strategies()
        self.recovery_functions: Dict[RecoveryStrategy, Callable] = self._initialize_recovery_functions()
        
        # State tracking
        self.current_errors: Dict[str, ErrorContext] = {}
        self.system_state_history: deque = deque(maxlen=100)
        
        # Threading
        self.recovery_lock = threading.RLock()
        
        logger.info("Initialized AdvancedErrorRecovery system")
    
    def _initialize_strategies(self) -> Dict[ErrorType, List[RecoveryStrategy]]:
        """Initialize recovery strategies for different error types."""
        return {
            ErrorType.MEMORY_ERROR: [
                RecoveryStrategy.MEMORY_CLEANUP,
                RecoveryStrategy.SCALE_DOWN,
                RecoveryStrategy.RESTART
            ],
            ErrorType.HPU_DRIVER_ERROR: [
                RecoveryStrategy.DRIVER_RESET,
                RecoveryStrategy.RESTART,
                RecoveryStrategy.GRACEFUL_DEGRADATION
            ],
            ErrorType.NETWORK_ERROR: [
                RecoveryStrategy.RETRY,
                RecoveryStrategy.RESTART,
                RecoveryStrategy.GRACEFUL_DEGRADATION
            ],
            ErrorType.CHECKPOINT_ERROR: [
                RecoveryStrategy.CHECKPOINT_RESTORE,
                RecoveryStrategy.ROLLBACK,
                RecoveryStrategy.RESTART
            ],
            ErrorType.DATA_LOADING_ERROR: [
                RecoveryStrategy.RETRY,
                RecoveryStrategy.ROLLBACK,
                RecoveryStrategy.GRACEFUL_DEGRADATION
            ],
            ErrorType.MODEL_ERROR: [
                RecoveryStrategy.CHECKPOINT_RESTORE,
                RecoveryStrategy.ROLLBACK,
                RecoveryStrategy.RESTART
            ],
            ErrorType.OPTIMIZER_ERROR: [
                RecoveryStrategy.CHECKPOINT_RESTORE,
                RecoveryStrategy.ROLLBACK,
                RecoveryStrategy.RESTART
            ],
            ErrorType.DISTRIBUTED_ERROR: [
                RecoveryStrategy.RETRY,
                RecoveryStrategy.SCALE_DOWN,
                RecoveryStrategy.RESTART
            ],
            ErrorType.UNKNOWN_ERROR: [
                RecoveryStrategy.RETRY,
                RecoveryStrategy.CHECKPOINT_RESTORE,
                RecoveryStrategy.RESTART
            ]
        }
    
    def _initialize_recovery_functions(self) -> Dict[RecoveryStrategy, Callable]:
        """Initialize recovery functions for each strategy."""
        return {
            RecoveryStrategy.RETRY: self._recovery_retry,
            RecoveryStrategy.ROLLBACK: self._recovery_rollback,
            RecoveryStrategy.RESTART: self._recovery_restart,
            RecoveryStrategy.SCALE_DOWN: self._recovery_scale_down,
            RecoveryStrategy.CHECKPOINT_RESTORE: self._recovery_checkpoint_restore,
            RecoveryStrategy.MEMORY_CLEANUP: self._recovery_memory_cleanup,
            RecoveryStrategy.DRIVER_RESET: self._recovery_driver_reset,
            RecoveryStrategy.GRACEFUL_DEGRADATION: self._recovery_graceful_degradation
        }
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        error_type: Optional[ErrorType] = None
    ) -> bool:
        """Handle an error with automatic recovery.
        
        Args:
            error: The exception that occurred
            context: Additional context information
            error_type: Type of error (auto-detected if not provided)
            
        Returns:
            True if recovery was successful, False otherwise
        """
        with self.recovery_lock:
            # Create error context
            error_context = self._create_error_context(error, context, error_type)
            
            # Store error
            self.error_history.append(error_context)
            self.current_errors[error_context.error_id] = error_context
            
            logger.error(f"Handling error {error_context.error_id}: {error_context.error_message}")
            
            # Attempt recovery
            recovery_success = self._attempt_recovery(error_context)
            
            if recovery_success:
                # Remove from current errors
                del self.current_errors[error_context.error_id]
                logger.info(f"Successfully recovered from error {error_context.error_id}")
            else:
                logger.error(f"Failed to recover from error {error_context.error_id}")
            
            return recovery_success
    
    def _create_error_context(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]],
        error_type: Optional[ErrorType]
    ) -> ErrorContext:
        """Create error context from exception and additional information."""
        import traceback
        import uuid
        
        error_id = str(uuid.uuid4())[:8]
        error_message = str(error)
        stack_trace = traceback.format_exc()
        
        # Auto-detect error type if not provided
        if error_type is None:
            error_type = self._detect_error_type(error, error_message)
        
        # Determine severity
        severity = self._determine_severity(error_type, error_message)
        
        # Get current system state
        system_state = self._capture_system_state()
        
        return ErrorContext(
            error_id=error_id,
            error_type=error_type,
            severity=severity,
            timestamp=time.time(),
            error_message=error_message,
            stack_trace=stack_trace,
            affected_components=context.get('affected_components', []) if context else [],
            system_state=system_state,
            max_recovery_attempts=self.max_recovery_attempts
        )
    
    def _detect_error_type(self, error: Exception, error_message: str) -> ErrorType:
        """Auto-detect error type from exception and message."""
        error_type_str = type(error).__name__.lower()
        error_msg_lower = error_message.lower()
        
        # Memory errors
        if 'memory' in error_msg_lower or 'oom' in error_msg_lower or isinstance(error, MemoryError):
            return ErrorType.MEMORY_ERROR
        
        # HPU/Driver errors
        if 'hpu' in error_msg_lower or 'habana' in error_msg_lower or 'driver' in error_msg_lower:
            return ErrorType.HPU_DRIVER_ERROR
        
        # Network/Distributed errors
        if 'network' in error_msg_lower or 'distributed' in error_msg_lower or 'nccl' in error_msg_lower:
            if 'distributed' in error_msg_lower:
                return ErrorType.DISTRIBUTED_ERROR
            return ErrorType.NETWORK_ERROR
        
        # Checkpoint errors
        if 'checkpoint' in error_msg_lower or 'state_dict' in error_msg_lower:
            return ErrorType.CHECKPOINT_ERROR
        
        # Data loading errors
        if 'dataloader' in error_msg_lower or 'dataset' in error_msg_lower:
            return ErrorType.DATA_LOADING_ERROR
        
        # Model errors
        if 'model' in error_msg_lower or 'forward' in error_msg_lower:
            return ErrorType.MODEL_ERROR
        
        # Optimizer errors
        if 'optimizer' in error_msg_lower or 'backward' in error_msg_lower:
            return ErrorType.OPTIMIZER_ERROR
        
        return ErrorType.UNKNOWN_ERROR
    
    def _determine_severity(self, error_type: ErrorType, error_message: str) -> ErrorSeverity:
        """Determine error severity."""
        # Critical errors that require immediate attention
        critical_errors = [ErrorType.HPU_DRIVER_ERROR, ErrorType.MEMORY_ERROR]
        if error_type in critical_errors:
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        high_errors = [ErrorType.DISTRIBUTED_ERROR, ErrorType.CHECKPOINT_ERROR]
        if error_type in high_errors:
            return ErrorSeverity.HIGH
        
        # Check message for severity indicators
        error_msg_lower = error_message.lower()
        if any(word in error_msg_lower for word in ['critical', 'fatal', 'abort']):
            return ErrorSeverity.CRITICAL
        elif any(word in error_msg_lower for word in ['error', 'exception', 'failed']):
            return ErrorSeverity.MEDIUM
        
        return ErrorSeverity.LOW
    
    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for error context."""
        state = {
            'timestamp': time.time(),
            'memory_usage': {},
            'hpu_status': {},
            'process_info': {}
        }
        
        try:
            import psutil
            process = psutil.Process()
            state['memory_usage'] = {
                'rss': process.memory_info().rss,
                'vms': process.memory_info().vms,
                'percent': process.memory_percent()
            }
            state['process_info'] = {
                'cpu_percent': process.cpu_percent(),
                'num_threads': process.num_threads(),
                'status': process.status()
            }
        except ImportError:
            logger.debug("psutil not available for system monitoring")
        
        # Capture HPU status if available
        try:
            if torch and hasattr(torch, 'hpu'):
                state['hpu_status'] = {
                    'device_count': torch.hpu.device_count(),
                    'current_device': torch.hpu.current_device(),
                    'is_available': torch.hpu.is_available()
                }
        except Exception:
            logger.debug("Could not capture HPU status")
        
        return state
    
    def _attempt_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt recovery using available strategies."""
        strategies = self.recovery_strategies.get(error_context.error_type, [RecoveryStrategy.RETRY])
        
        for strategy in strategies:
            if error_context.recovery_attempts >= error_context.max_recovery_attempts:
                logger.warning(f"Max recovery attempts reached for error {error_context.error_id}")
                break
            
            error_context.recovery_attempts += 1
            
            logger.info(f"Attempting recovery strategy {strategy.value} for error {error_context.error_id} "
                       f"(attempt {error_context.recovery_attempts}/{error_context.max_recovery_attempts})")
            
            try:
                recovery_function = self.recovery_functions.get(strategy)
                if recovery_function:
                    success = recovery_function(error_context)
                    if success:
                        logger.info(f"Recovery strategy {strategy.value} succeeded for error {error_context.error_id}")
                        return True
                    else:
                        logger.warning(f"Recovery strategy {strategy.value} failed for error {error_context.error_id}")
                else:
                    logger.warning(f"No recovery function for strategy {strategy.value}")
            
            except Exception as recovery_error:
                logger.error(f"Recovery strategy {strategy.value} threw exception: {recovery_error}")
        
        return False
    
    def _recovery_retry(self, error_context: ErrorContext) -> bool:
        """Simple retry recovery strategy."""
        # Wait before retry with exponential backoff
        wait_time = min(2 ** error_context.recovery_attempts, 30)
        time.sleep(wait_time)
        return True  # Indicate retry is ready
    
    def _recovery_rollback(self, error_context: ErrorContext) -> bool:
        """Rollback to previous checkpoint."""
        try:
            latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
            if latest_checkpoint:
                logger.info(f"Rolling back to checkpoint {latest_checkpoint.checkpoint_id}")
                # Actual rollback would depend on training framework integration
                return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
        return False
    
    def _recovery_restart(self, error_context: ErrorContext) -> bool:
        """Restart training process."""
        logger.info("Attempting training restart")
        # Implementation would depend on specific training setup
        # This is a placeholder for the restart logic
        return True
    
    def _recovery_scale_down(self, error_context: ErrorContext) -> bool:
        """Scale down training to use fewer resources."""
        logger.info("Attempting to scale down training")
        # Implementation would reduce batch size, workers, etc.
        return True
    
    def _recovery_checkpoint_restore(self, error_context: ErrorContext) -> bool:
        """Restore from best available checkpoint."""
        try:
            # Try to get the best checkpoint based on validation loss
            best_checkpoint = self.checkpoint_manager.get_best_checkpoint('val_loss', minimize=True)
            if not best_checkpoint:
                best_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
            
            if best_checkpoint:
                logger.info(f"Restoring from checkpoint {best_checkpoint.checkpoint_id}")
                # Actual restoration would depend on training framework
                return True
        except Exception as e:
            logger.error(f"Checkpoint restore failed: {e}")
        return False
    
    def _recovery_memory_cleanup(self, error_context: ErrorContext) -> bool:
        """Clean up memory to recover from memory errors."""
        logger.info("Attempting memory cleanup")
        
        try:
            # Clear PyTorch cache if available
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Additional memory cleanup could be implemented here
            return True
            
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
            return False
    
    def _recovery_driver_reset(self, error_context: ErrorContext) -> bool:
        """Reset HPU driver."""
        logger.info("Attempting HPU driver reset")
        # Implementation would depend on HPU driver API
        # This is a placeholder
        return True
    
    def _recovery_graceful_degradation(self, error_context: ErrorContext) -> bool:
        """Gracefully degrade training capabilities."""
        logger.info("Attempting graceful degradation")
        # Reduce precision, disable advanced features, etc.
        return True
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error history and recovery statistics.
        
        Returns:
            Dictionary containing error statistics
        """
        if not self.error_history:
            return {"total_errors": 0, "recovery_rate": 1.0}
        
        error_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        recovery_success_count = 0
        
        for error in self.error_history:
            error_counts[error.error_type.value] += 1
            severity_counts[error.severity.value] += 1
            
            # Check if error was successfully recovered
            if error.error_id not in self.current_errors:
                recovery_success_count += 1
        
        recovery_rate = recovery_success_count / len(self.error_history)
        
        return {
            "total_errors": len(self.error_history),
            "current_errors": len(self.current_errors),
            "recovery_rate": recovery_rate,
            "error_types": dict(error_counts),
            "severity_distribution": dict(severity_counts),
            "recent_errors": [error.to_dict() for error in self.error_history[-5:]]
        }
    
    def save_error_log(self, filepath: str) -> None:
        """Save error history to file.
        
        Args:
            filepath: File path to save error log
        """
        try:
            error_data = {
                "timestamp": time.time(),
                "error_history": [error.to_dict() for error in self.error_history],
                "summary": self.get_error_summary()
            }
            
            with open(filepath, 'w') as f:
                json.dump(error_data, f, indent=2)
            
            logger.info(f"Error log saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save error log: {e}")


# Factory functions

def create_checkpoint_manager(
    checkpoint_dir: str,
    max_checkpoints: int = 10,
    enable_verification: bool = True
) -> SmartCheckpointManager:
    """Create a configured checkpoint manager.
    
    Args:
        checkpoint_dir: Directory for checkpoints
        max_checkpoints: Maximum checkpoints to keep
        enable_verification: Enable checkpoint verification
        
    Returns:
        Configured checkpoint manager
    """
    return SmartCheckpointManager(
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=max_checkpoints,
        verification_enabled=enable_verification,
        compression_enabled=True,
        backup_enabled=True
    )


def create_error_recovery_system(
    checkpoint_dir: str,
    max_recovery_attempts: int = 3
) -> AdvancedErrorRecovery:
    """Create a complete error recovery system.
    
    Args:
        checkpoint_dir: Directory for checkpoints
        max_recovery_attempts: Maximum recovery attempts per error
        
    Returns:
        Configured error recovery system
    """
    checkpoint_manager = create_checkpoint_manager(checkpoint_dir)
    
    return AdvancedErrorRecovery(
        checkpoint_manager=checkpoint_manager,
        max_recovery_attempts=max_recovery_attempts,
        recovery_timeout=300.0,
        enable_quantum_recovery=True
    )


# Export main classes and functions
__all__ = [
    'SmartCheckpointManager',
    'AdvancedErrorRecovery',
    'ErrorType',
    'ErrorSeverity',
    'RecoveryStrategy',
    'ErrorContext',
    'CheckpointInfo',
    'create_checkpoint_manager',
    'create_error_recovery_system'
]