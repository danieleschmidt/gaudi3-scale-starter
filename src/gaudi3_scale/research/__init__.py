"""Research Framework for Gaudi 3 Scale Starter.

This module contains cutting-edge research implementations including:
- Adaptive batch size optimization with quantum-inspired algorithms
- Reinforcement learning-based hyperparameter tuning
- Novel scheduling algorithms for distributed training
- Advanced memory optimization techniques
- Quantum-hybrid computing approaches
- Performance-aware auto-scaling systems
- Advanced error recovery mechanisms
"""

from .adaptive_batch_optimizer import (
    AdaptiveBatchOptimizer,
    OptimizationStrategy,
    BatchMetrics,
    OptimizationResult,
    QuantumInspiredScheduler,
    ReinforcementLearningOptimizer,
    create_adaptive_optimizer
)

from .quantum_hybrid_scheduler import (
    QuantumHybridScheduler,
    Task,
    Node,
    TaskPriority,
    ResourceType,
    TaskState,
    SchedulingDecision,
    QuantumInspiredOptimizer,
    create_gaudi3_node,
    create_training_task
)

from .error_recovery import (
    SmartCheckpointManager,
    AdvancedErrorRecovery,
    ErrorType,
    ErrorSeverity,
    RecoveryStrategy,
    ErrorContext,
    CheckpointInfo,
    create_checkpoint_manager,
    create_error_recovery_system
)

from .performance_auto_scaler import (
    PerformanceAutoScaler,
    ScalingPolicy,
    ScalingDirection,
    WorkloadPattern,
    PerformanceMetrics,
    ScalingEvent,
    WorkloadPredictor,
    CostOptimizer,
    create_default_scaling_policy,
    create_performance_auto_scaler
)

__all__ = [
    # Adaptive Batch Optimization
    'AdaptiveBatchOptimizer',
    'OptimizationStrategy', 
    'BatchMetrics',
    'OptimizationResult',
    'QuantumInspiredScheduler',
    'ReinforcementLearningOptimizer',
    'create_adaptive_optimizer',
    
    # Quantum Hybrid Scheduling
    'QuantumHybridScheduler',
    'Task',
    'Node',
    'TaskPriority',
    'ResourceType',
    'TaskState',
    'SchedulingDecision',
    'QuantumInspiredOptimizer',
    'create_gaudi3_node',
    'create_training_task',
    
    # Error Recovery
    'SmartCheckpointManager',
    'AdvancedErrorRecovery',
    'ErrorType',
    'ErrorSeverity',
    'RecoveryStrategy',
    'ErrorContext',
    'CheckpointInfo',
    'create_checkpoint_manager',
    'create_error_recovery_system',
    
    # Performance Auto-Scaling
    'PerformanceAutoScaler',
    'ScalingPolicy',
    'ScalingDirection',
    'WorkloadPattern',
    'PerformanceMetrics',
    'ScalingEvent',
    'WorkloadPredictor',
    'CostOptimizer',
    'create_default_scaling_policy',
    'create_performance_auto_scaler'
]

__version__ = "2.0.0"
__author__ = "Terragon Labs Research Team"