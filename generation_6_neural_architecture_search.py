#!/usr/bin/env python3
"""
Generation 6: Advanced Neural Architecture Search (NAS)
======================================================

Next-generation Neural Architecture Search system with:
- Multi-objective optimization (performance, efficiency, latency)
- Progressive search with early stopping
- Hardware-aware architecture optimization
- Evolutionary and gradient-based search strategies
- Automated architecture morphing for different deployment targets
- Learned search space refinement
"""

import asyncio
import json
import logging
import time
import numpy as np
import random
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from pathlib import Path
from collections import defaultdict, deque
import pickle
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import ML libraries with fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from scipy.optimize import differential_evolution, minimize
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class ArchitectureSpec:
    """Specification for a neural architecture."""
    architecture_id: str
    layers: List[Dict[str, Any]]
    connections: List[Tuple[int, int]]  # (from_layer, to_layer)
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    estimated_params: int
    estimated_flops: float
    created_timestamp: float


@dataclass
class PerformanceMetrics:
    """Performance metrics for an architecture."""
    accuracy: float = 0.0
    training_time: float = 0.0
    inference_time: float = 0.0
    memory_usage: float = 0.0
    energy_consumption: float = 0.0
    model_size: float = 0.0
    convergence_epochs: int = 0
    stability_score: float = 0.0


@dataclass
class EvaluationResult:
    """Result of architecture evaluation."""
    architecture_id: str
    performance: PerformanceMetrics
    pareto_score: float
    rank: int
    evaluation_cost: float
    timestamp: float
    hardware_profile: str
    notes: str = ""


@dataclass
class SearchState:
    """State of the NAS process."""
    search_id: str
    current_generation: int = 0
    architectures_evaluated: int = 0
    pareto_front_size: int = 0
    best_accuracy: float = 0.0
    search_budget_used: float = 0.0
    early_stopping_patience: int = 10
    stagnation_counter: int = 0
    start_time: float = 0.0
    
    def __post_init__(self):
        if self.start_time == 0.0:
            self.start_time = time.time()


class NeuralArchitectureSearchEngine:
    """
    Generation 6: Advanced Neural Architecture Search Engine
    
    Features:
    - Multi-objective optimization balancing accuracy, efficiency, and latency
    - Progressive search space refinement based on promising regions
    - Hardware-aware optimization for different deployment targets
    - Evolutionary algorithms combined with Bayesian optimization
    - Early stopping and resource-aware search
    - Automated architecture morphing for deployment constraints
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.search_state = SearchState(
            search_id=f"nas_search_{int(time.time())}",
            early_stopping_patience=self.config.get('early_stopping_patience', 10)
        )
        
        # Search components
        self.search_space = self._initialize_search_space()
        self.population = []  # Current population for evolutionary search
        self.archive = []  # All evaluated architectures
        self.pareto_front = []  # Non-dominated solutions
        
        # Search strategies
        self.evolutionary_searcher = EvolutionarySearcher(self)
        self.bayesian_searcher = BayesianSearcher(self)
        self.progressive_searcher = ProgressiveSearcher(self)
        self.hardware_optimizer = HardwareAwareOptimizer(self)
        
        # Performance predictors
        self.performance_predictor = PerformancePredictor(self)
        self.early_stopping_predictor = EarlyStoppingPredictor(self)
        
        # Search space refinement
        self.promising_regions = {}
        self.search_space_history = []
        
        logger.info(f"🔬 Generation 6 Neural Architecture Search Engine initialized")
        logger.info(f"Search ID: {self.search_state.search_id}")
    
    def _initialize_search_space(self) -> Dict[str, Any]:
        """Initialize the neural architecture search space."""
        return {
            'layer_types': [
                'conv2d', 'depthwise_conv2d', 'pointwise_conv2d', 
                'conv1x1', 'dilated_conv', 'separable_conv',
                'linear', 'attention', 'multi_head_attention',
                'batch_norm', 'layer_norm', 'group_norm',
                'relu', 'gelu', 'swish', 'mish', 'elu',
                'max_pool', 'avg_pool', 'adaptive_pool',
                'dropout', 'spatial_dropout', 'stochastic_depth',
                'residual_block', 'dense_block', 'squeeze_excite'
            ],
            'conv_filters': [16, 32, 64, 128, 256, 512, 1024],
            'kernel_sizes': [1, 3, 5, 7, 9],
            'strides': [1, 2],
            'dilations': [1, 2, 4, 8],
            'attention_heads': [1, 2, 4, 8, 16],
            'hidden_sizes': [64, 128, 256, 512, 1024, 2048],
            'activation_functions': ['relu', 'gelu', 'swish', 'mish'],
            'normalization_types': ['batch_norm', 'layer_norm', 'group_norm'],
            'dropout_rates': [0.0, 0.1, 0.2, 0.3, 0.5],
            'depth_range': (3, 50),
            'width_multiplier_range': (0.25, 2.0),
            'skip_connection_probability': 0.3,
            'bottleneck_probability': 0.4
        }
    
    async def search_architectures(self, 
                                 search_budget: int = 100,
                                 hardware_constraints: Optional[Dict[str, Any]] = None,
                                 objectives: List[str] = ['accuracy', 'efficiency']) -> Dict[str, Any]:
        """
        Search for optimal neural architectures.
        
        Args:
            search_budget: Maximum number of architectures to evaluate
            hardware_constraints: Constraints like max_params, max_latency, etc.
            objectives: List of objectives to optimize ['accuracy', 'efficiency', 'latency']
        """
        
        logger.info(f"🚀 Starting NAS with budget: {search_budget}, objectives: {objectives}")
        
        search_results = {
            'search_id': self.search_state.search_id,
            'start_time': time.time(),
            'architectures_found': [],
            'pareto_front': [],
            'search_progress': [],
            'best_architectures': {},
            'search_statistics': {}
        }
        
        # Initialize search strategies
        strategies = ['evolutionary', 'bayesian', 'progressive']
        current_strategy = 0
        strategy_budget = search_budget // len(strategies)
        
        for generation in range(search_budget):
            self.search_state.current_generation = generation
            
            # Switch strategies periodically
            if generation > 0 and generation % strategy_budget == 0:
                current_strategy = (current_strategy + 1) % len(strategies)
                logger.info(f"🔄 Switching to {strategies[current_strategy]} search strategy")
            
            # Generate candidate architecture based on current strategy
            if strategies[current_strategy] == 'evolutionary':
                candidate = await self.evolutionary_searcher.generate_candidate()
            elif strategies[current_strategy] == 'bayesian':
                candidate = await self.bayesian_searcher.generate_candidate()
            else:  # progressive
                candidate = await self.progressive_searcher.generate_candidate()
            
            # Apply hardware constraints if specified
            if hardware_constraints:
                candidate = await self.hardware_optimizer.apply_constraints(
                    candidate, hardware_constraints
                )
            
            # Evaluate candidate architecture
            evaluation = await self._evaluate_architecture(candidate, objectives)
            
            # Update search state
            self.archive.append(evaluation)
            self.search_state.architectures_evaluated += 1
            self.search_state.search_budget_used = generation / search_budget
            
            # Update pareto front
            await self._update_pareto_front(evaluation)
            
            # Record progress
            search_results['search_progress'].append({
                'generation': generation,
                'strategy': strategies[current_strategy],
                'best_accuracy': self.search_state.best_accuracy,
                'pareto_front_size': len(self.pareto_front),
                'architecture_id': evaluation.architecture_id,
                'pareto_score': evaluation.pareto_score
            })
            
            # Update performance predictor
            await self.performance_predictor.update(candidate, evaluation)
            
            # Check for early stopping
            if await self._should_early_stop():
                logger.info(f"🛑 Early stopping triggered at generation {generation}")
                break
            
            # Refine search space periodically
            if generation % 20 == 0 and generation > 0:
                await self._refine_search_space()
            
            # Log progress
            if generation % 10 == 0:
                logger.info(f"📊 Generation {generation}: "
                          f"Best accuracy: {self.search_state.best_accuracy:.3f}, "
                          f"Pareto front size: {len(self.pareto_front)}")
        
        # Finalize search results
        search_results['end_time'] = time.time()
        search_results['total_time'] = search_results['end_time'] - search_results['start_time']
        search_results['architectures_found'] = [asdict(arch) for arch in self.archive]
        search_results['pareto_front'] = [asdict(arch) for arch in self.pareto_front]
        search_results['best_architectures'] = await self._get_best_architectures_by_objective(objectives)
        search_results['search_statistics'] = self._get_search_statistics()
        
        logger.info(f"✅ NAS completed: {len(self.archive)} architectures evaluated, "
                   f"{len(self.pareto_front)} in Pareto front")
        
        return search_results
    
    async def _evaluate_architecture(self, architecture: ArchitectureSpec,
                                   objectives: List[str]) -> EvaluationResult:
        """Evaluate a single architecture across multiple objectives."""
        
        eval_start = time.time()
        
        # Check if we can use early prediction to save computation
        early_prediction = await self.early_stopping_predictor.predict_early_stop(architecture)
        
        if early_prediction['should_skip']:
            # Return estimated poor performance to save computation
            performance = PerformanceMetrics(
                accuracy=early_prediction['estimated_accuracy'],
                training_time=early_prediction['estimated_training_time'],
                inference_time=0.001,  # Very fast inference for poor models
                memory_usage=self._estimate_memory_usage(architecture),
                model_size=self._estimate_model_size(architecture)
            )
        else:
            # Perform full evaluation (simulated)
            performance = await self._simulate_architecture_training(architecture)
        
        # Calculate multi-objective score
        pareto_score = self._calculate_pareto_score(performance, objectives)
        
        # Update best accuracy tracking
        if performance.accuracy > self.search_state.best_accuracy:
            self.search_state.best_accuracy = performance.accuracy
            self.search_state.stagnation_counter = 0
        else:
            self.search_state.stagnation_counter += 1
        
        evaluation_cost = time.time() - eval_start
        
        return EvaluationResult(
            architecture_id=architecture.architecture_id,
            performance=performance,
            pareto_score=pareto_score,
            rank=0,  # Will be updated when added to pareto front
            evaluation_cost=evaluation_cost,
            timestamp=time.time(),
            hardware_profile=self.config.get('hardware_profile', 'gpu_v100'),
            notes=f"Early skip: {early_prediction.get('should_skip', False)}" if early_prediction else ""
        )
    
    async def _simulate_architecture_training(self, architecture: ArchitectureSpec) -> PerformanceMetrics:
        """Simulate training an architecture to get performance metrics."""
        
        # Simulate training performance based on architecture characteristics
        layer_count = len(architecture.layers)
        param_count = architecture.estimated_params
        complexity_score = self._calculate_architecture_complexity(architecture)
        
        # Base performance with noise
        base_accuracy = 0.5 + random.uniform(-0.1, 0.1)
        
        # Architecture-dependent performance modifiers
        depth_bonus = min(0.3, layer_count * 0.01)  # Deeper can be better, with diminishing returns
        
        # Attention mechanism bonus
        attention_layers = sum(1 for layer in architecture.layers 
                             if layer.get('type') in ['attention', 'multi_head_attention'])
        attention_bonus = min(0.15, attention_layers * 0.05)
        
        # Skip connection bonus
        skip_connections = len([conn for conn in architecture.connections 
                              if abs(conn[1] - conn[0]) > 2])  # Non-adjacent connections
        skip_bonus = min(0.1, skip_connections * 0.02)
        
        # Regularization effects
        dropout_layers = sum(1 for layer in architecture.layers 
                           if layer.get('type') in ['dropout', 'spatial_dropout'])
        regularization_bonus = min(0.08, dropout_layers * 0.02)
        
        # Overfitting penalty for too many parameters
        param_penalty = max(0, (param_count - 1000000) / 10000000 * 0.2)  # Penalty for >1M params
        
        # Calculate final accuracy
        final_accuracy = base_accuracy + depth_bonus + attention_bonus + skip_bonus + regularization_bonus - param_penalty
        final_accuracy = max(0.1, min(0.98, final_accuracy + random.uniform(-0.05, 0.05)))
        
        # Training time (increases with complexity)
        base_training_time = 60.0  # 1 minute base
        training_time = base_training_time * (1 + complexity_score) * (1 + param_count / 1000000)
        training_time += random.uniform(-10, 10)  # Add noise
        
        # Inference time (depends on FLOPs)
        inference_time = max(0.001, architecture.estimated_flops / 1000000 + random.uniform(-0.002, 0.002))
        
        # Memory usage (depends on parameters and intermediate activations)
        memory_usage = param_count * 4 / (1024 * 1024)  # 4 bytes per param, convert to MB
        memory_usage += architecture.estimated_flops * 0.001  # Activation memory estimate
        
        # Energy consumption (correlated with training time and model size)
        energy_consumption = training_time * 0.5 + param_count / 1000000  # Arbitrary units
        
        # Model size (parameters * precision)
        model_size = param_count * 4 / (1024 * 1024)  # 4 bytes per param, MB
        
        # Convergence epochs (harder problems converge slower)
        convergence_epochs = max(5, int(50 * (1 - final_accuracy + 0.2)))
        
        # Stability score (how consistent the model performs)
        stability_score = min(1.0, final_accuracy + random.uniform(-0.1, 0.1))
        stability_score = max(0.0, stability_score)
        
        return PerformanceMetrics(
            accuracy=final_accuracy,
            training_time=max(1.0, training_time),
            inference_time=inference_time,
            memory_usage=memory_usage,
            energy_consumption=energy_consumption,
            model_size=model_size,
            convergence_epochs=convergence_epochs,
            stability_score=stability_score
        )
    
    def _calculate_architecture_complexity(self, architecture: ArchitectureSpec) -> float:
        """Calculate a complexity score for the architecture."""
        complexity = 0.0
        
        for layer in architecture.layers:
            layer_type = layer.get('type', 'linear')
            
            if layer_type in ['conv2d', 'depthwise_conv2d', 'separable_conv']:
                kernel_size = layer.get('kernel_size', 3)
                filters = layer.get('filters', 64)
                complexity += (kernel_size ** 2) * filters / 1000
            elif layer_type in ['attention', 'multi_head_attention']:
                heads = layer.get('heads', 8)
                dim = layer.get('dim', 512)
                complexity += heads * dim / 1000
            elif layer_type == 'linear':
                size = layer.get('size', 512)
                complexity += size / 1000
            else:
                complexity += 0.1  # Base complexity for other layers
        
        return complexity
    
    def _estimate_memory_usage(self, architecture: ArchitectureSpec) -> float:
        """Estimate memory usage in MB."""
        param_memory = architecture.estimated_params * 4 / (1024 * 1024)  # 4 bytes per param
        activation_memory = architecture.estimated_flops * 0.001  # Rough estimate
        return param_memory + activation_memory
    
    def _estimate_model_size(self, architecture: ArchitectureSpec) -> float:
        """Estimate model size in MB."""
        return architecture.estimated_params * 4 / (1024 * 1024)  # 4 bytes per param
    
    def _calculate_pareto_score(self, performance: PerformanceMetrics, 
                              objectives: List[str]) -> float:
        """Calculate multi-objective Pareto score."""
        
        objective_values = []
        weights = []
        
        for objective in objectives:
            if objective == 'accuracy':
                objective_values.append(performance.accuracy)
                weights.append(1.0)
            elif objective == 'efficiency':
                # Efficiency = accuracy / (training_time * model_size)
                efficiency = performance.accuracy / max(0.1, performance.training_time * performance.model_size / 100)
                objective_values.append(efficiency)
                weights.append(0.8)
            elif objective == 'latency':
                # Lower latency is better, so we use 1/latency
                latency_score = 1.0 / max(0.001, performance.inference_time)
                objective_values.append(latency_score)
                weights.append(0.6)
            elif objective == 'memory':
                # Lower memory usage is better
                memory_score = 1.0 / max(1.0, performance.memory_usage)
                objective_values.append(memory_score)
                weights.append(0.4)
            elif objective == 'stability':
                objective_values.append(performance.stability_score)
                weights.append(0.5)
        
        # Weighted geometric mean
        if not objective_values:
            return 0.0
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Calculate weighted geometric mean
        log_sum = sum(w * math.log(max(0.001, v)) for v, w in zip(objective_values, normalized_weights))
        pareto_score = math.exp(log_sum)
        
        return pareto_score
    
    async def _update_pareto_front(self, new_evaluation: EvaluationResult):
        """Update the Pareto front with a new evaluation."""
        
        # Check if new solution is dominated by existing solutions
        is_dominated = False
        dominates = []
        
        for i, existing in enumerate(self.pareto_front):
            dominance = self._check_dominance(new_evaluation, existing)
            
            if dominance == -1:  # New solution is dominated
                is_dominated = True
                break
            elif dominance == 1:  # New solution dominates existing
                dominates.append(i)
        
        # If not dominated, add to Pareto front
        if not is_dominated:
            # Remove dominated solutions
            for i in reversed(sorted(dominates)):
                del self.pareto_front[i]
            
            # Add new solution
            self.pareto_front.append(new_evaluation)
            
            # Update ranks
            for i, solution in enumerate(self.pareto_front):
                solution.rank = i + 1
            
            self.search_state.pareto_front_size = len(self.pareto_front)
            
            # Log new Pareto optimal solution
            logger.info(f"🎯 New Pareto optimal solution: {new_evaluation.architecture_id} "
                       f"(score: {new_evaluation.pareto_score:.3f})")
    
    def _check_dominance(self, solution1: EvaluationResult, solution2: EvaluationResult) -> int:
        """
        Check dominance relationship between two solutions.
        
        Returns:
            1 if solution1 dominates solution2
            -1 if solution2 dominates solution1
            0 if neither dominates
        """
        
        perf1 = solution1.performance
        perf2 = solution2.performance
        
        # Define objectives (higher is better for all)
        objectives1 = [
            perf1.accuracy,
            1.0 / max(0.001, perf1.inference_time),  # Lower latency is better
            1.0 / max(1.0, perf1.memory_usage),      # Lower memory is better
            perf1.stability_score
        ]
        
        objectives2 = [
            perf2.accuracy,
            1.0 / max(0.001, perf2.inference_time),
            1.0 / max(1.0, perf2.memory_usage),
            perf2.stability_score
        ]
        
        # Check if solution1 dominates solution2
        dominates_in_all = True
        strictly_better_in_one = False
        
        for obj1, obj2 in zip(objectives1, objectives2):
            if obj1 < obj2:
                dominates_in_all = False
                break
            elif obj1 > obj2:
                strictly_better_in_one = True
        
        if dominates_in_all and strictly_better_in_one:
            return 1
        
        # Check if solution2 dominates solution1
        dominates_in_all = True
        strictly_better_in_one = False
        
        for obj1, obj2 in zip(objectives1, objectives2):
            if obj2 < obj1:
                dominates_in_all = False
                break
            elif obj2 > obj1:
                strictly_better_in_one = True
        
        if dominates_in_all and strictly_better_in_one:
            return -1
        
        return 0  # Neither dominates
    
    async def _should_early_stop(self) -> bool:
        """Check if search should be stopped early."""
        
        # Stop if no improvement for too long
        if self.search_state.stagnation_counter >= self.search_state.early_stopping_patience:
            return True
        
        # Stop if Pareto front is large enough and stable
        if (len(self.pareto_front) >= 10 and 
            self.search_state.architectures_evaluated > 50 and
            self.search_state.stagnation_counter >= 5):
            return True
        
        return False
    
    async def _refine_search_space(self):
        """Refine search space based on promising regions."""
        
        if len(self.pareto_front) < 3:
            return
        
        logger.info("🔧 Refining search space based on promising architectures...")
        
        # Analyze characteristics of Pareto optimal solutions
        promising_characteristics = self._analyze_pareto_characteristics()
        
        # Update search space to focus on promising regions
        refined_space = self.search_space.copy()
        
        # Refine layer types based on successful architectures
        if promising_characteristics['layer_types']:
            top_layer_types = [lt for lt, count in promising_characteristics['layer_types'].items() 
                             if count >= 2]  # Appear in at least 2 good architectures
            if top_layer_types:
                refined_space['layer_types'] = top_layer_types
                logger.info(f"  Focused on layer types: {top_layer_types[:5]}")
        
        # Refine depth range
        if promising_characteristics['depths']:
            depths = promising_characteristics['depths']
            min_depth = max(3, min(depths) - 2)
            max_depth = min(50, max(depths) + 5)
            refined_space['depth_range'] = (min_depth, max_depth)
            logger.info(f"  Refined depth range: ({min_depth}, {max_depth})")
        
        # Update search space
        self.search_space = refined_space
        self.search_space_history.append(self.search_space.copy())
    
    def _analyze_pareto_characteristics(self) -> Dict[str, Any]:
        """Analyze characteristics of Pareto optimal solutions."""
        
        characteristics = {
            'layer_types': defaultdict(int),
            'depths': [],
            'skip_connection_ratios': [],
            'parameter_ranges': []
        }
        
        # Get architecture specs for Pareto solutions
        for solution in self.pareto_front:
            # Find corresponding architecture (simplified lookup)
            arch_spec = self._find_architecture_spec(solution.architecture_id)
            if arch_spec:
                # Count layer types
                for layer in arch_spec.layers:
                    layer_type = layer.get('type', 'unknown')
                    characteristics['layer_types'][layer_type] += 1
                
                # Record depth
                characteristics['depths'].append(len(arch_spec.layers))
                
                # Calculate skip connection ratio
                skip_connections = len([conn for conn in arch_spec.connections 
                                      if abs(conn[1] - conn[0]) > 1])
                total_connections = len(arch_spec.connections)
                skip_ratio = skip_connections / max(1, total_connections)
                characteristics['skip_connection_ratios'].append(skip_ratio)
                
                # Record parameter count
                characteristics['parameter_ranges'].append(arch_spec.estimated_params)
        
        return characteristics
    
    def _find_architecture_spec(self, architecture_id: str) -> Optional[ArchitectureSpec]:
        """Find architecture spec by ID (simplified implementation)."""
        # In a real implementation, this would maintain a mapping
        # For demo, we'll create a simple architecture spec
        return ArchitectureSpec(
            architecture_id=architecture_id,
            layers=[
                {'type': 'conv2d', 'filters': 64, 'kernel_size': 3},
                {'type': 'batch_norm'},
                {'type': 'relu'},
                {'type': 'conv2d', 'filters': 128, 'kernel_size': 3},
                {'type': 'attention', 'heads': 8},
                {'type': 'linear', 'size': 512}
            ],
            connections=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
            input_shape=(224, 224, 3),
            output_shape=(1000,),
            estimated_params=random.randint(100000, 5000000),
            estimated_flops=random.uniform(1e6, 1e9),
            created_timestamp=time.time()
        )
    
    async def _get_best_architectures_by_objective(self, objectives: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get best architecture for each objective."""
        
        best_architectures = {}
        
        for objective in objectives:
            best_arch = None
            best_score = -float('inf')
            
            for evaluation in self.archive:
                if objective == 'accuracy':
                    score = evaluation.performance.accuracy
                elif objective == 'efficiency':
                    score = evaluation.performance.accuracy / max(0.1, evaluation.performance.training_time)
                elif objective == 'latency':
                    score = 1.0 / max(0.001, evaluation.performance.inference_time)
                elif objective == 'memory':
                    score = 1.0 / max(1.0, evaluation.performance.memory_usage)
                elif objective == 'stability':
                    score = evaluation.performance.stability_score
                else:
                    continue
                
                if score > best_score:
                    best_score = score
                    best_arch = evaluation
            
            if best_arch:
                best_architectures[objective] = {
                    'architecture_id': best_arch.architecture_id,
                    'score': best_score,
                    'performance': asdict(best_arch.performance),
                    'pareto_score': best_arch.pareto_score
                }
        
        return best_architectures
    
    def _get_search_statistics(self) -> Dict[str, Any]:
        """Get comprehensive search statistics."""
        
        if not self.archive:
            return {'message': 'No architectures evaluated'}
        
        accuracies = [eval.performance.accuracy for eval in self.archive]
        training_times = [eval.performance.training_time for eval in self.archive]
        inference_times = [eval.performance.inference_time for eval in self.archive]
        
        return {
            'total_architectures_evaluated': len(self.archive),
            'pareto_front_size': len(self.pareto_front),
            'search_generations': self.search_state.current_generation + 1,
            'accuracy_statistics': {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'min': np.min(accuracies),
                'max': np.max(accuracies)
            },
            'training_time_statistics': {
                'mean': np.mean(training_times),
                'std': np.std(training_times),
                'min': np.min(training_times),
                'max': np.max(training_times)
            },
            'inference_time_statistics': {
                'mean': np.mean(inference_times),
                'std': np.std(inference_times),
                'min': np.min(inference_times),
                'max': np.max(inference_times)
            },
            'search_budget_utilization': self.search_state.search_budget_used,
            'early_stopping_triggered': self.search_state.stagnation_counter >= self.search_state.early_stopping_patience,
            'search_space_refinements': len(self.search_space_history)
        }


class EvolutionarySearcher:
    """Evolutionary search strategy for NAS."""
    
    def __init__(self, parent: NeuralArchitectureSearchEngine):
        self.parent = parent
        self.population_size = 20
        self.mutation_rate = 0.3
        self.crossover_rate = 0.7
    
    async def generate_candidate(self) -> ArchitectureSpec:
        """Generate candidate architecture using evolutionary approach."""
        
        if len(self.parent.population) < self.population_size:
            # Generate random architecture for initial population
            return self._generate_random_architecture()
        else:
            # Evolve existing population
            if random.random() < self.crossover_rate:
                return await self._crossover()
            else:
                return await self._mutate()
    
    def _generate_random_architecture(self) -> ArchitectureSpec:
        """Generate a random architecture."""
        
        search_space = self.parent.search_space
        
        # Random depth
        depth = random.randint(*search_space['depth_range'])
        
        # Generate layers
        layers = []
        for i in range(depth):
            layer_type = random.choice(search_space['layer_types'])
            
            layer = {'type': layer_type}
            
            if layer_type in ['conv2d', 'depthwise_conv2d']:
                layer.update({
                    'filters': random.choice(search_space['conv_filters']),
                    'kernel_size': random.choice(search_space['kernel_sizes']),
                    'stride': random.choice(search_space['strides'])
                })
            elif layer_type in ['attention', 'multi_head_attention']:
                layer.update({
                    'heads': random.choice(search_space['attention_heads']),
                    'dim': random.choice(search_space['hidden_sizes'])
                })
            elif layer_type == 'linear':
                layer.update({
                    'size': random.choice(search_space['hidden_sizes'])
                })
            elif layer_type in ['dropout', 'spatial_dropout']:
                layer.update({
                    'rate': random.choice(search_space['dropout_rates'])
                })
            
            layers.append(layer)
        
        # Generate connections (mostly sequential with some skip connections)
        connections = []
        for i in range(len(layers) - 1):
            connections.append((i, i + 1))  # Sequential connection
        
        # Add skip connections
        for i in range(len(layers)):
            if (i > 2 and random.random() < search_space['skip_connection_probability']):
                skip_target = random.randint(i + 1, min(i + 4, len(layers) - 1))
                connections.append((i, skip_target))
        
        # Estimate parameters and FLOPs
        estimated_params = self._estimate_parameters(layers)
        estimated_flops = self._estimate_flops(layers)
        
        architecture_id = f"evo_arch_{hashlib.md5(str(layers).encode()).hexdigest()[:8]}"
        
        return ArchitectureSpec(
            architecture_id=architecture_id,
            layers=layers,
            connections=connections,
            input_shape=(224, 224, 3),
            output_shape=(1000,),
            estimated_params=estimated_params,
            estimated_flops=estimated_flops,
            created_timestamp=time.time()
        )
    
    def _estimate_parameters(self, layers: List[Dict[str, Any]]) -> int:
        """Estimate number of parameters."""
        total_params = 0
        
        for layer in layers:
            layer_type = layer.get('type')
            
            if layer_type == 'conv2d':
                filters = layer.get('filters', 64)
                kernel_size = layer.get('kernel_size', 3)
                # Assuming input channels = output channels of previous layer
                in_channels = 64  # Simplified
                total_params += filters * in_channels * (kernel_size ** 2)
            elif layer_type == 'linear':
                size = layer.get('size', 512)
                in_size = 512  # Simplified
                total_params += size * in_size
            elif layer_type in ['attention', 'multi_head_attention']:
                dim = layer.get('dim', 512)
                heads = layer.get('heads', 8)
                total_params += heads * dim * dim * 3  # Q, K, V matrices
        
        return total_params
    
    def _estimate_flops(self, layers: List[Dict[str, Any]]) -> float:
        """Estimate FLOPs (floating point operations)."""
        total_flops = 0.0
        
        for layer in layers:
            layer_type = layer.get('type')
            
            if layer_type == 'conv2d':
                filters = layer.get('filters', 64)
                kernel_size = layer.get('kernel_size', 3)
                # Assuming 224x224 input with appropriate padding/stride
                output_size = 224 // layer.get('stride', 1)
                total_flops += filters * (kernel_size ** 2) * (output_size ** 2) * 64
            elif layer_type == 'linear':
                size = layer.get('size', 512)
                in_size = 512
                total_flops += size * in_size
            elif layer_type in ['attention', 'multi_head_attention']:
                dim = layer.get('dim', 512)
                seq_len = 196  # 14x14 for vision transformer
                total_flops += dim * seq_len * seq_len  # Attention computation
        
        return total_flops
    
    async def _crossover(self) -> ArchitectureSpec:
        """Create offspring through crossover of two parents."""
        
        # Select two parents (tournament selection)
        parent1 = self._tournament_selection()
        parent2 = self._tournament_selection()
        
        # Find their architecture specs
        parent1_spec = self._find_spec_by_evaluation(parent1)
        parent2_spec = self._find_spec_by_evaluation(parent2)
        
        if not parent1_spec or not parent2_spec:
            return self._generate_random_architecture()
        
        # Crossover layers (take first half from parent1, second half from parent2)
        crossover_point = len(parent1_spec.layers) // 2
        
        offspring_layers = (parent1_spec.layers[:crossover_point] + 
                          parent2_spec.layers[crossover_point:])
        
        # Generate new connections
        connections = []
        for i in range(len(offspring_layers) - 1):
            connections.append((i, i + 1))
        
        # Estimate parameters and FLOPs
        estimated_params = self._estimate_parameters(offspring_layers)
        estimated_flops = self._estimate_flops(offspring_layers)
        
        architecture_id = f"cross_arch_{hashlib.md5(str(offspring_layers).encode()).hexdigest()[:8]}"
        
        return ArchitectureSpec(
            architecture_id=architecture_id,
            layers=offspring_layers,
            connections=connections,
            input_shape=(224, 224, 3),
            output_shape=(1000,),
            estimated_params=estimated_params,
            estimated_flops=estimated_flops,
            created_timestamp=time.time()
        )
    
    async def _mutate(self) -> ArchitectureSpec:
        """Create offspring through mutation of a parent."""
        
        parent = self._tournament_selection()
        parent_spec = self._find_spec_by_evaluation(parent)
        
        if not parent_spec:
            return self._generate_random_architecture()
        
        # Copy parent layers and mutate
        mutated_layers = parent_spec.layers.copy()
        
        # Random mutations
        for i, layer in enumerate(mutated_layers):
            if random.random() < self.mutation_rate:
                mutated_layers[i] = self._mutate_layer(layer)
        
        # Possibly add or remove layers
        if random.random() < 0.1:  # 10% chance to add layer
            new_layer = self._generate_random_layer()
            insert_pos = random.randint(0, len(mutated_layers))
            mutated_layers.insert(insert_pos, new_layer)
        elif random.random() < 0.1 and len(mutated_layers) > 3:  # 10% chance to remove layer
            remove_pos = random.randint(0, len(mutated_layers) - 1)
            mutated_layers.pop(remove_pos)
        
        # Generate connections
        connections = []
        for i in range(len(mutated_layers) - 1):
            connections.append((i, i + 1))
        
        # Add some skip connections
        for i in range(len(mutated_layers)):
            if i > 2 and random.random() < 0.3:
                skip_target = random.randint(i + 1, min(i + 3, len(mutated_layers) - 1))
                connections.append((i, skip_target))
        
        # Estimate parameters and FLOPs
        estimated_params = self._estimate_parameters(mutated_layers)
        estimated_flops = self._estimate_flops(mutated_layers)
        
        architecture_id = f"mut_arch_{hashlib.md5(str(mutated_layers).encode()).hexdigest()[:8]}"
        
        return ArchitectureSpec(
            architecture_id=architecture_id,
            layers=mutated_layers,
            connections=connections,
            input_shape=(224, 224, 3),
            output_shape=(1000,),
            estimated_params=estimated_params,
            estimated_flops=estimated_flops,
            created_timestamp=time.time()
        )
    
    def _tournament_selection(self, tournament_size: int = 3) -> EvaluationResult:
        """Select parent using tournament selection."""
        
        candidates = random.sample(self.parent.archive, min(tournament_size, len(self.parent.archive)))
        return max(candidates, key=lambda x: x.pareto_score)
    
    def _find_spec_by_evaluation(self, evaluation: EvaluationResult) -> Optional[ArchitectureSpec]:
        """Find architecture spec for an evaluation (simplified)."""
        # In real implementation, maintain proper mapping
        return self._generate_random_architecture()  # Fallback
    
    def _mutate_layer(self, layer: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate a single layer."""
        
        mutated = layer.copy()
        layer_type = layer.get('type')
        search_space = self.parent.search_space
        
        if layer_type == 'conv2d':
            if 'filters' in mutated and random.random() < 0.5:
                mutated['filters'] = random.choice(search_space['conv_filters'])
            if 'kernel_size' in mutated and random.random() < 0.3:
                mutated['kernel_size'] = random.choice(search_space['kernel_sizes'])
        elif layer_type == 'attention':
            if 'heads' in mutated and random.random() < 0.4:
                mutated['heads'] = random.choice(search_space['attention_heads'])
        elif layer_type == 'linear':
            if 'size' in mutated and random.random() < 0.4:
                mutated['size'] = random.choice(search_space['hidden_sizes'])
        
        return mutated
    
    def _generate_random_layer(self) -> Dict[str, Any]:
        """Generate a random layer."""
        
        search_space = self.parent.search_space
        layer_type = random.choice(search_space['layer_types'])
        
        layer = {'type': layer_type}
        
        if layer_type == 'conv2d':
            layer.update({
                'filters': random.choice(search_space['conv_filters']),
                'kernel_size': random.choice(search_space['kernel_sizes'])
            })
        elif layer_type == 'attention':
            layer.update({
                'heads': random.choice(search_space['attention_heads']),
                'dim': random.choice(search_space['hidden_sizes'])
            })
        elif layer_type == 'linear':
            layer.update({
                'size': random.choice(search_space['hidden_sizes'])
            })
        
        return layer


class BayesianSearcher:
    """Bayesian optimization for NAS."""
    
    def __init__(self, parent: NeuralArchitectureSearchEngine):
        self.parent = parent
        self.gp_model = None
        self.acquisition_function = 'expected_improvement'
    
    async def generate_candidate(self) -> ArchitectureSpec:
        """Generate candidate using Bayesian optimization."""
        
        if len(self.parent.archive) < 5:
            # Not enough data for Bayesian optimization
            return self._generate_random_architecture()
        
        # Update Gaussian Process model
        await self._update_gp_model()
        
        # Optimize acquisition function to find next candidate
        candidate_encoding = await self._optimize_acquisition()
        
        # Convert encoding back to architecture
        return self._decode_architecture(candidate_encoding)
    
    def _generate_random_architecture(self) -> ArchitectureSpec:
        """Generate random architecture (fallback)."""
        # Use evolutionary searcher's random generation
        evo_searcher = EvolutionarySearcher(self.parent)
        return evo_searcher._generate_random_architecture()
    
    async def _update_gp_model(self):
        """Update Gaussian Process model with recent evaluations."""
        
        if not SKLEARN_AVAILABLE:
            return
        
        # Encode architectures and get their scores
        X = []
        y = []
        
        for evaluation in self.parent.archive:
            encoding = self._encode_architecture_simple(evaluation.architecture_id)
            X.append(encoding)
            y.append(evaluation.pareto_score)
        
        if len(X) < 3:
            return
        
        X = np.array(X)
        y = np.array(y)
        
        # Initialize or update Gaussian Process
        kernel = Matern(length_scale=1.0, nu=2.5)
        self.gp_model = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        self.gp_model.fit(X, y)
    
    def _encode_architecture_simple(self, architecture_id: str) -> List[float]:
        """Simple encoding of architecture for GP (placeholder)."""
        # In real implementation, this would encode architecture features
        # For demo, we'll use a hash-based encoding
        hash_val = hash(architecture_id)
        return [
            (hash_val % 1000) / 1000.0,  # Normalized hash component 1
            ((hash_val // 1000) % 1000) / 1000.0,  # Component 2
            ((hash_val // 1000000) % 1000) / 1000.0,  # Component 3
            random.uniform(0, 1),  # Random component
            random.uniform(0, 1)   # Another random component
        ]
    
    async def _optimize_acquisition(self) -> List[float]:
        """Optimize acquisition function to find next candidate."""
        
        if not self.gp_model or not SCIPY_AVAILABLE:
            # Fallback to random encoding
            return [random.uniform(0, 1) for _ in range(5)]
        
        # Define acquisition function
        def acquisition(x):
            x = np.array(x).reshape(1, -1)
            mean, std = self.gp_model.predict(x, return_std=True)
            
            if self.acquisition_function == 'expected_improvement':
                # Expected Improvement
                best_f = max(eval.pareto_score for eval in self.parent.archive)
                z = (mean - best_f) / (std + 1e-9)
                ei = (mean - best_f) * norm.cdf(z) + std * norm.pdf(z)
                return -ei[0]  # Minimize negative EI
            else:
                # Upper Confidence Bound
                return -(mean + 2.0 * std)[0]
        
        # Optimize acquisition function
        bounds = [(0, 1)] * 5  # 5D search space
        result = minimize(acquisition, 
                         x0=[random.uniform(0, 1) for _ in range(5)],
                         bounds=bounds,
                         method='L-BFGS-B')
        
        return result.x.tolist() if result.success else [random.uniform(0, 1) for _ in range(5)]
    
    def _decode_architecture(self, encoding: List[float]) -> ArchitectureSpec:
        """Decode architecture from encoding (simplified)."""
        # This is a simplified decoding - real implementation would be more sophisticated
        
        search_space = self.parent.search_space
        
        # Use encoding to determine architecture characteristics
        depth = int(encoding[0] * (search_space['depth_range'][1] - search_space['depth_range'][0]) + 
                   search_space['depth_range'][0])
        
        layers = []
        for i in range(depth):
            # Use different parts of encoding to select layer properties
            layer_type_idx = int(encoding[i % len(encoding)] * len(search_space['layer_types']))
            layer_type = search_space['layer_types'][layer_type_idx]
            
            layer = {'type': layer_type}
            
            if layer_type == 'conv2d':
                filter_idx = int(encoding[(i+1) % len(encoding)] * len(search_space['conv_filters']))
                kernel_idx = int(encoding[(i+2) % len(encoding)] * len(search_space['kernel_sizes']))
                layer.update({
                    'filters': search_space['conv_filters'][filter_idx],
                    'kernel_size': search_space['kernel_sizes'][kernel_idx]
                })
            elif layer_type == 'attention':
                head_idx = int(encoding[(i+1) % len(encoding)] * len(search_space['attention_heads']))
                dim_idx = int(encoding[(i+2) % len(encoding)] * len(search_space['hidden_sizes']))
                layer.update({
                    'heads': search_space['attention_heads'][head_idx],
                    'dim': search_space['hidden_sizes'][dim_idx]
                })
            
            layers.append(layer)
        
        # Generate connections
        connections = []
        for i in range(len(layers) - 1):
            connections.append((i, i + 1))
        
        # Estimate parameters and FLOPs
        estimated_params = sum(random.randint(1000, 100000) for _ in layers)
        estimated_flops = sum(random.uniform(1e4, 1e6) for _ in layers)
        
        architecture_id = f"bayes_arch_{hashlib.md5(str(encoding).encode()).hexdigest()[:8]}"
        
        return ArchitectureSpec(
            architecture_id=architecture_id,
            layers=layers,
            connections=connections,
            input_shape=(224, 224, 3),
            output_shape=(1000,),
            estimated_params=estimated_params,
            estimated_flops=estimated_flops,
            created_timestamp=time.time()
        )


class ProgressiveSearcher:
    """Progressive search that starts simple and increases complexity."""
    
    def __init__(self, parent: NeuralArchitectureSearchEngine):
        self.parent = parent
        self.complexity_schedule = [0.2, 0.4, 0.6, 0.8, 1.0]  # Complexity progression
        self.current_complexity_idx = 0
    
    async def generate_candidate(self) -> ArchitectureSpec:
        """Generate candidate with progressive complexity."""
        
        # Update complexity based on search progress
        progress = self.parent.search_state.search_budget_used
        target_complexity_idx = min(len(self.complexity_schedule) - 1, 
                                  int(progress * len(self.complexity_schedule)))
        
        if target_complexity_idx > self.current_complexity_idx:
            self.current_complexity_idx = target_complexity_idx
            logger.info(f"📈 Progressing to complexity level {self.current_complexity_idx + 1}")
        
        complexity_factor = self.complexity_schedule[self.current_complexity_idx]
        
        return self._generate_architecture_with_complexity(complexity_factor)
    
    def _generate_architecture_with_complexity(self, complexity_factor: float) -> ArchitectureSpec:
        """Generate architecture with specified complexity level."""
        
        search_space = self.parent.search_space
        
        # Adjust depth based on complexity
        min_depth, max_depth = search_space['depth_range']
        target_depth = int(min_depth + (max_depth - min_depth) * complexity_factor)
        depth = random.randint(max(3, target_depth - 3), target_depth + 3)
        
        # Generate layers with complexity-appropriate types
        layer_types = self._filter_layer_types_by_complexity(complexity_factor)
        
        layers = []
        for i in range(depth):
            layer_type = random.choice(layer_types)
            layer = {'type': layer_type}
            
            if layer_type == 'conv2d':
                # Filter complexity based on complexity factor
                available_filters = [f for f in search_space['conv_filters'] 
                                   if f <= search_space['conv_filters'][int(len(search_space['conv_filters']) * complexity_factor)]]
                layer.update({
                    'filters': random.choice(available_filters or [32]),
                    'kernel_size': random.choice(search_space['kernel_sizes'][:int(len(search_space['kernel_sizes']) * complexity_factor) + 1])
                })
            elif layer_type == 'attention':
                available_heads = [h for h in search_space['attention_heads']
                                 if h <= search_space['attention_heads'][int(len(search_space['attention_heads']) * complexity_factor)]]
                layer.update({
                    'heads': random.choice(available_heads or [1]),
                    'dim': random.choice(search_space['hidden_sizes'][:int(len(search_space['hidden_sizes']) * complexity_factor) + 1])
                })
            
            layers.append(layer)
        
        # Generate connections
        connections = []
        for i in range(len(layers) - 1):
            connections.append((i, i + 1))
        
        # Add skip connections based on complexity
        skip_prob = complexity_factor * search_space['skip_connection_probability']
        for i in range(len(layers)):
            if i > 1 and random.random() < skip_prob:
                skip_target = random.randint(i + 1, min(i + 3, len(layers) - 1))
                connections.append((i, skip_target))
        
        # Estimate parameters and FLOPs
        estimated_params = self._estimate_parameters_progressive(layers, complexity_factor)
        estimated_flops = self._estimate_flops_progressive(layers, complexity_factor)
        
        architecture_id = f"prog_arch_{complexity_factor:.1f}_{hashlib.md5(str(layers).encode()).hexdigest()[:8]}"
        
        return ArchitectureSpec(
            architecture_id=architecture_id,
            layers=layers,
            connections=connections,
            input_shape=(224, 224, 3),
            output_shape=(1000,),
            estimated_params=estimated_params,
            estimated_flops=estimated_flops,
            created_timestamp=time.time()
        )
    
    def _filter_layer_types_by_complexity(self, complexity_factor: float) -> List[str]:
        """Filter layer types based on complexity level."""
        
        all_types = self.parent.search_space['layer_types']
        
        # Define complexity levels for different layer types
        complexity_map = {
            'linear': 0.1,
            'conv2d': 0.2,
            'batch_norm': 0.1,
            'relu': 0.1,
            'dropout': 0.2,
            'max_pool': 0.2,
            'avg_pool': 0.2,
            'depthwise_conv2d': 0.4,
            'separable_conv': 0.5,
            'attention': 0.7,
            'multi_head_attention': 0.8,
            'residual_block': 0.6,
            'dense_block': 0.9,
            'squeeze_excite': 0.8
        }
        
        # Filter types that are appropriate for current complexity level
        available_types = [
            layer_type for layer_type in all_types
            if complexity_map.get(layer_type, 0.5) <= complexity_factor + 0.1  # Small tolerance
        ]
        
        return available_types or ['linear', 'conv2d', 'relu']  # Fallback to basics
    
    def _estimate_parameters_progressive(self, layers: List[Dict[str, Any]], 
                                       complexity_factor: float) -> int:
        """Estimate parameters with complexity awareness."""
        total_params = 0
        base_multiplier = 1.0 + complexity_factor
        
        for layer in layers:
            layer_type = layer.get('type')
            
            if layer_type == 'conv2d':
                filters = layer.get('filters', 32)
                kernel_size = layer.get('kernel_size', 3)
                total_params += int(filters * 32 * (kernel_size ** 2) * base_multiplier)
            elif layer_type == 'linear':
                size = layer.get('size', 128)
                total_params += int(size * 128 * base_multiplier)
            elif layer_type in ['attention', 'multi_head_attention']:
                dim = layer.get('dim', 256)
                heads = layer.get('heads', 4)
                total_params += int(heads * dim * dim * 3 * base_multiplier)
        
        return total_params
    
    def _estimate_flops_progressive(self, layers: List[Dict[str, Any]], 
                                  complexity_factor: float) -> float:
        """Estimate FLOPs with complexity awareness."""
        total_flops = 0.0
        base_multiplier = 1.0 + complexity_factor
        
        for layer in layers:
            layer_type = layer.get('type')
            
            if layer_type == 'conv2d':
                filters = layer.get('filters', 32)
                kernel_size = layer.get('kernel_size', 3)
                total_flops += filters * (kernel_size ** 2) * 224 * 224 * base_multiplier
            elif layer_type == 'linear':
                size = layer.get('size', 128)
                total_flops += size * 128 * base_multiplier
            elif layer_type in ['attention', 'multi_head_attention']:
                dim = layer.get('dim', 256)
                seq_len = 196
                total_flops += dim * seq_len * seq_len * base_multiplier
        
        return total_flops


class HardwareAwareOptimizer:
    """Hardware-aware architecture optimization."""
    
    def __init__(self, parent: NeuralArchitectureSearchEngine):
        self.parent = parent
        self.hardware_profiles = self._load_hardware_profiles()
    
    def _load_hardware_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load hardware constraint profiles."""
        return {
            'mobile': {
                'max_params': 10_000_000,  # 10M parameters
                'max_flops': 100_000_000,  # 100M FLOPs
                'max_memory': 100,  # 100MB
                'max_latency': 50,  # 50ms
                'preferred_layers': ['depthwise_conv2d', 'pointwise_conv2d', 'linear']
            },
            'edge': {
                'max_params': 50_000_000,  # 50M parameters
                'max_flops': 1_000_000_000,  # 1B FLOPs
                'max_memory': 500,  # 500MB
                'max_latency': 100,  # 100ms
                'preferred_layers': ['conv2d', 'separable_conv', 'attention']
            },
            'server': {
                'max_params': 1_000_000_000,  # 1B parameters
                'max_flops': 100_000_000_000,  # 100B FLOPs
                'max_memory': 8000,  # 8GB
                'max_latency': 1000,  # 1s
                'preferred_layers': ['multi_head_attention', 'dense_block', 'residual_block']
            }
        }
    
    async def apply_constraints(self, architecture: ArchitectureSpec,
                              constraints: Dict[str, Any]) -> ArchitectureSpec:
        """Apply hardware constraints to architecture."""
        
        hardware_type = constraints.get('hardware_type', 'server')
        profile = self.hardware_profiles.get(hardware_type, self.hardware_profiles['server'])
        
        # Check if architecture violates constraints
        violations = self._check_constraints(architecture, profile)
        
        if not violations:
            return architecture  # No violations, return as is
        
        # Apply fixes for violations
        fixed_architecture = await self._fix_constraint_violations(
            architecture, profile, violations
        )
        
        return fixed_architecture
    
    def _check_constraints(self, architecture: ArchitectureSpec,
                         profile: Dict[str, Any]) -> List[str]:
        """Check which constraints are violated."""
        violations = []
        
        if architecture.estimated_params > profile['max_params']:
            violations.append('max_params')
        
        if architecture.estimated_flops > profile['max_flops']:
            violations.append('max_flops')
        
        # Estimate memory usage
        estimated_memory = architecture.estimated_params * 4 / (1024 * 1024)  # MB
        if estimated_memory > profile['max_memory']:
            violations.append('max_memory')
        
        return violations
    
    async def _fix_constraint_violations(self, architecture: ArchitectureSpec,
                                       profile: Dict[str, Any],
                                       violations: List[str]) -> ArchitectureSpec:
        """Fix constraint violations by modifying architecture."""
        
        fixed_layers = architecture.layers.copy()
        
        # Strategy 1: Replace heavy layers with efficient alternatives
        if 'max_params' in violations or 'max_flops' in violations:
            fixed_layers = self._replace_heavy_layers(fixed_layers, profile)
        
        # Strategy 2: Reduce layer sizes
        if 'max_params' in violations:
            fixed_layers = self._reduce_layer_sizes(fixed_layers)
        
        # Strategy 3: Remove some layers if still too heavy
        if 'max_memory' in violations and len(fixed_layers) > 5:
            # Remove every 3rd layer (keeping essential ones)
            fixed_layers = [layer for i, layer in enumerate(fixed_layers) 
                          if i % 3 != 2 or i < 3 or i >= len(fixed_layers) - 2]
        
        # Regenerate connections
        connections = []
        for i in range(len(fixed_layers) - 1):
            connections.append((i, i + 1))
        
        # Recalculate estimates
        estimated_params = self._recalculate_params(fixed_layers)
        estimated_flops = self._recalculate_flops(fixed_layers)
        
        # Create fixed architecture
        fixed_id = f"hw_opt_{architecture.architecture_id}"
        
        return ArchitectureSpec(
            architecture_id=fixed_id,
            layers=fixed_layers,
            connections=connections,
            input_shape=architecture.input_shape,
            output_shape=architecture.output_shape,
            estimated_params=estimated_params,
            estimated_flops=estimated_flops,
            created_timestamp=time.time()
        )
    
    def _replace_heavy_layers(self, layers: List[Dict[str, Any]], 
                            profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Replace computationally heavy layers with efficient alternatives."""
        
        preferred_layers = profile.get('preferred_layers', [])
        fixed_layers = []
        
        for layer in layers:
            layer_type = layer.get('type')
            
            # Replace heavy layers with efficient alternatives
            if layer_type == 'conv2d' and 'depthwise_conv2d' in preferred_layers:
                # Replace conv2d with depthwise separable convolution
                new_layer = layer.copy()
                new_layer['type'] = 'depthwise_conv2d'
                # Reduce filter count
                if 'filters' in new_layer:
                    new_layer['filters'] = max(16, new_layer['filters'] // 2)
                fixed_layers.append(new_layer)
            elif layer_type == 'multi_head_attention' and layer_type not in preferred_layers:
                # Replace multi-head attention with simpler attention or linear layer
                if 'attention' in preferred_layers:
                    new_layer = {'type': 'attention', 'heads': 1, 'dim': layer.get('dim', 256) // 2}
                else:
                    new_layer = {'type': 'linear', 'size': layer.get('dim', 256) // 4}
                fixed_layers.append(new_layer)
            else:
                fixed_layers.append(layer)
        
        return fixed_layers
    
    def _reduce_layer_sizes(self, layers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reduce sizes of layers to fit parameter constraints."""
        
        reduced_layers = []
        
        for layer in layers:
            new_layer = layer.copy()
            
            if 'filters' in new_layer:
                new_layer['filters'] = max(8, new_layer['filters'] // 2)
            if 'size' in new_layer:
                new_layer['size'] = max(32, new_layer['size'] // 2)
            if 'dim' in new_layer:
                new_layer['dim'] = max(64, new_layer['dim'] // 2)
            if 'heads' in new_layer and new_layer['heads'] > 1:
                new_layer['heads'] = max(1, new_layer['heads'] // 2)
            
            reduced_layers.append(new_layer)
        
        return reduced_layers
    
    def _recalculate_params(self, layers: List[Dict[str, Any]]) -> int:
        """Recalculate parameter count for modified layers."""
        # Simplified recalculation
        total_params = 0
        
        for layer in layers:
            layer_type = layer.get('type')
            
            if layer_type in ['conv2d', 'depthwise_conv2d']:
                filters = layer.get('filters', 32)
                kernel_size = layer.get('kernel_size', 3)
                in_channels = 32  # Simplified assumption
                
                if layer_type == 'depthwise_conv2d':
                    total_params += filters * (kernel_size ** 2) + filters * in_channels  # Depthwise + pointwise
                else:
                    total_params += filters * in_channels * (kernel_size ** 2)
                    
            elif layer_type == 'linear':
                size = layer.get('size', 128)
                in_size = 128  # Simplified assumption
                total_params += size * in_size
                
            elif layer_type in ['attention', 'multi_head_attention']:
                dim = layer.get('dim', 256)
                heads = layer.get('heads', 1)
                total_params += heads * dim * dim * 3
        
        return total_params
    
    def _recalculate_flops(self, layers: List[Dict[str, Any]]) -> float:
        """Recalculate FLOP count for modified layers."""
        # Simplified recalculation
        total_flops = 0.0
        
        for layer in layers:
            layer_type = layer.get('type')
            
            if layer_type in ['conv2d', 'depthwise_conv2d']:
                filters = layer.get('filters', 32)
                kernel_size = layer.get('kernel_size', 3)
                output_size = 112  # Assuming some downsampling
                
                if layer_type == 'depthwise_conv2d':
                    # Depthwise: filters * kernel^2 * output^2
                    # Pointwise: filters * output^2 * in_channels
                    total_flops += filters * (kernel_size ** 2) * (output_size ** 2)
                    total_flops += filters * (output_size ** 2) * 32
                else:
                    total_flops += filters * (kernel_size ** 2) * (output_size ** 2) * 32
                    
            elif layer_type == 'linear':
                size = layer.get('size', 128)
                in_size = 128
                total_flops += size * in_size
                
            elif layer_type in ['attention', 'multi_head_attention']:
                dim = layer.get('dim', 256)
                seq_len = 196  # 14x14
                total_flops += dim * seq_len * seq_len
        
        return total_flops


class PerformancePredictor:
    """Predictor for architecture performance without full training."""
    
    def __init__(self, parent: NeuralArchitectureSearchEngine):
        self.parent = parent
        self.prediction_history = []
        self.feature_extractors = [
            self._extract_depth_features,
            self._extract_width_features,
            self._extract_layer_type_features,
            self._extract_connection_features,
            self._extract_parameter_features
        ]
    
    async def update(self, architecture: ArchitectureSpec, evaluation: EvaluationResult):
        """Update predictor with new architecture-performance pair."""
        
        features = self._extract_features(architecture)
        
        prediction_record = {
            'architecture_id': architecture.architecture_id,
            'features': features,
            'actual_performance': evaluation.performance.accuracy,
            'actual_pareto_score': evaluation.pareto_score,
            'timestamp': time.time()
        }
        
        self.prediction_history.append(prediction_record)
        
        # Keep only recent predictions
        if len(self.prediction_history) > 500:
            self.prediction_history = self.prediction_history[-500:]
    
    def _extract_features(self, architecture: ArchitectureSpec) -> List[float]:
        """Extract features from architecture for performance prediction."""
        
        all_features = []
        
        for extractor in self.feature_extractors:
            features = extractor(architecture)
            all_features.extend(features)
        
        return all_features
    
    def _extract_depth_features(self, architecture: ArchitectureSpec) -> List[float]:
        """Extract depth-related features."""
        return [
            len(architecture.layers),  # Total depth
            len(architecture.layers) / 50.0,  # Normalized depth
        ]
    
    def _extract_width_features(self, architecture: ArchitectureSpec) -> List[float]:
        """Extract width-related features."""
        
        conv_filters = []
        linear_sizes = []
        
        for layer in architecture.layers:
            if layer.get('type') == 'conv2d':
                conv_filters.append(layer.get('filters', 64))
            elif layer.get('type') == 'linear':
                linear_sizes.append(layer.get('size', 512))
        
        avg_conv_filters = np.mean(conv_filters) if conv_filters else 64
        avg_linear_size = np.mean(linear_sizes) if linear_sizes else 512
        
        return [
            avg_conv_filters / 1024.0,  # Normalized average conv filters
            avg_linear_size / 2048.0,   # Normalized average linear size
        ]
    
    def _extract_layer_type_features(self, architecture: ArchitectureSpec) -> List[float]:
        """Extract layer type distribution features."""
        
        layer_types = [layer.get('type', 'unknown') for layer in architecture.layers]
        total_layers = len(layer_types)
        
        type_counts = {
            'conv2d': layer_types.count('conv2d'),
            'linear': layer_types.count('linear'),
            'attention': layer_types.count('attention') + layer_types.count('multi_head_attention'),
            'normalization': (layer_types.count('batch_norm') + 
                            layer_types.count('layer_norm') + 
                            layer_types.count('group_norm')),
            'activation': (layer_types.count('relu') + 
                         layer_types.count('gelu') + 
                         layer_types.count('swish')),
        }
        
        # Convert to ratios
        return [count / max(1, total_layers) for count in type_counts.values()]
    
    def _extract_connection_features(self, architecture: ArchitectureSpec) -> List[float]:
        """Extract connection-related features."""
        
        total_connections = len(architecture.connections)
        sequential_connections = len(architecture.layers) - 1
        skip_connections = total_connections - sequential_connections
        
        return [
            skip_connections / max(1, len(architecture.layers)),  # Skip connection ratio
            total_connections / max(1, len(architecture.layers)),  # Connection density
        ]
    
    def _extract_parameter_features(self, architecture: ArchitectureSpec) -> List[float]:
        """Extract parameter and computation features."""
        
        return [
            architecture.estimated_params / 10_000_000,  # Normalized parameter count
            architecture.estimated_flops / 1_000_000_000,  # Normalized FLOP count
            np.log10(max(1, architecture.estimated_params)),  # Log parameter count
        ]
    
    async def predict_performance(self, architecture: ArchitectureSpec) -> Dict[str, float]:
        """Predict performance of architecture without training."""
        
        if len(self.prediction_history) < 5:
            # Not enough data for reliable prediction
            return {
                'predicted_accuracy': 0.7,
                'confidence': 0.1,
                'estimated_training_time': 100.0
            }
        
        features = self._extract_features(architecture)
        
        # Simple similarity-based prediction
        similarities = []
        for record in self.prediction_history:
            similarity = self._calculate_feature_similarity(features, record['features'])
            similarities.append((similarity, record))
        
        # Get top-k similar architectures
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_similar = similarities[:min(5, len(similarities))]
        
        if not top_similar:
            return {
                'predicted_accuracy': 0.7,
                'confidence': 0.1,
                'estimated_training_time': 100.0
            }
        
        # Weighted average based on similarity
        weighted_accuracy = 0.0
        weighted_training_time = 0.0
        total_weight = 0.0
        
        for similarity, record in top_similar:
            weight = similarity
            weighted_accuracy += weight * record['actual_performance']
            # Estimate training time from architecture complexity
            estimated_time = self._estimate_training_time_from_features(features)
            weighted_training_time += weight * estimated_time
            total_weight += weight
        
        if total_weight > 0:
            predicted_accuracy = weighted_accuracy / total_weight
            predicted_training_time = weighted_training_time / total_weight
        else:
            predicted_accuracy = 0.7
            predicted_training_time = 100.0
        
        # Calculate confidence based on similarity of top match
        confidence = top_similar[0][0] if top_similar else 0.1
        
        return {
            'predicted_accuracy': predicted_accuracy,
            'confidence': confidence,
            'estimated_training_time': predicted_training_time
        }
    
    def _calculate_feature_similarity(self, features1: List[float], 
                                    features2: List[float]) -> float:
        """Calculate similarity between feature vectors."""
        
        if len(features1) != len(features2):
            return 0.0
        
        # Cosine similarity
        dot_product = sum(f1 * f2 for f1, f2 in zip(features1, features2))
        norm1 = math.sqrt(sum(f1 * f1 for f1 in features1))
        norm2 = math.sqrt(sum(f2 * f2 for f2 in features2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _estimate_training_time_from_features(self, features: List[float]) -> float:
        """Estimate training time from architecture features."""
        
        # Extract relevant features for training time
        depth_normalized = features[1] if len(features) > 1 else 0.1
        param_normalized = features[-3] if len(features) >= 3 else 0.1
        flop_normalized = features[-2] if len(features) >= 2 else 0.1
        
        # Simple linear combination
        base_time = 60.0  # 1 minute base
        time_estimate = base_time * (1 + depth_normalized + param_normalized + flop_normalized)
        
        return max(10.0, time_estimate)  # At least 10 seconds


class EarlyStoppingPredictor:
    """Predictor for early stopping decisions to save computation."""
    
    def __init__(self, parent: NeuralArchitectureSearchEngine):
        self.parent = parent
        self.skip_threshold = 0.4  # Skip architectures predicted to perform below this
        self.confidence_threshold = 0.7  # Only skip if confidence is above this
    
    async def predict_early_stop(self, architecture: ArchitectureSpec) -> Dict[str, Any]:
        """Predict if architecture should be skipped for full evaluation."""
        
        # Get performance prediction
        prediction = await self.parent.performance_predictor.predict_performance(architecture)
        
        should_skip = (
            prediction['predicted_accuracy'] < self.skip_threshold and
            prediction['confidence'] > self.confidence_threshold
        )
        
        return {
            'should_skip': should_skip,
            'reason': 'Low predicted performance' if should_skip else 'Proceed with evaluation',
            'estimated_accuracy': prediction['predicted_accuracy'],
            'estimated_training_time': prediction['estimated_training_time'],
            'prediction_confidence': prediction['confidence']
        }


async def main():
    """Demonstrate Generation 6 Neural Architecture Search."""
    print("🔬 Generation 6: Advanced Neural Architecture Search")
    print("=" * 55)
    
    # Initialize NAS engine
    nas_config = {
        'early_stopping_patience': 8,
        'hardware_profile': 'server'
    }
    
    nas_engine = NeuralArchitectureSearchEngine(nas_config)
    
    print(f"🚀 NAS Engine initialized")
    print(f"Search ID: {nas_engine.search_state.search_id}")
    print(f"Search space layer types: {len(nas_engine.search_space['layer_types'])}")
    print()
    
    # Configure search parameters
    search_budget = 30  # Reduced for demo
    objectives = ['accuracy', 'efficiency', 'latency']
    hardware_constraints = {
        'hardware_type': 'edge',
        'max_params': 10_000_000,
        'max_latency': 100
    }
    
    print(f"🎯 Starting architecture search:")
    print(f"  Budget: {search_budget} evaluations")
    print(f"  Objectives: {objectives}")
    print(f"  Hardware: {hardware_constraints['hardware_type']}")
    print()
    
    # Run neural architecture search
    search_results = await nas_engine.search_architectures(
        search_budget=search_budget,
        hardware_constraints=hardware_constraints,
        objectives=objectives
    )
    
    print(f"✅ Architecture search completed!")
    print(f"  Total time: {search_results['total_time']:.2f} seconds")
    print(f"  Architectures evaluated: {len(search_results['architectures_found'])}")
    print(f"  Pareto front size: {len(search_results['pareto_front'])}")
    print()
    
    # Show best architectures for each objective
    print("🏆 Best architectures by objective:")
    for objective, arch_info in search_results['best_architectures'].items():
        print(f"  {objective.title()}: {arch_info['architecture_id']} (score: {arch_info['score']:.3f})")
        if 'performance' in arch_info:
            perf = arch_info['performance']
            print(f"    Accuracy: {perf['accuracy']:.3f}, Training time: {perf['training_time']:.1f}s")
    print()
    
    # Show search statistics
    stats = search_results['search_statistics']
    print("📊 Search Statistics:")
    print(f"  Accuracy range: {stats['accuracy_statistics']['min']:.3f} - {stats['accuracy_statistics']['max']:.3f}")
    print(f"  Average accuracy: {stats['accuracy_statistics']['mean']:.3f} ± {stats['accuracy_statistics']['std']:.3f}")
    print(f"  Training time range: {stats['training_time_statistics']['min']:.1f}s - {stats['training_time_statistics']['max']:.1f}s")
    print(f"  Early stopping: {stats['early_stopping_triggered']}")
    print(f"  Search space refinements: {stats['search_space_refinements']}")
    print()
    
    # Show Pareto front analysis
    if search_results['pareto_front']:
        print("🎯 Pareto Front Analysis:")
        pareto_architectures = search_results['pareto_front']
        
        # Sort by pareto score
        pareto_architectures.sort(key=lambda x: x['pareto_score'], reverse=True)
        
        print(f"  Top 3 Pareto optimal solutions:")
        for i, arch in enumerate(pareto_architectures[:3]):
            perf = arch['performance']
            print(f"    #{i+1}: {arch['architecture_id']}")
            print(f"        Pareto score: {arch['pareto_score']:.3f}")
            print(f"        Accuracy: {perf['accuracy']:.3f}")
            print(f"        Inference time: {perf['inference_time']:.3f}ms")
            print(f"        Model size: {perf['model_size']:.1f}MB")
    
    # Show search progress
    if search_results['search_progress']:
        print("\n📈 Search Progress (last 5 generations):")
        progress = search_results['search_progress'][-5:]
        for gen_info in progress:
            print(f"  Gen {gen_info['generation']}: "
                  f"Strategy={gen_info['strategy']}, "
                  f"Best accuracy={gen_info['best_accuracy']:.3f}, "
                  f"Pareto size={gen_info['pareto_front_size']}")
    
    # Demonstrate architecture prediction
    print("\n🔮 Performance Prediction Demonstration:")
    
    # Generate a test architecture
    test_arch = nas_engine.evolutionary_searcher._generate_random_architecture()
    prediction = await nas_engine.performance_predictor.predict_performance(test_arch)
    
    print(f"  Test architecture: {test_arch.architecture_id}")
    print(f"  Predicted accuracy: {prediction['predicted_accuracy']:.3f}")
    print(f"  Prediction confidence: {prediction['confidence']:.3f}")
    print(f"  Estimated training time: {prediction['estimated_training_time']:.1f}s")
    
    # Early stopping prediction
    early_stop = await nas_engine.early_stopping_predictor.predict_early_stop(test_arch)
    print(f"  Early stop recommendation: {early_stop['should_skip']} ({early_stop['reason']})")
    
    print("\n✨ Generation 6 Neural Architecture Search demonstration completed!")
    return search_results


if __name__ == "__main__":
    asyncio.run(main())