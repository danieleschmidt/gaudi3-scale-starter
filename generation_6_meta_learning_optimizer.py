#!/usr/bin/env python3
"""
Generation 6: Meta-Learning Optimization Engine
==============================================

Advanced meta-learning system that learns how to learn, adapting its learning
strategies based on experience across multiple domains and tasks.

Key Features:
- Meta-optimization algorithms that improve learning efficiency
- Cross-task knowledge transfer and adaptation
- Automatic hyperparameter optimization with learned priors
- Few-shot learning for rapid adaptation to new scenarios
- Self-improving optimization strategies
- Dynamic architecture morphing based on task characteristics
"""

import asyncio
import json
import logging
import time
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path
import random
from collections import defaultdict, deque
import pickle
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import optimization libraries with fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, using fallback implementations")

try:
    from scipy.optimize import minimize, differential_evolution
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available, using basic optimization")


@dataclass
class MetaLearningExperience:
    """Experience record for meta-learning system."""
    task_id: str
    domain: str
    learning_strategy: str
    hyperparameters: Dict[str, Any]
    performance_curve: List[float]
    convergence_time: float
    final_performance: float
    adaptation_speed: float
    transfer_effectiveness: float
    timestamp: float


@dataclass
class OptimizationStrategy:
    """Represents a learned optimization strategy."""
    strategy_id: str
    name: str
    hyperparameter_ranges: Dict[str, Tuple[float, float]]
    adaptation_rules: List[Dict[str, Any]]
    effectiveness_score: float
    applicability_domains: List[str]
    sample_efficiency: float
    convergence_reliability: float


@dataclass
class MetaLearningState:
    """State of the meta-learning system."""
    total_tasks_learned: int = 0
    optimization_strategies: int = 0
    successful_transfers: int = 0
    failed_transfers: int = 0
    meta_learning_iterations: int = 0
    average_adaptation_speed: float = 0.0
    cross_domain_knowledge: int = 0
    learned_priors: int = 0
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class MetaLearningOptimizer:
    """
    Generation 6: Meta-Learning Optimization Engine
    
    This system learns how to learn by:
    1. Analyzing optimization patterns across multiple tasks
    2. Building priors for hyperparameter initialization
    3. Developing task-specific adaptation strategies
    4. Transferring knowledge across domains
    5. Self-improving its learning algorithms
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.state = MetaLearningState()
        
        # Core components
        self.experience_buffer = deque(maxlen=5000)
        self.optimization_strategies = []
        self.learned_priors = {}
        self.task_embeddings = {}
        self.transfer_matrix = defaultdict(lambda: defaultdict(float))
        
        # Meta-optimization components
        self.meta_optimizer = MetaOptimizerEngine(self)
        self.few_shot_learner = FewShotLearningEngine(self)
        self.architecture_morpher = DynamicArchitectureMorpher(self)
        self.knowledge_transferer = CrossDomainTransferer(self)
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.adaptation_metrics = deque(maxlen=1000)
        
        logger.info("🧠 Generation 6 Meta-Learning Optimizer initialized")
    
    async def learn_from_task(self, task_config: Dict[str, Any], 
                             performance_data: List[float]) -> Dict[str, Any]:
        """Learn meta-optimization insights from a completed task."""
        
        task_id = task_config.get('task_id', f"task_{int(time.time())}")
        domain = task_config.get('domain', 'general')
        
        # Analyze the learning curve
        learning_analysis = self._analyze_learning_curve(performance_data)
        
        # Extract meta-learning insights
        meta_insights = await self._extract_meta_insights(
            task_config, performance_data, learning_analysis
        )
        
        # Create experience record
        experience = MetaLearningExperience(
            task_id=task_id,
            domain=domain,
            learning_strategy=task_config.get('strategy', 'standard'),
            hyperparameters=task_config.get('hyperparameters', {}),
            performance_curve=performance_data,
            convergence_time=learning_analysis['convergence_time'],
            final_performance=performance_data[-1] if performance_data else 0.0,
            adaptation_speed=learning_analysis['adaptation_speed'],
            transfer_effectiveness=learning_analysis.get('transfer_score', 0.5),
            timestamp=time.time()
        )
        
        self.experience_buffer.append(experience)
        
        # Update meta-learning state
        self.state.total_tasks_learned += 1
        self.state.average_adaptation_speed = self._update_average_adaptation_speed()
        
        # Learn from this experience
        await self._update_meta_knowledge(experience, meta_insights)
        
        logger.info(f"📚 Learned from task {task_id} in domain {domain}")
        return {
            'task_id': task_id,
            'meta_insights': meta_insights,
            'learning_analysis': learning_analysis,
            'updated_strategies': len(self.optimization_strategies)
        }
    
    def _analyze_learning_curve(self, performance_data: List[float]) -> Dict[str, Any]:
        """Analyze learning curve for meta-insights."""
        if len(performance_data) < 3:
            return {
                'convergence_time': 0.0,
                'adaptation_speed': 0.0,
                'stability_score': 0.0,
                'final_performance': 0.0
            }
        
        data = np.array(performance_data)
        
        # Calculate adaptation speed (early improvement rate)
        early_phase = min(10, len(data) // 4)
        if early_phase > 1:
            adaptation_speed = np.mean(np.diff(data[:early_phase]))
        else:
            adaptation_speed = 0.0
        
        # Estimate convergence time (when improvement rate drops below threshold)
        convergence_time = len(data)  # Default to full duration
        improvement_rates = np.diff(data)
        for i, rate in enumerate(improvement_rates):
            if abs(rate) < 0.01:  # Convergence threshold
                convergence_time = i + 1
                break
        
        # Calculate stability (variance in later stages)
        if len(data) > 10:
            later_phase = data[-10:]
            stability_score = 1.0 / (1.0 + np.var(later_phase))
        else:
            stability_score = 0.5
        
        return {
            'convergence_time': convergence_time,
            'adaptation_speed': adaptation_speed,
            'stability_score': stability_score,
            'final_performance': data[-1],
            'max_performance': np.max(data),
            'performance_variance': np.var(data)
        }
    
    async def _extract_meta_insights(self, task_config: Dict[str, Any], 
                                   performance_data: List[float],
                                   learning_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract meta-learning insights from task performance."""
        
        insights = {
            'hyperparameter_effectiveness': {},
            'strategy_performance': 0.0,
            'domain_characteristics': {},
            'transfer_potential': 0.0,
            'optimization_recommendations': []
        }
        
        # Analyze hyperparameter effectiveness
        hyperparams = task_config.get('hyperparameters', {})
        for param, value in hyperparams.items():
            effectiveness = self._evaluate_hyperparameter_effectiveness(
                param, value, learning_analysis
            )
            insights['hyperparameter_effectiveness'][param] = effectiveness
        
        # Evaluate strategy performance
        strategy = task_config.get('strategy', 'standard')
        insights['strategy_performance'] = learning_analysis['final_performance']
        
        # Identify domain characteristics
        domain = task_config.get('domain', 'general')
        insights['domain_characteristics'] = {
            'convergence_speed': learning_analysis['adaptation_speed'],
            'stability_requirement': learning_analysis['stability_score'],
            'complexity_level': self._estimate_task_complexity(performance_data)
        }
        
        # Assess transfer potential
        insights['transfer_potential'] = await self._assess_transfer_potential(
            domain, task_config, learning_analysis
        )
        
        # Generate optimization recommendations
        insights['optimization_recommendations'] = self._generate_optimization_recommendations(
            task_config, learning_analysis
        )
        
        return insights
    
    def _evaluate_hyperparameter_effectiveness(self, param: str, value: Any, 
                                             analysis: Dict[str, Any]) -> float:
        """Evaluate how effective a hyperparameter value was."""
        # Base effectiveness on final performance and adaptation speed
        base_score = analysis['final_performance'] * 0.7 + analysis['adaptation_speed'] * 0.3
        
        # Adjust based on parameter type and value
        if param == 'learning_rate':
            # Prefer moderate learning rates
            lr = float(value)
            if 0.001 <= lr <= 0.1:
                base_score *= 1.1
            elif lr < 0.0001 or lr > 0.5:
                base_score *= 0.8
        elif param == 'batch_size':
            # Larger batch sizes generally more stable
            batch_size = int(value)
            if 32 <= batch_size <= 256:
                base_score *= 1.05
        
        return max(0.0, min(1.0, base_score))
    
    def _estimate_task_complexity(self, performance_data: List[float]) -> str:
        """Estimate task complexity based on learning curve."""
        if len(performance_data) < 5:
            return 'unknown'
        
        data = np.array(performance_data)
        
        # Calculate learning curve characteristics
        initial_performance = data[0]
        final_performance = data[-1]
        improvement = final_performance - initial_performance
        variance = np.var(data)
        
        # Classify complexity
        if improvement > 0.3 and variance < 0.1:
            return 'simple'
        elif improvement > 0.2 and variance < 0.2:
            return 'moderate'
        elif improvement > 0.1:
            return 'complex'
        else:
            return 'very_complex'
    
    async def _assess_transfer_potential(self, domain: str, task_config: Dict[str, Any],
                                       analysis: Dict[str, Any]) -> float:
        """Assess how well insights from this task might transfer to others."""
        transfer_potential = 0.5  # Base score
        
        # Higher performance suggests more transferable insights
        if analysis['final_performance'] > 0.8:
            transfer_potential += 0.2
        
        # Stable learning curves are more transferable
        if analysis['stability_score'] > 0.8:
            transfer_potential += 0.15
        
        # Good adaptation speed suggests robust strategy
        if analysis['adaptation_speed'] > 0.1:
            transfer_potential += 0.1
        
        # Domain-specific adjustments
        if domain in ['vision', 'nlp', 'speech']:
            transfer_potential += 0.1  # Well-studied domains
        elif domain == 'multimodal':
            transfer_potential += 0.15  # High transfer value
        
        return max(0.0, min(1.0, transfer_potential))
    
    def _generate_optimization_recommendations(self, task_config: Dict[str, Any],
                                             analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        # Learning rate recommendations
        if analysis['adaptation_speed'] < 0.05:
            recommendations.append({
                'type': 'hyperparameter',
                'parameter': 'learning_rate',
                'suggestion': 'increase',
                'reason': 'Slow adaptation speed detected',
                'confidence': 0.7
            })
        elif analysis['stability_score'] < 0.5:
            recommendations.append({
                'type': 'hyperparameter',
                'parameter': 'learning_rate',
                'suggestion': 'decrease',
                'reason': 'High instability detected',
                'confidence': 0.8
            })
        
        # Architecture recommendations
        if analysis['convergence_time'] > 50:
            recommendations.append({
                'type': 'architecture',
                'suggestion': 'add_skip_connections',
                'reason': 'Slow convergence suggests gradient flow issues',
                'confidence': 0.6
            })
        
        # Strategy recommendations
        if analysis['final_performance'] < 0.6:
            recommendations.append({
                'type': 'strategy',
                'suggestion': 'curriculum_learning',
                'reason': 'Poor final performance suggests need for curriculum',
                'confidence': 0.65
            })
        
        return recommendations
    
    async def _update_meta_knowledge(self, experience: MetaLearningExperience,
                                   insights: Dict[str, Any]):
        """Update meta-knowledge based on new experience."""
        
        # Update learned priors for hyperparameters
        await self._update_hyperparameter_priors(experience, insights)
        
        # Update optimization strategies
        await self._update_optimization_strategies(experience, insights)
        
        # Update transfer knowledge
        await self._update_transfer_knowledge(experience, insights)
        
        # Update task embeddings for similarity matching
        await self._update_task_embeddings(experience)
        
        self.state.meta_learning_iterations += 1
    
    async def _update_hyperparameter_priors(self, experience: MetaLearningExperience,
                                          insights: Dict[str, Any]):
        """Update learned priors for hyperparameter initialization."""
        domain = experience.domain
        
        if domain not in self.learned_priors:
            self.learned_priors[domain] = {}
        
        for param, effectiveness in insights['hyperparameter_effectiveness'].items():
            if param not in self.learned_priors[domain]:
                self.learned_priors[domain][param] = {
                    'values': [],
                    'effectiveness_scores': [],
                    'recommended_range': None
                }
            
            # Add new data point
            param_value = experience.hyperparameters.get(param)
            if param_value is not None:
                self.learned_priors[domain][param]['values'].append(param_value)
                self.learned_priors[domain][param]['effectiveness_scores'].append(effectiveness)
                
                # Update recommended range based on top-performing values
                await self._update_recommended_range(domain, param)
        
        self.state.learned_priors = len(self.learned_priors)
    
    async def _update_recommended_range(self, domain: str, param: str):
        """Update recommended range for a hyperparameter."""
        prior_data = self.learned_priors[domain][param]
        
        if len(prior_data['values']) >= 3:
            # Find top-performing values
            paired_data = list(zip(prior_data['values'], prior_data['effectiveness_scores']))
            paired_data.sort(key=lambda x: x[1], reverse=True)
            
            # Take top 70% of values
            top_count = max(1, int(len(paired_data) * 0.7))
            top_values = [x[0] for x in paired_data[:top_count]]
            
            # Calculate recommended range
            if all(isinstance(v, (int, float)) for v in top_values):
                min_val = min(top_values)
                max_val = max(top_values)
                # Add some buffer
                range_buffer = (max_val - min_val) * 0.2
                prior_data['recommended_range'] = (
                    max(0, min_val - range_buffer),
                    max_val + range_buffer
                )
    
    async def _update_optimization_strategies(self, experience: MetaLearningExperience,
                                            insights: Dict[str, Any]):
        """Update or create optimization strategies based on experience."""
        
        strategy_name = experience.learning_strategy
        domain = experience.domain
        
        # Find existing strategy or create new one
        existing_strategy = None
        for strategy in self.optimization_strategies:
            if (strategy.name == strategy_name and 
                domain in strategy.applicability_domains):
                existing_strategy = strategy
                break
        
        if existing_strategy:
            # Update existing strategy
            await self._update_existing_strategy(existing_strategy, experience, insights)
        else:
            # Create new strategy
            new_strategy = await self._create_new_strategy(experience, insights)
            self.optimization_strategies.append(new_strategy)
            self.state.optimization_strategies += 1
    
    async def _update_existing_strategy(self, strategy: OptimizationStrategy,
                                      experience: MetaLearningExperience,
                                      insights: Dict[str, Any]):
        """Update an existing optimization strategy."""
        
        # Update effectiveness score (exponential moving average)
        alpha = 0.3  # Learning rate for updates
        new_score = experience.final_performance
        strategy.effectiveness_score = (
            alpha * new_score + (1 - alpha) * strategy.effectiveness_score
        )
        
        # Update sample efficiency
        convergence_efficiency = 1.0 / max(1.0, experience.convergence_time)
        strategy.sample_efficiency = (
            alpha * convergence_efficiency + (1 - alpha) * strategy.sample_efficiency
        )
        
        # Update hyperparameter ranges based on successful values
        for param, value in experience.hyperparameters.items():
            if param in strategy.hyperparameter_ranges:
                current_min, current_max = strategy.hyperparameter_ranges[param]
                
                # If this was a successful run, expand range to include this value
                if experience.final_performance > 0.7:
                    if isinstance(value, (int, float)):
                        new_min = min(current_min, value * 0.8)
                        new_max = max(current_max, value * 1.2)
                        strategy.hyperparameter_ranges[param] = (new_min, new_max)
    
    async def _create_new_strategy(self, experience: MetaLearningExperience,
                                 insights: Dict[str, Any]) -> OptimizationStrategy:
        """Create a new optimization strategy."""
        
        strategy_id = f"strategy_{experience.domain}_{int(time.time())}"
        
        # Initialize hyperparameter ranges
        hyperparameter_ranges = {}
        for param, value in experience.hyperparameters.items():
            if isinstance(value, (int, float)):
                # Create initial range around the observed value
                hyperparameter_ranges[param] = (value * 0.5, value * 1.5)
        
        # Create adaptation rules from recommendations
        adaptation_rules = []
        for rec in insights['optimization_recommendations']:
            if rec['confidence'] > 0.6:
                adaptation_rules.append({
                    'condition': rec.get('reason', 'generic'),
                    'action': rec.get('suggestion', 'tune'),
                    'parameter': rec.get('parameter', 'learning_rate'),
                    'confidence': rec['confidence']
                })
        
        return OptimizationStrategy(
            strategy_id=strategy_id,
            name=experience.learning_strategy,
            hyperparameter_ranges=hyperparameter_ranges,
            adaptation_rules=adaptation_rules,
            effectiveness_score=experience.final_performance,
            applicability_domains=[experience.domain],
            sample_efficiency=1.0 / max(1.0, experience.convergence_time),
            convergence_reliability=experience.adaptation_speed
        )
    
    async def _update_transfer_knowledge(self, experience: MetaLearningExperience,
                                       insights: Dict[str, Any]):
        """Update knowledge about cross-domain transfer effectiveness."""
        
        source_domain = experience.domain
        transfer_score = insights['transfer_potential']
        
        # Update transfer matrix for all other domains
        for target_domain in self.transfer_matrix:
            if target_domain != source_domain:
                # Use similarity and performance to estimate transfer effectiveness
                similarity_score = self._calculate_domain_similarity(
                    source_domain, target_domain
                )
                
                estimated_transfer = transfer_score * similarity_score
                
                # Update with exponential moving average
                current_score = self.transfer_matrix[source_domain][target_domain]
                alpha = 0.2
                self.transfer_matrix[source_domain][target_domain] = (
                    alpha * estimated_transfer + (1 - alpha) * current_score
                )
    
    def _calculate_domain_similarity(self, domain1: str, domain2: str) -> float:
        """Calculate similarity between two domains."""
        
        # Domain similarity mapping (could be learned or predefined)
        domain_groups = {
            'vision': ['image_classification', 'object_detection', 'segmentation'],
            'nlp': ['text_classification', 'translation', 'summarization'],
            'multimodal': ['vision_language', 'audio_visual', 'text_image'],
            'rl': ['control', 'games', 'robotics'],
            'scientific': ['physics', 'chemistry', 'biology']
        }
        
        # Find which groups the domains belong to
        group1 = None
        group2 = None
        
        for group, domains in domain_groups.items():
            if domain1 in domains:
                group1 = group
            if domain2 in domains:
                group2 = group
        
        # Calculate similarity
        if group1 == group2:
            return 0.8  # Same group - high similarity
        elif group1 and group2:
            # Different groups but both known
            cross_group_similarity = {
                ('vision', 'multimodal'): 0.6,
                ('nlp', 'multimodal'): 0.6,
                ('vision', 'scientific'): 0.4,
                ('nlp', 'scientific'): 0.4,
            }
            
            key = tuple(sorted([group1, group2]))
            return cross_group_similarity.get(key, 0.3)
        else:
            return 0.5  # Unknown domains - moderate similarity
    
    async def _update_task_embeddings(self, experience: MetaLearningExperience):
        """Update task embeddings for similarity matching."""
        
        task_id = experience.task_id
        
        # Create task embedding from experience characteristics
        embedding = {
            'domain': experience.domain,
            'final_performance': experience.final_performance,
            'convergence_time': experience.convergence_time,
            'adaptation_speed': experience.adaptation_speed,
            'hyperparameter_signature': self._create_hyperparameter_signature(
                experience.hyperparameters
            ),
            'complexity_level': self._estimate_task_complexity(experience.performance_curve)
        }
        
        self.task_embeddings[task_id] = embedding
    
    def _create_hyperparameter_signature(self, hyperparams: Dict[str, Any]) -> str:
        """Create a signature for hyperparameter configuration."""
        # Sort and hash hyperparameters for consistent signatures
        sorted_items = sorted(hyperparams.items())
        signature_string = str(sorted_items)
        return hashlib.md5(signature_string.encode()).hexdigest()[:8]
    
    def _update_average_adaptation_speed(self) -> float:
        """Update running average of adaptation speed."""
        if not self.experience_buffer:
            return 0.0
        
        recent_experiences = list(self.experience_buffer)[-20:]  # Last 20 experiences
        speeds = [exp.adaptation_speed for exp in recent_experiences]
        return sum(speeds) / len(speeds)
    
    async def suggest_hyperparameters(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest optimal hyperparameters for a new task based on learned priors."""
        
        domain = task_config.get('domain', 'general')
        task_characteristics = task_config.get('characteristics', {})
        
        suggestions = {}
        confidence_scores = {}
        
        # Use domain-specific priors if available
        if domain in self.learned_priors:
            domain_priors = self.learned_priors[domain]
            
            for param, prior_data in domain_priors.items():
                if prior_data['recommended_range']:
                    min_val, max_val = prior_data['recommended_range']
                    
                    # Suggest middle of the range as starting point
                    suggested_value = (min_val + max_val) / 2
                    suggestions[param] = suggested_value
                    
                    # Calculate confidence based on number of data points
                    data_points = len(prior_data['values'])
                    confidence = min(0.9, data_points / 10.0)  # Max confidence at 10+ points
                    confidence_scores[param] = confidence
        
        # Use similar task suggestions if no domain priors
        if not suggestions:
            similar_tasks = await self._find_similar_tasks(task_config)
            if similar_tasks:
                suggestions = await self._suggest_from_similar_tasks(similar_tasks)
                confidence_scores = {param: 0.5 for param in suggestions}  # Medium confidence
        
        # Default suggestions if nothing else available
        if not suggestions:
            suggestions = {
                'learning_rate': 0.001,
                'batch_size': 32,
                'optimizer': 'adam',
                'weight_decay': 0.0001
            }
            confidence_scores = {param: 0.3 for param in suggestions}  # Low confidence
        
        logger.info(f"💡 Suggested hyperparameters for {domain}: {suggestions}")
        
        return {
            'suggestions': suggestions,
            'confidence_scores': confidence_scores,
            'reasoning': f"Based on {len(self.experience_buffer)} previous experiences",
            'domain': domain
        }
    
    async def _find_similar_tasks(self, task_config: Dict[str, Any]) -> List[str]:
        """Find tasks similar to the given configuration."""
        
        target_domain = task_config.get('domain', 'general')
        target_characteristics = task_config.get('characteristics', {})
        
        similar_tasks = []
        
        for task_id, embedding in self.task_embeddings.items():
            similarity_score = self._calculate_task_similarity(
                task_config, embedding
            )
            
            if similarity_score > 0.6:  # Threshold for similarity
                similar_tasks.append(task_id)
        
        # Sort by similarity (approximate)
        return similar_tasks[:5]  # Return top 5 similar tasks
    
    def _calculate_task_similarity(self, task_config: Dict[str, Any], 
                                 embedding: Dict[str, Any]) -> float:
        """Calculate similarity between a task config and an embedding."""
        
        # Domain similarity
        config_domain = task_config.get('domain', 'general')
        embedding_domain = embedding.get('domain', 'general')
        domain_similarity = 1.0 if config_domain == embedding_domain else 0.5
        
        # Complexity similarity (if available)
        complexity_similarity = 0.5  # Default
        
        # Combined similarity (can be enhanced with more sophisticated methods)
        overall_similarity = domain_similarity * 0.7 + complexity_similarity * 0.3
        
        return overall_similarity
    
    async def _suggest_from_similar_tasks(self, similar_task_ids: List[str]) -> Dict[str, Any]:
        """Suggest hyperparameters based on similar tasks."""
        
        hyperparameter_collections = []
        
        # Collect hyperparameters from similar tasks
        for experience in self.experience_buffer:
            if experience.task_id in similar_task_ids:
                hyperparameter_collections.append(experience.hyperparameters)
        
        if not hyperparameter_collections:
            return {}
        
        # Calculate average values for numeric parameters
        suggestions = {}
        all_params = set()
        for hp_dict in hyperparameter_collections:
            all_params.update(hp_dict.keys())
        
        for param in all_params:
            values = []
            for hp_dict in hyperparameter_collections:
                if param in hp_dict and isinstance(hp_dict[param], (int, float)):
                    values.append(hp_dict[param])
            
            if values:
                # Use median for robustness
                suggestions[param] = np.median(values)
        
        return suggestions
    
    async def optimize_meta_strategy(self, performance_target: float = 0.9) -> Dict[str, Any]:
        """Optimize the meta-learning strategy itself."""
        
        logger.info(f"🎯 Optimizing meta-learning strategy for target performance: {performance_target}")
        
        optimization_results = {
            'initial_performance': self._calculate_current_meta_performance(),
            'optimization_steps': [],
            'final_performance': 0.0,
            'improved_strategies': 0,
            'new_insights': []
        }
        
        current_performance = optimization_results['initial_performance']
        
        # Iterative meta-optimization
        for iteration in range(5):  # Limit iterations for demo
            step_result = await self._meta_optimization_step(
                current_performance, performance_target
            )
            
            optimization_results['optimization_steps'].append(step_result)
            current_performance = step_result['new_performance']
            
            if current_performance >= performance_target:
                break
        
        optimization_results['final_performance'] = current_performance
        optimization_results['improved_strategies'] = len(self.optimization_strategies)
        
        # Generate new insights
        optimization_results['new_insights'] = await self._generate_meta_insights()
        
        logger.info(f"✨ Meta-optimization completed: {current_performance:.3f} performance")
        
        return optimization_results
    
    def _calculate_current_meta_performance(self) -> float:
        """Calculate current meta-learning system performance."""
        if not self.experience_buffer:
            return 0.5
        
        # Calculate based on recent performance trends
        recent_experiences = list(self.experience_buffer)[-10:]
        
        if not recent_experiences:
            return 0.5
        
        # Metrics: adaptation speed, final performance, transfer effectiveness
        adaptation_speeds = [exp.adaptation_speed for exp in recent_experiences]
        final_performances = [exp.final_performance for exp in recent_experiences]
        transfer_scores = [exp.transfer_effectiveness for exp in recent_experiences]
        
        # Weighted combination
        meta_performance = (
            0.4 * np.mean(final_performances) +
            0.3 * np.mean(adaptation_speeds) + 
            0.3 * np.mean(transfer_scores)
        )
        
        return max(0.0, min(1.0, meta_performance))
    
    async def _meta_optimization_step(self, current_performance: float,
                                    target_performance: float) -> Dict[str, Any]:
        """Perform one step of meta-optimization."""
        
        step_start = time.time()
        
        # Identify bottlenecks
        bottlenecks = self._identify_meta_bottlenecks()
        
        # Apply improvement strategies
        improvements = 0
        for bottleneck in bottlenecks[:3]:  # Address top 3 bottlenecks
            if await self._address_bottleneck(bottleneck):
                improvements += 1
        
        # Evaluate new performance
        new_performance = self._calculate_current_meta_performance()
        
        step_result = {
            'step_duration': time.time() - step_start,
            'bottlenecks_addressed': len(bottlenecks),
            'improvements_applied': improvements,
            'performance_change': new_performance - current_performance,
            'new_performance': new_performance
        }
        
        return step_result
    
    def _identify_meta_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify bottlenecks in the meta-learning system."""
        bottlenecks = []
        
        # Low adaptation speed
        if self.state.average_adaptation_speed < 0.1:
            bottlenecks.append({
                'type': 'slow_adaptation',
                'severity': 'high',
                'description': 'System adapts slowly to new tasks',
                'suggested_fix': 'improve_learning_rate_adaptation'
            })
        
        # Poor transfer effectiveness
        recent_transfers = list(self.experience_buffer)[-20:]
        if recent_transfers:
            avg_transfer = np.mean([exp.transfer_effectiveness for exp in recent_transfers])
            if avg_transfer < 0.6:
                bottlenecks.append({
                    'type': 'poor_transfer',
                    'severity': 'medium',
                    'description': 'Knowledge transfer between tasks is ineffective',
                    'suggested_fix': 'enhance_transfer_mechanisms'
                })
        
        # Limited strategy diversity
        if len(self.optimization_strategies) < 3:
            bottlenecks.append({
                'type': 'limited_strategies',
                'severity': 'medium',
                'description': 'Too few optimization strategies available',
                'suggested_fix': 'diversify_optimization_strategies'
            })
        
        return bottlenecks
    
    async def _address_bottleneck(self, bottleneck: Dict[str, Any]) -> bool:
        """Address a specific meta-learning bottleneck."""
        
        fix_type = bottleneck['suggested_fix']
        
        if fix_type == 'improve_learning_rate_adaptation':
            return await self._improve_learning_rate_adaptation()
        elif fix_type == 'enhance_transfer_mechanisms':
            return await self._enhance_transfer_mechanisms()
        elif fix_type == 'diversify_optimization_strategies':
            return await self._diversify_optimization_strategies()
        
        return False
    
    async def _improve_learning_rate_adaptation(self) -> bool:
        """Improve learning rate adaptation mechanisms."""
        
        # Analyze current learning rate effectiveness
        for domain in self.learned_priors:
            if 'learning_rate' in self.learned_priors[domain]:
                lr_data = self.learned_priors[domain]['learning_rate']
                
                # Refine recommended range based on adaptation speed
                if lr_data['recommended_range']:
                    min_lr, max_lr = lr_data['recommended_range']
                    
                    # If adaptation is slow, suggest higher learning rates
                    if self.state.average_adaptation_speed < 0.05:
                        new_min = min_lr * 1.5
                        new_max = max_lr * 2.0
                        lr_data['recommended_range'] = (new_min, new_max)
                        
                        logger.info(f"📈 Adjusted learning rate range for {domain}: ({new_min:.6f}, {new_max:.6f})")
                        return True
        
        return False
    
    async def _enhance_transfer_mechanisms(self) -> bool:
        """Enhance knowledge transfer mechanisms."""
        
        # Increase transfer scores between similar domains
        enhanced = 0
        
        for source_domain in self.transfer_matrix:
            for target_domain in self.transfer_matrix[source_domain]:
                current_score = self.transfer_matrix[source_domain][target_domain]
                
                # Boost transfer for domains with similar characteristics
                similarity = self._calculate_domain_similarity(source_domain, target_domain)
                if similarity > 0.6 and current_score < 0.7:
                    boosted_score = min(0.9, current_score + 0.2)
                    self.transfer_matrix[source_domain][target_domain] = boosted_score
                    enhanced += 1
        
        if enhanced > 0:
            logger.info(f"🔄 Enhanced {enhanced} transfer connections")
            self.state.cross_domain_knowledge += enhanced
            return True
        
        return False
    
    async def _diversify_optimization_strategies(self) -> bool:
        """Create additional optimization strategies for diversity."""
        
        # Generate new strategies based on underrepresented domains
        represented_domains = set()
        for strategy in self.optimization_strategies:
            represented_domains.update(strategy.applicability_domains)
        
        all_domains = set(exp.domain for exp in self.experience_buffer)
        underrepresented = all_domains - represented_domains
        
        new_strategies = 0
        for domain in list(underrepresented)[:2]:  # Add strategies for 2 domains
            new_strategy = await self._create_domain_strategy(domain)
            if new_strategy:
                self.optimization_strategies.append(new_strategy)
                new_strategies += 1
        
        if new_strategies > 0:
            logger.info(f"🎨 Created {new_strategies} new optimization strategies")
            self.state.optimization_strategies += new_strategies
            return True
        
        return False
    
    async def _create_domain_strategy(self, domain: str) -> Optional[OptimizationStrategy]:
        """Create a new optimization strategy for a specific domain."""
        
        # Find experiences from this domain
        domain_experiences = [
            exp for exp in self.experience_buffer if exp.domain == domain
        ]
        
        if not domain_experiences:
            return None
        
        # Analyze best-performing experiences
        domain_experiences.sort(key=lambda x: x.final_performance, reverse=True)
        top_experiences = domain_experiences[:3]
        
        # Extract common hyperparameter patterns
        hyperparameter_ranges = {}
        for param in ['learning_rate', 'batch_size', 'weight_decay']:
            values = []
            for exp in top_experiences:
                if param in exp.hyperparameters:
                    values.append(exp.hyperparameters[param])
            
            if values and all(isinstance(v, (int, float)) for v in values):
                min_val = min(values) * 0.8
                max_val = max(values) * 1.2
                hyperparameter_ranges[param] = (min_val, max_val)
        
        # Create strategy
        strategy_id = f"domain_strategy_{domain}_{int(time.time())}"
        
        return OptimizationStrategy(
            strategy_id=strategy_id,
            name=f"domain_optimized_{domain}",
            hyperparameter_ranges=hyperparameter_ranges,
            adaptation_rules=[],  # Could be populated with domain-specific rules
            effectiveness_score=np.mean([exp.final_performance for exp in top_experiences]),
            applicability_domains=[domain],
            sample_efficiency=np.mean([
                1.0 / max(1.0, exp.convergence_time) for exp in top_experiences
            ]),
            convergence_reliability=np.mean([exp.adaptation_speed for exp in top_experiences])
        )
    
    async def _generate_meta_insights(self) -> List[Dict[str, Any]]:
        """Generate insights about the meta-learning system itself."""
        
        insights = []
        
        # Learning efficiency insight
        if len(self.experience_buffer) > 20:
            recent_experiences = list(self.experience_buffer)[-20:]
            early_experiences = list(self.experience_buffer)[:20]
            
            recent_avg = np.mean([exp.final_performance for exp in recent_experiences])
            early_avg = np.mean([exp.final_performance for exp in early_experiences])
            
            improvement = recent_avg - early_avg
            
            insights.append({
                'type': 'learning_efficiency',
                'insight': f"Meta-learning system has improved by {improvement:.3f} over time",
                'confidence': 0.8,
                'data_points': len(self.experience_buffer)
            })
        
        # Domain specialization insight
        domain_performances = defaultdict(list)
        for exp in self.experience_buffer:
            domain_performances[exp.domain].append(exp.final_performance)
        
        best_domain = max(domain_performances.keys(), 
                         key=lambda d: np.mean(domain_performances[d]))
        
        insights.append({
            'type': 'domain_specialization',
            'insight': f"System performs best in {best_domain} domain",
            'confidence': 0.7,
            'average_performance': np.mean(domain_performances[best_domain])
        })
        
        # Transfer effectiveness insight
        if self.transfer_matrix:
            all_transfer_scores = []
            for source in self.transfer_matrix:
                for target in self.transfer_matrix[source]:
                    all_transfer_scores.append(self.transfer_matrix[source][target])
            
            if all_transfer_scores:
                avg_transfer = np.mean(all_transfer_scores)
                insights.append({
                    'type': 'transfer_effectiveness',
                    'insight': f"Average cross-domain transfer effectiveness: {avg_transfer:.3f}",
                    'confidence': 0.6,
                    'transfer_connections': len(all_transfer_scores)
                })
        
        return insights
    
    def get_meta_learning_report(self) -> Dict[str, Any]:
        """Generate comprehensive meta-learning report."""
        
        return {
            'meta_learning_state': asdict(self.state),
            'learned_priors_count': len(self.learned_priors),
            'optimization_strategies_count': len(self.optimization_strategies),
            'experience_buffer_size': len(self.experience_buffer),
            'task_embeddings_count': len(self.task_embeddings),
            'transfer_matrix_size': sum(len(targets) for targets in self.transfer_matrix.values()),
            'current_meta_performance': self._calculate_current_meta_performance(),
            'top_domains': self._get_top_performing_domains(),
            'strategy_effectiveness': self._get_strategy_effectiveness_summary(),
            'recent_insights': self._get_recent_performance_insights()
        }
    
    def _get_top_performing_domains(self) -> List[Dict[str, Any]]:
        """Get top performing domains."""
        
        domain_stats = defaultdict(lambda: {'performances': [], 'count': 0})
        
        for exp in self.experience_buffer:
            domain_stats[exp.domain]['performances'].append(exp.final_performance)
            domain_stats[exp.domain]['count'] += 1
        
        # Calculate averages and sort
        domain_rankings = []
        for domain, stats in domain_stats.items():
            if stats['count'] >= 3:  # Only domains with sufficient data
                avg_performance = np.mean(stats['performances'])
                domain_rankings.append({
                    'domain': domain,
                    'average_performance': avg_performance,
                    'task_count': stats['count'],
                    'performance_std': np.std(stats['performances'])
                })
        
        return sorted(domain_rankings, key=lambda x: x['average_performance'], reverse=True)[:5]
    
    def _get_strategy_effectiveness_summary(self) -> List[Dict[str, Any]]:
        """Get strategy effectiveness summary."""
        
        return [
            {
                'strategy_name': strategy.name,
                'effectiveness_score': strategy.effectiveness_score,
                'sample_efficiency': strategy.sample_efficiency,
                'applicable_domains': strategy.applicability_domains,
                'hyperparameter_count': len(strategy.hyperparameter_ranges)
            }
            for strategy in sorted(self.optimization_strategies, 
                                 key=lambda s: s.effectiveness_score, reverse=True)
        ]
    
    def _get_recent_performance_insights(self) -> List[str]:
        """Get recent performance insights."""
        
        insights = []
        
        if len(self.experience_buffer) >= 10:
            recent_perf = [exp.final_performance for exp in list(self.experience_buffer)[-10:]]
            trend = np.mean(np.diff(recent_perf))
            
            if trend > 0.01:
                insights.append("📈 Recent performance trend is positive - system is improving")
            elif trend < -0.01:
                insights.append("📉 Recent performance trend is negative - may need adjustment")
            else:
                insights.append("📊 Recent performance is stable")
        
        # Adaptation speed insights
        if self.state.average_adaptation_speed > 0.15:
            insights.append("⚡ System shows fast adaptation to new tasks")
        elif self.state.average_adaptation_speed < 0.05:
            insights.append("🐌 System adaptation is slow - consider learning rate adjustments")
        
        return insights


# Supporting engines for meta-learning optimization

class MetaOptimizerEngine:
    """Meta-optimizer for optimizing the meta-learning process itself."""
    
    def __init__(self, parent: MetaLearningOptimizer):
        self.parent = parent
        self.meta_performance_history = deque(maxlen=100)
    
    async def optimize_meta_parameters(self) -> Dict[str, Any]:
        """Optimize meta-learning parameters themselves."""
        # Implementation for meta-parameter optimization
        # This would adjust things like learning rates for the meta-learner,
        # exploration rates, confidence thresholds, etc.
        pass


class FewShotLearningEngine:
    """Engine for few-shot learning capabilities."""
    
    def __init__(self, parent: MetaLearningOptimizer):
        self.parent = parent
        self.few_shot_experiences = deque(maxlen=1000)
    
    async def learn_from_few_examples(self, task_config: Dict[str, Any], 
                                    examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Learn from just a few examples using meta-learned priors."""
        # Implementation for few-shot learning
        pass


class DynamicArchitectureMorpher:
    """Engine for dynamically morphing architectures based on task characteristics."""
    
    def __init__(self, parent: MetaLearningOptimizer):
        self.parent = parent
        self.architecture_history = []
    
    async def morph_architecture(self, task_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Dynamically adjust architecture based on task characteristics."""
        # Implementation for dynamic architecture morphing
        pass


class CrossDomainTransferer:
    """Engine for cross-domain knowledge transfer."""
    
    def __init__(self, parent: MetaLearningOptimizer):
        self.parent = parent
        self.transfer_success_history = deque(maxlen=500)
    
    async def transfer_knowledge(self, source_domain: str, target_domain: str,
                               task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer knowledge from source domain to target domain."""
        # Implementation for cross-domain knowledge transfer
        pass


async def main():
    """Demonstrate Generation 6 Meta-Learning Optimizer."""
    print("🧠 Generation 6: Meta-Learning Optimization Engine")
    print("=" * 55)
    
    # Initialize meta-learning optimizer
    meta_optimizer = MetaLearningOptimizer()
    
    print(f"🚀 Meta-Learning Optimizer initialized")
    print(f"Initial state: {meta_optimizer.state}")
    print()
    
    # Simulate learning from multiple tasks
    print("📚 Simulating meta-learning from multiple tasks...")
    
    tasks = [
        {
            'task_id': 'vision_task_1',
            'domain': 'vision',
            'strategy': 'adam_with_cosine_lr',
            'hyperparameters': {'learning_rate': 0.001, 'batch_size': 64},
            'performance_data': [0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.87, 0.88, 0.89, 0.90]
        },
        {
            'task_id': 'nlp_task_1',
            'domain': 'nlp',
            'strategy': 'adamw_with_warmup',
            'hyperparameters': {'learning_rate': 0.0005, 'batch_size': 32, 'weight_decay': 0.01},
            'performance_data': [0.2, 0.4, 0.6, 0.75, 0.82, 0.86, 0.88, 0.89, 0.89, 0.90]
        },
        {
            'task_id': 'vision_task_2',
            'domain': 'vision',
            'strategy': 'sgd_with_momentum',
            'hyperparameters': {'learning_rate': 0.01, 'batch_size': 128, 'momentum': 0.9},
            'performance_data': [0.05, 0.15, 0.3, 0.5, 0.65, 0.75, 0.82, 0.85, 0.86, 0.87]
        },
        {
            'task_id': 'multimodal_task_1',
            'domain': 'multimodal',
            'strategy': 'adam_with_cosine_lr',
            'hyperparameters': {'learning_rate': 0.0008, 'batch_size': 48},
            'performance_data': [0.15, 0.35, 0.55, 0.70, 0.80, 0.85, 0.88, 0.90, 0.91, 0.92]
        }
    ]
    
    # Learn from each task
    for task in tasks:
        result = await meta_optimizer.learn_from_task(task, task['performance_data'])
        print(f"✅ Learned from {task['task_id']}: {len(result['meta_insights'])} insights")
    
    print(f"\n📊 Meta-learning progress:")
    print(f"  Tasks learned from: {meta_optimizer.state.total_tasks_learned}")
    print(f"  Optimization strategies: {meta_optimizer.state.optimization_strategies}")
    print(f"  Learned priors: {meta_optimizer.state.learned_priors}")
    print(f"  Average adaptation speed: {meta_optimizer.state.average_adaptation_speed:.3f}")
    
    # Test hyperparameter suggestions
    print("\n💡 Testing hyperparameter suggestions...")
    
    new_task_configs = [
        {'domain': 'vision', 'characteristics': {'complexity': 'moderate'}},
        {'domain': 'nlp', 'characteristics': {'complexity': 'high'}},
        {'domain': 'multimodal', 'characteristics': {'complexity': 'low'}}
    ]
    
    for config in new_task_configs:
        suggestions = await meta_optimizer.suggest_hyperparameters(config)
        print(f"📋 Suggestions for {config['domain']}:")
        for param, value in suggestions['suggestions'].items():
            confidence = suggestions['confidence_scores'].get(param, 0.0)
            print(f"    {param}: {value} (confidence: {confidence:.2f})")
    
    # Perform meta-optimization
    print("\n🎯 Performing meta-optimization...")
    meta_opt_results = await meta_optimizer.optimize_meta_strategy(performance_target=0.85)
    
    print(f"📈 Meta-optimization results:")
    print(f"  Initial performance: {meta_opt_results['initial_performance']:.3f}")
    print(f"  Final performance: {meta_opt_results['final_performance']:.3f}")
    print(f"  Optimization steps: {len(meta_opt_results['optimization_steps'])}")
    print(f"  New insights generated: {len(meta_opt_results['new_insights'])}")
    
    # Show insights
    if meta_opt_results['new_insights']:
        print("\n🔍 New meta-learning insights:")
        for i, insight in enumerate(meta_opt_results['new_insights'][:3]):
            print(f"  {i+1}. [{insight['type']}] {insight['insight']} (confidence: {insight['confidence']:.2f})")
    
    # Generate comprehensive report
    print("\n📋 Generating comprehensive meta-learning report...")
    report = meta_optimizer.get_meta_learning_report()
    
    print(f"\n📊 Meta-Learning System Report:")
    print(f"  Current meta-performance: {report['current_meta_performance']:.3f}")
    print(f"  Experience buffer size: {report['experience_buffer_size']}")
    print(f"  Transfer connections: {report['transfer_matrix_size']}")
    
    if report['top_domains']:
        print(f"\n🏆 Top performing domains:")
        for i, domain_info in enumerate(report['top_domains'][:3]):
            print(f"    {i+1}. {domain_info['domain']}: {domain_info['average_performance']:.3f} "
                  f"({domain_info['task_count']} tasks)")
    
    if report['recent_insights']:
        print(f"\n💭 Recent insights:")
        for insight in report['recent_insights']:
            print(f"    {insight}")
    
    print("\n✨ Generation 6 Meta-Learning Optimizer demonstration completed!")
    return report


if __name__ == "__main__":
    asyncio.run(main())