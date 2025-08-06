"""Quantum Annealing Optimizer for HPU Cluster Resource Allocation.

Implements simulated quantum annealing algorithms for solving complex
optimization problems in distributed HPU cluster environments.
"""

import asyncio
import logging
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor

from ..exceptions import QuantumOptimizationError, ValidationError
from ..validation import DataValidator

logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """Optimization objectives for quantum annealing."""
    MINIMIZE_ENERGY = "minimize_energy"
    MAXIMIZE_THROUGHPUT = "maximize_throughput" 
    MINIMIZE_LATENCY = "minimize_latency"
    BALANCE_RESOURCES = "balance_resources"
    MINIMIZE_COST = "minimize_cost"


@dataclass
class AnnealingParameter:
    """Parameters for quantum annealing process."""
    initial_temperature: float = 100.0
    final_temperature: float = 0.01
    cooling_schedule: str = "exponential"  # "exponential", "linear", "logarithmic"
    cooling_rate: float = 0.95
    max_iterations: int = 1000
    equilibrium_steps: int = 10  # Steps at each temperature
    quantum_tunneling_rate: float = 0.1
    acceptance_threshold: float = 1e-6


@dataclass
class OptimizationVariable:
    """Decision variable in optimization problem."""
    name: str
    value: float = 0.0
    min_bound: float = 0.0
    max_bound: float = 1.0
    is_discrete: bool = False
    discrete_values: Optional[List[float]] = None


@dataclass
class OptimizationConstraint:
    """Constraint in optimization problem."""
    name: str
    constraint_type: str  # "equality", "inequality_le", "inequality_ge"
    target_value: float
    weight: float = 1.0
    violation_penalty: float = 1000.0


class QuantumAnnealingOptimizer:
    """Quantum annealing optimizer for complex resource allocation problems."""
    
    def __init__(self, 
                 annealing_params: AnnealingParameter = None,
                 parallel_chains: int = 4,
                 enable_quantum_tunneling: bool = True):
        """Initialize quantum annealing optimizer.
        
        Args:
            annealing_params: Parameters for annealing process
            parallel_chains: Number of parallel annealing chains
            enable_quantum_tunneling: Enable quantum tunneling effects
        """
        self.annealing_params = annealing_params or AnnealingParameter()
        self.parallel_chains = parallel_chains
        self.enable_quantum_tunneling = enable_quantum_tunneling
        
        # Optimization problem definition
        self.variables: Dict[str, OptimizationVariable] = {}
        self.constraints: Dict[str, OptimizationConstraint] = {}
        self.objective_function: Optional[Callable] = None
        self.objective_type: OptimizationObjective = OptimizationObjective.MINIMIZE_ENERGY
        
        # Optimization state
        self.current_solution: Dict[str, float] = {}
        self.best_solution: Dict[str, float] = {}
        self.best_energy: float = float('inf')
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=parallel_chains)
        
        logger.info(f"Initialized QuantumAnnealingOptimizer with {parallel_chains} parallel chains")
    
    async def add_variable(self, 
                         name: str, 
                         initial_value: float = 0.0,
                         min_bound: float = 0.0, 
                         max_bound: float = 1.0,
                         is_discrete: bool = False,
                         discrete_values: List[float] = None) -> OptimizationVariable:
        """Add optimization variable."""
        
        # Validate inputs
        validator = DataValidator()
        if not validator.validate_string(name, min_length=1):
            raise ValidationError(f"Invalid variable name: {name}")
        
        if name in self.variables:
            raise ValidationError(f"Variable {name} already exists")
        
        if min_bound >= max_bound:
            raise ValidationError(f"Invalid bounds for {name}: min={min_bound}, max={max_bound}")
        
        # Create variable
        variable = OptimizationVariable(
            name=name,
            value=initial_value,
            min_bound=min_bound,
            max_bound=max_bound,
            is_discrete=is_discrete,
            discrete_values=discrete_values
        )
        
        # Validate initial value
        if not await self._is_valid_variable_value(variable, initial_value):
            raise ValidationError(f"Invalid initial value {initial_value} for variable {name}")
        
        self.variables[name] = variable
        self.current_solution[name] = initial_value
        
        logger.info(f"Added optimization variable {name} with bounds [{min_bound}, {max_bound}]")
        return variable
    
    async def add_constraint(self,
                           name: str,
                           constraint_function: Callable,
                           constraint_type: str = "inequality_le",
                           target_value: float = 0.0,
                           weight: float = 1.0,
                           violation_penalty: float = 1000.0) -> OptimizationConstraint:
        """Add optimization constraint."""
        
        if name in self.constraints:
            raise ValidationError(f"Constraint {name} already exists")
        
        if constraint_type not in ["equality", "inequality_le", "inequality_ge"]:
            raise ValidationError(f"Invalid constraint type: {constraint_type}")
        
        constraint = OptimizationConstraint(
            name=name,
            constraint_type=constraint_type,
            target_value=target_value,
            weight=weight,
            violation_penalty=violation_penalty
        )
        
        # Store constraint function separately (not in dataclass for serialization)
        if not hasattr(self, '_constraint_functions'):
            self._constraint_functions = {}
        self._constraint_functions[name] = constraint_function
        
        self.constraints[name] = constraint
        
        logger.info(f"Added constraint {name} of type {constraint_type}")
        return constraint
    
    async def set_objective_function(self, 
                                   objective_function: Callable,
                                   objective_type: OptimizationObjective = OptimizationObjective.MINIMIZE_ENERGY):
        """Set objective function for optimization."""
        
        self.objective_function = objective_function
        self.objective_type = objective_type
        
        logger.info(f"Set objective function with type {objective_type.value}")
    
    async def optimize(self, max_runtime_seconds: float = 300.0) -> Dict[str, Any]:
        """Run quantum annealing optimization."""
        
        if not self.variables:
            raise QuantumOptimizationError("No variables defined for optimization")
        
        if self.objective_function is None:
            raise QuantumOptimizationError("No objective function defined")
        
        logger.info("Starting quantum annealing optimization")
        start_time = time.time()
        
        # Initialize best solution
        self.best_solution = self.current_solution.copy()
        self.best_energy = await self._evaluate_solution(self.current_solution)
        
        # Run parallel annealing chains
        optimization_tasks = []
        for chain_id in range(self.parallel_chains):
            task = asyncio.create_task(
                self._run_annealing_chain(chain_id, max_runtime_seconds)
            )
            optimization_tasks.append(task)
        
        # Wait for all chains to complete
        chain_results = await asyncio.gather(*optimization_tasks)
        
        # Select best result across all chains
        for result in chain_results:
            if result["best_energy"] < self.best_energy:
                self.best_energy = result["best_energy"]
                self.best_solution = result["best_solution"]
        
        optimization_time = time.time() - start_time
        
        # Final evaluation
        final_energy = await self._evaluate_solution(self.best_solution)
        constraint_violations = await self._evaluate_constraints(self.best_solution)
        
        optimization_result = {
            "success": True,
            "best_solution": self.best_solution.copy(),
            "best_energy": final_energy,
            "constraint_violations": constraint_violations,
            "optimization_time": optimization_time,
            "total_iterations": sum(r["iterations"] for r in chain_results),
            "parallel_chains": self.parallel_chains,
            "annealing_params": self.annealing_params.__dict__,
            "convergence_history": self.optimization_history[-100:]  # Last 100 steps
        }
        
        logger.info(f"Quantum annealing completed in {optimization_time:.2f}s with energy {final_energy:.6f}")
        return optimization_result
    
    async def _run_annealing_chain(self, chain_id: int, max_runtime_seconds: float) -> Dict[str, Any]:
        """Run single annealing chain."""
        
        # Initialize chain with random perturbation
        chain_solution = {}
        for var_name, variable in self.variables.items():
            # Add small random perturbation to avoid identical starting points
            perturbation = np.random.normal(0, 0.1) * (variable.max_bound - variable.min_bound)
            value = self.current_solution[var_name] + perturbation
            value = np.clip(value, variable.min_bound, variable.max_bound)
            
            if variable.is_discrete and variable.discrete_values:
                # Round to nearest discrete value
                value = min(variable.discrete_values, key=lambda x: abs(x - value))
            
            chain_solution[var_name] = value
        
        chain_best_solution = chain_solution.copy()
        chain_best_energy = await self._evaluate_solution(chain_solution)
        
        current_temperature = self.annealing_params.initial_temperature
        iteration = 0
        chain_start_time = time.time()
        
        while (current_temperature > self.annealing_params.final_temperature and 
               iteration < self.annealing_params.max_iterations and
               time.time() - chain_start_time < max_runtime_seconds):
            
            # Equilibrium steps at current temperature
            for _ in range(self.annealing_params.equilibrium_steps):
                
                # Generate neighbor solution using quantum tunneling
                neighbor_solution = await self._generate_neighbor_solution(
                    chain_solution, current_temperature
                )
                
                neighbor_energy = await self._evaluate_solution(neighbor_solution)
                current_energy = await self._evaluate_solution(chain_solution)
                
                # Quantum annealing acceptance criterion
                if await self._accept_solution(current_energy, neighbor_energy, current_temperature):
                    chain_solution = neighbor_solution
                    
                    # Update chain best
                    if neighbor_energy < chain_best_energy:
                        chain_best_energy = neighbor_energy
                        chain_best_solution = neighbor_solution.copy()
                        
                        # Store optimization step
                        self.optimization_history.append({
                            "iteration": iteration,
                            "chain_id": chain_id,
                            "temperature": current_temperature,
                            "energy": neighbor_energy,
                            "solution": neighbor_solution.copy()
                        })
            
            # Cool down temperature
            current_temperature = await self._update_temperature(
                current_temperature, iteration
            )
            iteration += 1
        
        return {
            "chain_id": chain_id,
            "best_solution": chain_best_solution,
            "best_energy": chain_best_energy,
            "iterations": iteration,
            "final_temperature": current_temperature
        }
    
    async def _generate_neighbor_solution(self, 
                                        current_solution: Dict[str, float], 
                                        temperature: float) -> Dict[str, float]:
        """Generate neighbor solution using quantum tunneling effects."""
        
        neighbor = current_solution.copy()
        
        # Select random variable to modify
        var_names = list(self.variables.keys())
        if not var_names:
            return neighbor
        
        selected_var = np.random.choice(var_names)
        variable = self.variables[selected_var]
        
        current_value = current_solution[selected_var]
        
        if self.enable_quantum_tunneling:
            # Quantum tunneling: allow larger jumps at high temperature
            tunneling_factor = self.annealing_params.quantum_tunneling_rate * temperature / self.annealing_params.initial_temperature
            jump_range = (variable.max_bound - variable.min_bound) * tunneling_factor
        else:
            # Classical annealing: smaller local changes
            jump_range = (variable.max_bound - variable.min_bound) * 0.1
        
        # Generate random perturbation
        perturbation = np.random.normal(0, jump_range)
        new_value = current_value + perturbation
        
        # Apply bounds
        new_value = np.clip(new_value, variable.min_bound, variable.max_bound)
        
        # Handle discrete variables
        if variable.is_discrete and variable.discrete_values:
            new_value = min(variable.discrete_values, key=lambda x: abs(x - new_value))
        
        neighbor[selected_var] = new_value
        
        return neighbor
    
    async def _accept_solution(self, 
                             current_energy: float, 
                             neighbor_energy: float, 
                             temperature: float) -> bool:
        """Quantum annealing acceptance criterion."""
        
        # Always accept better solutions
        if self.objective_type == OptimizationObjective.MINIMIZE_ENERGY:
            energy_delta = neighbor_energy - current_energy
        else:
            # For maximization objectives, flip the sign
            energy_delta = current_energy - neighbor_energy
        
        if energy_delta <= 0:
            return True
        
        # Quantum tunneling acceptance probability
        if temperature > 0:
            acceptance_prob = math.exp(-energy_delta / temperature)
            return np.random.random() < acceptance_prob
        
        return False
    
    async def _update_temperature(self, current_temp: float, iteration: int) -> float:
        """Update temperature according to cooling schedule."""
        
        if self.annealing_params.cooling_schedule == "exponential":
            return current_temp * self.annealing_params.cooling_rate
        
        elif self.annealing_params.cooling_schedule == "linear":
            temp_range = self.annealing_params.initial_temperature - self.annealing_params.final_temperature
            progress = iteration / self.annealing_params.max_iterations
            return self.annealing_params.initial_temperature - progress * temp_range
        
        elif self.annealing_params.cooling_schedule == "logarithmic":
            return self.annealing_params.initial_temperature / math.log(iteration + 2)
        
        else:
            # Default to exponential
            return current_temp * self.annealing_params.cooling_rate
    
    async def _evaluate_solution(self, solution: Dict[str, float]) -> float:
        """Evaluate objective function and constraints for a solution."""
        
        try:
            # Evaluate objective function
            if self.objective_function:
                objective_value = await self._safe_function_call(self.objective_function, solution)
            else:
                objective_value = 0.0
            
            # Evaluate constraints and add penalties
            constraint_penalty = 0.0
            
            for constraint_name, constraint in self.constraints.items():
                if constraint_name in getattr(self, '_constraint_functions', {}):
                    constraint_func = self._constraint_functions[constraint_name]
                    constraint_value = await self._safe_function_call(constraint_func, solution)
                    
                    # Calculate constraint violation
                    violation = 0.0
                    
                    if constraint.constraint_type == "equality":
                        violation = abs(constraint_value - constraint.target_value)
                    elif constraint.constraint_type == "inequality_le":
                        violation = max(0, constraint_value - constraint.target_value)
                    elif constraint.constraint_type == "inequality_ge":
                        violation = max(0, constraint.target_value - constraint_value)
                    
                    # Add penalty for constraint violation
                    if violation > 0:
                        constraint_penalty += constraint.weight * constraint.violation_penalty * violation
            
            total_energy = objective_value + constraint_penalty
            
            return total_energy
            
        except Exception as e:
            logger.error(f"Error evaluating solution: {e}")
            # Return high penalty for invalid solutions
            return float('inf')
    
    async def _safe_function_call(self, func: Callable, solution: Dict[str, float]) -> float:
        """Safely call objective/constraint function."""
        
        try:
            # Handle both sync and async functions
            if asyncio.iscoroutinefunction(func):
                result = await func(solution)
            else:
                result = func(solution)
            
            # Validate result
            if not isinstance(result, (int, float)) or math.isnan(result):
                logger.warning(f"Invalid function result: {result}")
                return float('inf')
            
            return float(result)
            
        except Exception as e:
            logger.error(f"Function call failed: {e}")
            return float('inf')
    
    async def _evaluate_constraints(self, solution: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Evaluate all constraints for a solution."""
        
        constraint_results = {}
        
        for constraint_name, constraint in self.constraints.items():
            if constraint_name in getattr(self, '_constraint_functions', {}):
                try:
                    constraint_func = self._constraint_functions[constraint_name]
                    constraint_value = await self._safe_function_call(constraint_func, solution)
                    
                    # Check satisfaction
                    satisfied = False
                    violation = 0.0
                    
                    if constraint.constraint_type == "equality":
                        violation = abs(constraint_value - constraint.target_value)
                        satisfied = violation < self.annealing_params.acceptance_threshold
                    elif constraint.constraint_type == "inequality_le":
                        violation = max(0, constraint_value - constraint.target_value)
                        satisfied = constraint_value <= constraint.target_value + self.annealing_params.acceptance_threshold
                    elif constraint.constraint_type == "inequality_ge":
                        violation = max(0, constraint.target_value - constraint_value)
                        satisfied = constraint_value >= constraint.target_value - self.annealing_params.acceptance_threshold
                    
                    constraint_results[constraint_name] = {
                        "value": constraint_value,
                        "target": constraint.target_value,
                        "type": constraint.constraint_type,
                        "satisfied": satisfied,
                        "violation": violation
                    }
                    
                except Exception as e:
                    logger.error(f"Error evaluating constraint {constraint_name}: {e}")
                    constraint_results[constraint_name] = {
                        "error": str(e),
                        "satisfied": False,
                        "violation": float('inf')
                    }
        
        return constraint_results
    
    async def _is_valid_variable_value(self, variable: OptimizationVariable, value: float) -> bool:
        """Check if value is valid for variable."""
        
        # Check bounds
        if value < variable.min_bound or value > variable.max_bound:
            return False
        
        # Check discrete values
        if variable.is_discrete and variable.discrete_values:
            if value not in variable.discrete_values:
                return False
        
        return True
    
    async def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get optimization performance metrics."""
        
        if not self.optimization_history:
            return {"no_optimization_run": True}
        
        # Calculate convergence statistics
        energies = [step["energy"] for step in self.optimization_history]
        
        return {
            "total_steps": len(self.optimization_history),
            "best_energy": self.best_energy,
            "energy_improvement": energies[0] - self.best_energy if energies else 0.0,
            "convergence_rate": len([e for e in energies if e <= self.best_energy * 1.01]) / len(energies),
            "average_energy": np.mean(energies) if energies else 0.0,
            "energy_std": np.std(energies) if energies else 0.0,
            "optimization_variables": len(self.variables),
            "optimization_constraints": len(self.constraints),
            "parallel_chains": self.parallel_chains,
            "quantum_tunneling_enabled": self.enable_quantum_tunneling
        }
    
    async def reset_optimizer(self):
        """Reset optimizer state."""
        
        self.current_solution.clear()
        self.best_solution.clear()
        self.best_energy = float('inf')
        self.optimization_history.clear()
        
        logger.info("Reset quantum annealing optimizer")
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)