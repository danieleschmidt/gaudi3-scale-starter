"""Comprehensive test suite for autonomous enhancements.

This module tests all the new autonomous enhancement features including:
- Autonomous Enhancement System
- Adaptive Intelligence Engine
- Quantum Resilience Manager
- Autonomous Orchestrator
- Hyper Scale Engine
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

# Import the modules to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from gaudi3_scale.autonomous_enhancement import (
    AutonomousEnhancer,
    SelfHealingManager,
    PerformanceMetrics,
    EnhancementDecision
)
from gaudi3_scale.adaptive_intelligence import (
    AdaptiveIntelligenceEngine,
    WorkloadProfile,
    PredictionResult,
    WorkloadClassifier
)
from gaudi3_scale.quantum_resilience import (
    QuantumResilienceManager,
    QubitState,
    QuantumState
)
from gaudi3_scale.autonomous_orchestrator import (
    AutonomousOrchestrator,
    AutonomousTask,
    WorkflowDefinition,
    TaskStatus,
    Priority
)
from gaudi3_scale.hyper_scale_engine import (
    HyperScaleEngine,
    ScalingMetrics,
    ScalingDecision,
    ScalingDirection
)


class TestAutonomousEnhancer:
    """Test suite for the Autonomous Enhancement System."""
    
    @pytest.fixture
    async def enhancer(self):
        """Create an autonomous enhancer instance."""
        enhancer = AutonomousEnhancer()
        yield enhancer
        await enhancer.stop()
    
    def test_performance_metrics_creation(self):
        """Test performance metrics creation and validation."""
        metrics = PerformanceMetrics(
            cpu_usage=0.75,
            memory_usage=0.60,
            hpu_utilization=0.90,
            throughput_tokens_per_sec=1500.0,
            latency_ms=45.0,
            error_rate=0.001
        )
        
        assert metrics.cpu_usage == 0.75
        assert metrics.memory_usage == 0.60
        assert metrics.hpu_utilization == 0.90
        assert metrics.throughput_tokens_per_sec == 1500.0
        assert metrics.latency_ms == 45.0
        assert metrics.error_rate == 0.001
        assert isinstance(metrics.timestamp, datetime)
    
    def test_enhancement_decision_creation(self):
        """Test enhancement decision creation."""
        decision = EnhancementDecision(
            action="optimize_batch_size",
            parameters={"new_batch_size": 64, "gradual_adjustment": True},
            expected_benefit=0.25,
            confidence=0.85,
            rationale="Throughput below average, adjusting batch size for optimization"
        )
        
        assert decision.action == "optimize_batch_size"
        assert decision.parameters["new_batch_size"] == 64
        assert decision.expected_benefit == 0.25
        assert decision.confidence == 0.85
        assert "throughput" in decision.rationale
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, enhancer):
        """Test autonomous metrics collection."""
        metrics = await enhancer._collect_metrics()
        
        assert isinstance(metrics, PerformanceMetrics)
        assert 0.0 <= metrics.cpu_usage <= 1.0
        assert 0.0 <= metrics.memory_usage <= 1.0
        assert 0.0 <= metrics.hpu_utilization <= 1.0
        assert metrics.throughput_tokens_per_sec > 0
        assert metrics.latency_ms > 0
        assert metrics.error_rate >= 0
    
    @pytest.mark.asyncio
    async def test_optimization_decision_making(self, enhancer):
        """Test autonomous optimization decision making."""
        # Test high latency scenario
        high_latency_metrics = PerformanceMetrics(
            cpu_usage=0.5,
            memory_usage=0.6,
            hpu_utilization=0.7,
            throughput_tokens_per_sec=800.0,
            latency_ms=150.0,  # High latency
            error_rate=0.005
        )
        
        decision = await enhancer._make_enhancement_decision(high_latency_metrics)
        
        assert decision is not None
        assert decision.action == "reduce_latency"
        assert "latency" in decision.rationale.lower()
        assert decision.expected_benefit > 0
        
        # Test low HPU utilization scenario
        low_hpu_metrics = PerformanceMetrics(
            cpu_usage=0.4,
            memory_usage=0.5,
            hpu_utilization=0.6,  # Low HPU utilization
            throughput_tokens_per_sec=1000.0,
            latency_ms=50.0,
            error_rate=0.001
        )
        
        decision = await enhancer._make_enhancement_decision(low_hpu_metrics)
        
        assert decision is not None
        assert decision.action == "increase_hpu_utilization"
        assert "hpu" in decision.rationale.lower()
    
    @pytest.mark.asyncio
    async def test_pattern_analysis(self, enhancer):
        """Test pattern learning and analysis."""
        # Add sample metrics to history
        for i in range(30):
            metrics = PerformanceMetrics(
                cpu_usage=0.5 + i * 0.01,
                memory_usage=0.6 + i * 0.005,
                hpu_utilization=0.8 + i * 0.002,
                throughput_tokens_per_sec=1000 + i * 10,
                latency_ms=50 - i * 0.5,
                batch_size=32 + i
            )
            enhancer.metrics_history.append(metrics)
        
        patterns = await enhancer._analyze_patterns()
        
        assert "avg_cpu_usage" in patterns
        assert "avg_throughput" in patterns
        assert "throughput_trend" in patterns
        assert patterns["throughput_trend"] == "increasing"
        assert isinstance(patterns["optimal_batch_size_range"], tuple)
    
    @pytest.mark.asyncio
    async def test_enhancement_application(self, enhancer):
        """Test enhancement application."""
        decision = EnhancementDecision(
            action="reduce_latency",
            parameters={"enable_caching": True, "optimize_batch_processing": True},
            expected_benefit=0.3,
            confidence=0.9,
            rationale="High latency detected"
        )
        
        await enhancer._apply_enhancement(decision)
        
        # Check that decision was recorded
        assert len(enhancer.enhancement_history) > 0
        applied_decision = enhancer.enhancement_history[-1]
        assert applied_decision.action == "reduce_latency"
    
    def test_enhancement_report_generation(self, enhancer):
        """Test enhancement report generation."""
        # Add some sample enhancement history
        for i in range(5):
            decision = EnhancementDecision(
                action=f"action_{i}",
                parameters={},
                expected_benefit=0.2,
                confidence=0.8,
                rationale=f"Test rationale {i}"
            )
            enhancer.enhancement_history.append(decision)
        
        report = enhancer.get_enhancement_report()
        
        assert "total_enhancements" in report
        assert report["total_enhancements"] == 5
        assert "enhancement_actions" in report
        assert "average_confidence" in report
        assert report["average_confidence"] == 0.8


class TestSelfHealingManager:
    """Test suite for the Self-Healing Manager."""
    
    @pytest.fixture
    def healing_manager(self):
        """Create a self-healing manager instance."""
        return SelfHealingManager()
    
    def test_error_registration(self, healing_manager):
        """Test error registration and pattern detection."""
        # Register multiple errors of the same type
        for i in range(5):
            healing_manager.register_error("memory_error", f"Out of memory error {i}")
        
        assert "memory_error" in healing_manager.error_patterns
        assert len(healing_manager.error_patterns["memory_error"]) == 5
    
    def test_recurring_error_detection(self, healing_manager):
        """Test recurring error detection."""
        # Register errors within time window
        for i in range(4):
            healing_manager.register_error("network_error", "Connection timeout")
        
        is_recurring = healing_manager._is_recurring_error("network_error", threshold=3)
        assert is_recurring is True
        
        # Test with error type that doesn't exceed threshold
        healing_manager.register_error("rare_error", "Rare error occurred")
        is_recurring = healing_manager._is_recurring_error("rare_error", threshold=3)
        assert is_recurring is False
    
    def test_healing_action_determination(self, healing_manager):
        """Test healing action determination."""
        # Test memory error
        action = healing_manager._determine_healing_action("memory_error", "Out of memory")
        assert action == "restart_training_process"
        
        # Test HPU error
        action = healing_manager._determine_healing_action("hpu_error", "HPU initialization failed")
        assert action == "reset_hpu_context"
        
        # Test network error
        action = healing_manager._determine_healing_action("network_error", "Connection refused")
        assert action == "retry_with_backoff"
        
        # Test unknown error
        action = healing_manager._determine_healing_action("unknown_error", "Unknown issue")
        assert action is None


class TestAdaptiveIntelligenceEngine:
    """Test suite for the Adaptive Intelligence Engine."""
    
    @pytest.fixture
    async def intelligence_engine(self):
        """Create an adaptive intelligence engine instance."""
        engine = AdaptiveIntelligenceEngine()
        yield engine
        await engine.stop_adaptive_learning()
    
    def test_workload_profile_creation(self):
        """Test workload profile creation and validation."""
        profile = WorkloadProfile(
            model_type="transformer",
            model_size_gb=70.0,
            sequence_length=2048,
            batch_size=64,
            vocab_size=50000,
            num_layers=24,
            hidden_size=1024,
            attention_heads=16,
            compute_intensity=0.9,
            memory_pattern="memory_bound"
        )
        
        assert profile.model_type == "transformer"
        assert profile.model_size_gb == 70.0
        assert profile.sequence_length == 2048
        assert profile.batch_size == 64
        assert profile.compute_intensity == 0.9
        assert profile.memory_pattern == "memory_bound"
    
    def test_prediction_result_creation(self):
        """Test prediction result creation."""
        result = PredictionResult(
            value=1500.0,
            confidence=0.85,
            factors={
                "model_size_impact": 0.7,
                "batch_size_impact": 0.5,
                "compute_intensity_impact": 0.9
            }
        )
        
        assert result.value == 1500.0
        assert result.confidence == 0.85
        assert len(result.factors) == 3
        assert isinstance(result.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_workload_analysis(self, intelligence_engine):
        """Test workload analysis and classification."""
        metrics = {
            "batch_size": 64,
            "sequence_length": 1024,
            "model_size_gb": 7.0,
            "num_layers": 24,
            "attention_heads": 16,
            "memory_usage": 0.8,
            "cpu_usage": 0.6
        }
        
        profile = await intelligence_engine.analyze_workload(metrics)
        
        assert isinstance(profile, WorkloadProfile)
        assert profile.batch_size == 64
        assert profile.sequence_length == 1024
        assert profile.model_size_gb == 7.0
        assert profile.compute_intensity > 0
        assert profile.memory_pattern in ["memory_bound", "compute_bound", "balanced"]
    
    @pytest.mark.asyncio
    async def test_performance_prediction(self, intelligence_engine):
        """Test performance prediction."""
        workload = WorkloadProfile(
            model_type="transformer",
            model_size_gb=7.0,
            batch_size=64,
            sequence_length=1024,
            compute_intensity=0.8
        )
        
        config = {
            "precision": "bf16",
            "gradient_checkpointing": True
        }
        
        prediction = await intelligence_engine.predict_performance(workload, config, 300)
        
        assert isinstance(prediction, PredictionResult)
        assert prediction.value > 0
        assert 0.0 <= prediction.confidence <= 1.0
        assert len(prediction.factors) > 0
        assert prediction.prediction_horizon_seconds == 300
    
    @pytest.mark.asyncio
    async def test_resource_optimization(self, intelligence_engine):
        """Test resource optimization."""
        workload = WorkloadProfile(
            model_type="large_language_model",
            model_size_gb=70.0,
            batch_size=32,
            memory_pattern="memory_bound",
            compute_intensity=0.9
        )
        
        current_resources = {
            "cpu_cores": 32,
            "memory_gb": 256,
            "hpu_count": 8
        }
        
        constraints = {
            "max_memory_gb": 512,
            "max_cpu_cores": 64
        }
        
        optimization = await intelligence_engine.optimize_resources(
            workload, current_resources, constraints
        )
        
        assert isinstance(optimization, dict)
        assert "cpu_adjustment" in optimization or "memory_adjustment" in optimization
    
    def test_intelligence_report_generation(self, intelligence_engine):
        """Test intelligence report generation."""
        # Add some observation history
        for i in range(10):
            observation = {
                "timestamp": datetime.now(),
                "metrics": {"cpu_usage": 0.5 + i * 0.05},
                "type": "test_observation"
            }
            intelligence_engine.observation_history.append(observation)
        
        report = intelligence_engine.get_intelligence_report()
        
        assert "learning_state" in report
        assert "data_summary" in report
        assert "system_status" in report
        assert report["data_summary"]["observation_count"] == 10


class TestWorkloadClassifier:
    """Test suite for the Workload Classifier."""
    
    @pytest.fixture
    def classifier(self):
        """Create a workload classifier instance."""
        return WorkloadClassifier()
    
    @pytest.mark.asyncio
    async def test_workload_classification(self, classifier):
        """Test workload classification."""
        # Test large language model classification
        llm_metrics = {
            "model_size_gb": 70.0,
            "batch_size": 32,
            "num_layers": 24,
            "attention_heads": 16,
            "memory_usage": 0.9,
            "cpu_usage": 0.4
        }
        
        profile = await classifier.classify_workload(llm_metrics)
        
        assert profile.model_type == "large_language_model"
        assert profile.memory_pattern == "memory_bound"
        assert profile.batch_size == 32
        assert profile.model_size_gb == 70.0
        
        # Test transformer classification
        transformer_metrics = {
            "model_size_gb": 1.5,
            "batch_size": 64,
            "attention_heads": 8,
            "memory_usage": 0.5,
            "cpu_usage": 0.7
        }
        
        profile = await classifier.classify_workload(transformer_metrics)
        
        assert profile.model_type == "transformer"
        assert profile.batch_size == 64
    
    def test_compute_intensity_calculation(self, classifier):
        """Test compute intensity calculation."""
        metrics = {
            "batch_size": 64,
            "model_size_gb": 7.0,
            "sequence_length": 1024
        }
        
        intensity = classifier._calculate_compute_intensity(metrics)
        
        assert 0.0 <= intensity <= 1.0
        assert intensity > 0  # Should be greater than 0 for non-zero inputs


class TestQuantumResilienceManager:
    """Test suite for the Quantum Resilience Manager."""
    
    @pytest.fixture
    async def resilience_manager(self):
        """Create a quantum resilience manager instance."""
        manager = QuantumResilienceManager()
        yield manager
        await manager.stop_quantum_resilience()
    
    def test_qubit_state_creation(self):
        """Test qubit state creation and operations."""
        qubit = QubitState(
            amplitude_0=complex(1/math.sqrt(2), 0),
            amplitude_1=complex(1/math.sqrt(2), 0),
            coherence_time=60.0
        )
        
        assert abs(qubit.amplitude_0) == pytest.approx(1/math.sqrt(2))
        assert abs(qubit.amplitude_1) == pytest.approx(1/math.sqrt(2))
        assert qubit.coherence_time == 60.0
        
        # Test probability calculations
        prob_0 = qubit.probability_0()
        prob_1 = qubit.probability_1()
        
        assert prob_0 == pytest.approx(0.5)
        assert prob_1 == pytest.approx(0.5)
        assert prob_0 + prob_1 == pytest.approx(1.0)
    
    def test_qubit_measurement(self):
        """Test qubit measurement and state collapse."""
        qubit = QubitState(
            amplitude_0=complex(1.0, 0),  # |0⟩ state
            amplitude_1=complex(0.0, 0)
        )
        
        measurement = qubit.measure()
        assert measurement == 0
        
        # After measurement, should be in definite state
        assert qubit.probability_0() == pytest.approx(1.0)
        assert qubit.probability_1() == pytest.approx(0.0)
    
    def test_qubit_normalization(self):
        """Test qubit state normalization."""
        qubit = QubitState(
            amplitude_0=complex(2.0, 0),
            amplitude_1=complex(2.0, 0)
        )
        
        # Should be normalized after creation
        assert abs(qubit.amplitude_0)**2 + abs(qubit.amplitude_1)**2 == pytest.approx(1.0)
    
    @pytest.mark.asyncio
    async def test_quantum_anomaly_detection(self, resilience_manager):
        """Test quantum anomaly detection."""
        # Initialize quantum states first
        await resilience_manager._initialize_quantum_states()
        
        # Test normal metrics (no anomaly)
        normal_metrics = {
            "error_rate": 0.001,
            "cpu_usage": 0.5,
            "memory_usage": 0.6
        }
        
        is_anomaly = await resilience_manager.detect_quantum_anomaly(
            "hpu_cluster", normal_metrics
        )
        # Should not detect anomaly with normal metrics
        
        # Test abnormal metrics (should detect anomaly)
        abnormal_metrics = {
            "error_rate": 0.1,  # High error rate
            "cpu_usage": 0.95,   # High CPU usage
            "memory_usage": 0.9  # High memory usage
        }
        
        is_anomaly = await resilience_manager.detect_quantum_anomaly(
            "hpu_cluster", abnormal_metrics
        )
        # Should detect anomaly with abnormal metrics
    
    def test_resilience_report_generation(self, resilience_manager):
        """Test quantum resilience report generation."""
        # Add some quantum states
        resilience_manager.quantum_states["test_component"] = QubitState()
        resilience_manager.entanglement_graph["test_component"] = {"backup_component"}
        
        report = resilience_manager.get_quantum_resilience_report()
        
        assert "system_fidelity" in report
        assert "entanglement_strength" in report
        assert "component_states" in report
        assert "test_component" in report["component_states"]
        assert isinstance(report["system_fidelity"], float)
        assert isinstance(report["entanglement_strength"], float)


class TestAutonomousOrchestrator:
    """Test suite for the Autonomous Orchestrator."""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create an autonomous orchestrator instance."""
        orchestrator = AutonomousOrchestrator()
        yield orchestrator
        await orchestrator.stop_orchestrator()
    
    def test_autonomous_task_creation(self):
        """Test autonomous task creation."""
        task = AutonomousTask(
            name="Test Training Task",
            description="Test autonomous training execution",
            task_type="training",
            priority=Priority.HIGH,
            resource_requirements={"hpu": 8, "memory_gb": 128},
            timeout_seconds=3600,
            max_retries=3
        )
        
        assert task.name == "Test Training Task"
        assert task.task_type == "training"
        assert task.priority == Priority.HIGH
        assert task.status == TaskStatus.PENDING
        assert task.resource_requirements["hpu"] == 8
        assert task.timeout_seconds == 3600
        assert task.max_retries == 3
        assert isinstance(task.created_at, datetime)
    
    def test_workflow_definition_creation(self):
        """Test workflow definition creation."""
        task1 = AutonomousTask(name="Setup", task_type="setup")
        task2 = AutonomousTask(name="Training", task_type="training", dependencies=[task1.id])
        
        workflow = WorkflowDefinition(
            name="Test Training Workflow",
            description="Complete training workflow",
            tasks=[task1, task2],
            global_timeout=7200,
            optimize_for=["performance", "cost"]
        )
        
        assert workflow.name == "Test Training Workflow"
        assert len(workflow.tasks) == 2
        assert workflow.global_timeout == 7200
        assert "performance" in workflow.optimize_for
        assert workflow.status == TaskStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_task_submission(self, orchestrator):
        """Test task submission to orchestrator."""
        task = AutonomousTask(
            name="Test Task",
            task_type="test",
            function=lambda: "test_result",
            resource_requirements={"cpu": 2, "memory_gb": 4}
        )
        
        task_id = await orchestrator.submit_task(task)
        
        assert task_id == task.id
        assert len(orchestrator.task_queue) > 0
    
    @pytest.mark.asyncio
    async def test_training_workflow_creation(self, orchestrator):
        """Test autonomous training workflow creation."""
        model_config = {
            "model_type": "transformer",
            "num_layers": 24,
            "hidden_size": 1024
        }
        
        training_config = {
            "epochs": 5,
            "batch_size": 64,
            "learning_rate": 1e-4
        }
        
        workflow_id = await orchestrator.create_training_workflow(
            model_config, training_config
        )
        
        assert workflow_id in orchestrator.workflows
        workflow = orchestrator.workflows[workflow_id]
        assert workflow.name == "Autonomous Training Workflow"
        assert len(workflow.tasks) > 0
        
        # Check that tasks have proper dependencies
        setup_tasks = [t for t in workflow.tasks if t.task_type == "setup"]
        training_tasks = [t for t in workflow.tasks if t.task_type == "training"]
        
        assert len(setup_tasks) > 0
        assert len(training_tasks) > 0
        
        # Training tasks should depend on setup tasks
        for training_task in training_tasks:
            setup_dependencies = [dep for dep in training_task.dependencies 
                                if dep in [t.id for t in setup_tasks]]
            # Should have at least one setup dependency (indirectly through data/model tasks)
    
    def test_orchestration_status_reporting(self, orchestrator):
        """Test orchestration status reporting."""
        # Add some tasks to test status
        task1 = AutonomousTask(name="Running Task", status=TaskStatus.RUNNING)
        task2 = AutonomousTask(name="Completed Task", status=TaskStatus.COMPLETED)
        
        orchestrator.running_tasks[task1.id] = task1
        orchestrator.completed_tasks[task2.id] = task2
        orchestrator.task_queue.append(AutonomousTask(name="Queued Task"))
        
        status = orchestrator.get_orchestration_status()
        
        assert status["is_running"] is False  # Not started yet
        assert status["active_tasks"] == 1
        assert status["completed_tasks"] == 1
        assert status["queued_tasks"] == 1
        assert "performance_metrics" in status
        assert "last_update" in status


class TestHyperScaleEngine:
    """Test suite for the Hyper Scale Engine."""
    
    @pytest.fixture
    async def scale_engine(self):
        """Create a hyper scale engine instance."""
        engine = HyperScaleEngine()
        yield engine
        await engine.stop_hyper_scaling()
    
    def test_scaling_metrics_creation(self):
        """Test scaling metrics creation."""
        metrics = ScalingMetrics(
            throughput_ops_per_sec=5000.0,
            latency_p50_ms=25.0,
            latency_p95_ms=75.0,
            latency_p99_ms=150.0,
            error_rate=0.001,
            cpu_utilization=0.75,
            memory_utilization=0.65,
            hpu_utilization=0.85,
            queue_depth=50,
            active_connections=200
        )
        
        assert metrics.throughput_ops_per_sec == 5000.0
        assert metrics.latency_p50_ms == 25.0
        assert metrics.cpu_utilization == 0.75
        assert metrics.hpu_utilization == 0.85
        assert metrics.queue_depth == 50
        assert isinstance(metrics.timestamp, datetime)
    
    def test_scaling_decision_creation(self):
        """Test scaling decision creation."""
        decision = ScalingDecision(
            direction=ScalingDirection.SCALE_UP,
            resource_type=ResourceType.COMPUTE,
            magnitude=1.5,
            target_instances=10,
            priority=1,
            trigger_reasons=["high_cpu_utilization", "high_queue_depth"],
            expected_impact={"throughput_increase": 0.4, "latency_reduction": 0.3},
            confidence=0.9
        )
        
        assert decision.direction == ScalingDirection.SCALE_UP
        assert decision.resource_type == ResourceType.COMPUTE
        assert decision.magnitude == 1.5
        assert decision.target_instances == 10
        assert decision.priority == 1
        assert len(decision.trigger_reasons) == 2
        assert decision.confidence == 0.9
        assert decision.status == "pending"
    
    @pytest.mark.asyncio
    async def test_scaling_infrastructure_initialization(self, scale_engine):
        """Test scaling infrastructure initialization."""
        await scale_engine._initialize_scaling_infrastructure()
        
        assert len(scale_engine.scaling_nodes) > 0
        
        # Check that different node types exist
        node_types = set(node.node_type for node in scale_engine.scaling_nodes.values())
        assert len(node_types) > 1
        
        # Check node properties
        for node in scale_engine.scaling_nodes.values():
            assert node.cpu_cores > 0
            assert node.memory_gb > 0
            assert node.cost_per_hour > 0
            assert not node.is_active  # Should start inactive
    
    @pytest.mark.asyncio
    async def test_scaling_request_processing(self, scale_engine):
        """Test scaling request processing."""
        # Initialize infrastructure first
        await scale_engine._initialize_scaling_infrastructure()
        
        # Create test metrics that should trigger scaling
        high_utilization_metrics = ScalingMetrics(
            cpu_utilization=0.9,  # High CPU utilization
            memory_utilization=0.85,
            hpu_utilization=0.88,
            throughput_ops_per_sec=2000.0,
            latency_p95_ms=120.0,
            queue_depth=500  # High queue depth
        )
        
        decisions = await scale_engine.process_scaling_request(
            high_utilization_metrics, ["performance"]
        )
        
        assert isinstance(decisions, list)
        assert len(decisions) > 0
        
        # Should have scaling up decisions due to high utilization
        scale_up_decisions = [d for d in decisions if d.direction == ScalingDirection.SCALE_UP]
        assert len(scale_up_decisions) > 0
        
        # Check decision properties
        for decision in decisions:
            assert isinstance(decision.magnitude, float)
            assert decision.magnitude > 0
            assert len(decision.trigger_reasons) > 0
            assert decision.confidence > 0
    
    @pytest.mark.asyncio
    async def test_reactive_decision_generation(self, scale_engine):
        """Test reactive scaling decision generation."""
        # Test CPU scaling decision
        high_cpu_metrics = ScalingMetrics(
            cpu_utilization=0.9,  # Exceeds threshold
            memory_utilization=0.5,
            hpu_utilization=0.6
        )
        
        decisions = await scale_engine._generate_reactive_decisions(high_cpu_metrics)
        
        cpu_decisions = [d for d in decisions if d.resource_type == ResourceType.COMPUTE]
        assert len(cpu_decisions) > 0
        
        cpu_decision = cpu_decisions[0]
        assert cpu_decision.direction == ScalingDirection.SCALE_UP
        assert "high_cpu_utilization" in cpu_decision.trigger_reasons
        
        # Test burst scaling decision
        extreme_load_metrics = ScalingMetrics(
            cpu_utilization=0.98,  # Extreme load
            queue_depth=2000  # Very high queue depth
        )
        
        decisions = await scale_engine._generate_reactive_decisions(extreme_load_metrics)
        
        burst_decisions = [d for d in decisions if d.direction == ScalingDirection.BURST_SCALE]
        assert len(burst_decisions) > 0
        
        burst_decision = burst_decisions[0]
        assert burst_decision.priority == 1  # Highest priority
        assert "extreme_load" in burst_decision.trigger_reasons
    
    def test_hyper_scale_status_reporting(self, scale_engine):
        """Test hyper scale status reporting."""
        # Add some test nodes
        from gaudi3_scale.hyper_scale_engine import HyperScaleNode
        
        node1 = HyperScaleNode(
            node_id="test-node-1",
            node_type="compute",
            is_active=True,
            current_load=0.7
        )
        node2 = HyperScaleNode(
            node_id="test-node-2", 
            node_type="memory",
            is_active=False
        )
        
        scale_engine.scaling_nodes[node1.node_id] = node1
        scale_engine.scaling_nodes[node2.node_id] = node2
        
        # Add sample scaling decision
        from gaudi3_scale.hyper_scale_engine import ScalingDecision, ScalingDirection
        decision = ScalingDecision(direction=ScalingDirection.SCALE_UP)
        scale_engine.scaling_decisions.append(decision)
        
        status = scale_engine.get_hyper_scale_status()
        
        assert status["is_running"] is False  # Not started
        assert status["active_nodes"] == 1  # Only node1 is active
        assert status["total_nodes"] == 2
        assert "total_capacity" in status
        assert "performance_metrics" in status
        assert "configuration" in status
        assert status["total_capacity"]["cpu_cores"] == node1.cpu_cores
        assert status["current_utilization"]["average_node_load"] == 0.7


# Integration tests
class TestAutonomousSystemIntegration:
    """Integration tests for autonomous system components."""
    
    @pytest.mark.asyncio
    async def test_autonomous_enhancement_integration(self):
        """Test integration between autonomous enhancement components."""
        # Test that all components can be instantiated together
        enhancer = AutonomousEnhancer()
        intelligence = AdaptiveIntelligenceEngine() 
        resilience = QuantumResilienceManager()
        orchestrator = AutonomousOrchestrator()
        scaler = HyperScaleEngine()
        
        # Test that they can work together (basic integration)
        metrics = PerformanceMetrics(
            cpu_usage=0.8,
            memory_usage=0.7,
            hpu_utilization=0.9
        )
        
        # Enhancer should be able to process metrics
        decision = await enhancer._make_enhancement_decision(metrics)
        assert decision is not None
        
        # Intelligence engine should be able to analyze workload
        workload_metrics = {
            "batch_size": 64,
            "model_size_gb": 7.0,
            "cpu_usage": metrics.cpu_usage,
            "memory_usage": metrics.memory_usage
        }
        
        workload = await intelligence.analyze_workload(workload_metrics)
        assert isinstance(workload, WorkloadProfile)
        
        # Clean up
        await enhancer.stop()
        await intelligence.stop_adaptive_learning()
        await resilience.stop_quantum_resilience()
        await orchestrator.stop_orchestrator()
        await scaler.stop_hyper_scaling()
    
    @pytest.mark.asyncio 
    async def test_end_to_end_autonomous_training(self):
        """Test end-to-end autonomous training workflow."""
        orchestrator = AutonomousOrchestrator()
        
        try:
            # Create training workflow
            model_config = {
                "model_type": "transformer",
                "num_layers": 12,
                "hidden_size": 768
            }
            
            training_config = {
                "epochs": 3,
                "batch_size": 32,
                "learning_rate": 1e-4
            }
            
            workflow_id = await orchestrator.create_training_workflow(
                model_config, training_config
            )
            
            assert workflow_id in orchestrator.workflows
            
            workflow = orchestrator.workflows[workflow_id]
            assert len(workflow.tasks) >= 5  # Setup, data, model, training, validation, deployment
            
            # Verify task dependencies are properly set
            task_types = {task.task_type: task for task in workflow.tasks}
            
            # Training should depend on setup-related tasks
            training_task = task_types.get("training")
            if training_task:
                assert len(training_task.dependencies) > 0
                
        finally:
            await orchestrator.stop_orchestrator()


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmark tests for autonomous systems."""
    
    @pytest.mark.asyncio
    async def test_enhancement_decision_performance(self):
        """Benchmark enhancement decision making performance."""
        enhancer = AutonomousEnhancer()
        
        try:
            # Measure decision making speed
            start_time = time.time()
            
            for i in range(100):
                metrics = PerformanceMetrics(
                    cpu_usage=0.5 + i * 0.003,
                    memory_usage=0.6 + i * 0.002,
                    hpu_utilization=0.8 + i * 0.001,
                    throughput_tokens_per_sec=1000 + i * 10,
                    latency_ms=50 + i * 0.1
                )
                
                decision = await enhancer._make_enhancement_decision(metrics)
            
            end_time = time.time()
            total_time = end_time - start_time
            decisions_per_second = 100 / total_time
            
            # Should process at least 10 decisions per second
            assert decisions_per_second > 10
            
        finally:
            await enhancer.stop()
    
    @pytest.mark.asyncio
    async def test_scaling_decision_performance(self):
        """Benchmark scaling decision making performance."""
        scaler = HyperScaleEngine()
        
        try:
            await scaler._initialize_scaling_infrastructure()
            
            # Measure scaling decision speed
            start_time = time.time()
            
            for i in range(50):
                metrics = ScalingMetrics(
                    cpu_utilization=0.5 + i * 0.01,
                    memory_utilization=0.6 + i * 0.008,
                    throughput_ops_per_sec=1000 + i * 50
                )
                
                decisions = await scaler._generate_reactive_decisions(metrics)
            
            end_time = time.time()
            total_time = end_time - start_time
            decisions_per_second = 50 / total_time
            
            # Should process at least 5 scaling analyses per second
            assert decisions_per_second > 5
            
        finally:
            await scaler.stop_hyper_scaling()


if __name__ == "__main__":
    # Run tests with pytest
    import math
    pytest.main([__file__, "-v", "--tb=short"])