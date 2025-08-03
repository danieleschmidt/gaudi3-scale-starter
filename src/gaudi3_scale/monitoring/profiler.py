"""Gaudi 3 performance profiling and analysis."""

import logging
import time
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ProfileEvent:
    """Single profiling event."""
    name: str
    category: str
    timestamp: float
    duration: float
    thread_id: int
    device_id: Optional[int] = None
    memory_usage: Optional[int] = None
    args: Optional[Dict[str, Any]] = None


@dataclass
class ProfileSession:
    """Profiling session data."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    events: List[ProfileEvent] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.events is None:
            self.events = []
        if self.metadata is None:
            self.metadata = {}


class GaudiProfiler:
    """Performance profiler for Gaudi 3 training workloads.
    
    Provides detailed profiling of training operations including
    HPU utilization, memory usage, and operation timing.
    """
    
    def __init__(self, output_dir: str = "./profiling_results"):
        """Initialize Gaudi profiler.
        
        Args:
            output_dir: Directory to save profiling results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_session: Optional[ProfileSession] = None
        self.profiling_enabled = False
        self.event_buffer: List[ProfileEvent] = []
        self.max_buffer_size = 10000
        
        # Profiling configuration
        self.profile_cpu = True
        self.profile_hpu = True
        self.profile_memory = True
        self.profile_operations = True
        
        self._setup_profiling_hooks()
    
    def start_profiling(self, session_name: str = None) -> str:
        """Start a profiling session.
        
        Args:
            session_name: Optional session name
            
        Returns:
            Session ID
        """
        if self.current_session is not None:
            logger.warning("Profiling session already active, stopping previous session")
            self.stop_profiling()
        
        session_id = session_name or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = ProfileSession(
            session_id=session_id,
            start_time=datetime.now(),
            metadata={
                "profiler_version": "1.0",
                "profiling_config": {
                    "cpu": self.profile_cpu,
                    "hpu": self.profile_hpu,
                    "memory": self.profile_memory,
                    "operations": self.profile_operations
                }
            }
        )
        
        self.profiling_enabled = True
        self.event_buffer.clear()
        
        logger.info(f"Started profiling session: {session_id}")
        return session_id
    
    def stop_profiling(self) -> Optional[str]:
        """Stop the current profiling session.
        
        Returns:
            Path to saved profiling results
        """
        if self.current_session is None:
            logger.warning("No active profiling session")
            return None
        
        self.profiling_enabled = False
        self.current_session.end_time = datetime.now()
        self.current_session.events = self.event_buffer.copy()
        
        # Save profiling results
        output_path = self._save_session(self.current_session)
        
        logger.info(f"Stopped profiling session: {self.current_session.session_id}")
        logger.info(f"Profiling results saved to: {output_path}")
        
        self.current_session = None
        self.event_buffer.clear()
        
        return str(output_path)
    
    def add_event(self, name: str, category: str, duration: float,
                 device_id: Optional[int] = None, memory_usage: Optional[int] = None,
                 args: Optional[Dict[str, Any]] = None) -> None:
        """Add a profiling event.
        
        Args:
            name: Event name
            category: Event category (compute, memory, communication, etc.)
            duration: Event duration in seconds
            device_id: Device ID if applicable
            memory_usage: Memory usage in bytes
            args: Additional event arguments
        """
        if not self.profiling_enabled:
            return
        
        event = ProfileEvent(
            name=name,
            category=category,
            timestamp=time.time(),
            duration=duration,
            thread_id=0,  # Simplified for now
            device_id=device_id,
            memory_usage=memory_usage,
            args=args
        )
        
        self.event_buffer.append(event)
        
        # Flush buffer if it gets too large
        if len(self.event_buffer) > self.max_buffer_size:
            self._flush_buffer()
    
    def profile_training_step(self, step_function: Callable, step_num: int) -> Dict[str, Any]:
        """Profile a single training step.
        
        Args:
            step_function: Function to profile
            step_num: Step number
            
        Returns:
            Profiling results for the step
        """
        step_start = time.time()
        
        # Profile memory before step
        memory_before = self._get_hpu_memory_usage()
        
        # Execute step
        try:
            result = step_function()
        except Exception as e:
            logger.error(f"Error during profiled training step: {e}")
            raise
        
        step_end = time.time()
        step_duration = step_end - step_start
        
        # Profile memory after step
        memory_after = self._get_hpu_memory_usage()
        memory_delta = memory_after - memory_before
        
        # Add step event
        self.add_event(
            name=f"training_step_{step_num}",
            category="training",
            duration=step_duration,
            memory_usage=memory_after,
            args={
                "step_num": step_num,
                "memory_delta": memory_delta,
                "throughput": result.get("throughput", 0) if isinstance(result, dict) else 0
            }
        )
        
        return {
            "step_num": step_num,
            "duration": step_duration,
            "memory_before": memory_before,
            "memory_after": memory_after,
            "memory_delta": memory_delta
        }
    
    def profile_model_forward(self, forward_function: Callable, batch_size: int) -> Dict[str, Any]:
        """Profile model forward pass.
        
        Args:
            forward_function: Forward pass function
            batch_size: Batch size
            
        Returns:
            Forward pass profiling results
        """
        forward_start = time.time()
        
        try:
            result = forward_function()
        except Exception as e:
            logger.error(f"Error during profiled forward pass: {e}")
            raise
        
        forward_end = time.time()
        forward_duration = forward_end - forward_start
        
        self.add_event(
            name="model_forward",
            category="compute",
            duration=forward_duration,
            args={
                "batch_size": batch_size,
                "samples_per_second": batch_size / forward_duration
            }
        )
        
        return {
            "duration": forward_duration,
            "batch_size": batch_size,
            "samples_per_second": batch_size / forward_duration
        }
    
    def analyze_session(self, session_path: str) -> Dict[str, Any]:
        """Analyze profiling session results.
        
        Args:
            session_path: Path to session file
            
        Returns:
            Analysis results
        """
        try:
            with open(session_path, 'r') as f:
                session_data = json.load(f)
            
            events = [ProfileEvent(**event) for event in session_data["events"]]
            
            # Analyze events
            analysis = {
                "session_summary": {
                    "total_events": len(events),
                    "session_duration": session_data["metadata"].get("duration_seconds", 0),
                    "start_time": session_data["start_time"],
                    "end_time": session_data.get("end_time")
                },
                "category_breakdown": self._analyze_categories(events),
                "performance_metrics": self._analyze_performance(events),
                "memory_analysis": self._analyze_memory(events),
                "bottleneck_analysis": self._analyze_bottlenecks(events)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing session: {e}")
            return {"error": str(e)}
    
    def generate_report(self, analysis: Dict[str, Any], output_path: str = None) -> str:
        """Generate HTML profiling report.
        
        Args:
            analysis: Analysis results
            output_path: Output path for report
            
        Returns:
            Path to generated report
        """
        if output_path is None:
            output_path = self.output_dir / "profiling_report.html"
        
        html_content = self._generate_html_report(analysis)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated profiling report: {output_path}")
        return str(output_path)
    
    def _setup_profiling_hooks(self) -> None:
        """Setup profiling hooks for automatic event capture."""
        # This would integrate with Habana profiling APIs
        # For now, we'll use manual event addition
        pass
    
    def _get_hpu_memory_usage(self) -> int:
        """Get current HPU memory usage.
        
        Returns:
            Memory usage in bytes
        """
        try:
            import habana_frameworks.torch as htorch
            return htorch.hpu.memory_allocated()
        except ImportError:
            return 0
    
    def _flush_buffer(self) -> None:
        """Flush event buffer to current session."""
        if self.current_session is not None:
            self.current_session.events.extend(self.event_buffer)
            self.event_buffer.clear()
    
    def _save_session(self, session: ProfileSession) -> Path:
        """Save profiling session to file.
        
        Args:
            session: Profiling session to save
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / f"{session.session_id}.json"
        
        # Calculate session duration
        if session.end_time:
            duration = (session.end_time - session.start_time).total_seconds()
            session.metadata["duration_seconds"] = duration
        
        # Convert to serializable format
        session_data = {
            "session_id": session.session_id,
            "start_time": session.start_time.isoformat(),
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "metadata": session.metadata,
            "events": [asdict(event) for event in session.events]
        }
        
        with open(output_path, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        return output_path
    
    def _analyze_categories(self, events: List[ProfileEvent]) -> Dict[str, Any]:
        """Analyze events by category.
        
        Args:
            events: List of profiling events
            
        Returns:
            Category analysis
        """
        category_stats = {}
        
        for event in events:
            if event.category not in category_stats:
                category_stats[event.category] = {
                    "count": 0,
                    "total_duration": 0,
                    "avg_duration": 0,
                    "max_duration": 0
                }
            
            stats = category_stats[event.category]
            stats["count"] += 1
            stats["total_duration"] += event.duration
            stats["max_duration"] = max(stats["max_duration"], event.duration)
        
        # Calculate averages
        for stats in category_stats.values():
            if stats["count"] > 0:
                stats["avg_duration"] = stats["total_duration"] / stats["count"]
        
        return category_stats
    
    def _analyze_performance(self, events: List[ProfileEvent]) -> Dict[str, Any]:
        """Analyze performance metrics.
        
        Args:
            events: List of profiling events
            
        Returns:
            Performance analysis
        """
        training_events = [e for e in events if "training_step" in e.name]
        
        if not training_events:
            return {"error": "No training events found"}
        
        durations = [e.duration for e in training_events]
        throughputs = []
        
        for event in training_events:
            if event.args and "throughput" in event.args:
                throughputs.append(event.args["throughput"])
        
        return {
            "training_steps": len(training_events),
            "avg_step_duration": sum(durations) / len(durations),
            "min_step_duration": min(durations),
            "max_step_duration": max(durations),
            "avg_throughput": sum(throughputs) / len(throughputs) if throughputs else 0,
            "total_training_time": sum(durations)
        }
    
    def _analyze_memory(self, events: List[ProfileEvent]) -> Dict[str, Any]:
        """Analyze memory usage patterns.
        
        Args:
            events: List of profiling events
            
        Returns:
            Memory analysis
        """
        memory_events = [e for e in events if e.memory_usage is not None]
        
        if not memory_events:
            return {"error": "No memory events found"}
        
        memory_values = [e.memory_usage for e in memory_events]
        
        return {
            "peak_memory": max(memory_values),
            "avg_memory": sum(memory_values) / len(memory_values),
            "min_memory": min(memory_values),
            "memory_events": len(memory_events)
        }
    
    def _analyze_bottlenecks(self, events: List[ProfileEvent]) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks.
        
        Args:
            events: List of profiling events
            
        Returns:
            List of identified bottlenecks
        """
        bottlenecks = []
        
        # Find longest duration events
        sorted_events = sorted(events, key=lambda e: e.duration, reverse=True)
        
        # Top 10 longest events
        for i, event in enumerate(sorted_events[:10]):
            bottlenecks.append({
                "rank": i + 1,
                "event_name": event.name,
                "category": event.category,
                "duration": event.duration,
                "percentage_of_total": (event.duration / sum(e.duration for e in events)) * 100
            })
        
        return bottlenecks
    
    def _generate_html_report(self, analysis: Dict[str, Any]) -> str:
        """Generate HTML report from analysis.
        
        Args:
            analysis: Analysis results
            
        Returns:
            HTML content
        """
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Gaudi 3 Profiling Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
        .metric {{ margin: 10px 0; }}
        .bottleneck {{ background: #fff3cd; padding: 10px; margin: 5px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Gaudi 3 Training Profiling Report</h1>
    
    <div class="section">
        <h2>Session Summary</h2>
        <div class="metric">Total Events: {analysis.get('session_summary', {}).get('total_events', 0)}</div>
        <div class="metric">Session Duration: {analysis.get('session_summary', {}).get('session_duration', 0):.2f} seconds</div>
    </div>
    
    <div class="section">
        <h2>Performance Metrics</h2>
"""
        
        perf = analysis.get('performance_metrics', {})
        if 'error' not in perf:
            html += f"""
        <div class="metric">Training Steps: {perf.get('training_steps', 0)}</div>
        <div class="metric">Average Step Duration: {perf.get('avg_step_duration', 0):.4f} seconds</div>
        <div class="metric">Average Throughput: {perf.get('avg_throughput', 0):.0f} tokens/sec</div>
        <div class="metric">Total Training Time: {perf.get('total_training_time', 0):.2f} seconds</div>
"""
        
        html += """
    </div>
    
    <div class="section">
        <h2>Bottleneck Analysis</h2>
"""
        
        bottlenecks = analysis.get('bottleneck_analysis', [])
        for bottleneck in bottlenecks[:5]:  # Top 5
            html += f"""
        <div class="bottleneck">
            <strong>#{bottleneck['rank']}: {bottleneck['event_name']}</strong><br>
            Category: {bottleneck['category']}<br>
            Duration: {bottleneck['duration']:.4f} seconds ({bottleneck['percentage_of_total']:.1f}% of total)
        </div>
"""
        
        html += """
    </div>
</body>
</html>
"""
        
        return html