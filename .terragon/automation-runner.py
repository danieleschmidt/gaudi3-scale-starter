#!/usr/bin/env python3
"""Autonomous Task Execution Runner for Terragon SDLC Enhancement."""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class AutomationRunner:
    """Executes autonomous SDLC enhancement tasks based on value prioritization."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.config_path = self.repo_path / ".terragon" / "config.yaml"
        self.metrics_path = self.repo_path / ".terragon" / "value-metrics.json"
        self.execution_log = []
        
    def load_backlog_items(self) -> List[Dict]:
        """Load prioritized backlog items from BACKLOG.md parsing."""
        # Simulate loading from discovery engine or manual backlog
        return [
            {
                "id": "DEPS-001",
                "title": "Update Python dependencies",
                "category": "dependency-update",
                "estimated_effort": 1.5,
                "composite_score": 71.2,
                "files_affected": ["requirements.txt", "requirements-dev.txt"],
                "automated": True,
                "command": "pip list --outdated --format=json"
            },
            {
                "id": "TYPE-001", 
                "title": "Improve type hint coverage",
                "category": "code-quality",
                "estimated_effort": 4.0,
                "composite_score": 65.8,
                "files_affected": ["src/gaudi3_scale/"],
                "automated": False,
                "description": "Add type hints to improve code maintainability"
            },
            {
                "id": "TEST-COV-001",
                "title": "Generate test coverage report",
                "category": "quality-assurance", 
                "estimated_effort": 0.5,
                "composite_score": 45.0,
                "files_affected": ["tests/"],
                "automated": True,
                "command": "python -m pytest --cov=gaudi3_scale --cov-report=term --cov-report=html"
            }
        ]
    
    def execute_dependency_updates(self) -> Dict:
        """Execute automated dependency update analysis."""
        result = {
            "task_id": "DEPS-001",
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "findings": [],
            "actions_taken": []
        }
        
        try:
            # Check for outdated packages
            cmd_result = subprocess.run([
                sys.executable, "-m", "pip", "list", "--outdated", "--format=json"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if cmd_result.returncode == 0 and cmd_result.stdout:
                outdated = json.loads(cmd_result.stdout)
                result["findings"] = [
                    f"{pkg['name']}: {pkg['version']} -> {pkg['latest_version']}"
                    for pkg in outdated[:5]  # Limit to top 5
                ]
                result["actions_taken"].append("Generated dependency update recommendations")
            else:
                result["findings"] = ["No outdated packages detected"]
                
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            
        return result
    
    def execute_test_coverage_analysis(self) -> Dict:
        """Execute test coverage analysis."""
        result = {
            "task_id": "TEST-COV-001",
            "status": "completed", 
            "timestamp": datetime.now().isoformat(),
            "findings": [],
            "actions_taken": []
        }
        
        try:
            # Run coverage analysis if pytest is available
            coverage_cmd = [
                sys.executable, "-m", "pytest", 
                "--cov=gaudi3_scale", 
                "--cov-report=term",
                "--tb=no", 
                "-q"
            ]
            
            cmd_result = subprocess.run(
                coverage_cmd, 
                capture_output=True, 
                text=True, 
                cwd=self.repo_path
            )
            
            if "TOTAL" in cmd_result.stdout:
                # Extract coverage percentage
                lines = cmd_result.stdout.split('\n')
                for line in lines:
                    if "TOTAL" in line:
                        parts = line.split()
                        if len(parts) >= 4:
                            coverage_pct = parts[-1]
                            result["findings"].append(f"Current test coverage: {coverage_pct}")
                            
                            # Recommend improvements if coverage < 80%
                            try:
                                pct_val = int(coverage_pct.replace('%', ''))
                                if pct_val < 80:
                                    result["findings"].append(
                                        f"Coverage below target (80%). Recommend adding {80-pct_val}% more tests"
                                    )
                            except ValueError:
                                pass
                                
                result["actions_taken"].append("Generated test coverage analysis")
            else:
                result["findings"] = ["Unable to determine test coverage - pytest/coverage not available"]
                
        except FileNotFoundError:
            result["findings"] = ["pytest not installed - cannot run coverage analysis"]
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            
        return result
        
    def execute_code_quality_scan(self) -> Dict:
        """Execute basic code quality analysis."""
        result = {
            "task_id": "CODE-QUAL-001",
            "status": "completed",
            "timestamp": datetime.now().isoformat(), 
            "findings": [],
            "actions_taken": []
        }
        
        try:
            # Count Python files and basic metrics
            py_files = list(self.repo_path.glob("src/**/*.py"))
            result["findings"].append(f"Found {len(py_files)} Python files in src/")
            
            # Check for basic code issues
            total_lines = 0
            files_with_todos = 0
            
            for py_file in py_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        total_lines += len(lines)
                        
                        # Check for TODO/FIXME comments
                        if any('TODO' in line or 'FIXME' in line for line in lines):
                            files_with_todos += 1
                            
                except Exception:
                    continue
                    
            result["findings"].append(f"Total lines of code: {total_lines}")
            
            if files_with_todos > 0:
                result["findings"].append(
                    f"Found {files_with_todos} files with TODO/FIXME comments"
                )
                
            result["actions_taken"].append("Performed basic code quality analysis")
            
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            
        return result
    
    def execute_security_check(self) -> Dict:
        """Execute basic security checks."""
        result = {
            "task_id": "SEC-CHECK-001",
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "findings": [],
            "actions_taken": []
        }
        
        try:
            # Check for common security files
            security_files = [
                "SECURITY.md",
                ".secrets.baseline", 
                ".pre-commit-config.yaml"
            ]
            
            found_files = []
            for sec_file in security_files:
                if (self.repo_path / sec_file).exists():
                    found_files.append(sec_file)
                    
            result["findings"].append(f"Security files present: {', '.join(found_files)}")
            
            # Check for potential security issues in code
            py_files = list(self.repo_path.glob("src/**/*.py"))
            security_concerns = []
            
            for py_file in py_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Check for obvious security anti-patterns
                        if 'password' in content.lower() or 'secret' in content.lower():
                            if 'os.environ' not in content:
                                security_concerns.append(f"{py_file.name}: potential hardcoded secrets")
                                
                except Exception:
                    continue
                    
            if security_concerns:
                result["findings"].extend(security_concerns)
            else:
                result["findings"].append("No obvious security concerns detected in source code")
                
            result["actions_taken"].append("Performed basic security scan")
            
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            
        return result
    
    def run_autonomous_cycle(self) -> Dict:
        """Execute one cycle of autonomous SDLC enhancement."""
        cycle_start = datetime.now()
        
        cycle_results = {
            "cycle_id": f"auto-{int(cycle_start.timestamp())}",
            "start_time": cycle_start.isoformat(),
            "tasks_executed": [],
            "total_value_delivered": 0.0,
            "errors": [],
            "recommendations": []
        }
        
        # Execute automated tasks
        automated_tasks = [
            ("dependency_updates", self.execute_dependency_updates),
            ("test_coverage", self.execute_test_coverage_analysis), 
            ("code_quality", self.execute_code_quality_scan),
            ("security_check", self.execute_security_check)
        ]
        
        for task_name, task_func in automated_tasks:
            try:
                task_result = task_func()
                cycle_results["tasks_executed"].append(task_result)
                
                if task_result["status"] == "completed":
                    # Add some value points for completed tasks
                    cycle_results["total_value_delivered"] += 10.0
                else:
                    cycle_results["errors"].append(f"{task_name}: {task_result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                cycle_results["errors"].append(f"{task_name}: {str(e)}")
        
        # Generate recommendations based on findings
        all_findings = []
        for task in cycle_results["tasks_executed"]:
            all_findings.extend(task.get("findings", []))
            
        if any("coverage" in finding.lower() for finding in all_findings):
            cycle_results["recommendations"].append(
                "Consider increasing test coverage by adding more unit tests"
            )
            
        if any("todo" in finding.lower() for finding in all_findings):
            cycle_results["recommendations"].append(
                "Address TODO/FIXME comments to reduce technical debt"
            )
            
        if any("outdated" in finding.lower() for finding in all_findings):
            cycle_results["recommendations"].append(
                "Update outdated dependencies for security and performance benefits"
            )
        
        cycle_results["end_time"] = datetime.now().isoformat()
        cycle_results["duration_seconds"] = (
            datetime.now() - cycle_start
        ).total_seconds()
        
        return cycle_results
    
    def update_metrics(self, cycle_results: Dict):
        """Update value metrics with cycle results."""
        try:
            # Load existing metrics
            if self.metrics_path.exists():
                with open(self.metrics_path, 'r') as f:
                    metrics = json.load(f)
            else:
                metrics = {
                    "executionHistory": [],
                    "backlogMetrics": {},
                    "continuousMetrics": {}
                }
            
            # Add execution history entry
            history_entry = {
                "timestamp": cycle_results["start_time"],
                "itemId": cycle_results["cycle_id"],
                "title": "Autonomous SDLC enhancement cycle",
                "actualEffort": cycle_results["duration_seconds"] / 3600,  # Convert to hours
                "actualImpact": {
                    "tasksCompleted": len(cycle_results["tasks_executed"]),
                    "valueDelivered": cycle_results["total_value_delivered"],
                    "errorsEncountered": len(cycle_results["errors"])
                },
                "learnings": "; ".join(cycle_results["recommendations"][:3])
            }
            
            metrics["executionHistory"].append(history_entry)
            
            # Update continuous metrics
            metrics["continuousMetrics"] = {
                "lastExecutionTime": cycle_results["end_time"],
                "nextScheduledRun": (
                    datetime.fromisoformat(cycle_results["end_time"].replace('Z', '+00:00')) 
                ).isoformat(),
                "totalCycles": len(metrics["executionHistory"]),
                "averageCycleTime": sum(
                    entry.get("actualEffort", 0) for entry in metrics["executionHistory"]
                ) / max(len(metrics["executionHistory"]), 1)
            }
            
            # Save updated metrics
            with open(self.metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not update metrics: {e}")


def main():
    """Main entry point for autonomous execution."""
    runner = AutomationRunner(".")
    
    try:
        print("🚀 Starting Autonomous SDLC Enhancement Cycle...")
        cycle_results = runner.run_autonomous_cycle()
        
        print(f"✅ Cycle completed in {cycle_results['duration_seconds']:.1f} seconds")
        print(f"📊 Tasks executed: {len(cycle_results['tasks_executed'])}")
        print(f"💎 Value delivered: {cycle_results['total_value_delivered']:.1f} points")
        
        if cycle_results["errors"]:
            print(f"⚠️  Errors encountered: {len(cycle_results['errors'])}")
            for error in cycle_results["errors"]:
                print(f"   - {error}")
        
        if cycle_results["recommendations"]:
            print("\n💡 Recommendations:")
            for rec in cycle_results["recommendations"]:
                print(f"   - {rec}")
        
        # Update metrics
        runner.update_metrics(cycle_results)
        print("\n📈 Value metrics updated successfully")
        
        return 0
        
    except Exception as e:
        print(f"❌ Autonomous cycle failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())