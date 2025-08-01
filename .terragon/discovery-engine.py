#!/usr/bin/env python3
"""Autonomous Value Discovery Engine for Terragon SDLC Enhancement."""

import json
import os
import subprocess
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ValueItem:
    """Represents a discoverable work item with value scoring."""
    
    id: str
    title: str
    description: str
    category: str
    source: str
    files_affected: List[str]
    estimated_effort: float
    wsjf_score: float
    ice_score: float
    technical_debt_score: float
    composite_score: float
    discovery_date: str
    dependencies: List[str] = None
    risk_level: float = 0.0
    security_related: bool = False
    compliance_related: bool = False


class ValueDiscoveryEngine:
    """Autonomous engine for discovering and prioritizing value items."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.config_path = self.repo_path / ".terragon" / "config.yaml"
        self.metrics_path = self.repo_path / ".terragon" / "value-metrics.json"
        self.config = self._load_config()
        self.metrics = self._load_metrics()
    
    def _load_config(self) -> Dict:
        """Load Terragon configuration."""
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    def _load_metrics(self) -> Dict:
        """Load current value metrics."""                                                                                                                                                              
        with open(self.metrics_path) as f:
            return json.load(f)
    
    def discover_from_git_history(self) -> List[ValueItem]:
        """Discover value items from Git commit history and comments."""
        items = []
        
        # Find TODO/FIXME comments in code
        try:
            result = subprocess.run([
                "grep", "-r", "-n", "-E", 
                "(TODO|FIXME|HACK|DEPRECATED|XXX).*",
                "src/", "tests/"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            for i, line in enumerate(result.stdout.split('\n')[:10]):  # Limit items
                if line.strip():
                    parts = line.split(':', 2)
                    if len(parts) >= 3:
                        file_path, line_num, comment = parts
                        items.append(ValueItem(
                            id=f"code-debt-{i+1:03d}",
                            title=f"Address technical debt in {Path(file_path).name}",
                            description=comment.strip(),
                            category="technical-debt",
                            source="code-comments",
                            files_affected=[file_path],
                            estimated_effort=2.0,
                            wsjf_score=self._calculate_wsjf(category="technical-debt"),
                            ice_score=self._calculate_ice(impact=6, confidence=8, ease=7),
                            technical_debt_score=45.0,
                            composite_score=0.0,  # Will be calculated
                            discovery_date=datetime.now().isoformat()
                        ))
        except Exception:
            pass  # Skip if grep fails
        
        return items
    
    def discover_from_static_analysis(self) -> List[ValueItem]:
        """Discover issues from static analysis tools."""
        items = []
        
        # Run basic Python checks
        try:
            # Check for missing type hints
            result = subprocess.run([
                "grep", "-r", "-n", "-E",
                "def .*\\(.*\\) ->",
                "src/"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            untyped_functions = []
            all_functions = subprocess.run([
                "grep", "-r", "-n", "-E", "def ",
                "src/"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            typed_count = len(result.stdout.split('\n')) if result.stdout else 0
            total_count = len(all_functions.stdout.split('\n')) if all_functions.stdout else 1
            
            if typed_count / total_count < 0.8:  # Less than 80% typed
                items.append(ValueItem(
                    id="type-hints-001",
                    title="Improve type hint coverage",
                    description=f"Only {typed_count}/{total_count} functions have type hints",
                    category="code-quality",
                    source="static-analysis",
                    files_affected=["src/"],
                    estimated_effort=4.0,
                    wsjf_score=self._calculate_wsjf(category="code-quality"),
                    ice_score=self._calculate_ice(impact=7, confidence=9, ease=6),
                    technical_debt_score=35.0,
                    composite_score=0.0,
                    discovery_date=datetime.now().isoformat()
                ))
        except Exception:
            pass
        
        return items
    
    def discover_from_dependencies(self) -> List[ValueItem]:
        """Discover dependency-related improvements."""
        items = []
        
        # Check for outdated dependencies (simulate)
        requirements_file = self.repo_path / "requirements.txt"
        if requirements_file.exists():
            items.append(ValueItem(
                id="deps-update-001",
                title="Update Python dependencies to latest versions",
                description="Regular dependency updates for security and performance",
                category="dependency-update",
                source="dependency-scanner",
                files_affected=["requirements.txt", "requirements-dev.txt"],
                estimated_effort=1.5,
                wsjf_score=self._calculate_wsjf(category="dependency-update"),
                ice_score=self._calculate_ice(impact=5, confidence=9, ease=8),
                technical_debt_score=20.0,
                composite_score=0.0,
                discovery_date=datetime.now().isoformat(),
                security_related=True
            ))
        
        return items
    
    def discover_infrastructure_gaps(self) -> List[ValueItem]:
        """Discover missing infrastructure components."""
        items = []
        
        # Check for missing CI/CD workflows
        workflows_dir = self.repo_path / ".github" / "workflows"
        if not workflows_dir.exists() or not any(workflows_dir.glob("*.yml")):
            items.append(ValueItem(
                id="cicd-setup-001", 
                title="Implement GitHub Actions CI/CD workflows",
                description="Set up automated testing, building, and deployment pipelines",
                category="infrastructure",
                source="infrastructure-analysis",
                files_affected=[".github/workflows/"],
                estimated_effort=6.0,
                wsjf_score=self._calculate_wsjf(category="infrastructure"),
                ice_score=self._calculate_ice(impact=9, confidence=8, ease=6),
                technical_debt_score=60.0,
                composite_score=0.0,
                discovery_date=datetime.now().isoformat()
            ))
        
        # Check for missing terraform infrastructure
        terraform_dir = self.repo_path / "terraform"
        if not terraform_dir.exists():
            items.append(ValueItem(
                id="terraform-infra-001",
                title="Implement Terraform infrastructure as code",
                description="Create reusable infrastructure templates for Gaudi 3 clusters",
                category="infrastructure",
                source="infrastructure-analysis", 
                files_affected=["terraform/"],
                estimated_effort=12.0,
                wsjf_score=self._calculate_wsjf(category="infrastructure"),
                ice_score=self._calculate_ice(impact=10, confidence=7, ease=4),
                technical_debt_score=80.0,
                composite_score=0.0,
                discovery_date=datetime.now().isoformat()
            ))
        
        return items
    
    def _calculate_wsjf(self, category: str) -> float:
        """Calculate Weighted Shortest Job First score."""
        # Simplified WSJF calculation based on category
        category_weights = {
            "security": 50.0,
            "infrastructure": 40.0,
            "technical-debt": 30.0,
            "code-quality": 25.0,
            "dependency-update": 20.0,
            "documentation": 15.0
        }
        return category_weights.get(category, 20.0)
    
    def _calculate_ice(self, impact: int, confidence: int, ease: int) -> float:
        """Calculate Impact, Confidence, Ease score."""
        return float(impact * confidence * ease)
    
    def _calculate_composite_score(self, item: ValueItem) -> float:
        """Calculate final composite score using configured weights."""
        weights = self.config["scoring"]["weights"]["maturing"]
        
        # Normalize scores to 0-100 scale
        normalized_wsjf = min(item.wsjf_score / 50.0 * 100, 100)
        normalized_ice = min(item.ice_score / 1000.0 * 100, 100)
        normalized_debt = min(item.technical_debt_score, 100)
        
        composite = (
            weights["wsjf"] * normalized_wsjf +
            weights["ice"] * normalized_ice +
            weights["technicalDebt"] * normalized_debt
        )
        
        # Apply boosts
        if item.security_related:
            composite *= self.config["scoring"]["thresholds"]["securityBoost"]
        
        if item.compliance_related:
            composite *= self.config["scoring"]["thresholds"]["complianceBoost"]
        
        return min(composite, 100.0)
    
    def discover_all_value_items(self) -> List[ValueItem]:
        """Run all discovery methods and return prioritized items."""
        all_items = []
        
        # Run discovery from multiple sources
        all_items.extend(self.discover_from_git_history())
        all_items.extend(self.discover_from_static_analysis())
        all_items.extend(self.discover_from_dependencies())
        all_items.extend(self.discover_infrastructure_gaps())
        
        # Calculate composite scores
        for item in all_items:
            item.composite_score = self._calculate_composite_score(item)
        
        # Sort by composite score (highest first)
        all_items.sort(key=lambda x: x.composite_score, reverse=True)
        
        return all_items
    
    def update_metrics(self, discovered_items: List[ValueItem]):
        """Update value metrics with newly discovered items."""
        self.metrics["backlogMetrics"]["totalItems"] = len(discovered_items)
        self.metrics["continuousMetrics"]["lastDiscoveryRun"] = datetime.now().isoformat()
        self.metrics["continuousMetrics"]["nextScheduledRun"] = (
            datetime.now() + timedelta(hours=1)
        ).isoformat()
        
        # Save updated metrics
        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)


if __name__ == "__main__":
    engine = ValueDiscoveryEngine(".")
    items = engine.discover_all_value_items()
    
    print(f"Discovered {len(items)} value items:")
    for item in items[:5]:  # Show top 5
        print(f"  {item.id}: {item.title} (Score: {item.composite_score:.1f})")
    
    engine.update_metrics(items)