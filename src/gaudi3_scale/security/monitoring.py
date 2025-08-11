"""Enterprise security monitoring and threat detection system.

This module provides real-time security monitoring, threat detection,
intrusion detection, and incident response capabilities.
"""

import time
import json
import asyncio
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Set, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
from collections import defaultdict, deque
import statistics

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from pydantic import BaseModel, Field
except ImportError:
    # Fallback for environments without pydantic
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    def Field(default=None, **kwargs):
        return default
from ..logging_utils import get_logger
from ..database.connection import get_redis
from .audit_logging import AuditEvent, AuditLevel, EventCategory, SecurityAuditLogger

logger = get_logger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class AlertType(Enum):
    """Types of security alerts."""
    INTRUSION_DETECTION = "intrusion_detection"
    BRUTE_FORCE_ATTACK = "brute_force_attack"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    MALWARE_DETECTION = "malware_detection"
    DOS_ATTACK = "dos_attack"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    CONFIGURATION_TAMPERING = "configuration_tampering"
    SUSPICIOUS_NETWORK_ACTIVITY = "suspicious_network_activity"


class IncidentStatus(Enum):
    """Incident response status."""
    OPEN = "open"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"
    CLOSED = "closed"


@dataclass
class SecurityAlert:
    """Represents a security alert."""
    alert_id: str
    alert_type: AlertType
    threat_level: ThreatLevel
    title: str
    description: str
    timestamp: datetime
    source: str
    indicators: List[str] = field(default_factory=list)
    affected_resources: List[str] = field(default_factory=list)
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    confidence_score: float = 0.0  # 0-1 confidence in the alert
    false_positive_probability: float = 0.0  # 0-1 probability of false positive
    context: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    related_events: List[str] = field(default_factory=list)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "threat_level": self.threat_level.value,
            "title": self.title,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "indicators": self.indicators,
            "affected_resources": self.affected_resources,
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "confidence_score": self.confidence_score,
            "false_positive_probability": self.false_positive_probability,
            "context": self.context,
            "recommendations": self.recommendations,
            "related_events": self.related_events,
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None
        }


@dataclass
class SecurityIncident:
    """Represents a security incident."""
    incident_id: str
    title: str
    description: str
    status: IncidentStatus
    severity: ThreatLevel
    created_at: datetime
    updated_at: datetime
    assigned_to: Optional[str] = None
    alerts: List[str] = field(default_factory=list)  # Alert IDs
    events: List[str] = field(default_factory=list)  # Event IDs
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    impact_assessment: Dict[str, Any] = field(default_factory=dict)
    response_actions: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    
    def add_timeline_entry(self, action: str, details: str, user: Optional[str] = None):
        """Add entry to incident timeline."""
        self.timeline.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "details": details,
            "user": user
        })
        self.updated_at = datetime.now(timezone.utc)


class ThreatDetector:
    """Base class for threat detection algorithms."""
    
    def __init__(self, name: str, sensitivity: float = 0.8):
        """Initialize threat detector.
        
        Args:
            name: Detector name
            sensitivity: Detection sensitivity (0-1, higher = more sensitive)
        """
        self.name = name
        self.sensitivity = sensitivity
        self.enabled = True
        self.logger = logger.getChild(f"{self.__class__.__name__}.{name}")
    
    def detect(self, events: List[AuditEvent], context: Dict[str, Any]) -> List[SecurityAlert]:
        """Detect threats from audit events."""
        raise NotImplementedError("Subclasses must implement detect method")
    
    def update_sensitivity(self, sensitivity: float):
        """Update detector sensitivity."""
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        self.logger.info(f"Sensitivity updated to {self.sensitivity}")


class BruteForceDetector(ThreatDetector):
    """Detects brute force authentication attacks."""
    
    def __init__(self, 
                 max_failures: int = 5,
                 time_window: int = 300,  # 5 minutes
                 sensitivity: float = 0.8):
        super().__init__("brute_force", sensitivity)
        self.max_failures = max_failures
        self.time_window = time_window
        self.failure_counts = defaultdict(list)
    
    def detect(self, events: List[AuditEvent], context: Dict[str, Any]) -> List[SecurityAlert]:
        alerts = []
        current_time = datetime.now(timezone.utc)
        
        # Track failed login attempts by IP address
        for event in events:
            if (event.event_type == "login_failure" and 
                event.ip_address and
                event.category == EventCategory.AUTHENTICATION):
                
                ip_address = event.ip_address
                
                # Clean old attempts outside time window
                cutoff_time = current_time - timedelta(seconds=self.time_window)
                self.failure_counts[ip_address] = [
                    timestamp for timestamp in self.failure_counts[ip_address]
                    if timestamp > cutoff_time
                ]
                
                # Add current attempt
                self.failure_counts[ip_address].append(event.timestamp)
                
                # Check if threshold exceeded
                failure_count = len(self.failure_counts[ip_address])
                adjusted_threshold = self.max_failures * (2 - self.sensitivity)
                
                if failure_count >= adjusted_threshold:
                    alert = SecurityAlert(
                        alert_id=f"bf_{ip_address}_{int(time.time())}",
                        alert_type=AlertType.BRUTE_FORCE_ATTACK,
                        threat_level=ThreatLevel.HIGH,
                        title=f"Brute Force Attack Detected from {ip_address}",
                        description=f"Detected {failure_count} failed login attempts from {ip_address} "
                                  f"within {self.time_window} seconds",
                        timestamp=current_time,
                        source=self.name,
                        indicators=[f"failed_logins:{failure_count}", f"source_ip:{ip_address}"],
                        ip_address=ip_address,
                        confidence_score=min(1.0, failure_count / self.max_failures),
                        context={
                            "failure_count": failure_count,
                            "time_window": self.time_window,
                            "usernames_attempted": list(set(
                                event.username for event in events 
                                if event.ip_address == ip_address and event.event_type == "login_failure"
                            ))
                        },
                        recommendations=[
                            f"Block IP address {ip_address}",
                            "Monitor for distributed attacks from multiple IPs",
                            "Review and strengthen password policies",
                            "Implement CAPTCHA after failed attempts"
                        ]
                    )
                    alerts.append(alert)
        
        return alerts


class AnomalyDetector(ThreatDetector):
    """Detects anomalous behavior patterns using statistical analysis."""
    
    def __init__(self, 
                 window_size: int = 100,
                 std_threshold: float = 3.0,
                 sensitivity: float = 0.8):
        super().__init__("anomaly", sensitivity)
        self.window_size = window_size
        self.std_threshold = std_threshold
        self.baseline_data = defaultdict(deque)
    
    def detect(self, events: List[AuditEvent], context: Dict[str, Any]) -> List[SecurityAlert]:
        if not NUMPY_AVAILABLE:
            return []
        
        alerts = []
        
        # Analyze patterns by user
        user_events = defaultdict(list)
        for event in events:
            if event.username:
                user_events[event.username].append(event)
        
        for username, user_event_list in user_events.items():
            anomaly_alerts = self._detect_user_anomalies(username, user_event_list)
            alerts.extend(anomaly_alerts)
        
        return alerts
    
    def _detect_user_anomalies(self, username: str, events: List[AuditEvent]) -> List[SecurityAlert]:
        """Detect anomalies for a specific user."""
        alerts = []
        
        # Check for unusual activity volume
        current_activity_rate = len(events)
        baseline_key = f"{username}_activity_rate"
        
        if baseline_key in self.baseline_data:
            baseline_rates = list(self.baseline_data[baseline_key])
            if len(baseline_rates) >= 5:  # Need minimum baseline data
                mean_rate = statistics.mean(baseline_rates)
                std_rate = statistics.stdev(baseline_rates) if len(baseline_rates) > 1 else 0
                
                if std_rate > 0:
                    z_score = abs(current_activity_rate - mean_rate) / std_rate
                    adjusted_threshold = self.std_threshold * (2 - self.sensitivity)
                    
                    if z_score > adjusted_threshold:
                        alert = SecurityAlert(
                            alert_id=f"anomaly_{username}_{int(time.time())}",
                            alert_type=AlertType.ANOMALOUS_BEHAVIOR,
                            threat_level=ThreatLevel.MEDIUM if z_score < 5 else ThreatLevel.HIGH,
                            title=f"Anomalous Activity Volume for {username}",
                            description=f"User {username} showing unusual activity rate: "
                                      f"{current_activity_rate} (baseline: {mean_rate:.1f}±{std_rate:.1f})",
                            timestamp=datetime.now(timezone.utc),
                            source=self.name,
                            indicators=[f"z_score:{z_score:.2f}", f"activity_rate:{current_activity_rate}"],
                            user_id=username,
                            confidence_score=min(1.0, z_score / 10),
                            context={
                                "current_rate": current_activity_rate,
                                "baseline_mean": mean_rate,
                                "baseline_std": std_rate,
                                "z_score": z_score
                            },
                            recommendations=[
                                f"Investigate {username}'s recent activities",
                                "Check if account is compromised",
                                "Review access patterns and locations"
                            ]
                        )
                        alerts.append(alert)
        
        # Update baseline data
        self.baseline_data[baseline_key].append(current_activity_rate)
        if len(self.baseline_data[baseline_key]) > self.window_size:
            self.baseline_data[baseline_key].popleft()
        
        return alerts


class PrivilegeEscalationDetector(ThreatDetector):
    """Detects privilege escalation attempts."""
    
    def __init__(self, sensitivity: float = 0.8):
        super().__init__("privilege_escalation", sensitivity)
        self.admin_actions = {
            "user_creation", "user_deletion", "role_assignment", 
            "permission_grant", "config_change", "system_access"
        }
    
    def detect(self, events: List[AuditEvent], context: Dict[str, Any]) -> List[SecurityAlert]:
        alerts = []
        
        # Group events by user
        user_events = defaultdict(list)
        for event in events:
            if event.username:
                user_events[event.username].append(event)
        
        for username, user_event_list in user_events.items():
            escalation_alerts = self._detect_escalation_attempts(username, user_event_list)
            alerts.extend(escalation_alerts)
        
        return alerts
    
    def _detect_escalation_attempts(self, username: str, events: List[AuditEvent]) -> List[SecurityAlert]:
        """Detect privilege escalation for specific user."""
        alerts = []
        
        # Look for patterns indicating escalation
        admin_event_count = 0
        failed_admin_attempts = 0
        suspicious_patterns = []
        
        for event in events:
            # Check for admin actions by non-admin users
            if (event.category == EventCategory.ADMIN_ACTION or
                any(action in str(event.details) for action in self.admin_actions)):
                admin_event_count += 1
                
                # Check if this was a failed attempt
                if "denied" in event.message.lower() or "unauthorized" in event.message.lower():
                    failed_admin_attempts += 1
                    suspicious_patterns.append(f"Failed admin action: {event.event_type}")
            
            # Check for role/permission changes
            if "role" in str(event.details).lower() or "permission" in str(event.details).lower():
                if event.username == username:  # User modifying their own permissions
                    suspicious_patterns.append(f"Self-permission modification: {event.event_type}")
        
        # Calculate risk score
        risk_score = (failed_admin_attempts * 2 + len(suspicious_patterns)) / 10
        adjusted_threshold = 0.3 * self.sensitivity
        
        if risk_score > adjusted_threshold and (failed_admin_attempts > 0 or suspicious_patterns):
            alert = SecurityAlert(
                alert_id=f"privesc_{username}_{int(time.time())}",
                alert_type=AlertType.PRIVILEGE_ESCALATION,
                threat_level=ThreatLevel.HIGH,
                title=f"Potential Privilege Escalation by {username}",
                description=f"User {username} showing patterns consistent with privilege escalation: "
                          f"{failed_admin_attempts} failed admin attempts, "
                          f"{len(suspicious_patterns)} suspicious patterns",
                timestamp=datetime.now(timezone.utc),
                source=self.name,
                indicators=suspicious_patterns[:5],  # Limit to first 5
                user_id=username,
                confidence_score=min(1.0, risk_score),
                context={
                    "admin_event_count": admin_event_count,
                    "failed_admin_attempts": failed_admin_attempts,
                    "suspicious_patterns": suspicious_patterns,
                    "risk_score": risk_score
                },
                recommendations=[
                    f"Review {username}'s role assignments and permissions",
                    "Check if account is compromised",
                    "Monitor future activities closely",
                    "Consider temporary access restrictions"
                ]
            )
            alerts.append(alert)
        
        return alerts


class DataExfiltrationDetector(ThreatDetector):
    """Detects potential data exfiltration attempts."""
    
    def __init__(self, 
                 large_access_threshold: int = 100,
                 unusual_time_window: Tuple[int, int] = (22, 6),  # 10 PM to 6 AM
                 sensitivity: float = 0.8):
        super().__init__("data_exfiltration", sensitivity)
        self.large_access_threshold = large_access_threshold
        self.unusual_time_start, self.unusual_time_end = unusual_time_window
    
    def detect(self, events: List[AuditEvent], context: Dict[str, Any]) -> List[SecurityAlert]:
        alerts = []
        
        # Group data access events by user
        user_access_events = defaultdict(list)
        for event in events:
            if (event.category == EventCategory.DATA_ACCESS and 
                event.username):
                user_access_events[event.username].append(event)
        
        for username, access_events in user_access_events.items():
            exfiltration_alerts = self._detect_exfiltration_patterns(username, access_events)
            alerts.extend(exfiltration_alerts)
        
        return alerts
    
    def _detect_exfiltration_patterns(self, username: str, events: List[AuditEvent]) -> List[SecurityAlert]:
        """Detect data exfiltration patterns for specific user."""
        alerts = []
        
        # Check for large volume access
        access_count = len(events)
        adjusted_threshold = self.large_access_threshold * self.sensitivity
        
        if access_count > adjusted_threshold:
            # Check for unusual timing
            unusual_time_count = 0
            for event in events:
                hour = event.timestamp.hour
                if (hour >= self.unusual_time_start or hour <= self.unusual_time_end):
                    unusual_time_count += 1
            
            # Check for diverse resource access
            unique_resources = len(set(event.resource_id for event in events if event.resource_id))
            
            # Calculate suspicion score
            suspicion_score = (
                (access_count / self.large_access_threshold) * 0.4 +
                (unusual_time_count / access_count) * 0.4 +
                (min(unique_resources / 20, 1.0)) * 0.2
            )
            
            if suspicion_score > 0.5:
                threat_level = (ThreatLevel.CRITICAL if suspicion_score > 0.8 
                              else ThreatLevel.HIGH if suspicion_score > 0.6 
                              else ThreatLevel.MEDIUM)
                
                alert = SecurityAlert(
                    alert_id=f"exfil_{username}_{int(time.time())}",
                    alert_type=AlertType.DATA_EXFILTRATION,
                    threat_level=threat_level,
                    title=f"Potential Data Exfiltration by {username}",
                    description=f"User {username} accessed {access_count} resources "
                              f"({unusual_time_count} during unusual hours)",
                    timestamp=datetime.now(timezone.utc),
                    source=self.name,
                    indicators=[
                        f"large_volume_access:{access_count}",
                        f"unusual_time_access:{unusual_time_count}",
                        f"unique_resources:{unique_resources}"
                    ],
                    user_id=username,
                    confidence_score=min(1.0, suspicion_score),
                    context={
                        "access_count": access_count,
                        "unusual_time_count": unusual_time_count,
                        "unique_resources": unique_resources,
                        "suspicion_score": suspicion_score,
                        "accessed_resource_types": list(set(
                            event.resource_type for event in events if event.resource_type
                        ))
                    },
                    recommendations=[
                        f"Immediately investigate {username}'s data access patterns",
                        "Check if account credentials are compromised",
                        "Review data loss prevention policies",
                        "Consider restricting data access temporarily",
                        "Monitor network traffic for data transfers"
                    ]
                )
                alerts.append(alert)
        
        return alerts


class SecurityMonitor:
    """Main security monitoring system that orchestrates threat detection."""
    
    def __init__(self, 
                 audit_logger: SecurityAuditLogger,
                 redis_client=None,
                 monitoring_interval: int = 60,  # seconds
                 event_batch_size: int = 1000):
        """Initialize security monitor.
        
        Args:
            audit_logger: Audit logger instance
            redis_client: Redis client for caching
            monitoring_interval: Monitoring check interval
            event_batch_size: Number of events to process in each batch
        """
        self.audit_logger = audit_logger
        self.redis_client = redis_client or get_redis().get_client()
        self.monitoring_interval = monitoring_interval
        self.event_batch_size = event_batch_size
        
        # Threat detectors
        self.detectors: List[ThreatDetector] = [
            BruteForceDetector(),
            AnomalyDetector(),
            PrivilegeEscalationDetector(),
            DataExfiltrationDetector()
        ]
        
        # Alert storage
        self.alerts: Dict[str, SecurityAlert] = {}
        self.incidents: Dict[str, SecurityIncident] = {}
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[SecurityAlert], None]] = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.last_processed_time = datetime.now(timezone.utc) - timedelta(hours=1)
        
        self.logger = logger.getChild(self.__class__.__name__)
    
    def add_detector(self, detector: ThreatDetector):
        """Add a custom threat detector."""
        self.detectors.append(detector)
        self.logger.info(f"Added threat detector: {detector.name}")
    
    def remove_detector(self, detector_name: str):
        """Remove a threat detector by name."""
        self.detectors = [d for d in self.detectors if d.name != detector_name]
        self.logger.info(f"Removed threat detector: {detector_name}")
    
    def add_alert_callback(self, callback: Callable[[SecurityAlert], None]):
        """Add callback function to be called when alerts are generated."""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start continuous security monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="SecurityMonitor"
        )
        self.monitoring_thread.start()
        
        self.logger.info("Security monitoring started")
    
    def stop_monitoring(self):
        """Stop security monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        self.logger.info("Security monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                start_time = time.time()
                
                # Process new events
                self._process_new_events()
                
                # Process existing alerts (age out old ones)
                self._process_alerts()
                
                processing_time = time.time() - start_time
                sleep_time = max(0, self.monitoring_interval - processing_time)
                
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _process_new_events(self):
        """Process new audit events for threat detection."""
        current_time = datetime.now(timezone.utc)
        
        # Query for new events since last processing
        from .audit_logging import AuditFilter
        audit_filter = AuditFilter(
            start_time=self.last_processed_time,
            end_time=current_time,
            limit=self.event_batch_size
        )
        
        events = self.audit_logger.query_events(audit_filter)
        
        if events:
            self.logger.debug(f"Processing {len(events)} new events")
            
            # Run threat detection
            context = {
                "monitoring_interval": self.monitoring_interval,
                "batch_size": len(events)
            }
            
            for detector in self.detectors:
                if not detector.enabled:
                    continue
                
                try:
                    alerts = detector.detect(events, context)
                    
                    for alert in alerts:
                        self._handle_new_alert(alert)
                        
                except Exception as e:
                    self.logger.error(f"Error in detector {detector.name}: {e}")
        
        self.last_processed_time = current_time
    
    def _handle_new_alert(self, alert: SecurityAlert):
        """Handle a new security alert."""
        # Store alert
        self.alerts[alert.alert_id] = alert
        
        # Store in Redis for persistence
        try:
            self.redis_client.setex(
                f"security_alert:{alert.alert_id}",
                86400,  # 24 hours
                json.dumps(alert.to_dict())
            )
        except Exception as e:
            self.logger.error(f"Failed to store alert in Redis: {e}")
        
        # Log the alert
        self.audit_logger.log_security_event(
            event_type=f"alert_generated_{alert.alert_type.value}",
            risk_score=alert.threat_level.value * 2,  # Convert to 0-10 scale
            threat_indicators=alert.indicators
        )
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
        
        # Auto-create incident for critical alerts
        if alert.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.EMERGENCY]:
            self._create_incident_from_alert(alert)
        
        self.logger.warning(f"Security alert generated: {alert.title} ({alert.threat_level.name})")
    
    def _process_alerts(self):
        """Process existing alerts (cleanup old ones, etc.)."""
        current_time = datetime.now(timezone.utc)
        expired_alert_ids = []
        
        for alert_id, alert in self.alerts.items():
            # Mark alerts older than 24 hours as expired
            age_hours = (current_time - alert.timestamp).total_seconds() / 3600
            if age_hours > 24:
                expired_alert_ids.append(alert_id)
        
        # Clean up expired alerts
        for alert_id in expired_alert_ids:
            del self.alerts[alert_id]
            try:
                self.redis_client.delete(f"security_alert:{alert_id}")
            except Exception:
                pass
    
    def _create_incident_from_alert(self, alert: SecurityAlert):
        """Create a security incident from a critical alert."""
        import uuid
        
        incident_id = str(uuid.uuid4())
        
        incident = SecurityIncident(
            incident_id=incident_id,
            title=f"Security Incident: {alert.title}",
            description=f"Critical security alert triggered: {alert.description}",
            status=IncidentStatus.OPEN,
            severity=alert.threat_level,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            alerts=[alert.alert_id],
            tags={alert.alert_type.value, "auto_created"}
        )
        
        incident.add_timeline_entry(
            "incident_created",
            f"Incident automatically created from alert {alert.alert_id}",
            "system"
        )
        
        self.incidents[incident_id] = incident
        
        self.logger.critical(f"Security incident created: {incident.title}")
        
        return incident_id
    
    def get_alerts(self, 
                   threat_level: Optional[ThreatLevel] = None,
                   alert_type: Optional[AlertType] = None,
                   acknowledged: Optional[bool] = None,
                   limit: int = 100) -> List[SecurityAlert]:
        """Get security alerts based on filters."""
        alerts = list(self.alerts.values())
        
        # Apply filters
        if threat_level is not None:
            alerts = [a for a in alerts if a.threat_level == threat_level]
        
        if alert_type is not None:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        return alerts[:limit]
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge a security alert."""
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        alert.acknowledged = True
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.now(timezone.utc)
        
        # Update in Redis
        try:
            self.redis_client.setex(
                f"security_alert:{alert_id}",
                86400,
                json.dumps(alert.to_dict())
            )
        except Exception as e:
            self.logger.error(f"Failed to update alert in Redis: {e}")
        
        self.logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
        return True
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security monitoring dashboard data."""
        current_time = datetime.now(timezone.utc)
        
        # Count alerts by threat level
        alert_counts = defaultdict(int)
        for alert in self.alerts.values():
            alert_counts[alert.threat_level.name] += 1
        
        # Count alerts by type
        type_counts = defaultdict(int)
        for alert in self.alerts.values():
            type_counts[alert.alert_type.value] += 1
        
        # Recent activity
        recent_alerts = [
            a for a in self.alerts.values() 
            if (current_time - a.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        # Detector status
        detector_status = {
            detector.name: {
                "enabled": detector.enabled,
                "sensitivity": detector.sensitivity
            }
            for detector in self.detectors
        }
        
        return {
            "monitoring_active": self.monitoring_active,
            "total_alerts": len(self.alerts),
            "unacknowledged_alerts": len([a for a in self.alerts.values() if not a.acknowledged]),
            "critical_alerts": alert_counts.get("CRITICAL", 0) + alert_counts.get("EMERGENCY", 0),
            "alert_counts_by_level": dict(alert_counts),
            "alert_counts_by_type": dict(type_counts),
            "recent_alerts": len(recent_alerts),
            "total_incidents": len(self.incidents),
            "open_incidents": len([i for i in self.incidents.values() if i.status == IncidentStatus.OPEN]),
            "detector_status": detector_status,
            "last_processed": self.last_processed_time.isoformat()
        }


class SecurityAlerts:
    """Security alerting and notification system."""
    
    def __init__(self, security_monitor: SecurityMonitor):
        """Initialize security alerts."""
        self.security_monitor = security_monitor
        self.notification_handlers: Dict[str, Callable] = {}
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Register as alert callback
        self.security_monitor.add_alert_callback(self._handle_alert)
    
    def add_notification_handler(self, name: str, handler: Callable[[SecurityAlert], None]):
        """Add notification handler for alerts."""
        self.notification_handlers[name] = handler
        self.logger.info(f"Added notification handler: {name}")
    
    def _handle_alert(self, alert: SecurityAlert):
        """Handle new security alert."""
        # Send notifications based on threat level
        if alert.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.EMERGENCY]:
            self._send_notifications(alert, urgent=True)
        elif alert.threat_level == ThreatLevel.HIGH:
            self._send_notifications(alert, urgent=False)
        # Medium and Low alerts are logged but not immediately notified
    
    def _send_notifications(self, alert: SecurityAlert, urgent: bool = False):
        """Send notifications for security alert."""
        for name, handler in self.notification_handlers.items():
            try:
                handler(alert)
                self.logger.info(f"Notification sent via {name} for alert {alert.alert_id}")
            except Exception as e:
                self.logger.error(f"Failed to send notification via {name}: {e}")


class ThreatDetection:
    """High-level threat detection orchestration."""
    
    def __init__(self, 
                 security_monitor: SecurityMonitor,
                 machine_learning_enabled: bool = False):
        """Initialize threat detection system."""
        self.security_monitor = security_monitor
        self.ml_enabled = machine_learning_enabled and NUMPY_AVAILABLE
        
        # Threat intelligence feeds (mock implementation)
        self.threat_intelligence = {
            "malicious_ips": set(),
            "known_attack_patterns": [],
            "compromised_accounts": set()
        }
        
        self.logger = logger.getChild(self.__class__.__name__)
    
    def update_threat_intelligence(self, 
                                 malicious_ips: Optional[Set[str]] = None,
                                 attack_patterns: Optional[List[str]] = None,
                                 compromised_accounts: Optional[Set[str]] = None):
        """Update threat intelligence data."""
        if malicious_ips:
            self.threat_intelligence["malicious_ips"].update(malicious_ips)
        
        if attack_patterns:
            self.threat_intelligence["known_attack_patterns"].extend(attack_patterns)
        
        if compromised_accounts:
            self.threat_intelligence["compromised_accounts"].update(compromised_accounts)
        
        self.logger.info("Threat intelligence updated")
    
    def analyze_threat_landscape(self) -> Dict[str, Any]:
        """Analyze current threat landscape."""
        alerts = self.security_monitor.get_alerts(limit=1000)
        
        # Threat trends
        threat_types = defaultdict(int)
        for alert in alerts:
            threat_types[alert.alert_type.value] += 1
        
        # Geographic analysis (if IP data available)
        ip_countries = defaultdict(int)
        for alert in alerts:
            if alert.ip_address:
                # In real implementation, use GeoIP lookup
                ip_countries["unknown"] += 1
        
        # Time-based analysis
        hourly_distribution = defaultdict(int)
        for alert in alerts:
            hour = alert.timestamp.hour
            hourly_distribution[hour] += 1
        
        return {
            "total_threats": len(alerts),
            "threat_types": dict(threat_types),
            "geographic_distribution": dict(ip_countries),
            "hourly_distribution": dict(hourly_distribution),
            "threat_intelligence_stats": {
                "malicious_ips": len(self.threat_intelligence["malicious_ips"]),
                "attack_patterns": len(self.threat_intelligence["known_attack_patterns"]),
                "compromised_accounts": len(self.threat_intelligence["compromised_accounts"])
            }
        }


class IncidentResponse:
    """Automated incident response system."""
    
    def __init__(self, security_monitor: SecurityMonitor):
        """Initialize incident response system."""
        self.security_monitor = security_monitor
        self.response_playbooks: Dict[AlertType, List[Callable]] = {}
        self.logger = logger.getChild(self.__class__.__name__)
    
    def add_response_playbook(self, alert_type: AlertType, actions: List[Callable]):
        """Add automated response playbook for alert type."""
        self.response_playbooks[alert_type] = actions
        self.logger.info(f"Added response playbook for {alert_type.value}")
    
    def execute_response(self, alert: SecurityAlert) -> List[str]:
        """Execute automated response for security alert."""
        actions_taken = []
        
        if alert.alert_type in self.response_playbooks:
            for action in self.response_playbooks[alert.alert_type]:
                try:
                    result = action(alert)
                    actions_taken.append(f"Executed {action.__name__}: {result}")
                    self.logger.info(f"Response action executed for {alert.alert_id}: {action.__name__}")
                except Exception as e:
                    error_msg = f"Failed to execute {action.__name__}: {e}"
                    actions_taken.append(error_msg)
                    self.logger.error(error_msg)
        
        return actions_taken
    
    def create_incident(self, 
                       title: str,
                       description: str,
                       severity: ThreatLevel,
                       assigned_to: Optional[str] = None) -> str:
        """Create new security incident."""
        return self.security_monitor._create_incident_from_alert(
            SecurityAlert(
                alert_id="manual_incident",
                alert_type=AlertType.SECURITY_EVENT,
                threat_level=severity,
                title=title,
                description=description,
                timestamp=datetime.now(timezone.utc),
                source="manual"
            )
        )