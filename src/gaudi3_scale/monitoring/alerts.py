"""Alert management and notification system."""

import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class Alert:
    """Alert representation."""
    id: str
    name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    source: str
    labels: Dict[str, str]
    created_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None


class AlertRule:
    """Alert rule definition."""
    
    def __init__(self, name: str, condition: Callable[[Dict[str, Any]], bool],
                 severity: AlertSeverity, message_template: str):
        """Initialize alert rule.
        
        Args:
            name: Alert rule name
            condition: Function that returns True if alert should fire
            severity: Alert severity level
            message_template: Alert message template
        """
        self.name = name
        self.condition = condition
        self.severity = severity
        self.message_template = message_template
        self.cooldown_seconds = 300  # 5 minutes default cooldown
        self.last_fired = None


class AlertManager:
    """Manages alerts and notifications for training infrastructure."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_handlers: List[Callable[[Alert], None]] = []
        self.suppression_rules: List[Dict[str, Any]] = []
        
        # Setup default alert rules
        self._setup_default_rules()
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add alert rule.
        
        Args:
            rule: Alert rule to add
        """
        self.rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def add_notification_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add notification handler.
        
        Args:
            handler: Function to handle alert notifications
        """
        self.notification_handlers.append(handler)
        logger.info("Added notification handler")
    
    def check_alerts(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Check all alert rules and fire alerts if conditions are met.
        
        Args:
            metrics: Current metrics to evaluate
            
        Returns:
            List of newly fired alerts
        """
        new_alerts = []
        current_time = datetime.now()
        
        for rule in self.rules:
            # Check cooldown period
            if (rule.last_fired and 
                current_time - rule.last_fired < timedelta(seconds=rule.cooldown_seconds)):
                continue
            
            # Evaluate rule condition
            try:
                if rule.condition(metrics):
                    alert = self._create_alert(rule, metrics)
                    
                    # Check if this alert is already active
                    alert_key = f"{rule.name}_{alert.source}"
                    if alert_key not in self.active_alerts:
                        self.active_alerts[alert_key] = alert
                        self.alert_history.append(alert)
                        new_alerts.append(alert)
                        rule.last_fired = current_time
                        
                        # Send notifications
                        self._send_notifications(alert)
                        
                        logger.warning(f"Alert fired: {alert.name} - {alert.message}")
            
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule.name}: {e}")
        
        return new_alerts
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve an active alert.
        
        Args:
            alert_id: Alert identifier
            resolved_by: Who resolved the alert
            
        Returns:
            True if alert was resolved
        """
        # Find alert in active alerts
        for key, alert in self.active_alerts.items():
            if alert.id == alert_id:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()
                del self.active_alerts[key]
                
                logger.info(f"Alert resolved: {alert.name} by {resolved_by}")
                return True
        
        return False
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert.
        
        Args:
            alert_id: Alert identifier
            acknowledged_by: Who acknowledged the alert
            
        Returns:
            True if alert was acknowledged
        """
        # Find alert in active alerts
        for alert in self.active_alerts.values():
            if alert.id == alert_id:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.now()
                alert.acknowledged_by = acknowledged_by
                
                logger.info(f"Alert acknowledged: {alert.name} by {acknowledged_by}")
                return True
        
        return False
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts.
        
        Args:
            severity: Optional severity filter
            
        Returns:
            List of active alerts
        """
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return sorted(alerts, key=lambda a: a.created_at, reverse=True)
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics.
        
        Returns:
            Alert summary information
        """
        active_alerts = list(self.active_alerts.values())
        
        # Count by severity
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len([
                a for a in active_alerts if a.severity == severity
            ])
        
        # Recent alerts (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_alerts = [
            a for a in self.alert_history 
            if a.created_at > recent_cutoff
        ]
        
        return {
            "active_alerts": len(active_alerts),
            "total_alerts_24h": len(recent_alerts),
            "severity_breakdown": severity_counts,
            "critical_alerts": severity_counts.get("critical", 0),
            "emergency_alerts": severity_counts.get("emergency", 0)
        }
    
    def _setup_default_rules(self) -> None:
        """Setup default alert rules for common issues."""
        
        # High HPU utilization
        self.add_rule(AlertRule(
            name="high_hpu_utilization",
            condition=lambda m: m.get("hpu_utilization", 0) > 95,
            severity=AlertSeverity.WARNING,
            message_template="HPU utilization is {hpu_utilization:.1f}%, exceeding 95% threshold"
        ))
        
        # Low HPU utilization (waste of resources)
        self.add_rule(AlertRule(
            name="low_hpu_utilization",
            condition=lambda m: m.get("hpu_utilization", 100) < 30,
            severity=AlertSeverity.INFO,
            message_template="HPU utilization is {hpu_utilization:.1f}%, below 30% threshold"
        ))
        
        # High memory usage
        self.add_rule(AlertRule(
            name="high_memory_usage",
            condition=lambda m: m.get("memory_usage_percent", 0) > 90,
            severity=AlertSeverity.CRITICAL,
            message_template="Memory usage is {memory_usage_percent:.1f}%, exceeding 90% threshold"
        ))
        
        # Training job failure
        self.add_rule(AlertRule(
            name="training_job_failed",
            condition=lambda m: m.get("job_status") == "failed",
            severity=AlertSeverity.CRITICAL,
            message_template="Training job {job_id} failed: {error_message}"
        ))
        
        # High training cost
        self.add_rule(AlertRule(
            name="high_training_cost",
            condition=lambda m: m.get("hourly_cost", 0) > 500,
            severity=AlertSeverity.WARNING,
            message_template="Training cost is ${hourly_cost:.2f}/hour, exceeding $500/hour threshold"
        ))
        
        # HPU temperature warning
        self.add_rule(AlertRule(
            name="high_hpu_temperature",
            condition=lambda m: m.get("hpu_temperature", 0) > 85,
            severity=AlertSeverity.WARNING,
            message_template="HPU temperature is {hpu_temperature:.1f}°C, exceeding 85°C threshold"
        ))
        
        # Loss not decreasing
        self.add_rule(AlertRule(
            name="loss_not_decreasing",
            condition=lambda m: (m.get("loss_trend", 0) > 0 and 
                                m.get("training_steps", 0) > 1000),
            severity=AlertSeverity.WARNING,
            message_template="Training loss has not decreased in recent steps (trend: {loss_trend:.4f})"
        ))
        
        # Gradient explosion
        self.add_rule(AlertRule(
            name="gradient_explosion",
            condition=lambda m: m.get("gradient_norm", 0) > 10.0,
            severity=AlertSeverity.CRITICAL,
            message_template="Gradient norm is {gradient_norm:.2f}, indicating potential gradient explosion"
        ))
        
        # Low throughput
        self.add_rule(AlertRule(
            name="low_throughput",
            condition=lambda m: (m.get("throughput", float('inf')) < 
                                m.get("expected_throughput", 0) * 0.7),
            severity=AlertSeverity.WARNING,
            message_template="Throughput is {throughput:.0f} tokens/sec, below 70% of expected {expected_throughput:.0f}"
        ))
        
        # Disk space low
        self.add_rule(AlertRule(
            name="low_disk_space",
            condition=lambda m: m.get("disk_usage_percent", 0) > 85,
            severity=AlertSeverity.WARNING,
            message_template="Disk usage is {disk_usage_percent:.1f}%, exceeding 85% threshold"
        ))
    
    def _create_alert(self, rule: AlertRule, metrics: Dict[str, Any]) -> Alert:
        """Create alert from rule and metrics.
        
        Args:
            rule: Alert rule that fired
            metrics: Metrics that triggered the alert
            
        Returns:
            Created alert
        """
        # Generate unique alert ID
        alert_id = f"{rule.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Format message with metrics
        try:
            message = rule.message_template.format(**metrics)
        except KeyError:
            message = rule.message_template
        
        # Extract labels from metrics
        labels = {
            "rule": rule.name,
            "severity": rule.severity.value,
            "source": metrics.get("source", "unknown")
        }
        
        # Add relevant metric labels
        for key in ["job_id", "model_name", "node_id", "user_id"]:
            if key in metrics:
                labels[key] = str(metrics[key])
        
        return Alert(
            id=alert_id,
            name=rule.name,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            message=message,
            source=metrics.get("source", "gaudi3-scale"),
            labels=labels,
            created_at=datetime.now()
        )
    
    def _send_notifications(self, alert: Alert) -> None:
        """Send alert notifications.
        
        Args:
            alert: Alert to send notifications for
        """
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error sending alert notification: {e}")


def create_slack_notification_handler(slack_notifier) -> Callable[[Alert], None]:
    """Create Slack notification handler.
    
    Args:
        slack_notifier: Slack notifier instance
        
    Returns:
        Notification handler function
    """
    def handle_alert(alert: Alert) -> None:
        """Handle alert by sending to Slack."""
        severity_icons = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.EMERGENCY: "🔥"
        }
        
        icon = severity_icons.get(alert.severity, "⚠️")
        title = f"{icon} {alert.severity.value.upper()}: {alert.name}"
        
        fields = {
            "Severity": alert.severity.value.upper(),
            "Source": alert.source,
            "Time": alert.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        }
        
        # Add relevant labels as fields
        for key, value in alert.labels.items():
            if key not in ["rule", "severity", "source"]:
                fields[key.replace("_", " ").title()] = value
        
        slack_notifier.send_custom_notification(
            title=title,
            message_text=alert.message,
            fields=fields,
            color="#ff0000" if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY] else "#ffaa00"
        )
    
    return handle_alert


def create_email_notification_handler(email_client, recipients: List[str]) -> Callable[[Alert], None]:
    """Create email notification handler.
    
    Args:
        email_client: Email client instance
        recipients: List of email recipients
        
    Returns:
        Notification handler function
    """
    def handle_alert(alert: Alert) -> None:
        """Handle alert by sending email."""
        subject = f"[GAUDI3-SCALE] {alert.severity.value.upper()}: {alert.name}"
        
        # Create email body
        body = f"""
Alert Details:
- Name: {alert.name}
- Severity: {alert.severity.value.upper()}
- Status: {alert.status.value}
- Source: {alert.source}
- Time: {alert.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")}

Message:
{alert.message}

Labels:
"""
        for key, value in alert.labels.items():
            body += f"- {key}: {value}\n"
        
        body += f"""

Alert ID: {alert.id}

This alert was generated by the Gaudi 3 Scale monitoring system.
"""
        
        email_client.send_email(
            to_addresses=recipients,
            subject=subject,
            html_body=body.replace("\n", "<br>"),
            text_body=body
        )
    
    return handle_alert