"""Slack notifications for training events."""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    import requests
except ImportError:
    # Fallback for environments without requests
    requests = None

try:
    from pydantic import BaseModel, ConfigDict, HttpUrl
except ImportError:
    # Fallback for environments without pydantic
    from typing import Any
    
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class ConfigDict:
        def __init__(self, **kwargs):
            pass
    
    # Simple HttpUrl fallback - just use str
    HttpUrl = str

logger = logging.getLogger(__name__)


class SlackConfig(BaseModel):
    """Slack configuration settings."""
    model_config = ConfigDict(extra='forbid')
    
    webhook_url: HttpUrl
    channel: str = "#training"
    username: str = "Gaudi3-Scale"
    icon_emoji: str = ":rocket:"


class SlackMessage(BaseModel):
    """Slack message structure."""
    model_config = ConfigDict(extra='forbid')
    
    text: str
    channel: Optional[str] = None
    username: Optional[str] = None
    icon_emoji: Optional[str] = None
    attachments: Optional[List[Dict[str, Any]]] = None
    blocks: Optional[List[Dict[str, Any]]] = None


class SlackNotifier:
    """Slack notifier for training events.
    
    Sends structured notifications to Slack channels about
    training progress, completions, failures, and metrics.
    """
    
    def __init__(self, config: SlackConfig):
        """Initialize Slack notifier.
        
        Args:
            config: Slack configuration settings
        """
        if requests is None:
            raise ImportError("requests library is required for Slack notifications. Install with: pip install requests")
        
        self.config = config
        self.session = requests.Session()
    
    def send_message(self, message: SlackMessage) -> bool:
        """Send message to Slack.
        
        Args:
            message: Slack message to send
            
        Returns:
            True if message was sent successfully
        """
        payload = {
            "text": message.text,
            "channel": message.channel or self.config.channel,
            "username": message.username or self.config.username,
            "icon_emoji": message.icon_emoji or self.config.icon_emoji
        }
        
        if message.attachments:
            payload["attachments"] = message.attachments
        
        if message.blocks:
            payload["blocks"] = message.blocks
        
        try:
            response = self.session.post(
                str(self.config.webhook_url),
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            logger.info("Slack notification sent successfully")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    def notify_training_started(self, model_name: str, config: Dict[str, Any]) -> bool:
        """Send training started notification.
        
        Args:
            model_name: Name of the model being trained
            config: Training configuration
            
        Returns:
            True if notification was sent successfully
        """
        message = SlackMessage(
            text=f"🚀 Training Started: {model_name}",
            blocks=[
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"🚀 Training Started: {model_name}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Model:*\n{model_name}"
                        },
                        {
                            "type": "mrkdwn", 
                            "text": f"*Started:*\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
                        }
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*Configuration:*"
                    }
                }
            ]
        )
        
        # Add configuration fields
        config_fields = []
        for key, value in config.items():
            if key not in ['password', 'token', 'secret']:  # Skip sensitive data
                config_fields.append({
                    "type": "mrkdwn",
                    "text": f"*{key.replace('_', ' ').title()}:*\n{value}"
                })
        
        if config_fields:
            message.blocks.append({
                "type": "section",
                "fields": config_fields[:10]  # Limit to 10 fields
            })
        
        message.blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": "🔥 Powered by Intel Gaudi 3 HPUs"
                }
            ]
        })
        
        return self.send_message(message)
    
    def notify_training_completed(self, model_name: str, metrics: Dict[str, Any], 
                                duration: str, artifacts_url: Optional[str] = None) -> bool:
        """Send training completed notification.
        
        Args:
            model_name: Name of the trained model
            metrics: Training metrics
            duration: Training duration
            artifacts_url: URL to download artifacts
            
        Returns:
            True if notification was sent successfully
        """
        message = SlackMessage(
            text=f"✅ Training Completed: {model_name}",
            blocks=[
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"✅ Training Completed: {model_name}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Model:*\n{model_name}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Duration:*\n{duration}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Completed:*\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
                        }
                    ]
                }
            ]
        )
        
        # Add metrics section
        if metrics:
            metric_fields = []
            for key, value in metrics.items():
                if isinstance(value, float):
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)
                
                metric_fields.append({
                    "type": "mrkdwn",
                    "text": f"*{key.replace('_', ' ').title()}:*\n{formatted_value}"
                })
            
            if metric_fields:
                message.blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*📊 Training Metrics:*"
                    }
                })
                message.blocks.append({
                    "type": "section",
                    "fields": metric_fields[:10]  # Limit to 10 fields
                })
        
        # Add artifacts link if available
        if artifacts_url:
            message.blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*📦 Model Artifacts:*\n<{artifacts_url}|Download Model>"
                }
            })
        
        message.blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": "🎉 Success! Model ready for deployment."
                }
            ]
        })
        
        return self.send_message(message)
    
    def notify_training_failed(self, model_name: str, error_message: str, 
                             duration: str, logs_url: Optional[str] = None) -> bool:
        """Send training failed notification.
        
        Args:
            model_name: Name of the model that failed
            error_message: Error description
            duration: Training duration before failure
            logs_url: URL to view logs
            
        Returns:
            True if notification was sent successfully
        """
        message = SlackMessage(
            text=f"❌ Training Failed: {model_name}",
            blocks=[
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"❌ Training Failed: {model_name}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Model:*\n{model_name}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Duration:*\n{duration}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Failed:*\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
                        }
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*🚨 Error:*\n```{error_message[:500]}{'...' if len(error_message) > 500 else ''}```"
                    }
                }
            ]
        )
        
        # Add logs link if available
        if logs_url:
            message.blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*📝 Logs:*\n<{logs_url}|View Full Logs>"
                }
            })
        
        message.blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": "🔧 Check logs and configuration, then retry training."
                }
            ]
        })
        
        return self.send_message(message)
    
    def notify_checkpoint_saved(self, model_name: str, epoch: int, 
                              metrics: Dict[str, Any], checkpoint_path: str) -> bool:
        """Send checkpoint saved notification.
        
        Args:
            model_name: Name of the model
            epoch: Current epoch number
            metrics: Current metrics
            checkpoint_path: Path to saved checkpoint
            
        Returns:
            True if notification was sent successfully
        """
        message = SlackMessage(
            text=f"💾 Checkpoint Saved: {model_name} (Epoch {epoch})",
            blocks=[
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"💾 *Checkpoint Saved:* {model_name} (Epoch {epoch})"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Epoch:*\n{epoch}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Time:*\n{datetime.now().strftime('%H:%M:%S UTC')}"
                        }
                    ]
                }
            ]
        )
        
        # Add key metrics
        if metrics:
            key_metrics = ["loss", "accuracy", "val_loss", "val_accuracy", "learning_rate"]
            metric_fields = []
            
            for key in key_metrics:
                if key in metrics:
                    value = metrics[key]
                    if isinstance(value, float):
                        formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = str(value)
                    
                    metric_fields.append({
                        "type": "mrkdwn",
                        "text": f"*{key.replace('_', ' ').title()}:*\n{formatted_value}"
                    })
            
            if metric_fields:
                message.blocks.append({
                    "type": "section",
                    "fields": metric_fields
                })
        
        return self.send_message(message)
    
    def notify_resource_alert(self, alert_type: str, message_text: str, 
                            severity: str = "warning") -> bool:
        """Send resource alert notification.
        
        Args:
            alert_type: Type of alert (memory, disk, network, etc.)
            message_text: Alert message
            severity: Alert severity (info, warning, critical)
            
        Returns:
            True if notification was sent successfully
        """
        severity_icons = {
            "info": "ℹ️",
            "warning": "⚠️", 
            "critical": "🚨"
        }
        
        severity_colors = {
            "info": "#36a64f",
            "warning": "#ffaa00",
            "critical": "#ff0000"
        }
        
        icon = severity_icons.get(severity, "⚠️")
        color = severity_colors.get(severity, "#ffaa00")
        
        message = SlackMessage(
            text=f"{icon} Resource Alert: {alert_type}",
            attachments=[
                {
                    "color": color,
                    "fields": [
                        {
                            "title": f"{alert_type.title()} Alert",
                            "value": message_text,
                            "short": False
                        },
                        {
                            "title": "Severity",
                            "value": severity.upper(),
                            "short": True
                        },
                        {
                            "title": "Time",
                            "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
                            "short": True
                        }
                    ],
                    "footer": "Gaudi 3 Scale Monitoring",
                    "ts": int(datetime.now().timestamp())
                }
            ]
        )
        
        return self.send_message(message)
    
    def send_custom_notification(self, title: str, message_text: str, 
                               fields: Optional[Dict[str, str]] = None,
                               color: str = "#36a64f") -> bool:
        """Send custom notification.
        
        Args:
            title: Notification title
            message_text: Main message
            fields: Optional additional fields
            color: Message color
            
        Returns:
            True if notification was sent successfully
        """
        attachment_fields = []
        if fields:
            for key, value in fields.items():
                attachment_fields.append({
                    "title": key.replace('_', ' ').title(),
                    "value": str(value),
                    "short": len(str(value)) < 30
                })
        
        message = SlackMessage(
            text=title,
            attachments=[
                {
                    "color": color,
                    "text": message_text,
                    "fields": attachment_fields,
                    "footer": "Gaudi 3 Scale Platform",
                    "ts": int(datetime.now().timestamp())
                }
            ]
        )
        
        return self.send_message(message)