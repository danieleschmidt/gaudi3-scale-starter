"""Email client for training notifications."""

import logging
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional, Dict, Any
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr

logger = logging.getLogger(__name__)


class EmailConfig(BaseModel):
    """Email configuration settings."""
    model_config = ConfigDict(extra='forbid')
    
    smtp_server: str
    smtp_port: int = 587
    username: str
    password: str
    use_tls: bool = True
    from_address: EmailStr
    from_name: str = "Gaudi 3 Scale"


class TrainingNotification(BaseModel):
    """Training notification data."""
    model_config = ConfigDict(extra='forbid')
    
    event_type: str  # started, completed, failed, checkpoint
    model_name: str
    timestamp: datetime
    details: Dict[str, Any]
    metrics: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None


class EmailClient:
    """Email client for sending training notifications.
    
    Provides functionality to send structured email notifications
    about training events, status updates, and alerts.
    """
    
    def __init__(self, config: EmailConfig):
        """Initialize email client.
        
        Args:
            config: Email configuration settings
        """
        self.config = config
    
    def _create_connection(self) -> smtplib.SMTP:
        """Create SMTP connection.
        
        Returns:
            SMTP connection object
            
        Raises:
            smtplib.SMTPException: If connection fails
        """
        try:
            if self.config.use_tls:
                server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
                server.starttls(context=ssl.create_default_context())
            else:
                server = smtplib.SMTP_SSL(self.config.smtp_server, self.config.smtp_port)
            
            server.login(self.config.username, self.config.password)
            return server
        except Exception as e:
            logger.error(f"Failed to create SMTP connection: {e}")
            raise
    
    def send_email(self, to_addresses: List[str], subject: str, 
                   html_body: str, text_body: Optional[str] = None) -> bool:
        """Send email to recipients.
        
        Args:
            to_addresses: List of recipient email addresses
            subject: Email subject
            html_body: HTML email body
            text_body: Plain text email body (optional)
            
        Returns:
            True if email was sent successfully
        """
        try:
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = f"{self.config.from_name} <{self.config.from_address}>"
            message["To"] = ", ".join(to_addresses)
            
            # Add text version if provided
            if text_body:
                text_part = MIMEText(text_body, "plain")
                message.attach(text_part)
            
            # Add HTML version
            html_part = MIMEText(html_body, "html")
            message.attach(html_part)
            
            # Send email
            with self._create_connection() as server:
                server.sendmail(self.config.from_address, to_addresses, message.as_string())
            
            logger.info(f"Email sent successfully to {len(to_addresses)} recipients")
            return True
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def send_training_notification(self, notification: TrainingNotification, 
                                 to_addresses: List[str]) -> bool:
        """Send training event notification.
        
        Args:
            notification: Training notification data
            to_addresses: List of recipient email addresses
            
        Returns:
            True if notification was sent successfully
        """
        subject = self._create_subject(notification)
        html_body = self._create_html_body(notification)
        text_body = self._create_text_body(notification)
        
        return self.send_email(to_addresses, subject, html_body, text_body)
    
    def _create_subject(self, notification: TrainingNotification) -> str:
        """Create email subject for training notification.
        
        Args:
            notification: Training notification data
            
        Returns:
            Email subject string
        """
        status_emoji = {
            "started": "🚀",
            "completed": "✅",
            "failed": "❌",
            "checkpoint": "💾"
        }
        
        emoji = status_emoji.get(notification.event_type, "📊")
        status = notification.event_type.title()
        
        return f"{emoji} Training {status}: {notification.model_name}"
    
    def _create_html_body(self, notification: TrainingNotification) -> str:
        """Create HTML email body for training notification.
        
        Args:
            notification: Training notification data
            
        Returns:
            HTML email body
        """
        style = """
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px 8px 0 0; }
            .content { padding: 20px; background: #f9f9f9; }
            .metrics { background: white; padding: 15px; border-radius: 8px; margin: 15px 0; }
            .metric { display: inline-block; margin: 0 15px 10px 0; padding: 8px 12px; background: #e8f4fd; border-radius: 4px; }
            .error { background: #fff5f5; border-left: 4px solid #e53e3e; padding: 15px; margin: 15px 0; }
            .footer { background: #2d3748; color: white; padding: 15px; text-align: center; border-radius: 0 0 8px 8px; }
            .timestamp { color: #666; font-size: 0.9em; }
        </style>
        """
        
        # Status-specific content
        if notification.event_type == "started":
            status_color = "#3182ce"
            status_message = "Training has started successfully"
        elif notification.event_type == "completed":
            status_color = "#38a169"
            status_message = "Training completed successfully"
        elif notification.event_type == "failed":
            status_color = "#e53e3e"
            status_message = "Training failed"
        else:  # checkpoint
            status_color = "#805ad5"
            status_message = "Training checkpoint saved"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Training Notification</title>
            {style}
        </head>
        <body>
            <div class="header" style="background-color: {status_color};">
                <h2>🔥 Gaudi 3 Scale Training Update</h2>
                <p>{status_message}</p>
            </div>
            
            <div class="content">
                <h3>Model: {notification.model_name}</h3>
                <p class="timestamp">Timestamp: {notification.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                
                <div class="metrics">
                    <h4>Training Details</h4>
        """
        
        # Add training details
        for key, value in notification.details.items():
            html += f"<div class=\"metric\"><strong>{key.replace('_', ' ').title()}:</strong> {value}</div>"
        
        # Add metrics if available
        if notification.metrics:
            html += "<h4>Performance Metrics</h4>"
            for metric, value in notification.metrics.items():
                if isinstance(value, float):
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)
                html += f"<div class=\"metric\"><strong>{metric.replace('_', ' ').title()}:</strong> {formatted_value}</div>"
        
        html += "</div>"
        
        # Add error message if present
        if notification.error_message:
            html += f"""
                <div class="error">
                    <h4>Error Details</h4>
                    <pre>{notification.error_message}</pre>
                </div>
            """
        
        html += """
            </div>
            
            <div class="footer">
                <p>This notification was sent by Gaudi 3 Scale Training Platform</p>
                <p>Intel Gaudi 3 HPU Infrastructure for Production ML Training</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_text_body(self, notification: TrainingNotification) -> str:
        """Create plain text email body for training notification.
        
        Args:
            notification: Training notification data
            
        Returns:
            Plain text email body
        """
        text = f"""
GAUDI 3 SCALE TRAINING NOTIFICATION
==================================

Event: {notification.event_type.upper()}
Model: {notification.model_name}
Timestamp: {notification.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

Training Details:
"""
        
        for key, value in notification.details.items():
            text += f"  {key.replace('_', ' ').title()}: {value}\n"
        
        if notification.metrics:
            text += "\nPerformance Metrics:\n"
            for metric, value in notification.metrics.items():
                if isinstance(value, float):
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)
                text += f"  {metric.replace('_', ' ').title()}: {formatted_value}\n"
        
        if notification.error_message:
            text += f"\nError Details:\n{notification.error_message}\n"
        
        text += """
---
This notification was sent by Gaudi 3 Scale Training Platform
Intel Gaudi 3 HPU Infrastructure for Production ML Training
"""
        
        return text
    
    def send_training_summary(self, model_name: str, training_stats: Dict[str, Any], 
                            to_addresses: List[str]) -> bool:
        """Send training summary report.
        
        Args:
            model_name: Name of the trained model
            training_stats: Complete training statistics
            to_addresses: List of recipient email addresses
            
        Returns:
            True if summary was sent successfully
        """
        subject = f"📊 Training Summary: {model_name}"
        
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .summary {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .stat {{ background: white; padding: 15px; margin: 10px; border-radius: 8px; display: inline-block; min-width: 200px; }}
            </style>
        </head>
        <body>
            <h2>🎯 Training Summary Report</h2>
            <h3>Model: {model_name}</h3>
            
            <div class="summary">
                <h4>Training Statistics</h4>
        """
        
        for key, value in training_stats.items():
            html_body += f"<div class=\"stat\"><strong>{key.replace('_', ' ').title()}:</strong> {value}</div>"
        
        html_body += """
            </div>
            
            <p>Training completed successfully on Intel Gaudi 3 HPUs.</p>
            <p><em>Gaudi 3 Scale Training Platform</em></p>
        </body>
        </html>
        """
        
        return self.send_email(to_addresses, subject, html_body)