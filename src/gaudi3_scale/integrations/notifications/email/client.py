"""Email client for training notifications."""

import logging
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

from ....optional_deps import PYDANTIC, OptionalDependencyError, warn_missing_dependency

# Import pydantic conditionally
if PYDANTIC:
    try:
        from pydantic import BaseModel, ConfigDict, EmailStr
    except ImportError:
        # Handle case where pydantic is available but email-validator is not
        from pydantic import BaseModel, ConfigDict
        EmailStr = str
        warn_missing_dependency('email-validator', 'Email validation', 
                               'Basic string validation will be used instead')
else:
    # Fallback base classes when pydantic is not available
    class BaseModel:
        def __init__(self, **data):
            for key, value in data.items():
                setattr(self, key, value)
        
        def dict(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    class ConfigDict(dict):
        pass
    
    EmailStr = str

logger = logging.getLogger(__name__)


class EmailConfig(BaseModel):
    """Email configuration settings."""
    
    def __init__(self, **data):
        if PYDANTIC:
            # Use pydantic validation if available
            super().__init__(**data)
        else:
            # Manual validation for required fields
            required_fields = ['smtp_server', 'username', 'password', 'from_address']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Required field '{field}' is missing")
            
            # Set defaults
            data.setdefault('smtp_port', 587)
            data.setdefault('use_tls', True)
            data.setdefault('from_name', 'Gaudi 3 Scale')
            
            super().__init__(**data)
    
    # Define attributes for type hints and IDE support
    smtp_server: str
    smtp_port: int = 587
    username: str
    password: str
    use_tls: bool = True
    from_address: str  # Would be EmailStr if pydantic with email-validator is available
    from_name: str = "Gaudi 3 Scale"


class TrainingNotification(BaseModel):
    """Training notification data."""
    
    def __init__(self, **data):
        if PYDANTIC:
            super().__init__(**data)
        else:
            # Manual validation for required fields
            required_fields = ['subject', 'recipient']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Required field '{field}' is missing")
            
            # Set defaults
            data.setdefault('timestamp', datetime.utcnow())
            
            super().__init__(**data)
    
    subject: str
    recipient: str  # Would be EmailStr if pydantic with email-validator is available
    message: str = ""
    training_job_id: Optional[str] = None
    status: str = "unknown"
    timestamp: datetime = datetime.utcnow()


class EmailClient:
    """Email client for sending training notifications."""
    
    def __init__(self, config):
        """Initialize email client.
        
        Args:
            config: Email configuration (EmailConfig instance or dict)
        """
        if isinstance(config, dict):
            self.config = EmailConfig(**config)
        else:
            self.config = config
        
        self.logger = logger.getChild(self.__class__.__name__)
    
    def send_notification(self, notification) -> bool:
        """Send training notification email.
        
        Args:
            notification: Notification data (TrainingNotification instance or dict)
            
        Returns:
            True if email sent successfully, False otherwise
        """
        if isinstance(notification, dict):
            notification = TrainingNotification(**notification)
        
        try:
            # Create email message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = notification.subject
            msg['From'] = f"{self.config.from_name} <{self.config.from_address}>"
            msg['To'] = notification.recipient
            
            # Create HTML and text versions
            text_content = self._create_text_content(notification)
            html_content = self._create_html_content(notification)
            
            # Add text and HTML parts
            text_part = MIMEText(text_content, 'plain', 'utf-8')
            html_part = MIMEText(html_content, 'html', 'utf-8')
            
            msg.attach(text_part)
            msg.attach(html_part)
            
            # Send email
            return self._send_email(msg)
            
        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}")
            return False
    
    def send_training_started(self, job_id: str, recipient: str, job_config: Dict[str, Any]) -> bool:
        """Send training started notification.
        
        Args:
            job_id: Training job ID
            recipient: Email recipient
            job_config: Training job configuration
            
        Returns:
            True if email sent successfully
        """
        notification = TrainingNotification(
            subject=f"Training Job Started - {job_id}",
            recipient=recipient,
            message=f"Training job {job_id} has been started successfully.",
            training_job_id=job_id,
            status="started",
            timestamp=datetime.utcnow()
        )
        
        return self.send_notification(notification)
    
    def send_training_completed(self, job_id: str, recipient: str, results: Dict[str, Any]) -> bool:
        """Send training completed notification.
        
        Args:
            job_id: Training job ID
            recipient: Email recipient
            results: Training results summary
            
        Returns:
            True if email sent successfully
        """
        notification = TrainingNotification(
            subject=f"Training Job Completed - {job_id}",
            recipient=recipient,
            message=f"Training job {job_id} has completed successfully.",
            training_job_id=job_id,
            status="completed",
            timestamp=datetime.utcnow()
        )
        
        return self.send_notification(notification)
    
    def send_training_failed(self, job_id: str, recipient: str, error: str) -> bool:
        """Send training failed notification.
        
        Args:
            job_id: Training job ID
            recipient: Email recipient
            error: Error message
            
        Returns:
            True if email sent successfully
        """
        notification = TrainingNotification(
            subject=f"Training Job Failed - {job_id}",
            recipient=recipient,
            message=f"Training job {job_id} has failed: {error}",
            training_job_id=job_id,
            status="failed",
            timestamp=datetime.utcnow()
        )
        
        return self.send_notification(notification)
    
    def _create_text_content(self, notification: TrainingNotification) -> str:
        """Create plain text email content."""
        content = f"""
Gaudi 3 Scale Training Notification
==================================

Subject: {notification.subject}
Status: {notification.status.upper()}
Time: {notification.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

{notification.message}

"""
        
        if notification.training_job_id:
            content += f"Training Job ID: {notification.training_job_id}\n\n"
        
        content += """
This is an automated notification from Gaudi 3 Scale.
"""
        
        return content
    
    def _create_html_content(self, notification: TrainingNotification) -> str:
        """Create HTML email content."""
        status_color = {
            'started': '#2196F3',
            'completed': '#4CAF50', 
            'failed': '#F44336',
            'unknown': '#757575'
        }.get(notification.status, '#757575')
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Gaudi 3 Scale Notification</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .status {{ display: inline-block; padding: 6px 12px; border-radius: 4px; color: white; background-color: {status_color}; font-weight: bold; }}
        .content {{ background-color: #ffffff; padding: 20px; border: 1px solid #dee2e6; border-radius: 8px; }}
        .footer {{ margin-top: 20px; font-size: 12px; color: #6c757d; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Gaudi 3 Scale Training Notification</h1>
            <p><span class="status">{notification.status.upper()}</span></p>
        </div>
        
        <div class="content">
            <h2>{notification.subject}</h2>
            <p><strong>Time:</strong> {notification.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
"""
        
        if notification.training_job_id:
            html += f"<p><strong>Training Job ID:</strong> {notification.training_job_id}</p>"
        
        html += f"""
            <p>{notification.message}</p>
        </div>
        
        <div class="footer">
            <p>This is an automated notification from Gaudi 3 Scale.</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    def _send_email(self, msg: MIMEMultipart) -> bool:
        """Send email using SMTP.
        
        Args:
            msg: Email message to send
            
        Returns:
            True if sent successfully
        """
        try:
            # Create SMTP connection
            if self.config.use_tls:
                context = ssl.create_default_context()
                server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
                server.starttls(context=context)
            else:
                server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            
            # Login and send
            server.login(self.config.username, self.config.password)
            text = msg.as_string()
            server.sendmail(self.config.from_address, msg['To'], text)
            server.quit()
            
            self.logger.info(f"Email sent successfully to {msg['To']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test SMTP connection.
        
        Returns:
            True if connection successful
        """
        try:
            if self.config.use_tls:
                context = ssl.create_default_context()
                server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
                server.starttls(context=context)
            else:
                server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            
            server.login(self.config.username, self.config.password)
            server.quit()
            
            self.logger.info("SMTP connection test successful")
            return True
            
        except Exception as e:
            self.logger.error(f"SMTP connection test failed: {e}")
            return False