"""Enterprise-grade audit logging and compliance system.

This module provides comprehensive audit logging, security event tracking,
and compliance monitoring for regulatory requirements (SOX, GDPR, HIPAA, etc.).
"""

import json
import time
import hashlib
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union, Set
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from queue import Queue, Empty
from contextlib import contextmanager

try:
    import cryptography
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

from pydantic import BaseModel, Field, validator
from ..logging_utils import get_logger
from ..database.connection import get_database
from .config_security import EncryptionManager

logger = get_logger(__name__)


class AuditLevel(Enum):
    """Audit log levels for security events."""
    INFO = "INFO"
    WARNING = "WARNING" 
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SECURITY = "SECURITY"
    COMPLIANCE = "COMPLIANCE"


class EventCategory(Enum):
    """Categories of audit events."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SYSTEM_ACCESS = "system_access"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_EVENT = "compliance_event"
    ERROR_EVENT = "error_event"
    ADMIN_ACTION = "admin_action"


class ComplianceStandard(Enum):
    """Compliance standards for audit requirements."""
    SOX = "SOX"  # Sarbanes-Oxley
    GDPR = "GDPR"  # General Data Protection Regulation
    HIPAA = "HIPAA"  # Health Insurance Portability and Accountability Act
    PCI_DSS = "PCI_DSS"  # Payment Card Industry Data Security Standard
    ISO27001 = "ISO27001"  # Information Security Management
    SOC2 = "SOC2"  # Service Organization Control 2


@dataclass
class AuditEvent:
    """Represents an audit event with all required metadata."""
    
    # Core event data
    event_id: str
    timestamp: datetime
    level: AuditLevel
    category: EventCategory
    event_type: str
    message: str
    
    # User and session context
    user_id: Optional[str] = None
    username: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # System context
    hostname: Optional[str] = None
    process_id: Optional[int] = None
    thread_id: Optional[int] = None
    
    # Request context
    request_id: Optional[str] = None
    method: Optional[str] = None
    url: Optional[str] = None
    status_code: Optional[int] = None
    
    # Resource context
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    resource_name: Optional[str] = None
    
    # Event details
    details: Dict[str, Any] = field(default_factory=dict)
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None
    
    # Security context
    risk_score: float = 0.0
    threat_indicators: List[str] = field(default_factory=list)
    security_tags: Set[str] = field(default_factory=set)
    
    # Compliance context
    compliance_standards: Set[ComplianceStandard] = field(default_factory=set)
    retention_period: Optional[int] = None  # Days to retain
    
    # Integrity verification
    checksum: Optional[str] = None
    signature: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary."""
        data = asdict(self)
        
        # Convert enum values to strings
        data['level'] = self.level.value
        data['category'] = self.category.value
        data['compliance_standards'] = [std.value for std in self.compliance_standards]
        data['security_tags'] = list(self.security_tags)
        
        # Convert datetime to ISO format
        data['timestamp'] = self.timestamp.isoformat()
        
        return data
    
    def calculate_checksum(self) -> str:
        """Calculate integrity checksum for the event."""
        # Create deterministic string representation
        data = self.to_dict()
        # Remove checksum and signature for calculation
        data.pop('checksum', None)
        data.pop('signature', None)
        
        event_string = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(event_string.encode('utf-8')).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify event integrity using checksum."""
        if not self.checksum:
            return False
        
        calculated_checksum = self.calculate_checksum()
        return calculated_checksum == self.checksum


class AuditFilter(BaseModel):
    """Filter criteria for audit log queries."""
    
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    levels: Optional[List[AuditLevel]] = None
    categories: Optional[List[EventCategory]] = None
    event_types: Optional[List[str]] = None
    user_ids: Optional[List[str]] = None
    usernames: Optional[List[str]] = None
    ip_addresses: Optional[List[str]] = None
    resource_types: Optional[List[str]] = None
    resource_ids: Optional[List[str]] = None
    min_risk_score: Optional[float] = None
    compliance_standards: Optional[List[ComplianceStandard]] = None
    text_search: Optional[str] = None
    limit: int = 1000
    offset: int = 0


class SecurityAuditLogger:
    """Enterprise security audit logger with encryption and compliance features."""
    
    def __init__(self,
                 storage_path: Optional[Path] = None,
                 encryption_manager: Optional[EncryptionManager] = None,
                 enable_encryption: bool = True,
                 enable_database_storage: bool = True,
                 enable_file_storage: bool = True,
                 buffer_size: int = 100,
                 flush_interval: int = 5):
        """Initialize security audit logger.
        
        Args:
            storage_path: Path for audit log files
            encryption_manager: Encryption manager for sensitive data
            enable_encryption: Whether to encrypt audit logs
            enable_database_storage: Whether to store in database
            enable_file_storage: Whether to store in files
            buffer_size: Buffer size for batching events
            flush_interval: Interval in seconds to flush buffer
        """
        self.storage_path = storage_path or Path.home() / ".gaudi3_scale" / "audit"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.enable_encryption = enable_encryption and CRYPTO_AVAILABLE
        self.encryption_manager = encryption_manager
        if self.enable_encryption and not self.encryption_manager:
            self.encryption_manager = EncryptionManager()
        
        self.enable_database_storage = enable_database_storage
        self.enable_file_storage = enable_file_storage
        
        # Event buffering for performance
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.event_buffer: Queue = Queue()
        self.buffer_lock = threading.Lock()
        
        # Background thread for processing events
        self.processing_thread = None
        self.stop_processing = threading.Event()
        
        # Current log file
        self.current_log_file = None
        self.log_rotation_size = 50 * 1024 * 1024  # 50MB
        
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Start background processing
        self._start_processing()
    
    def _start_processing(self):
        """Start background event processing thread."""
        if self.processing_thread is not None:
            return
        
        self.processing_thread = threading.Thread(
            target=self._process_events,
            daemon=True,
            name="AuditLogProcessor"
        )
        self.processing_thread.start()
        
        self.logger.info("Audit log processing started")
    
    def _process_events(self):
        """Process audit events in background thread."""
        buffer = []
        last_flush_time = time.time()
        
        while not self.stop_processing.is_set():
            try:
                # Try to get event from queue with timeout
                try:
                    event = self.event_buffer.get(timeout=1.0)
                    buffer.append(event)
                    self.event_buffer.task_done()
                except Empty:
                    pass
                
                current_time = time.time()
                
                # Flush buffer if full or time interval reached
                should_flush = (
                    len(buffer) >= self.buffer_size or
                    (buffer and current_time - last_flush_time >= self.flush_interval)
                )
                
                if should_flush:
                    self._flush_buffer(buffer)
                    buffer.clear()
                    last_flush_time = current_time
                    
            except Exception as e:
                self.logger.error(f"Error processing audit events: {e}")
        
        # Flush remaining events on shutdown
        if buffer:
            self._flush_buffer(buffer)
    
    def _flush_buffer(self, events: List[AuditEvent]):
        """Flush buffered events to storage."""
        if not events:
            return
        
        try:
            # Store to database if enabled
            if self.enable_database_storage:
                self._store_to_database(events)
            
            # Store to file if enabled
            if self.enable_file_storage:
                self._store_to_file(events)
                
        except Exception as e:
            self.logger.error(f"Failed to flush audit events: {e}")
    
    def _store_to_database(self, events: List[AuditEvent]):
        """Store audit events to database."""
        try:
            # This would integrate with the database layer
            # For now, we'll skip actual database storage
            pass
        except Exception as e:
            self.logger.error(f"Failed to store audit events to database: {e}")
    
    def _store_to_file(self, events: List[AuditEvent]):
        """Store audit events to log files."""
        try:
            # Rotate log file if needed
            self._rotate_log_file_if_needed()
            
            # Prepare log entries
            log_entries = []
            for event in events:
                # Calculate integrity checksum
                if not event.checksum:
                    event.checksum = event.calculate_checksum()
                
                # Convert to JSON
                entry_data = event.to_dict()
                
                if self.enable_encryption:
                    # Encrypt sensitive data
                    entry_json = json.dumps(entry_data)
                    encrypted_entry = self.encryption_manager.encrypt(entry_json)
                    log_entries.append(f"ENCRYPTED:{encrypted_entry}\n")
                else:
                    log_entries.append(json.dumps(entry_data) + "\n")
            
            # Write to file
            with open(self.current_log_file, 'a', encoding='utf-8') as f:
                f.writelines(log_entries)
                f.flush()
                
        except Exception as e:
            self.logger.error(f"Failed to store audit events to file: {e}")
    
    def _rotate_log_file_if_needed(self):
        """Rotate log file if it exceeds size limit."""
        if self.current_log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_log_file = self.storage_path / f"audit_{timestamp}.log"
            return
        
        try:
            if self.current_log_file.exists() and self.current_log_file.stat().st_size > self.log_rotation_size:
                # Archive current file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archived_name = f"audit_{timestamp}_archived.log"
                archived_path = self.storage_path / archived_name
                self.current_log_file.rename(archived_path)
                
                # Create new log file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.current_log_file = self.storage_path / f"audit_{timestamp}.log"
                
                self.logger.info(f"Audit log rotated: {archived_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to rotate audit log: {e}")
    
    def log_event(self, 
                  level: AuditLevel,
                  category: EventCategory,
                  event_type: str,
                  message: str,
                  **kwargs) -> str:
        """Log an audit event.
        
        Args:
            level: Audit level
            category: Event category
            event_type: Specific event type
            message: Event message
            **kwargs: Additional event data
            
        Returns:
            Event ID
        """
        import uuid
        import os
        import platform
        
        # Generate event ID
        event_id = str(uuid.uuid4())
        
        # Create audit event
        event = AuditEvent(
            event_id=event_id,
            timestamp=datetime.now(timezone.utc),
            level=level,
            category=category,
            event_type=event_type,
            message=message,
            hostname=platform.node(),
            process_id=os.getpid(),
            thread_id=threading.get_ident(),
            **kwargs
        )
        
        # Add to buffer for processing
        try:
            self.event_buffer.put_nowait(event)
        except Exception as e:
            self.logger.error(f"Failed to queue audit event: {e}")
        
        return event_id
    
    def log_authentication_event(self,
                                success: bool,
                                username: str,
                                ip_address: Optional[str] = None,
                                user_agent: Optional[str] = None,
                                failure_reason: Optional[str] = None,
                                **kwargs) -> str:
        """Log authentication event."""
        event_type = "login_success" if success else "login_failure"
        level = AuditLevel.INFO if success else AuditLevel.WARNING
        
        message = f"Authentication {'successful' if success else 'failed'} for user: {username}"
        if not success and failure_reason:
            message += f" - {failure_reason}"
        
        details = {
            "success": success,
            "failure_reason": failure_reason
        }
        
        return self.log_event(
            level=level,
            category=EventCategory.AUTHENTICATION,
            event_type=event_type,
            message=message,
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details,
            compliance_standards={ComplianceStandard.SOX, ComplianceStandard.ISO27001},
            **kwargs
        )
    
    def log_authorization_event(self,
                               success: bool,
                               username: str,
                               permission: str,
                               resource: Optional[str] = None,
                               **kwargs) -> str:
        """Log authorization event."""
        event_type = "authorization_granted" if success else "authorization_denied"
        level = AuditLevel.INFO if success else AuditLevel.WARNING
        
        message = f"Authorization {'granted' if success else 'denied'} for {username}"
        message += f" - Permission: {permission}"
        if resource:
            message += f" - Resource: {resource}"
        
        details = {
            "success": success,
            "permission": permission,
            "resource": resource
        }
        
        return self.log_event(
            level=level,
            category=EventCategory.AUTHORIZATION,
            event_type=event_type,
            message=message,
            username=username,
            details=details,
            compliance_standards={ComplianceStandard.SOX},
            **kwargs
        )
    
    def log_data_access_event(self,
                             username: str,
                             resource_type: str,
                             resource_id: str,
                             action: str,
                             **kwargs) -> str:
        """Log data access event."""
        message = f"Data access: {username} performed {action} on {resource_type}:{resource_id}"
        
        return self.log_event(
            level=AuditLevel.INFO,
            category=EventCategory.DATA_ACCESS,
            event_type="data_access",
            message=message,
            username=username,
            resource_type=resource_type,
            resource_id=resource_id,
            details={"action": action},
            compliance_standards={ComplianceStandard.GDPR, ComplianceStandard.HIPAA},
            **kwargs
        )
    
    def log_configuration_change(self,
                                username: str,
                                config_type: str,
                                before_state: Optional[Dict[str, Any]] = None,
                                after_state: Optional[Dict[str, Any]] = None,
                                **kwargs) -> str:
        """Log configuration change event."""
        message = f"Configuration change: {username} modified {config_type}"
        
        return self.log_event(
            level=AuditLevel.INFO,
            category=EventCategory.CONFIGURATION_CHANGE,
            event_type="config_change",
            message=message,
            username=username,
            before_state=before_state,
            after_state=after_state,
            details={"config_type": config_type},
            compliance_standards={ComplianceStandard.SOX, ComplianceStandard.ISO27001},
            **kwargs
        )
    
    def log_security_event(self,
                          event_type: str,
                          risk_score: float,
                          threat_indicators: List[str],
                          **kwargs) -> str:
        """Log security event."""
        level = AuditLevel.CRITICAL if risk_score >= 8.0 else AuditLevel.SECURITY
        
        message = f"Security event: {event_type} - Risk Score: {risk_score}"
        
        return self.log_event(
            level=level,
            category=EventCategory.SECURITY_EVENT,
            event_type=event_type,
            message=message,
            risk_score=risk_score,
            threat_indicators=threat_indicators,
            security_tags={"security_incident"},
            compliance_standards={ComplianceStandard.ISO27001, ComplianceStandard.SOC2},
            **kwargs
        )
    
    def log_admin_action(self,
                        username: str,
                        action: str,
                        target: Optional[str] = None,
                        **kwargs) -> str:
        """Log administrative action."""
        message = f"Admin action: {username} performed {action}"
        if target:
            message += f" on {target}"
        
        return self.log_event(
            level=AuditLevel.INFO,
            category=EventCategory.ADMIN_ACTION,
            event_type="admin_action",
            message=message,
            username=username,
            details={"action": action, "target": target},
            compliance_standards={ComplianceStandard.SOX},
            **kwargs
        )
    
    def query_events(self, audit_filter: AuditFilter) -> List[AuditEvent]:
        """Query audit events based on filter criteria."""
        # This is a simplified implementation
        # In production, this would query the database
        events = []
        
        try:
            # Read from current and archived log files
            log_files = list(self.storage_path.glob("audit_*.log"))
            
            for log_file in log_files:
                events.extend(self._read_events_from_file(log_file, audit_filter))
            
            # Apply filters and sorting
            filtered_events = self._apply_filters(events, audit_filter)
            
            return filtered_events[audit_filter.offset:audit_filter.offset + audit_filter.limit]
            
        except Exception as e:
            self.logger.error(f"Failed to query audit events: {e}")
            return []
    
    def _read_events_from_file(self, log_file: Path, audit_filter: AuditFilter) -> List[AuditEvent]:
        """Read and parse events from log file."""
        events = []
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        if line.startswith("ENCRYPTED:"):
                            # Decrypt the line
                            encrypted_data = line[10:]  # Remove "ENCRYPTED:" prefix
                            decrypted_data = self.encryption_manager.decrypt(encrypted_data)
                            event_data = json.loads(decrypted_data)
                        else:
                            event_data = json.loads(line)
                        
                        # Convert back to AuditEvent
                        event = self._dict_to_audit_event(event_data)
                        events.append(event)
                        
                    except Exception as e:
                        self.logger.error(f"Failed to parse audit event: {e}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Failed to read audit log file {log_file}: {e}")
        
        return events
    
    def _dict_to_audit_event(self, data: Dict[str, Any]) -> AuditEvent:
        """Convert dictionary back to AuditEvent."""
        # Convert string enums back to enum types
        data['level'] = AuditLevel(data['level'])
        data['category'] = EventCategory(data['category'])
        
        if 'compliance_standards' in data:
            data['compliance_standards'] = {
                ComplianceStandard(std) for std in data['compliance_standards']
            }
        
        if 'security_tags' in data:
            data['security_tags'] = set(data['security_tags'])
        
        # Convert timestamp back to datetime
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        return AuditEvent(**data)
    
    def _apply_filters(self, events: List[AuditEvent], audit_filter: AuditFilter) -> List[AuditEvent]:
        """Apply filter criteria to events."""
        filtered = events
        
        # Time range filter
        if audit_filter.start_time:
            filtered = [e for e in filtered if e.timestamp >= audit_filter.start_time]
        
        if audit_filter.end_time:
            filtered = [e for e in filtered if e.timestamp <= audit_filter.end_time]
        
        # Level filter
        if audit_filter.levels:
            filtered = [e for e in filtered if e.level in audit_filter.levels]
        
        # Category filter
        if audit_filter.categories:
            filtered = [e for e in filtered if e.category in audit_filter.categories]
        
        # Event type filter
        if audit_filter.event_types:
            filtered = [e for e in filtered if e.event_type in audit_filter.event_types]
        
        # User filters
        if audit_filter.user_ids:
            filtered = [e for e in filtered if e.user_id in audit_filter.user_ids]
        
        if audit_filter.usernames:
            filtered = [e for e in filtered if e.username in audit_filter.usernames]
        
        # IP address filter
        if audit_filter.ip_addresses:
            filtered = [e for e in filtered if e.ip_address in audit_filter.ip_addresses]
        
        # Resource filters
        if audit_filter.resource_types:
            filtered = [e for e in filtered if e.resource_type in audit_filter.resource_types]
        
        if audit_filter.resource_ids:
            filtered = [e for e in filtered if e.resource_id in audit_filter.resource_ids]
        
        # Risk score filter
        if audit_filter.min_risk_score is not None:
            filtered = [e for e in filtered if e.risk_score >= audit_filter.min_risk_score]
        
        # Compliance standard filter
        if audit_filter.compliance_standards:
            filtered = [e for e in filtered if 
                       any(std in e.compliance_standards for std in audit_filter.compliance_standards)]
        
        # Text search
        if audit_filter.text_search:
            search_term = audit_filter.text_search.lower()
            filtered = [e for e in filtered if 
                       search_term in e.message.lower() or
                       search_term in str(e.details).lower()]
        
        # Sort by timestamp (newest first)
        filtered.sort(key=lambda x: x.timestamp, reverse=True)
        
        return filtered
    
    def generate_compliance_report(self, 
                                 standard: ComplianceStandard,
                                 start_date: datetime,
                                 end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report for specified standard."""
        audit_filter = AuditFilter(
            start_time=start_date,
            end_time=end_date,
            compliance_standards=[standard],
            limit=10000  # Large limit for reports
        )
        
        events = self.query_events(audit_filter)
        
        # Generate report based on compliance standard
        report = {
            "standard": standard.value,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_events": len(events),
            "event_breakdown": {},
            "risk_analysis": {},
            "recommendations": []
        }
        
        # Event breakdown by category
        category_counts = {}
        for event in events:
            category = event.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        report["event_breakdown"] = category_counts
        
        # Risk analysis
        high_risk_events = [e for e in events if e.risk_score >= 7.0]
        report["risk_analysis"] = {
            "high_risk_events": len(high_risk_events),
            "average_risk_score": sum(e.risk_score for e in events) / len(events) if events else 0,
            "threat_indicators": list(set().union(*[e.threat_indicators for e in events]))
        }
        
        # Compliance-specific recommendations
        report["recommendations"] = self._get_compliance_recommendations(standard, events)
        
        return report
    
    def _get_compliance_recommendations(self, 
                                     standard: ComplianceStandard, 
                                     events: List[AuditEvent]) -> List[str]:
        """Get compliance-specific recommendations."""
        recommendations = []
        
        if standard == ComplianceStandard.SOX:
            # Sarbanes-Oxley specific recommendations
            failed_logins = [e for e in events if e.event_type == "login_failure"]
            if len(failed_logins) > 100:
                recommendations.append("High number of failed login attempts detected. Consider implementing account lockout policies.")
            
            config_changes = [e for e in events if e.category == EventCategory.CONFIGURATION_CHANGE]
            if len(config_changes) > 50:
                recommendations.append("Frequent configuration changes detected. Ensure proper approval workflows are in place.")
        
        elif standard == ComplianceStandard.GDPR:
            # GDPR specific recommendations
            data_access = [e for e in events if e.category == EventCategory.DATA_ACCESS]
            if len(data_access) > 1000:
                recommendations.append("High volume of data access events. Ensure data processing is lawful and documented.")
        
        elif standard == ComplianceStandard.ISO27001:
            # ISO 27001 specific recommendations
            security_events = [e for e in events if e.category == EventCategory.SECURITY_EVENT]
            if len(security_events) > 10:
                recommendations.append("Multiple security events detected. Review incident response procedures.")
        
        return recommendations
    
    def shutdown(self):
        """Shutdown audit logger and flush remaining events."""
        self.logger.info("Shutting down audit logger...")
        
        # Stop processing thread
        self.stop_processing.set()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=10)
        
        # Flush remaining events
        remaining_events = []
        try:
            while True:
                event = self.event_buffer.get_nowait()
                remaining_events.append(event)
        except Empty:
            pass
        
        if remaining_events:
            self._flush_buffer(remaining_events)
        
        self.logger.info("Audit logger shutdown complete")


class ComplianceLogger:
    """Specialized compliance logger for regulatory requirements."""
    
    def __init__(self, audit_logger: SecurityAuditLogger):
        """Initialize compliance logger."""
        self.audit_logger = audit_logger
        self.logger = logger.getChild(self.__class__.__name__)
    
    def log_data_processing(self, 
                           data_subject: str,
                           processing_purpose: str,
                           data_categories: List[str],
                           legal_basis: str,
                           **kwargs) -> str:
        """Log GDPR data processing activity."""
        details = {
            "data_subject": data_subject,
            "processing_purpose": processing_purpose,
            "data_categories": data_categories,
            "legal_basis": legal_basis
        }
        
        return self.audit_logger.log_event(
            level=AuditLevel.COMPLIANCE,
            category=EventCategory.COMPLIANCE_EVENT,
            event_type="gdpr_data_processing",
            message=f"GDPR data processing: {processing_purpose} for {data_subject}",
            details=details,
            compliance_standards={ComplianceStandard.GDPR},
            retention_period=2555,  # 7 years
            **kwargs
        )
    
    def log_financial_transaction(self,
                                 transaction_id: str,
                                 amount: float,
                                 currency: str,
                                 user: str,
                                 **kwargs) -> str:
        """Log SOX financial transaction."""
        details = {
            "transaction_id": transaction_id,
            "amount": amount,
            "currency": currency
        }
        
        return self.audit_logger.log_event(
            level=AuditLevel.COMPLIANCE,
            category=EventCategory.COMPLIANCE_EVENT,
            event_type="sox_financial_transaction",
            message=f"Financial transaction: {transaction_id} - {amount} {currency}",
            username=user,
            details=details,
            compliance_standards={ComplianceStandard.SOX},
            retention_period=2555,  # 7 years
            **kwargs
        )
    
    def log_patient_data_access(self,
                               patient_id: str,
                               healthcare_provider: str,
                               access_reason: str,
                               **kwargs) -> str:
        """Log HIPAA patient data access."""
        details = {
            "patient_id": patient_id,
            "healthcare_provider": healthcare_provider,
            "access_reason": access_reason
        }
        
        return self.audit_logger.log_event(
            level=AuditLevel.COMPLIANCE,
            category=EventCategory.COMPLIANCE_EVENT,
            event_type="hipaa_patient_access",
            message=f"Patient data access: {patient_id} by {healthcare_provider}",
            details=details,
            compliance_standards={ComplianceStandard.HIPAA},
            retention_period=2190,  # 6 years
            **kwargs
        )


@contextmanager
def audit_context(audit_logger: SecurityAuditLogger,
                 event_type: str,
                 category: EventCategory = EventCategory.SYSTEM_ACCESS,
                 **context_data):
    """Context manager for auditing operations."""
    start_time = datetime.now(timezone.utc)
    
    # Log operation start
    start_event_id = audit_logger.log_event(
        level=AuditLevel.INFO,
        category=category,
        event_type=f"{event_type}_start",
        message=f"Operation started: {event_type}",
        **context_data
    )
    
    try:
        yield start_event_id
        
        # Log successful completion
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        
        audit_logger.log_event(
            level=AuditLevel.INFO,
            category=category,
            event_type=f"{event_type}_success",
            message=f"Operation completed successfully: {event_type}",
            details={"duration_seconds": duration, "start_event_id": start_event_id},
            **context_data
        )
        
    except Exception as e:
        # Log failure
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        
        audit_logger.log_event(
            level=AuditLevel.ERROR,
            category=category,
            event_type=f"{event_type}_failure",
            message=f"Operation failed: {event_type} - {str(e)}",
            details={
                "error": str(e),
                "error_type": type(e).__name__,
                "duration_seconds": duration,
                "start_event_id": start_event_id
            },
            **context_data
        )
        raise