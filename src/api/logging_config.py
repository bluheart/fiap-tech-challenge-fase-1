"""
Custom JSON logging configuration for the MLP API.

This module sets up structured JSON logging for the MLP churn prediction API,
providing consistent, parseable log output formatted for log aggregation systems
like ELK Stack, Loki, or CloudWatch. All log entries include standardized fields
for timestamp, severity level, service identification, and logger name.

Features:
    - JSON-formatted log output for easy parsing by log aggregators
    - Automatic timestamp injection for each log entry
    - Standardized service and level fields for filtering and alerting
    - Console-based output to stdout for container/cloud-native environments
    - Configurable log level via environment or configuration

Classes:
    CustomJsonFormatter: Extended JSON formatter with additional metadata fields

Functions:
    setup_logging: Configure and return the application logger instance

Usage:
    from logger import logger
    
    logger.info("Model loaded successfully", extra={"model_version": "1.2.3"})
    logger.error("Prediction failed", extra={"error_type": "model_timeout"})
"""

from pythonjsonlogger import jsonlogger
import logging
from datetime import datetime
import sys


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON log formatter that enriches log records with additional metadata.
    
    Extends the base jsonlogger.JsonFormatter to automatically add standardized
    fields to every log entry, ensuring consistent log structure across the
    application. This makes logs easier to search, filter, and analyze in
    centralized logging platforms.
    
    Added fields:
        timestamp: ISO 8601 timestamp of when the log was created
        level: Log severity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        service: Service identifier for multi-service architectures ('mlp-api')
        logger: Name of the logger that generated the entry
        message: Human-readable log message (if not already present)
    
    Example output:
        {
            "timestamp": "2026-04-30T14:23:45.123456",
            "level": "INFO",
            "service": "mlp-api",
            "logger": "api",
            "message": "Request processed successfully"
        }
    """
    
    def add_fields(self, log_record, record, message_dict):
        """
        Add custom fields to the log record before JSON serialization.
        
        This method is called by the logging framework for each log entry.
        It extends the base implementation to inject application-specific
        metadata fields into the structured log output.
        
        Args:
            log_record (dict): The dictionary that will be serialized to JSON
            record (logging.LogRecord): The original log record from Python's logging
            message_dict (dict): Additional fields passed via the 'extra' parameter
            
        Note:
            The message field is only set if not already provided, allowing
            custom messages to override the default behavior while maintaining
            backward compatibility.
        """
        super().add_fields(log_record, record, message_dict)
        log_record['timestamp'] = datetime.now()
        log_record['level'] = record.levelname
        log_record['service'] = 'mlp-api'
        log_record['logger'] = record.name
        if not log_record.get('message'):
            log_record['message'] = record.getMessage()


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure and initialize the application logger with JSON formatting.
    
    Sets up a logger named 'api' with JSON formatting and stdout output.
    Any existing handlers are removed before configuration to prevent
    duplicate log entries. The function is designed for containerized
    environments where logs should be written to stdout for collection
    by the container runtime or orchestration platform.
    
    Args:
        level (str): Logging level as a string. Defaults to "INFO".
            Valid values: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
            Case-insensitive.
    
    Returns:
        logging.Logger: Configured logger instance ready for use
        
    Example:
        # Default INFO level
        logger = setup_logging()
        
        # Debug level for development
        logger = setup_logging(level="DEBUG")
        
        # Set level from environment variable
        logger = setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))
    """
    logger = logging.getLogger("api")
    logger.setLevel(getattr(logging, level.upper()))

    # Create handler that writes to stdout (standard for containerized apps)
    handler = logging.StreamHandler(sys.stdout)
    formatter = CustomJsonFormatter('%(timestamp)s %(level)s %(service)s %(message)s %(logger)s')
    handler.setFormatter(formatter)

    # Clear any existing handlers to prevent duplicate log entries
    # This is important when the module is reloaded or reinitialized
    logger.handlers = []
    logger.addHandler(handler)
    return logger

# Module-level logger instance
# Initialized at import time with default INFO level.
# Can be reinitialized with different settings by calling setup_logging() again.
logger = setup_logging()