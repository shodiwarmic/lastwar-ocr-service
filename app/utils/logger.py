"""
app/utils/logger.py

Configures structured JSON logging for Google Cloud Run.

Cloud Run captures all stdout/stderr and ships it to Cloud Logging automatically.
By emitting JSON, each log entry becomes queryable by individual fields in the
Cloud Logging console — making it easy to filter to just classification failures,
specific images, or fallback OCR events without grepping raw text.

Usage in any module:
    from app.utils.logger import get_logger, log_classification_event
    logger = get_logger(__name__)
    logger.info("Processing started")
"""

import json
import logging
import sys
from typing import Optional


class StructuredJsonFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON objects.

    Cloud Logging recognises the 'severity' field automatically and maps it
    to the correct log level in the console. The 'message' field is displayed
    as the primary log text. All extra fields passed via the 'extra' kwarg on
    the logger call are included at the top level for easy filtering.
    """

    # Map Python log level names to Cloud Logging severity strings
    SEVERITY_MAP = {
        "DEBUG": "DEBUG",
        "INFO": "INFO",
        "WARNING": "WARNING",
        "ERROR": "ERROR",
        "CRITICAL": "CRITICAL",
    }

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "severity": self.SEVERITY_MAP.get(record.levelname, "DEFAULT"),
            "message": record.getMessage(),
            "logger": record.name,
        }

        # Include any extra fields passed via logger.info("msg", extra={...})
        skip_fields = {
            "message", "asctime", "created", "exc_info", "exc_text", "filename",
            "funcName", "id", "levelname", "levelno", "lineno", "module",
            "msecs", "msg", "name", "pathname", "process", "processName",
            "relativeCreated", "stack_info", "thread", "threadName", "args",
        }
        for key, value in record.__dict__.items():
            if key not in skip_fields:
                log_entry[key] = value

        # Append exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger instance configured for Cloud Run structured logging.

    All loggers share the same stream handler (stdout) so Cloud Run captures
    every log entry. Calling this multiple times with the same name returns
    the same logger instance — safe to call at module level.

    Args:
        name: Typically __name__ from the calling module. Used as the 'logger'
              field in the JSON output so you can filter by module in Cloud Logging.

    Returns:
        A configured logging.Logger instance.

    Example:
        logger = get_logger(__name__)
        logger.info("Batch received", extra={"image_count": 10})
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredJsonFormatter())
    logger.addHandler(handler)

    # Prevent log records from propagating to the root logger and being
    # double-printed in environments that also have a root handler configured
    logger.propagate = False

    return logger


def log_classification_event(
    logger: logging.Logger,
    filename: str,
    resolution: tuple[int, int],
    pass1_result: Optional[str],
    pass1_confidence: float,
    pass2_result: Optional[str],
    ocr_triggered: bool,
    error: Optional[str] = None,
) -> None:
    """
    Emits a structured log entry for a single image classification attempt.

    Always called for every image so Cloud Logging has a complete audit trail.
    Uses WARNING severity when the OCR fallback was triggered so failures are
    easy to filter in the Cloud Logging console using:
        severity=WARNING logName=... jsonPayload.ocr_triggered=true

    Args:
        logger:           Logger instance from get_logger().
        filename:         Original uploaded filename, used to trace back to source.
        resolution:       (width, height) tuple of the image.
        pass1_result:     Category string from pre-OCR classification, or None if
                          classification could not be attempted.
        pass1_confidence: Float 0.0–1.0 confidence score from pass 1.
        pass2_result:     Category string resolved by OCR fallback, or None if
                          fallback was not triggered or also failed.
        ocr_triggered:    True if the image was sent unstitched for OCR-assisted
                          classification rather than being grouped for batch OCR.
        error:            Optional error message if classification failed entirely.

    Example Cloud Logging query to find all fallback events:
        jsonPayload.ocr_triggered = true
    """
    width, height = resolution
    extra = {
        "image_filename": filename,
        "resolution": f"{width}x{height}",
        "pass1_result": pass1_result,
        "pass1_confidence": round(pass1_confidence, 3),
        "pass2_result": pass2_result,
        "ocr_triggered": ocr_triggered,
    }

    if error:
        extra["error"] = error

    if error:
        logger.error("Classification failed entirely", extra=extra)
    elif ocr_triggered:
        logger.warning("Classification fallback triggered — image sent unstitched for OCR", extra=extra)
    else:
        logger.info("Classification succeeded on pass 1", extra=extra)
