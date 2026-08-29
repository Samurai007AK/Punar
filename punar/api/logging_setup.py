"""Structured (JSON) logging with correlation IDs and PII redaction.

The previous implementation called ``logging.basicConfig`` at import time --
hijacking the host application's root logger -- and interpolated case ids,
customer ids and rupee amounts straight into INFO messages.

This module instead:

* configures logging only when :func:`configure_logging` is called explicitly
  (from the app factory), never as an import side effect;
* emits one JSON object per line, so logs are queryable in any log platform;
* attaches a request/correlation id to every record via a ``ContextVar``;
* runs a :class:`RedactingFilter` over both the message and the structured
  fields so emails, phone numbers, card-like digit runs, payment links, API
  keys and signatures can never be written to a log sink.
"""
from __future__ import annotations

import json
import logging
import re
import sys
import uuid
from contextvars import ContextVar
from typing import Any

__all__ = [
    "configure_logging",
    "RedactingFilter",
    "JsonFormatter",
    "request_id_var",
    "new_request_id",
    "get_logger",
]

request_id_var: ContextVar[str] = ContextVar("punar_request_id", default="-")

_REDACTIONS = (
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[email-redacted]"),
    (re.compile(r"https?://[^\s<>\"']+"), "[link-redacted]"),
    (re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"), "[card-redacted]"),
    (re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?\d{10}(?!\d)"), "[phone-redacted]"),
    (re.compile(r"(?i)\b(secret|api[_-]?key|authorization|signature|token|password)"
                r"\s*[=:]\s*\S+"), r"\1=[redacted]"),
    (re.compile(r"(?i)\bcust(?:omer)?[_-]?id\s*[=:]\s*[\w-]+"), "customer_id=[redacted]"),
)

#: Structured-extra keys whose values are always dropped.
_FORBIDDEN_KEYS = frozenset({
    "customer_id", "customer_email", "customer_phone", "email", "phone",
    "payment_link", "signature", "api_key", "authorization", "secret",
    "amount_inr", "amount",
})

_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message", "asctime", "taskName"}


def redact(text: str) -> str:
    """Apply every redaction pattern to a free-text string."""
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


class RedactingFilter(logging.Filter):
    """Scrubs PII from the formatted message and from structured extras."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a bad format string must not lose the log
            message = str(record.msg)
        record.msg = redact(message)
        record.args = ()
        for key in list(record.__dict__):
            if key in _RESERVED:
                continue
            value = record.__dict__[key]
            if key.lower() in _FORBIDDEN_KEYS:
                record.__dict__[key] = "[redacted]"
            elif isinstance(value, str):
                record.__dict__[key] = redact(value)
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None) or request_id_var.get(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED or key in payload:
                continue
            if key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            # Stack traces stay server-side; they are never surfaced to callers.
            payload["exc_type"] = getattr(record.exc_info[0], "__name__", "Exception")
            payload["traceback"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "request_id", None):
            record.request_id = request_id_var.get()
        return True


def new_request_id() -> str:
    """Fresh correlation id for an inbound request."""
    return uuid.uuid4().hex


def configure_logging(level: str = "INFO", json_output: bool = True,
                      stream: Any | None = None) -> logging.Logger:
    """Install Punar's handler on the ``punar`` logger only.

    The root logger is deliberately left alone so that embedding Punar in a
    larger application does not reconfigure that application's logging.
    """
    logger = logging.getLogger("punar")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stdout)
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"))
    handler.addFilter(_RequestIdFilter())
    handler.addFilter(RedactingFilter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger(name: str = "punar.api") -> logging.Logger:
    return logging.getLogger(name)
