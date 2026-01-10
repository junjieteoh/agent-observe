"""
PII (Personally Identifiable Information) handling for agent-observe.

Intercepts data before storage and applies redaction, hashing, tokenization,
or flagging based on configurable patterns.
"""

from __future__ import annotations

import hashlib
import logging
import re
import secrets
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class PIIAction(Enum):
    """Actions to take when PII is detected."""

    REDACT = "redact"  # Replace with placeholder like [EMAIL_REDACTED]
    HASH = "hash"  # Replace with one-way hash
    TOKENIZE = "tokenize"  # Replace with reversible token (requires token store)
    FLAG = "flag"  # Keep data but flag it as containing PII


# Built-in PII patterns
BUILTIN_PATTERNS: dict[str, str] = {
    # Email addresses
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    # Phone numbers (various formats)
    "phone": r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
    # US Social Security Numbers
    "ssn": r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
    # Credit card numbers (basic)
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    # IP addresses (IPv4)
    "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    # Dates of birth (various formats)
    "date_of_birth": r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b",
    # API keys / secrets (generic patterns)
    "api_key": r"\b(?:sk|pk|api|key|secret|token)[-_]?[a-zA-Z0-9]{20,}\b",
    # AWS keys
    "aws_key": r"\b(?:AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}\b",
}


@dataclass
class PIIConfig:
    """Configuration for PII detection and handling."""

    enabled: bool = False
    action: str = "redact"  # "redact", "hash", "tokenize", "flag"

    # Pattern configuration
    # Key is pattern name, value is:
    # - True: use builtin pattern
    # - False: disable pattern
    # - str: custom regex pattern
    patterns: dict[str, bool | str] = field(default_factory=lambda: {
        "email": True,
        "phone": True,
        "ssn": True,
        "credit_card": True,
    })

    # What data to scan
    scan: list[str] = field(default_factory=lambda: ["span.attrs", "event.payload"])

    # Hash salt for consistent hashing (generated if not provided)
    hash_salt: str | None = None

    # Callback when PII is detected (for alerting/logging)
    on_detect: Callable[[str, str, str], None] | None = None

    def __post_init__(self) -> None:
        """Initialize hash salt if not provided."""
        if self.action == "hash" and self.hash_salt is None:
            # Generate a random salt for this session
            self.hash_salt = secrets.token_hex(16)


@dataclass
class PIIDetection:
    """A single PII detection result."""

    pattern_name: str
    original_value: str
    start: int
    end: int
    replacement: str | None = None


class PIIHandler:
    r"""
    Handles PII detection and redaction/hashing.

    Usage:
        handler = PIIHandler(PIIConfig(
            enabled=True,
            action="redact",
            patterns={"email": True, "employee_id": r"EMP-\d{6}"},
        ))

        # Process data before storage
        clean_data = handler.process(data)
    """

    def __init__(self, config: PIIConfig):
        """Initialize the PII handler."""
        self.config = config
        self._compiled_patterns: dict[str, re.Pattern] = {}
        self._token_store: dict[str, str] = {}  # For tokenization

        if config.enabled:
            self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficient matching."""
        for name, value in self.config.patterns.items():
            if value is False:
                continue  # Pattern disabled

            if value is True:
                # Use builtin pattern
                if name in BUILTIN_PATTERNS:
                    pattern_str = BUILTIN_PATTERNS[name]
                else:
                    logger.warning(f"Unknown builtin PII pattern: {name}")
                    continue
            else:
                # Custom pattern
                pattern_str = value

            try:
                self._compiled_patterns[name] = re.compile(pattern_str, re.IGNORECASE)
            except re.error as e:
                logger.error(f"Invalid PII pattern '{name}': {e}")

    def detect(self, text: str) -> list[PIIDetection]:
        """
        Detect PII in text.

        Args:
            text: The text to scan for PII.

        Returns:
            List of PII detections with pattern name and location.
        """
        if not self.config.enabled or not isinstance(text, str):
            return []

        detections = []
        for name, pattern in self._compiled_patterns.items():
            for match in pattern.finditer(text):
                detections.append(PIIDetection(
                    pattern_name=name,
                    original_value=match.group(),
                    start=match.start(),
                    end=match.end(),
                ))

        return detections

    def process_string(self, text: str) -> tuple[str, list[PIIDetection]]:
        """
        Process a string, applying the configured PII action.

        Args:
            text: The text to process.

        Returns:
            Tuple of (processed_text, detections).
        """
        if not self.config.enabled or not isinstance(text, str):
            return text, []

        detections = self.detect(text)
        if not detections:
            return text, []

        # Sort by position (reverse) to replace from end to start
        detections.sort(key=lambda d: d.start, reverse=True)

        action = PIIAction(self.config.action)
        result = text

        for detection in detections:
            replacement = self._get_replacement(detection, action)
            detection.replacement = replacement
            result = result[:detection.start] + replacement + result[detection.end:]

            # Call detection callback if configured
            if self.config.on_detect:
                try:
                    self.config.on_detect(
                        detection.pattern_name,
                        detection.original_value,
                        replacement,
                    )
                except Exception as e:
                    logger.warning(f"PII on_detect callback error: {e}")

        return result, detections

    def _get_replacement(self, detection: PIIDetection, action: PIIAction) -> str:
        """Get the replacement string for a PII detection."""
        if action == PIIAction.REDACT:
            return f"[{detection.pattern_name.upper()}_REDACTED]"

        elif action == PIIAction.HASH:
            # One-way hash with salt
            salt = self.config.hash_salt or ""
            hash_input = f"{salt}{detection.original_value}"
            hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
            return f"[{detection.pattern_name.upper()}:{hash_value}]"

        elif action == PIIAction.TOKENIZE:
            # Reversible tokenization
            if detection.original_value in self._token_store:
                token = self._token_store[detection.original_value]
            else:
                token = secrets.token_hex(8)
                self._token_store[detection.original_value] = token
            return f"[TOKEN:{token}]"

        elif action == PIIAction.FLAG:
            # Keep original but mark it
            return f"[PII:{detection.pattern_name}]{detection.original_value}[/PII]"

        return detection.original_value

    def process(self, data: Any, path: str = "") -> Any:
        """
        Process any data structure, recursively handling dicts, lists, and strings.

        Args:
            data: The data to process (dict, list, str, or other).
            path: Current path for scan filtering (e.g., "span.attrs").

        Returns:
            The processed data with PII handled.
        """
        if not self.config.enabled:
            return data

        if isinstance(data, str):
            result, _ = self.process_string(data)
            return result

        elif isinstance(data, dict):
            return {
                key: self.process(value, f"{path}.{key}" if path else key)
                for key, value in data.items()
            }

        elif isinstance(data, list):
            return [
                self.process(item, f"{path}[{i}]")
                for i, item in enumerate(data)
            ]

        # Other types (int, float, bool, None) pass through unchanged
        return data

    def detokenize(self, token: str) -> str | None:
        """
        Reverse tokenization (if tokenize action was used).

        Args:
            token: The token to look up.

        Returns:
            Original value if found, None otherwise.
        """
        for original, stored_token in self._token_store.items():
            if stored_token == token:
                return original
        return None

    def get_stats(self) -> dict[str, int]:
        """Get statistics about PII detections."""
        return {
            "patterns_enabled": len(self._compiled_patterns),
            "tokens_stored": len(self._token_store),
        }


def create_pii_handler(config: PIIConfig | dict | None) -> PIIHandler | None:
    """
    Create a PII handler from configuration.

    Args:
        config: PIIConfig object, dict of config values, or None.

    Returns:
        PIIHandler if enabled, None otherwise.
    """
    if config is None:
        return None

    if isinstance(config, dict):
        config = PIIConfig(**config)

    if not config.enabled:
        return None

    return PIIHandler(config)
