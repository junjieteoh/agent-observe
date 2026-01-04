"""
Hashing utilities for agent-observe.

Provides consistent SHA256 hashing for content fingerprinting.
Used in metadata_only mode to store hashes instead of raw content.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def hash_bytes(data: bytes) -> str:
    """
    Compute SHA256 hash of bytes.

    Args:
        data: Raw bytes to hash.

    Returns:
        Hexadecimal SHA256 hash string.
    """
    return hashlib.sha256(data).hexdigest()


def hash_string(data: str, encoding: str = "utf-8") -> str:
    """
    Compute SHA256 hash of a string.

    Args:
        data: String to hash.
        encoding: String encoding (default: utf-8).

    Returns:
        Hexadecimal SHA256 hash string.
    """
    return hash_bytes(data.encode(encoding))


def hash_json(data: Any) -> str:
    """
    Compute SHA256 hash of JSON-serializable data.

    Uses deterministic serialization (sorted keys, no whitespace)
    to ensure consistent hashes regardless of dict ordering.

    Args:
        data: JSON-serializable Python object.

    Returns:
        Hexadecimal SHA256 hash string.

    Raises:
        TypeError: If data is not JSON-serializable.
    """
    # Use sort_keys and separators for deterministic output
    serialized = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hash_string(serialized)


def hash_content(content: bytes | str | Any) -> tuple[str, int]:
    """
    Hash arbitrary content and return hash with size.

    Args:
        content: Content to hash (bytes, string, or JSON-serializable).

    Returns:
        Tuple of (hash, size_in_bytes).
    """
    if isinstance(content, bytes):
        return hash_bytes(content), len(content)
    elif isinstance(content, str):
        encoded = content.encode("utf-8")
        return hash_bytes(encoded), len(encoded)
    else:
        # JSON-serialize for other types
        serialized = json.dumps(content, sort_keys=True, separators=(",", ":"), default=str)
        encoded = serialized.encode("utf-8")
        return hash_bytes(encoded), len(encoded)


def truncate_for_storage(
    content: bytes | str, max_bytes: int, encoding: str = "utf-8"
) -> bytes:
    """
    Truncate content to fit within byte limit.

    For strings, ensures valid UTF-8 boundary truncation.

    Args:
        content: Content to truncate.
        max_bytes: Maximum size in bytes.
        encoding: String encoding.

    Returns:
        Truncated bytes.
    """
    if isinstance(content, str):
        content = content.encode(encoding)

    if len(content) <= max_bytes:
        return content

    # Truncate and ensure valid UTF-8 if applicable
    truncated = content[:max_bytes]
    try:
        # Try to decode - if it fails, back off until valid
        truncated.decode(encoding)
    except UnicodeDecodeError:
        # Back off byte by byte until valid
        for i in range(1, 4):  # UTF-8 chars are max 4 bytes
            try:
                truncated = content[: max_bytes - i]
                truncated.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

    return truncated
