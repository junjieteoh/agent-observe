"""Tests for hashing module."""

from __future__ import annotations

from agent_observe.hashing import (
    hash_bytes,
    hash_content,
    hash_json,
    hash_string,
    truncate_for_storage,
)


class TestHashing:
    """Tests for hashing functions."""

    def test_hash_bytes(self) -> None:
        """Test hashing bytes."""
        data = b"hello world"
        h = hash_bytes(data)

        assert len(h) == 64  # SHA256 hex is 64 chars
        assert h == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"

    def test_hash_bytes_empty(self) -> None:
        """Test hashing empty bytes."""
        h = hash_bytes(b"")
        assert len(h) == 64
        # Empty string has known hash
        assert h == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_hash_string(self) -> None:
        """Test hashing string."""
        h = hash_string("hello world")
        assert h == hash_bytes(b"hello world")

    def test_hash_json_dict(self) -> None:
        """Test hashing dict."""
        data = {"key": "value", "number": 42}
        h = hash_json(data)

        assert len(h) == 64
        # Same data should produce same hash
        assert h == hash_json({"number": 42, "key": "value"})

    def test_hash_json_deterministic(self) -> None:
        """Test that JSON hashing is deterministic regardless of key order."""
        data1 = {"a": 1, "b": 2, "c": 3}
        data2 = {"c": 3, "a": 1, "b": 2}

        assert hash_json(data1) == hash_json(data2)

    def test_hash_json_nested(self) -> None:
        """Test hashing nested structures."""
        data = {
            "outer": {
                "inner": [1, 2, 3],
                "nested": {"deep": "value"},
            }
        }
        h = hash_json(data)
        assert len(h) == 64

    def test_hash_content_bytes(self) -> None:
        """Test hash_content with bytes."""
        content = b"test content"
        h, size = hash_content(content)

        assert h == hash_bytes(content)
        assert size == len(content)

    def test_hash_content_string(self) -> None:
        """Test hash_content with string."""
        content = "test content"
        h, size = hash_content(content)

        assert h == hash_string(content)
        assert size == len(content.encode("utf-8"))

    def test_hash_content_dict(self) -> None:
        """Test hash_content with dict."""
        content = {"key": "value"}
        h, size = hash_content(content)

        assert h == hash_json(content)
        assert size > 0


class TestTruncation:
    """Tests for content truncation."""

    def test_truncate_short_content(self) -> None:
        """Test that short content is not truncated."""
        content = b"short"
        result = truncate_for_storage(content, 100)
        assert result == content

    def test_truncate_long_bytes(self) -> None:
        """Test truncating long bytes."""
        content = b"x" * 1000
        result = truncate_for_storage(content, 100)

        assert len(result) == 100
        assert result == b"x" * 100

    def test_truncate_string(self) -> None:
        """Test truncating string."""
        content = "x" * 1000
        result = truncate_for_storage(content, 100)

        assert len(result) == 100

    def test_truncate_utf8_boundary(self) -> None:
        """Test truncation at UTF-8 character boundary."""
        # Each emoji is 4 bytes in UTF-8
        content = "🎉" * 10  # 40 bytes total
        result = truncate_for_storage(content, 10)

        # Should truncate to valid UTF-8
        decoded = result.decode("utf-8")
        assert len(decoded) <= 3  # At most 2 emojis (8 bytes) + partial

    def test_truncate_exact_limit(self) -> None:
        """Test content exactly at limit."""
        content = b"x" * 100
        result = truncate_for_storage(content, 100)
        assert result == content
