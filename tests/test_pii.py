"""
Tests for PII detection and handling.
"""

import pytest

from agent_observe.pii import PIIAction, PIIConfig, PIIHandler, create_pii_handler


class TestPIIConfig:
    """Tests for PIIConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PIIConfig()
        assert config.enabled is False
        assert config.action == "redact"
        assert "email" in config.patterns
        assert "phone" in config.patterns

    def test_hash_salt_generated(self):
        """Test that hash salt is auto-generated for hash action."""
        config = PIIConfig(enabled=True, action="hash")
        assert config.hash_salt is not None
        assert len(config.hash_salt) == 32  # 16 bytes = 32 hex chars


class TestPIIHandler:
    """Tests for PIIHandler."""

    def test_disabled_handler_returns_unchanged(self):
        """Test that disabled handler returns data unchanged."""
        config = PIIConfig(enabled=False)
        handler = PIIHandler(config)

        data = {"email": "user@example.com"}
        result = handler.process(data)

        assert result == data

    def test_detect_email(self):
        """Test email detection."""
        config = PIIConfig(enabled=True, patterns={"email": True})
        handler = PIIHandler(config)

        text = "Contact me at user@example.com for details"
        detections = handler.detect(text)

        assert len(detections) == 1
        assert detections[0].pattern_name == "email"
        assert detections[0].original_value == "user@example.com"

    def test_detect_phone(self):
        """Test phone number detection."""
        config = PIIConfig(enabled=True, patterns={"phone": True})
        handler = PIIHandler(config)

        text = "Call me at 555-123-4567 or (555) 987-6543"
        detections = handler.detect(text)

        assert len(detections) == 2
        assert all(d.pattern_name == "phone" for d in detections)

    def test_detect_ssn(self):
        """Test SSN detection."""
        config = PIIConfig(enabled=True, patterns={"ssn": True})
        handler = PIIHandler(config)

        text = "My SSN is 123-45-6789"
        detections = handler.detect(text)

        assert len(detections) == 1
        assert detections[0].pattern_name == "ssn"

    def test_detect_credit_card(self):
        """Test credit card detection."""
        config = PIIConfig(enabled=True, patterns={"credit_card": True})
        handler = PIIHandler(config)

        text = "Card: 4111-1111-1111-1111"
        detections = handler.detect(text)

        assert len(detections) == 1
        assert detections[0].pattern_name == "credit_card"

    def test_redact_action(self):
        """Test PII redaction."""
        config = PIIConfig(enabled=True, action="redact", patterns={"email": True})
        handler = PIIHandler(config)

        text = "Contact user@example.com please"
        result, detections = handler.process_string(text)

        assert result == "Contact [EMAIL_REDACTED] please"
        assert len(detections) == 1
        assert detections[0].replacement == "[EMAIL_REDACTED]"

    def test_hash_action(self):
        """Test PII hashing."""
        config = PIIConfig(
            enabled=True,
            action="hash",
            patterns={"email": True},
            hash_salt="test_salt",
        )
        handler = PIIHandler(config)

        text = "Contact user@example.com please"
        result, detections = handler.process_string(text)

        assert "[EMAIL:" in result
        assert "user@example.com" not in result
        assert len(detections[0].replacement) > 0

    def test_hash_is_consistent(self):
        """Test that hash is consistent for same input."""
        config = PIIConfig(
            enabled=True,
            action="hash",
            patterns={"email": True},
            hash_salt="fixed_salt",
        )
        handler = PIIHandler(config)

        result1, _ = handler.process_string("user@example.com")
        result2, _ = handler.process_string("user@example.com")

        assert result1 == result2

    def test_tokenize_action(self):
        """Test PII tokenization."""
        config = PIIConfig(enabled=True, action="tokenize", patterns={"email": True})
        handler = PIIHandler(config)

        text = "Contact user@example.com please"
        result, _ = handler.process_string(text)

        assert "[TOKEN:" in result
        assert "user@example.com" not in result

    def test_tokenize_is_reversible(self):
        """Test that tokenization is reversible."""
        config = PIIConfig(enabled=True, action="tokenize", patterns={"email": True})
        handler = PIIHandler(config)

        original = "user@example.com"
        handler.process_string(original)

        # Get the token
        for original_val, token in handler._token_store.items():
            if original_val == original:
                # Should be able to reverse
                assert handler.detokenize(token) == original
                break

    def test_flag_action(self):
        """Test PII flagging (keeps data but marks it)."""
        config = PIIConfig(enabled=True, action="flag", patterns={"email": True})
        handler = PIIHandler(config)

        text = "Contact user@example.com please"
        result, _ = handler.process_string(text)

        assert "[PII:email]user@example.com[/PII]" in result

    def test_custom_pattern(self):
        """Test custom PII pattern."""
        config = PIIConfig(
            enabled=True,
            action="redact",
            patterns={
                "employee_id": r"EMP-\d{6}",
            }
        )
        handler = PIIHandler(config)

        text = "Employee EMP-123456 reported"
        result, detections = handler.process_string(text)

        assert result == "Employee [EMPLOYEE_ID_REDACTED] reported"
        assert detections[0].pattern_name == "employee_id"

    def test_disabled_builtin_pattern(self):
        """Test disabling a builtin pattern."""
        config = PIIConfig(
            enabled=True,
            patterns={
                "email": False,  # Disabled
                "phone": True,
            }
        )
        handler = PIIHandler(config)

        text = "Email: user@example.com Phone: 555-123-4567"
        detections = handler.detect(text)

        assert len(detections) == 1
        assert detections[0].pattern_name == "phone"

    def test_process_dict(self):
        """Test processing dictionary data."""
        config = PIIConfig(enabled=True, action="redact", patterns={"email": True})
        handler = PIIHandler(config)

        data = {
            "user": "Alice",
            "email": "alice@example.com",
            "nested": {
                "contact": "bob@example.com",
            }
        }

        result = handler.process(data)

        assert result["user"] == "Alice"  # Unchanged
        assert result["email"] == "[EMAIL_REDACTED]"
        assert result["nested"]["contact"] == "[EMAIL_REDACTED]"

    def test_process_list(self):
        """Test processing list data."""
        config = PIIConfig(enabled=True, action="redact", patterns={"email": True})
        handler = PIIHandler(config)

        data = ["alice@example.com", "bob@example.com", "no-email-here"]
        result = handler.process(data)

        assert result[0] == "[EMAIL_REDACTED]"
        assert result[1] == "[EMAIL_REDACTED]"
        assert result[2] == "no-email-here"

    def test_process_mixed_types(self):
        """Test processing mixed data types."""
        config = PIIConfig(enabled=True, action="redact", patterns={"email": True})
        handler = PIIHandler(config)

        data = {
            "count": 42,
            "active": True,
            "email": "user@example.com",
            "score": 3.14,
            "nullable": None,
        }

        result = handler.process(data)

        assert result["count"] == 42
        assert result["active"] is True
        assert result["email"] == "[EMAIL_REDACTED]"
        assert result["score"] == 3.14
        assert result["nullable"] is None

    def test_on_detect_callback(self):
        """Test that on_detect callback is called."""
        detections_logged = []

        def on_detect(pattern_name, original, replacement):
            detections_logged.append((pattern_name, original, replacement))

        config = PIIConfig(
            enabled=True,
            action="redact",
            patterns={"email": True},
            on_detect=on_detect,
        )
        handler = PIIHandler(config)

        handler.process_string("Contact user@example.com please")

        assert len(detections_logged) == 1
        assert detections_logged[0][0] == "email"
        assert detections_logged[0][1] == "user@example.com"
        assert detections_logged[0][2] == "[EMAIL_REDACTED]"

    def test_get_stats(self):
        """Test getting handler statistics."""
        config = PIIConfig(
            enabled=True,
            action="tokenize",
            patterns={"email": True, "phone": True},
        )
        handler = PIIHandler(config)

        handler.process_string("user@example.com")
        handler.process_string("555-123-4567")

        stats = handler.get_stats()
        assert stats["patterns_enabled"] == 2
        assert stats["tokens_stored"] == 2

    def test_multiple_pii_in_same_string(self):
        """Test handling multiple PII types in one string."""
        config = PIIConfig(
            enabled=True,
            action="redact",
            patterns={"email": True, "phone": True},
        )
        handler = PIIHandler(config)

        text = "Email: user@example.com, Phone: 555-123-4567"
        result, detections = handler.process_string(text)

        assert len(detections) == 2
        assert "[EMAIL_REDACTED]" in result
        assert "[PHONE_REDACTED]" in result


class TestCreatePIIHandler:
    """Tests for create_pii_handler helper."""

    def test_none_config_returns_none(self):
        """Test that None config returns None handler."""
        handler = create_pii_handler(None)
        assert handler is None

    def test_disabled_config_returns_none(self):
        """Test that disabled config returns None handler."""
        config = PIIConfig(enabled=False)
        handler = create_pii_handler(config)
        assert handler is None

    def test_enabled_config_returns_handler(self):
        """Test that enabled config returns handler."""
        config = PIIConfig(enabled=True)
        handler = create_pii_handler(config)
        assert handler is not None
        assert isinstance(handler, PIIHandler)

    def test_dict_config(self):
        """Test creating handler from dict config."""
        config_dict = {
            "enabled": True,
            "action": "hash",
            "patterns": {"email": True},
        }
        handler = create_pii_handler(config_dict)
        assert handler is not None
        assert handler.config.action == "hash"


class TestPIIIntegration:
    """Integration tests for PII with observe module."""

    def test_pii_with_observe_install(self, tmp_path):
        """Test PII configuration via observe.install()."""
        from agent_observe.config import CaptureMode, Config, Environment, SinkType
        from agent_observe.observe import Observe

        pii_config = PIIConfig(
            enabled=True,
            action="redact",
            patterns={"email": True},
        )

        config = Config(
            mode=CaptureMode.FULL,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=tmp_path / "pii_test.db",
        )

        obs = Observe()
        obs.install(config=config, pii=pii_config)

        # Verify PII handler was set on sink
        assert obs.sink.pii_handler is not None
        assert obs.sink.pii_handler.config.action == "redact"

        obs._cleanup()

    def test_pii_redaction_in_sink(self, tmp_path):
        """Test that PII is redacted before storage via sink."""
        from agent_observe.config import CaptureMode, Config, Environment, SinkType
        from agent_observe.observe import Observe
        from agent_observe.sinks.sqlite_sink import SQLiteSink

        pii_config = PIIConfig(
            enabled=True,
            action="redact",
            patterns={"email": True},
        )

        config = Config(
            mode=CaptureMode.FULL,
            env=Environment.DEV,
            sink_type=SinkType.SQLITE,
            sqlite_path=tmp_path / "pii_redact.db",
            pii=pii_config,  # Pass via config
        )

        obs = Observe()
        obs.install(config=config)

        # Verify PII handler is set
        assert obs.sink.pii_handler is not None

        # Test the sink's PII processing directly
        test_data = {
            "name": "test-span",
            "attrs": {
                "email": "user@example.com",
                "message": "Contact support@test.org for help",
            },
        }

        processed = obs.sink._process_pii(test_data)

        # Verify emails were redacted
        assert processed["attrs"]["email"] == "[EMAIL_REDACTED]"
        assert "[EMAIL_REDACTED]" in processed["attrs"]["message"]
        assert "user@example.com" not in processed["attrs"]["email"]
        assert "support@test.org" not in processed["attrs"]["message"]

        obs._cleanup()
