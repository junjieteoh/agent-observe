"""Tests for policy engine."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent_observe.policy import (
    Policy,
    PolicyEngine,
    PolicyViolationError,
    load_policy,
)


class TestPolicy:
    """Tests for Policy dataclass."""

    def test_default_policy(self) -> None:
        """Test default policy allows all tools."""
        policy = Policy()

        assert policy.allow_tools == ["*"]
        assert policy.deny_tools == []
        assert policy.max_tool_calls == 100
        assert policy.max_retries == 10

    def test_glob_pattern_compilation(self) -> None:
        """Test that glob patterns are compiled to regex."""
        policy = Policy(
            allow_tools=["db.*", "http.*"],
            deny_tools=["*.destructive"],
        )

        assert len(policy._allow_patterns) == 2
        assert len(policy._deny_patterns) == 1


class TestLoadPolicy:
    """Tests for load_policy function."""

    def test_load_nonexistent_file(self) -> None:
        """Test loading from nonexistent file returns defaults."""
        policy = load_policy(Path("/nonexistent/path.yml"))

        assert policy.allow_tools == ["*"]
        assert policy.deny_tools == []

    def test_load_none_returns_default(self) -> None:
        """Test loading with None returns default policy."""
        policy = load_policy(None)

        assert policy.allow_tools == ["*"]

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        """Test loading policy from YAML file."""
        policy_yaml = {
            "tools": {
                "allow": ["safe.*", "utility.*"],
                "deny": ["shell.*", "*.dangerous"],
            },
            "limits": {
                "max_tool_calls": 50,
                "max_retries": 5,
                "max_model_calls": 25,
            },
            "sql": {
                "destructive_blocklist": ["DROP", "TRUNCATE"],
                "require_where": ["DELETE"],
                "allowed_datasets": ["analytics.*"],
            },
            "network": {
                "allowed_domains": ["*.example.com"],
            },
        }

        policy_path = tmp_path / "policy.yml"
        with open(policy_path, "w") as f:
            yaml.dump(policy_yaml, f)

        policy = load_policy(policy_path)

        assert policy.allow_tools == ["safe.*", "utility.*"]
        assert policy.deny_tools == ["shell.*", "*.dangerous"]
        assert policy.max_tool_calls == 50
        assert policy.max_retries == 5
        assert policy.sql.destructive_blocklist == ["DROP", "TRUNCATE"]
        assert policy.network.allowed_domains == ["*.example.com"]


class TestPolicyEngine:
    """Tests for PolicyEngine."""

    def test_check_tool_allowed_default(self) -> None:
        """Test default policy allows all tools."""
        engine = PolicyEngine(Policy())

        assert engine.check_tool_allowed("any_tool") is None
        assert engine.check_tool_allowed("another.tool") is None

    def test_check_tool_denied(self) -> None:
        """Test tool denial."""
        policy = Policy(deny_tools=["dangerous.*"])
        engine = PolicyEngine(policy)

        violation = engine.check_tool_allowed("dangerous.tool")

        assert violation is not None
        assert violation.rule == "tool_denied"
        assert "dangerous.tool" in violation.message

    def test_deny_takes_precedence(self) -> None:
        """Test that deny patterns take precedence over allow."""
        policy = Policy(
            allow_tools=["*"],
            deny_tools=["shell.*"],
        )
        engine = PolicyEngine(policy)

        # Allowed
        assert engine.check_tool_allowed("safe_tool") is None

        # Denied despite allow_tools=["*"]
        violation = engine.check_tool_allowed("shell.exec")
        assert violation is not None
        assert violation.rule == "tool_denied"

    def test_check_tool_not_in_allow_list(self) -> None:
        """Test tool not in allow list."""
        policy = Policy(allow_tools=["db.*", "http.*"])
        engine = PolicyEngine(policy)

        violation = engine.check_tool_allowed("file.read")

        assert violation is not None
        assert violation.rule == "tool_not_allowed"

    def test_check_tool_call_limit(self) -> None:
        """Test tool call limit checking."""
        policy = Policy(max_tool_calls=10)
        engine = PolicyEngine(policy)

        # Under limit
        assert engine.check_tool_call_limit(5) is None
        assert engine.check_tool_call_limit(9) is None

        # At/over limit
        violation = engine.check_tool_call_limit(10)
        assert violation is not None
        assert violation.rule == "tool_call_limit"

    def test_check_retry_limit(self) -> None:
        """Test retry limit checking."""
        policy = Policy(max_retries=5)
        engine = PolicyEngine(policy)

        assert engine.check_retry_limit(4) is None

        violation = engine.check_retry_limit(5)
        assert violation is not None
        assert violation.rule == "retry_limit"

    def test_fail_on_violation_raises(self) -> None:
        """Test that fail_on_violation raises exception."""
        policy = Policy(deny_tools=["blocked"])
        engine = PolicyEngine(policy, fail_on_violation=True)

        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_tool_allowed("blocked")

        assert exc_info.value.rule == "tool_denied"

    def test_check_sql_blocked_statement(self) -> None:
        """Test SQL statement blocking."""
        policy = Policy()
        engine = PolicyEngine(policy)

        # DROP should be blocked by default
        violation = engine.check_sql("DROP TABLE users")
        assert violation is not None
        assert violation.rule == "sql_destructive"
        assert "DROP" in violation.message

    def test_check_sql_require_where(self) -> None:
        """Test SQL WHERE requirement."""
        policy = Policy()
        engine = PolicyEngine(policy)

        # DELETE without WHERE should be flagged
        # But first, DELETE is in blocklist by default
        # Let's test with custom policy
        from agent_observe.policy import SQLPolicy

        custom_sql = SQLPolicy(
            destructive_blocklist=[],  # Allow all for this test
            require_where=["DELETE", "UPDATE"],
        )
        custom_policy = Policy(sql=custom_sql)
        engine = PolicyEngine(custom_policy)

        # Without WHERE
        violation = engine.check_sql("DELETE FROM users")
        assert violation is not None
        assert violation.rule == "sql_require_where"

        # With WHERE
        assert engine.check_sql("DELETE FROM users WHERE id = 1") is None

    def test_check_network_allowed_domain(self) -> None:
        """Test network domain checking."""
        from agent_observe.policy import NetworkPolicy

        network = NetworkPolicy(allowed_domains=["*.example.com", "api.safe.org"])
        policy = Policy(network=network)
        engine = PolicyEngine(policy)

        # Allowed
        assert engine.check_network("https://api.example.com/data") is None
        assert engine.check_network("https://sub.example.com/path") is None
        assert engine.check_network("https://api.safe.org/endpoint") is None

        # Not allowed
        violation = engine.check_network("https://malicious.com/steal")
        assert violation is not None
        assert violation.rule == "network_domain_not_allowed"

    def test_check_network_no_restrictions(self) -> None:
        """Test that empty allowed_domains means no restrictions."""
        policy = Policy()  # Default has no network restrictions
        engine = PolicyEngine(policy)

        assert engine.check_network("https://any.domain.com") is None
