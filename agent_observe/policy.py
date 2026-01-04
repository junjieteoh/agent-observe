"""
Policy engine for agent-observe.

Provides YAML-based policy rules for controlling agent behavior:
- Tool allow/deny patterns (glob-based)
- Limits (max tool calls, retries)
- SQL safety rules
- Network domain restrictions
"""

from __future__ import annotations

import fnmatch
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class PolicyViolationError(Exception):
    """Raised when a policy violation blocks execution."""

    def __init__(self, message: str, rule: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.rule = rule
        self.details = details or {}


@dataclass
class PolicyViolation:
    """Represents a policy violation."""

    rule: str
    message: str
    tool_name: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "message": self.message,
            "tool_name": self.tool_name,
            "details": self.details,
        }


@dataclass
class SQLPolicy:
    """SQL-specific policy rules."""

    # SQL statements that are never allowed
    destructive_blocklist: list[str] = field(
        default_factory=lambda: [
            "DROP", "TRUNCATE", "DELETE", "ALTER", "CREATE", "INSERT", "UPDATE"
        ]
    )
    # Require WHERE clause for certain statements
    require_where: list[str] = field(default_factory=lambda: ["DELETE", "UPDATE"])
    # Allowed dataset/table patterns
    allowed_datasets: list[str] = field(default_factory=list)


@dataclass
class NetworkPolicy:
    """Network-specific policy rules."""

    # Allowed domain patterns
    allowed_domains: list[str] = field(default_factory=list)


@dataclass
class Policy:
    """Complete policy configuration."""

    # Tool patterns
    allow_tools: list[str] = field(default_factory=lambda: ["*"])
    deny_tools: list[str] = field(default_factory=list)

    # Limits
    max_tool_calls: int = 100
    max_retries: int = 10
    max_model_calls: int = 50

    # Sub-policies
    sql: SQLPolicy = field(default_factory=SQLPolicy)
    network: NetworkPolicy = field(default_factory=NetworkPolicy)

    # Compiled patterns (internal)
    _allow_patterns: list[re.Pattern[str]] = field(default_factory=list, repr=False)
    _deny_patterns: list[re.Pattern[str]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Compile glob patterns to regex for faster matching."""
        self._allow_patterns = [self._glob_to_regex(p) for p in self.allow_tools]
        self._deny_patterns = [self._glob_to_regex(p) for p in self.deny_tools]

    @staticmethod
    def _glob_to_regex(pattern: str) -> re.Pattern[str]:
        """Convert glob pattern to compiled regex."""
        regex = fnmatch.translate(pattern)
        return re.compile(regex, re.IGNORECASE)


def load_policy(path: Path | None = None) -> Policy:
    """
    Load policy from YAML file.

    Expected YAML structure:
    ```yaml
    tools:
      allow:
        - "*"
      deny:
        - "shell.*"
        - "*.destructive"

    limits:
      max_tool_calls: 100
      max_retries: 10
      max_model_calls: 50

    sql:
      destructive_blocklist:
        - DROP
        - TRUNCATE
      require_where:
        - DELETE
        - UPDATE
      allowed_datasets:
        - "analytics.*"
        - "public.*"

    network:
      allowed_domains:
        - "*.example.com"
        - "api.openai.com"
    ```

    Args:
        path: Path to YAML policy file. If None, returns default policy.

    Returns:
        Loaded Policy instance.
    """
    if path is None:
        return Policy()

    if not path.exists():
        logger.warning(f"Policy file not found: {path}, using defaults")
        return Policy()

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse policy file {path}: {e}")
        return Policy()

    # Parse tools section
    tools = data.get("tools", {})
    allow_tools = tools.get("allow", ["*"])
    deny_tools = tools.get("deny", [])

    # Parse limits section
    limits = data.get("limits", {})
    max_tool_calls = limits.get("max_tool_calls", 100)
    max_retries = limits.get("max_retries", 10)
    max_model_calls = limits.get("max_model_calls", 50)

    # Parse SQL section
    sql_data = data.get("sql", {})
    sql = SQLPolicy(
        destructive_blocklist=sql_data.get(
            "destructive_blocklist",
            ["DROP", "TRUNCATE", "DELETE", "ALTER", "CREATE", "INSERT", "UPDATE"],
        ),
        require_where=sql_data.get("require_where", ["DELETE", "UPDATE"]),
        allowed_datasets=sql_data.get("allowed_datasets", []),
    )

    # Parse network section
    network_data = data.get("network", {})
    network = NetworkPolicy(
        allowed_domains=network_data.get("allowed_domains", []),
    )

    return Policy(
        allow_tools=allow_tools,
        deny_tools=deny_tools,
        max_tool_calls=max_tool_calls,
        max_retries=max_retries,
        max_model_calls=max_model_calls,
        sql=sql,
        network=network,
    )


class PolicyEngine:
    """
    Policy enforcement engine.

    Checks tool calls and other operations against policy rules.
    """

    def __init__(self, policy: Policy, fail_on_violation: bool = False):
        """
        Initialize policy engine.

        Args:
            policy: Policy configuration.
            fail_on_violation: If True, raise PolicyViolationError on violations.
        """
        self.policy = policy
        self.fail_on_violation = fail_on_violation

    def check_tool_allowed(self, tool_name: str) -> PolicyViolation | None:
        """
        Check if a tool is allowed by policy.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            PolicyViolation if denied, None if allowed.
        """
        # Check deny patterns first (deny takes precedence)
        for pattern in self.policy._deny_patterns:
            if pattern.match(tool_name):
                violation = PolicyViolation(
                    rule="tool_denied",
                    message=f"Tool '{tool_name}' is denied by policy",
                    tool_name=tool_name,
                    details={"pattern": pattern.pattern},
                )
                if self.fail_on_violation:
                    raise PolicyViolationError(
                        violation.message, violation.rule, violation.details
                    )
                return violation

        # Check allow patterns
        for pattern in self.policy._allow_patterns:
            if pattern.match(tool_name):
                return None

        # Not in allow list
        violation = PolicyViolation(
            rule="tool_not_allowed",
            message=f"Tool '{tool_name}' is not in allow list",
            tool_name=tool_name,
        )
        if self.fail_on_violation:
            raise PolicyViolationError(violation.message, violation.rule, violation.details)
        return violation

    def check_tool_call_limit(self, current_count: int) -> PolicyViolation | None:
        """Check if tool call limit has been reached."""
        if current_count >= self.policy.max_tool_calls:
            violation = PolicyViolation(
                rule="tool_call_limit",
                message=f"Tool call limit reached ({self.policy.max_tool_calls})",
                details={"limit": self.policy.max_tool_calls, "current": current_count},
            )
            if self.fail_on_violation:
                raise PolicyViolationError(violation.message, violation.rule, violation.details)
            return violation
        return None

    def check_retry_limit(self, current_count: int) -> PolicyViolation | None:
        """Check if retry limit has been reached."""
        if current_count >= self.policy.max_retries:
            violation = PolicyViolation(
                rule="retry_limit",
                message=f"Retry limit reached ({self.policy.max_retries})",
                details={"limit": self.policy.max_retries, "current": current_count},
            )
            if self.fail_on_violation:
                raise PolicyViolationError(violation.message, violation.rule, violation.details)
            return violation
        return None

    def check_model_call_limit(self, current_count: int) -> PolicyViolation | None:
        """Check if model call limit has been reached."""
        if current_count >= self.policy.max_model_calls:
            violation = PolicyViolation(
                rule="model_call_limit",
                message=f"Model call limit reached ({self.policy.max_model_calls})",
                details={"limit": self.policy.max_model_calls, "current": current_count},
            )
            if self.fail_on_violation:
                raise PolicyViolationError(violation.message, violation.rule, violation.details)
            return violation
        return None

    def check_sql(self, query: str) -> PolicyViolation | None:
        """
        Check SQL query against policy.

        Args:
            query: SQL query string.

        Returns:
            PolicyViolation if query violates policy, None otherwise.
        """
        query_upper = query.strip().upper()

        # Check for blocked statements
        for stmt in self.policy.sql.destructive_blocklist:
            # Match statement at start of query or after common prefixes
            if re.search(rf"\b{stmt}\b", query_upper):
                violation = PolicyViolation(
                    rule="sql_destructive",
                    message=f"SQL statement '{stmt}' is blocked by policy",
                    details={"blocked_statement": stmt, "query_preview": query[:100]},
                )
                if self.fail_on_violation:
                    raise PolicyViolationError(
                        violation.message, violation.rule, violation.details
                    )
                return violation

        # Check for WHERE clause requirement
        for stmt in self.policy.sql.require_where:
            stmt_matches = query_upper.startswith(stmt) or re.search(
                rf"^\s*{stmt}\b", query_upper
            )
            if stmt_matches and "WHERE" not in query_upper:
                violation = PolicyViolation(
                    rule="sql_require_where",
                    message=f"SQL statement '{stmt}' requires WHERE clause",
                    details={"statement": stmt},
                )
                if self.fail_on_violation:
                    raise PolicyViolationError(
                        violation.message, violation.rule, violation.details
                    )
                return violation

        # Check allowed datasets (if configured)
        if self.policy.sql.allowed_datasets:
            # Extract table references (basic heuristic)
            tables = self._extract_table_refs(query)
            for table in tables:
                if not self._matches_any_pattern(table, self.policy.sql.allowed_datasets):
                    violation = PolicyViolation(
                        rule="sql_dataset_not_allowed",
                        message=f"Table/dataset '{table}' is not in allowed list",
                        details={"table": table},
                    )
                    if self.fail_on_violation:
                        raise PolicyViolationError(
                            violation.message, violation.rule, violation.details
                        )
                    return violation

        return None

    def check_network(self, url: str) -> PolicyViolation | None:
        """
        Check network URL against policy.

        Args:
            url: URL being accessed.

        Returns:
            PolicyViolation if URL violates policy, None otherwise.
        """
        if not self.policy.network.allowed_domains:
            # No restrictions configured
            return None

        # Extract domain from URL
        from urllib.parse import urlparse

        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Remove port if present
        if ":" in domain:
            domain = domain.split(":")[0]

        if not self._matches_any_pattern(domain, self.policy.network.allowed_domains):
            violation = PolicyViolation(
                rule="network_domain_not_allowed",
                message=f"Domain '{domain}' is not in allowed list",
                details={"domain": domain, "url": url},
            )
            if self.fail_on_violation:
                raise PolicyViolationError(violation.message, violation.rule, violation.details)
            return violation

        return None

    @staticmethod
    def _extract_table_refs(query: str) -> list[str]:
        """Extract table references from SQL query (basic heuristic)."""
        tables = []

        # Match FROM/JOIN clauses
        patterns = [
            r"\bFROM\s+([`\"\']?[\w.]+[`\"\']?)",
            r"\bJOIN\s+([`\"\']?[\w.]+[`\"\']?)",
            r"\bINTO\s+([`\"\']?[\w.]+[`\"\']?)",
            r"\bUPDATE\s+([`\"\']?[\w.]+[`\"\']?)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                # Clean up quotes
                table = match.strip("`\"'")
                if table:
                    tables.append(table)

        return tables

    @staticmethod
    def _matches_any_pattern(value: str, patterns: list[str]) -> bool:
        """Check if value matches any glob pattern."""
        return any(fnmatch.fnmatch(value.lower(), pattern.lower()) for pattern in patterns)
