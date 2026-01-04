"""
Policy configuration example for agent-observe.

Shows how to use the policy engine to allow/deny tool patterns.
"""

from agent_observe import observe, tool
from agent_observe.config import Config
from agent_observe.policy import PolicyViolation

# First, create a policy file at .riff/observe.policy.yml:
EXAMPLE_POLICY_YAML = """
# .riff/observe.policy.yml
tools:
  allow:
    - "db.read"
    - "db.query"
    - "http.*"
    - "cache.*"
  deny:
    - "db.delete"
    - "db.drop"
    - "shell.*"
    - "*.destructive"

limits:
  max_tool_calls: 100
  max_model_calls: 50
"""


@tool(name="db.read", kind="database")
def read_from_db(query: str) -> list:
    """Allowed: matches db.read pattern."""
    return [{"id": 1, "name": "Example"}]


@tool(name="db.delete", kind="database")
def delete_from_db(table: str) -> bool:
    """Denied: matches db.delete pattern."""
    return True


@tool(name="shell.exec", kind="system")
def run_shell_command(cmd: str) -> str:
    """Denied: matches shell.* pattern."""
    return "output"


def main():
    # Install with policy enforcement
    config = Config(
        mode="full",
        env="dev",
        policy_file=".riff/observe.policy.yml",
        fail_on_violation=True,  # Raises exception on violation
    )
    observe.install(config=config)

    with observe.run("policy-demo"):
        # This works - db.read is allowed
        data = read_from_db("SELECT * FROM users")
        print(f"Read data: {data}")

        # This would raise PolicyViolation if fail_on_violation=True
        # Or just log warning and increment policy_violations counter
        try:
            delete_from_db("users")
        except PolicyViolation as e:
            print(f"Blocked: {e}")


if __name__ == "__main__":
    print("Example policy YAML:")
    print(EXAMPLE_POLICY_YAML)
    # main()  # Uncomment after creating policy file
