"""
Capture modes example for agent-observe.

Demonstrates the different capture modes and when to use each.
"""

from agent_observe import observe, tool
from agent_observe.config import Config

# Option 1: Full capture (development/debugging)
# Stores everything - inputs, outputs, timing
config_full = Config(mode="full", env="dev")

# Option 2: Evidence only (debugging with size limits)
# Stores small content + hashes
config_evidence = Config(mode="evidence_only", env="staging")

# Option 3: Metadata only (production default)
# Stores only hashes and timing - no raw content
config_metadata = Config(mode="metadata_only", env="prod")

# Option 4: Off (disable completely)
config_off = Config(mode="off")


@tool(name="process_pii", kind="internal")
def process_user_data(user_email: str, _ssn: str) -> dict:
    """Tool that handles sensitive data (ssn intentionally unused in demo)."""
    return {"processed": True, "user": user_email[:3] + "***"}


def demo_metadata_mode():
    """
    In metadata_only mode (production default):
    - Input "user@example.com" is stored as hash: "4a8c9d2e..."
    - Output is stored as hash: "b7f3a1c0..."
    - No PII leakage, but you can still verify consistency
    """
    observe.install(config=Config(mode="metadata_only"))

    with observe.run("pii-processor"):
        result = process_user_data("user@example.com", "123-45-6789")
        print(f"Result: {result}")

    # Query later: you'll see timing, success/failure, but no raw data


def demo_full_mode():
    """
    In full mode (development only):
    - Input "user@example.com" is stored in full
    - Output is stored in full
    - Great for debugging, but never use in production with real PII
    """
    observe.install(config=Config(mode="full", env="dev"))

    with observe.run("debug-agent"):
        result = process_user_data("test@example.com", "000-00-0000")
        print(f"Result: {result}")


if __name__ == "__main__":
    demo_metadata_mode()
