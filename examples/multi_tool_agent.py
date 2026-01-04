"""
Multi-Tool Agent Example

Demonstrates advanced usage of agent-observe including:
- Multiple tool types (db, http, file)
- Policy enforcement
- Replay mode for testing
- Risk scoring

Run with:
    python examples/multi_tool_agent.py

Then view results with:
    agent-observe view
"""

import os
import time

from agent_observe import model_call, observe, tool


# Database tool
@tool(name="db.query", kind="db", version="2")
def query_database(sql: str) -> list[dict]:
    """Execute a SQL query (simulated)."""
    # Simulate database latency
    time.sleep(0.1)

    # Return mock data
    if "users" in sql.lower():
        return [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"},
        ]
    elif "orders" in sql.lower():
        return [
            {"id": 101, "user_id": 1, "amount": 99.99},
            {"id": 102, "user_id": 2, "amount": 149.99},
        ]
    return []


# HTTP tool
@tool(name="http.get", kind="http")
def fetch_url(url: str) -> dict:
    """Fetch data from a URL (simulated)."""
    time.sleep(0.05)

    if "api.example.com" in url:
        return {"status": "ok", "data": {"message": "Hello from API"}}
    return {"status": "error", "message": "Unknown endpoint"}


# File tool
@tool(name="file.write", kind="file")
def write_report(filename: str, content: str) -> bool:
    """Write a report to a file (simulated)."""
    # In a real agent, this would write to disk
    print(f"[Simulated] Writing {len(content)} chars to {filename}")
    return True


# Dangerous tool (will trigger policy if denied)
@tool(name="shell.execute", kind="shell")
def execute_shell(command: str) -> str:
    """Execute a shell command (simulated, dangerous!)."""
    # This should be denied by policy in production
    return f"[Simulated] Executed: {command}"


# Model call
@model_call(provider="openai", model="gpt-4")
def analyze_data(data: list[dict]) -> dict:
    """Analyze data using an LLM (simulated)."""
    time.sleep(0.1)

    total = sum(item.get("amount", 0) for item in data if isinstance(item.get("amount"), (int, float)))
    return {
        "analysis": "Data analysis complete",
        "total_amount": total,
        "record_count": len(data),
    }


def run_multi_tool_agent():
    """Run the multi-tool data analysis agent."""
    # Install with explicit configuration
    observe.install(mode="metadata_only")

    # Task definition
    task = {
        "goal": "Analyze user orders and generate report",
        "sources": ["database", "api"],
        "output": "report.txt",
    }

    with observe.run("data-analysis-agent", task=task, agent_version="2.1.0"):
        observe.emit_event("workflow.started", {"step": "data_collection"})

        # Collect data from multiple sources
        users = query_database("SELECT * FROM users")
        orders = query_database("SELECT * FROM orders WHERE user_id IN (1, 2)")

        # Fetch supplementary data from API
        api_data = fetch_url("https://api.example.com/enrichment")

        observe.emit_event("workflow.progress", {"step": "analysis", "records": len(orders)})

        # Analyze the data
        analysis = analyze_data(orders)

        # Generate report
        report_content = f"""
        Data Analysis Report
        ====================
        Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}

        Summary:
        - Users analyzed: {len(users)}
        - Orders analyzed: {len(orders)}
        - Total order value: ${analysis['total_amount']:.2f}

        API Status: {api_data['status']}
        """

        # Write report
        write_report("report.txt", report_content)

        # Emit artifact
        observe.emit_artifact(
            "analysis_report",
            {
                "users": len(users),
                "orders": len(orders),
                "total_value": analysis["total_amount"],
            },
            provenance=["db.query", "http.get"],
        )

        observe.emit_event("workflow.completed", {"success": True})

        print(report_content)
        return analysis


def demonstrate_policy_violation():
    """Demonstrate policy violation handling."""
    print("\n--- Demonstrating Policy Violation ---\n")

    # Create a policy file
    from pathlib import Path

    import yaml

    policy_dir = Path(".riff")
    policy_dir.mkdir(exist_ok=True)

    policy = {
        "tools": {
            "allow": ["db.*", "http.*", "file.*"],
            "deny": ["shell.*"],  # Block shell commands
        },
        "limits": {
            "max_tool_calls": 10,
            "max_model_calls": 5,
        },
        "sql": {
            "destructive_blocklist": ["DROP", "DELETE", "TRUNCATE"],
        },
    }

    policy_path = policy_dir / "observe.policy.yml"
    with open(policy_path, "w") as f:
        yaml.dump(policy, f)

    # Set environment variable for policy file
    os.environ["AGENT_OBSERVE_POLICY_FILE"] = str(policy_path)
    os.environ["AGENT_OBSERVE_ENV"] = "dev"

    # Create new observe instance with policy
    from agent_observe.observe import Observe

    obs = Observe()
    obs.install()

    with obs.run("policy-demo-agent"):
        # This should work
        users = query_database("SELECT * FROM users")
        print(f"Query succeeded, got {len(users)} users")

        # This should trigger a policy violation (but not block since fail_on_violation=False)
        try:
            result = execute_shell("ls -la")
            print(f"Shell executed (policy violation recorded): {result}")
        except Exception as e:
            print(f"Shell blocked by policy: {e}")

    # Check the risk score
    runs = obs.sink.get_runs(name="policy-demo")
    if runs:
        print(f"\nRisk Score: {runs[0].get('risk_score', 'N/A')}")
        print(f"Eval Tags: {runs[0].get('eval_tags', [])}")
        print(f"Policy Violations: {runs[0].get('policy_violations', 0)}")


if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Tool Agent Example")
    print("=" * 60)

    # Run the main agent
    result = run_multi_tool_agent()

    print("\n" + "=" * 60)
    print("Analysis Result:", result)
    print("=" * 60)

    # Demonstrate policy
    demonstrate_policy_violation()

    print("\n" + "=" * 60)
    print("View traces with: agent-observe view")
    print("=" * 60)
