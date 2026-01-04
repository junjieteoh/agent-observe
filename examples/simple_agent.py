"""
Simple Agent Example

Demonstrates basic usage of agent-observe with a simple agent that:
- Uses the @tool decorator
- Uses the @model_call decorator
- Emits custom events and artifacts

Run with:
    python examples/simple_agent.py

Then view results with:
    agent-observe view
"""

from agent_observe import model_call, observe, tool


# Define tools with the @tool decorator
@tool(name="calculator.add", kind="compute", version="1")
def add_numbers(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool(name="calculator.multiply", kind="compute", version="1")
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool(name="text.format", kind="text")
def format_result(value: float, label: str) -> str:
    """Format a result for display."""
    return f"{label}: {value:.2f}"


# Simulate an LLM call with @model_call
@model_call(provider="mock", model="calculator-v1")
def plan_calculation(_prompt: str) -> dict:
    """Mock LLM that plans a calculation."""
    # In a real agent, this would call an LLM
    return {
        "operations": [
            {"op": "add", "args": [10, 20]},
            {"op": "multiply", "args": ["result", 3]},
        ],
        "reasoning": "First add, then multiply the result",
    }


def run_simple_agent():
    """Run the simple calculator agent."""
    # Install observability (zero-config, uses SQLite in dev mode)
    observe.install()

    # Start a run context
    with observe.run("simple-calculator-agent", task={"goal": "compute (10+20)*3"}):
        # Emit a custom event
        observe.emit_event("agent.started", {"goal": "compute (10+20)*3"})

        # Plan the calculation (simulated LLM call)
        _plan = plan_calculation("How do I compute (10+20)*3?")

        # Execute the plan
        result = add_numbers(10, 20)
        result = multiply_numbers(result, 3)

        # Format and emit artifact
        formatted = format_result(result, "Final Result")
        observe.emit_artifact("calculation_result", {
            "expression": "(10+20)*3",
            "result": result,
            "formatted": formatted,
        })

        # Emit completion event
        observe.emit_event("agent.completed", {"success": True, "result": result})

        print(formatted)
        return result


if __name__ == "__main__":
    result = run_simple_agent()
    print(f"\nComputation complete. Result: {result}")
    print("\nView the trace with: agent-observe view")
