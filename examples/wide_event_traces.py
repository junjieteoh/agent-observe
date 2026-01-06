#!/usr/bin/env python3
"""
Example: Wide Event Traces (v0.1.7+)

Demonstrates the v0.1.7 Wide Event trace capture features:
- Run attribution (user_id, session_id, prompt_version)
- Input/output capture
- Custom metadata
- Full LLM context capture
- Session continuity

Run with:
    python examples/wide_event_traces.py
"""

from agent_observe import observe, tool, model_call
from agent_observe.config import Config


# Mock LLM response for demo (replace with actual LLM call in production)
def mock_llm_response(messages: list) -> dict:
    """Simulate an LLM response."""
    user_msg = messages[-1]["content"] if messages else ""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": f"I received your message: '{user_msg[:50]}...'",
                }
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
    }


# Wrap your LLM call - the decorator captures full context
@model_call(provider="openai", model="gpt-4")
def call_llm(messages: list, temperature: float = 0.7, max_tokens: int = 1000) -> str:
    """Call LLM with full context capture."""
    response = mock_llm_response(messages)
    return response["choices"][0]["message"]["content"]


# Wrap your tools
@tool(name="lookup_order", kind="db")
def lookup_order(order_id: str) -> dict:
    """Look up order details from database."""
    return {
        "order_id": order_id,
        "status": "shipped",
        "tracking": "1Z999AA10123456784",
        "eta": "2024-01-20",
    }


@tool(name="search_faq", kind="http")
def search_faq(query: str) -> list:
    """Search FAQ database."""
    return [
        {"title": "Shipping times", "content": "Orders ship within 2-3 days."},
        {"title": "Returns", "content": "30-day return policy."},
    ]


def main():
    # Initialize with full capture (default in v0.1.7)
    config = Config(mode="full", env="dev")
    observe.install(config=config)

    print("=" * 60)
    print("Wide Event Traces Demo (v0.1.7)")
    print("=" * 60)

    # Simulate a user session with multiple turns
    session_id = "session_abc123"
    user_id = "customer_jane"

    # --- First turn: User asks about their order ---
    print("\n[Turn 1] User: Where is my order #12345?")

    with observe.run(
        "support-agent",
        user_id=user_id,
        session_id=session_id,
        prompt_version="v2.3",
        experiment_id="ab_test_new_rag",
        model_config={"model": "gpt-4", "temperature": 0.7},
    ) as run:
        # Capture the user's input
        user_message = "Where is my order #12345?"
        run.set_input(user_message)

        # Add custom metadata
        run.add_metadata("customer_tier", "premium")
        run.add_metadata("channel", "web_chat")

        # Agent processing
        order_info = lookup_order("12345")

        response = call_llm(
            messages=[
                {"role": "system", "content": "You are a helpful support agent."},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": f"Let me look that up... Order info: {order_info}"},
            ],
            temperature=0.7,
        )

        # Capture the agent's output
        final_response = f"Your order #{order_info['order_id']} is {order_info['status']}. Tracking: {order_info['tracking']}"
        run.set_output(final_response)

        print(f"[Agent]: {final_response}")

    # --- Second turn: Follow-up question (same session) ---
    print("\n[Turn 2] User: What's your return policy?")

    with observe.run(
        "support-agent",
        user_id=user_id,
        session_id=session_id,  # Same session!
        prompt_version="v2.3",
    ) as run:
        user_message = "What's your return policy?"
        run.set_input(user_message)

        # Search FAQ
        faq_results = search_faq("return policy")

        response = call_llm(
            messages=[
                {"role": "system", "content": "You are a helpful support agent."},
                {"role": "user", "content": user_message},
            ],
        )

        final_response = f"We have a {faq_results[1]['content']}"
        run.set_output(final_response)

        print(f"[Agent]: {final_response}")

    # --- Third turn: Different user, different session ---
    print("\n[Turn 3] Different user: How long does shipping take?")

    with observe.run(
        "support-agent",
        user_id="customer_bob",
        session_id="session_xyz789",  # New session
        prompt_version="v2.3",
    ) as run:
        user_message = "How long does shipping take?"
        run.set_input(user_message)

        faq_results = search_faq("shipping times")

        response = call_llm(
            messages=[
                {"role": "system", "content": "You are a helpful support agent."},
                {"role": "user", "content": user_message},
            ],
        )

        final_response = faq_results[0]["content"]
        run.set_output(final_response)

        print(f"[Agent]: {final_response}")

    print("\n" + "=" * 60)
    print("Done! View traces with: agent-observe view")
    print("=" * 60)

    # Show what was captured
    print("\n--- What was captured (Wide Event) ---")
    print("""
Each run now includes:
- user_id: Who made the request
- session_id: Links runs in a conversation
- prompt_version: Which prompt version was used
- prompt_hash: Auto-calculated hash of system prompt
- input_json/input_text: What the user asked
- output_json/output_text: What the agent responded
- metadata: Custom key-value pairs
- Full LLM context in each @model_call span

Query examples:
  SELECT * FROM runs WHERE session_id = 'session_abc123'
  SELECT * FROM runs WHERE user_id = 'customer_jane'
  SELECT * FROM runs WHERE prompt_version = 'v2.3'
""")


if __name__ == "__main__":
    main()
