"""
Basic usage example for agent-observe.

Shows how to set up observability with minimal configuration.
"""

from agent_observe import observe, tool, model_call

# Initialize with zero-config (reads from environment)
observe.install()


@tool(name="search", kind="http")
def search_web(query: str) -> dict:
    """Example tool that would call a search API."""
    # In real code: return requests.get(f"https://api.search.com?q={query}").json()
    return {"results": [{"title": "AI News", "url": "https://example.com"}]}


@model_call(provider="openai", model="gpt-4")
def call_llm(prompt: str) -> str:
    """Example model call."""
    # In real code: return openai.chat.completions.create(...).choices[0].message.content
    return f"Analysis of: {prompt[:50]}..."


def main():
    # Everything inside observe.run() is tracked
    with observe.run("research-agent", task={"goal": "Research AI trends"}):
        results = search_web("AI agents 2025")
        analysis = call_llm(f"Analyze these results: {results}")
        print(analysis)


if __name__ == "__main__":
    main()
