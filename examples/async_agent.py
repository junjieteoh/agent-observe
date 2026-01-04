"""
Async agent example for agent-observe.

Shows how to use observe with async/await patterns.
"""

import asyncio

from agent_observe import observe, tool, model_call
from agent_observe.config import Config


@tool(name="fetch_url", kind="http")
async def fetch_url(url: str) -> str:
    """Async tool that fetches a URL."""
    # In real code: async with aiohttp.ClientSession() as session:
    await asyncio.sleep(0.1)  # Simulate network delay
    return f"Content from {url}"


@model_call(provider="anthropic", model="claude-sonnet-4-20250514")
async def analyze_async(text: str) -> str:
    """Async model call."""
    # In real code: await anthropic.messages.create(...)
    await asyncio.sleep(0.1)
    return f"Analysis: {text[:50]}..."


async def main():
    observe.install(config=Config(mode="full", env="dev"))

    # Use arun() for async context
    async with observe.arun("async-agent", task={"urls": ["https://example.com"]}):
        # Parallel async tool calls
        contents = await asyncio.gather(
            fetch_url("https://example.com/page1"),
            fetch_url("https://example.com/page2"),
            fetch_url("https://example.com/page3"),
        )

        # Process results
        combined = "\n".join(contents)
        analysis = await analyze_async(combined)
        print(analysis)


if __name__ == "__main__":
    asyncio.run(main())
