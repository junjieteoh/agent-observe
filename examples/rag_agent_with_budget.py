"""
RAG Agent with Budget Enforcement

A realistic RAG (Retrieval-Augmented Generation) agent that demonstrates:
- Document retrieval and embedding generation
- Budget tracking and enforcement (stop before overspending)
- Semantic caching (skip LLM calls for similar queries)
- Multi-step reasoning with tool chains
- Error handling and retries

This shows how to build a production-ready RAG system with cost controls.
"""

import hashlib
import time
from dataclasses import dataclass
from typing import Optional

from agent_observe import observe, tool, model_call, HookResult, CircuitBreakerConfig
from agent_observe.config import Config, CaptureMode, Environment


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BudgetConfig:
    """Budget configuration for the agent."""
    max_cost_usd: float = 1.00  # Maximum spend per session
    max_tokens: int = 10000    # Maximum tokens per session
    warn_at_percent: float = 0.8  # Warn at 80% of budget


# Global budget tracker
class BudgetTracker:
    def __init__(self, config: BudgetConfig):
        self.config = config
        self.total_cost = 0.0
        self.total_tokens = 0
        self.calls = 0

    def add(self, cost: float, tokens: int):
        self.total_cost += cost
        self.total_tokens += tokens
        self.calls += 1

    @property
    def remaining_budget(self) -> float:
        return max(0, self.config.max_cost_usd - self.total_cost)

    @property
    def is_over_budget(self) -> bool:
        return self.total_cost >= self.config.max_cost_usd

    @property
    def should_warn(self) -> bool:
        return self.total_cost >= self.config.max_cost_usd * self.config.warn_at_percent


budget = BudgetTracker(BudgetConfig(max_cost_usd=0.50))


# =============================================================================
# SEMANTIC CACHE
# =============================================================================

class SemanticCache:
    """
    Simple semantic cache using query hashing.

    In production, use vector similarity with embeddings:
    - Store (embedding, response) pairs
    - Find similar queries by cosine similarity
    - Return cached response if similarity > threshold
    """

    def __init__(self, similarity_threshold: float = 0.95):
        self.cache: dict[str, dict] = {}
        self.threshold = similarity_threshold
        self.hits = 0
        self.misses = 0

    def _hash_query(self, query: str) -> str:
        """Simple hash for demo. Use embeddings in production."""
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def get(self, query: str) -> Optional[dict]:
        key = self._hash_query(query)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def set(self, query: str, response: dict):
        key = self._hash_query(query)
        self.cache[key] = response

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


cache = SemanticCache()


# =============================================================================
# SIMULATED VECTOR STORE
# =============================================================================

class MockVectorStore:
    """Simulated vector database for demo purposes."""

    DOCUMENTS = [
        {
            "id": "doc1",
            "content": "Python is a programming language known for its simplicity and readability. "
                       "It was created by Guido van Rossum and first released in 1991.",
            "metadata": {"source": "wiki", "topic": "programming"},
        },
        {
            "id": "doc2",
            "content": "Machine learning is a subset of artificial intelligence that enables systems "
                       "to learn from data. Deep learning uses neural networks with multiple layers.",
            "metadata": {"source": "textbook", "topic": "ai"},
        },
        {
            "id": "doc3",
            "content": "RAG (Retrieval-Augmented Generation) combines information retrieval with "
                       "text generation. It first retrieves relevant documents, then generates answers.",
            "metadata": {"source": "research", "topic": "ai"},
        },
        {
            "id": "doc4",
            "content": "Vector databases store embeddings for semantic search. Popular options include "
                       "Pinecone, Weaviate, Milvus, and ChromaDB.",
            "metadata": {"source": "docs", "topic": "databases"},
        },
        {
            "id": "doc5",
            "content": "LangChain is a framework for building LLM applications. It provides chains, "
                       "agents, and tools for common patterns like RAG and agents.",
            "metadata": {"source": "docs", "topic": "frameworks"},
        },
    ]

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Simulate semantic search. In production, use actual embeddings."""
        # Simple keyword matching for demo
        query_terms = set(query.lower().split())
        scored_docs = []

        for doc in self.DOCUMENTS:
            content_terms = set(doc["content"].lower().split())
            overlap = len(query_terms & content_terms)
            if overlap > 0:
                scored_docs.append((overlap, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]


vector_store = MockVectorStore()


# =============================================================================
# TOOLS
# =============================================================================

@tool(name="embeddings.generate", kind="model")
def generate_embedding(text: str) -> dict:
    """Generate embedding for text. In production: call OpenAI embeddings API."""
    # Simulated embedding generation
    time.sleep(0.1)  # Simulate API latency
    return {
        "embedding": [0.1] * 1536,  # Simulated 1536-dim embedding
        "tokens": len(text.split()),
        "model": "text-embedding-3-small",
    }


@tool(name="vectordb.search", kind="database")
def search_documents(query: str, top_k: int = 3) -> dict:
    """Search vector database for relevant documents."""
    results = vector_store.search(query, top_k)
    return {
        "query": query,
        "results": results,
        "count": len(results),
    }


@tool(name="vectordb.get", kind="database")
def get_document(doc_id: str) -> dict:
    """Retrieve a specific document by ID."""
    for doc in vector_store.DOCUMENTS:
        if doc["id"] == doc_id:
            return {"found": True, "document": doc}
    return {"found": False, "error": f"Document {doc_id} not found"}


# =============================================================================
# MODEL CALLS
# =============================================================================

@model_call(provider="openai", model="gpt-4")
def generate_answer(query: str, context: str) -> dict:
    """Generate answer using retrieved context."""
    # In production: call OpenAI API
    # response = openai.chat.completions.create(
    #     model="gpt-4",
    #     messages=[
    #         {"role": "system", "content": f"Answer based on context:\n{context}"},
    #         {"role": "user", "content": query},
    #     ],
    # )

    # Simulated response
    answer = f"Based on the retrieved documents: {context[:200]}... The answer to '{query}' is synthesized from the above context."

    return {
        "content": answer,
        "usage": {
            "prompt_tokens": len(context.split()) + len(query.split()) + 20,
            "completion_tokens": len(answer.split()),
        },
    }


@model_call(provider="openai", model="gpt-3.5-turbo")
def rewrite_query(original_query: str) -> dict:
    """Rewrite query for better retrieval."""
    # Simulated query rewriting
    rewritten = f"detailed information about {original_query}"

    return {
        "content": rewritten,
        "usage": {
            "prompt_tokens": len(original_query.split()) + 10,
            "completion_tokens": len(rewritten.split()),
        },
    }


# =============================================================================
# HOOKS - Budget Enforcement, Caching, Cost Tracking
# =============================================================================

PRICING = {
    "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000},
    "gpt-3.5-turbo": {"input": 0.001 / 1000, "output": 0.002 / 1000},
    "text-embedding-3-small": {"input": 0.00002 / 1000},
}


@observe.hooks.before_model
def check_budget(ctx):
    """Block model calls if over budget."""
    if budget.is_over_budget:
        return HookResult.block(
            f"Budget exceeded: ${budget.total_cost:.4f} / ${budget.config.max_cost_usd:.2f}"
        )

    if budget.should_warn:
        print(f"[WARNING] Budget at {budget.total_cost / budget.config.max_cost_usd * 100:.0f}%")

    return HookResult.proceed()


@observe.hooks.after_model
def track_cost(ctx, result):
    """Track cost after each model call."""
    if isinstance(result, dict) and "usage" in result:
        usage = result["usage"]
        model = ctx.model or "gpt-4"
        pricing = PRICING.get(model, PRICING["gpt-4"])

        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_tokens = input_tokens + output_tokens

        input_cost = input_tokens * pricing.get("input", 0)
        output_cost = output_tokens * pricing.get("output", 0)
        call_cost = input_cost + output_cost

        budget.add(call_cost, total_tokens)

        # Record in span
        ctx.span.set_attribute("cost_usd", round(call_cost, 6))
        ctx.span.set_attribute("tokens_input", input_tokens)
        ctx.span.set_attribute("tokens_output", output_tokens)
        ctx.span.set_attribute("budget_remaining", round(budget.remaining_budget, 4))

    return result


# =============================================================================
# RAG PIPELINE
# =============================================================================

def rag_query(query: str, use_query_rewriting: bool = True) -> dict:
    """
    Execute a RAG query with:
    1. Optional query rewriting
    2. Semantic search for relevant documents
    3. Context assembly
    4. Answer generation
    5. Response caching
    """

    with observe.run(
        "rag-agent",
        user_id="demo-user",
    ) as run:
        run.set_input({"query": query, "use_query_rewriting": use_query_rewriting})

        # Check cache first
        cached = cache.get(query)
        if cached:
            print(f"[CACHE HIT] Returning cached response")
            observe.emit_event("cache.hit", {"query_hash": hashlib.sha256(query.encode()).hexdigest()[:8]})
            run.set_output(cached)
            return cached

        # Step 1: Query rewriting (optional)
        search_query = query
        if use_query_rewriting:
            print(f"[1] Rewriting query...")
            rewrite_result = rewrite_query(query)
            search_query = rewrite_result["content"]
            print(f"    Original: {query}")
            print(f"    Rewritten: {search_query}")

        # Step 2: Retrieve documents
        print(f"[2] Searching documents...")
        search_results = search_documents(search_query, top_k=3)
        print(f"    Found {search_results['count']} relevant documents")

        if search_results["count"] == 0:
            result = {"answer": "No relevant documents found.", "sources": []}
            run.set_output(result)
            return result

        # Step 3: Assemble context
        context_parts = []
        sources = []
        for doc in search_results["results"]:
            context_parts.append(f"[{doc['id']}]: {doc['content']}")
            sources.append({
                "id": doc["id"],
                "source": doc["metadata"]["source"],
                "topic": doc["metadata"]["topic"],
            })

        context = "\n\n".join(context_parts)

        # Step 4: Generate answer
        print(f"[3] Generating answer...")
        answer_result = generate_answer(query, context)

        result = {
            "answer": answer_result["content"],
            "sources": sources,
            "tokens_used": budget.total_tokens,
            "cost_usd": budget.total_cost,
        }

        # Cache the result
        cache.set(query, result)
        observe.emit_event("cache.miss", {"query_hash": hashlib.sha256(query.encode()).hexdigest()[:8]})

        run.set_output(result)
        return result


def batch_queries(queries: list[str]) -> list[dict]:
    """
    Process multiple queries with budget awareness.
    Stops processing if budget is exhausted.
    """

    results = []

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}/{len(queries)}: {query}")
        print(f"Budget: ${budget.total_cost:.4f} / ${budget.config.max_cost_usd:.2f}")
        print("=" * 60)

        if budget.is_over_budget:
            print("[STOPPED] Budget exhausted, skipping remaining queries")
            results.append({
                "query": query,
                "error": "Budget exhausted",
                "skipped": True,
            })
            continue

        try:
            result = rag_query(query)
            results.append({"query": query, **result})
        except Exception as e:
            results.append({"query": query, "error": str(e)})

    return results


# =============================================================================
# RUN THE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Configure with circuit breaker for resilience
    observe.install(
        config=Config(mode=CaptureMode.FULL, env=Environment.DEV),
    )

    # Enable circuit breaker for hooks
    observe.hooks.set_circuit_breaker(CircuitBreakerConfig(
        enabled=True,
        failure_threshold=3,
        window_seconds=60,
        recovery_seconds=120,
    ))

    print("=" * 60)
    print("RAG AGENT WITH BUDGET ENFORCEMENT")
    print("=" * 60)
    print(f"Budget: ${budget.config.max_cost_usd:.2f}")
    print(f"Warning at: {budget.config.warn_at_percent * 100:.0f}%")

    # Test queries
    queries = [
        "What is RAG?",
        "How does machine learning work?",
        "What is RAG?",  # Should hit cache
        "Tell me about Python programming",
        "What are vector databases?",
    ]

    results = batch_queries(queries)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for r in results:
        status = "CACHED" if cache.get(r["query"]) else ("SKIPPED" if r.get("skipped") else "OK")
        print(f"  [{status}] {r['query'][:40]}...")

    print(f"\nBudget Usage:")
    print(f"  Total Cost: ${budget.total_cost:.4f}")
    print(f"  Total Tokens: {budget.total_tokens:,}")
    print(f"  LLM Calls: {budget.calls}")

    print(f"\nCache Stats:")
    print(f"  Hits: {cache.hits}")
    print(f"  Misses: {cache.misses}")
    print(f"  Hit Rate: {cache.hit_rate * 100:.1f}%")

    observe.sink.flush()
