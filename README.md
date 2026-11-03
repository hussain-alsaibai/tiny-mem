# tiny-mem

> **Local-first memory for AI agents.** Zero dependencies. JSON, vector, or hybrid backends. Drop-in for tiny-agent, LangChain, or raw OpenAI.

[![PyPI](https://img.shields.io/badge/pypi-coming-blue)](#) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](#) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue)](#) [![Zero deps](https://img.shields.io/badge/deps-zero-brightgreen)](#)

```bash
pip install tiny-mem
```

## The problem

Building an AI agent with memory shouldn't mean shipping a 200MB Docker image with Qdrant, Redis, Postgres, and a sidecar for the embedding service.

It also shouldn't mean wiring up mem0, paying $0.001 per stored fact, and praying your user's PII doesn't get logged on someone else's server.

**tiny-mem is what you reach for when you need memory to *just work* locally.**

## What you get

- 🧠 **3 backends, one API**: `JSONMemory` (small/single-user), `VectorMemory` (semantic search, pure-Python cosine), `HybridMemory` (keyword + vector)
- 📦 **Zero dependencies.** Just stdlib. (NumPy optional for 10× faster vector ops — falls back to pure Python automatically.)
- 🔌 **Adapter-friendly**: works with any embedding function (OpenAI, Cohere, local, even a hash-based "no model needed" embedder)
- 💾 **SQLite-backed persistence** with WAL mode for safe concurrent reads
- ⏱️ **Time-aware recall** — filter by recency, importance, or session
- 🧬 **Scoped memory** — per-user, per-agent, per-session isolation
- 🪶 **~600 lines of code** — readable in one sitting

## Quickstart

```python
from tiny_mem import HybridMemory, OpenAIEmbedder

# Any function that takes a string and returns a List[float] works
def embed(text: str) -> list[float]:
    # your OpenAI / Cohere / local-model call here
    ...

mem = HybridMemory(embedder=embed)

# Add memories (auto-extracts key facts from a conversation)
mem.add("User's name is AlSaibai. They prefer dark mode. Building fintech in Bahrain.", user_id="u1")
mem.add("They asked about MCP servers yesterday, then switched to tiny-mem today.", user_id="u1")

# Recall — returns relevant memories ranked by vector similarity + recency
results = mem.recall("What does this user care about?", user_id="u1", limit=5)
for r in results:
    print(f"[{r.score:.2f}] {r.text}")

# > [0.91] User's name is AlSaibai. They prefer dark mode. Building fintech in Bahrain.
# > [0.78] They asked about MCP servers yesterday, then switched to tiny-mem today.
```

## Why this exists

| Tool | Lock-in | Size | Offline | Memory type |
|---|---|---|---|---|
| mem0 | Cloud-first | ~50MB+ | ❌ | Graph + vector |
| Zep | Cloud-only | 500MB+ Docker | ❌ | Graph |
| LlamaIndex | Many | ~80MB | ⚠️ | Vector + graph |
| LangChain Memory | Many | ~100MB | ⚠️ | Pluggable |
| **tiny-mem** | **None** | **<10KB** | **✅** | **JSON / Vector / Hybrid** |

## API surface

```python
# Three backends, identical API
from tiny_mem import JSONMemory, VectorMemory, HybridMemory

mem = JSONMemory()                          # keyword-only, ~50ns/op
mem = VectorMemory(embedder=embed)          # pure-Python cosine
mem = HybridMemory(embedder=embed)          # best of both

# Lifecycle
mem.add(text, user_id=..., session_id=..., metadata={...})
mem.recall(query, user_id=..., limit=5, min_score=0.5)
mem.recall_by_session(session_id)
mem.forget(user_id=...)                     # GDPR-style right-to-delete
mem.compact()                               # merge duplicates, prune old

# Scoping
mem.user("u1").add("...")                   # chained scope
mem.user("u1").session("s42").add("...")
```

## Zero-dep vector search?

Yep. Pure Python cosine similarity over `list[float]`. Slow for >10k vectors, but for <10k it's faster than disk seek. Use SQLite + NumPy for big stores.

```python
# No API key? No model? Hash-embedder works for demos and tests.
from tiny_mem import HashEmbedder

mem = HybridMemory(embedder=HashEmbedder(dim=384))
mem.add("the quick brown fox")
mem.recall("fast fox")  # → returns the fox, ranked
```

## Roadmap

- [ ] Compressed vector index (HNSW-lite, ~300 LOC, no deps)
- [ ] Lance/parquet export for cold storage
- [ ] Async API (`AsyncHybridMemory`)
- [ ] TypeScript port (`tiny-mem-js`) for edge runtimes

## License

MIT
