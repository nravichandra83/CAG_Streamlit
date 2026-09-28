# Understanding the KV Cache

Every caching mechanism this project touches — Gemini's explicit `CachedContent`,
OpenAI's automatic prompt caching, Anthropic's `cache_control` breakpoints,
vLLM's prefix caching — is a product-level wrapper around the **same
underlying idea inside the transformer**: the KV cache. This doc explains
what that actually is, why it exists, and how it's implemented at the GPU
level to make serving LLMs economically viable.

## 1. Where it comes from: self-attention

At every transformer layer, for every token, the model projects that token's
embedding into three vectors:

- **Q**uery — "what am I looking for?"
- **K**ey — "what do I contain?" (used to be matched against queries)
- **V**alue — "what do I actually contribute if matched?"

Attention for a token is computed as:

```
Attention(Q, K, V) = softmax(Q · Kᵀ / √d) · V
```

To generate token *n*, the model's Query for token *n* needs to attend over
the **Key and Value vectors of every token that came before it** (tokens
`1..n-1`), plus its own. That's the mechanism that lets token 500 "see"
token 3 — it attends to token 3's K/V vectors.

## 2. The problem: naive autoregressive generation is wasteful

LLMs generate one token at a time. Naively, to generate token *n* you'd
re-run the full forward pass over tokens `1..n`, recomputing K and V for
every single earlier token *again*, even though those vectors never change
once computed (they only depend on the tokens before them, which are fixed).

For a sequence of length `n`, generating all `n` tokens naively costs
`O(n²)` worth of K/V projection work — almost all of it pure repetition.

## 3. The fix: cache K and V, don't recompute them

The **KV cache** is simply: store every token's Key and Value vectors (per
layer, per attention head) in GPU memory the first time they're computed,
and reuse them on every later step instead of recomputing.

This splits inference into two distinct phases:

| Phase | What happens | Cost profile |
|---|---|---|
| **Prefill** | The entire input prompt is processed in one parallel forward pass. K/V vectors for *every* prompt token are computed and written into the KV cache. | Compute-bound — heavy matrix multiplication, highly parallel on GPU. This is the expensive part of a long prompt. |
| **Decode** | One new token is generated at a time. Only the *new* token's Q/K/V are computed; its Query attends over the entire cached K/V history; its new K/V are appended to the cache. | Memory-bandwidth-bound — the bottleneck is reading the growing KV cache from GPU memory for every single new token, not computing it. |

This is why a long prompt with a short answer still takes a moment before
the first output token appears (prefill), but then streams token-by-token
fairly quickly (decode) — the two phases have completely different
performance characteristics.

## 4. Why this eats GPU memory

The KV cache isn't free — it has to live in GPU VRAM for the entire
lifetime of a request, growing with every generated token. Its size is:

```
KV cache size = 2 (K and V) × layers × kv_heads × head_dim × sequence_length × batch_size × bytes_per_value
```

**Worked example** (illustrative — using a known open architecture like
Llama-3-8B's dimensions, since Gemini/GPT internals aren't public):
32 layers, 8 KV heads (grouped-query attention), head_dim 128, FP16 (2 bytes):

```
per token per layer = 2 × 8 × 128 = 2,048 values
per token (all layers) = 2,048 × 32 = 65,536 values
per token in bytes = 65,536 × 2 bytes ≈ 128 KB
```

For our HR policy document (~2,400 tokens once cached), that's roughly
`2,400 × 128 KB ≈ 300 MB` of GPU memory just to hold that one document's
cached state for a single sequence — and that scales linearly with context
length and with how many concurrent sequences the server is holding. This
is exactly why long-context models are expensive to serve: the KV cache,
not the model weights, often becomes the dominant consumer of GPU memory at
scale.

## 5. GPU-side optimizations built around the KV cache

These are the actual engineering techniques inference servers use to keep
KV cache memory under control and throughput high:

- **Grouped-Query / Multi-Query Attention (GQA/MQA)** — an architecture
  choice (used in Llama 3, Mistral, and most modern models) where many Query
  heads share a smaller number of Key/Value heads. Fewer KV heads means a
  proportionally smaller KV cache per token — this is a model-design-level
  optimization, baked in before the model is ever served.
- **PagedAttention (vLLM)** — borrows the OS's virtual-memory paging idea:
  instead of allocating one large contiguous memory block per sequence
  (which fragments and wastes GPU memory), the KV cache is split into
  fixed-size blocks that can be allocated non-contiguously and shared. This
  is also *how* vLLM implements automatic prefix caching: if two requests
  share an identical token prefix, they literally point at the same
  physical KV blocks instead of duplicating them.
- **KV cache quantization** — storing K/V in INT8/FP8 instead of FP16/BF16,
  roughly halving or quartering memory footprint at a small accuracy cost.
- **Sliding-window / windowed attention** — for very long contexts, older
  KV entries outside a fixed window are evicted rather than kept forever,
  bounding memory growth at the cost of losing far-back context.
- **Cross-request prefix caching** — the piece that's directly relevant to
  this project. If many requests share an identical prefix (our HR document
  followed by a different question each time), the server can reuse the
  *same* cached K/V blocks across all of them instead of re-running prefill
  per request. This is what every vendor-level "prompt/context caching"
  feature is really doing under the hood.

## 6. How each provider exposes cross-request prefix caching

| Provider | Cache model | How you use it | Signal to the server |
|---|---|---|---|
| **Google Gemini** | Explicit, stateful object (`CachedContent`) | Create a cache once via `client.caches.create(...)`, get back a handle, pass `cached_content=<handle>` on every later call. You manage its TTL and deletion. | An API resource ID |
| **OpenAI** | Fully automatic, stateless | Nothing to call — just send the same static prefix (e.g. document in the system prompt) before the varying part every time. | Nothing — inferred from exact token-prefix match, if ≥1024 tokens and reused within the recent time window |
| **Anthropic Claude** | Explicit, lightweight annotation | No separate object — mark a `cache_control: {"type": "ephemeral"}` breakpoint on a content block (e.g. the system prompt). Resend the identical block later to hit the cache. | An annotation on the request itself, not a separate resource |
| **vLLM (self-hosted)** | Automatic, engine-level | Set `enable_prefix_caching=True` when launching the server. All requests through that server transparently share KV blocks for matching prefixes. | Nothing at the API level — it's a server flag |
| **llama.cpp (self-hosted)** | Explicit, file-based | Run once with `--prompt-cache file.bin` to persist the computed KV state to disk; later runs referencing the same file skip recomputing the shared prefix. | A local file path |

Three different *interfaces* (stateful object, silent automation, request
annotation) sitting on top of one *mechanism* (skip recomputing K/V for
tokens you've already processed before).

## 7. Why this matters for cost and latency

Skipping prefill for cached tokens means the GPU doesn't run those matrix
multiplications again. Concretely, for the provider/server, that means:

- **Lower cost per request** — less compute per call, which is why cached
  input tokens are billed at a fraction of the normal rate (or free, on some
  platforms).
- **Lower latency (time-to-first-token)** — the expensive prefill phase is
  shortened to just the new, uncached tail of the prompt.
- **Higher throughput** — GPU cycles freed from redundant prefill can serve
  other concurrent requests, which is the actual commercial reason vendors
  built these features: it lets them serve more traffic on the same
  hardware.

This is the real payoff of Cache-Augmented Generation as a pattern: load a
document into context once, and every subsequent question against it barely
touches the GPU for the document portion at all.
