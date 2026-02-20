"""GPT-2-like architecture for microgpt: stateless forward pass with KV cache."""

from my_microgpt.autograd import Value
from my_microgpt.parameters import Matrix, ModelConfig, ModelParameters

# Fully populated shape: [layer][time_step][embd_dim]
# Starts as [layer][] (empty inner lists), grows to 3 levels as gpt() appends K/V vectors
KVCache = list[list[list[Value]]]


def linear(x: list[Value], w: Matrix) -> list[Value]:
    """Matrix-vector multiply: one dot product per row of w."""
    return [sum((wi * xi for wi, xi in zip(wo, x)), Value(0.0)) for wo in w]  # We need to initialize the sum with a Value(0.0) to avoid type errors.
    # This is equivalent to the non-list comprehension version:
    # results: list[Value] = []
    # for wo in w:
    #     result = Value(0.0)
    #     for wi, xi in zip(wo, x):
    #         result += wi * xi
    #     results.append(result)
    # return results


def softmax(logits: list[Value]) -> list[Value]:
    """Convert raw scores into a probability distribution."""
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x: list[Value]) -> list[Value]:
    """Root Mean Square Normalization: rescale to unit RMS."""
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def gpt(
    token_id: int,
    pos_id: int,
    model: ModelParameters,
    keys: KVCache,
    values: KVCache,
) -> list[Value]:
    """Forward pass for a single token. Returns logits over the vocabulary.

    Processes one token at a time, so no causal mask is needed: the KV cache only
    contains positions 0..t (current and past). Future positions don't exist in the
    cache yet, making causality implicit rather than requiring an explicit mask.

    NOTE: keys and values are mutated in place — each call appends the current
    token's K/V vectors to the cache. Callers must pass the same cache objects
    across sequential calls to maintain context between positions.
    """
    sd = model.state_dict
    cfg = model.config

    # Embeddings: token + position
    tok_emb = sd["wte"][token_id]  # token embedding (vocab_size x n_embd)
    pos_emb = sd["wpe"][pos_id]  # position embedding (block_size x n_embd)
    x = [t + p for t, p in zip(tok_emb, pos_emb)]  # token + position embeddings (n_embd)
    x = rmsnorm(x)

    for li in range(cfg.n_layer):
        # 1) Multi-head attention block
        x_residual = x  # residual connection
        x = rmsnorm(x)
        q = linear(x, sd[f"layer{li}.attn_wq"])  # query projection (n_embd x n_embd)
        k = linear(x, sd[f"layer{li}.attn_wk"])  # key projection (n_embd x n_embd)
        v = linear(x, sd[f"layer{li}.attn_wv"])  # value projection (n_embd x n_embd)
        # Mutate KV cache in place: append current token's K/V so future positions can attend to it
        keys[li].append(k)   # keys[li] grows: [] -> [k0] -> [k0, k1] -> ...
        values[li].append(v)  # values[li] grows in parallel
        # Concatenation of all head outputs (We're gonna chunk them below and fill this list in every loop). Dimension: n_embd
        x_attn: list[Value] = []
        for h in range(cfg.n_head):  # Iterate over all attention heads
            hs = h * cfg.head_dim  # Head start index. Dimension: head_dim
            # Slice the query, keys, and values for the current head. Dimension: head_dim
            q_h = q[hs : hs + cfg.head_dim]  # Query for the current head. Dimension: head_dim
            k_h = [ki[hs : hs + cfg.head_dim] for ki in keys[li]]  # Keys for the current head. Dimension: head_dim
            v_h = [vi[hs : hs + cfg.head_dim] for vi in values[li]]  # Values for the current head. Dimension: head_dim
            # Scaled dot-product attention: Q @ K^T / sqrt(d_k) on head-sized chunks
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(cfg.head_dim)) / cfg.head_dim**0.5
                for t in range(len(k_h))
            ]  # One score per cached position. Length: num_cached_positions (not head_dim)
            attn_weights = softmax(attn_logits)  # Attention weights. Dimension: len(k_h)
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(cfg.head_dim)
            ]  # Weighted sum of cached values. Dimension: cfg.head_dim
            x_attn.extend(head_out)  # Concatenation of all head outputs. Dimension: n_embd
        x = linear(x_attn, sd[f"layer{li}.attn_wo"])  # Output Projection. Dimension: n_embd x n_embd
        x = [a + b for a, b in zip(x, x_residual)]  # Residual connection. Dimension: n_embd

        # 2) MLP block
        x_residual = x  # residual connection. Dimension: n_embd
        x = rmsnorm(x)  # Root Mean Square Normalization. Dimension: n_embd
        x = linear(x, sd[f"layer{li}.mlp_fc1"])  # First MLP Projection. Dimension: 4 * n_embd x n_embd
        x = [xi.relu() for xi in x]
        x = linear(x, sd[f"layer{li}.mlp_fc2"])  # Second MLP Projection. Dimension: n_embd x 4 * n_embd
        x = [a + b for a, b in zip(x, x_residual)]  # Residual connection. Dimension: n_embd

    return linear(x, sd["lm_head"])  # Output Projection. Dimension: vocab_size x n_embd


def make_kv_cache(config: ModelConfig) -> tuple[KVCache, KVCache]:
    """Create empty KV caches for all layers.

    Returns mutable lists that gpt() appends to on each forward pass.
    Pass the same objects across sequential calls to accumulate context.
    """
    keys: KVCache = [[] for _ in range(config.n_layer)]
    values: KVCache = [[] for _ in range(config.n_layer)]
    return keys, values


def main() -> None:
    from my_microgpt.dataset import load_docs
    from my_microgpt.tokenization import Tokenizer

    docs = load_docs()
    tok = Tokenizer.from_docs(docs)
    model = ModelParameters.create(tok.vocab_size)
    keys, vals = make_kv_cache(model.config)
    print(keys)
    print(vals)

    # Run a single forward pass: BOS token (id=26) at the first sequence position (pos=0)
    first_position = 0
    logits = gpt(token_id=tok.bos, pos_id=first_position, model=model, keys=keys, values=vals)
    probs = softmax(logits)
    print(f"vocab_size: {tok.vocab_size}")
    print(f"logits length: {len(logits)}")
    print(f"sum of probs: {sum(p.data for p in probs):.6f}")
    top_ids = sorted(range(len(probs)), key=lambda i: probs[i].data, reverse=True)[:5]
    print(f"top 5 predictions: {top_ids}")
    print("top 5 predictions:")
    for rank, idx in enumerate(top_ids, 1):
        char = tok.id_to_char.get(idx, "<BOS>")
        print(f"  {rank}. '{char}' (id={idx}, prob={probs[idx].data:.4f})")


if __name__ == "__main__":
    main()
