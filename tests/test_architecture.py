import math

from my_microgpt.architecture import gpt, linear, make_kv_cache, rmsnorm, softmax
from my_microgpt.autograd import Value
from my_microgpt.parameters import ModelConfig, ModelParameters


def test_linear():
    x = [Value(1.0), Value(2.0)]
    w = [[Value(3.0), Value(4.0)], [Value(5.0), Value(6.0)]]
    result = linear(x, w)
    assert len(result) == 2
    assert result[0].data == 11.0  # 3*1 + 4*2
    assert result[1].data == 17.0  # 5*1 + 6*2


def test_softmax_sums_to_one():
    logits = [Value(1.0), Value(2.0), Value(3.0)]
    probs = softmax(logits)
    assert len(probs) == 3
    assert abs(sum(p.data for p in probs) - 1.0) < 1e-6


def test_softmax_ordering():
    logits = [Value(1.0), Value(3.0), Value(2.0)]
    probs = softmax(logits)
    assert probs[1].data > probs[2].data > probs[0].data


def test_rmsnorm_unit_rms():
    x = [Value(3.0), Value(4.0)]
    normed = rmsnorm(x)
    rms = math.sqrt(sum(v.data**2 for v in normed) / len(normed))
    assert abs(rms - 1.0) < 1e-3


def test_gpt_output_shape():
    """Forward pass produces one logit per vocabulary token."""
    vocab_size = 5
    cfg = ModelConfig(n_embd=4, n_head=2, n_layer=1, block_size=4)
    model = ModelParameters.create(vocab_size, cfg)
    keys, vals = make_kv_cache(cfg)
    logits = gpt(0, 0, model, keys, vals)
    assert len(logits) == vocab_size


def test_gpt_kv_cache_grows():
    """Each forward pass appends one entry to the KV cache per layer."""
    cfg = ModelConfig(n_embd=4, n_head=2, n_layer=2, block_size=4)
    model = ModelParameters.create(5, cfg)
    keys, vals = make_kv_cache(cfg)
    for pos in range(3):
        gpt(0, pos, model, keys, vals)
    for li in range(cfg.n_layer):
        assert len(keys[li]) == 3
        assert len(vals[li]) == 3


def test_gpt_logits_are_differentiable():
    """Logits are Value nodes connected to the computation graph."""
    cfg = ModelConfig(n_embd=4, n_head=2, n_layer=1, block_size=4)
    model = ModelParameters.create(5, cfg)
    keys, vals = make_kv_cache(cfg)
    logits = gpt(0, 0, model, keys, vals)
    loss = sum(logits)
    loss.backward()
    # At least some parameters should have non-zero gradients
    grads = [p.grad for p in model.params]
    assert any(g != 0.0 for g in grads)


# --- post_mlp_hook ---


def test_gpt_no_hook_matches_default():
    """Passing post_mlp_hook=None produces the same logits as omitting it."""
    cfg = ModelConfig(n_embd=4, n_head=2, n_layer=2, block_size=4)
    model = ModelParameters.create(5, cfg)
    keys1, vals1 = make_kv_cache(cfg)
    keys2, vals2 = make_kv_cache(cfg)
    logits_default = gpt(0, 0, model, keys1, vals1)
    logits_none = gpt(0, 0, model, keys2, vals2, post_mlp_hook=None)
    for a, b in zip(logits_default, logits_none):
        assert a.data == b.data


def test_hook_fires_at_every_layer():
    """The hook is called once per layer with the correct layer index."""
    cfg = ModelConfig(n_embd=4, n_head=2, n_layer=3, block_size=4)
    model = ModelParameters.create(5, cfg)
    keys, vals = make_kv_cache(cfg)
    fired_layers: list[int] = []

    def record_hook(x: list[Value], li: int) -> list[Value]:
        fired_layers.append(li)
        return x

    gpt(0, 0, model, keys, vals, post_mlp_hook=record_hook)
    assert fired_layers == [0, 1, 2]


def test_hook_can_capture_residual_stream():
    """A capture hook reads the residual stream without modifying logits."""
    cfg = ModelConfig(n_embd=4, n_head=2, n_layer=2, block_size=4)
    model = ModelParameters.create(5, cfg)
    captured: list[list[float]] = []

    def capture_hook(x: list[Value], li: int) -> list[Value]:
        if li == 1:  # capture at last layer
            captured.append([v.data for v in x])
        return x  # pass-through

    keys1, vals1 = make_kv_cache(cfg)
    logits_hooked = gpt(0, 0, model, keys1, vals1, post_mlp_hook=capture_hook)

    keys2, vals2 = make_kv_cache(cfg)
    logits_plain = gpt(0, 0, model, keys2, vals2)

    # Captured one vector of the right dimension
    assert len(captured) == 1
    assert len(captured[0]) == cfg.n_embd

    # Pass-through hook doesn't change logits
    for a, b in zip(logits_hooked, logits_plain):
        assert a.data == b.data


def test_hook_can_modify_residual_stream():
    """An injection hook changes the logits."""
    cfg = ModelConfig(n_embd=4, n_head=2, n_layer=2, block_size=4)
    model = ModelParameters.create(5, cfg)

    def inject_hook(x: list[Value], li: int) -> list[Value]:
        if li == 1:
            return [xi + 10.0 for xi in x]  # large perturbation
        return x

    keys1, vals1 = make_kv_cache(cfg)
    logits_plain = gpt(0, 0, model, keys1, vals1)

    keys2, vals2 = make_kv_cache(cfg)
    logits_steered = gpt(0, 0, model, keys2, vals2, post_mlp_hook=inject_hook)

    # Logits should differ after injection
    diffs = [abs(a.data - b.data) for a, b in zip(logits_plain, logits_steered)]
    assert max(diffs) > 0.01


def test_no_grad_with_hook_produces_same_logits():
    """Value.no_grad() + hook produces the same .data as normal gpt() + hook."""
    cfg = ModelConfig(n_embd=4, n_head=2, n_layer=2, block_size=4)
    model = ModelParameters.create(5, cfg)

    def passthrough(x: list[Value], li: int) -> list[Value]:
        return x

    keys1, vals1 = make_kv_cache(cfg)
    logits_grad = gpt(0, 0, model, keys1, vals1, post_mlp_hook=passthrough)

    keys2, vals2 = make_kv_cache(cfg)
    with Value.no_grad():
        logits_nograd = gpt(0, 0, model, keys2, vals2, post_mlp_hook=passthrough)

    for a, b in zip(logits_grad, logits_nograd):
        assert abs(a.data - b.data) < 1e-10
