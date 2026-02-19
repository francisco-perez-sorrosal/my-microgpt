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
