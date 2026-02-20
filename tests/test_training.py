from my_microgpt.parameters import ModelConfig, ModelParameters
from my_microgpt.tokenization import Tokenizer
from my_microgpt.training import AdamOptimizer, TrainConfig, train_step


def _make_tiny_model() -> tuple[list[str], Tokenizer, ModelParameters]:
    docs = ["ab", "ba", "aa", "bb"]
    tok = Tokenizer.from_docs(docs)
    cfg = ModelConfig(n_embd=4, n_head=2, n_layer=1, block_size=8)
    model = ModelParameters.create(tok.vocab_size, cfg)
    return docs, tok, model


def test_train_step_returns_loss():
    docs, tok, model = _make_tiny_model()
    adam = AdamOptimizer.create(len(model))
    train_cfg = TrainConfig(num_steps=10)
    loss = train_step(docs[0], tok, model, adam, train_cfg, step_no=0)
    assert isinstance(loss, float)
    assert loss > 0


def test_loss_decreases_over_steps():
    docs, tok, model = _make_tiny_model()
    adam = AdamOptimizer.create(len(model))
    train_cfg = TrainConfig(num_steps=20, learning_rate=0.01)
    losses = []
    for step in range(20):
        loss = train_step(docs[step % len(docs)], tok, model, adam, train_cfg, step)
        losses.append(loss)
    # Average of first 5 should be higher than average of last 5
    assert sum(losses[:5]) / 5 > sum(losses[-5:]) / 5


def test_gradients_reset_after_step():
    docs, tok, model = _make_tiny_model()
    adam = AdamOptimizer.create(len(model))
    train_cfg = TrainConfig(num_steps=10)
    train_step(docs[0], tok, model, adam, train_cfg, step_no=0)
    assert all(p.grad == 0 for p in model.params)


def test_adam_state_updated():
    docs, tok, model = _make_tiny_model()
    adam = AdamOptimizer.create(len(model))
    train_cfg = TrainConfig(num_steps=10)
    train_step(docs[0], tok, model, adam, train_cfg, step_no=0)
    # At least some moment buffers should be non-zero after a step
    assert any(m != 0.0 for m in adam.m)
    assert any(v != 0.0 for v in adam.v)
