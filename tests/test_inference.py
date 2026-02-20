from my_microgpt.inference import InferenceConfig, generate, inference, sample_token
from my_microgpt.parameters import ModelConfig, ModelParameters
from my_microgpt.tokenization import Tokenizer
from my_microgpt.training import AdamOptimizer, TrainConfig, train_step


def _make_tiny_trained_model() -> tuple[Tokenizer, ModelParameters]:
    docs = ["ab", "ba", "aa", "bb"]
    tok = Tokenizer.from_docs(docs)
    cfg = ModelConfig(n_embd=4, n_head=2, n_layer=1, block_size=8)
    model = ModelParameters.create(tok.vocab_size, cfg)
    adam = AdamOptimizer.create(len(model))
    train_cfg = TrainConfig(num_steps=10)
    for step in range(10):
        train_step(docs[step % len(docs)], tok, model, adam, train_cfg, step_no=step)
    return tok, model


def test_generate_returns_string():
    tok, model = _make_tiny_trained_model()
    name = generate(model, tok, temperature=0.5)
    assert isinstance(name, str)


def test_generate_only_produces_valid_chars():
    tok, model = _make_tiny_trained_model()
    valid_chars = set(tok.chars)
    for _ in range(10):
        name = generate(model, tok, temperature=0.8)
        assert all(c in valid_chars for c in name)


def test_inference_returns_requested_count():
    tok, model = _make_tiny_trained_model()
    cfg = InferenceConfig(num_samples=5, temperature=0.5)
    samples = inference(model, tok, cfg)
    assert len(samples) == 5


def test_sample_token_returns_valid_index():
    probs = [0.1, 0.2, 0.3, 0.4]
    for _ in range(20):
        idx = sample_token(probs)
        assert 0 <= idx < len(probs)
