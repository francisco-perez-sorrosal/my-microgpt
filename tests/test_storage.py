import json
from pathlib import Path

from my_microgpt.parameters import ModelConfig, ModelParameters
from my_microgpt.storage import load_model, model_filename, save_model
from my_microgpt.tokenization import Tokenizer
from my_microgpt.training import TrainingInfo


def _make_model_and_tok() -> tuple[ModelParameters, Tokenizer, TrainingInfo]:
    docs = ["ab", "ba", "aa", "bb"]
    tok = Tokenizer.from_docs(docs)
    cfg = ModelConfig(n_embd=4, n_head=2, n_layer=1, block_size=8)
    model = ModelParameters.create(tok.vocab_size, cfg)
    info = TrainingInfo(num_steps=100, final_loss=2.5)
    return model, tok, info


def test_save_creates_json_file(tmp_path: Path):
    model, tok, info = _make_model_and_tok()
    path = str(tmp_path / "model.json")
    save_model(model, tok, info, path=path)
    assert Path(path).exists()
    data = json.loads(Path(path).read_text())
    assert "config" in data
    assert "training" in data
    assert "chars" in data
    assert "state_dict" in data


def test_training_info_round_trip(tmp_path: Path):
    model, tok, info = _make_model_and_tok()
    path = str(tmp_path / "model.json")
    save_model(model, tok, info, path=path)
    _, _, loaded_info = load_model(path)
    assert loaded_info.num_steps == 100
    assert loaded_info.final_loss == 2.5


def test_round_trip_preserves_weights(tmp_path: Path):
    model, tok, info = _make_model_and_tok()
    path = str(tmp_path / "model.json")
    save_model(model, tok, info, path=path)
    loaded_model, loaded_tok, _ = load_model(path)

    assert loaded_model.config.n_embd == model.config.n_embd
    assert loaded_tok.chars == tok.chars
    assert len(loaded_model) == len(model)
    for key in model.keys:
        original = model.state_dict[key]
        loaded = loaded_model.state_dict[key]
        for orig_row, load_row in zip(original, loaded, strict=True):
            for orig_val, load_val in zip(orig_row, load_row, strict=True):
                assert orig_val.data == load_val.data


def test_loaded_model_has_no_grads(tmp_path: Path):
    model, tok, info = _make_model_and_tok()
    for p in model.params:
        p.grad = 1.0
    path = str(tmp_path / "model.json")
    save_model(model, tok, info, path=path)
    loaded_model, _, _ = load_model(path)
    assert all(p.grad == 0 for p in loaded_model.params)


def test_model_filename():
    docs = ["ab"]
    tok = Tokenizer.from_docs(docs)
    cfg = ModelConfig(n_embd=16, n_head=4, n_layer=1, block_size=16)
    model = ModelParameters.create(tok.vocab_size, cfg)
    info = TrainingInfo(num_steps=1000, final_loss=2.3)
    assert model_filename(model, info) == "model_e16_h4_l1_s1000.json"
