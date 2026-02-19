from my_microgpt.autograd import Value
from my_microgpt.parameters import ModelConfig, ModelParameters, make_matrix


def test_make_matrix_shape():
    mat = make_matrix(3, 4)
    assert len(mat) == 3
    assert all(len(row) == 4 for row in mat)
    assert all(isinstance(v, Value) for row in mat for v in row)


def test_model_config_defaults():
    cfg = ModelConfig()
    assert cfg.n_embd == 16
    assert cfg.n_head == 4
    assert cfg.n_layer == 1
    assert cfg.block_size == 16
    assert cfg.head_dim == 4


def test_model_parameters_param_count():
    """With vocab_size=27, default config: expect 4192 params (matching microgpt)."""
    model = ModelParameters.create(vocab_size=27)
    assert len(model) == 4192
    assert len(model.params) == 4192


def test_model_parameters_state_dict_keys():
    params = ModelParameters.create(vocab_size=10, config=ModelConfig(n_layer=2))
    keys = set(params.state_dict.keys())
    assert "wte" in keys
    assert "wpe" in keys
    assert "lm_head" in keys
    for i in range(2):
        for suffix in ["attn_wq", "attn_wk", "attn_wv", "attn_wo", "mlp_fc1", "mlp_fc2"]:
            assert f"layer{i}.{suffix}" in keys


def test_model_parameters_all_values():
    params = ModelParameters.create(vocab_size=5)
    assert all(isinstance(p, Value) for p in params.params)


def test_custom_config():
    cfg = ModelConfig(n_embd=8, n_head=2, n_layer=2, block_size=8)
    assert cfg.head_dim == 4
    params = ModelParameters.create(vocab_size=10, config=cfg)
    assert len(params.state_dict) == 3 + 6 * 2  # 3 global + 6 per layer * 2 layers
