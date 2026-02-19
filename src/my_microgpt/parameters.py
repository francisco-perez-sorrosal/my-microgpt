"""Model parameters for microgpt: initialization and organization of all trainable weights."""

import random
from dataclasses import dataclass

from my_microgpt.autograd import Value

DEFAULT_N_EMBD = 16
DEFAULT_N_HEAD = 4
DEFAULT_N_LAYER = 1
DEFAULT_BLOCK_SIZE = 16  # maximum sequence length
DEFAULT_INIT_STD = 0.08

Matrix = list[list[Value]]


def make_matrix(nout: int, nin: int, std: float = DEFAULT_INIT_STD) -> Matrix:
    """Create a matrix of Value objects initialized from a Gaussian distribution."""
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters that define the model architecture."""

    n_embd: int = DEFAULT_N_EMBD  # embedding dimension
    n_head: int = DEFAULT_N_HEAD  # number of attention heads
    n_layer: int = DEFAULT_N_LAYER  # number of layers
    block_size: int = DEFAULT_BLOCK_SIZE  # maximum sequence length
    init_std: float = DEFAULT_INIT_STD  # standard deviation for weight initialization

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head  # Dimension of each head. Dimension: n_embd // n_head


@dataclass
class ModelParameters:
    """All trainable parameters organized in a state dict (borrowing PyTorch's terminology)."""

    config: ModelConfig
    state_dict: dict[str, Matrix]

    @property
    def params(self) -> list[Value]:
        """Flat list of all trainable parameters across all matrices."""
        return [p for mat in self.state_dict.values() for row in mat for p in row]

    @property
    def keys(self) -> list[str]:
        """Names of all parameter matrices in the state dict."""
        return list(self.state_dict.keys())

    def __len__(self) -> int:
        return len(self.params)

    def __repr__(self) -> str:
        cfg = self.config
        keys_str = "\n".join(f"    {k:20s} {len(self.state_dict[k])}x{len(self.state_dict[k][0])}" for k in self.keys)
        return (
            f"ModelParameters({len(self):,} params)\n"
            f"  config: n_embd={cfg.n_embd}, n_head={cfg.n_head}, n_layer={cfg.n_layer}, "
            f"block_size={cfg.block_size}, head_dim={cfg.head_dim}\n"
            f"  layers:\n{keys_str}"
        )

    @staticmethod
    def create(vocab_size: int, config: ModelConfig | None = None) -> "ModelParameters":
        cfg = config or ModelConfig()
        std = cfg.init_std
        sd: dict[str, Matrix] = {
            "wte": make_matrix(vocab_size, cfg.n_embd, std), # Embedding Table dimension: vocab_size x n_embd
            "wpe": make_matrix(cfg.block_size, cfg.n_embd, std), # Positional Encoding dimension: block_size x n_embd
            "lm_head": make_matrix(vocab_size, cfg.n_embd, std), # Output Projection dimension: vocab_size x n_embd (final linear layer)
        }
        for i in range(cfg.n_layer):
            sd[f"layer{i}.attn_wq"] = make_matrix(cfg.n_embd, cfg.n_embd, std)  # Query Projection dimension: n_embd x n_embd 
            sd[f"layer{i}.attn_wk"] = make_matrix(cfg.n_embd, cfg.n_embd, std)  # Key Projection dimension: n_embd x n_embd
            sd[f"layer{i}.attn_wv"] = make_matrix(cfg.n_embd, cfg.n_embd, std)  # Value Projection dimension: n_embd x n_embd
            sd[f"layer{i}.attn_wo"] = make_matrix(cfg.n_embd, cfg.n_embd, std)  # Output Projection dimension: n_embd x n_embd
            sd[f"layer{i}.mlp_fc1"] = make_matrix(4 * cfg.n_embd, cfg.n_embd, std)  # First MLP Projection dimension: 4 * n_embd x n_embd
            sd[f"layer{i}.mlp_fc2"] = make_matrix(cfg.n_embd, 4 * cfg.n_embd, std)  # Second MLP Projection dimension: n_embd x 4 * n_embd
        return ModelParameters(config=cfg, state_dict=sd)


def main() -> None:
    from my_microgpt.dataset import load_docs
    from my_microgpt.tokenization import Tokenizer

    docs = load_docs()
    tok = Tokenizer.from_docs(docs)
    cfg = None  # ModelConfig(n_layer=2)
    model_params = ModelParameters.create(tok.vocab_size, cfg)
    print(model_params)


if __name__ == "__main__":
    main()
