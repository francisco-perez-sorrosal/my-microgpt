"""Inference for microgpt: sample new names from a trained model."""

import random
from dataclasses import dataclass

from my_microgpt.architecture import gpt, make_kv_cache, softmax
from my_microgpt.parameters import ModelParameters
from my_microgpt.tokenization import Tokenizer

DEFAULT_TEMPERATURE = 0.5
DEFAULT_NUM_SAMPLES = 20


@dataclass
class InferenceConfig:
    """Inference hyperparameters."""

    temperature: float = DEFAULT_TEMPERATURE  # controls randomness: lower = conservative, higher = diverse
    num_samples: int = DEFAULT_NUM_SAMPLES  # number of samples (names) to generate


def sample_token(probs: list[float]) -> int:
    """Sample a token id from a probability distribution."""
    return random.choices(range(len(probs)), weights=probs)[0]


def generate(
    model: ModelParameters,
    tok: Tokenizer,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """Generate a single name from the model.

    Starts with BOS, feeds each generated token back as the next input,
    and stops when the model produces BOS again or hits block_size.
    Temperature controls randomness: lower values are more conservative,
    higher values produce more diverse output.
    """
    cfg = model.config
    keys, values = make_kv_cache(cfg)
    token_id = tok.bos
    chars: list[str] = []

    for pos_id in range(cfg.block_size):
        logits = gpt(token_id=token_id, pos_id=pos_id, model=model, keys=keys, values=values)
        # Divide logits by temperature before softmax to control randomness
        probs = softmax([logit / temperature for logit in logits])
        token_id = sample_token([p.data for p in probs])
        if token_id == tok.bos:  # BOS again means "I'm done"
            break
        chars.append(tok.id_to_char[token_id])

    return "".join(chars)


def inference(
    model: ModelParameters,
    tok: Tokenizer,
    inference_cfg: InferenceConfig | None = None,
) -> list[str]:
    """Generate multiple names from the model. Returns list of generated names."""
    cfg = inference_cfg or InferenceConfig()
    samples: list[str] = []

    print(f"\n--- inference (temperature={cfg.temperature}) ---")
    for i in range(cfg.num_samples):
        name = generate(model, tok, temperature=cfg.temperature)
        samples.append(name)
        print(f"sample {i + 1:2d}: {name}")

    return samples


def main() -> None:
    import sys

    from my_microgpt.storage import load_model

    # Load a previously trained model — no dataset or training needed
    path = sys.argv[1] if len(sys.argv) > 1 else "model.json"
    model, tok, _info = load_model(path)
    print(f"Model loaded from {path}. Info: {_info}")
    inference(model, tok)


if __name__ == "__main__":
    main()
