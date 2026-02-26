"""Inference for microgpt: sample new names from a trained model."""

import random
from collections.abc import Callable
from dataclasses import dataclass

from my_microgpt.architecture import gpt, make_kv_cache, softmax
from my_microgpt.autograd import Value
from my_microgpt.parameters import ModelParameters
from my_microgpt.tokenization import Tokenizer

DEFAULT_TEMPERATURE = 0.5
DEFAULT_NUM_SAMPLES = 3781


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
    post_mlp_hook: Callable[[list[Value], int], list[Value]] | None = None,
) -> str:
    """Generate a single name from the model.

    Starts with BOS, feeds each generated token back as the next input,
    and stops when the model produces BOS again or hits block_size.
    Temperature controls randomness: lower values are more conservative,
    higher values produce more diverse output.

    The optional post_mlp_hook is passed through to gpt() — used by steered
    inference to inject concept vectors into the residual stream.
    """
    cfg = model.config
    keys, values = make_kv_cache(cfg)
    token_id = tok.bos
    chars: list[str] = []

    for pos_id in range(cfg.block_size):
        logits = gpt(token_id=token_id, pos_id=pos_id, model=model, keys=keys, values=values, post_mlp_hook=post_mlp_hook)
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
    import argparse

    from my_microgpt.storage import load_model

    parser = argparse.ArgumentParser(description="Generate names from a trained microgpt model")
    parser.add_argument("--model", type=str, default="model.json", help="path to the trained model JSON (default: model.json)")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="sampling temperature")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES, help="number of names to generate (default: 3781)")
    parser.add_argument("--concept", type=str, default="fran", help="substring to count in generated names (default: fran)")
    args = parser.parse_args()

    model, tok, _info = load_model(args.model)
    print(f"Model loaded from {args.model}. Info: {_info}")
    samples = inference(model, tok, InferenceConfig(temperature=args.temperature, num_samples=args.num_samples))

    concept = args.concept.lower()
    matches = sum(1 for s in samples if concept in s.lower())
    print(f"\n-> {matches}/{len(samples)} contain '{args.concept}' ({matches / len(samples) * 100:.1f}%)")


if __name__ == "__main__":
    main()
