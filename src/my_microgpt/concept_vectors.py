"""Concept vector extraction for microgpt: contrastive activation analysis.

Extracts concept vectors by comparing the residual stream activations of
names that match a concept (e.g. "fran") against a random baseline.
The resulting vector captures the model's internal representation of the concept
and can be injected at inference time to steer generation.

Uses Value.no_grad() to run the existing gpt() forward pass without autograd
overhead — same code path as training, but without building the computation graph.
"""

import math
import random

from my_microgpt.architecture import gpt, make_kv_cache
from my_microgpt.autograd import Value
from my_microgpt.mc_helpers import ConceptVectorData, save_concept_vector
from my_microgpt.parameters import ModelParameters
from my_microgpt.tokenization import Tokenizer


def extract_activations(
    model: ModelParameters,
    token_ids: list[int],
    layer: int,
) -> list[list[float]]:
    """Run one name through the model and return residual stream vectors at each position.

    Returns a list of n_embd-dimensional vectors, one per token position.
    Uses a capture hook to intercept the residual stream at the target layer.
    """
    cfg = model.config
    keys, values = make_kv_cache(cfg)
    activations: list[list[float]] = []

    def capture_hook(x: list[Value], li: int) -> list[Value]:
        if li == layer:  # We make sure that the layer is the one we want to capture the activations from! ;-)
            activations.append([v.data for v in x])
        return x

    with Value.no_grad():
        for pos_id, token_id in enumerate(token_ids):
            gpt(token_id, pos_id, model, keys, values, post_mlp_hook=capture_hook)

    return activations


def mean_vector(vectors: list[list[float]]) -> list[float]:
    """Average a list of same-length vectors element-wise."""
    n = len(vectors)
    dim = len(vectors[0])
    return [sum(vectors[i][d] for i in range(n)) / n for d in range(dim)]


def compute_concept_vector(
    model: ModelParameters,
    tok: Tokenizer,
    positive_names: list[str],
    negative_names: list[str],
    layer: int,
) -> list[float]:
    """Compute a concept vector via contrastive activation subtraction.

    1. For each positive name, extract activations at `layer` and average across positions.
    2. Do the same for negative names.
    3. Concept vector = mean(positive) - mean(negative).
    """

    def collect_mean_activations(names: list[str]) -> list[list[float]]:
        per_name_means: list[list[float]] = []
        for name in names:
            token_ids = tok.encode(name)
            activations = extract_activations(model, token_ids, layer)
            if activations:
                per_name_means.append(mean_vector(activations))
        return per_name_means

    print(f"extracting activations for {len(positive_names)} positive names...")
    positive_means = collect_mean_activations(positive_names)
    print(f"extracting activations for {len(negative_names)} negative names...")
    negative_means = collect_mean_activations(negative_names)

    mean_pos = mean_vector(positive_means)
    mean_neg = mean_vector(negative_means)

    # Here is where Contrastive Subtraction happens: concept vector = mean(positive) - mean(negative)
    concept_vector = [p - n for p, n in zip(mean_pos, mean_neg)]
    norm = math.sqrt(sum(v * v for v in concept_vector))
    print(f"concept vector norm: {norm:.4f}")
    return concept_vector


def main() -> None:
    import argparse

    from my_microgpt.dataset import load_docs
    from my_microgpt.storage import load_model

    parser = argparse.ArgumentParser(description="Extract a concept vector from a trained microgpt model")
    parser.add_argument("--model", type=str, required=True, help="path to the trained model JSON")
    parser.add_argument("--concept", type=str, default="fran", help="substring to match for positive names (default: fran)")
    parser.add_argument("--layer", type=int, default=2, help="layer to extract activations from (default: 2, the last layer)")
    parser.add_argument("--num-negative", type=int, default=50, help="number of random negative examples (default: 50)")
    parser.add_argument("--output", type=str, default=None, help="output path for the concept vector JSON")
    args = parser.parse_args()

    model, tok, _ = load_model(args.model)

    if args.layer >= model.config.n_layer:
        parser.error(f"--layer {args.layer} is out of range: model has {model.config.n_layer} layers (valid: 0-{model.config.n_layer - 1})")

    # Build positive/negative sets from the training data
    docs = load_docs()
    concept_lower = args.concept.lower()
    positive_names = [name for name in docs if concept_lower in name.lower()]
    negative_candidates = [name for name in docs if concept_lower not in name.lower()]
    negative_names = random.sample(negative_candidates, min(args.num_negative, len(negative_candidates)))

    print(f"concept: '{args.concept}'")
    print(f"positive names ({len(positive_names)}): {positive_names}")
    print(f"negative names ({len(negative_names)}): {negative_names}")

    vector = compute_concept_vector(model, tok, positive_names, negative_names, args.layer)

    output_path = args.output or f"concept_{args.concept}_layer{args.layer}.json"
    data = ConceptVectorData(
        concept=args.concept,
        layer=args.layer,
        hook_point="post_mlp",
        alpha=1.0,
        num_positive=len(positive_names),
        num_negative=len(negative_names),
        vector=vector,
    )
    save_concept_vector(data, output_path)


if __name__ == "__main__":
    main()
