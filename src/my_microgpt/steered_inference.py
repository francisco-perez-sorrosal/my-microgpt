"""Steered inference for microgpt: inject concept vectors to steer name generation.

Builds on the existing generate() function by passing a post-MLP hook that
injects the concept vector into the residual stream. Value.no_grad() skips
graph construction since steered inference never needs gradients.
"""

from dataclasses import dataclass

from my_microgpt.autograd import Value
from my_microgpt.mc_helpers import ConceptVectorData, load_concept_vector
from my_microgpt.inference import generate
from my_microgpt.parameters import ModelParameters
from my_microgpt.tokenization import Tokenizer

DEFAULT_TEMPERATURE = 0.5
DEFAULT_NUM_SAMPLES = 3781


@dataclass(frozen=True)
class SteeredInferenceConfig:
    """Hyperparameters for steered generation."""

    alphas: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
    temperature: float = DEFAULT_TEMPERATURE
    num_samples: int = DEFAULT_NUM_SAMPLES


def steered_inference(
    model: ModelParameters,
    tok: Tokenizer,
    cv_data: ConceptVectorData,
    inference_cfg: SteeredInferenceConfig | None = None,
) -> dict[float, list[str]]:
    """Generate names at multiple alpha values. Returns {alpha: [names]}."""
    cfg_inf = inference_cfg or SteeredInferenceConfig()
    results: dict[float, list[str]] = {}

    for alpha in cfg_inf.alphas:
        samples: list[str] = []
        print(f"\n--- alpha={alpha:.1f} (temperature={cfg_inf.temperature}) ---")

        def inject_hook(x: list[Value], li: int) -> list[Value]:
            if li == cv_data.layer:
                return [xi + alpha * ci for xi, ci in zip(x, cv_data.vector)]
            return x

        with Value.no_grad():
            for i in range(cfg_inf.num_samples):
                name = generate(model, tok, cfg_inf.temperature, post_mlp_hook=inject_hook)
                samples.append(name)
                print(f"  sample {i + 1:2d}: {name}")

        results[alpha] = samples

        # Count concept matches
        concept = cv_data.concept.lower()
        matches = sum(1 for s in samples if concept in s.lower())
        print(f"  -> {matches}/{len(samples)} contain '{cv_data.concept}'")

    return results


def main() -> None:
    import argparse

    from my_microgpt.storage import load_model

    parser = argparse.ArgumentParser(description="Generate names with concept vector steering")
    parser.add_argument("--model", type=str, required=True, help="path to the trained model JSON")
    parser.add_argument("--concept-vector", type=str, required=True, help="path to the concept vector JSON")
    parser.add_argument(
        "--alpha", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 5.0],
        help="steering strengths to try (default: 0.0 0.5 1.0 2.0 5.0)",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="sampling temperature")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES, help="names per alpha value")
    args = parser.parse_args()

    model, tok, _ = load_model(args.model)
    cv_data = load_concept_vector(args.concept_vector)

    inference_cfg = SteeredInferenceConfig(
        alphas=tuple(args.alpha),
        temperature=args.temperature,
        num_samples=args.num_samples,
    )
    steered_inference(model, tok, cv_data, inference_cfg)


if __name__ == "__main__":
    main()
