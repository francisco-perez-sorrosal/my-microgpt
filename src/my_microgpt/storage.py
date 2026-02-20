"""Model persistence for microgpt: save and load a trained model as JSON.

A saved model is a single JSON file containing everything needed for inference:
- config: the model architecture hyperparameters
- training: metadata about how the model was trained (steps, final loss)
- chars: the tokenizer's character set (from which all mappings are derived)
- state_dict: all weight matrices as 2D arrays of floats

Only the raw float values (Value.data) are saved — gradients and the
autograd computation graph are transient training state, not needed at inference.

Filename convention: model_e{n_embd}_h{n_head}_l{n_layer}_s{num_steps}.json
so different training runs are distinguishable at a glance.
"""

import json
from pathlib import Path

from my_microgpt.autograd import Value
from my_microgpt.parameters import ModelConfig, ModelParameters
from my_microgpt.tokenization import Tokenizer
from my_microgpt.training import TrainingInfo


def model_filename(model: ModelParameters, training_info: TrainingInfo) -> str:
    """Generate a descriptive filename from config and training metadata."""
    cfg = model.config
    return f"model_e{cfg.n_embd}_h{cfg.n_head}_l{cfg.n_layer}_s{training_info.num_steps}.json"


def save_model(
    model: ModelParameters,
    tok: Tokenizer,
    training_info: TrainingInfo,
    path: str | None = None,
) -> str:
    """Save a trained model, tokenizer, and training metadata to a JSON file.

    If no path is given, generates one from the model config and training info.
    Returns the path where the model was saved.
    """
    filepath = path or model_filename(model, training_info)
    data = {
        "config": {
            "n_embd": model.config.n_embd,
            "n_head": model.config.n_head,
            "n_layer": model.config.n_layer,
            "block_size": model.config.block_size,
        },
        "training": {
            "num_steps": training_info.num_steps,
            "final_loss": training_info.final_loss,
        },
        "chars": list(tok.chars),
        # Extract .data from each Value — discard grad and autograd graph
        "state_dict": {
            name: [[v.data for v in row] for row in matrix]
            for name, matrix in model.state_dict.items()
        },
    }
    Path(filepath).write_text(json.dumps(data))
    print(f"model saved to {filepath} ({Path(filepath).stat().st_size:,} bytes)")
    return filepath


def load_model(path: str) -> tuple[ModelParameters, Tokenizer, TrainingInfo]:
    """Load a model, tokenizer, and training metadata from a JSON file.

    Returns a (ModelParameters, Tokenizer, TrainingInfo) tuple ready for inference.
    The Value objects are created with the saved weights but no autograd graph,
    which is exactly what inference needs.
    """
    data = json.loads(Path(path).read_text())

    # Reconstruct tokenizer from the saved character set
    tok = Tokenizer.from_docs(["".join(data["chars"])])

    # Reconstruct config
    config = ModelConfig(**data["config"])

    # Reconstruct state_dict: wrap each float back into a Value
    state_dict = {
        name: [[Value(v) for v in row] for row in matrix]
        for name, matrix in data["state_dict"].items()
    }

    model = ModelParameters(config=config, state_dict=state_dict)

    # Reconstruct training metadata
    training_info = TrainingInfo(**data["training"])

    steps, loss = training_info.num_steps, training_info.final_loss
    print(f"model loaded from {path} ({len(model):,} params, {steps} steps, loss={loss:.4f})")
    return model, tok, training_info
