"""Training loop for microgpt: forward pass, cross-entropy loss, backpropagation, and Adam optimizer."""

from dataclasses import dataclass

from my_microgpt.architecture import gpt, make_kv_cache, softmax
from my_microgpt.autograd import Value
from my_microgpt.parameters import ModelParameters
from my_microgpt.tokenization import Tokenizer

DEFAULT_NUM_STEPS = 1000
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_BETA1 = 0.85
DEFAULT_BETA2 = 0.99
DEFAULT_EPS_ADAM = 1e-8


@dataclass
class AdamOptimizer:
    """Adam optimizer: hyperparameters and moment buffers."""

    beta1: float
    beta2: float
    eps: float
    m: list[float]  # first moment (mean of gradients)
    v: list[float]  # second moment (mean of squared gradients)

    def __len__(self) -> int:
        """Total size of the optimizer state (m + v buffers)."""
        return len(self.m) + len(self.v)

    def step(self, params: list[Value], lr: float, step: int) -> None:
        """Update parameters using Adam with bias-corrected moments."""
        for i, p in enumerate(params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad**2
            m_hat = self.m[i] / (1 - self.beta1 ** (step + 1))  # bias correction
            v_hat = self.v[i] / (1 - self.beta2 ** (step + 1))  # bias correction
            p.data -= lr * m_hat / (v_hat**0.5 + self.eps)
            p.grad = 0  # reset gradient for next step

    @staticmethod
    def create(
        num_params: int,
        beta1: float = DEFAULT_BETA1,
        beta2: float = DEFAULT_BETA2,
        eps: float = DEFAULT_EPS_ADAM,
    ) -> "AdamOptimizer":
        adam = AdamOptimizer(beta1=beta1, beta2=beta2, eps=eps, m=[0.0] * num_params, v=[0.0] * num_params)
        print(f"adam optimizer state: {len(adam):,} floats ({len(adam) // 2:,} params x 2 buffers)")
        return adam


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    num_steps: int = DEFAULT_NUM_STEPS
    learning_rate: float = DEFAULT_LEARNING_RATE


@dataclass
class TrainingInfo:
    """Metadata about how a model was trained. Saved alongside the weights."""

    num_steps: int
    final_loss: float


def train_step(
    doc: str,
    tok: Tokenizer,
    model: ModelParameters,
    adam: AdamOptimizer,
    train_cfg: TrainConfig,
    step_no: int,
) -> float:
    """Run one training step on a single document. Returns the loss value.

    1. Tokenize the document (wrapped with BOS on both sides)
    2. Forward pass: run each token through the model, compute cross-entropy loss
    3. Backward pass: backpropagate to get gradients for all parameters
    4. Adam update: adjust parameters using adaptive learning rates
    """
    cfg = model.config
    tokens = tok.encode(doc)
    n = min(cfg.block_size, len(tokens) - 1)  # -1 because we're predicting the next token

    # Forward pass: build computation graph from tokens to loss
    # Fresh KV cache per step — each document is an independent sequence.
    # gpt() mutates it in place (append), so it accumulates context across positions within the loop.
    keys, values = make_kv_cache(cfg)
    losses: list[Value] = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id=token_id, pos_id=pos_id, model=model, keys=keys, values=values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()  # cross-entropy: -log(p(correct token))
        losses.append(loss_t)

    loss = (1 / n) * sum(losses, Value(0.0))  # average loss over the document

    # Backward pass: compute gradients for all parameters
    loss.backward()

    # Adam optimizer update with linear learning rate decay
    lr_t = train_cfg.learning_rate * (1 - step_no / train_cfg.num_steps)
    adam.step(model.params, lr=lr_t, step=step_no)

    return loss.data


def train(
    dataset: list[str],
    tok: Tokenizer,
    model: ModelParameters,
    train_cfg: TrainConfig | None = None,
) -> list[float]:
    """Train the model on the dataset. Returns list of per-step losses."""
    cfg = train_cfg or TrainConfig()
    adam = AdamOptimizer.create(len(model))
    losses: list[float] = []

    for step in range(cfg.num_steps):
        doc = dataset[step % len(dataset)]
        # print(f"doc: {doc}")
        loss = train_step(doc, tok, model, adam, cfg, step)
        losses.append(loss)
        print(f"step {step + 1:4d} / {cfg.num_steps:4d} | loss {loss:.4f}")

    return losses


def main() -> None:
    from my_microgpt.dataset import load_docs
    from my_microgpt.storage import save_model

    dataset = load_docs()
    tok = Tokenizer.from_docs(dataset)
    model = ModelParameters.create(tok.vocab_size)
    print(model)

    train_cfg = TrainConfig(num_steps=1000)
    losses = train(dataset, tok, model, train_cfg)
    print(f"\nloss: {losses[0]:.4f} -> {losses[-1]:.4f}")

    # Save the trained model so inference can load it without retraining
    info = TrainingInfo(num_steps=train_cfg.num_steps, final_loss=losses[-1])
    save_model(model, tok, info)


if __name__ == "__main__":
    main()
