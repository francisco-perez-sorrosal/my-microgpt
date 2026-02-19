# my-microgpt

Playing with AK's [micro gpt-based architecture](https://karpathy.github.io/2026/02/12/microgpt/).

## About microgpt

A single file of ~200 lines of pure Python by Andrej Karpathy with no dependencies that trains and runs inference on a GPT. The file contains the full algorithmic content: dataset of documents, tokenizer, autograd engine, a GPT-2-like neural network architecture, the Adam optimizer, training loop, and inference loop. Everything else is just efficiency.

It is dissected here for educational purposes.

## Table of Contents

1. [Dataset](#1-dataset)
2. [Tokenizer](#2-tokenizer)
3. [Autograd](#3-autograd-the-hardcore-section-that-enables-backprogagation-or-implementing-what-pytorch-gives-you-for-free)
4. [Parameters](#4-parameters-neurons)

## 1. Dataset

The input for LLMs is usually a stream of text data, which can represent a document or a set of documents. Each document can be a web page; here, if we don't provide anything else will be a 32K names, one per line downloaded from a URL:

Each name will represent a document:

```text
emma
olivia
ava
isabella
sophia
charlotte
mia
amelia
harper
... (~32,000 names)
```

The model will learn patterns found in the data and will then generate similar new documents sharing statistical patterns within. By the end of the training script, the model will generate ("hallucinate"?) new, plausible-sounding names:

```text
sample  1: kamon
sample  2: ann
sample  3: karai
sample  4: jaire
sample  5: vialan
sample  6: karia
sample  7: yeran
sample  8: anna
sample  9: areli
sample 10: kaina
```

From ChatGPT's PoV, a conversation is just a differently structured document. When a document is initialized with a particular prompt, the model's response is a statistical document completion.

### Run dataset

Run the [`dataset`](src/my_microgpt/dataset.py) module:

```bash
uv run dataset
```

## 2. Tokenizer

NN works with floating-point numbers, not chars; A tokenizer converts initially text into a sequence of integer ids per token (a char in our case) and back. Production-grade tokenizers (e.g. tiktoken by OpenAI) operate on chunks of characters for efficiency. Here, for simplicity, our DB assigns an integer to each unique character in the dataset.

Procedure:

- We collect all the unique chars in the dataset; this will form our vocabulary
- We sort the vocabulary
- We assign each letter an id by its index
- We add one special token: BOS (Begining of Sequence) to delimit documents

At training time, each doc will get wrapped with BOS on both sides. Example:

`[BOS, f, r, a, n, c, i, s, c, o, BOS]`

As training continues, the model will learn the pattern "a BOS token initiates a new sequence of letters; when we encounter another BOS ends the sequence".

The final vocabulary has 27 tokens (26 lowercase characters a-z + 1 BOS token). The BOS won't have representation in the chars tuple, but we'll use it as an index in the encoding/decoding process.

### Run tokenization

Run the [`tokenization`](src/my_microgpt/tokenization.py) module:

```bash
uv run tokenization
```

## 3. Autograd: The `Hardcore` Section that Enables Backprogagation (or implementing what Pytorch gives you for free)

Derivatives is everything here. Training NNs requires gradients; that is, if each parameter in the model is nudged up or down, how this affects the loss (up or down) and by how much?. The model can be seen as a computation graph that has many inputs (the model parameters and input tokens) but in the end funnels down to a single scalar output: the loss.

Backpropagation starts at that output and works backwards through the graph, computing the gradient of the loss with respect to every input using the chain rule from calculus. As the title mentions, libraries like PyTorch, or JaX handle gradient calulation automatically. Here it is implemented from scratch in a single class called `Value`.

`Value` contains:

1. A single scalar number (`.data`) and tracks how it is computed. If we see each operation as a building block: it takes inputs, produces an output (forward pass).
2. The block also knows how its output changes with respect to each input (the gradient).
3. When an operation is performed with a `Value` (e.g. `multiply`), the result is a new `Value` object that remembers its inputs (`_children`) and the local derivative of that operation (`_local_grads`).

The supported operations and their local gradients:

| Operation  | Forward  | Local gradients        |
| ---------- | -------- | ---------------------- |
| `a + b`    | a + b    | d/da=1, d/db=1         |
| `a * b`    | a * b    | d/da=b, d/db=a         |
| `a ** n`   | a^n      | d/da=n\*a^(n-1)        |
| `log(a)`   | ln(a)    | d/da=1/a               |
| `exp(a)`   | e^a      | d/da=e^a               |
| `relu(a)`  | max(0,a) | d/da=1 if a>0, else 0  |

### Chain-Rule Intuition

If a car travels twice as fast as a bicycle and the bicycle is four times as fast as a walking man, then the car travels 2 x 4 = 8 times as fast as the man. The chain rule is the same idea: you multiply the rates of change along the path.

So, the `backward()` method walks the graph from loss back to parameters (that is, in reverse topological order), applying the chain rule at each step. If `L` is the loss, `v` has a child `c` with local gradient `dv/dc`, the key accumulation formula is:

```text
dL/dc += (dv/dc) * (dL/dv)
```

Accumulation (`+=` rather than `=`) matters: when a value is used in multiple places in the graph, gradients flow back along each branch independently and must be added. This is a consequence of the multivariable chain rule: if `c` contributes to `L` through multiple paths, the total derivative is the sum of contributions from each path.

### Example

Below, `a` is used twice, so its gradient accumulates from both paths:

```python
a = Value(2.0)
b = Value(3.0)
c = a * b       # c = 6.0
L = c + a       # L = 8.0
L.backward()
print(a.grad)   # 4.0 (dL/da = b + 1 = 3 + 1, via both paths)
print(b.grad)   # 2.0 (dL/db = a = 2)
```

This is conceptually the identical algorithm that PyTorch's `loss.backward()` uses. But, instead of operating on scalars, Pytorch operates on tensors.

The gradients encode the direction (sign) and steepness (magnitude) of each parameter's influence on the loss, enabling iterative parameter updates that lower the loss and improve predictions.

So, as shown above, if `L = a*b + a`, being `a=2` and `b=3`, then `a.grad = 4.0` tells us about the local influence of `a` on `L`. If we wiggle the input `a`, in what direction is `L` changing? Here, the derivative of `L` w.r.t. `a` is `4.0`, meaning that if we increase `a` by a tiny amount (say `0.001`), `L` would increase by about **4x** that (`0.004`). Similarly, `b.grad = 2.0` means the same nudge to `b` would increase `L` by about **2x** that (`0.002`).

### Run autograd

Run the [`autograd`](src/my_microgpt/autograd.py) module:

```bash
uv run autograd
```

## 4. Parameters (Neurons)

**Parameters** can be seen as the neurons that hold the knowledge of the model. These floating-point numbers, wrapped in `Value` class for getting autograd, are going to be iteratively optimized during training through the learning process provided by the forward and backward progagation passes.

The so-called **hyperparameters** control the architecture shape:

- `n_embd = 16` — embedding dimension
- `n_head = 4` — number of attention heads
- `n_layer = 1` — number of layers
- `block_size = 16` — maximum sequence length
- `head_dim = n_embd // n_head` — dimension of each head

Parameters are initialized to random numbers drawn from a **Gaussian**, and they're gonna be organized into matrices. The `state_dict` structure organizes those matrices in: embedding tables (`wte`, `wpe`), attention weights (`attn_wq`, `attn_wk`, `attn_wv`, `attn_wo`), MLP weights (`mlp_fc1`, `mlp_fc2`), and a final output projection (`lm_head`). All parameters are also flattened into a single list `params` so the optimizer can loop over them later.

The weight matrices are stored as `(nout, nin)` following PyTorch's convention: each row holds the weights for one output neuron, so the forward pass is `output = W @ input`.

```text
state_dict (default config: n_embd=16, n_head=4, n_layer=1, block_size=16)
├── Embeddings
│   ├── wte              27 x 16     token embeddings (vocab_size x n_embd)
│   └── wpe              16 x 16     position embeddings (block_size x n_embd)
│
├── Layer 0
│   ├── Attention
│   │   ├── attn_wq      16 x 16     query projection (n_embd x n_embd)
│   │   ├── attn_wk      16 x 16     key projection
│   │   ├── attn_wv      16 x 16     value projection
│   │   └── attn_wo      16 x 16     output projection
│   └── MLP
│       ├── mlp_fc1      64 x 16     expand (4*n_embd x n_embd)
│       └── mlp_fc2      16 x 64     contract (n_embd x 4*n_embd)
│
└── Output
    └── lm_head          27 x 16     final projection (vocab_size x n_embd)

Total: 4,192 parameters
```

With our default configuration and a vocabulary of 27 tokens, this comes out to 4,192 parameters. For comparison GPT-2 had ~1.6 billion.

### Run parameters

Run the [`parameters`](src/my_microgpt/parameters.py) module:

```bash
uv run parameters
```
