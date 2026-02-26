# Micro-Mechanistic Interpretability: Exploring the Activation Steering in microgpt

This variation aims at extracting concept vectors from the model's residual stream and inject them at inference time to steer name generation toward specific patterns.

Inspired by the talk in our CS group on Jack Lindsey's ["Emergent Introspective Awareness in Large Language Models"](https://arxiv.org/abs/2601.01828), this is a "quick and dirty" adaptation for the ~4K parameter character-level transformer from [AK's microgpt](https://karpathy.github.io/2026/02/12/microgpt/).

AFAIK, there's no published steering work exists below 124M parameters. Let's see if this makes sense. **Disclaimer**: I'm not an expert on MecInt, just exploring here...

## Lessons to Learn???

Even in a tiny model, the residual stream should encode structure that can be extracted and manipulated. The same mathematical framework (contrastive activation subtraction) that the paper describes working **(more or less)** on **(some)** billion-parameter models should apply apply here. **The main and maybe only question is whether the model's capacity is sufficient for the concept to be linearly separable in the activation space**.

This is the core insight of mechanistic interpretability: neural networks operate as mathematical functions on vectors, and their behavior can be understood by studying those vectors directly.

## Table of Contents

1. [Background](#1-background)
2. [Adapting for microgpt](#2-adapting-for-microgpt)
3. [The Concept Vector Pipeline](#3-the-concept-vector-pipeline)
4. [Extraction](#4-extraction)
5. [Injection](#5-injection)
6. [Experiments](#6-experiments)
7. [Interpreting Results](#7-interpreting-results)
8. [TODOs](#8-todo-other-experiments)

## 1. Background

### Activation Steering 101

In a nutshell, LLMs build internal representations as data from the input flows through their transformers and linear layers. At each layer, the model maintains the so-called a **residual stream**, which is a vector that goes and goes compressing information from the previous attention and MLP blocks. What **activation steering** is trying to do is to exploit this: by adding a carefully chosen vector to the residual stream, in theory, you can "nudge" the model's behavior without changing any weights. In other words, "invasive artificial brain poking".

### Contrastive activation subtraction

The core technique explained in more detail in the recent paper from Anthropic ["Emergent Introspective Awareness in Large Language Models"](https://arxiv.org/abs/2601.01828) but in essence:

1. Concept data collection:

    1.1 Collect the model's internal activations on inputs that exhibit a target concept (in our case I'm gonna choose names containing "fran"). Note that in the paper, they just use one word, but because of the differences in the size we are going to have all the words that contain, let me call it, the "fran-ness" ;-) whatever that implies... :)

    1.2. Collect activations on inputs that do not exhibit the concept

3. Subtract the mean of the negative set from the mean of the positive set
4. The resulting **concept vector** is supposed to capture the direction in activation space that distinguishes the concept. In other words the "fran-ness".

The peaper tries to demonstrate this at scale (billions of parameters) in Anthropic's models, for concepts like honesty, refusal, and sentiment. The aim here is to apply the same algorithm to AK's tiny model operating on character-level patterns and see what are the results.

## 2. Adapting for microgpt

This is a mini summary of the key differences from production-scale steering experiments:

| Aspect | Production LLMs | microgpt |
|--------|-----------------|----------|
| Parameters | 124M–405B | ~4K |
| Residual stream dim | 768–8192 | 16 |
| Tokenization | Subword (BPE) | Character-level |
| Concepts | Semantic (honesty, sentiment) | Character patterns (name substrings) |
| Sequence context | Prompts with instructions | Just the name characters (The AK's goal hasn't been changed) |
| Layers | 12–128 | 3 (in as per my training) |

In theory, the 16-dimensional residual stream is the biggest constraint, as it has to compress all the information in that small size vector. In a 768-dim model, a concept vector has room to encode nuance. In our tiny 16 dimensions, the vector must capture the entire concept in a very compressed space. This makes the technique more fragile but also more interpretable as in theory we can inspect all 16 values directly.

The original AK's model successfully learns to generate plausible names from ~4K parameters. What this implies is that despite being tiny, the 16-dim residual stream encodes enough structure to distinguish character patterns of the ~32K names we have in the training dataset. So, again, in theory, if the model has learned a direction for "fran-ness names" vs "other names" not that cool ;-), contrastive subtraction should "manipulate" it.

## 3. The Concept Vector Pipeline

```text
┌─────────────────────┐
│  Trained Model      │  model_e16_h4_l3_s1000.json
│  (frozen weights)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐     ┌──────────────────┐
│  concept_vectors.py │◄────│  Dataset          │
│                     │     │  (positive/       │
│  1. Select positive │     │   negative names) │
│     & negative names│     └──────────────────┘
│  2. Data-only       │
│     forward pass    │
│  3. Capture residual│
│     stream at layer │
│  4. Average across  │
│     positions       │
│  5. Subtract means  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  concept_fran_      │  16-dim JSON vector
│  layer2.json        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  steered_inference  │
│                     │
│  For each token:    │
│  1. Forward pass    │
│  2. Add alpha *     │
│     concept_vector  │
│     at target layer │
│  3. Sample next     │
│     token           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Steered names      │  "francisco", "frankie", "francool", "frankenstein?" ...
└─────────────────────┘
```

### Adpating the AK's Original Code with Minimal Changes

Both extraction and injection of the concept vectors will require a forward pass over the architecture without autograd overhead.

In order to do so, rather than duplicating `gpt()` (and probably all its helpers `linear`, `softmax`, `rmsnorm`, `relu`) with float-only equivalent vectors, I've just added a no_grad() method in the `Value` class to avoid backprop, and added an optional hook to the orignial `gpt()` function, which works for both training, and inference, extraction, and injection. Let's see those in more detail.

#### Change 1: `Value.no_grad()` context manager ([`autograd.py`](src/my_microgpt/autograd.py))

As AK's explain in his blogpost, during training, every `Value` operation (`+`, `*`, `.relu()`, ...) records not only its inputs, but also the local gradients so `backward()` can compute derivatives. In our new extraction and injection phases for MC, we only need the forward-pass numbers. Adding a class-level `_no_grad` flag makes `__init__` skip storing `_children` and `_local_grads` when set:

```python
# In Value.__init__:
if Value._no_grad:
    self._children = ()
    self._local_grads = ()
else:
    self._children = children
    self._local_grads = local_grads
```

The `Value.no_grad()` has been implemented as a context manager idiom in Python, which allows to flips the flag on entry and restores it on exit, micmicking what PyTorch's `torch.no_grad()` is doing. All existing operations (`__add__`, `__mul__`, `.relu()`, etc.) keep working like no change would have happened;s o they still pass `children` and `local_grads` to `__init__`, but now `__init__` will ignore them if the flag is on. This is how it's used now in a nutshell:

```python
with Value.no_grad():
    # All Value operations here produce correct .data
    # but skip graph construction — faster, no memory for gradients
    logits = gpt(token_id, pos_id, model, keys, values)
```

#### Change 2: `post_mlp_hook` parameter in `gpt()` ([`architecture.py`](src/my_microgpt/architecture.py))

In concept extraction we need to **read** the so-called residual stream at a specific layer. Later, in the steered inference we need to **modify** it. Both happen at the same point: after the MLP residual add, and before the next layer begins. We achieve this with one optional parameter and two lines inside the layer loop:

```python
def gpt(token_id, pos_id, model, keys, values,
        post_mlp_hook=None):  # <-- new, defaults to None
    ...
    for li in range(cfg.n_layer):
        # bla bla bla
        ...  # attention + MLP blocks (unchanged)
        x = [a + b for a, b in zip(x, x_residual)]  # post-MLP residual

        if post_mlp_hook is not None:   # <-- new (2 lines)
            x = post_mlp_hook(x, li)

    return linear(x, sd["lm_head"])
```

The hook receives `(residual_stream, layer_index)` and returns a (possibly modified) residual stream. The current existing callers (`training.py`, `inference.py`) dont' pass a hook, so `post_mlp_hook` the condion on the `if` will never be entered and they won't be impmacted by the change.

#### Extra Change: Iterative Topological Sort in Autograd

This is necessary to avoid recursion-depth limits on training larger models (e.g. dims > 16). I've found this when executing bigger representations.

```python
stack: list[tuple[Value, int]] = [(self, 0)]
while stack:
    node, idx = stack[-1]
    if id(node) in visited:
        stack.pop()
        continue
    if idx < len(node._children):
        stack[-1] = (node, idx + 1)
        child = node._children[idx]
        if id(child) not in visited:
            stack.append((child, 0))
    else:
        visited.add(id(node))
        topo.append(node)
        stack.pop()
```

...instead of calling `build_topo(self)` recursively. See Section #7 [Interpreting Results](#7-interpreting-results) below.

#### How These Changes are Used in Extraction and Injection

We basically define a hook for each task and we pass it to the brand new `gpt()` function signature.

**Capture hook** (in [`concept_vectors.py`](src/my_microgpt/concept_vectors.py)): reads the residual stream, returns it unchanged.

```python
activations = []
def capture_hook(x, li):
    if li == target_layer:  # <- this is where we pay attention to the layer where to extract the concept vector
        activations.append([v.data for v in x])
    return x  # pass-through — don't modify

Then, later gradients are inhibited:

with Value.no_grad():
    ...
    gpt(token_id, pos_id, model, keys, values, post_mlp_hook=capture_hook)  # <- we pass the hook here
```

**Injection hook** (in [`steered_inference.py`](src/my_microgpt/steered_inference.py)): adds the scaled concept vector to the residual stream. The parameter `alpha` below is what in the paper is called `strenght` as per our discussion yesterday.

```python
def inject_hook(x, li):
    if li == injection_layer:  # <- this is where we pay attention to the layer to inject the concept vector
        return [xi + alpha * ci for xi, ci in zip(x, concept_vector)]
    return x

Then, later, nested in the hierarchy of calls, the inhibition of the gradients happens too:

with Value.no_grad():
    ...
    logits = gpt(token_id, pos_id, model, keys, values, post_mlp_hook=inject_hook)  # <- we pass the hook here
```

Because `Value` supports `__add__` and `__rmul__` with plain floats, the injection `xi + alpha * ci` (where `xi` is a `Value` and `alpha * ci` is a `float`) works naturally — and under `no_grad()`, the result skips graph construction.

## 4. Extraction

### How it Works at a Glance

For each name in the positive and negative sets:

1. Encode the name as token ids (wrapped with BOS)
2. Run `gpt()` (with `Value.no_grad()`) with a capture hook, intercepting the residual stream after the MLP block at the target layer
3. This produces one 16-dim vector per character position
4. Average across all positions to get a single 16-dim vector for the name

Then, conceptually:

```text
mean_positive = average of all positive name vectors
mean_negative = average of all negative name vectors
concept_vector = mean_positive - mean_negative
```

### Token Position Strategy

A difference with the proposed solution in the paper, the `word` become a set of poistive names, which won't be all the same length unfortunately ("fran" = 4 chars, "francisco" = 9 chars). To compensate this, we average across all positions to capture the "distributed character-level concept" (again the "fran-ness"???), rather than relying on a specific position. Every position contributes to the model's internal representation of what kind of name this is (yes, there are more names apart of Fran, even in San Francisco).

### Running extraction



First, train a model (or use an existing one):

```bash
uv run training --num-steps 1000 --output model.json
```

Then extract the concept vector:

```bash
uv run python -m my_microgpt.concept_vectors \
    --model model.json \
    --concept fran \
    --layer 2 \
    --num-negative 50 \
    --output concept_fran_layer2.json
```

Output:

```json
{
  "concept": "fran",
  "layer": 2,
  "hook_point": "post_mlp",
  "alpha": 1.0,
  "num_positive": 12,
  "num_negative": 50,
  "vector": [0.12, -0.34, ...]
}
```

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | required | Path to trained model JSON |
| `--concept` | `fran` | Substring to match for positive examples |
| `--layer` | `2` | Layer to extract from (0-indexed, 2 = last layer in default 3-layer model) |
| `--num-negative` | `50` | Number of random negative examples |
| `--output` | auto | Output path (default: `concept_{concept}_layer{layer}.json`) |

## 5. Injection

### How steering works

During generation, at each forward pass step:

1. Process the token through the network normally
2. After the MLP block at the injection layer, add `alpha * concept_vector` to the residual stream
3. Continue the forward pass with the modified activations
4. Sample the next token from the steered logits

The injection happens at every generation step, continuously nudging the model's internal state toward the concept direction.

### The alpha parameter

Alpha controls steering strength:

- **alpha = 0.0** — No steering. Identical to normal inference (baseline)
- **alpha = 0.5** — Gentle steering. Subtle bias toward the concept
- **alpha = 1.0** — Unit steering. The concept vector at its extracted magnitude
- **alpha = 2.0–5.0** — Strong steering. Increasingly dominant concept influence
- **alpha > 5.0** — Likely degenerate output (repetitive patterns or garbage)
- **alpha < 0.0** — Reverse steering. Push *away* from the concept

The strenght `alpha` paramteres we're gonna use here are way smaller than the ones used in the paper to adjust to the model size.

### Running steered inference

```bash
uv run python -m my_microgpt.steered_inference \
    --model model.json \
    --concept-vector concept_fran_layer2.json \
    --alpha 0.0 0.5 1.0 2.0 5.0 \
    --temperature 0.5 \
    --num-samples 20
```

This generates 20 names at each alpha value and reports how many contain the concept substring.

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | required | Path to trained model JSON |
| `--concept-vector` | required | Path to concept vector JSON |
| `--alpha` | `0.0 0.5 1.0 2.0 5.0` | Steering strengths to sweep |
| `--temperature` | `0.5` | Sampling temperature |
| `--num-samples` | `20` | Names to generate per alpha |

## 6. Experiments

### Limitations

As we introduced before, the major points to cosider with the success or not of the approach are gonna be related mainly to:

- **low 16-dim compression representation:** The residual stream may not have enough capacity to encode separable concepts. Multiple patterns may share dimensions.
- **Small dataset overlap:** With only ~12 "fran" names out of 32K, the positive set is small. The concept vector estimate may have high variance. But the article has only one... who knows...
- **Character-level concepts:** Unlike semantic concepts in large models, we're steering toward character patterns. The model may not represent these as "clean" linear directions. Averaging all over all the chars may help???
- **No causal interpretation:** A steering effect doesn't prove the model "understands" the concept. It may just be exploiting a statistical correlation in activation space.

### Experiment 1: Finding ~~Nemo~~ the Right Vector for "fran-ness"

The main experiment. The dataset contains 39 names with "fran" (francisco, francis, frankie, etc.). Extract the concept vector and sweep alpha (what is called strenght in the paper above.)

**Expected outcome:** At `alpha=0`, the baseline rate of "fran" names should be very, very low (< ~0.122%, as there are only 39 fran-like names out of 32033 total names) as it should behave like no manimpulation has been done, that is regular inference. At increasing alpha, the rate should climb. At very high alpha, output should degenerate.

So for 39 in 32,033 (approximately 0.122% as we said above), the number of samples needed to detect at least one "fran-like name" is:

1. With 95% Confidence: At least 2,460 samples
2. With 99% Confidence: At least 3,781 samples

Let's use that number (3,781) to set a baseline for the number of generated names to detect one fran-like, although this is not ideal is reasonable. Being a generative model, then it doesn't reproduce the training distribution literally. It will generate novel names from learned character-level patterns. 3,781 is a conservative lower bound for the baseline, meaning that any "fran" hit under steering is clearly attributable to the concept vector, not chance. For the steered experiments, 3,781 samples is more than enough. If steering works, we should see the rate climb well above 0.122%! Even a few percent would be a strong signal. At that rate, a few hundred samples would suffice to detect the effect but I want to keep 3,781 to give us more chance of seeing the effect of the steering. I wish...

We will inject first in the last layer (paper said best one was 2/3), but we will start there. Then move to the middle of the model in layer 1.

**Expected outcome:** Earlier layers capture more syntactic features (character n-gram patterns), later layers capture more abstract patterns. Layer 2 (the last) likely should work better because the residual stream has accumulated the most context, but in a tiny model is still to be seen (or, most likely, not observed ;-)


#### Extract the Concept Vector for "fran-ness"

**Layer 2**

I use the model I trained, and I extract it from **layer 2** as first trial (more or less like in the paper, although more layers would be nice to have. Train more in another experiment maybe if this goes wrong?)

```sh
uv run python -m my_microgpt.concept_vectors \
    --model model_e16_h4_l3_s32033.json \
    --concept fran \
    --layer 2
```

Output:
```text
model loaded from model_e16_h4_l3_s32033.json (10,336 params, 32033 steps, loss=2.2816)
num docs: 32033
concept: 'fran'
positive names (39): ['franchesca', 'frandy', 'franklynn', 'franco', 'franklyn', 'frankee', 'francine', 'franky', 'francois', 'gianfranco', 'franklyn', 'franz', 'frankie', 'fransisco', 'francheska', 'francis', 'franklin', 'frances', 'franceska', 'franziska', 'frans', 'francia', 'francesco', 'francis', 'francisca', 'franck', 'frantz', 'francely', 'franciszek', 'frank', 'franki', 'francisco', 'franko', 'francesca', 'franklin', 'francie', 'efran', 'maryfrances', 'frankie']
negative names (50): ['sareena', 'taw', 'oktavia', 'jovonni', 'rocko', 'demetrios', 'md', 'aben', 'syra', 'eshan', 'misael', 'jaccob', 'emarion', 'aowyn', 'jyree', 'aleea', 'mads', 'acari', 'caiden', 'hajra', 'eydan', 'kymbree', 'zaye', 'bretton', 'zaylia', 'stefano', 'jeanpierre', 'jamera', 'tinsleigh', 'bento', 'jajuan', 'nicole', 'olympia', 'kyrielle', 'avonlea', 'naraly', 'cressida', 'kadynce', 'elleigh', 'jamya', 'leddy', 'kalijah', 'adah', 'saer', 'annabell', 'mamie', 'aneri', 'ady', 'brunella', 'rebeca']
extracting activations for 39 positive names...
extracting activations for 50 negative names...
concept vector norm: 0.8941
concept vector saved to concept_fran_layer2.json (547 bytes)
```

Looks promising. The concept vector is > 0. And there were 39 names "fran" related names.

And this is the content of the generated concept vector in `conept_fran_layer2.json`:

```json
{
  "concept": "fran",
  "layer": 2,
  "hook_point": "post_mlp",
  "alpha": 1.0,
  "num_positive": 39,
  "num_negative": 50,
  "vector": [
    -0.002970869491156991,
    -0.4459307489660401,
    0.19463724019509096,
    -0.07137764696850213,
    0.1471926794933648,
    0.46005105986789013,
    0.290263520437014,
    0.2165130977241848,
    0.033745366729877366,
    -0.09087748044745154,
    0.00115286520860447,
    0.005639343200178171,
    -0.08285039893418557,
    -0.06217326734172707,
    0.009735010716916126,
    0.4157107692608888
  ]
}
```

Regular inference run (at default temperature of 0.5 as in AK's blog):

```sh
uv run src/my_microgpt/inference.py --model model_e16_h4_l3_s32033.json 
```

List of names generated with regular inference with the pretrained model:

```text
model loaded from model_e16_h4_l3_s32033.json (10,336 params, 32033 steps, loss=2.2816)
Model loaded from model_e16_h4_l3_s32033.json. Info: TrainingInfo(num_steps=32033, final_loss=2.281591759987865)

--- inference (temperature=0.5) ---
sample  1: anan
sample  2: mani
sample  3: chenyle
sample  4: carar
sample  5: shanie
sample  6: lani
sample  7: anni
sample  8: maren
sample  9: jamba
sample 10: adara
sample 11: kavin
sample 12: aristhi
sample 13: kellyn
sample 14: arala
sample 15: lian
sample 16: rilliana
sample 17: adidan
sample 18: rona
sample 19: miki
sample 20: rayana
...
sample 3775: javenn
sample 3776: saba
sample 3777: analiah
sample 3778: aria
sample 3779: saina
sample 3780: layna
sample 3781: amaria

-> 0/3781 contain 'fran' (0.0%)
```

Looks more or less "ok". rilliana and jamba are cool! But "frans" are zero, which confirmsthat the model's actual "fran" generation rate is even lower than the dataset base %. Yes, I know that 3,781 it's "only" 99% confidence...

Now let's try the steered version but with alpha 0, which should behave as the regular inference above:

```sh
# Run steered inference
uv run python -m my_microgpt.steered_inference \
    --model model_e16_h4_l3_s32033.json \
    --concept-vector concept_fran_layer2.json \
    --alpha 0.0
```

This is the output:

```text
model loaded from model_e16_h4_l3_s32033.json (10,336 params, 32033 steps, loss=2.2816)
concept vector loaded from concept_fran_layer2.json (concept='fran', layer=2, dim=16)

--- alpha=0.0 (temperature=0.5) ---
  sample  1: adray
  sample  2: analiah
  sample  3: araela
  sample  4: ardee
  sample  5: jelen
  sample  6: zatha
  sample  7: jayley
  sample  8: eleny
  sample  9: kaillen
  sample 10: jasha
...
  sample 3775: laylen
  sample 3776: aylon
  sample 3777: raylian
  sample 3778: bristan
  sample 3779: kasten
  sample 3780: ten
  sample 3781: tarien
  -> 0/3781 contain 'fran'
```

Cool up to now... Analiah? That name would be a favorite for Syrus...

Now let's try the real steered version with alpha > 0, 0.5 in this case:

```sh
# Run steered inference
uv run python -m my_microgpt.steered_inference \
    --model model_e16_h4_l3_s32033.json \
    --concept-vector concept_fran_layer2.json \
    --alpha 0.5
```

Output:

```text
model loaded from model_e16_h4_l3_s32033.json (10,336 params, 32033 steps, loss=2.2816)
concept vector loaded from concept_fran_layer2.json (concept='fran', layer=2, dim=16)

--- alpha=0.5 (temperature=0.5) ---
  sample  1: shie
  sample  2: laysen
  sample  3: jarie
  sample  4: tayani
  sample  5: jaellin
  sample  6: aviri
  sample  7: adee
  sample  8: nison
  sample  9: amelio
  sample 10: andeley
...
  sample 3775: avirl
  sample 3776: aman
  sample 3777: kenn
  sample 3778: yani
  sample 3779: sana
  sample 3780: saie
  sample 3781: aalin
  -> 0/3781 contain 'fran'
```

Nothing there! Uhmmmm...

More power.. Now with alpha 1.0:

```text
model loaded from model_e16_h4_l3_s32033.json (10,336 params, 32033 steps, loss=2.2816)
concept vector loaded from concept_fran_layer2.json (concept='fran', layer=2, dim=16)

--- alpha=1.0 (temperature=0.5) ---
  sample  1: cary
  sample  2: sarie
  sample  3: aniyah
  sample  4: jasa
  sample  5: aries
  sample  6: aavah
  sample  7: kaysi
  sample  8: elason
  sample  9: aniton
  sample 10: nara
...
  sample 3775: avine
  sample 3776: nika
  sample 3777: asona
  sample 3778: aie
  sample 3779: ave
  sample 3780: emeri
  sample 3781: roani
  -> 0/3781 contain 'fran'
```

Zero...

Now with alpha 2.0:

```
  sample 3775: isina
  sample 3776: eon
  sample 3777: elea
  sample 3778: eri
  sample 3779: iton
  sample 3780: aalie
  sample 3781: elieie
  -> 0/3781 contain 'fran'
```

Also nothing...

Push to alpha 5.0...

```text
model loaded from model_e16_h4_l3_s32033.json (10,336 params, 32033 steps, loss=2.2816)
concept vector loaded from concept_fran_layer2.json (concept='fran', layer=2, dim=16)

--- alpha=5.0 (temperature=0.5) ---
  sample  1: eee
  sample  2: eie
  sample  3: ioie
  sample  4: eiya
  sample  5: iiee
  sample  6: eeie
  sample  7: eoie
  sample  8: aie
  sample  9: ieia
  sample 10: iee
...
```

Nonsese appears, so it's too much.

So we haven't found anything yet... Apart that the generated names are very short and maybe even the fran root is too much.

**Layer 1**

Let's try to extract and inject in layer 1, the one in the middle...

```sh
uv run python -m my_microgpt.concept_vectors \
    --model model_e16_h4_l3_s32033.json \
    --concept fran \
    --layer 1
```

Output:

```text
model loaded from model_e16_h4_l3_s32033.json (10,336 params, 32033 steps, loss=2.2816)
num docs: 32033
concept: 'fran'
positive names (39): ['frans', 'frankie', 'franklin', 'francesco', 'francheska', 'franklynn', 'frantz', 'frances', 'franki', 'frankie', 'francia', 'franceska', 'efran', 'franco', 'franz', 'francois', 'franko', 'franklyn', 'franck', 'franklin', 'francely', 'frankee', 'frandy', 'gianfranco', 'francis', 'francesca', 'maryfrances', 'frank', 'fransisco', 'francisco', 'franziska', 'francine', 'franciszek', 'franky', 'francie', 'franchesca', 'franklyn', 'francisca', 'francis']
negative names (50): ['kailea', 'advik', 'kynsley', 'kennidy', 'santhiago', 'xayah', 'lotanna', 'amori', 'penelope', 'hamdan', 'rari', 'alyra', 'nithya', 'clever', 'silvano', 'skylarr', 'lynnlee', 'cotton', 'tayvon', 'stirling', 'raymond', 'trinidy', 'victoria', 'zarria', 'aroosh', 'oralia', 'keyshawn', 'farhan', 'issabelle', 'ousmane', 'roderick', 'amrit', 'logynn', 'sahir', 'athaleyah', 'roseann', 'anai', 'min', 'chantelle', 'daviya', 'maleik', 'remee', 'jood', 'avaan', 'ysabelle', 'renlee', 'kejuan', 'tyga', 'brittain', 'keyton']
extracting activations for 39 positive names...
extracting activations for 50 negative names...
concept vector norm: 0.6110
concept vector saved to concept_fran_layer1.json (551 bytes)
```

The norm of the concept vector is still > 0... This is the concept vector:

```json
{
  "concept": "fran",
  "layer": 1,
  "hook_point": "post_mlp",
  "alpha": 1.0,
  "num_positive": 39,
  "num_negative": 50,
  "vector": [
    0.034433102109322766,
    -0.03322745623768508,
    0.015156969908565333,
    -0.14656224491184633,
    0.02830354323316639,
    0.3431210688364943,
    0.13958706814056818,
    0.1918884856887468,
    0.09431457600884241,
    -0.1572334261776819,
    -0.0012706060265759156,
    -0.14299608324641236,
    0.07893781409880456,
    -0.11046096058530727,
    -0.017453413724432088,
    0.3188778780435215
  ]
}
```

I've done quick trials with alpha 0.5 and 1.0 with null results. So I'm gonna stop for now.

## 7. Interpreting Results

### Signs of success

- The normalized concept vector is non-trivial (not near zero) -> That was promising... 0.8 and 0.6. Check!
- alpha=0 produces the same distribution as normal inference (should see all kinds of names) -> Checked!
- Clear monotonic increase in concept-match rate as alpha increases from 0 -> Not observed (Yet!)
- Generated names at moderate alpha are still plausible names -> Checked! Till 0.2 or so it's ok. I have the intution that the names are sorter with higher alphas, but I haven't checked the fact nor thougth about why. But with alpha 5.0 go nuts as expected...

### Signs of failure

- No change in concept-match rate across alpha values; the model hasn't learned a separable concept direction -> Yup!
- Concept vector is near-zero — positive and negative activations are too similar -> Nope!
- Even low alpha produces garbage — the residual stream is too sensitive to perturbation in 16 dimensions -> Nope!

### Possible causes

- Again main one is that most likely everything is too tiny, model, representations, dataset, training regime, etc.
- The steering **seems to be working** but not toward "fran"
    The concept vector is clearly modifying the model's behavior. At *alpha=5.0*, output degenerates to vowel-heavy nonsense (*"eee", "eie", "ioie"*). That is proof the vector is perturbing the residual stream in a consistent direction. The question is what that direction actually represents

- The contrastive subtraction computes: "what's statistically different about the activations of fran-names vs. random names, averaged across all character positions." It works when the concept is **linearly separable** in the model's representation space which may not be possible with our 16 vector dim.

    In a 768-dim model, there many room for many orthogonal directions: one could encode "contains the substring fran" as a near-independent feature. In our 16 dimensions, the residual stream must compress everything: character identity, positional patterns, name structure, length tendencies; so there may not be room for "fran-ness" as a clean separable direction. 16-dim space encode distributional features, not compositional ones; the model can't represent "f -> r -> a -> n" as a single direction because that's sequential/compositional, which requires multiple dimensions working across positions that can't be represented in the 16 dim embeding vector.

- What the vector likely captures instead are statistical correlates of fran-names:
  - Vowel/consonant distribution patterns
  - Average name length tendencies
  - Character bigram frequencies

- Could that explain the high-alpha degeneration pattern? getting short, vowel-heavy names, not fran-like names. The vector is steering toward the average statistical profile of fran-names, not the specific character sequence?

- Examining the two vector values above, let's pay attention to the Layer 2's dominant dimensions:

```text
┌─────┬────────┬─────────────────┐
│ Dim │ Value  │                 │
├─────┼────────┼─────────────────┤
│ 1   │ -0.446 │ strong negative │
├─────┼────────┼─────────────────┤
│ 5   │ +0.460 │ strong positive │
├─────┼────────┼─────────────────┤
│ 15  │ +0.416 │ strong positive │
├─────┼────────┼─────────────────┤
│ 6   │ +0.290 │ moderate        │
├─────┼────────┼─────────────────┤
│ 7   │ +0.217 │ moderate        │
└─────┴────────┴─────────────────┘
```
   - ... and Layer 1 looks similar more or less (dominant positive dimensions are 5, 6, 7, 15). This cross-layer consistency may be suggesting taht these dimensions encode a real structural feature, but it's a broad distributional feature, not "fran" like specifically :-(


## 8. TODO: Other Experiments

**I'll keep experimenting another time... For now I'm training a bigger model with bigger representation capacity 3x to try to make my "fran-ness" linearly separable: `model_e48_h4_l5_s32033.json` (Experiment 1 below)**


### Experiment 2: Bigger Model!

More dimensions... E X T R E M E L Y - S L OOOOOOOOOOOOO W E R to train in my Mac. Still one epoch to avoid overfitting...


### Experiment 3: Different concepts

Try concepts beyond "fran". He's not the center of the universe ;-):

- `--concept an` — names containing "an" (a very common pattern)
- `--concept ia` — names ending in "ia" (maria, sophia, olivia)
- `--concept ch` — names with "ch" (charlotte, rachel, zachary)

### Experiment 4: Negative alpha

Use `--alpha -1.0 -2.0` to steer **away** from a concept. With the "fran" vector at negative alpha, generated names should avoid "fran"-like patterns. What a pity...

### Experiment 5: Training duration

Compare concept vectors from models trained for different durations (100, 500, 1000, 5000 steps). A barely-trained model won't have meaningful internal representations, so its concept vectors should be weak.
