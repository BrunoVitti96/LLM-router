# Decision-Aligned LLM Router

A research-grade, quality-preserving router that learns when a faster open-weight
model can replace a stronger fallback. The routing encoder is
[`nomic-ai/modernbert-embed-base`](https://huggingface.co/nomic-ai/modernbert-embed-base)
with rank-4 LoRA adapters and three prediction heads: replacement safety, latency,
and output tokens.

The repository is intentionally split into two layers:

- the notebook explains and runs the experiment;
- the Python package owns reusable training, inference, selection, and evaluation
  logic.

The repository supports two complementary experiments: a lightweight public-data
experiment on [LLMRouterBench](https://github.com/ynulihao/LLMRouterBench) to
establish headroom and compare simple baselines, and the original
hardware-specific v4 experiment that trains ModernBERT on the measured
three-model panel.

## Repository layout

```text
.
├── notebooks/
│   └── 01_train_modernbert_router.ipynb
├── src/llm_router/
│   ├── benchmark_cli.py
│   ├── public_benchmark.py
│   ├── config.py
│   ├── models/
│   │   ├── qwen2_5_1_5b_inference.py
│   │   ├── fast_dllm_v2_1_5b_inference.py
│   │   ├── qwen2_5_7b_4bit_inference.py
│   │   ├── modernbert_router.py
│   │   └── modernbert_router_inference.py
│   └── utils/
│       ├── artifacts.py
│       ├── calibration.py
│       ├── data.py
│       ├── evaluation.py
│       ├── routing.py
│       └── training.py
├── tests/
├── llm_router_project_scope_4.md
└── pyproject.toml
```

Each candidate has a dedicated inference class while sharing a small base adapter
for the common chat-template and timing contract. The ModernBERT artifact has its
own inference runtime and does not depend on notebook state.

## Objective function and training loss

There are two related—but different—objectives in this project: the deployment
objective defines what a useful routing policy must accomplish, while the
training loss teaches the neural network to produce the predictions used by that
policy.

### Deployment objective: save resources subject to quality

Let (f) be the strongest training-selected fallback, let \(\pi(x)\) be the
model selected for prompt (x), let (Q_m(x)) be the measured quality of model
(m), and let (R_m(x)) be the resource being minimized. In the local v4
experiment (R) is latency; in the public benchmark it can be cost or simulated
latency. Policy selection solves:

$$
\min_{\pi}\; \frac{1}{N}\sum_x
\left[R_{\pi(x)}(x) + R_{router}(x)\right]
$$

subject to:

$$
\operatorname{LCB}_{95\%}\left(
\frac{\mathbb{E}[Q_{\pi(x)}(x)]}{\mathbb{E}[Q_f(x)]}
\right) \ge 0.98.
$$

In plain language: choose the cheapest or fastest eligible model for every
prompt, but activate the router only when the **one-sided 95% lower confidence
bound** says it retains at least 98% of fallback quality. The current
implementation uses a paired normal-approximation bound. Net savings must also
remain positive after router overhead.

For each non-fallback candidate (m), the supervised safety target is:

$$
y_m(x)=\mathbf{1}\left[Q_m(x)\ge Q_f(x)-\epsilon_q\right].
$$

With the default \(\epsilon_q=0\), a candidate is safe when its observed score is
no worse than the fallback's. At inference time it is eligible only when its
calibrated safety probability exceeds the validation-selected threshold and its
predicted resource use is at least 2% below the fallback. The fallback is always
eligible.

### Training loss: learn safety, latency, and output length

ModernBERT emits replacement-safety logits \(s_m\), log latency
\(\widehat{\ell}_m\), and log output tokens \(\widehat{t}_m\). Its total loss is:

$$
\mathcal L =
1.00\,\mathcal L_{safety}
+0.75\,\mathcal L_{latency}
+0.10\,\mathcal L_{tokens}
+0.25\,\mathcal L_{opportunity}.
$$

Each term has a separate job:

- **Safety loss:** weighted binary cross-entropy learns whether each smaller
  model can replace the fallback. Unsafe examples receive extra weight based on
  their quality drop; safe examples receive extra weight based on the available
  latency saving.
- **Latency loss:** Smooth L1 regression on `log1p(generation_seconds)` learns
  latency ordering while limiting the influence of timing outliers.
- **Token loss:** Smooth L1 regression on `log1p(output_tokens)` provides an
  auxiliary response-length signal, an important driver of latency and cost.
- **Opportunity-margin loss:** for a candidate that was safe and faster,
  `softplus(margin - safety_logit) * normalized_gain` pushes its safety logit
  upward. Larger missed savings create a stronger push; slower candidates get
  no reward.

The safety-example weight is:

$$
w_m(x)=
\begin{cases}
1 + 2g_m(x), & y_m(x)=1,\\
1 + 4d_m(x), & y_m(x)=0,
\end{cases}
$$

where (g_m(x)) is the normalized positive latency opportunity and (d_m(x))
is the fallback-relative quality drop. Thus an unsafe switch is penalized more
heavily than a missed speedup without changing the binary safety label.

The neural loss does **not** directly select the deployed model. It produces
safety and resource estimates; validation calibrates probabilities, selects
thresholds, includes router overhead, and applies the confidence-aware
deployment constraint. This separation is necessary because the final routing
decision is discrete and constrained.

## Public benchmark proof of concept

Download and extract the pre-collected LLMRouterBench results, then inspect the
available dataset and model directory names:

```bash
llm-router-benchmark \
  --data-root /path/to/LLMRouterBench \
  --list-inventory
```

Copy `configs/benchmark_scenario.example.json` and replace every illustrative
price, TTFT, throughput, and queue value with dated measurements or cited
provider statistics. Select a small complementary pool—ideally four to six
models on the empirical quality/resource Pareto frontier—and run:

```bash
llm-router-benchmark \
  --data-root /path/to/LLMRouterBench \
  --scenario configs/my_scenario.json \
  --models model-a,model-b,model-c,model-d \
  --objective cost \
  --split-mode dataset_ood \
  --output-dir reports_benchmark/cost_ood
```

The workflow reports the best single model, cheapest single model,
dataset-lookup baseline, outcome-aware oracle, and a TF-IDF safety router. The
default `dataset_ood` split holds out entire datasets; `random` provides an easier
interpolation comparison. Validation selects a shared safety threshold using the
95% lower quality-retention bound, and test outcomes stay sealed until the
threshold and activation guard are frozen.

Scenario cost comes from per-example input/output tokens and dated token prices.
Simulated latency is:

```text
network + queue + TTFT + completion_tokens / output_tokens_per_second
```

Realized completion tokens are used only to evaluate a frozen policy. Routing
uses train-only dataset/model resource medians, backing off to global training
medians for unseen datasets, so it never sees a test response's length before
choosing. The exported manifest records every scenario assumption. Simulated
latency should not be described as measured production latency.

## Train in Google Colab

The experiment reuses the immutable v3 evidence under
`/content/drive/MyDrive/llm_router_v3`. Run it on the same GPU type recorded in
`run_manifest_v3.json`, because router overhead and candidate latency must be
comparable.

```bash
git clone --branch develop https://github.com/BrunoVitti96/LLM-router.git
cd LLM-router
pip install -e ".[notebook,quantization]"
```

Then open `notebooks/01_train_modernbert_router.ipynb` and run it top to bottom.
The notebook audits all 900 prompts and 2,700 measurements before training.

## Use the exported router

```python
from llm_router.models.modernbert_router_inference import ModernBERTRouterInference

router = ModernBERTRouterInference.from_artifact(
    "/path/to/artifacts_v4/<evidence-tag>__<router-tag>"
)

decision = router.route(
    prompt="Choose the best answer ...",
    task="mmlu",
    subject="abstract_algebra",
    num_choices=4,
    length_bin="s",
)
print(decision.selected_model, decision.router_active)
```

The artifact's deployment guard is authoritative. If the one-sided 95% validation
lower bound did not retain at least 98% of fallback quality while reducing net
latency, inference returns the fallback without running the encoder.

## Candidate inference

```python
from llm_router.models import QwenOnePointFiveBInference

model = QwenOnePointFiveBInference(revision="<revision from run_manifest_v3.json>")
result = model.generate(prompt, task="gsm8k")
print(result.response)
```

Available classes are `QwenOnePointFiveBInference`,
`FastDLLMV2OnePointFiveBInference`, and `QwenSevenB4BitInference`. Load one model
per worker; keeping all candidates resident is outside this experiment's scope.

## Quality contract

An alternative is safe for prompt `x` when its measured quality is no worse than
the training-selected fallback. Validation selects per-candidate Platt calibration,
safety thresholds, and a neural/empirical latency blend. A router checkpoint is
deployable only when it achieves both:

```text
one-sided 95% quality-retention LCB >= 0.98
net latency reduction > 0 (including router overhead)
```

See [`llm_router_project_scope_4.md`](llm_router_project_scope_4.md) for the full
experimental contract.

## Development checks

```bash
pip install -e ".[dev]"
pytest
python -m compileall -q src
```
