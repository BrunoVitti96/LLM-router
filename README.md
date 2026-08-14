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

## Repository layout

```text
.
├── notebooks/
│   └── 01_train_modernbert_router.ipynb
├── src/llm_router/
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

## Train in Google Colab

The experiment reuses the immutable v3 evidence under
`/content/drive/MyDrive/llm_router_v3`. Run it on the same GPU type recorded in
`run_manifest_v3.json`, because router overhead and candidate latency must be
comparable.

```bash
git clone --branch main https://github.com/BrunoVitti96/LLM-router.git
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

The artifact's deployment guard is authoritative. If validation did not retain at
least 98% of fallback quality while reducing net latency, inference returns the
fallback without running the encoder.

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
quality retention >= 0.98
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
