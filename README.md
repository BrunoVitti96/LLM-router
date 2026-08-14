# Latency-Aware Open-Weight LLM Router

This project tests whether a lightweight prompt router can reduce warm inference latency across already-loaded open-weight LLM workers while retaining near-strongest-model answer quality.

## Run the risk-controlled v5 router

First complete `notebooks/03_quality_safe_colab_router.ipynb` so its fingerprinted prompt and measurement parquet files exist in Google Drive. Then open `notebooks/05_risk_controlled_paired_outcome_router.ipynb` on the GPU type recorded by the v3 manifest and run it from top to bottom. The synchronized design is documented in `llm_router_project_scope_5.md`.

V5 replaces the weak binary safety target with four paired outcomes (`loss`, `wrong_tie`, `correct_tie`, and `gain`), replaces the 149M-parameter transformer router with fast word/character TF-IDF linear heads, reserves separate calibration and policy-validation prompts, models arithmetic-mean latency with a task-level p90 tail guard, and activates only when paired-bootstrap lower bounds preserve 98% quality and show positive net latency savings. V3 and v4 evidence and reports remain unchanged; v5 writes to separate `reports_v5` and `artifacts_v5` directories.

## Run the decision-aligned v4 router

First complete `notebooks/03_quality_safe_colab_router.ipynb` so its fingerprinted prompt and measurement parquet files exist in Google Drive. Then open `notebooks/04_decision_aligned_quality_router.ipynb` on the same GPU type recorded by the v3 manifest and run it from top to bottom. The synchronized design is documented in `llm_router_project_scope_4.md`.

V4 reuses all 900 prompts and 2,700 candidate measurements from v3 without regenerating answers. It trains a new fallback-relative router with rank-4 LoRA, decision-aware safety loss, per-candidate Platt calibration and thresholds, validation-selected latency blending, and overhead-inclusive checkpoint selection. V3 reports and artifacts remain unchanged; v4 writes to separate `reports_v4` and `artifacts_v4` directories.

## Run the quality-safe v3 experiment

Open `notebooks/03_quality_safe_colab_router.ipynb` in Google Colab, select a GPU runtime, and run it from top to bottom. Its synchronized design is documented in `llm_router_project_scope_3.md`. It performs real local inference, stores revision- and configuration-fingerprinted caches in Google Drive, scores responses against public benchmark references, adapts ModernBERT with LoRA plus quality-safety, direct-latency, and auxiliary output-token heads, calibrates safety on validation data, and evaluates the resulting selector on a sealed test split. The notebook is written as a guided walkthrough with architecture diagrams, stage explanations, expected outputs, and inline implementation comments.

The default v3 experiment uses 300 examples from each of GSM8K, MMLU, and ARC-Challenge: 900 prompts and 2,700 model generations. If a Colab session is short, set `MODELS_TO_RUN` to one candidate; later runs reuse compatible completed rows. The final notebook section provides an interactive browser for reviewing every prompt, the router's selected model, calibrated safety probabilities, measured and predicted latency, strict-format compliance, and every candidate response.

The previous `02_real_colab_router.ipynb` is the completed v2 real experiment and remains useful as a baseline. The older `01_latency_aware_llm_router_mvp.ipynb` is a synthetic pipeline dry run; its numbers are not model results.

## Important interpretation

Measured latency is warm, batch-size-one generation on one Colab GPU, with each model benchmarked sequentially. A deployed router would need the candidate models already loaded on separate workers; dynamic model loading is not included.

RouterBench is not used directly to score new generations because its released 0-shot table contains previous models' outputs and performance values but not the ground-truth references needed for new responses. The real notebooks instead load three of RouterBench's underlying benchmark families from their source datasets.
