# GPT-2 Modernization Roadmap (2026)

## Executive Summary

This document proposes a focused modernization plan for the `AlexKitipov/gpt-2` repository, which currently reflects an older GPT-2 codebase centered on TensorFlow 1.x inference utilities. The goal is to evolve it into a maintainable, testable, trainable, and production-ready library aligned with 2026 engineering standards.

The roadmap is intentionally split into phases so the project can deliver value early (packaging, CI, docs, basic PyTorch inference) while incrementally adding full training and release infrastructure.

---

## 1) Critical Issues in the Current Codebase

### 1.1 Outdated framework and APIs

- **TensorFlow 1.x dependency** (with graph/session-style execution) is deprecated and no longer a good default for modern NLP work.
- Likely use of **`tf.contrib`-era patterns** and static graph assumptions that do not map cleanly to current ecosystems.
- Existing architecture is difficult to integrate with modern tooling (Torch compile, AMP, DDP/FSDP, HF ecosystem).

### 1.2 Missing training stack

- Repository is effectively **inference-first**, with no robust, first-party training/fine-tuning loop.
- No integrated support for:
  - gradient accumulation
  - mixed precision (bf16/fp16)
  - distributed training
  - checkpoint lifecycle management
  - resume/restart semantics

### 1.3 Weak dependency and packaging posture

- Legacy-style `requirements.txt` constraints are likely stale and vulnerable.
- No modern package metadata/standards workflow (`pyproject.toml`, extras, lock strategy).
- Lack of compatibility matrix and version policy.

### 1.4 Testing and quality gaps

- Minimal or no automated tests.
- No CI gates for linting, type checking, and unit/integration testing.
- No reproducibility checks for tokenization/model outputs/checkpoint load behavior.

### 1.5 Documentation and maintainability deficits

- Missing architectural docs, contribution guide, and API reference.
- Sparse usage examples (training, fine-tuning, distributed launch, checkpoint conversion).
- Limited type hints and inconsistent module boundaries impede contribution velocity.

---

## 2) Modernization Goals and Success Metrics

### Primary goals

1. Replace TensorFlow 1.x runtime with **PyTorch 2.x-native implementation**.
2. Introduce a full **training and evaluation pipeline**.
3. Establish modern **dataset ingestion + preprocessing architecture**.
4. Achieve robust **testing, documentation, CI/CD**, and release hygiene.

### Success metrics

- 100% removal of TensorFlow runtime dependencies in core path.
- End-to-end training script supporting single-GPU and multi-GPU.
- >=80% coverage on critical modules (tokenizer wrappers, model blocks, training utils).
- Passing CI on Linux/macOS with Python compatibility matrix.
- Published package + container + model artifacts workflow (PyPI, Docker, HF Hub).

---

## 3) TensorFlow 1.x → PyTorch 2.x Migration Plan

## 3.1 Migration principles

- **Behavioral parity first**, then performance optimization.
- Keep model math faithful to GPT-2 reference design.
- Introduce compatibility utilities for legacy checkpoint conversion.

## 3.2 Work breakdown

### Phase A: Baseline parity

- Implement `GPT2Config` dataclass/pydantic config.
- Port core modules:
  - token/position embeddings
  - masked multi-head self-attention
  - MLP block
  - residual + layer norm stack
  - LM head + tied embeddings
- Implement autoregressive generation with sampling controls:
  - top-k / top-p / temperature / repetition penalty

### Phase B: Checkpoint interoperability

- Build conversion utility for TF checkpoint to PyTorch `state_dict`.
- Add strict key mapping and shape validation.
- Validate with golden-text parity tests on fixed prompts.

### Phase C: PyTorch 2.x optimization

- Add optional `torch.compile()` path.
- Ensure AMP autocast support for inference/training.
- Add flash-attention/xformers abstraction hooks (optional feature flag).

### 3.3 Deliverables

- `src/gpt2/modeling_gpt2.py`
- `src/gpt2/config.py`
- `src/gpt2/generation.py`
- `tools/convert_tf_checkpoint.py`
- migration + parity tests

---

## 4) Proposed New Project Structure

```text
gpt-2/
├── pyproject.toml
├── README.md
├── LICENSE
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── release.yml
│       └── docs.yml
├── src/
│   └── gpt2/
│       ├── __init__.py
│       ├── config.py
│       ├── modeling_gpt2.py
│       ├── generation.py
│       ├── tokenization.py
│       ├── optim/
│       │   ├── __init__.py
│       │   ├── schedulers.py
│       │   └── param_groups.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── datasets.py
│       │   ├── collators.py
│       │   └── streaming.py
│       ├── train/
│       │   ├── __init__.py
│       │   ├── loop.py
│       │   ├── evaluate.py
│       │   ├── checkpointing.py
│       │   └── distributed.py
│       └── utils/
│           ├── logging.py
│           ├── seed.py
│           └── io.py
├── scripts/
│   ├── train.py
│   ├── eval.py
│   ├── generate.py
│   └── export_hf.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
└── docs/
    ├── architecture.md
    ├── training.md
    ├── datasets.md
    ├── api.md
    └── migration_from_tf.md
```

### Structure rationale

- `src/` layout avoids import ambiguity and improves packaging correctness.
- Clear separation between model, data, training, and utility layers.
- Script entrypoints remain thin wrappers over library APIs.

---

## 5) Data Pipeline Redesign

### 5.1 Current pain points

- Manual data ingestion and ad-hoc preprocessing.
- No scalable abstraction for large corpora or streaming datasets.
- Weak reproducibility controls (shuffling seeds, deterministic splits).

### 5.2 Target architecture

- Unified dataset interfaces:
  - `TextDataset` (plain text)
  - `JsonlDataset` (instruction/record format)
  - `StreamingDataset` (web/local shards)
- Tokenization pipeline with:
  - pre-tokenized caching
  - memory-mapped shards
  - optional on-the-fly packing
- Data collators for:
  - causal LM fixed-length blocks
  - dynamic padding (for variable-length batches)

### 5.3 Performance features

- Multi-worker data loading with pinned memory.
- Prefetch and persistent workers.
- Optional dataset streaming for limited disk environments.
- Dataset fingerprinting to invalidate stale token caches safely.

### 5.4 Quality controls

- Dataset schema validation and stats report (length distribution, vocabulary diagnostics).
- Deterministic split generation with persisted manifests.

---

## 6) Training Loop Implementation Plan

### 6.1 Core capabilities

- Single-node single-GPU and multi-GPU (DDP).
- Gradient accumulation for effective large batch sizes.
- AMP (`bf16` preferred, `fp16` fallback).
- Gradient clipping + anomaly checks.
- Configurable evaluation cadence and checkpoint intervals.

### 6.2 Optimizers and schedulers (modern defaults)

- Optimizers:
  - `AdamW` baseline
  - optional `Lion`/`Adafactor` behind feature flags
- Learning rate schedules:
  - linear warmup + cosine decay (default)
  - constant with warmup
  - one-cycle (optional)

### 6.3 Checkpointing and resume

- Save:
  - model state
  - optimizer/scheduler state
  - scaler state (AMP)
  - RNG states and dataloader progress
- Provide robust resume behavior across preemptions.
- Keep “best”, “last”, and periodic checkpoints with retention policy.

### 6.4 Observability

- Structured logging (JSON + console).
- TensorBoard/W&B optional integrations.
- Per-step and per-epoch metrics:
  - loss
  - perplexity
  - throughput (tokens/s)
  - GPU memory footprint

---

## 7) Code Clarity, Modularity, and Type Safety

### 7.1 Refactoring standards

- Full type hints for public APIs.
- Dataclass/pydantic-based config schema.
- Strict lint + formatting (`ruff`, `black`, `isort`).
- `mypy` in CI for static typing guarantees.

### 7.2 API design principles

- Stable high-level interfaces:
  - `load_model(...)`
  - `train(...)`
  - `generate(...)`
- Keep internals replaceable (attention kernels, optimizer backends).
- Minimize hidden global state.

### 7.3 Reliability improvements

- Centralized exception messages for config/data errors.
- Reproducibility helpers (seed control, deterministic flags).

---

## 8) Memory Efficiency Strategy

### 8.1 Training memory reductions

- Mixed precision (`bf16/fp16`).
- Gradient checkpointing in transformer blocks.
- Activation recomputation toggles.
- Sequence packing to reduce padding waste.

### 8.2 Model/runtime optimizations

- Optional fused kernels and optimized attention backends.
- Parameter-efficient fine-tuning adapters (LoRA-ready extension points).
- CPU/NVMe offload compatibility path (for constrained GPUs).

### 8.3 Inference efficiency

- KV-cache optimized generation.
- Batch generation utilities.
- Optional quantization/export path (8-bit/4-bit via supported backends).

---

## 9) Testing Strategy

### 9.1 Test pyramid

- **Unit tests**
  - tensor shapes and mask correctness
  - tokenizer edge cases
  - scheduler and optimizer parameter grouping
- **Integration tests**
  - tiny end-to-end training run (toy dataset)
  - checkpoint save/load/resume consistency
  - generation determinism on fixed seed
- **Regression tests**
  - parity baselines for known prompts/checkpoints

### 9.2 CI matrix

- Python versions (e.g., 3.10–3.12).
- CPU-first required checks; optional GPU nightly workflow.
- Required gates:
  - lint
  - type check
  - unit/integration tests
  - docs build

### 9.3 Coverage and quality targets

- >=80% line coverage in core package.
- 100% coverage for critical numerics paths (attention masking, checkpoint conversion).

---

## 10) Documentation Plan

### 10.1 Core docs to add

- `README.md` refresh with quickstart and feature matrix.
- `docs/architecture.md` for model/training internals.
- `docs/training.md` with single/multi-GPU examples.
- `docs/datasets.md` for ingestion formats and preprocessing.
- `docs/api.md` for public API.
- `docs/migration_from_tf.md` for legacy users.

### 10.2 Developer documentation

- `CONTRIBUTING.md` with style, testing, and release process.
- `SECURITY.md` with vulnerability reporting guidance.
- `CODE_OF_CONDUCT.md` (community health).

### 10.3 Examples

- Minimal inference example.
- Fine-tuning example on small corpus.
- Distributed training launch example.
- Checkpoint conversion example (TF -> PyTorch).

---

## 11) Release and Distribution Plan

### 11.1 PyPI

- Adopt `pyproject.toml` with optional extras:
  - `train`
  - `dev`
  - `docs`
- Automated semantic versioning and changelog generation.
- Signed artifacts where feasible.

### 11.2 Docker

- Multi-stage Dockerfile:
  - runtime image
  - training image
- GPU-enabled image variant with pinned CUDA/cuDNN compatibility.
- Publish images via GitHub Container Registry.

### 11.3 Hugging Face Hub

- Export utility for model + tokenizer + config.
- Standardized model card template.
- Optional automated push pipeline on tagged release.

---

## 12) Suggested Timeline (10 Weeks)

### Weeks 1–2: Foundation

- Modern packaging (`pyproject.toml`), lint/type/test scaffolding, CI.
- Repository restructuring into `src/` layout.
- Baseline documentation refresh.

### Weeks 3–5: Core Migration

- PyTorch model implementation.
- Generation utilities.
- TF checkpoint conversion + parity tests.

### Weeks 6–8: Data + Training

- Dataset abstractions, collators, caching/streaming.
- Full training loop with AMP/DDP/checkpointing.
- Basic evaluation and metrics reporting.

### Weeks 9–10: Hardening + Release

- Test coverage expansion and performance profiling.
- Documentation completion.
- First modern release to PyPI + Docker + HF Hub.

---

## 13) Risk Register and Mitigations

- **Risk:** TF-to-PyTorch parity drift.
  - **Mitigation:** golden prompt tests + checkpoint-level numerical checks.
- **Risk:** Training instability on mixed precision.
  - **Mitigation:** conservative defaults, scaler monitoring, fallback modes.
- **Risk:** Scope creep (features vs. maintainability).
  - **Mitigation:** enforce milestone-based acceptance criteria.

---

## 14) Recommended Immediate Next Steps

1. Merge this roadmap and align maintainers on priorities.
2. Open milestone issues for each phase with explicit acceptance criteria.
3. Land Phase 1 (packaging + CI + structure) before model migration.
4. Start PyTorch parity implementation behind a feature branch.

---

## Conclusion

Modernizing GPT-2 is highly feasible with a phased plan that prioritizes compatibility, quality, and maintainability. Completing this roadmap will transform the repository from a legacy inference-oriented project into a contemporary, trainable, and production-ready foundation suitable for research and practical deployment in 2026.
