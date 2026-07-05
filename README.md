# EnzOpt-ML: Active-Learning Pipeline for pLM-Driven Enzyme Thermostability Optimization

An enterprise-grade, end-to-end computational biology platform leveraging Pre-trained Protein Language Models (pLMs) and automated Molecular Dynamics (MD) simulations to predict, engineer, and validate high-stability enzyme variants.

## 🧬 Platform Architecture (DBTL Cycle)

This platform implements an industrial **Design-Build-Test-Learn (DBTL)** engineering cycle entirely in silicon:

1. **DESIGN (Sequence Space ML):** Zero-shot variant effect prediction using HuggingFace-hosted Protein Language Models (`ESM-2`). Computes masked marginal log-likelihood ratios across the entire mutational landscape of a target wild-type sequence.
2. **BUILD (Structural In Silico Generation):** Automated high-throughput 3D structure generation using `ESMFold`/`AlphaFold 3` orchestrators for top-tier candidate sequences.
3. **TEST (Physics-Based MD Validation):** Automated physical validation layer using `OpenMM`. Executes equilibrium simulations under simulated thermal stress to calculate Root-Mean-Square Fluctuation (RMSF) and Solvent-Accessible Surface Area (SASA) of the catalytic triad.
4. **LEARN (Data Layer & Analytics UI):** Aggregates relational sequence-to-structure-to-physics metrics inside an `SQLite` data warehouse, surfacing candidate prioritizations via an interactive `Streamlit` analytical dashboard.

## 🚀 Repository Blueprint
```text
enzopt-ml/
├── config/             # Centralized YAML configuration tracking
├── data/               # Structured local data warehouses (Raw, Processed, Results)
├── logs/               # Standardized runtime system telemetry
├── src/                # Modular operational pipeline source code
│   ├── phase1_design/  # pLM inference & mutation matrix calculations
│   ├── phase2_build/   # High-throughput 3D folding orchestration
│   ├── phase3_test/    # OpenMM simulation & trajectory analysis
│   ├── phase4_learn/   # Relational DB ingestion & analytical UI
│   └── utils/          # Cross-module shared utilities
└── tests/              # Automated unit testing frameworks
🛠️ Current Status
[x] Phase 0: Repository Infrastructure & Telemetry Freeze

[ ] Phase 1: Sequence Design & Zero-Shot Inference Engine [In Progress]

[ ] Phase 2: High-Throughput Structural Generation

[ ] Phase 3: Automated Molecular Dynamics Validation

[ ] Phase 4: Relational Orchestration & Analytics UI
