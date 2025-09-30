# Analysis for “Navigating the Synchrony–Stability Frontier in Adaptive Chatbots”

This repository contains the supplementary material and a fully reproducible artifact for the paper “Navigating the Synchrony–Stability Frontier in Adaptive Chatbots.” The artifact is designed to regenerate all tables and figures from the paper in a single, automated run.

| Key Information      | Details                                                                 |
| -------------------- | ----------------------------------------------------------------------- |
| **Artifact Goal**    | Reproduce all paper figures and tables                                  |
| **Main Script**      | `run.sh`                                                                 |
| **Environment**      | Python 3.10+ (see `env/requirements-lock.txt`)                           |
| **Platform**         | Unix-like (Linux, macOS, WSL)                                           |
| **Network**          | Required for first-run dataset downloads                                |
| **Expected Runtime** | Typically a few minutes on a standard laptop                            |
| **Outputs**          | Generated under `reports/`                                              |

---

## Data and Reproducibility

**Scope.** This artifact supports a *computational* framework and its derived results. Details of any live user studies are contained in the paper.

**IRB & privacy.** To protect participants, raw conversational logs are not included. The pipeline reads pre-computed numerical summaries sufficient to reproduce the validation and analysis (e.g., `data/human_logs_derived/turn_level_metrics.csv`).

---

## Artifact Structure

The `reports/` and `data/external_raw/` directories are initially empty and will be populated by the main script.

```

.
├── backend/                   # Core, reusable logic (policies, lightweight vectorizer)
├── data/
│   ├── assets/                # Scalers and centroids for each dataset (generated/provided)
│   ├── external/              # Script to fetch public datasets
│   ├── external_raw/          # (Initially empty) Target for downloaded datasets
│   └── human_logs_derived/    # Derived numerical data for validation (no raw text)
├── env/                       # Python environment files
├── llm_replay/                # Numerical outputs from LLM-in-the-loop simulations (subsets)
├── reports/                   # All generated outputs
├── src/                       # Analysis pipeline scripts
└──run.sh                     # Single script to run the entire reproduction pipeline

````

---

## Requirements

- **Python**: 3.10+
- **OS tools**: `wget`, `unzip`, and `tar` (for dataset download/unpack)
- **Disk**: A few GB free (first run downloads public corpora)

All Python dependencies are pinned in `env/requirements-lock.txt` to ensure a reproducible environment.

---

## Reproduction and Verification Steps

### 1) Set up the Environment
Create a virtual environment and install pinned dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r env/requirements-lock.txt
````

### 2) Run the Reproduction Pipeline

Make scripts executable (first run only), then execute the pipeline:

```bash
chmod +x run.sh data/external/fetch_datasets.sh
./run.sh
```

The pipeline will:

1. **Fetch External Datasets** (public corpora used for cross-corpus replication).
2. **Prepare Assets** (build scalers/centroids for the external datasets).
3. **Write Ground-Truth Results** (sim summaries saved to `reports/simulation_outputs/`).
4. **Run Statistical Analyses** (validation and policy effects).
5. **Generate Figures** (PDFs saved to `reports/figures/`).
6. **Generate License Manifest** (third-party license summary).
7. **Verify Consistency** (checks reproduced results against expected values).

### 3) Expected Outcome

On success, the terminal prints:

```
✅ ALL CONSISTENCY CHECKS PASSED.
```

All generated outputs are under `reports/`:

* `reports/figures/` — all figures (PDF)
* `reports/simulation_outputs/` — simulation summary tables (CSV)
* `reports/statistical_analysis/` — analysis outputs and manifests

---

## External Datasets

Cross-corpus replications use the following publicly available datasets, which are fetched automatically by the pipeline on first run:

* DailyDialog
* Persona-Chat
* EmpatheticDialogues

(See `data/external/fetch_datasets.sh` for source locations used by the pipeline.)

---

## Methodological Notes

* **No live API calls during reproduction.** Analyses operate on pre-generated numerical outputs (e.g., `llm_replay/*.jsonl`) and derived metrics.
* **Figures/plots.** All figures are generated with the provided scripts in `src/` and written to `reports/figures/`.
* **Policy implementations.** Adaptation policies are defined in `backend/policies/`; the lightweight style-vectorizer used for external datasets is in `backend/style_vector/`.

---

## Troubleshooting

* **Network blocked or offline?** If dataset download fails, re-run `./run.sh` after ensuring network access, or place the datasets under `data/external_raw/` in the expected structure and re-run.
* **Python/ABI mismatch.** Verify that your interpreter is Python 3.10+ and reinstall using `env/requirements-lock.txt`.
* **Permissions.** If you see permission errors, re-run `chmod +x run.sh data/external/fetch_datasets.sh`.

---

## License

* **Code**: MIT (see `LICENSE`)
* **Third-party licenses**: summarized in `LICENSES.md`; a full manifest is emitted at `reports/statistical_analysis/THIRD_PARTY_LICENSES_FULL.md` during the run.
* **Datasets**: subject to their original licenses/terms; fetched by the script from their official sources.