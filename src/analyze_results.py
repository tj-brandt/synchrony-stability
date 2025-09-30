# src/analyze_results.py
"""
Performs statistical analyses for the adaptive chatbot study.

This script contains two main analyses:
1.  Simulation Validation: Compares the high-fidelity LLM-in-the-loop simulation
    against the ground-truth human-subjects study data to test for statistical
    equivalence (reproducing Table 3).
2.  Policy Effect Analysis: Calculates the mean effect and 95% bootstrap
    confidence intervals for the 'Hybrid' policy compared to the 'Uncapped'
    baseline across different LLM backends (reproducing Table 2).
"""

import pandas as pd
import numpy as np
import sys
import argparse
from pathlib import Path
import json

# --- Setup Project Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")
LLM_REPLAY_DIR = Path("llm_replay")
ANALYSIS_OUTPUT_DIR = REPORTS_DIR / "statistical_analysis"
GROUND_TRUTH_FILE = DATA_DIR / "human_logs_derived" / "turn_level_metrics.csv"
OPENAI_REPLAY_FILE = LLM_REPLAY_DIR / "gpt4.1_nano_subset.jsonl"
ANTHROPIC_REPLAY_FILE = LLM_REPLAY_DIR / "claude_sonnet4_subset.jsonl"

# --- Constants ---
N_BOOTSTRAPS, CI_LEVEL = 10000, 0.95
TOST_STABILITY_BOUND, TOST_SYNCHRONY_BOUND = 0.10, 0.10

# --- Helper Functions ---

def bootstrap_ci(values, n_resamples=N_BOOTSTRAPS, ci=CI_LEVEL, seed=42):
    """Calculates the mean and bootstrap confidence interval for an array of values."""
    values = np.asarray(values, dtype=float)[np.isfinite(values)]
    if values.size == 0:
        return (np.nan, np.nan, np.nan, 0)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_resamples, values.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return (float(values.mean()), lo, hi, values.size)

def format_ci_string(mean, lo, hi):
    """Formats a mean and confidence interval into a standard string."""
    if mean is None or np.isnan(mean):
        return "NaN"
    return f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]"

def tost_equivalence_test(g1, g2, bound, alpha=0.05):
    """
    Performs a Two One-Sided Test (TOST) for statistical equivalence between
    two independent samples.
    """
    g1, g2 = np.asarray(g1)[~np.isnan(g1)], np.asarray(g2)[~np.isnan(g2)]
    if g1.size == 0 or g2.size == 0:
        return "Skipped (no data)"
    try:
        from statsmodels.stats.weightstats import ttost_ind
        p = ttost_ind(g1, g2, low=-bound, upp=bound, usevar="unequal")[0]
        return f"{'Equivalent' if p < alpha else 'Not Equivalent'} (p = {p:.3f})"
    except ImportError:
        return "Skipped (statsmodels not installed)"

def run_simulation_validation():
    """Generates the data for Table 3: Simulation Framework Validation.

    Consumes:
    - data/human_logs_derived/turn_level_metrics.csv
    - llm_replay/gpt4.1_nano_subset.jsonl

    Produces:
    - reports/statistical_analysis/table_3_simulation_validation.csv
    """
    print("\n--- Running Simulation Framework Validation (reproduces Table 3) ---")
    df_truth = pd.read_csv(GROUND_TRUTH_FILE)
    truth_pid_means = df_truth[df_truth['condition'] == 'adaptive'].groupby('participant_id')[['synchrony_embedding', 'stability_score']].mean()

    df_sim = pd.read_json(OPENAI_REPLAY_FILE, lines=True)
    sim_pid_means = df_sim[df_sim['policy'] == 'Uncapped'].groupby('pid')[['synchrony_embed', 'stability']].mean()

    truth_sync_ci = bootstrap_ci(truth_pid_means['synchrony_embedding'])
    truth_stab_ci = bootstrap_ci(truth_pid_means['stability_score'])
    sim_sync_ci = bootstrap_ci(sim_pid_means['synchrony_embed'])
    sim_stab_ci = bootstrap_ci(sim_pid_means['stability'])

    tost_sync = tost_equivalence_test(truth_pid_means['synchrony_embedding'].values, sim_pid_means['synchrony_embed'].values, TOST_SYNCHRONY_BOUND)
    tost_stab = tost_equivalence_test(truth_pid_means['stability_score'].values, sim_pid_means['stability'].values, TOST_STABILITY_BOUND)

    validation_data = {
        "Metric": ["Synchrony (Embedding)", "Stability (Vector)"],
        "Human-Subjects (Observed)": [f"{ts[0]:.3f} [{ts[1]:.3f}, {ts[2]:.3f}]" for ts in [truth_sync_ci, truth_stab_ci]],
        "LLM-in-the-Loop Sim (Real-World)": [f"{ss[0]:.3f} [{ss[1]:.3f}, {ss[2]:.3f}]" for ss in [sim_sync_ci, sim_stab_ci]],
        "TOST Equivalence (p-value)": [tost_sync, tost_stab]
    }
    df_validation = pd.DataFrame(validation_data)
    print("\n" + df_validation.to_string(index=False))

    df_validation.to_csv(ANALYSIS_OUTPUT_DIR / "table_3_simulation_validation.csv", index=False)
    print(f"\n  ✅ Validation summary table saved.")

def run_policy_effect_analysis():
    """Generates the data for Table 2: Policy Effect Analysis.

    Consumes:
    - llm_replay/gpt4.1_nano_subset.jsonl
    - llm_replay/claude_sonnet4_subset.jsonl

    Produces:
    - reports/statistical_analysis/table_2_policy_effect_ci.csv
    - reports/statistical_analysis/table_2_policy_effect_ci.json
    """
    print("\n--- Running Policy Effect Analysis (reproduces Table 2) ---")
    replay_files = {"GPT-4.1 nano": OPENAI_REPLAY_FILE, "Claude Sonnet 4": ANTHROPIC_REPLAY_FILE}
    all_ci_results = {}
    for model_name, file_path in replay_files.items():
        if not file_path.exists():
            continue
        print(f"\nProcessing {model_name} results...")
        df_sim = pd.read_json(file_path, lines=True)
        metrics_to_process = ["stability", "synchrony", "coherence"]
        pid_means = df_sim.groupby(["pid", "policy"])[metrics_to_process].mean()

        wide = pid_means.unstack(level='policy')
        wide.columns = ['_'.join(col).strip() for col in wide.columns.values]

        delta_df = pd.DataFrame()
        delta_df['Δ Stability'] = wide['stability_Hybrid (EMA+Cap)'] - wide['stability_Uncapped']
        delta_df['Δ Synchrony'] = wide['synchrony_Hybrid (EMA+Cap)'] - wide['synchrony_Uncapped']
        if 'coherence_Hybrid (EMA+Cap)' in wide.columns:
            delta_df['Δ Coherence'] = wide['coherence_Hybrid (EMA+Cap)'] - wide['coherence_Uncapped']

        ci_results = {}
        for col in delta_df.columns:
            metric_name = col.split(' ')[1].lower()
            mean, lo, hi, _ = bootstrap_ci(delta_df[col].dropna().to_numpy(), seed=0)
            ci_results[metric_name] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
        all_ci_results[model_name] = ci_results

    table_rows = []
    for metric in ["Stability", "Synchrony", "Coherence"]:
        row_data = {"Metric": f"Δ {metric}"}
        for model in ["GPT-4.1 nano", "Claude Sonnet 4"]:
            stats = all_ci_results.get(model, {}).get(metric.lower(), {})
            row_data[model] = format_ci_string(stats.get('mean'), stats.get('ci_lo'), stats.get('ci_hi'))
        table_rows.append(row_data)

    df_summary = pd.DataFrame(table_rows)
    print("\n" + df_summary.to_string(index=False))

    df_summary.to_csv(ANALYSIS_OUTPUT_DIR / "table_2_policy_effect_ci.csv", index=False)
    with open(ANALYSIS_OUTPUT_DIR / "table_2_policy_effect_ci.json", "w") as f:
        json.dump(all_ci_results, f, indent=2)
    print(f"\n  ✅ Policy effect summary tables saved.")

def main(args):
    """Orchestrates the statistical analysis pipeline."""
    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.analysis_type in ["validation", "all"]:
        run_simulation_validation()
    if args.analysis_type in ["policy_effects", "all"]:
        run_policy_effect_analysis()
    print("\n--- All statistical analyses complete. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run statistical analyses for the paper.")
    parser.add_argument(
        "analysis_type",
        choices=["validation", "policy_effects", "all"],
        default="all",
        nargs="?",
        help="The specific analysis to run."
    )
    args = parser.parse_args()
    main(args)