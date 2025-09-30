# src/generate_figures.py
"""
Generates all figures for the paper from the processed simulation and
analysis data.

This script reads summary CSV and JSON files from the `reports/` directory
and produces all frontier plots, bar charts, and facet grids as PDF files
in the `reports/figures/` directory.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys
import argparse
from pathlib import Path
from typing import List
from math import ceil
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection

# --- Setup Project Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
REPORTS_DIR = Path("reports")
SIMULATION_OUTPUT_DIR = REPORTS_DIR / "simulation_outputs"
ANALYSIS_OUTPUT_DIR = REPORTS_DIR / "statistical_analysis"
LLM_REPLAY_DIR = Path("llm_replay")
FIG_DIR = REPORTS_DIR / "figures" 

# --- Constants ---
EXTERNAL_DATASETS = ["daily_dialog", "persona_chat", "empathetic_dialogues"]

# -----------------------------
# Accessible, cohesive palette (Okabe–Ito)
# -----------------------------
OI = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "yellow": "#F0E442",
    "purple": "#CC79A7",
    "red":    "#D55E00",
    "black":  "#000000",
    "grey":   "#7F7F7F",
}
NEUTRAL = {
    "bg": "#FAFAFC",
    "grid": "#E6E8EF",
    "axis": "#2C2C2C",
    "text": "#111111",
    "muted": "#B9BDC9",
}

POLICY_PALETTE = {
    "Static Baseline": "#9AA0A6",
    "Uncapped": OI["red"],
    "Cap": OI["orange"],
    "EMA": OI["blue"],
    "Dead-band": OI["purple"],
    "Hybrid": OI["green"],
    "Hybrid+Radius": "#30B0A6",
    "Hybrid+Cache": OI["yellow"],
    "Echo Ceiling": "#34495E",
}

MODEL_PALETTE = {
    "GPT-4.1 nano": OI["blue"],
    "Claude Sonnet 4": OI["orange"],
    "Claude-Sonnet-4": OI["orange"],
    "Claude-Sonnet-4.1": OI["orange"],
}

POLICY_NORMALIZE = {
    "Cap (0.25)": "Cap",
    "EMA (α=0.5)": "EMA",
    "EMA (alpha=0.5)": "EMA",
    "Dead-Band (ε=0.1)": "Dead-band",
    "Dead-Band (eps=0.1)": "Dead-band",
    "Hybrid (EMA+Cap)": "Hybrid",
}

def _normalize_policy_names(df, col="policy"):
    if col in df.columns:
        df[col] = (df[col]
                   .replace(POLICY_NORMALIZE)
                   .replace({"  ": " "}, regex=True))
    return df

# ----------------------------------
# Theme
# ----------------------------------
def setup_plot_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False,
        "axes.unicode_minus": True,
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.title_fontsize": 12,
        "axes.grid": True,
        "grid.color": NEUTRAL["grid"],
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "axes.edgecolor": NEUTRAL["grid"],
        "axes.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "figure.autolayout": False,
    })

def _soften(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(NEUTRAL["grid"])
    ax.spines["bottom"].set_color(NEUTRAL["grid"])
    ax.tick_params(axis="both", which="both", length=0)

def _apply_labels(ax, title=None, xlabel=None, ylabel=None, title_pad=10, label_pad=6, **kwargs):
    if title is not None:
        ax.set_title(title, pad=title_pad, **kwargs)
    if xlabel is not None:
        ax.set_xlabel(xlabel, labelpad=label_pad)
    if ylabel is not None:
        ax.set_ylabel(ylabel, labelpad=label_pad)

def _legend_inside(ax, handles, labels, loc, title=None, ncol=1, align="left"):
    leg = ax.legend(handles, labels, title=title, loc=loc,
                    frameon=False, ncol=ncol, borderaxespad=0.6)
    if leg:
        leg.get_title().set_fontweight('bold')
    if align == "left" and leg and leg._legend_box:
        for t in leg.get_texts():
            t.set_ha("left")
            t.set_position((5, 0))
    return leg

def _scatter_aesthetics(ax):
    for p in ax.collections:
        p.set_edgecolor("#FFFFFF")
        p.set_linewidth(0.8)
        p.set_alpha(0.95)

def _value_labels_on_bars(ax, fmt="{:.2f}", dy=0.01):
    """Add numeric labels on bars; dy is fraction of y-range."""
    y_min, y_max = ax.get_ylim()
    off = (y_max - y_min) * dy if y_max > y_min else 0.01
    
    for p in ax.patches:
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        
        ax.text(x, y + off, fmt.format(y), ha="center", va="bottom", fontsize=9)


# ------------------------
# Plotting Functions
# ------------------------

def plot_single_frontier_on_axis(ax, summary_df: pd.DataFrame, title: str, s=160, show_axis_labels: bool=True):
    df = _normalize_policy_names(summary_df.copy())
    sns.scatterplot(
        data=df, x='stability', y='synchrony',
        hue='policy', style='policy',
        palette=POLICY_PALETTE, s=s, zorder=3, ax=ax, legend=True
    )
    _scatter_aesthetics(ax)
    if show_axis_labels:
        _apply_labels(ax, title=title,
                      xlabel="Stability (higher is better)",
                      ylabel="Synchrony (higher is better)")
    else:
        _apply_labels(ax, title=title)
    ax.grid(True, which='major', linestyle='-', linewidth=0.6)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.6)
    ax.minorticks_on()
    _soften(ax)

def plot_combined_external_frontiers(output_path: Path):
    print("- Generating combined external frontiers plot (vertical)...")

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7.0, 11.0), sharex=True, sharey=True)

    datasets_to_plot = {
        "DailyDialog": "daily_dialog",
        "Persona-Chat": "persona_chat",
        "EmpatheticDialogues": "empathetic_dialogues",
    }

    all_sync_values, all_stab_values = [], []
    for dataset_key in datasets_to_plot.values():
        try:
            summary_file = SIMULATION_OUTPUT_DIR / f"policy_summary_{dataset_key}.csv"
            df_ext = pd.read_csv(summary_file)
            all_sync_values.extend(df_ext['synchrony'].tolist())
            all_stab_values.extend(df_ext['stability'].tolist())
        except (FileNotFoundError, KeyError):
            continue

    pad = 0.05
    xlim = (min(all_stab_values) - pad, max(all_stab_values) + pad)
    ylim = (min(all_sync_values) - pad, max(all_sync_values) + pad)
    x_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    y_ticks = [-0.1, 0.0, 0.5, 1.0]

    all_handles, all_labels = [], []

    for i, (display_name, dataset_key) in enumerate(datasets_to_plot.items()):
        ax = axes[i]
        try:
            summary_file = SIMULATION_OUTPUT_DIR / f"policy_summary_{dataset_key}.csv"
            df_ext = pd.read_csv(summary_file)
            plot_single_frontier_on_axis(ax, df_ext, display_name, s=160, show_axis_labels=False)
            
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)

            h, l = ax.get_legend_handles_labels()
            if h and l:
                all_handles.extend(h)
                all_labels.extend(l)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        except FileNotFoundError:
            ax.text(0.5, 0.5, 'Data not found', ha='center', va='center', fontsize=12, color=OI["red"])
            _apply_labels(ax, title=display_name)
            _soften(ax)
            
        ax.label_outer()
        ax.set_xlabel(None)
        ax.set_ylabel(None)

    fig.supxlabel("Stability (higher is better)")
    fig.supylabel("Synchrony (higher is better)")

    from collections import OrderedDict
    by_label = OrderedDict()
    for h, l in zip(all_handles, all_labels):
        if l not in by_label: by_label[l] = h
    if by_label:
        leg = fig.legend(by_label.values(), by_label.keys(), title="Policy",
                         frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.965))
        if leg: leg.get_title().set_fontweight('bold')

    fig.suptitle("Synchrony–Stability Frontier Across Public Datasets", y=1.02, fontsize=16)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f" -> Saved combined frontiers plot: {output_path.name}")

def plot_policy_frontier(summary_df: pd.DataFrame, title: str, output_path: Path):
    df = _normalize_policy_names(summary_df.copy())
    fig, ax = plt.subplots(figsize=(7.2, 6.8))
    is_ci = {'synchrony_mean', 'synchrony_low', 'synchrony_high'}.issubset(df.columns)
    x_col, y_col = ('stability_mean', 'synchrony_mean') if is_ci else ('stability', 'synchrony')

    if is_ci:
        xerr = np.vstack([df[x_col] - df['stability_low'], df['stability_high'] - df[x_col]])
        yerr = np.vstack([df[y_col] - df['synchrony_low'], df['synchrony_high'] - df[y_col]])
        ax.errorbar(df[x_col], df[y_col], xerr=xerr, yerr=yerr, fmt='none',
                    ecolor=NEUTRAL["muted"], elinewidth=1.2, capsize=2, zorder=2)

    sns.scatterplot(data=df, x=x_col, y=y_col, hue='policy', style='policy',
                    palette=POLICY_PALETTE, s=150, zorder=3, ax=ax)
    _scatter_aesthetics(ax)
    _apply_labels(ax, title=title,
                  xlabel="Stability (higher is better)", ylabel="Synchrony (higher is better)")
    _soften(ax)
    ax.grid(True, which='major', linestyle='-', linewidth=0.6)
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.6)

    handles, labels = ax.get_legend_handles_labels()
    for handle in handles:
        if isinstance(handle, PathCollection):
            handle.set_sizes([40])

    if ax.get_legend() is not None: ax.get_legend().remove()
    by_label = dict(zip(labels, handles))
    _legend_inside(ax, list(by_label.values()), list(by_label.keys()),
                   loc="lower left", title="Policy", ncol=1, align="left")

    plt.tight_layout(rect=[0.04, 0.06, 0.98, 0.94])
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved frontier plot: {output_path.name}")

def plot_model_generalization_frontier(combined_df: pd.DataFrame, output_path: Path):
    df = _normalize_policy_names(combined_df.copy())
    fig, ax = plt.subplots(figsize=(7.3, 6.6))
    
    model_names = df['model'].unique()
    markers = {model_names[0]: 'o', model_names[1]: 'X'} if len(model_names) > 1 else {model_names[0]: 'o'}

    sns.scatterplot(data=df, x='stability', y='synchrony', hue='policy', style='model',
                    s=150, zorder=3, palette=POLICY_PALETTE, markers=markers, ax=ax, legend='full')
    _scatter_aesthetics(ax)

    for model_name, model_df in df.groupby('model'):
        frontier_policies = ['Static Baseline', 'Hybrid', 'Uncapped']
        present = [p for p in frontier_policies if p in model_df['policy'].values]
        if present:
            fdf = model_df.set_index('policy').loc[present].reset_index()
            ax.plot(fdf['stability'], fdf['synchrony'], linestyle='--', linewidth=1.2,
                    alpha=0.7, color=NEUTRAL["muted"], label=f'_{model_name} Frontier')

    _apply_labels(ax, title="Model Generalization: Synchrony–Stability Frontier",
                  xlabel="Stability (higher is better)", ylabel="Synchrony (higher is better)")
    _soften(ax)
    ax.grid(True, which='major', linestyle='-', linewidth=0.6)
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.6)
    
    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None: ax.get_legend().remove()
    
    policy_labels = df['policy'].unique().tolist()
    model_labels = df['model'].unique().tolist()
    policy_map = {lbl: h for h, lbl in zip(handles, labels) if lbl in policy_labels}
    model_map = {lbl: h for h, lbl in zip(handles, labels) if lbl in model_labels}
    final_handles = [policy_map[p] for p in policy_labels if p in policy_map] + [model_map[m] for m in model_labels if m in model_map]
    final_labels = [p for p in policy_labels if p in policy_map] + [m for m in model_labels if m in model_map]
    _legend_inside(ax, final_handles, final_labels, loc="upper left", title="Legend", ncol=1, align="left")

    plt.tight_layout(rect=[0.04, 0.04, 0.98, 0.94])
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved model generalization plot: {output_path.name}")

def plot_policy_deltas(ci_data: dict, output_path: Path):
    rows = []
    for model_key, metrics in ci_data.items():
        if not isinstance(metrics, dict): continue
        model_name = "GPT-4.1 nano" if "GPT-4.1 nano" in model_key or "OPENAI" in model_key else "Claude Sonnet 4"
        for metric, stats in metrics.items():
             if metric in ['synchrony', 'stability', 'coherence']:
                rows.append([model_name, metric.capitalize(), stats["mean"], stats["ci_lo"], stats["ci_hi"]])

    plot_df = pd.DataFrame(rows, columns=["model", "metric", "mean", "ci_lo", "ci_hi"])
    x_order = ["Synchrony", "Stability", "Coherence"]
    hue_order = sorted(plot_df["model"].unique().tolist())
    
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    sns.barplot(data=plot_df, x='metric', y='mean', hue='model', order=x_order,
                hue_order=hue_order, palette=MODEL_PALETTE, ax=ax, edgecolor="none", zorder=3)

    num_metrics = len(x_order)
    num_models = len(hue_order)
    dodge_width = 0.8
    bar_width = dodge_width / num_models
    for i, metric in enumerate(x_order):
        for j, model in enumerate(hue_order):
            data_row = plot_df[(plot_df['metric'] == metric) & (plot_df['model'] == model)]
            if data_row.empty: continue
            
            group_start_x = i - dodge_width / 2
            bar_offset = j * bar_width
            bar_center_x = group_start_x + bar_offset + (bar_width / 2)
            
            ci_lo, ci_hi = data_row['ci_lo'].iloc[0], data_row['ci_hi'].iloc[0]
            ax.vlines(bar_center_x, ci_lo, ci_hi, colors=NEUTRAL["axis"], linewidth=1.2, zorder=4)
            cap_width = 0.25 * bar_width
            ax.hlines([ci_lo, ci_hi], bar_center_x - cap_width, bar_center_x + cap_width, colors=NEUTRAL["axis"], linewidth=1.2, zorder=4)

    ax.axhline(0, color=NEUTRAL["muted"], linestyle='--', linewidth=1, zorder=2)
    _apply_labels(ax, title="Effect of Hybrid Policy (Δ vs. Uncapped) with 95% CIs",
                  xlabel="Metric", ylabel="Mean Difference (Hybrid — Uncapped)")
    _soften(ax)
    handles, labels = ax.get_legend_handles_labels()
    _legend_inside(ax, handles, labels, loc="upper left", title="Model", ncol=1, align="left")

    plt.tight_layout(rect=[0.04, 0.06, 0.98, 0.92])
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved policy deltas plot: {output_path.name}")

def plot_window_size_ablation(summary_df: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.plot(summary_df['window_size'], summary_df['mean'], marker='o', zorder=3)
    ax.fill_between(summary_df['window_size'], summary_df['mean'] - 1.96 * summary_df['sem'],
                    summary_df['mean'] + 1.96 * summary_df['sem'], alpha=0.15, zorder=2)
    _apply_labels(ax, title="Impact of Analysis Window Size on Predictive Synchrony",
                  xlabel="User History Window (turns)", ylabel="Predictive Synchrony")
    _soften(ax)
    ax.grid(True, axis='y', linestyle='-', linewidth=0.6, zorder=0)
    ax.set_xticks(sorted(summary_df['window_size'].unique()))

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved window size plot: {output_path.name}")

def plot_external_facets(df: pd.DataFrame, dataset: str, output_path: Path):
    df = _normalize_policy_names(df.copy())
    metric_specs = [
        ("synchrony", "Synchrony"),
        ("stability", "Stability"),
        ("coherence", "Coherence"),
        ("legibility", "Legibility"),
        ("register_flip_rate", "Register Flip Rate"),
        ("cache_hit_rate", "Cache Hit Rate"),
    ]
    present = [(k, lab) for k, lab in metric_specs if k in df.columns]
    n = len(present)
    if n == 0:
        print("  - No facet metrics found; skipping.")
        return

    ncols = 3 if n >= 5 else 2
    nrows = ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11.0, 3.2*nrows), sharex=False)
    axes = np.atleast_1d(axes).ravel()

    order = ["Uncapped", "Cap", "EMA", "Dead-band", "Hybrid", "Hybrid+Radius", "Hybrid+Cache", "Static Baseline"]
    present_policies = [p for p in order if p in df["policy"].values]
    xticks = np.arange(len(present_policies))

    for ax, (key, label) in zip(axes, present):
        sub = df.set_index("policy").loc[present_policies][key]
        colors = [POLICY_PALETTE.get(p, "#888888") for p in present_policies]
        ax.bar(xticks, sub.values, color=colors, edgecolor="none", zorder=3)

        title_text = f"{label} (higher is better)" if "Rate" not in label else f"{label} (lower is better)"
        _apply_labels(ax,
            title=title_text,
            xlabel=None,
            ylabel=label,
            fontsize=13
        )
        _soften(ax)
        ax.grid(True, axis='y', linestyle='-', linewidth=0.6, zorder=0)
        ax.set_xticks(xticks)
        tick_abbr = {"Hybrid+Radius": "Hybrid+Rad.", "Hybrid+Cache": "Hybrid+Cache"}
        tick_text = [tick_abbr.get(p, p) for p in present_policies]
        ax.set_xticklabels(tick_text, rotation=30, ha="center", fontsize=9)

        _value_labels_on_bars(ax, fmt="{:.2f}")

        if key == "register_flip_rate" and not np.all(sub.values == 0):
            ax.set_ylim(top=max(sub.max() * 1.15, 1e-3))

    for j in range(len(present), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Policy Facets: {dataset.replace('_', ' ').title()}", fontsize=16, y=0.98)
    fig.subplots_adjust(top=0.9, bottom=0.18, left=0.10, right=0.98, wspace=0.25, hspace=0.6)

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved facets plot: {output_path.name}")


# ----------------
# Orchestration
# ----------------
def main(figures_to_generate: List[str]):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"--- Generating Paper Figures (Output to: {FIG_DIR}) ---")
    setup_plot_style()

    if "frontier" in figures_to_generate:
        print("\n[1] Generating Main Policy Frontier Plot (Figure 1)...")
        try:
            df_summary = pd.read_csv(SIMULATION_OUTPUT_DIR / "policy_summary_original.csv")
            plot_policy_frontier(df_summary, "Synchrony–Stability Frontier",
                                 FIG_DIR / "fig_1_main_frontier.pdf")
        except FileNotFoundError:
            print("  - Skipping: `policy_summary_original.csv` not found. Run `src/run_simulations.py` first.")

    if "model_gen" in figures_to_generate:
        print("\n[2] Generating Model Generalization Frontier Plot (Figure 2)...")
        try:
            # Load the aggregated results from the LLM replay simulations
            df_openai = pd.read_json(LLM_REPLAY_DIR / "gpt4.1_nano_subset.jsonl", lines=True)
            df_openai_agg = df_openai.groupby('policy')[['synchrony', 'stability']].mean().reset_index()
            df_openai_agg['model'] = 'GPT-4.1 nano'

            df_claude = pd.read_json(LLM_REPLAY_DIR / "claude_sonnet4_subset.jsonl", lines=True)
            df_claude_agg = df_claude.groupby('policy')[['synchrony', 'stability']].mean().reset_index()
            df_claude_agg['model'] = 'Claude Sonnet 4'

            # Define the Static Baseline point programmatically.
            # This is a theoretical anchor for this specific plot and is not part of the
            # idealized vector-space simulation results. Its synchrony value is taken
            # from the analysis of the human-subjects study (Table 1 in the paper).
            static_sync = 0.079
            static_baseline = pd.DataFrame([
                {'policy': 'Static Baseline', 'synchrony': static_sync, 'stability': 1.0, 'model': m}
                for m in ['GPT-4.1 nano', 'Claude Sonnet 4']
            ])
            
            # Combine all dataframes to create the final dataset for plotting
            df_combined = pd.concat([df_openai_agg, df_claude_agg, static_baseline], ignore_index=True)

            # Generate the plot
            plot_model_generalization_frontier(df_combined, output_path=FIG_DIR / "fig_2_model_generalization.pdf")

        except FileNotFoundError:
            print("  - Skipping: One or more required .jsonl files were not found.")
        except Exception as e:
            print(f"  - An error occurred while plotting model generalization: {e}")


    if "deltas" in figures_to_generate:
        print("\n[3] Generating Policy Deltas Bar Plot (Figure 4)...")
        try:
            with open(ANALYSIS_OUTPUT_DIR / "table_2_policy_effect_ci.json", 'r') as f:
                ci_data = json.load(f)
            plot_policy_deltas(ci_data, FIG_DIR / "fig_4_policy_deltas.pdf")
        except FileNotFoundError:
            print("  - Skipping: CI data not found. Run `src/analyze_results.py` first.")

    if "window" in figures_to_generate:
        print("\n[4] Generating Window Size Ablation Plot (Figure 5)...")
        try:
            df_window = pd.read_csv(SIMULATION_OUTPUT_DIR / "window_size_summary.csv")
            plot_window_size_ablation(df_window, FIG_DIR / "fig_5_window_size.pdf")
        except FileNotFoundError:
            print("  - Skipping: Window size summary not found. Run `src/run_simulations.py` first.")

    if "external" in figures_to_generate:
        print("\n[5] Generating External Dataset Replication Plots (Figure 6)...")
        plot_combined_external_frontiers(FIG_DIR / "fig_6_external_frontiers_combined.pdf")
        
        for ds in EXTERNAL_DATASETS:
            try:
                df_facets = pd.read_csv(SIMULATION_OUTPUT_DIR / f"policy_summary_{ds}.csv")
                plot_external_facets(df_facets, ds, FIG_DIR / f"appendix_facets_{ds}.pdf")
            except FileNotFoundError:
                print(f"  - Skipping facet plot for {ds}: summary file not found.")

    print("\n--- All plotting tasks complete. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate all figures for the paper artifact.")
    parser.add_argument(
        "figures", nargs='*', default=["all"],
        help="Space-separated list of figures to generate. Choices: [frontier, model_gen, deltas, window, external, all]."
    )
    args = parser.parse_args()

    figure_list = args.figures
    if "all" in figure_list:
        figure_list = ["frontier", "model_gen", "deltas", "window", "external"]

    main(figure_list)