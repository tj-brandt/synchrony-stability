# src/prepare_assets.py
"""
Prepares the necessary assets for running style adaptation simulations on the
publicly available external datasets.

For each specified dataset, this script performs the following steps:
1.  Reads all conversational utterances from the raw data files.
2.  Vectorizes each utterance into an 8-dimensional style vector using the
    lightweight, dependency-free vectorizer.
3.  Fits a scikit-learn `StandardScaler` to the full set of raw vectors.
4.  Calculates a "persona centroid," which is the mean of the standardized
    (z-scored) style vectors.
5.  Saves the fitted scaler (`.joblib`) and the centroid (`.npy`) to the
    `data/assets/` directory for use in the simulation script.
"""

import pandas as pd
import numpy as np
import sys
import json
import joblib
import argparse
import csv
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# --- Setup Project Paths ---
# This allows the script to be run from the root directory via run.sh
try:
    from backend.style_vector.vectorizer import vectorize_text
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    from backend.style_vector.vectorizer import vectorize_text

# --- Constants ---
DATA_DIR = Path("data")
ASSETS_DIR = DATA_DIR / "assets"
RAW_EXTERNAL_DIR = DATA_DIR / "external_raw"
ALL_DATASETS = ["daily_dialog", "persona_chat", "empathetic_dialogues"]

def get_external_texts(dataset_name: str) -> list:
    """Extracts a flat list of all text utterances from a given raw dataset."""
    dataset_path = RAW_EXTERNAL_DIR / dataset_name
    all_texts = []
    if dataset_name == "daily_dialog":
        # Dialogues are concatenated with '__eou__' as a separator.
        for txt_file in sorted(dataset_path.glob("**/dialogues_text.txt")):
            all_texts.extend(txt_file.read_text(encoding='utf-8').strip().split('__eou__'))
    elif dataset_name == "persona_chat":
        json_file = dataset_path / "personachat_self_original.json"
        if json_file.exists():
            with json_file.open('r', encoding='utf-8') as f:
                data = json.load(f)
                for split in data.values():
                    for conversation in split:
                        for utterance in conversation.get('utterances', []):
                            all_texts.extend(utterance.get('history', []))
    elif dataset_name == "empathetic_dialogues":
        for csv_file in sorted(dataset_path.glob("**/empatheticdialogues/*.csv")):
            try:
                # Read CSV, skipping potential bad lines in the dataset.
                df = pd.read_csv(csv_file, quoting=csv.QUOTE_MINIMAL, on_bad_lines='skip')
                all_texts.extend(df['utterance'].dropna().tolist())
            except Exception as e:
                print(f"Warning: Could not process {csv_file}: {e}")
    
    # Return a clean list of non-empty strings.
    return [text.strip() for text in all_texts if text.strip()]

def build_external_assets(dataset_name: str):
    """Builds and saves the scaler and centroid for a single external dataset."""
    print(f"\n--- Building assets for external dataset: {dataset_name} ---")
    all_texts = get_external_texts(dataset_name)
    if not all_texts:
        print(f"  [Warning] No text extracted for '{dataset_name}'. Skipping.")
        return

    print(f"Vectorizing {len(all_texts):,} utterances from '{dataset_name}'...")
    raw_vectors = np.array([vectorize_text(text) for text in tqdm(all_texts, unit=" utterance")])

    # Fit a scaler to the raw vector dimensions.
    scaler = StandardScaler().fit(raw_vectors)
    scaler_path = ASSETS_DIR / f"{dataset_name}_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to: {scaler_path}")

    # Transform the vectors and compute the mean to find the centroid.
    standardized_vectors = scaler.transform(raw_vectors)
    centroid = np.mean(standardized_vectors, axis=0)
    centroid_path = ASSETS_DIR / f"{dataset_name}_persona_centroid.npy"
    np.save(centroid_path, centroid)
    print(f"Saved centroid to: {centroid_path}")

def main(args):
    """Main function to orchestrate asset preparation for specified datasets."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    datasets_to_process = args.datasets
    
    # Default to all datasets if none are specified.
    if not datasets_to_process or "all" in datasets_to_process:
        datasets_to_process = ALL_DATASETS
        
    for ds in datasets_to_process:
        if ds in ALL_DATASETS:
            build_external_assets(ds)
        else:
            print(f"[Warning] Unknown dataset '{ds}' specified. Skipping.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare style vectorization assets (scalers and centroids) for public external datasets."
    )
    parser.add_argument(
        "datasets",
        nargs='*',
        help=f"A space-separated list of datasets to process. Choices: {ALL_DATASETS}. "
             "If omitted, all datasets will be processed."
    )
    args = parser.parse_args()
    main(args)