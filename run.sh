#!/bin/bash
#
# Main reproduction pipeline for the IUI 2026 paper artifact:
# "Navigating the Synchrony-Stability Frontier in Adaptive Chatbots"
#
# This script is self-contained and uses the executables from the local
# 'venv' directory to ensure perfect reproducibility.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration & Pre-flight Check ---
VENV_DIR="venv"

# Check if the virtual environment directory exists.
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: Virtual environment directory '$VENV_DIR' not found."
    echo "Please create the environment and install dependencies first by running:"
    echo "  python3 -m venv venv"
    echo "  pip install -r env/requirements-lock.txt"
    exit 1
fi

# Define paths to the executables inside the virtual environment
PYTHON_EXEC="$VENV_DIR/bin/python3"
PIP_LICENSES_EXEC="$VENV_DIR/bin/pip-licenses"


# --- Pipeline Execution ---
echo "================================================================="
echo "        IUI Artifact Reproduction Pipeline"
echo "================================================================="

echo -e "\n--- [Step 1/7] Fetching External Datasets ---"
./data/external/fetch_datasets.sh

echo -e "\n--- [Step 2/7] Preparing Scaler and Centroid Assets ---"
echo "NOTE: This step only builds assets for the PUBLIC external datasets."
$PYTHON_EXEC src/prepare_assets.py daily_dialog persona_chat empathetic_dialogues

echo -e "\n--- [Step 3/7] Writing Ground-Truth Simulation Results ---"
$PYTHON_EXEC src/run_simulations.py policy_suite --datasets all
$PYTHON_EXEC src/run_simulations.py window_size

echo -e "\n--- [Step 4/7] Running Statistical Analyses ---"
$PYTHON_EXEC src/analyze_results.py all

echo -e "\n--- [Step 5/7] Generating All Paper Figures ---"
$PYTHON_EXEC src/generate_figures.py all

echo -e "\n--- [Step 6/7] Generating Third-Party License Manifest ---"
# This is the corrected command block, with the unsupported argument removed.
$PIP_LICENSES_EXEC \
  --from=meta \
  --format=markdown \
  --output-file=reports/statistical_analysis/THIRD_PARTY_LICENSES_FULL.md
echo "Full license manifest saved to reports/statistical_analysis/THIRD_PARTY_LICENSES_FULL.md"

echo -e "\n--- [Step 7/7] Verifying Artifact Consistency ---"
$PYTHON_EXEC src/check_consistency.py

echo ""
echo "================================================================="
echo "✅ Reproduction Pipeline Complete!"
echo "   All steps executed successfully and all consistency checks passed."
echo "   Outputs are in the 'reports/' directory."
echo "================================================================="