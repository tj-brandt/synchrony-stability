#!/bin/bash
#
# This script downloads and extracts the three public datasets required for
# the cross-corpus replication analysis. It fetches the original raw files
# that correspond to the data hosted on Hugging Face.
#
# It places them in the `data/external_raw/` directory.

# Exit immediately if a command exits with a non-zero status.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
TARGET_DIR_BASE="$SCRIPT_DIR/../external_raw"

echo "--- Fetching External Datasets ---"
echo "Target directory: $TARGET_DIR_BASE"
echo ""

# --- 1. DailyDialog ---
echo "[1/3] Processing DailyDialog..."
TARGET_DIR_DD="$TARGET_DIR_BASE/daily_dialog"
ZIP_FILE_DD="$TARGET_DIR_DD/ijcnlp_dailydialog.zip"
URL_DD="http://yanran.li/files/ijcnlp_dailydialog.zip"
FINAL_FILE_CHECK_DD="$TARGET_DIR_DD/dialogues_text.txt"

mkdir -p "$TARGET_DIR_DD"
if [ -f "$FINAL_FILE_CHECK_DD" ]; then
    echo "  -> DailyDialog already appears to be downloaded and extracted. Skipping."
else
    echo "  -> Downloading from $URL_DD..."
    wget --quiet --show-progress -O "$ZIP_FILE_DD" "$URL_DD"
    echo "  -> Extracting..."
    unzip -q "$ZIP_FILE_DD" -d "$TARGET_DIR_DD"
    mv "$TARGET_DIR_DD"/ijcnlp_dailydialog/* "$TARGET_DIR_DD"/
    rm -rf "$TARGET_DIR_DD"/ijcnlp_dailydialog
    rm "$ZIP_FILE_DD"
    echo "  -> DailyDialog downloaded and extracted successfully."
fi
echo ""

# --- 2. Persona-Chat ---
echo "[2/3] Processing Persona-Chat..."
TARGET_DIR_PC="$TARGET_DIR_BASE/persona_chat"
JSON_FILE_PC="$TARGET_DIR_PC/personachat_self_original.json"
URL_PC="https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"

mkdir -p "$TARGET_DIR_PC"
if [ -f "$JSON_FILE_PC" ]; then
    echo "  -> Persona-Chat file already exists. Skipping."
else
    echo "  -> Downloading from $URL_PC..."
    wget --quiet --show-progress -O "$JSON_FILE_PC" "$URL_PC"
    echo "  -> Persona-Chat downloaded successfully."
fi
echo ""

# --- 3. EmpatheticDialogues ---
echo "[3/3] Processing EmpatheticDialogues..."
TARGET_DIR_ED="$TARGET_DIR_BASE/empathetic_dialogues"
TAR_FILE_ED="$TARGET_DIR_ED/empatheticdialogues.tar.gz"
URL_ED="https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"
FINAL_FILE_CHECK_ED="$TARGET_DIR_ED/empatheticdialogues/train.csv"

mkdir -p "$TARGET_DIR_ED"
if [ -f "$FINAL_FILE_CHECK_ED" ]; then
    echo "  -> EmpatheticDialogues already appears to be downloaded and extracted. Skipping."
else
    echo "  -> Downloading from $URL_ED..."
    wget --quiet --show-progress -O "$TAR_FILE_ED" "$URL_ED"
    echo "  -> Extracting..."
    tar -xzf "$TAR_FILE_ED" -C "$TARGET_DIR_ED"
    rm "$TAR_FILE_ED"
    echo "  -> EmpatheticDialogues downloaded and extracted successfully."
fi
echo ""

echo "--- All external datasets are ready. ---"