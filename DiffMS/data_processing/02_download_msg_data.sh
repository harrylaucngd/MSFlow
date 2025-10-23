#!/bin/bash
# script adapted from MIST repo: https://github.com/samgoldman97/mist/blob/main_v2/data_processing/canopus_train/00_download_canopus_data.sh
# This script downloads preprocessed data from the MassSpecGym project
# Original MassSpecGym code/data: https://github.com/pluskal-lab/MassSpecGym

export_link="https://zenodo.org/records/15008938/files/msg_preprocessed.tar.gz"

# Define target directory
target_dir="/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/data/ms_data"

# Create directory if it doesn't exist
mkdir -p "$target_dir"

# Move into target directory
cd "$target_dir" || exit

# Download data
wget -O msg_preprocessed.tar.gz "$export_link"

# Extract the tar.gz file
tar -xvzf msg_preprocessed.tar.gz

# Clean up compressed file
rm -f msg_preprocessed.tar.gz

echo "✅ MassSpecGym preprocessed data downloaded and extracted to: $target_dir"
