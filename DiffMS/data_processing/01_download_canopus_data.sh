# script adapted from MIST repo: https://github.com/samgoldman97/mist/blob/main_v2/data_processing/canopus_train/00_download_canopus_data.sh

# Original data link
# SVM_URL="https://bio.informatik.uni-jena.de/wp/wp-content/uploads/2020/08/svm_training_data.zip"

export_link="https://zenodo.org/record/8316682/files/canopus_train_export_v2.tar"

# Define target directory
target_dir="/lustre/groups/ml01/workspace/ghaith.mqawass/2025_ghaith_de_novo_design/data/ms_data"

# Create directory if it doesn't exist
mkdir -p "$target_dir"

# Move into target directory
cd "$target_dir" || exit

# Download data
wget -O canopus_train_export.tar "$export_link"

# Extract and organize
tar -xvf canopus_train_export.tar
mv canopus_train_export canopus

# Clean up
rm -f canopus_train_export.tar

echo "✅ Data successfully downloaded and extracted to: $target_dir/canopus"
