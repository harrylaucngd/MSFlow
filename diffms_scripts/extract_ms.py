import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm


data_dir = "../spec_files"   # <-- set your folder here
bin_size = 0.1                  # m/z resolution
max_mz = 2960                   # maximum m/z range for vector length
min_intensity_fraction = 0.01   # filter peaks <1% base peak


def spectrum_to_vector(peaks, bin_size=0.1, max_mz=1500):
    """
    Convert list of (mz, intensity) to a fixed-length binned vector.
    """
    num_bins = int(max_mz / bin_size)
    vec = np.zeros(num_bins)

    if len(peaks) == 0:
        return vec

    mzs = np.array([p[0] for p in peaks])
    intensities = np.array([p[1] for p in peaks])

    # Normalize intensities
    intensities = intensities / intensities.max()

    # Remove noise
    mask = intensities >= min_intensity_fraction
    mzs = mzs[mask]
    intensities = intensities[mask]

    # Bin assignment
    for mz, inten in zip(mzs, intensities):
        idx = int(mz / bin_size)
        if 0 <= idx < num_bins:
            vec[idx] +=  inten     #max aggregation: vec[idx] = max(vec[idx], inten)

    return vec

def parse_ms_file(filepath):
    """
    Extract smiles, spectrumid, and peak list from a GNPS-style .ms file.
    The peak lines are recorded ONLY AFTER a '>ms2peaks' marker.
    Returns: (smiles, specid, peaks)
    """
    smiles = None
    inchi = None
    peaks = []

    in_peak_section = False

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            # Extract SMILES
            if line.startswith("#smiles") or line.lower().startswith(">smiles"):
                parts = line.split()
                if len(parts) > 1:
                    smiles = parts[1]
            if line.startswith("#InChI") or line.lower().startswith("#inchi"):
                parts = line.split()
                if len(parts) > 1:
                    inchi = parts[1]

            parts = line.split()
            if len(parts) >= 2:
                try:
                    mz = float(parts[0])
                    inten = float(parts[1])
                    peaks.append((mz, inten))
                except ValueError:
                    pass  # skip invalid lines

    return smiles,inchi, peaks

rows = []

for filename in tqdm(os.listdir(data_dir)):
    if filename.endswith(".ms"):
        filepath = os.path.join(data_dir, filename)
        smiles,inchi, peaks = parse_ms_file(filepath)

        vector = spectrum_to_vector(peaks, bin_size, max_mz)

        rows.append({
            "smiles": smiles,
            "inchi": inchi,
            "specid": os.path.splitext(filename)[0],
            "raw_peaks": peaks,      # <-- NEW: store raw float tuples
            "spectrum_vector": vector
        })

df = pd.DataFrame(rows)
print(df.head())
print(df.shape)
df.dropna(inplace=True)
print(df.shape)
df.to_pickle("../data_binned_spectra.pkl")     