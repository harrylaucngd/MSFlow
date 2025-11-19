import os
from tqdm import tqdm

min_mz = float("inf")
max_mz = 0
directory = '../spec_files/'  # path to mass spec files dir

for fname in tqdm(os.listdir(directory)):
    if not fname.endswith(".ms"):
        continue

    with open(os.path.join(directory, fname)) as f:
        in_peaks_section = False

        for line in f:
            line = line.strip()

            # Detect start of peak block
            if line.lower() == ">ms2peaks":
                in_peaks_section = True
                continue

            # Ignore lines until >ms2peaks appears
            if not in_peaks_section:
                continue

            # Once inside ms2peaks: expect "mz intensity"
            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                mz = float(parts[0])
                intensity = float(parts[1])
            except ValueError:
                continue

            # Skip invalid peaks
            if mz <= 0 or intensity <= 0:
                continue

            # Update global bounds
            min_mz = min(min_mz, mz)
            max_mz = max(max_mz, mz)

# Fallback if no peaks found
if min_mz == float("inf"):
    min_mz = 0

print(min_mz, max_mz)
