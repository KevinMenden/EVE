"""
Extract the best MSA form a EVcouplings align run
"""

import os
import glob
import pandas as pd

output_path = "/home/kevin/qbic/variant_effects/couplings_test/output"
prefix = "PTEN"

# Collect the different runs

results = glob.glob(output_path + os.sep + prefix + "*.done")
results = [os.path.basename(x) for x in results]
results = [x.replace(".done", "") for x in results]

best_msa = None
seq_no_cutoff = 100000
seq_cov_cutoff_factor = 0.8

msa_list = []
for res in results:
    stats_path = os.path.join(
        output_path, res, "align", res + "_alignment_statistics.csv"
    )
    msa_path = os.path.join(output_path, res, "align", res + ".a2m")

    msa = pd.read_csv(stats_path)
    msa_list.append(msa)

    # If no MSA exists, use this one as best MSA
    if best_msa is None:
        best_msa = msa
        continue

    if (
        int(msa["num_cov"]) > int(best_msa["num_cov"])
        and int(msa["num_seqs"]) <= seq_no_cutoff
    ):
        best_msa = msa

print(best_msa["prefix"])
