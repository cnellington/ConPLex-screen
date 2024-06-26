import pandas as pd
from pyfaidx import Fasta
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--files", nargs="+")
parser.add_argument("--output", type=str)
args = parser.parse_args()

cols = ['proteinID', 'proteinSequence']
rows = []
for i, file in enumerate(args.files):
    print(len(args.files), i, end='\r')
    try:
        proteins = Fasta(file, read_long_names=True)
    except:
        continue
    for key, seq in proteins.items():
        rows.append([key, seq])
print()
pd.DataFrame(data=rows, columns=cols).to_csv(args.output, index=False, header=False, sep='\t')
