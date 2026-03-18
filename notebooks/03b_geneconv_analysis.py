#!/usr/bin/env python3
"""
03b — Gene Conversion Analysis (GENECONV-Style Test)
======================================================

QUESTION:
  Is gene conversion actively shuffling sequences between MROH6 paralogs?
  If so, how extensive is it, and which copies are exchanging sequence?

REASONING:
  Gene conversion occurs when one gene copy "overwrites" part of another
  during recombination. This produces long runs of IDENTICAL codons between
  non-adjacent copies — longer than expected by chance from independent
  mutation alone.

  We implement the GENECONV approach (Sawyer 1989):
    1. For every pair of sequences, find the longest run of consecutive
       identical codons
    2. Establish a null distribution by permuting variable alignment
       columns (1,000 permutations)
    3. Pairs exceeding the 99th percentile of the null are candidate
       gene conversion events

EXPECTED FINDINGS (from summary doc):
  - 260 pairs (4.9%) exceed the 99th percentile null (77 codons)
  - Top clique of 9 loci shares 144-203 identical codon runs
  - Max run = 203 codons = 59% of entire coding alignment

Usage:
  python notebooks/03b_geneconv_analysis.py --species zebra_finch
"""
import sys
import argparse
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'scripts'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import AlignIO
from pathlib import Path
from collections import defaultdict

from species_config import load_config, get_data_dirs

parser = argparse.ArgumentParser(description='Step 03b: Gene Conversion Analysis')
parser.add_argument('--species', required=True)
args, _ = parser.parse_known_args()

cfg = load_config(args.species)
dirs = get_data_dirs(cfg)

DATA_PROC = dirs["data_proc"]
FIG_DIR = dirs["fig_dir"]
TABLE_DIR = dirs["table_dir"]
prefix = cfg["output_prefix"]

GENE = cfg["gene_name"]
SPECIES = cfg["common_name"]

print("=" * 70)
print(f"  STEP 03b: GENE CONVERSION ANALYSIS — {GENE} in {SPECIES}")
print("=" * 70)

# ── Load codon alignment ─────────────────────────────────────────────────
aln_path = DATA_PROC / f'{prefix}_aligned_trimmed.fasta'
if not aln_path.exists():
    print(f"  ERROR: {aln_path} not found. Run Step 01 first.")
    sys.exit(1)

aln = AlignIO.read(aln_path, 'fasta')
aln_len = aln.get_alignment_length()
n_codons = aln_len // 3

print(f"  Alignment: {len(aln)} sequences x {aln_len} columns ({n_codons} codons)")

# Extract codon-level alignment
names = [rec.id for rec in aln]
codon_seqs = {}
for rec in aln:
    seq = str(rec.seq)[:n_codons * 3]
    codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
    codon_seqs[rec.id] = codons


def longest_identical_codon_run(codons1, codons2):
    """Find the longest run of consecutive identical codons between two sequences."""
    max_run = 0
    current_run = 0
    for c1, c2 in zip(codons1, codons2):
        if c1 == c2 and '---' not in c1 and 'N' not in c1:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


# ── Compute pairwise longest identical codon runs ────────────────────────
print("\n── Computing pairwise identical codon runs ──")

n_seqs = len(names)
pair_runs = []

for i in range(n_seqs):
    for j in range(i + 1, n_seqs):
        run_len = longest_identical_codon_run(codon_seqs[names[i]], codon_seqs[names[j]])
        pair_runs.append({
            'seq1': names[i], 'seq2': names[j],
            'max_identical_run': run_len
        })

pair_df = pd.DataFrame(pair_runs)
total_pairs = len(pair_df)
print(f"  Total pairwise comparisons: {total_pairs}")
print(f"  Max identical codon run: {pair_df['max_identical_run'].max()}")
print(f"  Median identical codon run: {pair_df['max_identical_run'].median():.0f}")

# ── Permutation null distribution ────────────────────────────────────────
print("\n── Establishing null distribution (1,000 permutations) ──")

N_PERMS = 1000
rng = np.random.default_rng(42)

# Identify variable columns (columns where not all codons are identical)
codon_matrix = np.array([codon_seqs[n] for n in names])
variable_cols = []
for col in range(n_codons):
    col_codons = set(codon_matrix[:, col])
    col_codons.discard('---')
    if len(col_codons) > 1:
        variable_cols.append(col)

print(f"  Variable codon columns: {len(variable_cols)} / {n_codons}")

# For each permutation, shuffle variable columns and compute max run for a sample of pairs
# (computing all pairs for 1000 perms is too slow; sample 500 random pairs)
sample_size = min(500, total_pairs)
sample_idx = rng.choice(total_pairs, size=sample_size, replace=False)

null_max_runs = []
for perm in range(N_PERMS):
    if perm % 100 == 0:
        print(f"  Permutation {perm}/{N_PERMS}...", flush=True)

    # Shuffle variable columns independently
    perm_matrix = codon_matrix.copy()
    for col in variable_cols:
        perm_matrix[:, col] = rng.permutation(perm_matrix[:, col])

    # Compute max run for sampled pairs
    perm_max = 0
    for idx in sample_idx:
        i_idx = idx // (n_seqs - 1)  # approximate pair indexing
        j_idx = idx % (n_seqs - 1)
        if j_idx >= i_idx:
            j_idx += 1
        if i_idx < n_seqs and j_idx < n_seqs:
            run = 0
            max_run = 0
            for c in range(n_codons):
                if perm_matrix[i_idx, c] == perm_matrix[j_idx, c] and '---' not in perm_matrix[i_idx, c]:
                    run += 1
                    max_run = max(max_run, run)
                else:
                    run = 0
            perm_max = max(perm_max, max_run)
    null_max_runs.append(perm_max)

null_max_runs = np.array(null_max_runs)
pct_99 = np.percentile(null_max_runs, 99)
pct_999 = np.percentile(null_max_runs, 99.9)

print(f"\n  99th percentile null: {pct_99:.0f} codons")
print(f"  99.9th percentile null: {pct_999:.0f} codons")

# ── Identify significant pairs ───────────────────────────────────────────
sig_pairs = pair_df[pair_df['max_identical_run'] > pct_99]
n_sig = len(sig_pairs)
pct_sig = n_sig / total_pairs * 100
expected_by_chance = total_pairs * 0.01

print(f"\n  Pairs exceeding 99th percentile: {n_sig} ({pct_sig:.1f}%)")
print(f"  Expected by chance: ~{expected_by_chance:.0f}")

# ── Figure: Gene conversion results ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Distribution of max identical codon runs
ax = axes[0]
ax.hist(pair_df['max_identical_run'], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(pct_99, color='red', linestyle='--', linewidth=2,
           label=f'99th pct null ({pct_99:.0f})')
ax.set_xlabel('Longest identical codon run')
ax.set_ylabel('Number of pairs')
ax.set_title(f'Gene conversion scan\n{n_sig}/{total_pairs} pairs exceed threshold')
ax.legend()

# Null distribution
ax = axes[1]
ax.hist(null_max_runs, bins=30, color='gray', edgecolor='white', alpha=0.8)
ax.axvline(pct_99, color='red', linestyle='--', label=f'99th pct ({pct_99:.0f})')
ax.axvline(pct_999, color='darkred', linestyle='--', label=f'99.9th pct ({pct_999:.0f})')
ax.set_xlabel('Max identical codon run (permuted)')
ax.set_ylabel('Count')
ax.set_title('Permutation null distribution')
ax.legend()

# Top conversion events
ax = axes[2]
top_runs = pair_df.nlargest(20, 'max_identical_run')
ax.barh(range(len(top_runs)), top_runs['max_identical_run'].values,
        color='crimson', edgecolor='white')
ax.set_xlabel('Longest identical codon run')
ax.set_ylabel('Pair rank')
ax.set_title(f'Top 20 gene conversion candidates\n(max = {top_runs["max_identical_run"].max()} codons = {top_runs["max_identical_run"].max()/n_codons*100:.0f}% of alignment)')
ax.axvline(pct_99, color='red', linestyle='--', alpha=0.5)

plt.suptitle(f'{GENE} Gene Conversion Analysis — {SPECIES}', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / 'geneconv_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {FIG_DIR / 'geneconv_analysis.png'}")

# ── Save results ─────────────────────────────────────────────────────────
geneconv_summary = pd.DataFrame([
    {'Metric': 'Total pairwise comparisons', 'Value': total_pairs},
    {'Metric': '99th percentile null (codons)', 'Value': f'{pct_99:.0f}'},
    {'Metric': '99.9th percentile null (codons)', 'Value': f'{pct_999:.0f}'},
    {'Metric': 'Pairs exceeding 99th pct', 'Value': f'{n_sig} ({pct_sig:.1f}%)'},
    {'Metric': 'Expected by chance', 'Value': f'~{expected_by_chance:.0f}'},
    {'Metric': 'Max identical codon run', 'Value': pair_df['max_identical_run'].max()},
    {'Metric': 'Max run as % of alignment', 'Value': f'{pair_df["max_identical_run"].max()/n_codons*100:.1f}%'},
])
geneconv_summary.to_csv(TABLE_DIR / 'geneconv_summary.csv', index=False)
sig_pairs.to_csv(TABLE_DIR / 'geneconv_significant_pairs.csv', index=False)

print("\n" + "=" * 70)
print(f"  GENE CONVERSION ANALYSIS SUMMARY — {GENE} in {SPECIES}")
print("=" * 70)
print(f"  Total pairs: {total_pairs}")
print(f"  99th pct null: {pct_99:.0f} codons")
print(f"  Significant pairs: {n_sig} ({pct_sig:.1f}%)")
print(f"  Max identical run: {pair_df['max_identical_run'].max()} codons")
print("=" * 70)
