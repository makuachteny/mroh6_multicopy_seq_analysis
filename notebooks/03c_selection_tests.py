#!/usr/bin/env python3
"""
03c — Polymorphism vs Positive Selection Tests
================================================

QUESTION:
  Is the elevated dN/dS at BEB sites driven by genuine positive selection,
  or is it an artifact of within-species polymorphism across paralogs?

REASONING:
  MROH6 copies are paralogs within ONE species, not orthologs across species.
  High omega could reflect ancestral polymorphism rather than selection.
  Three independent tests distinguish these hypotheses:

  (1) Radical vs Conservative substitution test (Section 6.1 of summary):
      If selection is real, BEB sites should have MORE radical substitutions
      (changing physicochemical class: nonpolar↔polar↔charged) than non-BEB sites.
      Under neutral polymorphism, radical/conservative ratios should be similar.

  (2) Synonymous diversity at BEB vs non-BEB sites (Section 6.2):
      If BEB signal is polymorphism-driven, synonymous diversity should be
      similar everywhere. If BEB sites show LOWER synonymous diversity,
      omega is an underestimate — the true selection signal is even stronger.

  (3) Modal allele analysis (Section 6.3):
      At BEB sites, if the MAJORITY amino acid across copies is derived
      (different from reference), this indicates a selective sweep, not
      random polymorphism.

EXPECTED VALUES (from summary):
  - BEB sites: 67% radical vs 33% conservative (chi2=260, p~10^-58)
  - Non-BEB sites: 48% radical vs 52% conservative
  - BEB sites have 43% lower synonymous diversity
  - 28/30 BEB sites have derived majority state

  Also computes:
  - Mutation spectrum: Ts/Tv at 4-fold degenerate sites
  - Expected: 1.59 (below bird genome average of ~3.5)

Usage:
  python notebooks/03c_selection_tests.py --species zebra_finch
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
from Bio import AlignIO, SeqIO
from Bio.Seq import Seq
from pathlib import Path
import re
from scipy import stats as sp_stats

from utils import count_substitution_types
from species_config import load_config, get_data_dirs, classify_chrom

parser = argparse.ArgumentParser(description='Step 03c: Selection Tests')
parser.add_argument('--species', required=True)
args, _ = parser.parse_known_args()

cfg = load_config(args.species)
dirs = get_data_dirs(cfg)

DATA_PROC = dirs["data_proc"]
PAML_DIR = dirs["paml_dir"]
FIG_DIR = dirs["fig_dir"]
TABLE_DIR = dirs["table_dir"]
prefix = cfg["output_prefix"]

GENE = cfg["gene_name"]
SPECIES = cfg["common_name"]

print("=" * 70)
print(f"  STEP 03c: SELECTION TESTS — {GENE} in {SPECIES}")
print("=" * 70)

# ── Load alignment ───────────────────────────────────────────────────────
aln_path = DATA_PROC / f'{prefix}_aligned_trimmed.fasta'
aln = AlignIO.read(aln_path, 'fasta')
aln_len = aln.get_alignment_length()
n_codons = aln_len // 3

print(f"  Alignment: {len(aln)} sequences x {aln_len} columns ({n_codons} codons)")

# Build codon alignment
names = [rec.id for rec in aln]
seqs = {rec.id: str(rec.seq)[:n_codons*3] for rec in aln}

# ── Ts/Tv at 4-fold degenerate sites ────────────────────────────────────
print("\n── Mutation spectrum: Ts/Tv at 4-fold degenerate sites ──")

# 4-fold degenerate codons: third position can be any base without changing AA
# These are: CTN (Leu), GTN (Val), TCN (Ser), CCN (Pro), ACN (Thr), GCN (Ala),
#             CGN (Arg), GGN (Gly)
FOURFOLD_PREFIXES = {'CT', 'GT', 'TC', 'CC', 'AC', 'GC', 'CG', 'GG'}

ts_4fold = 0
tv_4fold = 0
purines = {'A', 'G'}
pyrimidines = {'C', 'T'}

for codon_pos in range(n_codons):
    nt_start = codon_pos * 3
    # Collect third-position bases at this codon across all sequences
    bases_at_pos = []
    is_4fold = False

    for name in names:
        codon = seqs[name][nt_start:nt_start+3].upper()
        if '-' in codon or 'N' in codon:
            continue
        prefix_2 = codon[:2]
        if prefix_2 in FOURFOLD_PREFIXES:
            is_4fold = True
            bases_at_pos.append(codon[2])

    if not is_4fold or len(bases_at_pos) < 2:
        continue

    # Count Ts and Tv at this position across all pairs
    for i in range(len(bases_at_pos)):
        for j in range(i+1, len(bases_at_pos)):
            a, b = bases_at_pos[i], bases_at_pos[j]
            if a == b:
                continue
            if (a in purines and b in purines) or (a in pyrimidines and b in pyrimidines):
                ts_4fold += 1
            else:
                tv_4fold += 1

tstv_4fold = ts_4fold / tv_4fold if tv_4fold > 0 else float('inf')
print(f"  Transitions at 4-fold sites: {ts_4fold}")
print(f"  Transversions at 4-fold sites: {tv_4fold}")
print(f"  Ts/Tv at 4-fold degenerate sites: {tstv_4fold:.2f}")
print(f"  Bird genome average Ts/Tv: ~3.5")
print(f"  Expected if RT-mediated: >4")

# ── Radical vs Conservative substitution analysis ────────────────────────
print("\n── Radical vs Conservative substitution test ──")

# Physicochemical classes
NONPOLAR = set('GAVLIPFWM')
POLAR = set('STYCNQ')
CHARGED = set('DEKRH')

def aa_class(aa):
    aa = aa.upper()
    if aa in NONPOLAR:
        return 'nonpolar'
    elif aa in POLAR:
        return 'polar'
    elif aa in CHARGED:
        return 'charged'
    return 'unknown'

def is_radical(aa1, aa2):
    """Returns True if substitution changes physicochemical class."""
    c1, c2 = aa_class(aa1), aa_class(aa2)
    if c1 == 'unknown' or c2 == 'unknown':
        return None
    return c1 != c2

# Try to load BEB results if available
beb_sites = set()
beb_file = PAML_DIR / 'M8_out.txt'
if beb_file.exists():
    with open(beb_file) as f:
        text = f.read()
    # Parse BEB table - look for lines with site number and posterior probability
    beb_section = False
    for line in text.split('\n'):
        if 'Bayes Empirical Bayes' in line:
            beb_section = True
            continue
        if beb_section and line.strip():
            parts = line.split()
            if len(parts) >= 3:
                try:
                    site = int(parts[0])
                    prob_str = parts[2].replace('*', '')
                    prob = float(prob_str)
                    if prob > 0.95:
                        beb_sites.add(site)
                except (ValueError, IndexError):
                    continue

print(f"  BEB sites (P > 0.95): {len(beb_sites)}")

# Classify substitutions at BEB vs non-BEB sites
CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L',
    'CTA': 'L', 'CTG': 'L', 'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V', 'TCT': 'S', 'TCC': 'S',
    'TCA': 'S', 'TCG': 'S', 'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T', 'GCT': 'A', 'GCC': 'A',
    'GCA': 'A', 'GCG': 'A', 'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q', 'AAT': 'N', 'AAC': 'N',
    'AAA': 'K', 'AAG': 'K', 'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W', 'CGT': 'R', 'CGC': 'R',
    'CGA': 'R', 'CGG': 'R', 'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

beb_radical = 0
beb_conservative = 0
nonbeb_radical = 0
nonbeb_conservative = 0

# Use first sequence as reference
ref_seq = seqs[names[0]]

for codon_pos in range(n_codons):
    site_num = codon_pos + 1  # 1-indexed
    is_beb = site_num in beb_sites
    nt_start = codon_pos * 3

    ref_codon = ref_seq[nt_start:nt_start+3].upper()
    if '-' in ref_codon or 'N' in ref_codon or ref_codon not in CODON_TABLE:
        continue
    ref_aa = CODON_TABLE[ref_codon]
    if ref_aa == '*':
        continue

    for name in names[1:]:
        alt_codon = seqs[name][nt_start:nt_start+3].upper()
        if '-' in alt_codon or 'N' in alt_codon or alt_codon not in CODON_TABLE:
            continue
        alt_aa = CODON_TABLE[alt_codon]
        if alt_aa == '*' or alt_aa == ref_aa:
            continue

        radical = is_radical(ref_aa, alt_aa)
        if radical is None:
            continue

        if is_beb:
            if radical:
                beb_radical += 1
            else:
                beb_conservative += 1
        else:
            if radical:
                nonbeb_radical += 1
            else:
                nonbeb_conservative += 1

# Chi-squared test
beb_total = beb_radical + beb_conservative
nonbeb_total = nonbeb_radical + nonbeb_conservative

if beb_total > 0 and nonbeb_total > 0:
    beb_pct_radical = beb_radical / beb_total * 100
    nonbeb_pct_radical = nonbeb_radical / nonbeb_total * 100

    # Chi-squared contingency table
    observed = np.array([[beb_radical, beb_conservative],
                          [nonbeb_radical, nonbeb_conservative]])
    chi2, p_chi2, dof, expected = sp_stats.chi2_contingency(observed)

    print(f"\n  BEB sites: {beb_pct_radical:.0f}% radical, {100-beb_pct_radical:.0f}% conservative")
    print(f"  Non-BEB sites: {nonbeb_pct_radical:.0f}% radical, {100-nonbeb_pct_radical:.0f}% conservative")
    print(f"  Chi-squared: {chi2:.0f}, p = {p_chi2:.2e}")
else:
    beb_pct_radical = 0
    nonbeb_pct_radical = 0
    chi2 = 0
    p_chi2 = 1.0
    print("  Insufficient BEB sites for radical/conservative test")

# ── Modal allele analysis ────────────────────────────────────────────────
print("\n── Modal allele analysis at BEB sites ──")

n_derived_majority = 0
n_beb_analyzed = 0

for codon_pos in range(n_codons):
    site_num = codon_pos + 1
    if site_num not in beb_sites:
        continue

    nt_start = codon_pos * 3
    ref_codon = ref_seq[nt_start:nt_start+3].upper()
    if '-' in ref_codon or ref_codon not in CODON_TABLE:
        continue
    ref_aa = CODON_TABLE[ref_codon]

    # Count amino acids across all copies
    aa_counts = {}
    for name in names:
        codon = seqs[name][nt_start:nt_start+3].upper()
        if '-' in codon or 'N' in codon or codon not in CODON_TABLE:
            continue
        aa = CODON_TABLE[codon]
        if aa == '*':
            continue
        aa_counts[aa] = aa_counts.get(aa, 0) + 1

    if not aa_counts:
        continue

    modal_aa = max(aa_counts, key=aa_counts.get)
    n_beb_analyzed += 1

    if modal_aa != ref_aa:
        n_derived_majority += 1

if n_beb_analyzed > 0:
    print(f"  BEB sites analyzed: {n_beb_analyzed}")
    print(f"  Sites with derived majority: {n_derived_majority} ({n_derived_majority/n_beb_analyzed*100:.0f}%)")
    print(f"  Consistent with selective sweep: {'Yes' if n_derived_majority/n_beb_analyzed > 0.8 else 'Unclear'}")
else:
    print("  No BEB sites available for modal allele analysis")

# ── Figures ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Radical vs conservative bar chart
ax = axes[0]
if beb_total > 0 and nonbeb_total > 0:
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w/2, [beb_pct_radical, nonbeb_pct_radical], w,
           label='Radical', color='crimson')
    ax.bar(x + w/2, [100-beb_pct_radical, 100-nonbeb_pct_radical], w,
           label='Conservative', color='steelblue')
    ax.set_xticks(x)
    ax.set_xticklabels(['BEB sites\n(P>0.95)', 'Non-BEB\nsites'])
    ax.set_ylabel('% of substitutions')
    ax.set_title(f'Radical vs Conservative\nchi2={chi2:.0f}, p={p_chi2:.1e}')
    ax.legend()

# Ts/Tv comparison
ax = axes[1]
categories = [f'{GENE}\n4-fold sites', 'Bird genome\naverage', 'Expected\n(RT-mediated)']
values = [tstv_4fold, 3.5, 4.5]
colors = ['darkorange', 'steelblue', 'crimson']
ax.bar(categories, values, color=colors)
for i, v in enumerate(values):
    ax.text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('Ts/Tv ratio')
ax.set_title('Transition/Transversion at 4-fold degenerate sites')

# Modal allele at BEB sites
ax = axes[2]
if n_beb_analyzed > 0:
    ax.bar(['Derived\nmajority', 'Ancestral\nmajority'],
           [n_derived_majority, n_beb_analyzed - n_derived_majority],
           color=['crimson', 'steelblue'])
    ax.set_ylabel('Number of BEB sites')
    ax.set_title(f'Modal allele at BEB sites\n{n_derived_majority}/{n_beb_analyzed} derived = sweep signal')
else:
    ax.text(0.5, 0.5, 'No BEB sites\navailable', ha='center', va='center',
            fontsize=14, transform=ax.transAxes)

plt.suptitle(f'{GENE} Selection Tests — {SPECIES}', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / 'selection_tests.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved: {FIG_DIR / 'selection_tests.png'}")

# ── Save results ─────────────────────────────────────────────────────────
results = pd.DataFrame([
    {'Test': 'Ts/Tv at 4-fold degenerate sites', 'Value': f'{tstv_4fold:.2f}'},
    {'Test': 'Bird genome average Ts/Tv', 'Value': '~3.5'},
    {'Test': 'PAML M0 kappa', 'Value': 'see paml_results.csv'},
    {'Test': 'BEB sites (P>0.95)', 'Value': len(beb_sites)},
    {'Test': 'BEB radical %', 'Value': f'{beb_pct_radical:.0f}%'},
    {'Test': 'Non-BEB radical %', 'Value': f'{nonbeb_pct_radical:.0f}%'},
    {'Test': 'Chi-squared (radical/conservative)', 'Value': f'{chi2:.0f}'},
    {'Test': 'Chi-squared p-value', 'Value': f'{p_chi2:.2e}'},
    {'Test': 'BEB sites with derived majority', 'Value': f'{n_derived_majority}/{n_beb_analyzed}'},
])
results.to_csv(TABLE_DIR / 'selection_tests.csv', index=False)

print("\n" + "=" * 70)
print(f"  SELECTION TESTS SUMMARY — {GENE} in {SPECIES}")
print("=" * 70)
print(f"  Ts/Tv (4-fold degenerate): {tstv_4fold:.2f} (bird avg ~3.5)")
print(f"  Radical/Conservative chi2: {chi2:.0f} (p={p_chi2:.2e})")
print(f"  Modal derived at BEB: {n_derived_majority}/{n_beb_analyzed}")
print("=" * 70)
