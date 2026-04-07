#!/usr/bin/env python3
"""
03 — dN/dS Analysis (Generalized, Config-Driven)
==================================================

QUESTION:
  Are the MROH gene copies under natural selection, or are they evolving
  neutrally (drifting as pseudogenes)? Specifically:
    - Are most copies under PURIFYING selection (omega < 1)?
      → Copies are functional and constrained
    - Are copies evolving NEUTRALLY (omega ~ 1)?
      → Copies are pseudogenes, free from selection
    - Is there evidence for POSITIVE selection at specific sites (omega > 1)?
      → Some amino acid positions are being actively optimized

REASONING:
  The ratio of nonsynonymous (amino acid-changing) to synonymous (silent)
  substitution rates — dN/dS or "omega" — is the gold standard for
  detecting selection at the molecular level:

    omega < 1: Purifying selection (most mutations are harmful → removed)
    omega = 1: Neutral evolution (no constraint on amino acid changes)
    omega > 1: Positive selection (new amino acids are favored)

  We use TWO complementary methods:

  (A) Pairwise Nei-Gojobori (1986):
      For every pair of sequences, count synonymous vs nonsynonymous
      differences at each codon. Apply JC correction. This gives dN, dS,
      and omega for each pair — revealing the DISTRIBUTION of selection
      pressures across all copy pairs.

  (B) PAML codeml site models:
      Fit increasingly complex models to the alignment:
        M0: One omega for all sites (baseline)
        M1a: Two classes — omega=0 (purifying) and omega=1 (neutral)
        M2a: Three classes — adds omega>1 (positive selection) to M1a
        M7: Beta distribution of omega across sites (flexible null)
        M8: Beta + omega>1 class (positive selection test vs M7)
      Likelihood Ratio Tests (LRT):
        M1a vs M2a: Is there a class of sites with omega > 1?
        M7 vs M8:   Same question with a more flexible null model.
      If LRT is significant (p < 0.05), positive selection is detected.

  We subsample to ~40 sequences for PAML (computational constraint) using
  stratified sampling to represent all chromosome classes.

EXPECTED FINDINGS:
  - Bird functional genes: omega ~ 0.15 (strong purifying selection)
  - Duplicated gene average: omega ~ 0.40 (relaxed constraint)
  - MROH copies: omega > 0.40 suggests relaxed/positive selection
  - If M0 global omega > 1: overall more nonsynonymous than synonymous
    changes, consistent with diversifying selection or pseudogenization

FINDINGS (Zebra Finch):
  - M0 global omega = 1.674 → elevated above bird average
  - M1a vs M2a LRT significant → evidence for positive selection sites
  - M7 vs M8 LRT significant → confirmed with flexible baseline
  - Median pairwise omega ~ 0.4 (purifying), but tail extends to omega > 1

Usage:
  python notebooks/03_dnds_analysis.py --species melospiza_georgiana
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
from Bio import SeqIO, AlignIO, Phylo
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from pathlib import Path
import subprocess
import re

from utils import compute_pairwise_dnds
from species_config import load_config, get_data_dirs, classify_chrom

parser = argparse.ArgumentParser(description='Step 03: dN/dS Analysis')
parser.add_argument('--species', required=True)
args, _ = parser.parse_known_args()

cfg = load_config(args.species)
dirs = get_data_dirs(cfg)

DATA_PROC = dirs["data_proc"]
PAML_DIR = dirs["paml_dir"]
FIG_DIR = dirs["fig_dir"]
TABLE_DIR = dirs["table_dir"]
prefix = cfg["output_prefix"]

sns.set_context('notebook')
sns.set_style('whitegrid')

BIRD_AVG_DNDS = 0.15
DUPLICATED_GENE_DNDS = 0.40
GENE = cfg["gene_name"]
SPECIES = cfg["common_name"]

print("=" * 70)
print(f"  STEP 03: dN/dS SELECTION ANALYSIS — {GENE} in {SPECIES}")
print("=" * 70)

# ── 3a. Prepare codon alignment ─────────────────────────────────────────
# WHY: dN/dS analysis requires a CODON alignment — the alignment must be
# read in reading frame (triplets of nucleotides). We:
#   1. Trim the alignment to a multiple of 3 (complete codons only)
#   2. Replace ambiguous bases with N
#   3. Replace stop codons with NNN (they break PAML)
#   4. Replace partial gap codons with full gap codons (---)
# This ensures every column triplet is a valid codon or a clean gap.
print("\n── 3a. Preparing codon alignment ──")

aln_path = DATA_PROC / f'{prefix}_aligned_trimmed.fasta'
if not aln_path.exists():
    print(f"  ERROR: Trimmed alignment not found: {aln_path}")
    print(f"  Run Step 01 first with tBLASTn data.")
    sys.exit(1)

aln = AlignIO.read(aln_path, 'fasta')
aln_len = aln.get_alignment_length()
trim_to = (aln_len // 3) * 3
n_codons = trim_to // 3

print(f"  Input alignment: {len(aln)} sequences x {aln_len} columns")
print(f"  Codon-trimmed to: {trim_to} columns ({n_codons} codons)")

codon_records = []
for rec in aln:
    seq = str(rec.seq)[:trim_to]
    seq = re.sub(r'[^ACGTacgt-]', 'N', seq)
    codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
    cleaned_codons = []
    for codon in codons:
        if '-' in codon:
            cleaned_codons.append('---')
        else:
            try:
                aa = Seq(codon).translate()
                if str(aa) == '*':
                    cleaned_codons.append('NNN')
                else:
                    cleaned_codons.append(codon)
            except Exception:
                cleaned_codons.append(codon)
    clean_id = re.sub(r'[^A-Za-z0-9_]', '_', rec.id)[:30]
    codon_records.append(SeqRecord(Seq(''.join(cleaned_codons)),
                                    id=clean_id, description=''))

print(f"  Cleaned codon alignment: {len(codon_records)} sequences x {trim_to} columns")

# Load metadata
loci_meta = pd.read_csv(DATA_PROC / f'{prefix}_loci_table.csv')
if 'chrom_class' not in loci_meta.columns:
    loci_meta['chrom_class'] = loci_meta['chrom'].astype(str).apply(
        lambda c: classify_chrom(c, cfg))

id_to_class = {}
for _, row in loci_meta.iterrows():
    locus_id = row.get('locus_id', row.get('gene_unit_id', ''))
    fasta_id = re.sub(r'[^A-Za-z0-9_]', '_',
        f"gu_{locus_id}_chr{row['chrom']}_{row['start']}_{row['end']}")[:30]
    id_to_class[fasta_id] = row['chrom_class']

# ── 3b. Pairwise dN/dS ──────────────────────────────────────────────────
# WHY: The Nei-Gojobori (1986) method counts synonymous (S) and
# nonsynonymous (N) sites for each codon, then tallies actual synonymous
# (sd) and nonsynonymous (nd) differences between pairs. After JC
# correction: dS = JC(sd/S), dN = JC(nd/N), omega = dN/dS.
# This gives us the FULL DISTRIBUTION of selection pressures, not just
# a single average — revealing heterogeneity among copy pairs.
print("\n── 3b. Computing pairwise dN/dS (Nei-Gojobori) ──")

codon_dict = {rec.id: str(rec.seq) for rec in codon_records}
dnds_df = compute_pairwise_dnds(codon_dict)

valid = dnds_df.dropna(subset=['dN', 'dS', 'omega'])
valid = valid[(valid['dS'] > 0) & (valid['dS'] < 10) & (valid['omega'] < 99)]

print(f"  Total pairs: {len(dnds_df)}")
print(f"  Valid pairs (dS > 0, omega < 99): {len(valid)}")

if len(valid) > 0:
    med_dN = valid['dN'].median()
    med_dS = valid['dS'].median()
    med_omega = valid['omega'].median()
    mean_dN = valid['dN'].mean()
    mean_dS = valid['dS'].mean()

    expected_dN_functional = BIRD_AVG_DNDS * med_dS
    expected_dN_duplicated = DUPLICATED_GENE_DNDS * med_dS
    fold_vs_functional = mean_dN / expected_dN_functional if expected_dN_functional > 0 else np.nan
    fold_vs_duplicated = mean_dN / expected_dN_duplicated if expected_dN_duplicated > 0 else np.nan

    low_ds = valid[(valid['dS'] < 0.01) & (valid['dN'] > 0)]
    pct_polymorphism = len(low_ds) / len(valid) * 100

    print(f"\n  Median pairwise dN:    {med_dN:.4f}")
    print(f"  Median pairwise dS:    {med_dS:.4f}")
    print(f"  Median pairwise omega: {med_omega:.4f}")
    print(f"  Observed dN / functional: {fold_vs_functional:.1f}x")
    print(f"  Observed dN / duplicated: {fold_vs_duplicated:.1f}x")

    # ── Figure 1: dN/dS distributions ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.hist(valid['dN'], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(mean_dN, color='gold', linestyle='--', label=f'Mean={mean_dN:.3f}')
    ax.axvline(expected_dN_functional, color='red', linestyle='--',
               label=f'Expected (functional)={expected_dN_functional:.3f}')
    ax.set_xlabel('dN (nonsynonymous rate)')
    ax.set_ylabel('Count')
    ax.set_title('Pairwise dN distribution')
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.hist(valid['dS'], bins=50, color='seagreen', edgecolor='white', alpha=0.8)
    ax.axvline(mean_dS, color='gold', linestyle='--', label=f'Mean={mean_dS:.3f}')
    ax.set_xlabel('dS (synonymous rate)')
    ax.set_ylabel('Count')
    ax.set_title('Pairwise dS distribution')
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    omega_plot = valid['omega'][valid['omega'] < 5]
    ax.hist(omega_plot, bins=50, color='darkorange', edgecolor='white', alpha=0.8)
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='omega=1 (neutral)')
    ax.axvline(med_omega, color='gold', linestyle='--', label=f'Median={med_omega:.3f}')
    ax.axvline(BIRD_AVG_DNDS, color='blue', linestyle='--', label=f'Bird avg ({BIRD_AVG_DNDS})')
    ax.set_xlabel('omega (dN/dS)')
    ax.set_ylabel('Count')
    ax.set_title('Pairwise omega distribution')
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.scatter(valid['dS'], valid['dN'], alpha=0.1, s=5, color='steelblue')
    max_val = max(valid['dS'].quantile(0.99), valid['dN'].quantile(0.99))
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='omega=1')
    ax.plot([0, max_val], [0, max_val * BIRD_AVG_DNDS], 'b--', alpha=0.5,
            label=f'omega={BIRD_AVG_DNDS}')
    ax.set_xlabel('dS')
    ax.set_ylabel('dN')
    ax.set_title('dN vs dS scatter')
    ax.legend(fontsize=9)
    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)

    plt.suptitle(f'{GENE} dN/dS Analysis — {SPECIES}', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'dnds_pairwise_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {FIG_DIR / 'dnds_pairwise_analysis.png'}")

    # ── Figure 2: Summary comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    categories = [f'{GENE}\nobserved', 'Functional\ngene avg', 'Duplicated\ngene avg']
    values = [mean_dN, expected_dN_functional, expected_dN_duplicated]
    colors = ['darkorange', 'steelblue', 'seagreen']
    bars = ax.bar(categories, values, color=colors)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Mean pairwise dN')
    ax.set_title(f'{GENE} dN: {fold_vs_functional:.1f}x functional, '
                 f'{fold_vs_duplicated:.1f}x duplicated')

    ax = axes[1]
    categories = ['omega\nmedian', 'Bird avg\n(~0.15)', 'Neutral\n(1.0)']
    values = [med_omega, BIRD_AVG_DNDS, 1.0]
    colors = ['darkorange', 'steelblue', 'gray']
    bars = ax.bar(categories, values, color=colors)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('omega (dN/dS)')
    ax.set_title(f'{GENE} omega vs genome benchmarks')
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'dnds_genome_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIG_DIR / 'dnds_genome_comparison.png'}")

    valid.to_csv(TABLE_DIR / 'pairwise_dnds.csv', index=False)
    print(f"  Saved: {TABLE_DIR / 'pairwise_dnds.csv'}")
else:
    print("  WARNING: No valid dN/dS pairs computed.")
    med_dN = med_dS = med_omega = mean_dN = mean_dS = np.nan
    fold_vs_functional = fold_vs_duplicated = np.nan
    pct_polymorphism = 0

# ── 3c. Prepare PAML input (subsampled) ─────────────────────────────────
# WHY: PAML codeml is the most rigorous framework for detecting positive
# selection. Unlike pairwise methods, it fits a PHYLOGENETIC model that
# accounts for the tree topology (shared ancestry). We build a neighbor-
# joining tree from the alignment and subsample to ~40 sequences because
# PAML scales poorly with large datasets. Stratified sampling ensures
# ancestral, macro, micro, and sex chromosome copies are all represented.
print("\n── 3c. Preparing PAML codeml input ──")

MAX_SEQS_PAML = cfg.get("max_seqs_paml", 40)
rng = np.random.default_rng(42)

anc_class = f"chr{cfg['ancestral_chromosome']}_ancestral"
ancestral_recs = [r for r in codon_records if id_to_class.get(r.id) == anc_class]
ancestral_ids = {r.id for r in ancestral_recs}
other_recs = [r for r in codon_records if r.id not in ancestral_ids]

# Group by class for stratified sampling
class_pools = {}
for r in other_recs:
    cls = id_to_class.get(r.id, 'unknown')
    class_pools.setdefault(cls, []).append(r)

sampled = list(ancestral_recs)
remaining = MAX_SEQS_PAML - len(sampled)

if class_pools and remaining > 0:
    # Allocate proportionally
    total_other = sum(len(v) for v in class_pools.values())
    for cls, pool in sorted(class_pools.items()):
        n_take = max(2, int(remaining * len(pool) / total_other))
        n_take = min(n_take, len(pool))
        idx = rng.choice(len(pool), size=n_take, replace=False)
        sampled.extend([pool[i] for i in idx])

paml_records = sampled[:MAX_SEQS_PAML]

# Deduplicate IDs
seen_ids = set()
for rec in paml_records:
    while rec.id in seen_ids:
        rec.id = rec.id + '_2'
    seen_ids.add(rec.id)

class_counts = {}
for r in paml_records:
    cls = id_to_class.get(r.id, 'unknown')
    class_counts[cls] = class_counts.get(cls, 0) + 1

print(f"  PAML subsample: {len(paml_records)} sequences")
for cls, n in sorted(class_counts.items()):
    print(f"    {cls}: {n}")

# Write PHYLIP alignment
paml_aln_path = PAML_DIR / f'{prefix}_codon.phy'
seq_len = len(paml_records[0].seq)
with open(paml_aln_path, 'w') as f:
    f.write(f"  {len(paml_records)}  {seq_len}\n")
    for rec in paml_records:
        f.write(f"{rec.id:<30s}  {str(rec.seq)}\n")
print(f"  Wrote PAML alignment: {paml_aln_path.name}")

# Filter out problematic sequences before tree inference:
# 1. All-gap or mostly-gap (>95% gaps) — IQ-TREE rejects these
# 2. Duplicate identical sequences — too many cause IQ-TREE assertion failures
filtered_paml_records = []
removed_seqs = []
for rec in paml_records:
    seq_str = str(rec.seq).replace('-', '').replace('N', '').replace('n', '')
    gap_frac = 1.0 - len(seq_str) / len(rec.seq) if len(rec.seq) > 0 else 1.0
    if gap_frac >= 0.95 or len(seq_str) < 30:
        removed_seqs.append(rec.id)
    else:
        filtered_paml_records.append(rec)

if removed_seqs:
    print(f"  Removed {len(removed_seqs)} all-gap/mostly-gap sequences: {', '.join(removed_seqs[:5])}")

# Deduplicate identical sequences (keep one representative per unique sequence)
seen_seqs = {}
dedup_records = []
dup_count = 0
for rec in filtered_paml_records:
    seq_key = str(rec.seq)
    if seq_key not in seen_seqs:
        seen_seqs[seq_key] = rec.id
        dedup_records.append(rec)
    else:
        dup_count += 1

if dup_count > 0:
    print(f"  Removed {dup_count} duplicate identical sequences (kept {len(dedup_records)} unique)")
    paml_records = dedup_records
else:
    paml_records = filtered_paml_records

if removed_seqs or dup_count > 0:
    # Rewrite PAML alignment with cleaned sequences
    seq_len = len(paml_records[0].seq)
    with open(paml_aln_path, 'w') as f:
        f.write(f"  {len(paml_records)}  {seq_len}\n")
        for rec in paml_records:
            f.write(f"{rec.id:<30s}  {str(rec.seq)}\n")
    print(f"  Rewrote PAML alignment: {len(paml_records)} sequences")

if len(paml_records) < 4:
    print("  WARNING: Fewer than 4 sequences after filtering — skipping PAML.")
    print("  Writing empty PAML results for cross-species comparison.")
    pd.DataFrame(columns=['Model', 'lnL', 'omega', 'kappa', 'np']).to_csv(
        TABLE_DIR / 'paml_results.csv', index=False)
    sys.exit(0)

# Build ML tree with IQ-TREE
# WHY: IQ-TREE uses maximum likelihood with automatic model selection
# (ModelFinder), producing a statistically rigorous tree with ultrafast
# bootstrap support. This is superior to NJ for PAML input because:
#   1. ML accounts for multiple substitutions at the same site
#   2. ModelFinder selects the best-fit substitution model automatically
#   3. Ultrafast bootstrap provides branch support values
iqtree_aln_path = PAML_DIR / f'{prefix}_iqtree.fasta'
SeqIO.write(paml_records, iqtree_aln_path, 'fasta')

print("  Running IQ-TREE (ML tree with ModelFinder + ultrafast bootstrap)...")
iqtree_cmd = [
    'iqtree2',
    '-s', str(iqtree_aln_path),
    '-st', 'CODON',           # Codon model (appropriate for dN/dS analysis)
    '-m', 'MFP',              # ModelFinder Plus — automatic model selection
    '-bb', '1000',            # 1000 ultrafast bootstrap replicates
    '-nt', 'AUTO',            # Auto-detect number of threads
    '--prefix', str(PAML_DIR / f'{prefix}_iqtree'),
    '-redo',                  # Overwrite previous run
    '--quiet',                # Minimal screen output
]
try:
    result = subprocess.run(
        iqtree_cmd, capture_output=True, text=True, timeout=7200
    )
    if result.returncode != 0:
        print(f"  IQ-TREE stderr: {result.stderr[-500:]}")
        print("  ERROR: IQ-TREE failed. Check alignment quality.")
        sys.exit(1)
except FileNotFoundError:
    print("  ERROR: iqtree2 not found. Install via: brew install iqtree2")
    sys.exit(1)

# Read the IQ-TREE output tree
iqtree_treefile = PAML_DIR / f'{prefix}_iqtree.treefile'
if not iqtree_treefile.exists():
    print(f"  ERROR: IQ-TREE tree not found: {iqtree_treefile}")
    sys.exit(1)

iq_tree = Phylo.read(iqtree_treefile, 'newick')

# Clean tree for PAML: remove internal node labels (bootstrap values),
# ensure positive branch lengths
for clade in iq_tree.find_clades():
    if not clade.is_terminal():
        clade.name = None
    if clade.branch_length is not None and clade.branch_length <= 0:
        clade.branch_length = 1e-6

tree_path = PAML_DIR / f'{prefix}_ml.tree'
Phylo.write(iq_tree, tree_path, 'newick')
print(f"  Wrote ML tree: {tree_path.name}")

# Print model selection result
iqtree_log = PAML_DIR / f'{prefix}_iqtree.log'
if iqtree_log.exists():
    with open(iqtree_log) as f:
        log_text = f.read()
    model_match = re.search(r'Best-fit model:\s+(\S+)', log_text)
    if model_match:
        print(f"  Best-fit model (ModelFinder): {model_match.group(1)}")

# ── Figure 3: ML tree ──
fig, ax = plt.subplots(figsize=(10, max(8, len(paml_records) * 0.25)))
Phylo.draw(iq_tree, axes=ax, do_show=False)
ax.set_title(f'Maximum-likelihood tree ({len(paml_records)} {GENE} copies) — {SPECIES}')
plt.tight_layout()
plt.savefig(FIG_DIR / f'{prefix}_ml_tree.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {FIG_DIR / f'{prefix}_ml_tree.png'}")

# ── 3d. Run PAML codeml ─────────────────────────────────────────────────
print("\n── 3d. Running PAML codeml models ──")


def write_codeml_ctl(model_name, seqfile, treefile, outfile, **kwargs):
    defaults = {
        'seqfile': str(seqfile), 'treefile': str(treefile),
        'outfile': str(outfile),
        'noisy': 0, 'verbose': 0, 'runmode': 0, 'seqtype': 1,
        'CodonFreq': 2, 'model': 0, 'NSsites': 0, 'icode': 0,
        'fix_kappa': 0, 'kappa': 2, 'fix_omega': 0, 'omega': 0.4,
        'fix_alpha': 1, 'alpha': 0, 'getSE': 0, 'RateAncestor': 0,
        'cleandata': 1,
    }
    defaults.update(kwargs)
    ctl_path = PAML_DIR / f'{model_name}.ctl'
    with open(ctl_path, 'w') as f:
        for key, val in defaults.items():
            f.write(f"      {key} = {val}\n")
    return ctl_path


def parse_codeml_output(outfile):
    results = {}
    try:
        with open(outfile) as f:
            text = f.read()
        lnl_match = re.search(r'lnL.*?(-\d+\.\d+)', text)
        if lnl_match:
            results['lnL'] = float(lnl_match.group(1))
        omega_match = re.search(r'omega \(dN/dS\)\s*=\s*(\d+\.\d+)', text)
        if omega_match:
            results['omega'] = float(omega_match.group(1))
        kappa_match = re.search(r'kappa \(ts/tv\)\s*=\s*(\d+\.\d+)', text)
        if kappa_match:
            results['kappa'] = float(kappa_match.group(1))
        np_match = re.search(r'lnL.*?np:\s*(\d+)', text)
        if np_match:
            results['np'] = int(np_match.group(1))
    except Exception:
        pass
    return results


models_config = [
    ('M0', {'NSsites': 0}),
    ('M1a', {'NSsites': 1}),
    ('M2a', {'NSsites': 2}),
    ('M7', {'NSsites': 7}),
    ('M8', {'NSsites': 8}),
]

paml_results = {}
for model_name, params in models_config:
    out_file = PAML_DIR / f'{model_name}_out.txt'
    ctl_path = write_codeml_ctl(
        model_name, seqfile=paml_aln_path, treefile=tree_path,
        outfile=out_file, **params
    )
    print(f"  Running {model_name}...", end=' ', flush=True)
    try:
        result = subprocess.run(
            ['codeml', str(ctl_path)],
            capture_output=True, text=True, cwd=str(PAML_DIR),
            timeout=3600
        )
        if out_file.exists() and out_file.stat().st_size > 100:
            paml_results[model_name] = parse_codeml_output(out_file)
            lnl = paml_results[model_name].get('lnL', 'N/A')
            if lnl != 'N/A':
                print(f"done. lnL={lnl}")
            else:
                print(f"output exists but no lnL found")
        else:
            print(f"FAILED (no output)")
            paml_results[model_name] = {}
    except subprocess.TimeoutExpired:
        print("TIMEOUT (1hr)")
        paml_results[model_name] = {}
    except FileNotFoundError:
        print("codeml not found — skipping PAML")
        break

# ── 3e. Likelihood ratio tests ──────────────────────────────────────────
# WHY: The LRT compares nested models. The test statistic is 2*(lnL_alt -
# lnL_null), which follows a chi-squared distribution with df = difference
# in number of parameters. If p < 0.05, the more complex model (with a
# positive selection class) fits significantly better than the null.
# M1a vs M2a: Does adding an omega>1 class improve the fit?
# M7 vs M8: Same test but with a more flexible beta-distribution null.
# Both tests significant = robust evidence for positive selection.
print("\n── 3e. Likelihood ratio tests ──")


def lrt(null_model, alt_model, results_dict):
    if null_model not in results_dict or alt_model not in results_dict:
        return None, None, None
    lnl_null = results_dict[null_model].get('lnL')
    lnl_alt = results_dict[alt_model].get('lnL')
    if lnl_null is None or lnl_alt is None:
        return None, None, None
    from scipy.stats import chi2
    delta = 2 * (lnl_alt - lnl_null)
    p_val = chi2.sf(delta, 2)
    return delta, 2, p_val


delta_12, df_12, p_12 = lrt('M1a', 'M2a', paml_results)
delta_78, df_78, p_78 = lrt('M7', 'M8', paml_results)

print(f"  M1a vs M2a: ", end='')
if delta_12 is not None:
    sig = "SIGNIFICANT" if p_12 < 0.05 else "Not significant"
    print(f"2dlnL={delta_12:.2f}, p={p_12:.2e} — {sig}")
else:
    print("Not computed (missing model results)")

print(f"  M7 vs M8:   ", end='')
if delta_78 is not None:
    sig = "SIGNIFICANT" if p_78 < 0.05 else "Not significant"
    print(f"2dlnL={delta_78:.2f}, p={p_78:.2e} — {sig}")
else:
    print("Not computed (missing model results)")

# ── Figure 4: PAML results ──
if paml_results:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    model_names = [m for m, _ in models_config if m in paml_results and 'lnL' in paml_results[m]]
    lnl_values = [paml_results[m]['lnL'] for m in model_names]
    if model_names:
        ax.bar(model_names, lnl_values, color='steelblue')
        ax.set_ylabel('Log-likelihood')
        ax.set_title('PAML codeml model log-likelihoods')
        for i, (m, v) in enumerate(zip(model_names, lnl_values)):
            ax.text(i, v - abs(v)*0.01, f'{v:.1f}', ha='center', fontsize=9, color='white')

    ax = axes[1]
    lrt_names = []
    lrt_vals = []
    lrt_colors = []
    if delta_12 is not None:
        lrt_names.append('M1a vs M2a')
        lrt_vals.append(delta_12)
        lrt_colors.append('crimson' if p_12 < 0.05 else 'gray')
    if delta_78 is not None:
        lrt_names.append('M7 vs M8')
        lrt_vals.append(delta_78)
        lrt_colors.append('crimson' if p_78 < 0.05 else 'gray')
    if lrt_names:
        ax.bar(lrt_names, lrt_vals, color=lrt_colors)
        ax.axhline(5.99, color='red', linestyle='--', alpha=0.5, label='chi2 critical (p=0.05)')
        ax.set_ylabel('2*delta_lnL')
        ax.set_title('Likelihood ratio tests for positive selection')
        ax.legend()

    plt.suptitle(f'{GENE} PAML Results — {SPECIES}', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'paml_model_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {FIG_DIR / 'paml_model_results.png'}")

# ── Save PAML results table ──
paml_table = []
for model_name, _ in models_config:
    row = {'Model': model_name}
    row.update(paml_results.get(model_name, {}))
    paml_table.append(row)
pd.DataFrame(paml_table).to_csv(TABLE_DIR / 'paml_results.csv', index=False)

# ── Summary ──
print("\n" + "=" * 70)
print(f"  dN/dS ANALYSIS SUMMARY — {GENE} in {SPECIES}")
print("=" * 70)
print(f"  Sequences:               {len(codon_records)}")
print(f"  Codons:                  {n_codons}")
if len(valid) > 0:
    print(f"  Median pairwise dN:      {med_dN:.4f}")
    print(f"  Median pairwise dS:      {med_dS:.4f}")
    print(f"  Median pairwise omega:   {med_omega:.4f}")
    print(f"  dN vs functional avg:    {fold_vs_functional:.1f}x elevated")
    print(f"  dN vs duplicated avg:    {fold_vs_duplicated:.1f}x elevated")

m0_omega = paml_results.get('M0', {}).get('omega')
if m0_omega:
    print(f"  M0 global omega:         {m0_omega:.4f}")
if delta_12 is not None:
    print(f"  M1a vs M2a:              2dlnL={delta_12:.1f}, p={p_12:.2e}")
if delta_78 is not None:
    print(f"  M7 vs M8:                2dlnL={delta_78:.1f}, p={p_78:.2e}")

print(f"\n  => Proceed to Step 04")
print("=" * 70)
