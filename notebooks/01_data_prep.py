#!/usr/bin/env python3
"""
01 — Data Preparation (Generalized, Config-Driven)
====================================================

QUESTION:
  How many copies of the MROH gene exist in this genome, and where are they
  located? Are they clustered on one chromosome (suggesting tandem duplication)
  or dispersed across many chromosomes (suggesting retrotransposition)?

REASONING:
  MROH genes like MROH6 exist in hundreds of copies in songbird genomes. To
  study their evolution, we must first locate every copy and extract its
  nucleotide sequence. We use two complementary BLAST searches:

  (1) Exon 13 BLASTn (23-mer probe) — A short, highly conserved nucleotide
      probe from exon 13 is searched against the genome. Each hit marks the
      position of one MROH gene copy. This gives us a CENSUS: how many copies,
      on which chromosomes, at what positions. Tony counts ~728 copies in
      Melospiza georgiana and ~3,000 in zebra finch.

  (2) tBLASTn (protein query) — The full MROH protein is searched against the
      genome. This returns the actual nucleotide sequences encoding each copy,
      which we need for downstream mutation rate and selection analyses.

  We then merge overlapping tBLASTn hits into "gene units" (one per copy),
  assign each gene unit to its nearest exon 13 anchor, and apply a coverage
  filter to keep only copies with sufficient coding sequence for codon-based
  analysis.

DECISION TREE (Tony's strategy):
  First, try running with ALL 15 exons. If enough gene units (>=15) pass the
  50% coverage threshold, proceed. If not, fall back to exons 4-15 only
  (skipping the variable N-terminal exons 1-3). This was necessary in zebra
  finch because the N-terminal region is too divergent in many copies.

EXPECTED FINDINGS:
  - Copies dispersed across MANY chromosomes → supports retrotransposition
  - Copies clustered on one chromosome → suggests tandem duplication
  - The ancestral copy lives adjacent to the LSS gene (known synteny)
  - Micro-chromosomes often harbor the majority of derived copies

OUTPUTS:
  - Gene unit FASTA (filtered sequences for alignment)
  - Gene unit metadata table (chromosome, position, coverage)
  - MAFFT multiple sequence alignment (trimmed for gaps)
  - Figures: chromosome distribution, coverage filter, genomic positions

Usage:
  python notebooks/01_data_prep.py --species melospiza_georgiana
  python notebooks/01_data_prep.py --species zebra_finch
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
from Bio import SeqIO, AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pathlib import Path
import subprocess

from utils import (
    parse_blast_fasta, parse_exon13_blastn, parse_combined_tblastn,
    build_accession_to_chrom, define_gene_units, gene_units_to_fasta,
    merge_overlapping_hits, define_gene_units_from_anchors_only,
)
from species_config import load_config, resolve_path, get_data_dirs, classify_chrom

# ── Parse arguments ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Step 01: Data Preparation')
parser.add_argument('--species', required=True, help='Species config slug or path')
args, _ = parser.parse_known_args()

cfg = load_config(args.species)
dirs = get_data_dirs(cfg)

PROJECT = Path(cfg["_project_root"])
DATA_PROC = dirs["data_proc"]
FIG_DIR = dirs["fig_dir"]
TABLE_DIR = dirs["table_dir"]

sns.set_context('notebook')
sns.set_style('whitegrid')

MIN_COVERAGE = cfg["min_coverage"]
GENE = cfg["gene_name"]
SPECIES = cfg["common_name"]

print("=" * 70)
print(f"  STEP 01: DATA PREPARATION — {GENE} in {SPECIES}")
print(f"  Genome: {cfg['genome_assembly']} ({cfg['genome_accession']})")
print("=" * 70)

# ── 1a. Parse exon 13 anchors ────────────────────────────────────────────
# WHY: The 23-mer probe from exon 13 is highly conserved across all MROH copies.
# Each BLASTn hit = one gene copy. This gives us the total copy count and
# genomic positions before we even look at sequence data.
# The probe can come in two formats:
#   - Tabular BLASTn (zebra finch): tab-separated columns from NCBI
#   - FASTA (Melospiza georgiana): header encodes coordinates, sequence below
print("\n── 1a. Parsing exon 13 anchors ──")

anchor_path = resolve_path(cfg, cfg["exon13_anchor_file"])
anchor_fmt = cfg.get("exon13_anchor_format", "auto")
exon13 = parse_exon13_blastn(anchor_path, fmt=anchor_fmt)
print(f"  Exon 13 anchors: {len(exon13)} across {exon13['saccver'].nunique()} accessions")

# Build accession -> chromosome mapping from anchors (for FASTA format)
if anchor_fmt == "fasta" or (anchor_fmt == "auto" and anchor_path.suffix != '.txt'):
    # Re-parse as FASTA to get chromosome mapping
    anchor_fasta_df = parse_blast_fasta(anchor_path)
    acc_to_chrom_anchor = build_accession_to_chrom(anchor_fasta_df)
else:
    acc_to_chrom_anchor = {}

# ── 1b. Parse tBLASTn hits ───────────────────────────────────────────────
# WHY: tBLASTn searches the MROH protein against the genome, returning the
# nucleotide sequences that encode each copy. Unlike the exon 13 probe
# (which only marks positions), tBLASTn gives us the actual DNA sequences
# we need for alignment, divergence calculation, and codon-based dN/dS.
# Multiple protein queries can be combined (e.g., two isoforms) to capture
# diverged copies that one query might miss.
# If tBLASTn data is not yet available, the pipeline proceeds with anchor-
# only analysis (copy counting and chromosome distribution) and tells the
# user exactly what file to provide.
print("\n── 1b. Parsing tBLASTn hits ──")

tblastn_paths = [resolve_path(cfg, f) for f in cfg["tblastn_files"]]
existing_tblastn = [p for p in tblastn_paths if p.exists()]

if existing_tblastn:
    tblastn = parse_combined_tblastn(existing_tblastn)
    acc_to_chrom = build_accession_to_chrom(tblastn)
    # Merge with anchor-derived chromosome mapping
    acc_to_chrom.update({k: v for k, v in acc_to_chrom_anchor.items()
                         if k not in acc_to_chrom})
    print(f"  tBLASTn hits (combined, deduplicated): {len(tblastn)}")
    has_tblastn = True
else:
    print(f"  WARNING: No tBLASTn files found. Expected:")
    for p in tblastn_paths:
        print(f"    {p}")
    print(f"  Proceeding with anchor-only analysis (positions but no sequences).")
    print(f"  To complete the pipeline, provide tBLASTn alignment file(s).")
    tblastn = pd.DataFrame()
    acc_to_chrom = acc_to_chrom_anchor
    has_tblastn = False

# Show chromosome distribution from anchors
anchor_chroms = exon13['saccver'].map(acc_to_chrom).fillna('unknown')
print(f"\n  Anchor distribution by chromosome:")
for chrom, count in anchor_chroms.value_counts().head(15).items():
    print(f"    chr {chrom}: {count} anchors")

total_copies = len(exon13)
print(f"\n  Total MROH6 copies (by exon 13 count): {total_copies}")

# ── Figure 1: BLAST hit overview ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Anchor distribution by chromosome
chrom_counts = anchor_chroms.value_counts().head(20)
anc_chrom = cfg["ancestral_chromosome"]
colors = []
for c in chrom_counts.index:
    if c == anc_chrom:
        colors.append('crimson')
    elif c in cfg.get("sex_chromosomes", []):
        colors.append('mediumpurple')
    elif c in cfg.get("macro_chromosomes", []):
        colors.append('steelblue')
    else:
        colors.append('darkorange')
chrom_counts.plot.bar(ax=axes[0], color=colors)
axes[0].set_title(f'Exon 13 anchors per chromosome')
axes[0].set_xlabel('Chromosome')
axes[0].set_ylabel('Number of anchors')

# Hit length distribution (if tBLASTn available)
if has_tblastn and len(tblastn) > 0:
    axes[1].hist(tblastn['seq_len'], bins=50, color='steelblue', edgecolor='white')
    axes[1].set_title('Individual BLAST hit lengths')
    axes[1].set_xlabel('Hit length (bp)')
    axes[1].set_ylabel('Count')
else:
    axes[1].text(0.5, 0.5, 'tBLASTn data\nnot yet available',
                 ha='center', va='center', fontsize=14, transform=axes[1].transAxes)
    axes[1].set_title('tBLASTn hit lengths (pending)')

# Category breakdown
macro_chroms = cfg.get("macro_chromosomes", [])
sex_chroms = cfg.get("sex_chromosomes", [])
micro_chroms = [c for c in anchor_chroms.unique()
                if c not in macro_chroms and c not in sex_chroms and c != 'unknown']
cats = {
    f'Chr {anc_chrom}\n(ancestral)': (anchor_chroms == anc_chrom).sum(),
    f'Other macro': ((anchor_chroms.isin(macro_chroms)) & (anchor_chroms != anc_chrom)).sum(),
    'Micro': (anchor_chroms.isin(micro_chroms)).sum(),
    'Sex\n(Z/W)': (anchor_chroms.isin(sex_chroms)).sum(),
}
axes[2].bar(cats.keys(), cats.values(),
            color=['crimson', 'steelblue', 'darkorange', 'mediumpurple'])
axes[2].set_title(f'Copies: Ancestral chr{anc_chrom} vs Derived')
axes[2].set_ylabel('Number of copies')
for i, (k, v) in enumerate(cats.items()):
    axes[2].text(i, v + max(cats.values()) * 0.02, str(v),
                 ha='center', fontsize=10, fontweight='bold')

plt.suptitle(f'{GENE} in {SPECIES} ({cfg["genome_assembly"]})', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / 'blast_hit_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {FIG_DIR / 'blast_hit_overview.png'}")


# ── 1c. Define gene units ────────────────────────────────────────────────
# WHY: Raw tBLASTn output contains many small, overlapping hits per gene copy
# (one hit per exon or domain). We need to MERGE these into "gene units" —
# one record per MROH copy — by:
#   1. Using exon 13 anchors as reference points for each copy
#   2. Assigning nearby tBLASTn hits to the nearest anchor
#   3. Concatenating hit sequences within each gene unit
#   4. Applying a COVERAGE FILTER: only keep copies where the total
#      sequence covers >=50% of the expected CDS length
# This removes pseudogenes, fragments, and low-quality hits that would
# create gaps and noise in the downstream alignment and dN/dS analysis.
#
# DECISION TREE: Tony's strategy for handling diverged copies:
#   - First try ALL exons (1-15). If >=15 gene units pass → use them.
#   - If too few pass, fall back to exons 4-15 only (skip variable N-terminal).
#   - The exon 4-15 region is more conserved and captures more copies.
def run_gene_unit_definition(exon_range, expected_len_nt, label):
    """Run gene unit definition with a given exon range and return results."""
    print(f"\n── Defining gene units ({label}, {MIN_COVERAGE*100:.0f}% coverage) ──")
    print(f"  Expected CDS length: {expected_len_nt} nt")
    print(f"  Minimum sequence length: {int(expected_len_nt * MIN_COVERAGE)} nt")

    if has_tblastn:
        gu_all, gu_filtered = define_gene_units(
            exon13, tblastn, acc_to_chrom,
            expected_len_nt=expected_len_nt,
            max_dist=cfg["max_anchor_dist"],
            merge_gap=cfg["merge_gap"],
            min_coverage=MIN_COVERAGE
        )
    else:
        gu_all = define_gene_units_from_anchors_only(exon13, acc_to_chrom)
        gu_filtered = gu_all.copy()  # no sequence filtering possible

    # Classify chromosomes
    gu_all['chrom_class'] = gu_all['chrom'].apply(lambda c: classify_chrom(c, cfg))
    gu_filtered['chrom_class'] = gu_filtered['chrom'].apply(lambda c: classify_chrom(c, cfg))

    # Identify ancestral copy
    anc_range = cfg.get("ancestral_locus_range")
    anc_chr = cfg["ancestral_chromosome"]

    if anc_range:
        chr_gu = gu_filtered[
            (gu_filtered['chrom'] == anc_chr) &
            (gu_filtered['start'] > anc_range[0]) &
            (gu_filtered['end'] < anc_range[1])
        ]
    else:
        chr_gu = gu_filtered[gu_filtered['chrom'] == anc_chr]

    if len(chr_gu) > 0:
        if has_tblastn:
            ancestral_idx = chr_gu['total_seq_len'].idxmax()
        else:
            ancestral_idx = chr_gu.index[0]
        ancestral_id = gu_filtered.loc[ancestral_idx, 'gene_unit_id']
        gu_filtered['is_ancestral'] = gu_filtered['gene_unit_id'] == ancestral_id
        gu_all['is_ancestral'] = gu_all['gene_unit_id'] == ancestral_id
        print(f"  Ancestral copy: gu_{ancestral_id} (chr{anc_chr})")
    else:
        gu_filtered['is_ancestral'] = False
        gu_all['is_ancestral'] = False
        print(f"  WARNING: Could not identify ancestral locus on chr{anc_chr}")

    n_filtered = len(gu_filtered) if has_tblastn else len(gu_all)
    print(f"  Gene units defined: {len(gu_all)}")
    if has_tblastn:
        print(f"  After {MIN_COVERAGE*100:.0f}% coverage filter: {n_filtered} gene units")
        print(f"  Removed: {len(gu_all) - n_filtered} (insufficient coverage)")

    return gu_all, gu_filtered, n_filtered


# Try primary exon range
primary_range = cfg["analysis_exon_range"]
primary_len_nt = cfg["analysis_len_nt"]
primary_label = f"exons {primary_range[0]}-{primary_range[1]}"

print(f"\n── 1c. Gene unit definition ──")
gu_all, gu_filtered, n_units = run_gene_unit_definition(
    primary_range, primary_len_nt, primary_label
)

# Decision tree: fall back to narrower exon range if too few units pass
fallback_range = cfg.get("fallback_exon_range")
used_fallback = False

if has_tblastn and fallback_range and n_units < cfg.get("min_gene_units_for_paml", 15):
    print(f"\n  *** Only {n_units} gene units passed filter with {primary_label}.")
    print(f"  *** Falling back to exons {fallback_range[0]}-{fallback_range[1]}...")
    fallback_len_nt = cfg["fallback_len_nt"]
    fallback_label = f"exons {fallback_range[0]}-{fallback_range[1]}"
    gu_all, gu_filtered, n_units = run_gene_unit_definition(
        fallback_range, fallback_len_nt, fallback_label
    )
    used_fallback = True

# Report final classification
print(f"\n  Final chromosome classification:")
print(f"    {gu_filtered['chrom_class'].value_counts().to_string()}")


# ── Figure 2: Coverage filter visualization ───────────────────────────────
if has_tblastn:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Coverage distribution
    ax = axes[0]
    cov = gu_all['coverage_frac']
    ax.hist(cov[cov > 0], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(MIN_COVERAGE, color='red', linestyle='--', linewidth=2,
               label=f'{MIN_COVERAGE*100:.0f}% cutoff')
    n_pass = (cov >= MIN_COVERAGE).sum()
    n_total = (cov > 0).sum()
    ax.set_title(f'Coverage distribution\n'
                 f'{n_pass}/{n_total} pass {MIN_COVERAGE*100:.0f}% threshold')
    ax.set_xlabel('Fraction of analysis region covered')
    ax.set_ylabel('Count')
    ax.legend()

    # Before vs after filter
    ax = axes[1]
    before_class = gu_all[gu_all['total_seq_len'] > 0]['chrom_class'].value_counts()
    after_class = gu_filtered['chrom_class'].value_counts()
    x = range(len(before_class))
    w = 0.35
    ax.bar([i - w/2 for i in x], before_class.values, w,
           label='Before filter', color='lightcoral', alpha=0.8)
    after_vals = [after_class.get(c, 0) for c in before_class.index]
    ax.bar([i + w/2 for i in x], after_vals, w,
           label='After filter', color='steelblue', alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(before_class.index, rotation=20, ha='right')
    ax.set_ylabel('Gene units')
    ax.set_title('Gene units before/after coverage filter')
    ax.legend()

    # Sequence length distribution (filtered)
    ax = axes[2]
    ax.hist(gu_filtered['total_seq_len'], bins=40, color='steelblue', edgecolor='white')
    ax.set_xlabel('Total sequence length (bp)')
    ax.set_ylabel('Count')
    ax.set_title(f'Filtered gene unit lengths (n={len(gu_filtered)})')
    if len(gu_filtered) > 0:
        ax.axvline(gu_filtered['total_seq_len'].median(), color='red', linestyle='--',
                   label=f"Median={gu_filtered['total_seq_len'].median():.0f} bp")
        ax.legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'exon_coverage_filter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIG_DIR / 'exon_coverage_filter.png'}")


# ── Figure 3: Exon 13 anchor quality ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
if 'pident' in exon13.columns:
    ax.hist(exon13['pident'], bins=20, color='seagreen', edgecolor='white')
    ax.set_xlabel('% Identity to exon 13 probe')
    ax.set_ylabel('Count')
    ax.set_title(f'Exon 13 anchor identity (n={len(exon13)})')
    ax.axvline(exon13['pident'].median(), color='red', linestyle='--',
               label=f'Median={exon13["pident"].median():.1f}%')
    ax.legend()
else:
    ax.text(0.5, 0.5, 'Identity data\nnot available\n(FASTA format)',
            ha='center', va='center', fontsize=12, transform=ax.transAxes)

ax = axes[1]
exon13_chroms = exon13['saccver'].map(acc_to_chrom).fillna('unknown')
chrom_counts_e13 = exon13_chroms.value_counts().head(15)
chrom_counts_e13.plot.bar(ax=ax, color='seagreen')
ax.set_title('Exon 13 anchors per chromosome')
ax.set_xlabel('Chromosome')
ax.set_ylabel('Anchors')

ax = axes[2]
if has_tblastn and len(tblastn) > 0:
    blast_by_chrom = tblastn['chrom'].value_counts()
    e13_by_chrom = exon13_chroms.value_counts()
    compare_chroms = blast_by_chrom.head(12).index
    x = range(len(compare_chroms))
    w = 0.35
    ax.bar([i - w/2 for i in x],
           [blast_by_chrom.get(c, 0) for c in compare_chroms],
           w, label='tBLASTn hits', color='steelblue', alpha=0.8)
    ax.bar([i + w/2 for i in x],
           [e13_by_chrom.get(c, 0) * 10 for c in compare_chroms],
           w, label='Exon 13 anchors (x10)', color='seagreen', alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(compare_chroms, rotation=45)
    ax.set_title('tBLASTn hits vs exon 13 anchors')
    ax.legend()
else:
    ax.text(0.5, 0.5, 'tBLASTn data\nnot yet available',
            ha='center', va='center', fontsize=12, transform=ax.transAxes)

plt.suptitle(f'{GENE} exon 13 anchor analysis — {SPECIES}', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / 'exon13_anchor_quality.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {FIG_DIR / 'exon13_anchor_quality.png'}")


# ── Figure 4: Genomic distribution ───────────────────────────────────────
top_chroms = gu_filtered['chrom'].value_counts().head(8).index.tolist()
for c in [cfg["ancestral_chromosome"]] + cfg.get("sex_chromosomes", []):
    if c not in top_chroms and c in gu_filtered['chrom'].values:
        top_chroms.append(c)

if top_chroms:
    fig, axes = plt.subplots(len(top_chroms), 1,
                             figsize=(14, 2 * len(top_chroms)), sharex=False)
    if len(top_chroms) == 1:
        axes = [axes]

    for ax, chrom in zip(axes, top_chroms):
        chrom_gu = gu_filtered[gu_filtered['chrom'] == chrom]
        positions = chrom_gu['start'].values / 1e6
        if chrom == cfg["ancestral_chromosome"]:
            color = 'crimson'
        elif chrom in cfg.get("sex_chromosomes", []):
            color = 'mediumpurple'
        else:
            color = 'darkorange'
        ax.scatter(positions, [1]*len(positions), marker='|', s=100, color=color, alpha=0.7)
        ax.set_yticks([])
        ax.set_ylabel(f'chr {chrom}\n({len(chrom_gu)})', rotation=0, ha='right', va='center')
        if len(positions) > 0:
            ax.set_xlim(positions.min() - 0.5, positions.max() + 0.5)

    axes[-1].set_xlabel('Genomic position (Mb)')
    fig.suptitle(f'{GENE} gene unit distribution — {SPECIES}\n'
                 '(dispersed = retrotransposition; clustered = tandem duplication)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'gene_unit_genomic_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIG_DIR / 'gene_unit_genomic_distribution.png'}")


# ── Figure 5: Hits per gene unit + coverage scatter ───────────────────────
if has_tblastn and len(gu_filtered) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(gu_filtered['n_hits'], bins=30, color='steelblue', edgecolor='white')
    ax.set_xlabel('tBLASTn hits per gene unit')
    ax.set_ylabel('Count')
    ax.set_title(f'tBLASTn coverage per gene unit (median={gu_filtered["n_hits"].median():.0f})')

    ax = axes[1]
    color_map = {}
    for cls in gu_filtered['chrom_class'].unique():
        if 'ancestral' in cls:
            color_map[cls] = 'crimson'
        elif cls == 'macro_derived':
            color_map[cls] = 'steelblue'
        elif cls == 'sex_chrom':
            color_map[cls] = 'mediumpurple'
        else:
            color_map[cls] = 'darkorange'

    for cls, color in color_map.items():
        mask = gu_filtered['chrom_class'] == cls
        ax.scatter(gu_filtered.loc[mask, 'n_hits'],
                   gu_filtered.loc[mask, 'coverage_frac'],
                   alpha=0.5, s=20, c=color, label=cls)
    ax.set_xlabel('Number of tBLASTn hits')
    ax.set_ylabel('Coverage fraction')
    ax.set_title('Hits vs coverage per gene unit')
    ax.axhline(MIN_COVERAGE, color='red', linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'gene_unit_coverage.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIG_DIR / 'gene_unit_coverage.png'}")


# ── 1d. Write output files ───────────────────────────────────────────────
print("\n── 1d. Writing output files ──")

prefix = cfg["output_prefix"]
output_fasta = DATA_PROC / f'{prefix}_gene_units.fasta'

if has_tblastn:
    n_written = gene_units_to_fasta(gu_filtered, output_fasta)
    print(f"  Wrote {n_written} gene units to {output_fasta.name}")
else:
    n_written = 0
    print(f"  No sequences to write (tBLASTn data not available)")

# Save metadata
meta_cols = [c for c in gu_filtered.columns if c != 'sequence']
gu_filtered[meta_cols].to_csv(DATA_PROC / f'{prefix}_gene_units_table.csv', index=False)
loci_df = gu_filtered.rename(columns={'gene_unit_id': 'locus_id'})
loci_cols = [c for c in loci_df.columns if c != 'sequence']
loci_df[loci_cols].to_csv(DATA_PROC / f'{prefix}_loci_table.csv', index=False)
print(f"  Saved gene unit metadata to {prefix}_gene_units_table.csv")


# ── 1e. Align with MAFFT ─────────────────────────────────────────────────
aligned_fasta = DATA_PROC / f'{prefix}_aligned.fasta'
trimmed_path = DATA_PROC / f'{prefix}_aligned_trimmed.fasta'

if has_tblastn and n_written > 1:
    # WHY: MAFFT aligns all gene unit sequences to each other so that
    # homologous positions are in the same column. This is essential for:
    #   - Counting substitutions at each position (Step 02)
    #   - Identifying synonymous vs nonsynonymous changes (Step 03)
    # We use --auto mode (MAFFT chooses the best algorithm for the dataset
    # size) and --thread -1 (use all available CPU cores).
    # After alignment, we TRIM columns where >50% of sequences have gaps.
    # These gap-rich columns are alignment artifacts from insertions in a
    # few copies and would bias divergence estimates upward.
    print("\n── 1e. Running MAFFT alignment ──")

    result = subprocess.run(
        ['mafft', '--auto', '--thread', '-1', str(output_fasta)],
        capture_output=True, text=True
    )

    if result.returncode == 0:
        with open(aligned_fasta, 'w') as f:
            f.write(result.stdout)
        print(f"  MAFFT alignment complete: {aligned_fasta.name}")
    else:
        print(f"  MAFFT error: {result.stderr[:200]}")
        sys.exit(1)

    # Alignment QC
    alignment = AlignIO.read(aligned_fasta, 'fasta')
    n_seqs = len(alignment)
    aln_len = alignment.get_alignment_length()
    print(f"  Alignment: {n_seqs} sequences x {aln_len} columns")

    gap_fracs = np.array([
        alignment[:, i].count('-') / n_seqs for i in range(aln_len)
    ])
    print(f"  Columns with >50% gaps: {(gap_fracs > 0.5).sum()} / {aln_len}")

    # Figure 6: Gap distribution
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))

    axes[0].plot(gap_fracs, color='steelblue', linewidth=0.5)
    axes[0].axhline(0.5, color='red', linestyle='--', alpha=0.7, label='50% gap threshold')
    axes[0].set_xlabel('Alignment position')
    axes[0].set_ylabel('Gap fraction')
    axes[0].set_title('Gap distribution (raw alignment)')
    axes[0].legend()

    # Trim columns with >50% gaps
    keep_cols = np.where(gap_fracs <= 0.5)[0]
    trimmed_records = []
    for record in alignment:
        trimmed_seq = ''.join(str(record.seq)[i] for i in keep_cols)
        trimmed_records.append(SeqRecord(Seq(trimmed_seq), id=record.id, description=''))

    SeqIO.write(trimmed_records, trimmed_path, 'fasta')
    n_codons = len(keep_cols) // 3
    print(f"  Trimmed alignment: {len(trimmed_records)} sequences x {len(keep_cols)} columns ({n_codons} codons)")

    # Plot trimmed gaps
    trimmed_aln = AlignIO.read(trimmed_path, 'fasta')
    gap_fracs_trim = np.array([
        trimmed_aln[:, i].count('-') / len(trimmed_aln)
        for i in range(trimmed_aln.get_alignment_length())
    ])
    axes[1].plot(gap_fracs_trim, color='steelblue', linewidth=0.5)
    axes[1].set_xlabel('Alignment position')
    axes[1].set_ylabel('Gap fraction')
    axes[1].set_title(f'Gap distribution after trimming ({len(keep_cols)} columns)')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'alignment_gap_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIG_DIR / 'alignment_gap_distribution.png'}")

else:
    print("\n── 1e. MAFFT alignment skipped (need tBLASTn sequences) ──")
    n_seqs = 0
    n_codons = 0
    trimmed_records = []


# ── Summary ──────────────────────────────────────────────────────────────
anc_label = f"chr{cfg['ancestral_chromosome']}"
chr_anc = gu_filtered[gu_filtered['chrom'] == cfg['ancestral_chromosome']]
derived = gu_filtered[gu_filtered['chrom'] != cfg['ancestral_chromosome']]

print("\n" + "=" * 70)
print(f"  DATA PREPARATION SUMMARY — {GENE} in {SPECIES}")
print("=" * 70)
print(f"  Exon 13 anchors:           {len(exon13)}")
if has_tblastn:
    print(f"  Combined tBLASTn hits:     {len(tblastn)}")
print(f"  Gene units (all):          {len(gu_all)}")
if has_tblastn:
    exon_label = primary_label if not used_fallback else f"exons {fallback_range[0]}-{fallback_range[1]}"
    print(f"  Strategy used:             {exon_label}")
    print(f"  Coverage filter:           >= {MIN_COVERAGE*100:.0f}%")
    print(f"  Gene units (filtered):     {len(gu_filtered)}")
print(f"    {anc_label} (ancestral):     {len(chr_anc)}")
for cls in sorted(gu_filtered['chrom_class'].unique()):
    if 'ancestral' not in cls:
        print(f"    {cls}:{'':>{20-len(cls)}} {len(gu_filtered[gu_filtered['chrom_class']==cls])}")
if n_seqs > 0:
    print(f"  Alignment:                 {n_seqs} seqs x {len(keep_cols)} cols ({n_codons} codons)")
    print(f"  Expansion ratio:           {len(derived)/max(len(chr_anc),1):.1f}x")
if not has_tblastn:
    print(f"\n  *** NEXT: Provide tBLASTn alignment file to complete the pipeline.")
    print(f"  *** Run tBLASTn on NCBI using protein query against {cfg['genome_assembly']}")
    print(f"  *** Save alignment as FASTA and place at:")
    for p in tblastn_paths:
        print(f"      {p}")
print(f"\n  => Proceed to Step 02")
print("=" * 70)
