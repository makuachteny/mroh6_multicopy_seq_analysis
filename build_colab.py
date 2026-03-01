#!/usr/bin/env python3
"""Build a Google Colab notebook from the MROH6 research project."""
import json

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}

def code(source):
    return {"cell_type": "code", "metadata": {}, "source": [source], "outputs": [], "execution_count": None}

cells = []

# ═══════════════════════════════════════════════════════════════════
# TITLE & OVERVIEW
# ═══════════════════════════════════════════════════════════════════
cells.append(md("""# MROH6 Multicopy Expansion in the Zebra Finch
## Complete Analysis Pipeline — Google Colab Edition

**Research question:** What duplication mechanism generated thousands of MROH6 copies in the zebra finch genome?

**Central hypothesis:** MROH6 copies arose through RNA-mediated duplication (retrotransposition), not DNA-mediated mechanisms.

**Genome assembly:** GCF_003957565.2 (*Taeniopygia guttata*, bTaeGut7.mat)

### Pipeline overview
| Step | Notebook | Goal |
|------|----------|------|
| 01 | Data Preparation | Parse tBLASTn, merge hits, filter, align |
| 02 | Mutation Rate | Test if divergence is elevated (RT signature) |
| 03 | dN/dS Analysis | Test for selection pressure (PAML) |
| 04 | Transcriptome | Check if copies are expressed in song nuclei |
| 05 | Price Equation | Model evolutionary dynamics |
| 06 | Phylogenomic Hypercube | 3D cross-species visualization |"""))

# ═══════════════════════════════════════════════════════════════════
# SETUP & INSTALLATION
# ═══════════════════════════════════════════════════════════════════
cells.append(md("## 0. Environment Setup\n\nInstall all required packages and external tools for Google Colab."))

cells.append(code("""# ── Install Python packages ──
!pip install -q biopython>=1.84 scanpy>=1.12 anndata>=0.10.8 plotly>=5.0 \\
    statsmodels>=0.14 umap-learn>=0.5.7

# ── Install bioinformatics tools ──
!apt-get install -qq mafft > /dev/null 2>&1
!echo "MAFFT installed: $(mafft --version 2>&1 | head -1)"

# ── Install PAML ──
!apt-get install -qq paml > /dev/null 2>&1
!which codeml && echo "PAML codeml installed" || echo "PAML not found — dN/dS step will be skipped"

print("\\n✓ Setup complete")"""))

cells.append(code("""# ── Create directory structure ──
import os
from pathlib import Path

BASE = Path('/content/mroh6_project')
DATA_RAW = BASE / 'data' / 'raw'
DATA_PROC = BASE / 'data' / 'processed'
PAML_DIR = DATA_PROC / 'paml_input'
DATA_TRANS = BASE / 'data' / 'transcriptome' / 'colquitt_2021'
RESULTS = BASE / 'results'
FIGURES = RESULTS / 'figures'
TABLES = RESULTS / 'tables'

for d in [DATA_RAW, DATA_PROC, PAML_DIR, DATA_TRANS, FIGURES, TABLES]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Project root: {BASE}")
print("Directory structure created.")"""))

cells.append(md("""### Upload your BLAST data

Upload `MROH6_tBLASTn_Zebra_finch.txt` using the cell below. This is the raw tBLASTn output file (~828 KB)."""))

cells.append(code("""# ── Upload BLAST data file ──
from google.colab import files

print("Please upload: MROH6_tBLASTn_Zebra_finch.txt")
uploaded = files.upload()

# Move to data/raw/
import shutil
for fname, content in uploaded.items():
    dest = DATA_RAW / fname
    with open(dest, 'wb') as f:
        f.write(content)
    print(f"Saved to {dest} ({len(content)/1024:.0f} KB)")"""))

# ═══════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS (inlined from scripts/utils.py)
# ═══════════════════════════════════════════════════════════════════
cells.append(md("## Shared Utility Functions\n\nThese are inlined from `scripts/utils.py` — shared across all analysis steps."))

cells.append(code("""import re
import numpy as np
import pandas as pd
from Bio import SeqIO, AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import defaultdict
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('notebook')
sns.set_style('whitegrid')


def parse_blast_fasta(filepath):
    \"\"\"Parse tBLASTn FASTA output into a DataFrame with genomic coordinates.\"\"\"
    records = []
    for rec in SeqIO.parse(filepath, "fasta"):
        header = rec.description
        match = re.match(r'(\\S+):(c?)(\\d+)-(\\d+)\\s+(.*)', header)
        if not match:
            continue
        accession = match.group(1)
        is_complement = match.group(2) == 'c'
        coord1 = int(match.group(3))
        coord2 = int(match.group(4))
        desc = match.group(5)
        chrom_match = re.search(r'chromosome\\s+(\\S+)', desc)
        chrom = chrom_match.group(1).rstrip(',') if chrom_match else 'unknown'
        if is_complement:
            start, end = coord2, coord1
            strand = '-'
        else:
            start, end = coord1, coord2
            strand = '+'
        records.append({
            'accession': accession, 'chrom': chrom,
            'start': start, 'end': end, 'strand': strand,
            'seq_len': len(rec.seq), 'sequence': str(rec.seq), 'header': header,
        })
    return pd.DataFrame(records)


def _make_locus(locus_id, chrom, strand, start, end, hit_rows):
    hit_rows_sorted = sorted(hit_rows, key=lambda r: r['start'])
    concat_seq = ''.join(r['sequence'] for r in hit_rows_sorted)
    return {
        'locus_id': locus_id, 'chrom': chrom, 'strand': strand,
        'start': start, 'end': end, 'span': end - start + 1,
        'n_hits': len(hit_rows), 'total_seq_len': len(concat_seq), 'sequence': concat_seq,
    }


def merge_overlapping_hits(df, max_gap=500):
    \"\"\"Merge overlapping or nearby BLAST hits on the same chrom/strand into loci.\"\"\"
    loci = []
    locus_id = 0
    for (chrom, strand), group in df.groupby(['chrom', 'strand']):
        group = group.sort_values('start').reset_index(drop=True)
        current_start = group.iloc[0]['start']
        current_end = group.iloc[0]['end']
        current_seqs = [group.iloc[0]]
        for i in range(1, len(group)):
            row = group.iloc[i]
            if row['start'] <= current_end + max_gap:
                current_end = max(current_end, row['end'])
                current_seqs.append(row)
            else:
                loci.append(_make_locus(locus_id, chrom, strand, current_start, current_end, current_seqs))
                locus_id += 1
                current_start = row['start']
                current_end = row['end']
                current_seqs = [row]
        loci.append(_make_locus(locus_id, chrom, strand, current_start, current_end, current_seqs))
        locus_id += 1
    return pd.DataFrame(loci)


def loci_to_fasta(loci_df, outpath):
    \"\"\"Write loci DataFrame to FASTA file.\"\"\"
    records = []
    for _, row in loci_df.iterrows():
        rec = SeqRecord(
            Seq(row['sequence']),
            id=f"locus_{row['locus_id']}_chr{row['chrom']}_{row['start']}_{row['end']}_{row['strand']}",
            description=f"n_hits={row['n_hits']} span={row['span']}bp",
        )
        records.append(rec)
    SeqIO.write(records, outpath, "fasta")
    return len(records)


def jukes_cantor_distance(p):
    if np.isnan(p) or p >= 0.75:
        return np.nan
    return -0.75 * np.log(1.0 - (4.0 / 3.0) * p)


def count_substitution_types(seq1, seq2):
    purines = {'A', 'G'}
    pyrimidines = {'C', 'T'}
    ts = tv = identical = gaps = 0
    for a, b in zip(seq1.upper(), seq2.upper()):
        if a == '-' or b == '-' or a == 'N' or b == 'N':
            gaps += 1
            continue
        if a == b:
            identical += 1
        elif (a in purines and b in purines) or (a in pyrimidines and b in pyrimidines):
            ts += 1
        else:
            tv += 1
    total = ts + tv + identical
    return {
        'transitions': ts, 'transversions': tv, 'identical': identical,
        'gaps': gaps, 'total_compared': total,
        'raw_divergence': (ts + tv) / total if total > 0 else np.nan,
        'ts_tv_ratio': ts / tv if tv > 0 else np.inf,
    }


def pairwise_divergence_matrix(alignment_dict):
    names = list(alignment_dict.keys())
    n = len(names)
    raw_div = np.zeros((n, n))
    jc_div = np.zeros((n, n))
    ts_tv = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            stats = count_substitution_types(alignment_dict[names[i]], alignment_dict[names[j]])
            raw_div[i, j] = raw_div[j, i] = stats['raw_divergence']
            jc_div[i, j] = jc_div[j, i] = jukes_cantor_distance(stats['raw_divergence'])
            ts_tv[i, j] = ts_tv[j, i] = stats['ts_tv_ratio']
    return names, raw_div, jc_div, ts_tv


print("Utility functions loaded.")"""))

# ═══════════════════════════════════════════════════════════════════
# STEP 01 — DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════
cells.append(md("""---
# Step 01 — Data Preparation

**Goal:** Parse tBLASTn results, merge fragmented hits into loci, filter for quality, classify by chromosome, and produce a multiple sequence alignment.

**Input:** `MROH6_tBLASTn_Zebra_finch.txt` (3,039 hits)
**Outputs:** Filtered FASTA (596 loci), MAFFT alignment, loci metadata table"""))

cells.append(code("""# ── 1a. Parse BLAST results ──
blast_file = DATA_RAW / 'MROH6_tBLASTn_Zebra_finch.txt'
df = parse_blast_fasta(blast_file)
print(f"Total BLAST hits parsed: {len(df)}")
print(f"\\nHits per chromosome (top 15):")
print(df['chrom'].value_counts().head(15))
print(f"\\nSequence length stats:")
print(df['seq_len'].describe())"""))

cells.append(code("""# ── 1a. Visualize BLAST hit distribution ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

chrom_counts = df['chrom'].value_counts().head(20)
colors = ['crimson' if c == '7' else ('darkorange' if c in ['16','25','29','30','31','33','34','35','36','37'] else 'steelblue')
          for c in chrom_counts.index]
chrom_counts.plot.bar(ax=axes[0], color=colors)
axes[0].set_title('BLAST hits per chromosome')
axes[0].set_xlabel('Chromosome')
axes[0].set_ylabel('Number of hits')

axes[1].hist(df['seq_len'], bins=50, color='steelblue', edgecolor='white')
axes[1].set_title('BLAST hit length distribution')
axes[1].set_xlabel('Sequence length (bp)')
axes[1].set_ylabel('Count')

macro_chroms = ['1', '1A', '2', '3', '4', '4A', '5', '6', '7', '8']
micro_chroms = [c for c in df['chrom'].unique() if c not in macro_chroms and c not in ['Z', 'W', 'unknown']]
cats = {'Chr 7 (ancestral)': len(df[df['chrom'] == '7']),
        'Other macro (1-8)': len(df[(df['chrom'].isin(macro_chroms)) & (df['chrom'] != '7')]),
        'Micro (9-37)': len(df[df['chrom'].isin(micro_chroms)]),
        'Sex (Z/W)': len(df[df['chrom'].isin(['Z', 'W'])])}
axes[2].bar(cats.keys(), cats.values(), color=['crimson', 'steelblue', 'darkorange', 'mediumpurple'])
axes[2].set_title('Hits: Ancestral chr7 vs Derived')
axes[2].set_ylabel('Number of hits')
for i, (k, v) in enumerate(cats.items()):
    axes[2].text(i, v + 20, str(v), ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES / 'blast_hit_overview.png', dpi=150, bbox_inches='tight')
plt.show()"""))

cells.append(code("""# ── 1b. Merge, filter, classify ──
MIN_LENGTH = 300

loci = merge_overlapping_hits(df, max_gap=500)
print(f"Merged loci: {len(loci)}")

# Identify ancestral copy on chr7 near LSS (~28.8 Mb)
chr7 = loci[loci['chrom'] == '7'].sort_values('start')
ancestral_candidates = chr7[(chr7['start'] > 28_000_000) & (chr7['end'] < 29_500_000)]
if len(ancestral_candidates) > 0:
    ancestral_idx = ancestral_candidates['total_seq_len'].idxmax()
    ancestral_locus_id = loci.loc[ancestral_idx, 'locus_id']
    print(f"Ancestral copy: locus_{ancestral_locus_id} (span={loci.loc[ancestral_idx, 'span']} bp)")
    loci['is_ancestral'] = loci['locus_id'] == ancestral_locus_id
else:
    loci['is_ancestral'] = False

# Classify chromosomes
MACRO_CHROMS = {'1', '1A', '2', '3', '4', '4A', '5', '6', '7', '8'}
SEX_CHROMS = {'Z', 'W'}
def classify_chrom(chrom):
    if chrom == '7': return 'chr7_ancestral'
    elif chrom in MACRO_CHROMS: return 'macro_derived'
    elif chrom in SEX_CHROMS: return 'sex_chrom'
    else: return 'micro_derived'

loci['chrom_class'] = loci['chrom'].apply(classify_chrom)

# Apply length filter
loci_final = loci[loci['total_seq_len'] >= MIN_LENGTH].copy()
print(f"After filter (>= {MIN_LENGTH} bp): {len(loci_final)} loci retained")
print(f"Removed: {len(loci) - len(loci_final)} short fragments")
print(f"\\nChromosome classification:")
print(loci_final['chrom_class'].value_counts())"""))

cells.append(code("""# ── 1b. Save filtered loci ──
n_written = loci_to_fasta(loci_final, DATA_PROC / 'mroh6_copies_filtered.fasta')
print(f"Wrote {n_written} loci to mroh6_copies_filtered.fasta")

loci_final.drop(columns=['sequence']).to_csv(DATA_PROC / 'mroh6_loci_table.csv', index=False)
print("Saved loci metadata to mroh6_loci_table.csv")"""))

cells.append(code("""# ── 1c. MAFFT alignment ──
input_fasta = DATA_PROC / 'mroh6_copies_filtered.fasta'
output_fasta = DATA_PROC / 'mroh6_aligned.fasta'

result = subprocess.run(
    ['mafft', '--auto', '--thread', '-1', str(input_fasta)],
    capture_output=True, text=True
)

if result.returncode == 0:
    with open(output_fasta, 'w') as f:
        f.write(result.stdout)
    print(f"MAFFT alignment complete: {output_fasta}")
else:
    print(f"MAFFT error:\\n{result.stderr}")"""))

cells.append(code("""# ── 1c. Alignment QC and trimming ──
alignment = AlignIO.read(output_fasta, 'fasta')
n_seqs = len(alignment)
aln_len = alignment.get_alignment_length()
print(f"Alignment: {n_seqs} sequences x {aln_len} columns")

gap_fracs = []
for col_idx in range(aln_len):
    col = alignment[:, col_idx]
    gap_fracs.append(col.count('-') / n_seqs)
gap_fracs = np.array(gap_fracs)
print(f"Columns with >50% gaps: {(gap_fracs > 0.5).sum()} / {aln_len}")

# Trim columns with >50% gaps
keep_cols = np.where(gap_fracs <= 0.5)[0]
trimmed_records = []
for record in alignment:
    trimmed_seq = ''.join(str(record.seq)[i] for i in keep_cols)
    trimmed_records.append(SeqRecord(Seq(trimmed_seq), id=record.id, description=''))

trimmed_path = DATA_PROC / 'mroh6_aligned_trimmed.fasta'
SeqIO.write(trimmed_records, trimmed_path, 'fasta')
print(f"Trimmed alignment: {len(trimmed_records)} sequences x {len(keep_cols)} columns")

# Plot gap distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 3))
axes[0].plot(gap_fracs, color='steelblue', linewidth=0.5)
axes[0].axhline(0.5, color='red', linestyle='--', alpha=0.7, label='50% threshold')
axes[0].set_title('Gap distribution (before trimming)')
axes[0].set_xlabel('Position'); axes[0].set_ylabel('Gap fraction'); axes[0].legend()

trimmed_aln = AlignIO.read(trimmed_path, 'fasta')
gap_fracs_t = np.array([trimmed_aln[:, i].count('-') / len(trimmed_aln) for i in range(trimmed_aln.get_alignment_length())])
axes[1].plot(gap_fracs_t, color='steelblue', linewidth=0.5)
axes[1].set_title('Gap distribution (after trimming)')
axes[1].set_xlabel('Position'); axes[1].set_ylabel('Gap fraction')
plt.tight_layout()
plt.savefig(FIGURES / 'alignment_gap_distribution.png', dpi=150, bbox_inches='tight')
plt.show()"""))

cells.append(code("""# ── Step 01 Summary ──
print("=" * 60)
print("DATA PREPARATION SUMMARY")
print("=" * 60)
print(f"Raw BLAST hits:          {len(df)}")
print(f"Merged loci:             {len(loci)}")
print(f"Filtered loci:           {len(loci_final)}")
print(f"  Chr 7 (ancestral):     {len(loci_final[loci_final['chrom']=='7'])}")
print(f"  Macro-derived:         {len(loci_final[loci_final['chrom_class']=='macro_derived'])}")
print(f"  Micro-derived:         {len(loci_final[loci_final['chrom_class']=='micro_derived'])}")
print(f"  Sex chromosomes:       {len(loci_final[loci_final['chrom_class']=='sex_chrom'])}")
print(f"\\n=> Proceed to Step 02 with {len(loci_final)} filtered loci")"""))

# ═══════════════════════════════════════════════════════════════════
# STEP 02 — MUTATION RATE
# ═══════════════════════════════════════════════════════════════════
cells.append(md("""---
# Step 02 — Mutation Rate Analysis

**Primary question:** Is the substitution rate across MROH6 copies elevated compared to baseline?

**Decision criteria:**
| Fold difference vs baseline | p-value | Interpretation |
|---|---|---|
| > 3x | < 0.05 | **Robustly elevated** — supports RT hypothesis |
| 1.5-3x | < 0.05 | Moderately elevated |
| ~1x | any | No elevation — supports DNA-only hypothesis |"""))

cells.append(code("""# ── 2a. Load alignment and compute pairwise divergence ──
from scipy import stats as sp_stats

aln_dict = {}
for rec in SeqIO.parse(DATA_PROC / 'mroh6_aligned_trimmed.fasta', 'fasta'):
    aln_dict[rec.id] = str(rec.seq)

print(f"Loaded {len(aln_dict)} aligned sequences")
print("Computing pairwise divergence matrix (this may take a few minutes)...")

names, raw_div, jc_div, ts_tv_mat = pairwise_divergence_matrix(aln_dict)
print(f"Matrix computed: {len(names)} x {len(names)}")"""))

cells.append(code("""# ── 2b. Summary statistics ──
mask = np.triu(np.ones_like(raw_div, dtype=bool), k=1)
raw_vals = raw_div[mask]
jc_vals = jc_div[mask]
tstv_vals = ts_tv_mat[mask]

# Filter valid values
raw_valid = raw_vals[~np.isnan(raw_vals)]
jc_valid = jc_vals[~np.isnan(jc_vals)]
tstv_valid = tstv_vals[(~np.isnan(tstv_vals)) & (~np.isinf(tstv_vals))]

print(f"Pairwise divergence summary:")
print(f"  Mean raw divergence:   {np.mean(raw_valid):.4f} +/- {np.std(raw_valid):.4f}")
print(f"  Mean JC-corrected:     {np.nanmean(jc_valid):.4f}")
print(f"  Median Ts/Tv ratio:    {np.median(tstv_valid):.4f}")

# Compare to baseline
BASELINE = 0.05
fold_diff = np.mean(raw_valid) / BASELINE
t_stat, p_val = sp_stats.ttest_1samp(raw_valid, BASELINE)

print(f"\\nBaseline comparison:")
print(f"  Genomic baseline:      {BASELINE} subs/site")
print(f"  Fold difference:       {fold_diff:.1f}x")
print(f"  One-sample t-test:     t={t_stat:.2f}, p={p_val:.2e}")

# Bootstrap CI
n_boot = 1000
rng = np.random.default_rng(42)
boot_means = [np.mean(rng.choice(raw_valid, size=len(raw_valid), replace=True)) for _ in range(n_boot)]
ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
print(f"  Bootstrap 95% CI:      [{ci_lo:.4f}, {ci_hi:.4f}]")"""))

cells.append(code("""# ── 2c. Chr7 ancestral vs derived comparison ──
loci_meta = pd.read_csv(DATA_PROC / 'mroh6_loci_table.csv')
chr7_names = [n for n in names if 'chr7' in n]
derived_names = [n for n in names if 'chr7' not in n]
chr7_idx = [names.index(n) for n in chr7_names if n in names]
derived_idx = [names.index(n) for n in derived_names if n in names]

if chr7_idx and derived_idx:
    chr7_jc = jc_div[np.ix_(chr7_idx, chr7_idx)]
    chr7_vals = chr7_jc[np.triu(np.ones_like(chr7_jc, dtype=bool), k=1)]
    chr7_vals = chr7_vals[~np.isnan(chr7_vals)]

    derived_jc = jc_div[np.ix_(derived_idx, derived_idx)]
    derived_vals = derived_jc[np.triu(np.ones_like(derived_jc, dtype=bool), k=1)]
    derived_vals = derived_vals[~np.isnan(derived_vals)]

    print(f"Chr7 ancestral copies:   mean JC = {np.mean(chr7_vals):.4f} (n={len(chr7_vals)} pairs)")
    print(f"Derived copies:          mean JC = {np.mean(derived_vals):.4f} (n={len(derived_vals)} pairs)")
    u_stat, mw_p = sp_stats.mannwhitneyu(chr7_vals, derived_vals, alternative='two-sided')
    print(f"Mann-Whitney U test:     U={u_stat:.0f}, p={mw_p:.4f}")"""))

cells.append(code("""# ── 2d. Visualize mutation rate analysis ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (A) Divergence histogram
axes[0,0].hist(raw_valid, bins=50, color='steelblue', edgecolor='white', alpha=0.8, density=True)
axes[0,0].axvline(BASELINE, color='red', linestyle='--', linewidth=2, label=f'Baseline ({BASELINE})')
axes[0,0].axvline(np.mean(raw_valid), color='darkorange', linewidth=2, label=f'Mean ({np.mean(raw_valid):.3f})')
axes[0,0].set_xlabel('Raw pairwise divergence'); axes[0,0].set_ylabel('Density')
axes[0,0].set_title(f'A. Divergence distribution ({fold_diff:.1f}x baseline)'); axes[0,0].legend()

# (B) Chr7 vs derived
if chr7_idx and derived_idx:
    axes[0,1].hist(chr7_vals, bins=30, alpha=0.6, color='crimson', label=f'Chr7 (n={len(chr7_vals)})', density=True)
    axes[0,1].hist(derived_vals, bins=30, alpha=0.6, color='steelblue', label=f'Derived (n={len(derived_vals)})', density=True)
    axes[0,1].set_xlabel('JC-corrected divergence'); axes[0,1].set_ylabel('Density')
    axes[0,1].set_title('B. Chr7 ancestral vs derived'); axes[0,1].legend()

# (C) Ts/Tv ratio distribution
axes[1,0].hist(tstv_valid[tstv_valid < 5], bins=50, color='darkgreen', edgecolor='white', alpha=0.8)
axes[1,0].axvline(1.0, color='red', linestyle='--', label='Ts/Tv = 1.0 (RT bias)')
axes[1,0].axvline(0.5, color='blue', linestyle='--', label='Ts/Tv = 0.5 (random)')
axes[1,0].set_xlabel('Ts/Tv ratio'); axes[1,0].set_ylabel('Count')
axes[1,0].set_title(f'C. Transition/transversion ratio (median={np.median(tstv_valid):.2f})'); axes[1,0].legend()

# (D) Divergence heatmap (subsample for visibility)
sub_n = min(50, len(names))
sub_idx = np.linspace(0, len(names)-1, sub_n, dtype=int)
sub_matrix = raw_div[np.ix_(sub_idx, sub_idx)]
im = axes[1,1].imshow(sub_matrix, cmap='YlOrRd', aspect='auto')
plt.colorbar(im, ax=axes[1,1], label='Raw divergence')
axes[1,1].set_title('D. Pairwise divergence heatmap (subset)')

plt.tight_layout()
plt.savefig(FIGURES / 'mutation_rate_analysis.png', dpi=150, bbox_inches='tight')
plt.show()"""))

cells.append(code("""# ── 2e. Save mutation rate results ──
summary_data = {
    'Metric': ['N_loci', 'Mean_raw_divergence', 'SD_raw_divergence', 'Mean_JC_divergence',
               'Median_TsTv', 'Baseline', 'Fold_difference', 'T_statistic', 'P_value',
               'Bootstrap_CI_lo', 'Bootstrap_CI_hi'],
    'Value': [len(names), f"{np.mean(raw_valid):.4f}", f"{np.std(raw_valid):.4f}",
              f"{np.nanmean(jc_valid):.4f}", f"{np.median(tstv_valid):.4f}",
              str(BASELINE), f"{fold_diff:.1f}x", f"{t_stat:.2f}", f"{p_val:.2e}",
              f"{ci_lo:.4f}", f"{ci_hi:.4f}"]
}
pd.DataFrame(summary_data).to_csv(TABLES / 'mutation_rate_summary.csv', index=False)

# Per-copy divergence from ancestral
per_copy = []
for i, name in enumerate(names):
    chrom_class = 'chr7_ancestral' if 'chr7' in name else 'derived'
    is_chr7 = 'chr7' in name
    anc_divs = [raw_div[i, j] for j in chr7_idx if i != j] if chr7_idx else []
    mean_anc_div = np.nanmean(anc_divs) if anc_divs else np.nan
    jc_anc = jukes_cantor_distance(mean_anc_div) if not np.isnan(mean_anc_div) else np.nan
    per_copy.append({'name': name, 'chrom_class': chrom_class, 'is_chr7': is_chr7,
                     'raw_div_from_ancestral': mean_anc_div, 'jc_div_from_ancestral': jc_anc})
pd.DataFrame(per_copy).to_csv(TABLES / 'per_copy_divergence.csv', index=False)

print("Saved: mutation_rate_summary.csv, per_copy_divergence.csv")
print(f"\\n{'='*60}")
print(f"MUTATION RATE CONCLUSION: {fold_diff:.1f}x baseline ({'ROBUSTLY ELEVATED' if fold_diff > 3 and p_val < 0.05 else 'NOT elevated'})")
print(f"{'='*60}")"""))

# ═══════════════════════════════════════════════════════════════════
# STEP 03 — dN/dS
# ═══════════════════════════════════════════════════════════════════
cells.append(md("""---
# Step 03 — dN/dS Analysis (PAML codeml)

**Goal:** Test for purifying selection, neutrality, or positive selection using codon-based ML models.

**Note:** This step requires PAML (codeml) to be installed. If unavailable on Colab, the cell will skip gracefully."""))

cells.append(code("""# ── 3a. Prepare codon alignment ──
aln = AlignIO.read(DATA_PROC / 'mroh6_aligned_trimmed.fasta', 'fasta')
aln_len = aln.get_alignment_length()
trim_to = (aln_len // 3) * 3

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
                cleaned_codons.append('NNN' if str(aa) == '*' else codon)
            except:
                cleaned_codons.append(codon)
    clean_id = re.sub(r'[^A-Za-z0-9_]', '_', rec.id)[:30]
    codon_records.append(SeqRecord(Seq(''.join(cleaned_codons)), id=clean_id, description=''))

print(f"Codon alignment: {len(codon_records)} sequences x {trim_to} columns ({trim_to//3} codons)")"""))

cells.append(code("""# ── 3b. Stratified subsample for PAML ──
MAX_SEQS_PAML = 20

loci_meta = pd.read_csv(DATA_PROC / 'mroh6_loci_table.csv')
if 'chrom_class' not in loci_meta.columns:
    loci_meta['chrom_class'] = loci_meta['chrom'].astype(str).apply(classify_chrom)

id_to_class = {}
for _, row in loci_meta.iterrows():
    clean_id = re.sub(r'[^A-Za-z0-9_]', '_',
        f"locus_{row['locus_id']}_chr{row['chrom']}_{row['start']}_{row['end']}_{row['strand']}")[:30]
    id_to_class[clean_id] = row['chrom_class']

rng = np.random.default_rng(42)
ancestral_recs = [r for r in codon_records if 'chr7_28' in r.id or 'chr7_288' in r.id]
ancestral_ids = {r.id for r in ancestral_recs}
other_recs = [r for r in codon_records if r.id not in ancestral_ids]

sampled = list(ancestral_recs)
chr7_others = [r for r in other_recs if id_to_class.get(r.id) == 'chr7_ancestral']
macro_recs = [r for r in other_recs if id_to_class.get(r.id) == 'macro_derived']
micro_recs = [r for r in other_recs if id_to_class.get(r.id) == 'micro_derived']
sex_recs = [r for r in other_recs if id_to_class.get(r.id) == 'sex_chrom']

for pool, n_take in [(chr7_others, 3), (macro_recs, 4), (micro_recs, 10), (sex_recs, 2)]:
    if pool:
        take = min(n_take, len(pool))
        idx = rng.choice(len(pool), size=take, replace=False)
        sampled.extend([pool[i] for i in idx])

codon_records_paml = sampled[:MAX_SEQS_PAML]
print(f"Stratified sample: {len(codon_records_paml)} sequences")"""))

cells.append(code("""# ── 3c. Build NJ tree and write PAML input ──
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from Bio.Align import MultipleSeqAlignment
from Bio import Phylo

# Write PHYLIP alignment
paml_aln_path = PAML_DIR / 'mroh6_codon.phy'
seq_len = len(codon_records_paml[0].seq)
with open(paml_aln_path, 'w') as f:
    f.write(f"  {len(codon_records_paml)}  {seq_len}\\n")
    for rec in codon_records_paml:
        f.write(f"{rec.id:<30s}  {str(rec.seq)}\\n")

# Build NJ tree
paml_aln = MultipleSeqAlignment(codon_records_paml)
calculator = DistanceCalculator('identity')
dm = calculator.get_distance(paml_aln)
constructor = DistanceTreeConstructor()
nj_tree = constructor.nj(dm)

for clade in nj_tree.find_clades():
    if not clade.is_terminal():
        clade.name = None
    if clade.branch_length is not None and clade.branch_length <= 0:
        clade.branch_length = 1e-6

tree_path = PAML_DIR / 'mroh6_nj.tree'
Phylo.write(nj_tree, tree_path, 'newick')
print(f"Wrote NJ tree: {tree_path}")

fig, ax = plt.subplots(figsize=(10, max(8, len(codon_records_paml) * 0.2)))
Phylo.draw(nj_tree, axes=ax, do_show=False)
ax.set_title('Neighbor-joining tree of MROH6 copies')
plt.tight_layout()
plt.savefig(FIGURES / 'mroh6_nj_tree.png', dpi=150, bbox_inches='tight')
plt.show()"""))

cells.append(code("""# ── 3d. Run PAML codeml ──
import shutil

def write_codeml_ctl(model_name, seqfile, treefile, outfile, **kwargs):
    defaults = {
        'seqfile': str(seqfile), 'treefile': str(treefile), 'outfile': str(outfile),
        'noisy': 0, 'verbose': 0, 'runmode': 0, 'seqtype': 1,
        'CodonFreq': 2, 'model': 0, 'NSsites': 0, 'icode': 0,
        'fix_kappa': 0, 'kappa': 2, 'fix_omega': 0, 'omega': 0.4,
        'fix_alpha': 1, 'alpha': 0, 'getSE': 0, 'RateAncestor': 0, 'cleandata': 1,
    }
    defaults.update(kwargs)
    ctl_path = PAML_DIR / f'{model_name}.ctl'
    with open(ctl_path, 'w') as f:
        for key, val in defaults.items():
            f.write(f"      {key} = {val}\\n")
    return ctl_path

def parse_codeml_output(outfile):
    results = {}
    with open(outfile) as f:
        text = f.read()
    lnl_match = re.search(r'lnL.*?(-\\d+\\.\\d+)', text)
    if lnl_match: results['lnL'] = float(lnl_match.group(1))
    omega_match = re.search(r'omega \\(dN/dS\\)\\s*=\\s*(\\d+\\.\\d+)', text)
    if omega_match: results['omega'] = float(omega_match.group(1))
    kappa_match = re.search(r'kappa \\(ts/tv\\)\\s*=\\s*(\\d+\\.\\d+)', text)
    if kappa_match: results['kappa'] = float(kappa_match.group(1))
    np_match = re.search(r'lnL.*?np:\\s*(\\d+)', text)
    if np_match: results['np'] = int(np_match.group(1))
    return results

# Check if codeml is available
codeml_available = shutil.which('codeml') is not None

if codeml_available:
    models_cfg = [
        ('M0', {'model': 0, 'NSsites': 0}),
        ('M1a', {'model': 0, 'NSsites': 1}),
        ('M2a', {'model': 0, 'NSsites': 2}),
        ('M7', {'model': 0, 'NSsites': 7}),
        ('M8', {'model': 0, 'NSsites': 8}),
    ]
    results = {}
    for model, params in models_cfg:
        ctl = write_codeml_ctl(model, paml_aln_path, tree_path, PAML_DIR / f'{model}_out.txt', **params)
        print(f"Running codeml {model}...", end=' ', flush=True)
        try:
            r = subprocess.run(['codeml', str(ctl)], capture_output=True, text=True,
                               cwd=str(PAML_DIR), timeout=1800)
            out_file = PAML_DIR / f'{model}_out.txt'
            if r.returncode == 0 and out_file.exists():
                results[model] = parse_codeml_output(out_file)
                print(f"done. lnL = {results[model].get('lnL', 'N/A')}")
            else:
                print(f"FAILED"); results[model] = {}
        except subprocess.TimeoutExpired:
            print("TIMEOUT"); results[model] = {}
else:
    print("codeml not found — skipping PAML analysis.")
    print("To install: !apt-get install paml")
    results = {}"""))

cells.append(code("""# ── 3e. Likelihood ratio tests ──
from scipy.stats import chi2

if results:
    def lrt(null_m, alt_m):
        if null_m not in results or alt_m not in results: return None, None, None
        lnl_n = results[null_m].get('lnL'); lnl_a = results[alt_m].get('lnL')
        np_n = results[null_m].get('np', 0); np_a = results[alt_m].get('np', 0)
        if lnl_n is None or lnl_a is None: return None, None, None
        delta = 2 * (lnl_a - lnl_n); df = max(np_a - np_n, 2)
        return delta, df, chi2.sf(delta, df)

    d12, df12, p12 = lrt('M1a', 'M2a')
    d78, df78, p78 = lrt('M7', 'M8')

    print("LRT: M1a vs M2a (positive selection)")
    if d12 is not None:
        print(f"  2DlnL = {d12:.2f}, df = {df12}, p = {p12:.4e}")
    print("\\nLRT: M7 vs M8 (positive selection)")
    if d78 is not None:
        print(f"  2DlnL = {d78:.2f}, df = {df78}, p = {p78:.4e}")

    if 'M0' in results and 'omega' in results['M0']:
        omega_m0 = results['M0']['omega']
        print(f"\\nGlobal dN/dS (M0): {omega_m0:.4f}")
        print(f"Bird gene average: ~0.15")
        print(f"Ratio: {omega_m0/0.15:.1f}x bird average")
else:
    print("PAML results not available — skipping LRT.")"""))

# ═══════════════════════════════════════════════════════════════════
# STEP 04 — TRANSCRIPTOME
# ═══════════════════════════════════════════════════════════════════
cells.append(md("""---
# Step 04 — Transcriptome Overlay

**Goal:** Determine if MROH6 copies are expressed in zebra finch song-control brain regions.

**Data source:** Colquitt et al. 2021 (*Science* 371:6530), GEO: GSE148997

**Note:** This step downloads data from GEO (~several hundred MB). It may take a while."""))

cells.append(code("""# ── 4a. Install scanpy and download data ──
try:
    import scanpy as sc
    sc.settings.verbosity = 2
    sc.set_figure_params(dpi=100, frameon=False)
    scanpy_available = True
except ImportError:
    print("scanpy not available. Install with: !pip install scanpy")
    scanpy_available = False

if scanpy_available:
    geo_id = 'GSE148997'
    existing = list(DATA_TRANS.glob('*'))
    if existing:
        print(f"Found existing files in {DATA_TRANS}:")
        for f in existing:
            print(f"  {f.name}")
    else:
        print(f"Downloading from GEO ({geo_id})...")
        result = subprocess.run(
            ['wget', '-r', '-np', '-nd', '-P', str(DATA_TRANS),
             f'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE148nnn/{geo_id}/suppl/'],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            print("Download complete.")
        else:
            print(f"Download failed. Manual download needed from:")
            print(f"  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={geo_id}")"""))

cells.append(code("""# ── 4b. Load and preprocess ──
adata = None
found_genes = []

if scanpy_available:
    h5ad_files = list(DATA_TRANS.glob('*.h5ad'))
    mtx_files = list(DATA_TRANS.glob('*.mtx*'))
    h5_files = list(DATA_TRANS.glob('*.h5'))
    csv_files = list(DATA_TRANS.glob('*.csv*'))

    if h5ad_files:
        adata = sc.read_h5ad(h5ad_files[0])
    elif h5_files:
        adata = sc.read_10x_h5(h5_files[0])
    elif mtx_files:
        adata = sc.read_10x_mtx(DATA_TRANS)
    elif csv_files:
        df_trans = pd.read_csv(csv_files[0], index_col=0)
        adata = sc.AnnData(df_trans)

    if adata is not None:
        print(f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")
        if adata.X.max() > 100:
            adata.var_names_make_unique()
            sc.pp.filter_cells(adata, min_genes=200)
            sc.pp.filter_genes(adata, min_cells=3)
            adata.raw = adata.copy()
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=2000)
            sc.pp.pca(adata)
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
            sc.tl.leiden(adata)

        gene_names = list(adata.var_names)
        for term in ['MROH6', 'mroh6', 'Mroh6', 'LOC', 'maestro']:
            matches = [g for g in gene_names if term.lower() in g.lower()]
            if matches:
                found_genes.extend(matches)
                print(f"Matches for '{term}': {matches[:5]}")
        found_genes = list(set(found_genes))
    else:
        print("Could not load transcriptome data automatically.")
else:
    print("Skipping transcriptome analysis (scanpy not available).")"""))

cells.append(code("""# ── 4c. Visualize expression ──
if adata is not None and found_genes:
    region_col = None
    for col in ['brain_region', 'region', 'tissue', 'cluster', 'leiden']:
        if col in adata.obs.columns:
            region_col = col
            break

    sc.pl.umap(adata, color=found_genes[:3] + ([region_col] if region_col else []),
               ncols=2, save='_mroh6.png')

    if region_col:
        sc.pl.violin(adata, found_genes[:5], groupby=region_col, rotation=45, stripplot=False)
else:
    print("Skipping visualization — data or MROH6 gene not available.")
    print("This is expected if the GEO download failed or MROH6 is not annotated.")"""))

# ═══════════════════════════════════════════════════════════════════
# STEP 05 — PRICE EQUATION
# ═══════════════════════════════════════════════════════════════════
cells.append(md("""---
# Step 05 — Price Equation Model

**Goal:** Model evolutionary dynamics of multicopy expansion under selection, comparing DNA-only vs RNA-mediated pathways.

$$\\bar{w} \\Delta\\bar{z} = \\text{Cov}(w, z) + E(w \\Delta z)$$"""))

cells.append(code("""# ── 5a. Price equation simulation functions ──
def price_equation_step(z, w, delta_z):
    w_bar = np.mean(w)
    cov_wz = np.cov(w, z, ddof=0)[0, 1]
    e_w_dz = np.mean(w * delta_z)
    delta_z_bar = (cov_wz + e_w_dz) / w_bar
    return delta_z_bar, cov_wz / w_bar, e_w_dz / w_bar


def simulate_multicopy_evolution(
    n_copies=200, n_generations=500, mu_dna=1e-3, mu_rna=1e-2,
    rna_fraction=0.3, selection_strength=0.1, optimal_z=0.0,
    duplication_rate=0.02, loss_rate=0.02, seed=42
):
    rng = np.random.default_rng(seed)
    z = rng.normal(optimal_z, 0.01, size=n_copies)
    history = {k: [] for k in ['gen', 'n_copies', 'z_mean', 'z_var',
                                'cov_wz', 'e_w_dz', 'delta_z_bar',
                                'rna_copies_added', 'dna_copies_added']}
    for gen in range(n_generations):
        n = len(z)
        if n == 0: break
        w = np.exp(-selection_strength * (z - optimal_z)**2)
        delta_z = rng.normal(0, mu_dna, size=n)
        dz_bar, cov_term, ew_dz_term = price_equation_step(z, w, delta_z)
        z = z + delta_z
        n_dup = rng.binomial(n, duplication_rate)
        n_rna_dup = rng.binomial(n_dup, rna_fraction)
        n_dna_dup = n_dup - n_rna_dup
        if n_dna_dup > 0:
            parents = rng.choice(n, size=n_dna_dup)
            z = np.concatenate([z, z[parents] + rng.normal(0, mu_dna, size=n_dna_dup)])
        if n_rna_dup > 0:
            parents = rng.choice(n, size=n_rna_dup)
            z = np.concatenate([z, z[parents] + rng.normal(0, mu_rna, size=n_rna_dup)])
        n_total = len(z)
        if n_total > 0:
            w_full = np.exp(-selection_strength * (z - optimal_z)**2)
            survival_prob = np.clip((1 - loss_rate) * w_full / np.max(w_full), 0.01, 0.99)
            z = z[rng.random(n_total) < survival_prob]
        history['gen'].append(gen); history['n_copies'].append(len(z))
        history['z_mean'].append(np.mean(z) if len(z) > 0 else np.nan)
        history['z_var'].append(np.var(z) if len(z) > 0 else np.nan)
        history['cov_wz'].append(cov_term); history['e_w_dz'].append(ew_dz_term)
        history['delta_z_bar'].append(dz_bar)
        history['rna_copies_added'].append(n_rna_dup); history['dna_copies_added'].append(n_dna_dup)
    return {k: np.array(v) for k, v in history.items()}

print("Simulation functions ready.")"""))

cells.append(code("""# ── 5b. Load empirical parameters and run simulations ──
mu_dna_empirical = 1e-3
mu_rna_empirical = 1e-2

try:
    mut_summary = pd.read_csv(TABLES / 'mutation_rate_summary.csv')
    fold_row = mut_summary[mut_summary['Metric'].str.contains('Fold', na=False)]
    if len(fold_row) > 0:
        fold_str = str(fold_row['Value'].iloc[0]).replace('x', '').strip()
        fold = float(fold_str)
        if not np.isnan(fold) and fold > 0:
            mu_rna_empirical = mu_dna_empirical * fold
            print(f"Empirical fold difference: {fold}x  =>  mu_RNA = {mu_rna_empirical:.4f}")
except Exception:
    print("Using default parameters.")

params_base = dict(n_copies=200, n_generations=500, mu_dna=mu_dna_empirical,
                   selection_strength=0.1, duplication_rate=0.02, loss_rate=0.02)

hist_dna = simulate_multicopy_evolution(**params_base, mu_rna=mu_dna_empirical, rna_fraction=0.0, seed=42)
hist_rna_mod = simulate_multicopy_evolution(**params_base, mu_rna=mu_rna_empirical, rna_fraction=0.3, seed=42)
hist_rna_high = simulate_multicopy_evolution(**params_base, mu_rna=mu_rna_empirical * 3, rna_fraction=0.5, seed=42)
print("Simulations complete.")"""))

cells.append(code("""# ── 5c. Plot simulation results ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
scenarios = [('DNA only', hist_dna, 'steelblue'),
             ('DNA + RNA (30%)', hist_rna_mod, 'darkorange'),
             ('DNA + RNA (50%, high mu)', hist_rna_high, 'crimson')]

for label, hist, color in scenarios:
    axes[0,0].plot(hist['gen'], hist['n_copies'], color=color, label=label, alpha=0.8)
axes[0,0].set_xlabel('Generation'); axes[0,0].set_ylabel('Copies')
axes[0,0].set_title('A. Copy number dynamics'); axes[0,0].legend(fontsize=9)

for label, hist, color in scenarios:
    axes[0,1].plot(hist['gen'], hist['z_var'], color=color, label=label, alpha=0.8)
axes[0,1].set_xlabel('Generation'); axes[0,1].set_ylabel('Trait variance')
axes[0,1].set_title('B. Genetic variation'); axes[0,1].legend(fontsize=9)

window = 20
for label, hist, color in scenarios:
    cov_s = pd.Series(hist['cov_wz']).rolling(window).mean()
    ew_s = pd.Series(hist['e_w_dz']).rolling(window).mean()
    axes[1,0].plot(hist['gen'], cov_s, color=color, linestyle='-', alpha=0.7, label=f'{label} Cov(w,z)')
    axes[1,0].plot(hist['gen'], ew_s, color=color, linestyle='--', alpha=0.7, label=f'{label} E(wDz)')
axes[1,0].axhline(0, color='black', linewidth=0.5)
axes[1,0].set_xlabel('Generation'); axes[1,0].set_ylabel('Price component')
axes[1,0].set_title('C. Selection vs transmission'); axes[1,0].legend(fontsize=7, ncol=2)

for label, hist, color in scenarios:
    axes[1,1].plot(hist['gen'], hist['z_mean'], color=color, label=label, alpha=0.8)
axes[1,1].axhline(0, color='black', linewidth=0.5, linestyle=':')
axes[1,1].set_xlabel('Generation'); axes[1,1].set_ylabel('Mean trait')
axes[1,1].set_title('D. Mean trait evolution'); axes[1,1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES / 'price_equation_simulations.png', dpi=150, bbox_inches='tight')
plt.show()"""))

cells.append(code("""# ── 5d. Phase diagram ──
rna_fractions = np.linspace(0, 0.8, 9)
mu_rna_multipliers = np.logspace(0, 2, 9)
final_variance = np.zeros((len(rna_fractions), len(mu_rna_multipliers)))
final_copies = np.zeros_like(final_variance)

for i, rf in enumerate(rna_fractions):
    for j, mult in enumerate(mu_rna_multipliers):
        hist = simulate_multicopy_evolution(
            n_copies=100, n_generations=300, mu_dna=mu_dna_empirical,
            mu_rna=mu_dna_empirical * mult, rna_fraction=rf,
            selection_strength=0.1, duplication_rate=0.02, loss_rate=0.02, seed=42)
        final_variance[i, j] = np.nanmean(hist['z_var'][-50:])
        final_copies[i, j] = np.nanmean(hist['n_copies'][-50:])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
im = axes[0].pcolormesh(mu_rna_multipliers, rna_fractions, final_variance, cmap='YlOrRd', shading='auto')
axes[0].set_xscale('log'); axes[0].set_xlabel('RNA/DNA mutation rate ratio')
axes[0].set_ylabel('RNA pathway fraction'); axes[0].set_title('Trait variance')
plt.colorbar(im, ax=axes[0], label='Variance')

im = axes[1].pcolormesh(mu_rna_multipliers, rna_fractions, final_copies, cmap='YlGnBu', shading='auto')
axes[1].set_xscale('log'); axes[1].set_xlabel('RNA/DNA mutation rate ratio')
axes[1].set_ylabel('RNA pathway fraction'); axes[1].set_title('Equilibrium copy number')
plt.colorbar(im, ax=axes[1], label='Copies')

plt.suptitle('Phase diagram: RNA-mediated multicopy evolution', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES / 'price_phase_diagram.png', dpi=150, bbox_inches='tight')
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════
# STEP 06 — PHYLOGENOMIC HYPERCUBE
# ═══════════════════════════════════════════════════════════════════
cells.append(md("""---
# Step 06 — 3D Phylogenomic Hypercube

**Goal:** Interactive 3D visualization of gene divergence across 100 simulated species."""))

cells.append(code("""# ── 6a. Generate synthetic phylogenomic data ──
import plotly.express as px
import plotly.graph_objects as go

N_SPECIES = 100
np.random.seed(42)

rows = []
for sp_idx in range(N_SPECIES):
    species_name = f'Species_{sp_idx:03d}'
    rows.append({'Species': species_name, 'Species_Index': sp_idx,
                 'Locus_Type': 'Ancient', 'Locus_Index': 0,
                 'Divergence': 0.0, 'Confidence': 1.0})
    n_paralogs = np.random.randint(1, 6)
    phylo_factor = (sp_idx + 1) / N_SPECIES
    alpha_param, beta_param = 2.0, max(1.0, 5.0 - 4.0 * phylo_factor)
    raw_divs = np.sort(np.random.beta(alpha_param, beta_param, size=n_paralogs))
    paralog_divs = 0.1 + raw_divs * 0.7
    for p_idx, div in enumerate(paralog_divs, start=1):
        rows.append({'Species': species_name, 'Species_Index': sp_idx,
                     'Locus_Type': 'Paralogues', 'Locus_Index': p_idx,
                     'Divergence': round(float(div), 4),
                     'Confidence': round(float(1.0 - div * 0.5), 4)})

df_hyper = pd.DataFrame(rows)
print(f"Generated: {len(df_hyper)} data points across {N_SPECIES} species")"""))

cells.append(code("""# ── 6b. 3D scatter plot ──
fig = px.scatter_3d(
    df_hyper, x='Species', y='Locus_Index', z='Divergence',
    color='Divergence', symbol='Locus_Type', size='Confidence',
    color_continuous_scale='Viridis',
    title='3D Phylogenomic Matrix: Gene A Divergence across 100 Species',
    labels={'Locus_Index': 'Locus (0=Ancient)', 'Divergence': 'Sequence Distance'},
    hover_data=['Species', 'Locus_Type', 'Locus_Index', 'Divergence', 'Confidence'],
)
fig.update_layout(
    scene=dict(xaxis=dict(title='Species', showticklabels=False),
               yaxis=dict(title='Locus (0=Ancient)'),
               zaxis=dict(title='Sequence Distance')),
    width=900, height=700, margin=dict(l=0, r=0, b=0, t=40),
)
fig.show()"""))

cells.append(code("""# ── 6c. Save outputs ──
df_hyper.to_csv(TABLES / 'phylogenomic_hypercube_data.csv', index=False)

species_summary = df_hyper.groupby('Species').agg(
    n_loci=('Locus_Index', 'count'), n_paralogs=('Locus_Type', lambda x: (x == 'Paralogues').sum()),
    mean_div=('Divergence', 'mean'), max_div=('Divergence', 'max'),
).reset_index()
species_summary.to_csv(TABLES / 'phylogenomic_species_summary.csv', index=False)

html_path = FIGURES / 'phylogenomic_3d_hypercube.html'
fig.write_html(str(html_path))
print(f"Saved: CSV data, species summary, interactive HTML ({html_path})")"""))

# ═══════════════════════════════════════════════════════════════════
# DOWNLOAD RESULTS
# ═══════════════════════════════════════════════════════════════════
cells.append(md("""---
# Download Results

Download all generated figures, tables, and data files."""))

cells.append(code("""# ── Package and download all results ──
import zipfile

zip_path = '/content/mroh6_results.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for folder in [FIGURES, TABLES, DATA_PROC]:
        for fpath in folder.rglob('*'):
            if fpath.is_file():
                arcname = str(fpath.relative_to(BASE))
                zf.write(fpath, arcname)

print(f"Results packaged: {zip_path}")
print(f"Size: {os.path.getsize(zip_path) / 1e6:.1f} MB")

from google.colab import files
files.download(zip_path)"""))

cells.append(md("""---
## Pipeline Complete

### Summary of analyses:
1. **Data Prep:** 3,039 BLAST hits -> 812 merged loci -> 596 filtered loci
2. **Mutation Rate:** Divergence elevated vs genomic baseline (supports RT hypothesis)
3. **dN/dS:** PAML codeml models tested for selection signatures
4. **Transcriptome:** MROH6 expression in song nuclei (conditional on data availability)
5. **Price Equation:** RNA pathway increases trait variance; selection-transmission balance modeled
6. **Phylogenomic Hypercube:** 3D cross-species divergence visualization

### Key conclusion:
The dispersed chromosomal distribution and elevated divergence of MROH6 copies are consistent with RNA-mediated duplication (retrotransposition) rather than DNA-mediated mechanisms."""))

# ═══════════════════════════════════════════════════════════════════
# ASSEMBLE NOTEBOOK
# ═══════════════════════════════════════════════════════════════════
notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "name": "MROH6_Multicopy_Analysis.ipynb"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"}
    },
    "cells": cells
}

output_path = "/Users/makuachtenygatluak/Documents/Research/MROH6_Multicopy_Analysis_Colab.ipynb"
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Colab notebook written to: {output_path}")
print(f"Total cells: {len(cells)}")
print(f"  Markdown: {sum(1 for c in cells if c['cell_type'] == 'markdown')}")
print(f"  Code: {sum(1 for c in cells if c['cell_type'] == 'code')}")
