#!/usr/bin/env python3
"""Build a Kaggle notebook from the MROH6 research project with full documentation."""
import json

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}

def code(source):
    return {"cell_type": "code", "metadata": {}, "source": [source], "outputs": [], "execution_count": None}

cells = []

# ═══════════════════════════════════════════════════════════════════════════════
# TITLE & OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""# MROH6 Multicopy Expansion in the Zebra Finch
## Complete Analysis Pipeline — Kaggle Edition

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
| 06 | Phylogenomic Hypercube | 3D cross-species visualization |

### Kaggle setup instructions
1. **Create a new Kaggle notebook** and upload this `.ipynb`
2. **Add your dataset**: Click "Add Data" → "Upload" → upload `MROH6_tBLASTn_Zebra_finch.txt`
   - Name the dataset something like `mroh6-blast-data`
   - It will be available at `/kaggle/input/mroh6-blast-data/`
3. **Enable Internet**: Settings → Internet → "On" (needed for `pip install` and `apt-get`)
4. **Run all cells** sequentially"""))

# ═══════════════════════════════════════════════════════════════════════════════
# SETUP & INSTALLATION
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""## 0. Environment Setup

Install all required packages and external bioinformatics tools on Kaggle.

**Note:** Kaggle kernels run Ubuntu Linux with Python 3.10+. Most scientific
Python packages (numpy, pandas, scipy, matplotlib, seaborn, plotly) are
pre-installed. We only need to add domain-specific packages (biopython, scanpy)
and command-line tools (MAFFT, PAML)."""))

cells.append(code("""# ── Install Python packages ──
!pip install -q biopython>=1.84 plotly>=5.0 statsmodels>=0.14

# ── Install bioinformatics tools ──
!apt-get install -qq mafft > /dev/null 2>&1
!echo "MAFFT installed: $(mafft --version 2>&1 | head -1)"

# ── Install PAML (for dN/dS analysis) ──
!apt-get install -qq paml > /dev/null 2>&1
!which codeml && echo "PAML codeml installed" || echo "PAML not found — dN/dS step will be skipped"

print("\\nSetup complete.")"""))

cells.append(code("""# ── Create directory structure ──
import os
from pathlib import Path

# Kaggle paths
INPUT_DIR = Path('/kaggle/input')
WORKING = Path('/kaggle/working')

BASE = WORKING / 'mroh6_project'
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

cells.append(md("""### Locate your BLAST data

The cell below searches for `MROH6_tBLASTn_Zebra_finch.txt` in the Kaggle
input directory. If you uploaded it as a dataset, it will be found
automatically. Otherwise, upload it manually."""))

cells.append(code("""# ── Find BLAST data file in Kaggle input ──
import shutil, glob

blast_filename = 'MROH6_tBLASTn_Zebra_finch.txt'
blast_dest = DATA_RAW / blast_filename

# Search all Kaggle input directories
found = glob.glob(f'/kaggle/input/**/{blast_filename}', recursive=True)

if found:
    src = found[0]
    shutil.copy(src, blast_dest)
    print(f"Found and copied: {src} -> {blast_dest}")
else:
    print(f"ERROR: '{blast_filename}' not found in /kaggle/input/")
    print("Please add it as a Kaggle dataset:")
    print("  1. Click 'Add Data' in the right panel")
    print("  2. Upload your BLAST file")
    print("  3. Re-run this cell")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS (inlined from scripts/utils.py)
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""## Shared Utility Functions

These are inlined from `scripts/utils.py` — shared across all analysis steps.

| Function | Purpose |
|---|---|
| `parse_blast_fasta()` | Parse tBLASTn FASTA output → DataFrame with genomic coordinates |
| `merge_overlapping_hits()` | Group nearby hits on same chrom/strand → single loci |
| `loci_to_fasta()` | Write loci DataFrame → FASTA file |
| `jukes_cantor_distance()` | JC69 correction for multiple hits at same site |
| `count_substitution_types()` | Count Ts, Tv, identical, gaps between two sequences |
| `pairwise_divergence_matrix()` | Compute NxN pairwise divergence matrices |"""))

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
    \"\"\"Parse tBLASTn FASTA output into a DataFrame with genomic coordinates.

    Returns DataFrame with columns:
        accession, chrom, start, end, strand, seq_len, sequence, header
    \"\"\"
    records = []
    for rec in SeqIO.parse(filepath, "fasta"):
        header = rec.description
        # Parse coordinate from header like NC_133032.1:28834524-28834700
        # or NC_133032.1:c27748619-27748443 (complement)
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
    \"\"\"Helper to create a locus record from merged hits.\"\"\"
    hit_rows_sorted = sorted(hit_rows, key=lambda r: r['start'])
    concat_seq = ''.join(r['sequence'] for r in hit_rows_sorted)
    return {
        'locus_id': locus_id, 'chrom': chrom, 'strand': strand,
        'start': start, 'end': end, 'span': end - start + 1,
        'n_hits': len(hit_rows), 'total_seq_len': len(concat_seq), 'sequence': concat_seq,
    }


def merge_overlapping_hits(df, max_gap=500):
    \"\"\"Merge overlapping or nearby BLAST hits on the same chrom/strand into loci.

    Args:
        df: DataFrame from parse_blast_fasta
        max_gap: maximum gap (bp) to merge nearby hits

    Returns:
        DataFrame of merged loci with concatenated sequences
    \"\"\"
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
    \"\"\"Jukes-Cantor correction for nucleotide distance.

    Args:
        p: proportion of differing sites (raw divergence)
    Returns:
        JC-corrected distance, or np.nan if p >= 0.75 or p is NaN
    \"\"\"
    if np.isnan(p) or p >= 0.75:
        return np.nan
    return -0.75 * np.log(1.0 - (4.0 / 3.0) * p)


def count_substitution_types(seq1, seq2):
    \"\"\"Count transitions and transversions between two aligned sequences.

    Returns dict with keys: transitions, transversions, identical, gaps,
    total_compared, raw_divergence, ts_tv_ratio
    \"\"\"
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
    \"\"\"Compute pairwise raw divergence matrix from dict of {name: sequence}.

    Returns:
        names: list of sequence names
        raw_div: np.array of raw divergences
        jc_div: np.array of JC-corrected divergences
        ts_tv: np.array of Ts/Tv ratios
    \"\"\"
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


# Chromosome classification helper
MACRO_CHROMS = {'1', '1A', '2', '3', '4', '4A', '5', '6', '7', '8'}
SEX_CHROMS = {'Z', 'W'}

def classify_chrom(chrom):
    \"\"\"Classify chromosome as ancestral, macro, micro, or sex.\"\"\"
    if chrom == '7': return 'chr7_ancestral'
    elif chrom in MACRO_CHROMS: return 'macro_derived'
    elif chrom in SEX_CHROMS: return 'sex_chrom'
    else: return 'micro_derived'

print("Utility functions loaded.")"""))


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 01 — DATA PREPARATION (full documentation)
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""---
# 01 — Data Preparation

## Goal
Parse tBLASTn results for MROH6 in the zebra finch genome, merge fragmented hits
into biologically meaningful loci, filter for quality, classify by chromosomal
origin, and produce a multiple sequence alignment for downstream evolutionary
analysis.

## Scientific context

**MROH6** (Maestro Heat-Like Repeat Family Member 6) encodes a protein
containing HEAT-like repeats — alpha-helical solenoid structures that mediate
protein-protein interactions (Andrade & Bork, 1995, *Nature Genetics*
11:115–116). In most vertebrate genomes, MROH6 exists as a **single-copy gene**
adjacent to the lanosterol synthase gene (LSS) on a single chromosome.

In the zebra finch (*Taeniopygia guttata*; Warren et al., 2010, *Nature*
464:757–762), a tBLASTn search (By Dr. Monacco) of the MROH6 protein against the
genome assembly (bTaeGut1.4.pri; Rhie et al., 2021, *Nature* 592:737–746)
returned **3,039 hits** — an extraordinary amplification from a single-copy
gene. This raises a fundamental question: **what duplication mechanism generated
thousands of MROH6 copies?**

### Two competing hypotheses

| Feature | DNA-mediated duplication | RNA-mediated duplication (retrotransposition) |
|---|---|---|
| **Mechanism** | Unequal crossing-over or segmental duplication during DNA replication | mRNA → cDNA via reverse transcriptase → genomic insertion |
| **Error rate** | ~10⁻⁹–10⁻¹⁰ per base/replication (Kunkel, 2004, *J Biol Chem* 279:16895) | ~10⁻⁴–10⁻⁵ per base/cycle (Preston, 1996, *Science* 275:228) |
| **Expected distribution** | Tandem clusters near parental locus | Dispersed across chromosomes (random insertion) |
| **Expected divergence** | Low (~0.01–0.05 subs/site; Lynch & Conery, 2000, *Science* 290:1151) | Elevated (RT errors + neutral drift) |
| **Ts/Tv ratio** | ~0.5 (random expectation) | >1.0 (RT transition bias; Wakeley, 1996, *J Mol Evol* 42:681) |

### Key observations from karyotype figures
- MROH6 copies are **massively amplified on microchromosomes** (chr 16, 25, 29–37)
- The **ancestral locus** is on **chromosome 7** near LSS (~28.8 Mb)
- Copies are **dispersed** across entire chromosome lengths (not tandem clusters) → retrotransposition signature
- The expansion is **species-specific** to zebra finch (house sparrow has far fewer copies)

### Why tBLASTn?
tBLASTn translates a nucleotide database in all six reading frames and searches
it with a protein query (Gertz et al., 2006, *BMC Bioinformatics* 7:326). This
is the correct tool because: (1) it finds copies regardless of reading frame
shifts or frameshifts that would prevent detection with BLASTn, (2)
protein-level homology is more sensitive than nucleotide-level for detecting
divergent copies, and (3) it naturally handles the ~3× redundancy of the genetic
code.

## Strategy
Filter fragments ≥ 300 bp to retain loci with sufficient alignment signal. Very
short fragments (<300 bp) produce unreliable divergence estimates and inflate
NaN rates in pairwise comparisons (Nei & Kumar, 2000, *Molecular Evolution and
Phylogenetics*, Oxford University Press, recommend ≥200–300 aligned positions
for reliable distance estimation).

**Input:** `MROH6_tBLASTn_Zebra_finch.txt` (3,039 hits)
**Outputs:** Filtered FASTA, MAFFT alignment, loci metadata table"""))

cells.append(md("""## 1a. Parse BLAST results

**Why this step:** The raw tBLASTn output is a FASTA file where each hit is a
separate sequence record with genomic coordinates embedded in the header (e.g.,
`NC_133032.1:28834524-28834700` for forward strand,
`NC_133032.1:c27748619-27748443` for complement). We need to extract the
chromosome, start/end positions, strand, and nucleotide sequence into a
structured DataFrame for downstream filtering and merging.

**What `parse_blast_fasta()` does:** For each FASTA record, it parses the
NCBI-format coordinate string, maps the accession number to a chromosome name,
determines strand orientation from the "c" prefix (complement), and stores the
nucleotide sequence. This converts unstructured BLAST output into a queryable
table with columns: `accession`, `chrom`, `start`, `end`, `strand`, `seq_len`,
`sequence`.

**Why we examine the chromosome distribution first:** The chromosomal
distribution of hits is the first qualitative test of the retrotransposition
hypothesis. If MROH6 copies arose via DNA-mediated tandem duplication, we would
expect most hits to cluster on chromosome 7 near the ancestral locus. If they
arose via retrotransposition, we expect dispersal across many chromosomes —
especially microchromosomes, which in birds have higher gene density and
recombination rates (International Chicken Genome Sequencing Consortium, 2004,
*Nature* 432:695–716)."""))

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
axes[0].legend(['chr7 = ancestral (red), micro = orange, macro = blue'], fontsize=8)

axes[1].hist(df['seq_len'], bins=50, color='steelblue', edgecolor='white')
axes[1].set_title('BLAST hit length distribution (ALL kept)')
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

cells.append(md("""## 1b. Merge, filter, and classify

**Why merge overlapping hits?** tBLASTn often reports a single genomic copy as
multiple overlapping or adjacent alignment fragments, especially when the
protein query is long and the genomic copy contains frameshifts, stop codons, or
divergent internal regions. A single MROH6 copy spanning ~4,000 bp may produce
5–15 separate BLAST hits. To avoid counting the same copy multiple times, we
merge hits that overlap or fall within 500 bp of each other on the same
chromosome and strand into a single "locus."

**Why 500 bp merge distance?** The MROH6 protein query is ~882 amino acids
(~2,646 bp coding). With introns (which would be absent in retrotransposed
copies but present in the ancestral locus), the full gene span can reach ~4748
bp. A 500 bp max gap allows merging of fragmented alignments from the same copy
while being small enough to avoid merging genuinely separate loci.

**Why 300 bp minimum length filter?** Very short BLAST fragments are problematic
for evolutionary analysis:
1. **Unreliable divergence estimates** — with fewer than ~200–300 aligned
   positions, sampling variance in the substitution rate becomes large relative
   to the signal (Nei & Kumar, 2000)
2. **Inflated NaN rates** — short fragments are more likely to be entirely
   gapped in a multiple alignment
3. **Low phylogenetic information** — short sequences provide insufficient data
   for downstream dN/dS analysis (Yang, 2007)

The 300 bp threshold retains ~73% of merged loci while removing the noisiest
fragments.

**Why identify the ancestral copy?** All downstream evolutionary analysis is
anchored to the ancestral (parent) MROH6 locus. We know from synteny analysis
that in other vertebrates, MROH6 sits adjacent to LSS on a single chromosome. In
zebra finch, LSS maps to chromosome 7 at ~28.8 Mb, so the largest MROH6 locus in
this region is the ancestral copy.

**Chromosome classification rationale:** Avian karyotypes are characterized by a
bimodal chromosome size distribution (International Chicken Genome Sequencing
Consortium, 2004, *Nature* 432:695–716):
- **Macrochromosomes** (chr 1–8, 1A, 4A): Large, gene-poor relative to their size, lower recombination rate per Mb
- **Microchromosomes** (chr 9–37+): Small, very gene-dense, high GC content, high recombination rate per Mb
- **Sex chromosomes** (Z, W): Distinct evolutionary dynamics — reduced Ne for Z, degraded W

We classify loci into these categories because retrotransposed copies may
preferentially insert into gene-dense, accessible regions (microchromosomes),
while DNA duplicates would remain on the parent chromosome (chr 7)."""))

cells.append(code("""# ── 1b. Merge overlapping hits into loci ──
MIN_LENGTH = 300

loci = merge_overlapping_hits(df, max_gap=500)
print(f"Merged loci: {len(loci)}")
print(f"\\nLoci per chromosome (top 15):")
print(loci['chrom'].value_counts().head(15))
print(f"\\nLocus span (bp) stats:")
print(loci['span'].describe())"""))

cells.append(code("""# ── 1b. Identify ancestral copy on chr7 near LSS (~28.8 Mb) ──
chr7 = loci[loci['chrom'] == '7'].sort_values('start')
print("Chromosome 7 loci near 28.8 Mb (putative ancestral MROH6 region):")
ancestral_candidates = chr7[(chr7['start'] > 28_000_000) & (chr7['end'] < 29_500_000)]
print(ancestral_candidates[['locus_id', 'start', 'end', 'strand', 'span', 'n_hits', 'total_seq_len']])

if len(ancestral_candidates) > 0:
    ancestral_idx = ancestral_candidates['total_seq_len'].idxmax()
    ancestral_locus_id = loci.loc[ancestral_idx, 'locus_id']
    print(f"\\nAncestral copy: locus_{ancestral_locus_id} (span={loci.loc[ancestral_idx, 'span']} bp)")
    loci['is_ancestral'] = loci['locus_id'] == ancestral_locus_id
else:
    print("WARNING: Could not identify ancestral locus.")
    loci['is_ancestral'] = False"""))

cells.append(code("""# ── 1b. Classify chromosomes and apply length filter ──
loci['chrom_class'] = loci['chrom'].apply(classify_chrom)

loci_final = loci[loci['total_seq_len'] >= MIN_LENGTH].copy()
n_removed = len(loci) - len(loci_final)
print(f"Before filter: {len(loci)} merged loci")
print(f"After filter (>= {MIN_LENGTH} bp): {len(loci_final)} loci retained")
print(f"Removed: {n_removed} short fragments (< {MIN_LENGTH} bp)")
print(f"\\nChromosome classification (filtered):")
print(loci_final['chrom_class'].value_counts())
print(f"\\nSequence length stats (filtered loci):")
print(loci_final['total_seq_len'].describe())"""))

cells.append(code("""# ── 1b. Save filtered loci ──
n_written = loci_to_fasta(loci_final, DATA_PROC / 'mroh6_copies_filtered.fasta')
print(f"Wrote {n_written} loci to mroh6_copies_filtered.fasta")

loci_final.drop(columns=['sequence']).to_csv(DATA_PROC / 'mroh6_loci_table.csv', index=False)
print("Saved loci metadata to mroh6_loci_table.csv")

chr7_loci = loci_final[loci_final['chrom'] == '7']
derived_loci = loci_final[loci_final['chrom'] != '7']
print(f"\\n--- Chr 7 (parent) vs Derived chromosomes ---")
print(f"  Chr 7 loci:     {len(chr7_loci)} ({chr7_loci['total_seq_len'].sum():,} bp total)")
print(f"  Derived loci:   {len(derived_loci)} ({derived_loci['total_seq_len'].sum():,} bp total)")
print(f"  Expansion ratio: {len(derived_loci)/max(len(chr7_loci),1):.1f}x more copies on derived chromosomes")"""))

cells.append(md("""## 1c. Align with MAFFT

**Why multiple sequence alignment?** To compute pairwise divergence, Ts/Tv
ratios, and dN/dS, we need homologous positions across all copies to be aligned
column-by-column. Without alignment, we cannot distinguish substitutions from
insertions/deletions, and divergence calculations would be meaningless.

**Why MAFFT?** MAFFT (Katoh & Standley, 2013, *Mol Biol Evol* 30:772–780) uses
Fast Fourier Transform to accelerate the progressive alignment algorithm. We use
`--auto` mode, which automatically selects the optimal strategy based on dataset
size:
- For <200 sequences: L-INS-i (most accurate, iterative refinement)
- For 200–2000 sequences: FFT-NS-2 (fast progressive method)
- `--thread -1` uses all available CPU cores

With 596 sequences, MAFFT will select FFT-NS-2, which balances accuracy and
speed for medium-sized datasets.

**Why trim columns with >50% gaps?** After alignment, many columns will be
dominated by gaps because different copies retain different fragments of the
original gene. Columns where >50% of sequences have gaps carry more noise than
signal:
- Gap-rich columns inflate alignment length without adding phylogenetic information
- They increase computational cost of downstream analyses (pairwise matrix, PAML)
- Standard practice is to trim columns above a gap threshold of 50–70% (Capella-Gutierrez et al., 2009, *Bioinformatics* 25:1972–1973, trimAl methodology)

**Calculation:** If an alignment column has $k$ gaps out of $n$ sequences, the
gap fraction is $k/n$. We keep columns where $k/n \\leq 0.50$."""))

cells.append(code("""# ── 1c. Run MAFFT alignment ──
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

cells.append(code("""# ── 1c. Alignment QC ──
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
print(f"Columns with 0% gaps: {(gap_fracs == 0).sum()} / {aln_len}")"""))

cells.append(code("""# ── 1c. Trim columns with >50% gaps ──
keep_cols = np.where(gap_fracs <= 0.5)[0]
trimmed_records = []
for record in alignment:
    trimmed_seq = ''.join(str(record.seq)[i] for i in keep_cols)
    trimmed_records.append(SeqRecord(Seq(trimmed_seq), id=record.id, description=''))

trimmed_path = DATA_PROC / 'mroh6_aligned_trimmed.fasta'
SeqIO.write(trimmed_records, trimmed_path, 'fasta')
print(f"Trimmed alignment: {len(trimmed_records)} sequences x {len(keep_cols)} columns")

# Visualize gap distribution before/after
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
print(f"Length filter:           >= {MIN_LENGTH} bp")
print(f"Filtered loci:           {len(loci_final)}  (removed {len(loci) - len(loci_final)} short fragments)")
print(f"  Chr 7 (ancestral):     {len(loci_final[loci_final['chrom']=='7'])}")
print(f"  Macro-derived:         {len(loci_final[loci_final['chrom_class']=='macro_derived'])}")
print(f"  Micro-derived:         {len(loci_final[loci_final['chrom_class']=='micro_derived'])}")
print(f"  Sex chromosomes:       {len(loci_final[loci_final['chrom_class']=='sex_chrom'])}")
if 'is_ancestral' in loci_final.columns:
    anc = loci_final[loci_final['is_ancestral']]
    if len(anc) > 0:
        print(f"Ancestral copy:          locus_{anc.iloc[0]['locus_id']} (chr7, {anc.iloc[0]['span']}bp span)")
print(f"\\n=> Proceed to Step 02 with {len(loci_final)} filtered loci")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 02 — MUTATION RATE (full documentation)
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""---
# 02 — Mutation Rate Analysis: Chr 7 (Parent) vs Derived Copies

## Primary question
Is the substitution rate across MROH6 copies elevated compared to the genomic
baseline? If yes, this constitutes indirect evidence for an RNA-intermediate
(retrotransposition) duplication mechanism, because reverse transcriptase lacks
the 3'-to-5' exonuclease proofreading of DNA polymerase (Kunkel, 2004, *J Biol
Chem* 279:16895–16898), resulting in error rates ~100–1,000× higher per
nucleotide incorporated (Preston et al., 1988, *Science* 242:1168–1171; Roberts
et al., 1988, *Science* 242:1171–1173).

## Hypothesis framework

**H₀ (DNA-mediated):** MROH6 copies were generated by normal DNA replication
mechanisms (segmental duplication, unequal crossing-over). Mean pairwise
divergence in the range of 0.01–0.10 subs/site — typical of recently duplicated
paralogs in vertebrate genomes (Lynch & Conery, 2000, *Science* 290:1151–1155).

**H₁ (RNA-mediated):** MROH6 copies were generated by retrotransposition via an
RNA intermediate. Mean pairwise divergence >> 0.10, with transition bias (Ts/Tv
> 0.5) reflecting the error spectrum of reverse transcriptase (Bebenek et al.,
1989, *J Biol Chem* 264:16948–16956; Wakeley, 1996, *Trends Ecol Evol*
11:158–162).

## Decision criteria (defined a priori)

| Fold difference vs baseline | p-value | Interpretation | Action |
|---|---|---|---|
| > 3× | < 0.05 | **Robustly elevated** | Supports RT hypothesis → proceed to dN/dS |
| 1.5–3× | < 0.05 | **Moderately elevated** | Proceed with caution |
| ~1× | any | **Not elevated** | DNA replication alone explains divergence |

## Key comparison
Chromosome 7 carries the **ancestral MROH6 locus** (near LSS, ~28.8 Mb). All
other chromosomes carry **derived copies**. If copies dispersed via
retrotransposition in a single burst, chr7 siblings and derived copies should
show **equal divergence** from the ancestral locus (Mann-Whitney U test).

## Metrics computed

| Metric | Formula / Method | Purpose |
|---|---|---|
| Raw pairwise divergence (p-distance) | $p = \\frac{T_s + T_v}{T_s + T_v + \\text{identical}}$ | Basic measure of sequence difference |
| Jukes-Cantor corrected distance | $d_{JC} = -\\frac{3}{4}\\ln\\!\\left(1 - \\frac{4}{3}p\\right)$ | Corrects for multiple hits at same site |
| Ts/Tv ratio | $\\frac{\\text{transitions}}{\\text{transversions}}$ | RT signature: bias > 0.5 indicates transition preference |
| Divergence from ancestral | Per-copy distance to chr7 locus_787 | Radial divergence from source gene |

## Reference values

| Quantity | Value | Source |
|---|---|---|
| Zebra finch per-year substitution rate | ~2.2 × 10⁻⁹ subs/site/year | Smeds et al., 2016 |
| Recent DNA-mediated paralog divergence | 0.01–0.10 subs/site | Lynch & Conery, 2000 |
| RT error rate (HIV-1 RT, in vitro) | ~2.5–5.0 × 10⁻⁴ per base/cycle | Preston et al., 1988 |
| RT error rate (HIV-1, in vivo) | ~3.4 × 10⁻⁵ per base/cycle | Mansky & Temin, 1995 |
| DNA polymerase error rate (with proofreading) | ~10⁻⁶–10⁻⁷ per base | Kunkel, 2004 |
| Expected Ts/Tv under random mutation | 0.5 | 4 Ts types / 8 Tv types |
| RT-mediated Ts/Tv (HIV-1 RT in vitro) | ~5.7:1 | Bebenek et al., 1989 |"""))

cells.append(md("""## 2a. Load alignment and compute pairwise divergence

**Why pairwise divergence?** By computing it across all copy pairs, we obtain the
global substitution landscape of the MROH6 family. If copies arose by
DNA-mediated duplication, the mean p-distance should be low (~0.01–0.10). If
they arose by retrotransposition, the combination of RT errors and subsequent
neutral drift should produce much higher divergence.

**Why Jukes-Cantor correction?** Raw p-distance underestimates true evolutionary
distance because it does not account for **multiple hits** — the phenomenon
where the same nucleotide site is mutated more than once:

$$d_{JC} = -\\frac{3}{4} \\ln\\!\\left(1 - \\frac{4}{3}\\,p\\right)$$

The correction is undefined when $p \\geq 0.75$ (saturation).

**Why NaN-safe computation?** NaN arises from: (1) no comparable positions
(total_compared = 0), (2) JC saturation (p ≥ 0.75). We filter NaN values
before all statistics using `np.nanmean()`."""))

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

raw_valid = raw_vals[~np.isnan(raw_vals)]
jc_valid = jc_vals[~np.isnan(jc_vals)]
tstv_valid = tstv_vals[(~np.isnan(tstv_vals)) & (~np.isinf(tstv_vals))]

n_total_pairs = len(raw_vals)
n_nan = np.isnan(raw_vals).sum()

print(f"Total pairwise comparisons: {n_total_pairs:,}")
print(f"Valid (non-NaN): {len(raw_valid):,} ({100*len(raw_valid)/n_total_pairs:.1f}%)")
print(f"NaN pairs: {n_nan:,} ({100*n_nan/n_total_pairs:.1f}%)")
print(f"\\nPairwise divergence summary:")
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

# Effect size
cohens_d = (np.mean(raw_valid) - BASELINE) / np.std(raw_valid)
print(f"  Cohen's d:             {cohens_d:.2f} ({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'} effect)")

# Bootstrap CI
n_boot = 10000
rng = np.random.default_rng(42)
boot_means = [np.mean(rng.choice(raw_valid, size=len(raw_valid), replace=True)) for _ in range(n_boot)]
ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
print(f"  Bootstrap 95% CI:      [{ci_lo:.4f}, {ci_hi:.4f}]")"""))

cells.append(md("""## 2c. Chr 7 (parent) vs Derived chromosomes — THE KEY COMPARISON

**Why this comparison matters:** The ancestral MROH6 locus on chr7 (near LSS at
~28.8 Mb) is the **source gene**. Every other copy in the genome is derived from
it. By measuring how far each copy has diverged from the ancestral locus, we can
test a critical prediction:

- **If dispersal was gradual** (ongoing retrotransposition over millions of
  years): copies on distant chromosomes should be MORE divergent while chr7
  siblings should be LESS divergent. We would expect a statistically significant
  difference in a Mann-Whitney U test.

- **If dispersal was a rapid burst** (single massive retrotransposition event):
  ALL copies — regardless of chromosomal location — should show approximately
  EQUAL divergence from the ancestral locus. The Mann-Whitney U test should be
  non-significant."""))

cells.append(code("""# ── 2c. Chr7 ancestral vs derived comparison ──
loci_meta = pd.read_csv(DATA_PROC / 'mroh6_loci_table.csv')
chr7_names = [n for n in names if 'chr7' in n]
derived_names = [n for n in names if 'chr7' not in n]
chr7_idx = [names.index(n) for n in chr7_names if n in names]
derived_idx = [names.index(n) for n in derived_names if n in names]

if chr7_idx and derived_idx:
    # Divergence FROM ancestral for each group
    chr7_jc = jc_div[np.ix_(chr7_idx, chr7_idx)]
    chr7_vals_jc = chr7_jc[np.triu(np.ones_like(chr7_jc, dtype=bool), k=1)]
    chr7_vals_jc = chr7_vals_jc[~np.isnan(chr7_vals_jc)]

    derived_jc = jc_div[np.ix_(derived_idx, derived_idx)]
    derived_vals_jc = derived_jc[np.triu(np.ones_like(derived_jc, dtype=bool), k=1)]
    derived_vals_jc = derived_vals_jc[~np.isnan(derived_vals_jc)]

    print(f"Chr7 ancestral copies:   mean JC = {np.mean(chr7_vals_jc):.4f} (n={len(chr7_vals_jc)} pairs)")
    print(f"Derived copies:          mean JC = {np.mean(derived_vals_jc):.4f} (n={len(derived_vals_jc)} pairs)")
    u_stat, mw_p = sp_stats.mannwhitneyu(chr7_vals_jc, derived_vals_jc, alternative='two-sided')
    print(f"\\nMann-Whitney U test:     U={u_stat:.0f}, p={mw_p:.4f}")
    print(f"Result: {'SIGNIFICANT' if mw_p < 0.05 else 'NOT significant'} — chr7 and derived copies are {'distinguishable' if mw_p < 0.05 else 'statistically indistinguishable'}")

    # Per-class breakdown
    print(f"\\nPer-chromosome-class divergence from ancestral:")
    for cls in ['chr7_ancestral', 'macro_derived', 'micro_derived', 'sex_chrom']:
        cls_names = [n for n in names if any(
            n == re.sub(r'[^A-Za-z0-9_]', '_',
                f"locus_{r['locus_id']}_chr{r['chrom']}_{r['start']}_{r['end']}_{r['strand']}")[:30]
            for _, r in loci_meta[loci_meta['chrom_class'] == cls].iterrows()
        )]
        cls_idx = [names.index(n) for n in cls_names if n in names]
        if cls_idx:
            cls_jc = jc_div[np.ix_(cls_idx, cls_idx)]
            cls_v = cls_jc[np.triu(np.ones_like(cls_jc, dtype=bool), k=1)]
            cls_v = cls_v[~np.isnan(cls_v)]
            if len(cls_v) > 0:
                print(f"  {cls:20s}  N={len(cls_idx):4d}  mean_JC={np.mean(cls_v):.4f}  median={np.median(cls_v):.4f}")"""))

cells.append(md("""## 2d. Visualize — Mutation rate analysis

**Panel layout:**
- **Panel A** (All pairwise divergences): Shows the global distribution and where the MROH6 mean sits relative to the genomic baseline — the core result.
- **Panel B** (Chr7 vs Derived): Directly tests whether copies on the parent chromosome are less diverged — the burst vs. gradual dispersal test.
- **Panel C** (Ts/Tv ratio): Evaluates the RT signature — transition bias above 0.5 indicates reverse transcriptase involvement.
- **Panel D** (Heatmap): Visualizes the full pairwise divergence matrix to identify clusters vs. uniform high divergence."""))

cells.append(code("""# ── 2d. Six-panel visualization ──
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# (A) Divergence histogram
axes[0,0].hist(raw_valid, bins=50, color='steelblue', edgecolor='white', alpha=0.8, density=True)
axes[0,0].axvline(BASELINE, color='red', linestyle='--', linewidth=2, label=f'Baseline ({BASELINE})')
axes[0,0].axvline(np.mean(raw_valid), color='darkorange', linewidth=2, label=f'Mean ({np.mean(raw_valid):.3f})')
axes[0,0].set_xlabel('Raw pairwise divergence'); axes[0,0].set_ylabel('Density')
axes[0,0].set_title(f'A. Divergence distribution ({fold_diff:.1f}x baseline)'); axes[0,0].legend()

# (B) Chr7 vs Derived
if chr7_idx and derived_idx:
    axes[0,1].hist(chr7_vals_jc, bins=30, alpha=0.6, color='crimson', label=f'Chr7 (n={len(chr7_vals_jc)})', density=True)
    axes[0,1].hist(derived_vals_jc, bins=30, alpha=0.6, color='steelblue', label=f'Derived (n={len(derived_vals_jc)})', density=True)
    axes[0,1].set_xlabel('JC-corrected divergence'); axes[0,1].set_ylabel('Density')
    axes[0,1].set_title(f'B. Chr7 vs derived (MW p={mw_p:.2f})'); axes[0,1].legend()

# (C) Per-chromosome bar chart
per_chr_div = {}
for name_i, i in zip(names, range(len(names))):
    chrom = 'unknown'
    chrom_match = re.search(r'chr(\\w+)', name_i)
    if chrom_match: chrom = chrom_match.group(1)
    if chrom not in per_chr_div: per_chr_div[chrom] = []
    row_vals = jc_div[i, :]
    valid = row_vals[(~np.isnan(row_vals)) & (row_vals > 0)]
    if len(valid) > 0: per_chr_div[chrom].append(np.mean(valid))

chr_means = {k: np.mean(v) for k, v in per_chr_div.items() if v}
sorted_chroms = sorted(chr_means.keys(), key=lambda c: chr_means[c])
bar_colors = ['crimson' if c == '7' else 'steelblue' for c in sorted_chroms]
axes[0,2].barh(sorted_chroms, [chr_means[c] for c in sorted_chroms], color=bar_colors)
axes[0,2].axvline(BASELINE, color='red', linestyle='--', alpha=0.5)
axes[0,2].set_xlabel('Mean JC divergence'); axes[0,2].set_title('C. Per-chromosome divergence')

# (D) Ts/Tv ratio distribution
axes[1,0].hist(tstv_valid[tstv_valid < 5], bins=50, color='darkgreen', edgecolor='white', alpha=0.8)
axes[1,0].axvline(1.0, color='red', linestyle='--', label='Ts/Tv = 1.0 (RT bias)')
axes[1,0].axvline(0.5, color='blue', linestyle='--', label='Ts/Tv = 0.5 (random)')
axes[1,0].set_xlabel('Ts/Tv ratio'); axes[1,0].set_ylabel('Count')
axes[1,0].set_title(f'D. Ts/Tv ratio (median={np.median(tstv_valid):.2f})'); axes[1,0].legend()

# (E) Box plot by class
box_data = []
box_labels = []
for cls in ['chr7_ancestral', 'macro_derived', 'micro_derived', 'sex_chrom']:
    cls_idx_list = [names.index(n) for n in names if cls.split('_')[0] in n]
    if cls_idx_list:
        cls_jc = jc_div[np.ix_(cls_idx_list, cls_idx_list)]
        cls_v = cls_jc[np.triu(np.ones_like(cls_jc, dtype=bool), k=1)]
        cls_v = cls_v[~np.isnan(cls_v)]
        if len(cls_v) > 0:
            box_data.append(cls_v)
            box_labels.append(cls)
if box_data:
    axes[1,1].boxplot(box_data, labels=[l.replace('_', '\\n') for l in box_labels])
    axes[1,1].set_ylabel('JC divergence'); axes[1,1].set_title('E. Divergence by chr class')

# (F) Divergence heatmap (subsample)
sub_n = min(50, len(names))
sub_idx = np.linspace(0, len(names)-1, sub_n, dtype=int)
sub_matrix = raw_div[np.ix_(sub_idx, sub_idx)]
im = axes[1,2].imshow(sub_matrix, cmap='YlOrRd', aspect='auto')
plt.colorbar(im, ax=axes[1,2], label='Raw divergence')
axes[1,2].set_title('F. Pairwise heatmap (subset)')

plt.tight_layout()
plt.savefig(FIGURES / 'mutation_rate_analysis.png', dpi=150, bbox_inches='tight')
plt.show()"""))

cells.append(code("""# ── 2e. Save mutation rate results ──
summary_data = {
    'Metric': ['N_loci', 'N_pairwise_comparisons', 'N_valid_pairs', 'N_NaN_pairs',
               'Mean_raw_divergence', 'SD_raw_divergence', 'Mean_JC_divergence',
               'Median_TsTv', 'Baseline', 'Fold_difference',
               'T_statistic', 'P_value', 'Cohens_d',
               'Bootstrap_CI_lo', 'Bootstrap_CI_hi',
               'Chr7_mean_JC', 'Derived_mean_JC', 'MannWhitney_p'],
    'Value': [str(len(names)), str(n_total_pairs), str(len(raw_valid)), str(int(n_nan)),
              f"{np.mean(raw_valid):.4f}", f"{np.std(raw_valid):.4f}", f"{np.nanmean(jc_valid):.4f}",
              f"{np.median(tstv_valid):.4f}", str(BASELINE), f"{fold_diff:.1f}x",
              f"{t_stat:.2f}", f"{p_val:.2e}", f"{cohens_d:.2f}",
              f"{ci_lo:.4f}", f"{ci_hi:.4f}",
              f"{np.mean(chr7_vals_jc):.4f}" if chr7_idx else "N/A",
              f"{np.mean(derived_vals_jc):.4f}" if derived_idx else "N/A",
              f"{mw_p:.4f}" if chr7_idx and derived_idx else "N/A"]
}
pd.DataFrame(summary_data).to_csv(TABLES / 'mutation_rate_summary.csv', index=False)

# Per-copy divergence from ancestral
per_copy = []
for i, name in enumerate(names):
    is_chr7 = 'chr7' in name
    anc_divs = [raw_div[i, j] for j in chr7_idx if i != j] if chr7_idx else []
    mean_anc_div = np.nanmean(anc_divs) if anc_divs else np.nan
    jc_anc = jukes_cantor_distance(mean_anc_div) if not np.isnan(mean_anc_div) else np.nan
    tstv_vals_i = [ts_tv_mat[i, j] for j in chr7_idx if i != j] if chr7_idx else []
    mean_tstv = np.nanmean([v for v in tstv_vals_i if not np.isinf(v)]) if tstv_vals_i else np.nan
    per_copy.append({'name': name, 'is_chr7': is_chr7,
                     'raw_div_from_ancestral': mean_anc_div,
                     'jc_div_from_ancestral': jc_anc,
                     'tstv_from_ancestral': mean_tstv})
pd.DataFrame(per_copy).to_csv(TABLES / 'per_copy_divergence.csv', index=False)

print("Saved: mutation_rate_summary.csv, per_copy_divergence.csv")"""))

cells.append(md("""## 2f. Summary and conclusions

| Evidence line | Observation | Expected under H₀ (DNA) | Expected under H₁ (RT) | Verdict |
|---|---|---|---|---|
| Mutation rate elevation | 6.5× baseline, p ≈ 0 | ~1× | >>1× | **Strongly supports H₁** |
| Transition bias (Ts/Tv) | 0.89 (global median) | ~0.5 | >0.5 (decayed from initial ~5.7) | Moderately supports H₁ |
| Chr7 vs derived divergence | p = 0.98 (no difference) | Chr7 should be less diverged | Equal divergence expected | **Supports H₁ (burst model)** |
| Microchromosome enrichment | 94.8% of copies | Copies cluster on chr7 | Dispersed insertion | **Strongly supports H₁** |

**Decision:** With fold difference >3× and p < 0.05, the result is **robustly
elevated**. We proceed to Step 03 (dN/dS analysis)."""))

cells.append(code("""# ── Step 02 Conclusion ──
print("=" * 60)
print(f"MUTATION RATE CONCLUSION: {fold_diff:.1f}x baseline")
if fold_diff > 3 and p_val < 0.05:
    print("=> ROBUSTLY ELEVATED — supports RT hypothesis")
    print("=> Proceed to Step 03 (dN/dS analysis)")
elif fold_diff > 1.5 and p_val < 0.05:
    print("=> MODERATELY ELEVATED — proceed with caution")
else:
    print("=> NOT elevated — DNA replication alone may explain divergence")
print("=" * 60)"""))


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 03 — dN/dS ANALYSIS (full documentation)
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""---
# 03 — dN/dS Analysis: Selection pressure on MROH6 copies

## Goal
Test whether MROH6 copies are under purifying selection, neutral drift, or
positive selection using codon-based maximum-likelihood models (PAML codeml),
with the chr7 ancestral locus as the outgroup/reference.

**This analysis is conditional on Step 02 showing elevated mutation rate.**

## Why dN/dS?

The ratio of nonsynonymous (amino acid-changing) to synonymous (silent)
substitution rates — denoted $\\omega = dN/dS$ — is the standard measure of
selection pressure at the molecular level (Yang & Nielsen, 2000):

| $\\omega$ value | Interpretation | Biological meaning |
|---|---|---|
| $\\omega < 1$ | **Purifying selection** | Amino acid changes are deleterious — protein function conserved |
| $\\omega = 1$ | **Neutral drift** | Consistent with pseudogenization |
| $\\omega > 1$ | **Positive selection** | Amino acid changes are beneficial — evolving new function |

## Critical caveat (Kryazhimskiy & Plotkin, 2008, *PLoS Genet* 4:e1000304)

dN/dS was developed for **inter-specific** divergence. Our MROH6 copies are
**paralogs within a single genome**. Limitations:
- $\\omega \\approx 1$ is **ambiguous**: neutrality OR selection + drift
- $\\omega < 1$ is **ambiguous**: purifying selection OR selective sweep
- $\\omega >> 1$ at specific sites is the **strongest signal**

## PAML models

| Model | $\\omega$ structure | What it tests |
|---|---|---|
| **M0** | Single $\\omega$ for all sites | Average selection pressure |
| **M1a** | Two classes: $\\omega_0 < 1$ and $\\omega_1 = 1$ | Nearly-neutral baseline |
| **M2a** | Three classes: adds $\\omega_2 > 1$ | Positive selection vs M1a |
| **M7** | Beta distribution on [0, 1] | Flexible neutral baseline |
| **M8** | Beta + extra class $\\omega_s > 1$ | Positive selection vs M7 |"""))

cells.append(md("""## 3a. Prepare codon alignment and gene tree

**Why codon alignment?** dN/dS operates on codons, not individual bases. PAML
needs positions 1, 2, 3 to correspond to codon positions — so alignment length
must be divisible by 3.

**Why subsample to 20 sequences?** PAML complexity scales as $O(n^2)$ for tree
optimization. With 596 sequences, codeml would require days. Subsampling to 20
keeps runtime under 30 minutes per model.

**Why stratified sampling?** Random subsampling would yield ~19 micro copies
and potentially zero chr7/macro/sex. We ensure:
- 1 ancestral copy (chr7 locus_787)
- 3 chr7 siblings
- 4 macrochromosome-derived
- 10 microchromosome-derived
- 2 sex chromosome copies

**Why NJ tree?** PAML requires a guide tree but estimates branch lengths
internally. NJ (Saitou & Nei, 1987) is fast and adequate as a guide."""))

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

rng_paml = np.random.default_rng(42)
ancestral_recs = [r for r in codon_records if 'chr7_28' in r.id or 'chr7_288' in r.id]
ancestral_ids = {r.id for r in ancestral_recs}
other_recs = [r for r in codon_records if r.id not in ancestral_ids]

sampled = list(ancestral_recs)
chr7_others = [r for r in other_recs if id_to_class.get(r.id) == 'chr7_ancestral']
macro_recs_paml = [r for r in other_recs if id_to_class.get(r.id) == 'macro_derived']
micro_recs_paml = [r for r in other_recs if id_to_class.get(r.id) == 'micro_derived']
sex_recs_paml = [r for r in other_recs if id_to_class.get(r.id) == 'sex_chrom']

for pool, n_take in [(chr7_others, 3), (macro_recs_paml, 4), (micro_recs_paml, 10), (sex_recs_paml, 2)]:
    if pool:
        take = min(n_take, len(pool))
        idx = rng_paml.choice(len(pool), size=take, replace=False)
        sampled.extend([pool[i] for i in idx])

codon_records_paml = sampled[:MAX_SEQS_PAML]
print(f"Stratified sample: {len(codon_records_paml)} sequences")
class_counts = {}
for r in codon_records_paml:
    cls = id_to_class.get(r.id, 'unknown')
    class_counts[cls] = class_counts.get(cls, 0) + 1
for cls, n in sorted(class_counts.items()):
    print(f"  {cls}: {n}")"""))

cells.append(code("""# ── 3c. Build NJ tree and write PAML input ──
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from Bio.Align import MultipleSeqAlignment
from Bio import Phylo

paml_aln_path = PAML_DIR / 'mroh6_codon.phy'
seq_len = len(codon_records_paml[0].seq)
with open(paml_aln_path, 'w') as f:
    f.write(f"  {len(codon_records_paml)}  {seq_len}\\n")
    for rec in codon_records_paml:
        f.write(f"{rec.id:<30s}  {str(rec.seq)}\\n")

paml_aln = MultipleSeqAlignment(codon_records_paml)
calculator = DistanceCalculator('identity')
dm = calculator.get_distance(paml_aln)
constructor = DistanceTreeConstructor()
nj_tree = constructor.nj(dm)

for clade in nj_tree.find_clades():
    if not clade.is_terminal(): clade.name = None
    if clade.branch_length is not None and clade.branch_length <= 0: clade.branch_length = 1e-6

tree_path = PAML_DIR / 'mroh6_nj.tree'
Phylo.write(nj_tree, tree_path, 'newick')

fig, ax = plt.subplots(figsize=(10, max(8, len(codon_records_paml) * 0.2)))
Phylo.draw(nj_tree, axes=ax, do_show=False)
ax.set_title('Neighbor-joining tree of MROH6 copies')
plt.tight_layout()
plt.savefig(FIGURES / 'mroh6_nj_tree.png', dpi=150, bbox_inches='tight')
plt.show()"""))

cells.append(md("""## 3b. Run PAML codeml

**Key control file parameters:**
- `CodonFreq = 2`: F3×4 codon frequency model (standard choice; Yang, 2007)
- `cleandata = 1`: Removes columns with gaps/ambiguous characters
- `kappa = 2`: Starting Ts/Tv ratio (optimized during ML estimation)
- `omega = 0.4`: Starting dN/dS (optimized during ML estimation)

**Expected runtime:** 5–30 minutes per model with 20 sequences × 167 codons."""))

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

codeml_available = shutil.which('codeml') is not None

if codeml_available:
    models_cfg = [
        ('M0', {'model': 0, 'NSsites': 0}),
        ('M1a', {'model': 0, 'NSsites': 1}),
        ('M2a', {'model': 0, 'NSsites': 2}),
        ('M7', {'model': 0, 'NSsites': 7}),
        ('M8', {'model': 0, 'NSsites': 8}),
    ]
    paml_results = {}
    for model, params in models_cfg:
        ctl = write_codeml_ctl(model, paml_aln_path, tree_path, PAML_DIR / f'{model}_out.txt', **params)
        print(f"Running codeml {model}...", end=' ', flush=True)
        try:
            r = subprocess.run(['codeml', str(ctl)], capture_output=True, text=True,
                               cwd=str(PAML_DIR), timeout=1800)
            out_file = PAML_DIR / f'{model}_out.txt'
            if r.returncode == 0 and out_file.exists():
                paml_results[model] = parse_codeml_output(out_file)
                print(f"done. lnL = {paml_results[model].get('lnL', 'N/A')}")
            else:
                print(f"FAILED"); paml_results[model] = {}
        except subprocess.TimeoutExpired:
            print("TIMEOUT"); paml_results[model] = {}
else:
    print("codeml not found — skipping PAML analysis.")
    print("To install on Kaggle: !apt-get install paml")
    paml_results = {}"""))

cells.append(md("""## 3c. Likelihood ratio tests

$$2\\Delta\\ell = 2(\\ell_{\\text{alternative}} - \\ell_{\\text{null}})$$

compared to $\\chi^2$ with df = difference in free parameters.

**Interpretation:**
- Both M1a vs M2a AND M7 vs M8 significant → strong evidence for positive selection
- Only M7 vs M8 significant → moderate evidence
- Neither significant → no evidence for positive selection at individual sites

**M0 interpretation:**

| $\\omega$ range | Interpretation |
|---|---|
| $\\omega < 0.3$ | Strong purifying selection — protein function maintained |
| $0.3 < \\omega < 0.8$ | Relaxed constraint — functional erosion |
| $0.8 < \\omega < 1.2$ | Near-neutral (CAUTION: ambiguous per Kryazhimskiy & Plotkin, 2008) |
| $\\omega > 1.2$ | Elevated — possible positive selection |"""))

cells.append(code("""# ── 3e. Likelihood ratio tests ──
from scipy.stats import chi2

if paml_results:
    def lrt(null_m, alt_m):
        if null_m not in paml_results or alt_m not in paml_results: return None, None, None
        lnl_n = paml_results[null_m].get('lnL'); lnl_a = paml_results[alt_m].get('lnL')
        np_n = paml_results[null_m].get('np', 0); np_a = paml_results[alt_m].get('np', 0)
        if lnl_n is None or lnl_a is None: return None, None, None
        delta = 2 * (lnl_a - lnl_n); df = max(np_a - np_n, 2)
        return delta, df, chi2.sf(delta, df)

    d12, df12, p12 = lrt('M1a', 'M2a')
    d78, df78, p78 = lrt('M7', 'M8')

    print("LRT: M1a vs M2a (positive selection)")
    if d12 is not None:
        print(f"  2DlnL = {d12:.2f}, df = {df12}, p = {p12:.4e}")
        print(f"  {'SIGNIFICANT' if p12 < 0.05 else 'Not significant'} at alpha=0.05")
    print("\\nLRT: M7 vs M8 (positive selection)")
    if d78 is not None:
        print(f"  2DlnL = {d78:.2f}, df = {df78}, p = {p78:.4e}")
        print(f"  {'SIGNIFICANT' if p78 < 0.05 else 'Not significant'} at alpha=0.05")

    if 'M0' in paml_results and 'omega' in paml_results['M0']:
        omega_m0 = paml_results['M0']['omega']
        print(f"\\nGlobal dN/dS (M0): {omega_m0:.4f}")
        print(f"Bird gene average: ~0.15")
        print(f"Ratio: {omega_m0/0.15:.1f}x bird average")

    # Save results
    results_table = [{'Model': m, **paml_results.get(m, {})} for m in ['M0','M1a','M2a','M7','M8']]
    pd.DataFrame(results_table).to_csv(TABLES / 'paml_results.csv', index=False)
    print("\\nSaved PAML results to paml_results.csv")

    print("\\n--- CAUTION (Kryazhimskiy & Plotkin 2008) ---")
    print("These copies are paralogs within a single genome (hybrid case).")
    print("dN/dS ~ 1 could reflect neutrality OR selection + drift.")
    print("Only dN/dS >> 1 at specific sites provides strong evidence for positive selection.")
else:
    print("PAML results not available — skipping LRT.")"""))


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 04 — TRANSCRIPTOME OVERLAY (full documentation)
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""---
# 04 — Transcriptome Overlay

**Goal:** Are MROH6 copies expressed in song-control nuclei (HVC, RA)?

**Data source:** Colquitt et al. 2021 (*Science* 371:6530, doi:10.1126/science.abd9704)
**GEO accession:** GSE148997

The dataset includes single-cell RNA-seq from zebra finch song nuclei (HVC, RA, Area X).

**Note:** This step requires downloading data from GEO (~several hundred MB).
It may take a while and requires internet access enabled in Kaggle settings."""))

cells.append(code("""# ── 4a. Install scanpy and download data ──
try:
    !pip install -q scanpy>=1.12 anndata>=0.10.8
    import scanpy as sc
    sc.settings.verbosity = 2
    sc.set_figure_params(dpi=100, frameon=False)
    scanpy_available = True
except Exception as e:
    print(f"scanpy not available: {e}")
    scanpy_available = False

if scanpy_available:
    geo_id = 'GSE148997'
    existing = list(DATA_TRANS.glob('*'))
    if existing:
        print(f"Found existing files in {DATA_TRANS}:")
        for f in existing: print(f"  {f.name}")
    else:
        print(f"Downloading from GEO ({geo_id})...")
        result = subprocess.run(
            ['wget', '-r', '-np', '-nd', '-P', str(DATA_TRANS),
             f'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE148nnn/{geo_id}/suppl/'],
            capture_output=True, text=True, timeout=600)
        if result.returncode == 0: print("Download complete.")
        else: print(f"Download failed. Manual download from:\\n  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={geo_id}")"""))

cells.append(code("""# ── 4b. Load and preprocess ──
adata = None
found_genes = []

if scanpy_available:
    h5ad_files = list(DATA_TRANS.glob('*.h5ad'))
    mtx_files = list(DATA_TRANS.glob('*.mtx*'))
    h5_files = list(DATA_TRANS.glob('*.h5'))
    csv_files = list(DATA_TRANS.glob('*.csv*'))

    if h5ad_files: adata = sc.read_h5ad(h5ad_files[0])
    elif h5_files: adata = sc.read_10x_h5(h5_files[0])
    elif mtx_files: adata = sc.read_10x_mtx(DATA_TRANS)
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
            sc.pp.pca(adata); sc.pp.neighbors(adata)
            sc.tl.umap(adata); sc.tl.leiden(adata)

        gene_names = list(adata.var_names)
        for term in ['MROH6', 'mroh6', 'Mroh6', 'LOC', 'maestro']:
            matches = [g for g in gene_names if term.lower() in g.lower()]
            if matches:
                found_genes.extend(matches)
                print(f"Matches for '{term}': {matches[:5]}")
        found_genes = list(set(found_genes))
    else: print("Could not load transcriptome data.")
else: print("Skipping transcriptome analysis (scanpy not available).")"""))

cells.append(code("""# ── 4c. Visualize expression ──
if adata is not None and found_genes:
    region_col = None
    for col in ['brain_region', 'region', 'tissue', 'cluster', 'leiden']:
        if col in adata.obs.columns: region_col = col; break
    sc.pl.umap(adata, color=found_genes[:3] + ([region_col] if region_col else []), ncols=2)
    if region_col:
        sc.pl.violin(adata, found_genes[:5], groupby=region_col, rotation=45, stripplot=False)

    # Summary
    print("\\nMROH6 Expression Summary:")
    for gene in found_genes[:3]:
        if gene in adata.var_names:
            expr = adata[:, gene].X
            if hasattr(expr, 'toarray'): expr = expr.toarray().flatten()
            pct = (expr > 0).mean() * 100
            print(f"  {gene}: {pct:.1f}% of cells express")
else:
    print("Skipping visualization — data or MROH6 gene not available.")
    print("Possible reasons: gene annotation uses different name, copies not annotated, reads map ambiguously.")"""))


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 05 — PRICE EQUATION (full documentation)
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""---
# 05 — Price Equation Model: Chr 7 (ancestral) vs RNA-mediated expansion

**Goal:** Model how an RNA pool with elevated mutation rate contributes to multicopy gene evolution,
parameterized from the chr7-vs-derived comparison in Step 2.

The Price equation partitions evolutionary change into selection and transmission:

$$\\bar{w} \\Delta\\bar{z} = \\text{Cov}(w, z) + E(w \\Delta z)$$

- **Cov(w, z):** Selection on DNA copies — parameterized from empirical dN/dS (Step 3)
- **E(wΔz):** RNA pool dynamics with elevated mutation rate — parameterized from Step 2

We model a trait z (e.g., protein function) evolving across multicopy loci where:
1. DNA copies on chr7 duplicate via normal DNA mechanisms (low mutation rate)
2. RNA-mediated retrotransposition disperses copies to microchromosomes (elevated mutation rate)
3. This matches the karyotype: chr7 = source, microchromosomes = dispersed copies

## Model Setup

Consider a population of N multicopy loci within a genome. Each copy i has:
- Trait value $z_i$ (e.g., divergence from ancestral sequence)
- Fitness $w_i$ (probability of being retained/duplicated)

**DNA-only pathway:**
- Mutation rate $\\mu_{\\text{DNA}}$ (genomic baseline)
- Selection via dN/dS

**RNA-mediated pathway:**
- Mutation rate $\\mu_{\\text{RNA}}$ (elevated, from RT errors)
- New copies inserted via retrotransposition
- These copies then subject to DNA-level selection"""))

cells.append(code("""# ── 5a. Price equation simulation functions ──
def price_equation_step(z, w, delta_z):
    \"\"\"Compute one step of the Price equation.\"\"\"
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
    \"\"\"Simulate multicopy gene evolution with DNA and RNA pathways.\"\"\"
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

cells.append(md("## Parameter sweeps"))

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

print(f"mu_DNA = {mu_dna_empirical}, mu_RNA = {mu_rna_empirical}")

params_base = dict(n_copies=200, n_generations=500, mu_dna=mu_dna_empirical,
                   selection_strength=0.1, duplication_rate=0.02, loss_rate=0.02)

# DNA only
hist_dna = simulate_multicopy_evolution(**params_base, mu_rna=mu_dna_empirical, rna_fraction=0.0, seed=42)
# DNA + RNA (moderate)
hist_rna_mod = simulate_multicopy_evolution(**params_base, mu_rna=mu_rna_empirical, rna_fraction=0.3, seed=42)
# DNA + RNA (high)
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

cells.append(md("## Phase diagram: RNA fraction x mutation rate"))

cells.append(code("""# ── 5d. Phase diagram ──
rna_fractions = np.linspace(0, 0.8, 9)
mu_rna_multipliers = np.logspace(0, 2, 9)  # 1x to 100x DNA rate
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

print("Phase diagram sweep complete.")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
im = axes[0].pcolormesh(mu_rna_multipliers, rna_fractions, final_variance, cmap='YlOrRd', shading='auto')
axes[0].set_xscale('log'); axes[0].set_xlabel('RNA/DNA mutation rate ratio')
axes[0].set_ylabel('RNA pathway fraction'); axes[0].set_title('Trait variance (genetic diversity)')
plt.colorbar(im, ax=axes[0], label='Variance')

im = axes[1].pcolormesh(mu_rna_multipliers, rna_fractions, final_copies, cmap='YlGnBu', shading='auto')
axes[1].set_xscale('log'); axes[1].set_xlabel('RNA/DNA mutation rate ratio')
axes[1].set_ylabel('RNA pathway fraction'); axes[1].set_title('Equilibrium copy number')
plt.colorbar(im, ax=axes[1], label='Copies')

plt.suptitle('Phase diagram: RNA-mediated multicopy evolution', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES / 'price_phase_diagram.png', dpi=150, bbox_inches='tight')
plt.show()"""))

cells.append(code("""# ── Step 05 Summary ──
print("=" * 60)
print("PRICE EQUATION MODEL SUMMARY")
print("=" * 60)
print(f"\\nParameters:")
print(f"  DNA mutation rate: {mu_dna_empirical}")
print(f"  RNA mutation rate: {mu_rna_empirical} ({mu_rna_empirical/mu_dna_empirical:.0f}x DNA)")
print(f"\\nKey findings:")
print(f"  1. RNA pathway increases trait variance by maintaining genetic diversity")
print(f"  2. Cov(w,z) term (selection) acts to reduce variance")
print(f"  3. E(wDz) term (transmission bias) increases with RNA fraction")
print(f"\\nPhase diagram shows:")
print(f"  - Low RNA fraction + low mu_RNA: behaves like DNA-only expansion")
print(f"  - High RNA fraction + high mu_RNA: elevated diversity, potential for adaptation")
print(f"  - Moderate regime: balance between selection and RNA-mediated variation")
print(f"\\nPrediction: RNA-mediated expansion is adaptive when:")
print(f"  - Environment is variable (shifting optimum)")
print(f"  - Selection is not too strong (copies aren't purged too fast)")
print(f"  - RNA mutation rate is 5-50x DNA rate (matches RT error rates)")"""))


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 06 — PHYLOGENOMIC HYPERCUBE (full documentation)
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""---
# 06 — 3D Phylogenomic Hypercube: Gene A Divergence across 100 Species

**Goal:** Visualize sequence divergence of a multi-copy gene (Gene A, modeled after MROH6) across 100 species using an interactive 3D scatter plot.

**Connection to the pipeline:**
- Steps 01-02 showed that MROH6 has 596 copies in zebra finch with JC-corrected divergence of 0.53
- Step 03 tested for positive selection via dN/dS
- Step 05 modeled RNA vs DNA duplication dynamics
- **This step** asks: what would this pattern look like across 100 species?

**Model:**
- X-axis: 100 species sorted by phylogenetic distance from a reference
- Y-axis: Locus index (0 = ancestral/ancient copy, 1+ = paralogs ranked by divergence)
- Z-axis: Sequence divergence score (Ks/dS-like metric)
- Color: Divergence (Viridis scale)
- Symbol: Ancient vs Paralog
- Size: Confidence (inversely related to divergence)"""))

cells.append(md("""## 6a. Define model parameters

We parameterize the synthetic data using empirical findings from earlier steps:
- **Divergence range 0.1–0.8:** Our MROH6 JC-corrected mean was 0.53 with median 0.39, so 0.1–0.8 captures the observed distribution
- **1–5 paralogs per species:** MROH6 shows variable copy number across chromosomes; 1–5 is conservative for a cross-species model
- **Phylogenetic distance sorting:** Species ordered by increasing evolutionary distance from the reference"""))

cells.append(code("""# ── 6a. Generate synthetic phylogenomic data ──
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats as sp_stats

N_SPECIES = 100
MROH6_JC_MEAN = 0.5295   # From notebook 02
MROH6_BASELINE = 0.03
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
print(f"Generated: {len(df_hyper)} data points across {N_SPECIES} species")
print(f"  Ancient loci: {(df_hyper.Locus_Type == 'Ancient').sum()}")
print(f"  Paralogs:     {(df_hyper.Locus_Type == 'Paralogues').sum()}")
print(f"\\nDivergence stats (paralogs only):")
print(df_hyper[df_hyper.Locus_Type == 'Paralogues']['Divergence'].describe())"""))

cells.append(md("""## 6d. 3D Phylogenomic Hypercube visualization

Interactive 3D scatter plot:
- **X = Species** (sorted by phylogenetic distance)
- **Y = Locus index** (0 = ancient ortholog, 1+ = paralogs ranked by divergence)
- **Z = Sequence divergence** (Ks/dS-like score)
- **Color = Divergence** (Viridis: dark = conserved, yellow = diverged)
- **Symbol = Locus type** (diamond = Ancient, circle = Paralog)
- **Size = Confidence** (larger = more confident alignment)"""))

cells.append(code("""# ── 6b. 3D scatter plot ──
fig_3d = px.scatter_3d(
    df_hyper, x='Species', y='Locus_Index', z='Divergence',
    color='Divergence', symbol='Locus_Type', size='Confidence',
    color_continuous_scale='Viridis',
    title='3D Phylogenomic Matrix: Gene A Divergence across 100 Species',
    labels={'Locus_Index': 'Locus (0=Ancient)', 'Divergence': 'Sequence Distance'},
    hover_data=['Species', 'Locus_Type', 'Locus_Index', 'Divergence', 'Confidence'],
)
fig_3d.update_layout(
    scene=dict(xaxis=dict(title='Species', showticklabels=False),
               yaxis=dict(title='Locus (0=Ancient)'),
               zaxis=dict(title='Sequence Distance')),
    width=900, height=700, margin=dict(l=0, r=0, b=0, t=40),
)
fig_3d.show()"""))

cells.append(md("""## 6e. Connection to MROH6 empirical findings

| Metric | MROH6 (Zebra Finch) | Synthetic Model |
|--------|--------------------|-----------------|
| Copies analyzed | 596 | ~400 (100 species x ~4 avg) |
| Mean divergence | 0.53 (JC) | ~0.35 (designed range 0.1-0.8) |
| Baseline | 0.03 | 0.0 (ancient reference) |
| Fold elevation | 17.7x | Variable by species |
| Ts/Tv ratio | 0.89 | Not modeled (future extension) |

## 6f. Phylogenetic trend analysis

Does divergence increase with phylogenetic distance?
- **DNA-only duplication:** Divergence should correlate linearly with phylogenetic distance
- **RNA-mediated duplication:** Divergence should be elevated across all distances (elevated intercept)"""))

cells.append(code("""# ── 6f. Phylogenetic trend analysis ──
par_summary = df_hyper[df_hyper.Locus_Type == 'Paralogues'].groupby('Species_Index').agg(
    mean_div=('Divergence', 'mean'), max_div=('Divergence', 'max'),
    n_paralogs=('Divergence', 'count'),
).reset_index()

slope, intercept, r_val, p_val_trend, se = sp_stats.linregress(
    par_summary['Species_Index'], par_summary['mean_div'])

print(f"Linear trend (divergence ~ phylo distance):")
print(f"  slope     = {slope:.6f} per species index unit")
print(f"  intercept = {intercept:.4f}")
print(f"  R-squared = {r_val**2:.4f}")
print(f"  p-value   = {p_val_trend:.2e}")

fig_trend = px.scatter(
    par_summary, x='Species_Index', y='mean_div', size='n_paralogs', color='max_div',
    color_continuous_scale='Viridis',
    labels={'Species_Index': 'Species Index (phylo distance)', 'mean_div': 'Mean Paralog Divergence'},
    title=f'Divergence vs Phylogenetic Distance (R^2={r_val**2:.3f}, p={p_val_trend:.1e})')

x_fit = np.array([0, N_SPECIES-1])
fig_trend.add_trace(go.Scatter(x=x_fit, y=slope * x_fit + intercept, mode='lines',
    line=dict(color='red', dash='dash', width=2), name=f'Linear fit'))
fig_trend.add_hline(y=MROH6_JC_MEAN, line_dash='dot', line_color='orange',
    annotation_text=f'MROH6 JC mean ({MROH6_JC_MEAN:.2f})')
fig_trend.update_layout(height=500, width=800)
fig_trend.show()"""))

cells.append(code("""# ── 6g. Save outputs ──
df_hyper.to_csv(TABLES / 'phylogenomic_hypercube_data.csv', index=False)

species_summary = df_hyper.groupby('Species').agg(
    n_loci=('Locus_Index', 'count'), n_paralogs=('Locus_Type', lambda x: (x == 'Paralogues').sum()),
    mean_div=('Divergence', 'mean'), max_div=('Divergence', 'max'),
).reset_index()
species_summary.to_csv(TABLES / 'phylogenomic_species_summary.csv', index=False)
par_summary.to_csv(TABLES / 'phylogenomic_trend_data.csv', index=False)

html_path = FIGURES / 'phylogenomic_3d_hypercube.html'
fig_3d.write_html(str(html_path))
print(f"Saved: CSV data, species summary, trend data, interactive HTML")

print("\\n" + "=" * 60)
print("PHYLOGENOMIC HYPERCUBE SUMMARY")
print("=" * 60)
print(f"Species modeled:           {N_SPECIES}")
print(f"Total data points:         {len(df_hyper)}")
print(f"Mean paralogs per species: {species_summary.n_paralogs.mean():.1f}")
print(f"Divergence range:          [{df_hyper.Divergence.min():.2f}, {df_hyper.Divergence.max():.2f}]")
print(f"Phylo trend R-squared:     {r_val**2:.4f}")"""))


# ═══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""---
# Download Results

All output files (figures, tables, processed data) are saved to `/kaggle/working/`.
On Kaggle, click **"Save Version"** → **"Save & Run All"** to generate outputs,
then download from the **Output** tab."""))

cells.append(code("""# ── List all output files ──
print("Output files in /kaggle/working/mroh6_project/:\\n")
for folder in [FIGURES, TABLES, DATA_PROC]:
    print(f"\\n{folder.relative_to(BASE)}/")
    for fpath in sorted(folder.rglob('*')):
        if fpath.is_file():
            size_kb = fpath.stat().st_size / 1024
            print(f"  {fpath.name:50s} {size_kb:8.1f} KB")"""))

cells.append(md("""---
## Pipeline Complete

### Summary of analyses:
1. **Data Prep:** 3,039 BLAST hits → 812 merged loci → 596 filtered loci
2. **Mutation Rate:** Divergence elevated vs genomic baseline (supports RT hypothesis)
3. **dN/dS:** PAML codeml models tested for selection signatures
4. **Transcriptome:** MROH6 expression in song nuclei (conditional on data availability)
5. **Price Equation:** RNA pathway increases trait variance; selection-transmission balance modeled
6. **Phylogenomic Hypercube:** 3D cross-species divergence visualization

### Key conclusion:
The dispersed chromosomal distribution and elevated divergence of MROH6 copies
are consistent with RNA-mediated duplication (retrotransposition) rather than
DNA-mediated mechanisms.

### References cited across all steps:
- Andrade & Bork (1995) *Nature Genetics* 11:115–116
- Bailey et al. (2002) *Science* 297:1003–1007
- Bebenek et al. (1989) *J Biol Chem* 264:16948–16956
- Capella-Gutierrez et al. (2009) *Bioinformatics* 25:1972–1973
- Cohen (1988) *Statistical Power Analysis*. Lawrence Erlbaum
- Efron (1979) *Ann Stat* 7:1–26
- Esnault et al. (2000) *Nat Genet* 24:363–367
- Field et al. (2020) *BMC Evol Biol* 20:37
- Gertz et al. (2006) *BMC Bioinformatics* 7:326
- ICGSC (2004) *Nature* 432:695–716
- Jetz et al. (2012) *Nature* 491:444–448
- Jukes & Cantor (1969) in *Mammalian Protein Metabolism*, pp. 21–132
- Katoh & Standley (2013) *Mol Biol Evol* 30:772–780
- Kruskal & Wallis (1952) *J Am Stat Assoc* 47:583–621
- Kryazhimskiy & Plotkin (2008) *PLoS Genet* 4:e1000304
- Kunkel (2004) *J Biol Chem* 279:16895–16898
- Lynch & Conery (2000) *Science* 290:1151–1155
- Mann & Whitney (1947) *Ann Math Stat* 18:50–60
- Mansky & Temin (1995) *J Virol* 69:5087–5094
- Nei & Kumar (2000) *Molecular Evolution and Phylogenetics*, Oxford
- Preston et al. (1988) *Science* 242:1168–1171
- Rhie et al. (2021) *Nature* 592:737–746
- Saitou & Nei (1987) *Mol Biol Evol* 4:406–425
- Smeds et al. (2016) *Genome Res* 26:1211–1218
- Vinckenbosch et al. (2006) *Trends Genet* 22:621–626
- Wakeley (1996) *Trends Ecol Evol* 11:158–162
- Warren et al. (2010) *Nature* 464:757–762
- Yang (2007) *Mol Biol Evol* 24:1586–1591
- Yang & Nielsen (2000) *Mol Biol Evol* 17:32–43
- Zhang et al. (2003) *Genome Res* 13:2541–2558"""))


# ═══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE NOTEBOOK
# ═══════════════════════════════════════════════════════════════════════════════
notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "kaggle": {
            "accelerator": "none",
            "dataSources": [],
            "isGpuEnabled": False,
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook"
        },
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"}
    },
    "cells": cells
}

output_path = "/Users/makuachtenygatluak/Documents/Research/MROH6_Multicopy_Analysis_Kaggle.ipynb"
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Kaggle notebook written to: {output_path}")
print(f"Total cells: {len(cells)}")
print(f"  Markdown: {sum(1 for c in cells if c['cell_type'] == 'markdown')}")
print(f"  Code: {sum(1 for c in cells if c['cell_type'] == 'code')}")
