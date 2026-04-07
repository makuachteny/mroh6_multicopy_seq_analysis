#!/usr/bin/env python3
"""
03d — Per-Site dN/dS Visualization & BEB Selected Regions
==========================================================

QUESTION:
  Where along the protein are sites under positive selection?
  Are the positively selected sites clustered, or scattered?
  What are the specific mutations at each site, and what are the
  average dN/dS rates across the protein?

REASONING:
  PAML codeml M2a and M8 models identify sites under positive selection
  via Bayes Empirical Bayes (BEB) posterior probabilities. The rst file
  contains per-site posterior mean omega values for every codon in the
  alignment. Plotting these along the protein reveals:

    1. Regions of elevated omega (potential functional divergence)
    2. Clustering of BEB sites (suggests domain-level selection)
    3. The specific amino acid mutations at positively selected sites
    4. Average and per-region dN/dS patterns

  This step produces:
    - Per-site omega plot along the protein (from rst file)
    - BEB site map with significance levels (P>0.95 *, P>0.99 **)
    - Mutation matrix at BEB sites across all copies
    - Clustering analysis (sliding window, nearest-neighbor distances)
    - Summary table of all BEB sites with amino acids and omega values

Usage:
  python steps/03d_site_selection_plot.py --species zebra_finch
"""
import sys
import argparse
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'scripts'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from Bio import AlignIO, SeqIO
from Bio.Seq import Seq
from pathlib import Path
from collections import Counter
import re

from species_config import load_config, get_data_dirs

parser = argparse.ArgumentParser(description='Step 03d: Per-site dN/dS & BEB visualization')
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
print(f"  STEP 03d: PER-SITE dN/dS & BEB VISUALIZATION — {GENE} in {SPECIES}")
print("=" * 70)


# ── Helper: Parse BEB sites from codeml output ─────────────────────────
def parse_beb_sites(outfile):
    """Parse BEB (Bayes Empirical Bayes) results from codeml M2a or M8 output.

    Returns a list of dicts with site, aa, prob, omega, se fields.
    """
    sites = []
    if not outfile.exists():
        return sites
    with open(outfile) as f:
        text = f.read()

    # Find the BEB section (not NEB)
    beb_match = re.search(
        r'Bayes Empirical Bayes \(BEB\).*?Positively selected sites.*?\n\n'
        r'\s+Pr\(w>1\)\s+post mean \+- SE for w\n\n(.*?)(?:\n\n|\nThe grid)',
        text, re.DOTALL
    )
    if not beb_match:
        return sites

    for line in beb_match.group(1).strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            site_num = int(parts[0])
            aa = parts[1]
            prob = float(parts[2].replace('*', ''))
            omega_val = float(parts[3])
            se = float(parts[5]) if len(parts) >= 6 else 0.0
            sig = ''
            if '**' in line:
                sig = '**'
            elif '*' in line:
                sig = '*'
            sites.append({
                'site': site_num,
                'aa': aa,
                'prob': prob,
                'omega': omega_val,
                'se': se,
                'significance': sig
            })
        except (ValueError, IndexError):
            continue
    return sites


# ── Helper: Parse per-site omega from rst file ─────────────────────────
def parse_rst_per_site(rst_path):
    """Parse per-site posterior omega from PAML rst file (M8 model output).

    Returns a list of dicts with site, aa, postmean_w, p_w_gt1 fields.
    """
    sites = []
    if not rst_path.exists():
        return sites
    with open(rst_path) as f:
        text = f.read()

    # Find NEB section for M8 (last model in rst) — the per-site table
    # Format: "   1 L   0.81083 0.06303 ... ( 1)  0.421  0.000"
    neb_match = re.search(
        r'Naive Empirical Bayes \(NEB\) probabilities for \d+ classes.*?\n'
        r'\(amino acids refer to 1st sequence:.*?\)\n\n(.*?)(?:\nPositively selected|$)',
        text, re.DOTALL
    )
    if not neb_match:
        return sites

    for line in neb_match.group(1).strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            site_num = int(parts[0])
            aa = parts[1]
            # Post mean omega is second-to-last column, P(w>1) is last
            postmean_w = float(parts[-2])
            p_w_gt1 = float(parts[-1])
            sites.append({
                'site': site_num,
                'aa': aa,
                'postmean_w': postmean_w,
                'p_w_gt1': p_w_gt1
            })
        except (ValueError, IndexError):
            continue
    return sites


# ── 1. Parse BEB sites from M2a and M8 ────────────────────────────────
print("\n── 1. Parsing BEB sites from codeml output ──")

beb_m2a = parse_beb_sites(PAML_DIR / 'M2a_out.txt')
beb_m8 = parse_beb_sites(PAML_DIR / 'M8_out.txt')

print(f"  M2a BEB sites: {len(beb_m2a)} total, "
      f"{sum(1 for s in beb_m2a if s['significance'])} significant (P>0.95)")
print(f"  M8  BEB sites: {len(beb_m8)} total, "
      f"{sum(1 for s in beb_m8 if s['significance'])} significant (P>0.95)")

# Use M8 as primary (more powerful test), fall back to M2a
beb_sites = beb_m8 if beb_m8 else beb_m2a
beb_model = 'M8' if beb_m8 else 'M2a'
print(f"  Using {beb_model} BEB results for visualization")

if beb_sites:
    print("\n  BEB Positively Selected Sites:")
    print(f"  {'Site':>6s}  {'AA':>3s}  {'Pr(w>1)':>8s}  {'omega':>8s}  {'SE':>6s}  Sig")
    print(f"  {'─'*6}  {'─'*3}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*3}")
    for s in beb_sites:
        print(f"  {s['site']:>6d}  {s['aa']:>3s}  {s['prob']:>8.3f}  "
              f"{s['omega']:>8.3f}  {s['se']:>6.3f}  {s['significance']}")

# ── 2. Parse per-site omega from rst ───────────────────────────────────
print("\n── 2. Parsing per-site omega from rst file ──")

rst_sites = parse_rst_per_site(PAML_DIR / 'rst')
print(f"  Per-site omega values: {len(rst_sites)} codons")

if rst_sites:
    omegas = [s['postmean_w'] for s in rst_sites]
    print(f"  Mean per-site omega: {np.mean(omegas):.3f}")
    print(f"  Median per-site omega: {np.median(omegas):.3f}")
    print(f"  Sites with omega > 1: {sum(1 for w in omegas if w > 1)} "
          f"({sum(1 for w in omegas if w > 1)/len(omegas)*100:.1f}%)")
    print(f"  Sites with omega > 2: {sum(1 for w in omegas if w > 2)} "
          f"({sum(1 for w in omegas if w > 2)/len(omegas)*100:.1f}%)")

# ── 3. Load PAML alignment and count mutations per site ────────────────
print("\n── 3. Counting mutations per codon site ──")

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

paml_aln_path = PAML_DIR / f'{prefix}_codon.phy'
mutation_data = []

if paml_aln_path.exists():
    # Parse PHYLIP manually (sequential format)
    with open(paml_aln_path) as f:
        header = f.readline().split()
        n_seqs, seq_len = int(header[0]), int(header[1])
        records = {}
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                name = parts[0].strip()
                seq = parts[1].replace(' ', '')
                records[name] = seq

    names = list(records.keys())
    n_codons = seq_len // 3
    ref_name = names[0]
    ref_seq = records[ref_name]

    # If rst has fewer codons, PAML used cleandata=1 to remove gappy columns
    # Use the rst codon count as the authoritative number of sites
    if rst_sites and len(rst_sites) < n_codons:
        n_codons = len(rst_sites)
        print(f"  NOTE: Using {n_codons} codons (PAML cleandata removed gappy columns)")

    for codon_pos in range(n_codons):
        site_num = codon_pos + 1
        nt_start = codon_pos * 3

        ref_codon = ref_seq[nt_start:nt_start+3].upper()
        if '-' in ref_codon or 'N' in ref_codon:
            mutation_data.append({
                'site': site_num, 'ref_aa': '-', 'n_mutations': 0,
                'n_unique_aa': 0, 'aa_variants': '', 'mutation_freq': 0.0,
                'n_synonymous': 0, 'n_nonsynonymous': 0
            })
            continue

        ref_aa = CODON_TABLE.get(ref_codon, 'X')
        aa_counts = Counter()
        n_valid = 0
        n_syn = 0
        n_nonsyn = 0

        for name in names:
            codon = records[name][nt_start:nt_start+3].upper()
            if '-' in codon or 'N' in codon:
                continue
            aa = CODON_TABLE.get(codon, 'X')
            if aa == '*' or aa == 'X':
                continue
            aa_counts[aa] += 1
            n_valid += 1
            if codon != ref_codon:
                if aa == ref_aa:
                    n_syn += 1
                else:
                    n_nonsyn += 1

        n_mutations = n_syn + n_nonsyn
        unique_aas = sorted(aa_counts.keys())
        variants_str = ', '.join(f"{aa}({c})" for aa, c in aa_counts.most_common())

        mutation_data.append({
            'site': site_num,
            'ref_aa': ref_aa,
            'n_mutations': n_mutations,
            'n_unique_aa': len(unique_aas),
            'aa_variants': variants_str,
            'mutation_freq': n_mutations / n_valid if n_valid > 0 else 0.0,
            'n_synonymous': n_syn,
            'n_nonsynonymous': n_nonsyn
        })

    print(f"  Analyzed {n_codons} codon sites across {len(names)} sequences")
    total_mut = sum(m['n_mutations'] for m in mutation_data)
    total_nonsyn = sum(m['n_nonsynonymous'] for m in mutation_data)
    print(f"  Total mutations: {total_mut} ({total_nonsyn} nonsynonymous)")

mut_df = pd.DataFrame(mutation_data)

# ── 4. Merge per-site omega with mutation data ─────────────────────────
print("\n── 4. Building combined per-site table ──")

if rst_sites and len(mutation_data) > 0:
    rst_df = pd.DataFrame(rst_sites)
    combined = mut_df.merge(rst_df, on='site', how='left', suffixes=('', '_rst'))
    # Drop duplicate aa column
    if 'aa' in combined.columns:
        combined.drop('aa', axis=1, inplace=True)
else:
    combined = mut_df.copy()
    combined['postmean_w'] = np.nan
    combined['p_w_gt1'] = np.nan

# Mark BEB sites
beb_site_nums = {s['site'] for s in beb_sites}
beb_sig_95 = {s['site'] for s in beb_sites if s['prob'] > 0.95}
beb_sig_99 = {s['site'] for s in beb_sites if s['prob'] > 0.99}
combined['is_beb'] = combined['site'].isin(beb_site_nums)
combined['beb_95'] = combined['site'].isin(beb_sig_95)
combined['beb_99'] = combined['site'].isin(beb_sig_99)

# Add BEB omega and probability
beb_dict = {s['site']: s for s in beb_sites}
combined['beb_prob'] = combined['site'].map(lambda s: beb_dict.get(s, {}).get('prob', np.nan))
combined['beb_omega'] = combined['site'].map(lambda s: beb_dict.get(s, {}).get('omega', np.nan))

# ── 5. Exon boundary mapping ──────────────────────────────────────────
exon_boundaries = cfg.get('exon_boundaries_aa', {})
analysis_range = cfg.get('analysis_exon_range', [1, 15])

# ── 6. FIGURES ─────────────────────────────────────────────────────────
print("\n── 5. Generating per-site dN/dS figures ──")

sns.set_context('notebook')
sns.set_style('whitegrid')

if 'postmean_w' in combined.columns and combined['postmean_w'].notna().any():

    # ── Figure 1: Per-site omega along the protein ──────────────────
    fig, axes = plt.subplots(4, 1, figsize=(16, 16), gridspec_kw={'height_ratios': [3, 1.5, 1.5, 1]})
    fig.suptitle(f'{GENE} Per-Site dN/dS Analysis — {SPECIES}\n'
                 f'({beb_model} Bayes Empirical Bayes)', fontsize=14, y=0.98)

    # Panel A: Per-site omega with BEB highlights
    ax = axes[0]
    sites = combined['site'].values
    omegas = combined['postmean_w'].values

    # Background bars for all sites
    ax.bar(sites, omegas, width=1.0, color='steelblue', alpha=0.4, label='Per-site omega')

    # Highlight BEB sites
    beb_mask = combined['is_beb'].values
    beb95_mask = combined['beb_95'].values
    beb99_mask = combined['beb_99'].values

    if beb_mask.any():
        ax.bar(sites[beb_mask], omegas[beb_mask], width=1.0,
               color='darkorange', alpha=0.7, label='BEB candidate')
    if beb95_mask.any():
        ax.bar(sites[beb95_mask], omegas[beb95_mask], width=1.0,
               color='crimson', alpha=0.8, label='BEB P>0.95 (*)')
    if beb99_mask.any():
        ax.bar(sites[beb99_mask], omegas[beb99_mask], width=1.0,
               color='darkred', alpha=0.9, label='BEB P>0.99 (**)')

    # Reference lines
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Neutral (omega=1)')
    mean_omega = np.nanmean(omegas)
    ax.axhline(mean_omega, color='gold', linestyle='-', linewidth=2,
               alpha=0.8, label=f'Mean omega={mean_omega:.2f}')
    ax.axhline(0.15, color='blue', linestyle=':', linewidth=1, alpha=0.5,
               label='Bird functional avg (0.15)')

    # Annotate BEB ** sites
    for s in beb_sites:
        if s['significance'] == '**':
            ax.annotate(f"{s['site']}{s['aa']}", xy=(s['site'], omegas[s['site']-1]),
                        xytext=(0, 8), textcoords='offset points',
                        fontsize=7, fontweight='bold', color='darkred',
                        ha='center', va='bottom', rotation=45)

    ax.set_ylabel('Posterior mean omega (dN/dS)')
    ax.set_title('A. Per-site dN/dS ratio along the protein')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.set_xlim(0, len(sites) + 1)

    # Panel B: Mutation frequency per site
    ax = axes[1]
    nonsyn = combined['n_nonsynonymous'].values
    syn = combined['n_synonymous'].values

    ax.bar(sites, nonsyn, width=1.0, color='crimson', alpha=0.6, label='Nonsynonymous')
    ax.bar(sites, syn, width=1.0, bottom=nonsyn, color='steelblue', alpha=0.4, label='Synonymous')

    # Mark BEB sites
    for s in beb_sites:
        if s['significance']:
            ax.axvline(s['site'], color='red', alpha=0.3, linewidth=0.5)

    ax.set_ylabel('Mutation count')
    ax.set_title('B. Mutations per codon site (synonymous + nonsynonymous)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, len(sites) + 1)

    # Panel C: Amino acid diversity (unique AAs per site)
    ax = axes[2]
    n_unique = combined['n_unique_aa'].values
    colors = ['steelblue' if not b else ('crimson' if s else 'darkorange')
              for b, s in zip(beb_mask, beb95_mask)]
    ax.bar(sites, n_unique, width=1.0, color=colors, alpha=0.6)
    ax.axhline(np.mean(n_unique), color='gold', linestyle='-', linewidth=1.5,
               label=f'Mean={np.mean(n_unique):.1f}')
    ax.set_ylabel('Unique amino acids')
    ax.set_title('C. Amino acid diversity per site')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, len(sites) + 1)

    # Panel D: Exon map
    ax = axes[3]
    ax.set_xlim(0, len(sites) + 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title('D. Protein position (exon boundaries)')
    ax.set_xlabel('Codon position')

    exon_colors = plt.cm.Pastel1(np.linspace(0, 1, len(exon_boundaries)))
    for i, (exon_num, (start_aa, end_aa)) in enumerate(sorted(exon_boundaries.items(), key=lambda x: int(x[0]))):
        # Map amino acid positions to codon positions in our alignment
        # The alignment covers a subset of exons
        exon_int = int(exon_num)
        if analysis_range and (exon_int < analysis_range[0] or exon_int > analysis_range[1]):
            continue
        # Approximate mapping: offset by the start of analysis range
        first_exon_start = cfg['exon_boundaries_aa'].get(str(analysis_range[0]), [1, 1])[0]
        rel_start = start_aa - first_exon_start + 1
        rel_end = end_aa - first_exon_start + 1
        if rel_start < 1:
            rel_start = 1
        if rel_end > len(sites):
            rel_end = len(sites)
        if rel_start <= len(sites):
            ax.axvspan(rel_start, rel_end, alpha=0.3, color=exon_colors[i % len(exon_colors)])
            mid = (rel_start + rel_end) / 2
            ax.text(mid, 0.5, f'Ex{exon_num}', ha='center', va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'per_site_dnds.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIG_DIR / 'per_site_dnds.png'}")

    # ── Figure 2: BEB site detail — mutation heatmap ────────────────
    if beb_sites and paml_aln_path.exists():
        # Build amino acid matrix at BEB sites
        beb_site_list = sorted(beb_site_nums)
        aa_matrix = []
        seq_names = list(records.keys())

        for name in seq_names:
            row = []
            for s in beb_site_list:
                codon_start = (s - 1) * 3
                codon = records[name][codon_start:codon_start+3].upper()
                if '-' in codon or 'N' in codon:
                    row.append('-')
                else:
                    row.append(CODON_TABLE.get(codon, 'X'))
            aa_matrix.append(row)

        aa_df = pd.DataFrame(aa_matrix, index=seq_names,
                             columns=[f"{s} ({beb_dict[s]['aa']})" for s in beb_site_list])

        # Encode for heatmap: reference AA = 0, different = 1, gap = -1
        ref_row = aa_matrix[0]
        encoded = np.zeros((len(seq_names), len(beb_site_list)))
        for i, row in enumerate(aa_matrix):
            for j, aa in enumerate(row):
                if aa == '-':
                    encoded[i, j] = -0.5
                elif aa != ref_row[j]:
                    encoded[i, j] = 1.0

        fig, ax = plt.subplots(figsize=(max(8, len(beb_site_list) * 0.6),
                                         max(6, len(seq_names) * 0.25)))
        cmap = plt.cm.RdYlBu_r
        im = ax.imshow(encoded, aspect='auto', cmap=cmap, vmin=-0.5, vmax=1.0)

        ax.set_xticks(range(len(beb_site_list)))
        xlabels = []
        for s in beb_site_list:
            sig = '**' if s in beb_sig_99 else '*' if s in beb_sig_95 else ''
            omega_str = f"w={beb_dict[s]['omega']:.1f}"
            xlabels.append(f"{s}{beb_dict[s]['aa']}{sig}\n{omega_str}")
        ax.set_xticklabels(xlabels, fontsize=7, rotation=0)
        ax.set_yticks(range(len(seq_names)))
        ax.set_yticklabels([n[:20] for n in seq_names], fontsize=5)

        # Add actual AA letters
        for i, row in enumerate(aa_matrix):
            for j, aa in enumerate(row):
                if aa != '-':
                    color = 'white' if encoded[i, j] > 0.5 else 'black'
                    ax.text(j, i, aa, ha='center', va='center', fontsize=5, color=color)

        ax.set_xlabel('BEB positively selected sites')
        ax.set_ylabel('Gene copies')
        ax.set_title(f'{GENE} Amino Acid Variation at BEB Sites — {SPECIES}\n'
                     f'(Red = derived, Blue = reference/ancestral)')
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'beb_site_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {FIG_DIR / 'beb_site_heatmap.png'}")

    # ── Figure 3: BEB site clustering analysis ──────────────────────
    if len(beb_site_list) >= 3:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{GENE} BEB Site Clustering Analysis — {SPECIES}', fontsize=13, y=1.02)

        # Panel A: Sliding window average omega
        ax = axes[0]
        window_sizes = [5, 10, 15]
        for ws in window_sizes:
            if len(omegas) >= ws:
                smoothed = np.convolve(omegas, np.ones(ws)/ws, mode='valid')
                ax.plot(range(ws//2 + 1, len(smoothed) + ws//2 + 1), smoothed,
                        label=f'{ws}-codon window', alpha=0.8)
        ax.axhline(1.0, color='red', linestyle='--', alpha=0.5)
        ax.axhline(mean_omega, color='gold', linestyle='-', alpha=0.5,
                   label=f'Global mean={mean_omega:.2f}')
        for s in beb_site_list:
            ax.axvline(s, color='red', alpha=0.2, linewidth=0.5)
        ax.set_xlabel('Codon position')
        ax.set_ylabel('Sliding window omega')
        ax.set_title('A. Sliding window dN/dS')
        ax.legend(fontsize=8)

        # Panel B: Inter-BEB distances
        ax = axes[1]
        if len(beb_site_list) >= 2:
            distances = [beb_site_list[i+1] - beb_site_list[i]
                         for i in range(len(beb_site_list)-1)]
            ax.bar(range(len(distances)), distances, color='steelblue')
            ax.axhline(np.mean(distances), color='gold', linestyle='--',
                       label=f'Mean gap={np.mean(distances):.1f} codons')
            # Expected under random placement
            expected_gap = len(sites) / (len(beb_site_list) + 1)
            ax.axhline(expected_gap, color='red', linestyle=':',
                       label=f'Expected random={expected_gap:.1f}')
            ax.set_xlabel('BEB site pair index')
            ax.set_ylabel('Distance (codons)')
            ax.set_title('B. Distance between adjacent BEB sites')
            ax.legend(fontsize=8)

        # Panel C: BEB site probability vs position
        ax = axes[2]
        beb_positions = [s['site'] for s in beb_sites]
        beb_probs = [s['prob'] for s in beb_sites]
        beb_omegas = [s['omega'] for s in beb_sites]

        scatter = ax.scatter(beb_positions, beb_probs, c=beb_omegas,
                            cmap='YlOrRd', s=80, edgecolors='black', linewidth=0.5)
        ax.axhline(0.95, color='red', linestyle='--', alpha=0.5, label='P=0.95')
        ax.axhline(0.99, color='darkred', linestyle='--', alpha=0.5, label='P=0.99')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Posterior omega')
        ax.set_xlabel('Codon position')
        ax.set_ylabel('Pr(omega > 1)')
        ax.set_title('C. BEB probability vs protein position')
        ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(FIG_DIR / 'beb_clustering.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {FIG_DIR / 'beb_clustering.png'}")

    # ── Figure 4: Summary — dN/dS rate averages ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Average omega by region
    ax = axes[0]
    beb_avg = np.nanmean([omegas[s-1] for s in beb_site_nums if s-1 < len(omegas)])
    nonbeb_mask = ~combined['is_beb'].values
    nonbeb_avg = np.nanmean(omegas[nonbeb_mask]) if nonbeb_mask.any() else 0

    cats = [f'All sites\n(n={len(omegas)})',
            f'BEB sites\n(n={len(beb_site_nums)})',
            f'Non-BEB\n(n={sum(nonbeb_mask)})',
            'Bird avg\n(0.15)',
            'Neutral\n(1.0)']
    vals = [mean_omega, beb_avg, nonbeb_avg, 0.15, 1.0]
    colors = ['steelblue', 'crimson', 'gray', 'lightblue', 'lightgray']
    bars = ax.bar(cats, vals, color=colors)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.3)
    ax.set_ylabel('Mean omega (dN/dS)')
    ax.set_title(f'{GENE} Average dN/dS by Site Category')

    # Panel B: Omega distribution histogram
    ax = axes[1]
    ax.hist(omegas[~np.isnan(omegas)], bins=30, color='steelblue', alpha=0.6,
            edgecolor='white', label='All sites')
    beb_omegas_vals = [omegas[s-1] for s in beb_site_nums if s-1 < len(omegas)]
    if beb_omegas_vals:
        ax.hist(beb_omegas_vals, bins=15, color='crimson', alpha=0.7,
                edgecolor='white', label='BEB sites')
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Neutral')
    ax.axvline(mean_omega, color='gold', linestyle='-', linewidth=2,
               label=f'Mean={mean_omega:.2f}')
    ax.set_xlabel('Posterior mean omega')
    ax.set_ylabel('Number of sites')
    ax.set_title('Per-site omega distribution')
    ax.legend(fontsize=9)

    plt.suptitle(f'{GENE} dN/dS Rate Summary — {SPECIES}', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'dnds_rate_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIG_DIR / 'dnds_rate_summary.png'}")

else:
    print("  WARNING: No per-site omega data available from rst file")

# ── 7. Save tables ─────────────────────────────────────────────────────
print("\n── 6. Saving results tables ──")

# BEB sites detail table
if beb_sites:
    beb_df = pd.DataFrame(beb_sites)
    beb_df.to_csv(TABLE_DIR / 'beb_selected_sites.csv', index=False)
    print(f"  Saved: {TABLE_DIR / 'beb_selected_sites.csv'}")

# Per-site combined table
combined.to_csv(TABLE_DIR / 'per_site_dnds.csv', index=False)
print(f"  Saved: {TABLE_DIR / 'per_site_dnds.csv'}")

# ── 8. Clustering statistics ──────────────────────────────────────────
print("\n── 7. BEB site clustering statistics ──")

# beb_site_list is only defined when beb_sites is non-empty and alignment was parsed
beb_site_list = sorted({s['site'] for s in beb_sites}) if beb_sites else []
clustering_ratio = None

if len(beb_site_list) >= 2:
    distances = [beb_site_list[i+1] - beb_site_list[i]
                 for i in range(len(beb_site_list)-1)]
    n_total_sites = len(rst_sites) if rst_sites else len(beb_site_list) + 1
    expected_gap = n_total_sites / (len(beb_site_list) + 1)
    clustering_ratio = np.mean(distances) / expected_gap if expected_gap > 0 else None

    print(f"  BEB sites span: {beb_site_list[0]} to {beb_site_list[-1]} "
          f"(range: {beb_site_list[-1] - beb_site_list[0]} codons)")
    print(f"  Mean inter-BEB distance: {np.mean(distances):.1f} codons")
    print(f"  Min inter-BEB distance: {min(distances)} codons")
    print(f"  Max inter-BEB distance: {max(distances)} codons")
    if expected_gap > 0:
        print(f"  Expected distance (random): {expected_gap:.1f} codons")
        print(f"  Clustering ratio: {clustering_ratio:.2f} "
              f"({'CLUSTERED' if clustering_ratio < 0.5 else 'DISPERSED' if clustering_ratio > 1.5 else 'MODERATE'})")
else:
    print("  No BEB sites available for clustering analysis")

# ── Summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"  PER-SITE dN/dS SUMMARY — {GENE} in {SPECIES}")
print("=" * 70)
if rst_sites:
    rst_omegas = [s['postmean_w'] for s in rst_sites]
    print(f"  Total codon sites:         {len(rst_sites)}")
    print(f"  Mean per-site omega:       {np.mean(rst_omegas):.3f}")
    print(f"  Sites with omega > 1:      {sum(1 for w in rst_omegas if w > 1)} "
          f"({sum(1 for w in rst_omegas if w > 1)/len(rst_omegas)*100:.1f}%)")
print(f"  BEB sites ({beb_model}):        {len(beb_sites)}")
print(f"    P > 0.95 (*):            {len(beb_sig_95)}")
print(f"    P > 0.99 (**):           {len(beb_sig_99)}")
if beb_sites:
    print(f"  BEB mean omega:            {np.mean([s['omega'] for s in beb_sites]):.3f}")
if len(beb_site_list) >= 2 and clustering_ratio is not None:
    print(f"  BEB clustering:            {clustering_ratio:.2f} "
          f"({'Clustered' if clustering_ratio < 0.5 else 'Dispersed' if clustering_ratio > 1.5 else 'Moderate'})")
print(f"\n  Figures: per_site_dnds.png, beb_site_heatmap.png,")
print(f"           beb_clustering.png, dnds_rate_summary.png")
print(f"  Tables:  beb_selected_sites.csv, per_site_dnds.csv")
print("=" * 70)
