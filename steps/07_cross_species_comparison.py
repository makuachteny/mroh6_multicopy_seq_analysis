#!/usr/bin/env python3
"""
07 — Cross-Species Comparison Report
======================================

QUESTION:
  How do MROH gene copy dynamics compare across songbird species?
  Which species show the strongest evidence for RNA-mediated duplication
  and positive selection? Are the patterns consistent across lineages?

REASONING:
  Each species was analyzed independently through the same pipeline.
  Comparing the results reveals:

    1. Copy number variation — lineage-specific expansion rates
    2. Mutation rate elevation — strength of the RNA-mediated signal
    3. Selection regime — which species have positively selected sites
    4. Copy integrity — pseudogenization vs functional maintenance
    5. Repeat structure — conservation of HEAT repeat architecture

  A cross-species summary table and comparative figures enable direct
  hypothesis testing: if RNA-mediated duplication is a shared mechanism
  across songbirds, all species should show elevated mutation rates
  (>3x baseline) and dispersed genomic distribution.

Usage:
  python steps/07_cross_species_comparison.py
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'scripts'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy import stats as sp_stats

PROJECT = Path(__file__).resolve().parent.parent
CONFIGS = PROJECT / 'configs'
RESULTS = PROJECT / 'results'
DATA_PROC = PROJECT / 'data' / 'processed'

FIG_DIR = RESULTS / 'figures'
TABLE_DIR = RESULTS / 'tables'
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  STEP 07: CROSS-SPECIES COMPARISON REPORT")
print("=" * 70)

# ── 1. Load all species data ──────────────────────────────────────────
print("\n── 1. Loading results from all species ──")

species_slugs = sorted([p.stem for p in CONFIGS.glob('*.json')])
print(f"  Species found: {len(species_slugs)}")

all_data = []

from species_config import load_config

for slug in species_slugs:
    cfg = load_config(slug)

    row = {
        'species_slug': slug,
        'species_name': cfg['species_name'],
        'common_name': cfg['common_name'],
        'genome_assembly': cfg['genome_assembly'],
    }

    sp_results = RESULTS / slug / 'tables'
    sp_data = DATA_PROC / slug

    # Gene copy count
    loci_path = sp_data / f"{cfg['output_prefix']}_loci_table.csv"
    if loci_path.exists():
        loci_df = pd.read_csv(loci_path)
        row['n_copies'] = len(loci_df)
        row['n_intact'] = len(loci_df[loci_df['coverage_frac'] >= 0.80])
        row['n_partial'] = len(loci_df[(loci_df['coverage_frac'] >= 0.50) & (loci_df['coverage_frac'] < 0.80)])
        row['pct_intact'] = row['n_intact'] / row['n_copies'] * 100
        row['mean_coverage'] = loci_df['coverage_frac'].clip(upper=1.0).mean()
    else:
        row['n_copies'] = 0

    # Mutation rate summary
    mut_path = sp_results / 'mutation_rate_summary.csv'
    if mut_path.exists():
        mut_df = pd.read_csv(mut_path)
        mut_dict = dict(zip(mut_df.iloc[:, 0], mut_df.iloc[:, 1]))
        # Map various key names to standard names
        key_map = {
            'JC_mean': ['JC-corrected mean', 'JC_mean'],
            'JC_median': ['JC-corrected median', 'JC-corrected median divergence'],
            'TsTv_median': ['Ts/Tv median'],
            'fold_elevation': ['Fold elevation', 'fold_elevation'],
            'divergence_pvalue': ['P-value', 'T-test p-value (vs baseline)', 'P-value (vs baseline)'],
        }
        for target, candidates in key_map.items():
            for key in candidates:
                if key in mut_dict:
                    try:
                        row[target] = float(str(mut_dict[key]).replace('x', ''))
                    except (ValueError, TypeError):
                        pass
                    break

    # PAML results
    paml_path = sp_results / 'paml_results.csv'
    if paml_path.exists():
        paml_df = pd.read_csv(paml_path)
        for _, prow in paml_df.iterrows():
            model = prow.get('Model', '')
            if model == 'M0' and pd.notna(prow.get('omega')):
                row['M0_omega'] = prow['omega']
                row['M0_kappa'] = prow.get('kappa', np.nan)
            if model and pd.notna(prow.get('lnL')):
                row[f'{model}_lnL'] = prow['lnL']

        # Compute LRT
        if 'M1a_lnL' in row and 'M2a_lnL' in row:
            delta = 2 * (row['M2a_lnL'] - row['M1a_lnL'])
            row['LRT_M1a_vs_M2a'] = delta
            row['LRT_M1a_vs_M2a_p'] = sp_stats.chi2.sf(delta, 2) if delta > 0 else 1.0
        if 'M7_lnL' in row and 'M8_lnL' in row:
            delta = 2 * (row['M8_lnL'] - row['M7_lnL'])
            row['LRT_M7_vs_M8'] = delta
            row['LRT_M7_vs_M8_p'] = sp_stats.chi2.sf(delta, 2) if delta > 0 else 1.0

    # Pairwise dN/dS
    dnds_path = sp_results / 'pairwise_dnds.csv'
    if dnds_path.exists():
        dnds_df = pd.read_csv(dnds_path)
        valid = dnds_df.dropna(subset=['omega'])
        valid = valid[(valid['omega'] < 99) & (valid['dS'] > 0)]
        if len(valid) > 0:
            row['median_pairwise_omega'] = valid['omega'].median()
            row['mean_pairwise_dN'] = valid['dN'].mean()
            row['mean_pairwise_dS'] = valid['dS'].mean()
            row['n_pairs'] = len(valid)

    # Selection tests
    sel_path = sp_results / 'selection_tests.csv'
    if sel_path.exists():
        sel_df = pd.read_csv(sel_path)
        sel_dict = dict(zip(sel_df.iloc[:, 0], sel_df.iloc[:, 1]))
        try:
            row['BEB_sites_95'] = int(str(sel_dict.get('BEB sites (P>0.95)', 0)))
        except (ValueError, TypeError):
            row['BEB_sites_95'] = 0
        row['TsTv_4fold'] = sel_dict.get('Ts/Tv at 4-fold degenerate sites', '')

    # BEB selected sites
    beb_path = sp_results / 'beb_selected_sites.csv'
    if beb_path.exists():
        beb_df = pd.read_csv(beb_path)
        row['BEB_total'] = len(beb_df)
        row['BEB_sig_99'] = len(beb_df[beb_df['prob'] > 0.99]) if 'prob' in beb_df.columns else 0
        row['BEB_mean_omega'] = beb_df['omega'].mean() if 'omega' in beb_df.columns else np.nan
    else:
        row['BEB_total'] = 0
        row['BEB_sig_99'] = 0

    # Repeat summary
    rep_path = sp_results / 'repeat_summary.csv'
    if rep_path.exists():
        rep_df = pd.read_csv(rep_path)
        rep_dict = dict(zip(rep_df.iloc[:, 0], rep_df.iloc[:, 1]))
        try:
            row['mean_mroh_pct'] = float(str(rep_dict.get('Mean MROH fraction of span', '0')).replace('%', ''))
        except (ValueError, TypeError):
            pass
        try:
            row['mean_heat_repeats'] = float(rep_dict.get('Mean HEAT repeats per copy', 0))
        except (ValueError, TypeError):
            pass

    all_data.append(row)
    status = "FULL" if row.get('M0_omega') else "PARTIAL (no PAML)"
    print(f"  {slug:30s} — {row.get('n_copies', 0)} copies, {status}")

df = pd.DataFrame(all_data)

# ── 2. Summary table ──────────────────────────────────────────────────
print("\n── 2. Building cross-species comparison table ──")

summary_cols = {
    'common_name': 'Species',
    'n_copies': 'Gene Copies',
    'n_intact': 'Intact (>=80%)',
    'pct_intact': '% Intact',
    'JC_mean': 'Mean Divergence (JC)',
    'fold_elevation': 'Fold Elevation',
    'M0_omega': 'M0 omega (dN/dS)',
    'M0_kappa': 'M0 kappa (Ts/Tv)',
    'median_pairwise_omega': 'Median Pairwise omega',
    'BEB_total': 'BEB Sites (all)',
    'BEB_sites_95': 'BEB Sites (P>0.95)',
    'BEB_sig_99': 'BEB Sites (P>0.99)',
    'BEB_mean_omega': 'BEB Mean omega',
    'mean_mroh_pct': 'MROH % of Span',
    'mean_heat_repeats': 'HEAT Repeats/Copy',
}

available = {k: v for k, v in summary_cols.items() if k in df.columns}
summary = df[list(available.keys())].rename(columns=available)

# Round numeric columns
for col in summary.columns:
    if summary[col].dtype in ['float64', 'float32']:
        if 'pvalue' in col.lower() or 'p-value' in col.lower():
            summary[col] = summary[col].map(lambda x: f'{x:.2e}' if pd.notna(x) else '')
        elif '%' in col:
            summary[col] = summary[col].round(1)
        else:
            summary[col] = summary[col].round(3)

summary.to_csv(TABLE_DIR / 'cross_species_comparison.csv', index=False)
print(f"  Saved: {TABLE_DIR / 'cross_species_comparison.csv'}")

# Print table
print("\n  CROSS-SPECIES COMPARISON TABLE:")
print("  " + "─" * 80)
for col in summary.columns:
    vals = summary[col].astype(str).tolist()
    print(f"  {col:30s}  {'  |  '.join(vals)}")
print("  " + "─" * 80)

# ── 3. LRT significance table ────────────────────────────────────────
print("\n── 3. Likelihood ratio tests ──")

lrt_rows = []
for _, row in df.iterrows():
    lrt_row = {'Species': row['common_name']}
    for test, cols in [('M1a vs M2a', ('LRT_M1a_vs_M2a', 'LRT_M1a_vs_M2a_p')),
                       ('M7 vs M8', ('LRT_M7_vs_M8', 'LRT_M7_vs_M8_p'))]:
        delta = row.get(cols[0], np.nan)
        p = row.get(cols[1], np.nan)
        if pd.notna(delta):
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            lrt_row[f'{test} (2dlnL)'] = f'{delta:.2f}'
            lrt_row[f'{test} (p)'] = f'{p:.2e}'
            lrt_row[f'{test} (sig)'] = sig
        else:
            lrt_row[f'{test} (2dlnL)'] = 'N/A'
            lrt_row[f'{test} (p)'] = 'N/A'
            lrt_row[f'{test} (sig)'] = 'N/A'
    lrt_rows.append(lrt_row)

lrt_df = pd.DataFrame(lrt_rows)
lrt_df.to_csv(TABLE_DIR / 'cross_species_lrt.csv', index=False)
print(f"  Saved: {TABLE_DIR / 'cross_species_lrt.csv'}")

for _, row in lrt_df.iterrows():
    print(f"  {row['Species']:25s}  M1a/M2a: {row['M1a vs M2a (2dlnL)']:>8s} "
          f"(p={row['M1a vs M2a (p)']:>10s}) {row['M1a vs M2a (sig)']:>3s}  |  "
          f"M7/M8: {row['M7 vs M8 (2dlnL)']:>8s} "
          f"(p={row['M7 vs M8 (p)']:>10s}) {row['M7 vs M8 (sig)']:>3s}")

# ── 4. Figures ─────────────────────────────────────────────────────────
print("\n── 4. Generating cross-species comparison figures ──")

sns.set_context('notebook')
sns.set_style('whitegrid')

# Use species with data
species_order = df.sort_values('n_copies', ascending=False)['common_name'].tolist()
palette = sns.color_palette('Set2', n_colors=len(species_order))
color_map = dict(zip(species_order, palette))

# ── Figure 1: Multi-panel overview ─────────────────────────────────
fig = plt.figure(figsize=(20, 16))
gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.3)
fig.suptitle('MROH6 Cross-Species Comparison — 5 Songbird Species',
             fontsize=16, y=0.98, fontweight='bold')

# A: Gene copy number
ax = fig.add_subplot(gs[0, 0])
bars = ax.bar(df['common_name'], df['n_copies'], color=[color_map[n] for n in df['common_name']])
for bar, val in zip(bars, df['n_copies']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(int(val)), ha='center', fontsize=9, fontweight='bold')
ax.set_ylabel('Gene copies')
ax.set_title('A. MROH6 Copy Number')
ax.tick_params(axis='x', rotation=30)

# B: Copy integrity (stacked bar)
ax = fig.add_subplot(gs[0, 1])
intact_pct = df['pct_intact'].fillna(0).values
partial_pct = (df['n_partial'] / df['n_copies'] * 100).fillna(0).values
ax.bar(df['common_name'], intact_pct, color='seagreen', label='Intact (>=80%)')
ax.bar(df['common_name'], partial_pct, bottom=intact_pct, color='darkorange', label='Partial (50-80%)')
ax.set_ylabel('% of copies')
ax.set_title('B. Copy Integrity')
ax.legend(fontsize=8)
ax.tick_params(axis='x', rotation=30)

# C: Mutation rate elevation
ax = fig.add_subplot(gs[0, 2])
fold_vals = df['fold_elevation'].fillna(0).values
bars = ax.bar(df['common_name'], fold_vals, color=[color_map[n] for n in df['common_name']])
ax.axhline(3.0, color='red', linestyle='--', alpha=0.5, label='3x threshold (RNA signal)')
ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5, label='Genomic baseline')
for bar, val in zip(bars, fold_vals):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}x', ha='center', fontsize=9, fontweight='bold')
ax.set_ylabel('Fold elevation vs baseline')
ax.set_title('C. Mutation Rate Elevation')
ax.legend(fontsize=8)
ax.tick_params(axis='x', rotation=30)

# D: M0 global omega
ax = fig.add_subplot(gs[1, 0])
has_m0 = df['M0_omega'].notna()
if has_m0.any():
    m0_df = df[has_m0]
    bars = ax.bar(m0_df['common_name'], m0_df['M0_omega'],
                  color=[color_map[n] for n in m0_df['common_name']])
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Neutral (omega=1)')
    ax.axhline(0.15, color='blue', linestyle=':', alpha=0.5, label='Bird avg (0.15)')
    for bar, val in zip(bars, m0_df['M0_omega']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')
    ax.legend(fontsize=8)
ax.set_ylabel('Global omega (dN/dS)')
ax.set_title('D. PAML M0 Global omega')
ax.tick_params(axis='x', rotation=30)

# E: BEB sites
ax = fig.add_subplot(gs[1, 1])
beb95 = df['BEB_sites_95'].fillna(0).values
beb99 = df['BEB_sig_99'].fillna(0).values
beb_other = df['BEB_total'].fillna(0).values - beb95
x = np.arange(len(df))
ax.bar(x, beb99, color='darkred', label='P>0.99 (**)')
ax.bar(x, beb95 - beb99, bottom=beb99, color='crimson', label='P>0.95 (*)')
ax.bar(x, beb_other, bottom=beb95, color='darkorange', alpha=0.6, label='Candidate (P<0.95)')
ax.set_xticks(x)
ax.set_xticklabels(df['common_name'], rotation=30, ha='right')
ax.set_ylabel('Number of BEB sites')
ax.set_title('E. Positively Selected Sites (BEB)')
ax.legend(fontsize=8)

# F: LRT significance
ax = fig.add_subplot(gs[1, 2])
has_lrt = df['LRT_M1a_vs_M2a'].notna()
if has_lrt.any():
    lrt_data = df[has_lrt]
    x = np.arange(len(lrt_data))
    w = 0.35
    ax.bar(x - w/2, lrt_data['LRT_M1a_vs_M2a'], w, color='steelblue', label='M1a vs M2a')
    ax.bar(x + w/2, lrt_data['LRT_M7_vs_M8'], w, color='darkorange', label='M7 vs M8')
    ax.axhline(5.99, color='red', linestyle='--', alpha=0.5, label='chi2 critical (p=0.05)')
    ax.axhline(9.21, color='darkred', linestyle='--', alpha=0.3, label='chi2 critical (p=0.01)')
    ax.set_xticks(x)
    ax.set_xticklabels(lrt_data['common_name'], rotation=30, ha='right')
    ax.legend(fontsize=7)
ax.set_ylabel('2*delta_lnL')
ax.set_title('F. Likelihood Ratio Tests')

# G: Median pairwise omega comparison
ax = fig.add_subplot(gs[2, 0])
has_pw = df['median_pairwise_omega'].notna()
if has_pw.any():
    pw_df = df[has_pw].sort_values('median_pairwise_omega', ascending=False)
    bars = ax.barh(pw_df['common_name'], pw_df['median_pairwise_omega'],
                   color=[color_map[n] for n in pw_df['common_name']])
    ax.axvline(0.15, color='blue', linestyle=':', alpha=0.5, label='Bird avg (0.15)')
    ax.axvline(0.40, color='darkorange', linestyle=':', alpha=0.5, label='Duplicated gene avg')
    ax.axvline(1.0, color='red', linestyle='--', alpha=0.3, label='Neutral')
    for bar, val in zip(bars, pw_df['median_pairwise_omega']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    ax.legend(fontsize=7, loc='lower right')
ax.set_xlabel('Median pairwise omega')
ax.set_title('G. Pairwise dN/dS Distribution')

# H: MROH content fraction
ax = fig.add_subplot(gs[2, 1])
mroh_pct = df['mean_mroh_pct'].fillna(0).values
bars = ax.bar(df['common_name'], mroh_pct, color=[color_map[n] for n in df['common_name']])
for bar, val in zip(bars, mroh_pct):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}%', ha='center', fontsize=9, fontweight='bold')
ax.set_ylabel('Mean % of span that is MROH CDS')
ax.set_title('H. MROH Content Fraction')
ax.tick_params(axis='x', rotation=30)

# I: Divergence vs copy number scatter
ax = fig.add_subplot(gs[2, 2])
has_div = df['JC_mean'].notna()
if has_div.any():
    d = df[has_div]
    for _, row in d.iterrows():
        ax.scatter(row['n_copies'], row['JC_mean'],
                   c=[color_map[row['common_name']]], s=100, edgecolors='black',
                   linewidth=0.5, zorder=5)
        ax.annotate(row['common_name'], (row['n_copies'], row['JC_mean']),
                    xytext=(5, 5), textcoords='offset points', fontsize=7)
    # Add correlation if enough points
    if len(d) >= 3:
        r, p = sp_stats.pearsonr(d['n_copies'], d['JC_mean'])
        ax.set_title(f'I. Copy Number vs Divergence\n(r={r:.2f}, p={p:.3f})')
    else:
        ax.set_title('I. Copy Number vs Divergence')
    ax.axhline(0.03, color='blue', linestyle=':', alpha=0.5, label='Genomic baseline')
    ax.legend(fontsize=8)
ax.set_xlabel('Gene copies')
ax.set_ylabel('Mean JC-corrected divergence')

plt.savefig(FIG_DIR / 'cross_species_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {FIG_DIR / 'cross_species_comparison.png'}")

# ── Figure 2: Evidence summary heatmap ─────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

# Build evidence matrix
evidence_cols = [
    ('Copies\n(n)', 'n_copies', 100, 600),
    ('Intact\n(%)', 'pct_intact', 0, 100),
    ('Divergence\n(JC)', 'JC_mean', 0, 0.6),
    ('Fold\nElevation', 'fold_elevation', 0, 20),
    ('M0\nomega', 'M0_omega', 0, 2),
    ('Pairwise\nomega', 'median_pairwise_omega', 0, 0.5),
    ('BEB\nSites', 'BEB_total', 0, 60),
    ('BEB\nomega', 'BEB_mean_omega', 0, 5),
    ('MROH\n(%)', 'mean_mroh_pct', 0, 70),
    ('HEAT\nRepeats', 'mean_heat_repeats', 0, 15),
]

evidence_matrix = np.zeros((len(df), len(evidence_cols)))
col_labels = []
for j, (label, col, vmin, vmax) in enumerate(evidence_cols):
    col_labels.append(label)
    if col in df.columns:
        vals = df[col].fillna(0).values
        # Normalize to 0-1
        if vmax > vmin:
            evidence_matrix[:, j] = np.clip((vals - vmin) / (vmax - vmin), 0, 1)

im = ax.imshow(evidence_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)

ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, fontsize=9)
ax.set_yticks(range(len(df)))
ax.set_yticklabels([f"{row['common_name']}\n({row['species_name']})" for _, row in df.iterrows()],
                   fontsize=9)

# Add values
for i in range(len(df)):
    for j, (label, col, vmin, vmax) in enumerate(evidence_cols):
        val = df.iloc[i].get(col, 0)
        if pd.isna(val):
            txt = '—'
        elif isinstance(val, float):
            txt = f'{val:.2f}' if val < 10 else f'{val:.0f}'
        else:
            txt = str(int(val))
        color = 'white' if evidence_matrix[i, j] > 0.6 else 'black'
        ax.text(j, i, txt, ha='center', va='center', fontsize=8, color=color)

plt.colorbar(im, ax=ax, label='Normalized value', shrink=0.7)
ax.set_title('MROH6 Evidence Summary — All Species\n(Higher values = stronger signal)',
             fontsize=13, pad=15)
plt.tight_layout()
plt.savefig(FIG_DIR / 'cross_species_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {FIG_DIR / 'cross_species_heatmap.png'}")

# ── 4b. Presentation-quality comparison figures ──────────────────────
print("\n── 4b. Generating presentation-quality comparison figures ──")

COMP_FIG_DIR = RESULTS / 'cross_species_figures'
COMP_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Consistent species ordering and labels
SPECIES_DISPLAY_ORDER = [
    "Zebra finch", "Bengalese finch", "House sparrow",
    "White-throated sparrow", "Swamp sparrow",
]
SHORT_LABELS = ["ZF", "WRM", "HS", "WTS", "SS"]
FULL_LABELS = [
    "Zebra finch\n(T. guttata)",
    "White-rumped\nmunia\n(L. striata)",
    "House sparrow\n(P. domesticus)",
    "White-throated\nsparrow\n(Z. albicollis)",
    "Swamp sparrow\n(M. georgiana)",
]
PHENO_PAL = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261"]
CHROM_COLORS = {
    "micro":     "#D62828",
    "macro":     "#457B9D",
    "ancestral": "#2A9D8F",
    "sex":       "#F4A261",
}

# Build ordered data arrays from the DataFrame
def get_ordered(col, default=0):
    vals = []
    for sp in SPECIES_DISPLAY_ORDER:
        match = df[df['common_name'] == sp]
        if len(match) > 0:
            v = match.iloc[0].get(col, default)
            vals.append(v if pd.notna(v) else default)
        else:
            vals.append(default)
    return vals

x = np.arange(len(SPECIES_DISPLAY_ORDER))
width = 0.55

def add_val_labels(ax, bars, fmt="{:.0f}", fs=8, offset=0):
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                    fmt.format(h), ha="center", va="bottom", fontsize=fs)

# ── Fig P1: Copy number + intact fraction ────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

copies = get_ordered('n_copies')
bars = ax1.bar(x, copies, width, color=PHENO_PAL, edgecolor="white", linewidth=0.8)
add_val_labels(ax1, bars, offset=5)
ax1.set_xticks(x); ax1.set_xticklabels(SHORT_LABELS, fontsize=11)
ax1.set_ylabel("Total MROH6 copies", fontsize=12)
ax1.set_title("A.  MROH6 Copy Number", fontsize=13, fontweight="bold", loc="left")
ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
ax1.set_ylim(0, max(copies) * 1.15)

intact_pct_ord = get_ordered('pct_intact')
bars2 = ax2.bar(x, intact_pct_ord, width, color=PHENO_PAL, edgecolor="white", linewidth=0.8)
add_val_labels(ax2, bars2, fmt="{:.1f}%", offset=1)
ax2.set_xticks(x); ax2.set_xticklabels(SHORT_LABELS, fontsize=11)
ax2.set_ylabel("Intact copies (≥80% coverage), %", fontsize=12)
ax2.set_title("B.  Intact Copy Fraction", fontsize=13, fontweight="bold", loc="left")
ax2.axhline(50, color="grey", ls="--", lw=0.8, alpha=0.5)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
ax2.set_ylim(0, 100)

import matplotlib.patches as mpatches
legend_handles = [mpatches.Patch(color=PHENO_PAL[i], label=FULL_LABELS[i].replace("\n", " "))
                  for i in range(5)]
fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("MROH6 Copy Number Across Five Passeriformes", fontsize=15,
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(COMP_FIG_DIR / '01_copy_number_intact.png', dpi=200, bbox_inches='tight',
            facecolor='white')
plt.close()
print(f"  Saved: {COMP_FIG_DIR / '01_copy_number_intact.png'}")

# ── Fig P2: Chromosomal distribution ─────────────────────────────────
import csv as csv_mod

SPECIES_DIRS_MAP = {
    "Zebra finch": "zebra_finch",
    "Bengalese finch": "lonchura_striata",
    "House sparrow": "passer_domesticus",
    "White-throated sparrow": "zonotrichia_albicollis",
    "Swamp sparrow": "melospiza_georgiana",
}

chrom_counts = []
for sp in SPECIES_DISPLAY_ORDER:
    sp_dir = SPECIES_DIRS_MAP[sp]
    loci_path = DATA_PROC / sp_dir / f"mroh6_{sp_dir}_loci_table.csv"
    cats = {"micro": 0, "macro": 0, "ancestral": 0, "sex": 0}
    if loci_path.exists():
        with open(loci_path) as f:
            reader = csv_mod.DictReader(f)
            for r in reader:
                cc = r["chrom_class"]
                if "micro" in cc:
                    cats["micro"] += 1
                elif "sex" in cc:
                    cats["sex"] += 1
                elif "ancestral" in cc:
                    cats["ancestral"] += 1
                else:
                    cats["macro"] += 1
    chrom_counts.append(cats)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
category_order = ["micro", "macro", "ancestral", "sex"]
cat_labels = ["Microchromosome", "Macrochromosome", "Ancestral (chr7/10)", "Sex (Z/W)"]

bottoms = np.zeros(5)
for cat, label in zip(category_order, cat_labels):
    vals = np.array([chrom_counts[i][cat] for i in range(5)])
    ax1.bar(x, vals, width, bottom=bottoms, color=CHROM_COLORS[cat],
            edgecolor="white", linewidth=0.5, label=label)
    for i, v in enumerate(vals):
        if v > 20:
            ax1.text(x[i], bottoms[i] + v / 2, str(v),
                     ha="center", va="center", fontsize=7, color="white", fontweight="bold")
    bottoms += vals
ax1.set_xticks(x); ax1.set_xticklabels(SHORT_LABELS, fontsize=11)
ax1.set_ylabel("Number of MROH6 copies", fontsize=12)
ax1.set_title("A.  Chromosomal Distribution (counts)", fontsize=13, fontweight="bold", loc="left")
ax1.legend(fontsize=9, loc="upper left", framealpha=0.9)
ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

totals = np.array([sum(chrom_counts[i].values()) for i in range(5)], dtype=float)
bottoms = np.zeros(5)
for cat, label in zip(category_order, cat_labels):
    vals = np.array([chrom_counts[i][cat] for i in range(5)])
    pcts = vals / totals * 100
    ax2.bar(x, pcts, width, bottom=bottoms, color=CHROM_COLORS[cat],
            edgecolor="white", linewidth=0.5, label=label)
    for i, v in enumerate(pcts):
        if v > 8:
            ax2.text(x[i], bottoms[i] + v / 2, f"{v:.0f}%",
                     ha="center", va="center", fontsize=7, color="white", fontweight="bold")
    bottoms += pcts
ax2.set_xticks(x); ax2.set_xticklabels(SHORT_LABELS, fontsize=11)
ax2.set_ylabel("% of copies", fontsize=12)
ax2.set_title("B.  Chromosomal Distribution (proportional)", fontsize=13, fontweight="bold", loc="left")
ax2.set_ylim(0, 100)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

fig.suptitle("MROH6 Macro- vs Microchromosome Distribution", fontsize=15,
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(COMP_FIG_DIR / '02_chromosomal_distribution.png', dpi=200, bbox_inches='tight',
            facecolor='white')
plt.close()
print(f"  Saved: {COMP_FIG_DIR / '02_chromosomal_distribution.png'}")

# ── Fig P3: Divergence & fold elevation ──────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

divs = get_ordered('JC_mean')
bars = ax1.bar(x, divs, width, color=PHENO_PAL, edgecolor="white", linewidth=0.8)
add_val_labels(ax1, bars, fmt="{:.3f}", fs=9, offset=0.005)
ax1.axhline(0.03, color="#E63946", ls="--", lw=1.2, alpha=0.7)
ax1.text(4.4, 0.035, "Genomic\nbaseline\n(0.03)", fontsize=8, color="#E63946", ha="center", va="bottom")
ax1.set_xticks(x); ax1.set_xticklabels(SHORT_LABELS, fontsize=11)
ax1.set_ylabel("Mean JC-corrected divergence", fontsize=12)
ax1.set_title("A.  Pairwise Divergence", fontsize=13, fontweight="bold", loc="left")
ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
ax1.set_ylim(0, max(divs) * 1.2)

folds = get_ordered('fold_elevation')
bars2 = ax2.bar(x, folds, width, color=PHENO_PAL, edgecolor="white", linewidth=0.8)
add_val_labels(ax2, bars2, fmt="{:.1f}×", fs=10, offset=0.2)
ax2.axhline(1, color="grey", ls="--", lw=0.8, alpha=0.5)
ax2.set_xticks(x); ax2.set_xticklabels(SHORT_LABELS, fontsize=11)
ax2.set_ylabel("Fold elevation over genomic baseline", fontsize=12)
ax2.set_title("B.  Mutation Rate Elevation", fontsize=13, fontweight="bold", loc="left")
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
ax2.set_ylim(0, max(folds) * 1.15)

fig.suptitle("MROH6 Sequence Divergence Across Five Passeriformes", fontsize=15,
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(COMP_FIG_DIR / '03_divergence_fold_elevation.png', dpi=200, bbox_inches='tight',
            facecolor='white')
plt.close()
print(f"  Saved: {COMP_FIG_DIR / '03_divergence_fold_elevation.png'}")

# ── Fig P4: PAML dN/dS and LRT ──────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

omegas = get_ordered('M0_omega')
colors_omega = [PHENO_PAL[i] if omegas[i] > 0 else "#cccccc" for i in range(5)]
bars = ax1.bar(x, omegas, width, color=colors_omega, edgecolor="white", linewidth=0.8)
for i, w_val in enumerate(omegas):
    if w_val > 0:
        ax1.text(x[i], w_val + 0.02, f"{w_val:.3f}", ha="center", va="bottom", fontsize=10)
    else:
        ax1.text(x[i], 0.05, "N/A", ha="center", va="bottom", fontsize=9, color="#999999", style="italic")
ax1.axhline(1.0, color="#E63946", ls="--", lw=1.2, alpha=0.7)
ax1.text(-0.4, 1.03, "ω = 1\n(neutral)", fontsize=8, color="#E63946", va="bottom")
ax1.set_xticks(x); ax1.set_xticklabels(SHORT_LABELS, fontsize=11)
ax1.set_ylabel("M0 ω (dN/dS)", fontsize=12)
ax1.set_title("A.  Global dN/dS (M0 model)", fontsize=13, fontweight="bold", loc="left")
ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
valid_omegas = [w for w in omegas if w > 0]
ax1.set_ylim(0, max(valid_omegas) * 1.25 if valid_omegas else 2.0)

lrt_m1m2 = get_ordered('LRT_M1a_vs_M2a')
lrt_m7m8 = get_ordered('LRT_M7_vs_M8')
bw = 0.35
b1 = ax2.bar(x - bw / 2, lrt_m1m2, bw, color="#457B9D", edgecolor="white", linewidth=0.8, label="M1a vs M2a")
b2 = ax2.bar(x + bw / 2, lrt_m7m8, bw, color="#E9C46A", edgecolor="white", linewidth=0.8, label="M7 vs M8")
ax2.axhline(5.99, color="grey", ls=":", lw=1, alpha=0.6)
ax2.axhline(13.82, color="#E63946", ls="--", lw=1, alpha=0.6)
ax2.text(4.55, 6.5, "p = 0.05", fontsize=7, color="grey")
ax2.text(4.55, 14.5, "p = 0.001", fontsize=7, color="#E63946")
for i in range(5):
    if lrt_m1m2[i] > 0:
        ax2.text(x[i] - bw / 2, lrt_m1m2[i] + 0.8, "***", ha="center", fontsize=9, fontweight="bold", color="#457B9D")
    if lrt_m7m8[i] > 0:
        ax2.text(x[i] + bw / 2, lrt_m7m8[i] + 0.8, "***", ha="center", fontsize=9, fontweight="bold", color="#E9C46A")
    if lrt_m1m2[i] == 0 and lrt_m7m8[i] == 0:
        ax2.text(x[i], 1, "codeml\nnot converged", ha="center", va="bottom", fontsize=7, color="#999999", style="italic")
ax2.set_xticks(x); ax2.set_xticklabels(SHORT_LABELS, fontsize=11)
ax2.set_ylabel("2ΔlnL (LRT statistic)", fontsize=12)
ax2.set_title("B.  Positive Selection LRT", fontsize=13, fontweight="bold", loc="left")
ax2.legend(fontsize=10, loc="upper left", framealpha=0.9)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

fig.suptitle("MROH6 Selection Pressure (PAML codeml Site Models)", fontsize=15,
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(COMP_FIG_DIR / '04_paml_dnds_lrt.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {COMP_FIG_DIR / '04_paml_dnds_lrt.png'}")

# ── Fig P5: BEB lollipop plots ───────────────────────────────────────
beb_species = {
    "Zebra finch":            ("zebra_finch",            PHENO_PAL[0], "ZF"),
    "White-throated sparrow": ("zonotrichia_albicollis", PHENO_PAL[3], "WTS"),
    "Swamp sparrow":         ("melospiza_georgiana",     PHENO_PAL[4], "SS"),
}

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, gridspec_kw={"hspace": 0.35})
max_site = 0
beb_loaded = {}
for sp, (sp_dir, color, short) in beb_species.items():
    beb_path = RESULTS / sp_dir / 'tables' / 'beb_selected_sites.csv'
    sites = []
    if beb_path.exists():
        beb_df_sp = pd.read_csv(beb_path)
        for _, r in beb_df_sp.iterrows():
            sites.append({"site": int(r["site"]), "aa": r["aa"], "prob": float(r["prob"]),
                          "omega": float(r["omega"]), "significance": r.get("significance", "")})
        if sites:
            max_site = max(max_site, max(s["site"] for s in sites))
    beb_loaded[sp] = sites

for idx, (sp, (sp_dir, color, short)) in enumerate(beb_species.items()):
    ax = axes[idx]
    sites = beb_loaded[sp]
    if not sites:
        ax.text(0.5, 0.5, "No BEB sites", transform=ax.transAxes, ha="center", fontsize=12, color="grey")
        ax.set_ylabel(short, fontsize=12, fontweight="bold")
        continue

    for s in sites:
        alpha = 1.0 if s["significance"] == "**" else (0.8 if s["significance"] == "*" else 0.5)
        lw = 1.5 if s["significance"] == "**" else (1.2 if s["significance"] == "*" else 1.0)
        ax.vlines(s["site"], 0, s["prob"], color=color, alpha=alpha * 0.6, linewidth=lw)
        ax.plot(s["site"], s["prob"], "o", color=color, markersize=5 + s["prob"] * 3,
                alpha=alpha, markeredgecolor="white", markeredgewidth=0.3)
        if s["significance"] == "**":
            ax.text(s["site"], s["prob"] + 0.02, f'{s["aa"]}{s["site"]}', ha="center",
                    va="bottom", fontsize=7, fontweight="bold", color=color)

    ax.axhline(0.95, color="grey", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(0.99, color="#E63946", ls="--", lw=0.8, alpha=0.5)
    if idx == 0:
        ax.text(max_site + 1, 0.952, "P=0.95", fontsize=7, color="grey", va="center")
        ax.text(max_site + 1, 0.992, "P=0.99", fontsize=7, color="#E63946", va="center")
    n_95 = sum(1 for s in sites if s["prob"] >= 0.95)
    n_99 = sum(1 for s in sites if s["prob"] >= 0.99)
    ax.text(0.98, 0.95, f"n={len(sites)}  ({n_95} at P≥0.95, {n_99} at P≥0.99)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey", alpha=0.9))
    ax.set_ylabel(f"BEB P(ω>1)\n{short}", fontsize=10, fontweight="bold")
    ax.set_ylim(0.4, 1.08)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

axes[-1].set_xlabel("Alignment codon position", fontsize=12)
axes[-1].set_xlim(0, max_site + 5)
fig.suptitle("BEB Positively Selected Sites — MROH6 (M2a model)", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(COMP_FIG_DIR / '05_beb_selected_sites.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {COMP_FIG_DIR / '05_beb_selected_sites.png'}")

# ── Fig P6: Summary dashboard ────────────────────────────────────────
fig = plt.figure(figsize=(18, 11))
gs2 = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

# A: Copy number
ax = fig.add_subplot(gs2[0, 0])
bars = ax.bar(x, copies, width, color=PHENO_PAL, edgecolor="white", linewidth=0.8)
add_val_labels(ax, bars, offset=5, fs=8)
ax.set_xticks(x); ax.set_xticklabels(SHORT_LABELS, fontsize=10)
ax.set_ylabel("Total copies"); ax.set_title("A. Copy Number", fontweight="bold", loc="left")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.set_ylim(0, max(copies) * 1.18)

# B: Intact fraction
ax = fig.add_subplot(gs2[0, 1])
bars = ax.bar(x, intact_pct_ord, width, color=PHENO_PAL, edgecolor="white", linewidth=0.8)
add_val_labels(ax, bars, fmt="{:.0f}%", offset=1, fs=8)
ax.axhline(50, color="grey", ls="--", lw=0.8, alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels(SHORT_LABELS, fontsize=10)
ax.set_ylabel("% intact (≥80%)"); ax.set_title("B. Intact Fraction", fontweight="bold", loc="left")
ax.set_ylim(0, 105); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# C: Chromosomal distribution
ax = fig.add_subplot(gs2[0, 2])
totals_c = np.array([sum(chrom_counts[i].values()) for i in range(5)], dtype=float)
bottoms = np.zeros(5)
for cat, label in zip(category_order, ["Micro", "Macro", "Ancestral", "Sex"]):
    vals = np.array([chrom_counts[i][cat] for i in range(5)])
    pcts = vals / totals_c * 100
    ax.bar(x, pcts, width, bottom=bottoms, color=CHROM_COLORS[cat], edgecolor="white", linewidth=0.5, label=label)
    bottoms += pcts
ax.set_xticks(x); ax.set_xticklabels(SHORT_LABELS, fontsize=10)
ax.set_ylabel("% of copies"); ax.set_title("C. Chromosome Class", fontweight="bold", loc="left")
ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
ax.set_ylim(0, 100); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# D: Fold elevation
ax = fig.add_subplot(gs2[1, 0])
bars = ax.bar(x, folds, width, color=PHENO_PAL, edgecolor="white", linewidth=0.8)
add_val_labels(ax, bars, fmt="{:.1f}×", offset=0.2, fs=9)
ax.set_xticks(x); ax.set_xticklabels(SHORT_LABELS, fontsize=10)
ax.set_ylabel("Fold over baseline"); ax.set_title("D. Mutation Rate Elevation", fontweight="bold", loc="left")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.set_ylim(0, max(folds) * 1.18)

# E: dN/dS (M0)
ax = fig.add_subplot(gs2[1, 1])
colors_o2 = [PHENO_PAL[i] if omegas[i] > 0 else "#cccccc" for i in range(5)]
bars = ax.bar(x, omegas, width, color=colors_o2, edgecolor="white", linewidth=0.8)
for i, w_val in enumerate(omegas):
    if w_val > 0:
        ax.text(x[i], w_val + 0.02, f"{w_val:.2f}", ha="center", va="bottom", fontsize=9)
    else:
        ax.text(x[i], 0.05, "N/A", ha="center", fontsize=8, color="#999", style="italic")
ax.axhline(1.0, color="#E63946", ls="--", lw=1, alpha=0.6)
ax.set_xticks(x); ax.set_xticklabels(SHORT_LABELS, fontsize=10)
ax.set_ylabel("M0 ω (dN/dS)"); ax.set_title("E. Global dN/dS", fontweight="bold", loc="left")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.set_ylim(0, 2.0)

# F: BEB sites
ax = fig.add_subplot(gs2[1, 2])
beb_all_ord = get_ordered('BEB_total')
beb_95_ord = get_ordered('BEB_sites_95')
beb_99_ord = get_ordered('BEB_sig_99')
bw3 = 0.25
ax.bar(x - bw3, beb_all_ord, bw3, color=PHENO_PAL, edgecolor="white", label="All BEB", alpha=0.5)
ax.bar(x, beb_95_ord, bw3, color=PHENO_PAL, edgecolor="white", label="P ≥ 0.95", alpha=0.75)
ax.bar(x + bw3, beb_99_ord, bw3, color=PHENO_PAL, edgecolor="white", label="P ≥ 0.99")
ax.set_xticks(x); ax.set_xticklabels(SHORT_LABELS, fontsize=10)
ax.set_ylabel("Number of BEB sites"); ax.set_title("F. BEB Selected Sites", fontweight="bold", loc="left")
ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle("MROH6 Cross-Species Comparison — Five Passeriformes", fontsize=17, fontweight="bold", y=1.01)
plt.savefig(COMP_FIG_DIR / '06_summary_dashboard.png', dpi=250, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {COMP_FIG_DIR / '06_summary_dashboard.png'}")

print(f"\n  All presentation figures saved to {COMP_FIG_DIR}/")

# ── 5. Key findings ──────────────────────────────────────────────────
print("\n── 5. Key Findings ──")

# RNA-mediated duplication evidence
print("\n  RNA-MEDIATED DUPLICATION EVIDENCE:")
for _, row in df.iterrows():
    fold = row.get('fold_elevation', 0)
    if fold > 3:
        verdict = "STRONG (>3x baseline)"
    elif fold > 1:
        verdict = "Moderate"
    else:
        verdict = "Insufficient data"
    print(f"  {row['common_name']:25s}  {fold:>5.1f}x elevation — {verdict}")

# Positive selection evidence
print("\n  POSITIVE SELECTION EVIDENCE:")
for _, row in df.iterrows():
    m0 = row.get('M0_omega', np.nan)
    beb = row.get('BEB_total', 0)
    lrt_p = row.get('LRT_M1a_vs_M2a_p', np.nan)

    if pd.notna(m0) and pd.notna(lrt_p):
        if m0 > 1 and lrt_p < 0.05 and beb > 0:
            verdict = f"STRONG (omega={m0:.2f}, {beb} BEB sites, LRT p={lrt_p:.1e})"
        elif m0 > 1:
            verdict = f"Moderate (omega={m0:.2f}, LRT p={lrt_p:.1e})"
        else:
            verdict = f"Weak (omega={m0:.2f})"
    else:
        verdict = "N/A (PAML failed — gappy alignment)"
    print(f"  {row['common_name']:25s}  {verdict}")

# Consistency across species
n_elevated = sum(1 for _, r in df.iterrows() if r.get('fold_elevation', 0) > 3)
n_selection = sum(1 for _, r in df.iterrows()
                  if r.get('M0_omega', 0) > 1 and r.get('LRT_M1a_vs_M2a_p', 1) < 0.05)

print(f"\n  CONSISTENCY: {n_elevated}/{len(df)} species show >3x mutation rate elevation")
print(f"  CONSISTENCY: {n_selection}/{len(df)} species show significant positive selection (LRT)")

# ── Summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  CROSS-SPECIES COMPARISON SUMMARY")
print("=" * 70)
print(f"  Species analyzed:          {len(df)}")
print(f"  Total gene copies:         {df['n_copies'].sum()}")
print(f"  Copy range:                {df['n_copies'].min()} — {df['n_copies'].max()}")
mean_fold = df['fold_elevation'].mean()
if pd.notna(mean_fold):
    print(f"  Mean fold elevation:       {mean_fold:.1f}x")
print(f"  RNA signal (>3x):          {n_elevated}/{len(df)} species")
print(f"  Positive selection (LRT):  {n_selection}/{len(df)} species")
print(f"\n  Reports: cross_species_comparison.csv, cross_species_lrt.csv")
print(f"  Figures: cross_species_comparison.png, cross_species_heatmap.png")
print("=" * 70)
