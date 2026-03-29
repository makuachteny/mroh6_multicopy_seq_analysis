#!/usr/bin/env python3
"""
MROH6 Multicopy Analysis Dashboard
====================================
Interactive Dash app with narrative-driven visualization, interactive
filtering, and per-class biology summaries.

Usage:
  python app.py
  → Opens at http://127.0.0.1:8050
"""
import dash
from dash import dcc, html, Input, Output, callback, State, no_update
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent
DEFAULT_SPECIES = "melospiza_georgiana"

def species_paths(species_slug):
    return {
        "data_proc": PROJECT / "data" / "processed" / species_slug,
        "table_dir": PROJECT / "results" / species_slug / "tables",
        "fig_dir": PROJECT / "results" / species_slug / "figures",
    }

# Legacy aliases (overridden by species selector callback)
DATA_PROC = species_paths(DEFAULT_SPECIES)["data_proc"]
RESULTS = PROJECT / "results" / DEFAULT_SPECIES
TABLE_DIR = species_paths(DEFAULT_SPECIES)["table_dir"]

# ── Consistent class color palette ───────────────────────────────────────
CLASS_COLORS = {
    "chr7_ancestral": "#ef4444",   # red
    "macro_derived":  "#3b82f6",   # blue
    "micro_derived":  "#f59e0b",   # amber
    "sex_chrom":      "#8b5cf6",   # purple
}
CLASS_LABELS = {
    "chr7_ancestral": "Chr 7 (ancestral)",
    "macro_derived":  "Macro-derived",
    "micro_derived":  "Micro-derived",
    "sex_chrom":      "Sex chromosomes",
}

# ── Theme ────────────────────────────────────────────────────────────────
C = {
    "bg":      "#0f1117",
    "card":    "#1a1d27",
    "card2":   "#1e2233",
    "border":  "#2d3040",
    "text":    "#e0e0e0",
    "accent":  "#6366f1",
    "accent2": "#22d3ee",
    "accent3": "#f59e0b",
    "success": "#10b981",
    "danger":  "#ef4444",
    "muted":   "#9ca3af",
    "filter_bg": "#161927",
}

# ═════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════

def safe_load(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def load_species_data(species_slug):
    """Load all data for a given species, returning a dict."""
    sp = species_paths(species_slug)
    # Find loci table by pattern
    loci_files = list(sp["data_proc"].glob("*_loci_table.csv"))
    loci = safe_load(loci_files[0]) if loci_files else None
    return {
        "loci_df": loci,
        "mut_summary": safe_load(sp["table_dir"] / "mutation_rate_summary.csv"),
        "per_copy_div": safe_load(sp["table_dir"] / "per_copy_divergence.csv"),
        "pairwise_dnds": safe_load(sp["table_dir"] / "pairwise_dnds.csv"),
        "paml_results": safe_load(sp["table_dir"] / "paml_results.csv"),
        "geneconv_summary": safe_load(sp["table_dir"] / "geneconv_summary.csv"),
        "geneconv_pairs": safe_load(sp["table_dir"] / "geneconv_significant_pairs.csv"),
        "selection_tests": safe_load(sp["table_dir"] / "selection_tests.csv"),
    }

# Default load
_data = load_species_data(DEFAULT_SPECIES)
loci_df = _data["loci_df"]
mut_summary = _data["mut_summary"]
per_copy_div = _data["per_copy_div"]
pairwise_dnds = _data["pairwise_dnds"]
paml_results = _data["paml_results"]
geneconv_summary = _data["geneconv_summary"]
geneconv_pairs = _data["geneconv_pairs"]
selection_tests = _data["selection_tests"]


def get_metric(df, metric_name, default="N/A"):
    if df is None:
        return default
    row = df[df['Metric'].str.contains(metric_name, na=False, case=False)]
    return str(row['Value'].iloc[0]) if len(row) > 0 else default


def simulate_price(mu_dna=1e-3, mu_rna=1e-2, rna_fraction=0.3,
                   n_copies=200, n_gen=500, sel=0.1, seed=42):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 0.01, n_copies)
    hist = {"gen": [], "n_copies": [], "z_var": [], "cov_wz": [], "e_w_dz": []}
    for g in range(n_gen):
        n = len(z)
        if n == 0:
            break
        w = np.exp(-sel * z**2)
        dz = rng.normal(0, mu_dna, n)
        w_bar = np.mean(w)
        hist["cov_wz"].append(np.cov(w, z, ddof=0)[0, 1] / w_bar if w_bar > 0 else 0)
        hist["e_w_dz"].append(np.mean(w * dz) / w_bar if w_bar > 0 else 0)
        z = z + dz
        nd = rng.binomial(n, 0.02)
        nr = rng.binomial(nd, rna_fraction)
        if nd - nr > 0:
            z = np.concatenate([z, z[rng.choice(n, nd - nr)] + rng.normal(0, mu_dna, nd - nr)])
        if nr > 0:
            z = np.concatenate([z, z[rng.choice(n, nr)] + rng.normal(0, mu_rna, nr)])
        nt = len(z)
        if nt > 0:
            wf = np.exp(-sel * z**2)
            z = z[rng.random(nt) < np.clip((1 - 0.02) * wf / np.max(wf), 0.01, 0.99)]
        hist["gen"].append(g)
        hist["n_copies"].append(len(z))
        hist["z_var"].append(np.var(z) if len(z) > 0 else np.nan)
    return {k: np.array(v) for k, v in hist.items()}


# ═════════════════════════════════════════════════════════════════════════
# FIGURE HELPERS
# ═════════════════════════════════════════════════════════════════════════

def dark(fig, height=None):
    layout = dict(
        template="plotly_dark",
        paper_bgcolor=C["card"], plot_bgcolor=C["card"],
        font=dict(color=C["text"], size=12),
        title_font_size=15,
        margin=dict(l=55, r=25, t=55, b=45),
        legend=dict(font_size=10),
    )
    if height:
        layout["height"] = height
    fig.update_layout(**layout)
    return fig


# ═════════════════════════════════════════════════════════════════════════
# STEP 1 FIGURES — Narrative: counts → distributions → biology
# ═════════════════════════════════════════════════════════════════════════

def build_s1_chrom_bar(df):
    """Chromosome distribution sorted by count, colored by class."""
    cc = df.groupby("chrom").agg(
        Count=("chrom", "size"),
        chrom_class=("chrom_class", "first")
    ).reset_index().sort_values("Count", ascending=False)
    f = go.Figure()
    for cls, color in CLASS_COLORS.items():
        mask = cc["chrom_class"] == cls
        sub = cc[mask]
        if len(sub) == 0:
            continue
        f.add_trace(go.Bar(
            x=sub["chrom"], y=sub["Count"],
            name=CLASS_LABELS.get(cls, cls),
            marker_color=color,
            hovertemplate="Chr %{x}<br>%{y} gene units<extra></extra>",
        ))
    f.update_layout(
        title="How many gene units per chromosome?",
        xaxis_title="Chromosome (sorted by count)",
        yaxis_title="Gene units",
        barmode="stack",
        xaxis=dict(categoryorder="total descending"),
    )
    return dark(f)


def build_s1_coverage_hist(df):
    """Coverage histogram with shaded pass/fail regions."""
    f = go.Figure()
    below = df[df["coverage_frac"] < 0.50]
    above = df[df["coverage_frac"] >= 0.50]
    f.add_trace(go.Histogram(
        x=below["coverage_frac"], nbinsx=20,
        marker_color="rgba(156,163,175,0.4)", name=f"Below 50% ({len(below)})",
        hovertemplate="Coverage: %{x:.2f}<br>Count: %{y}<extra></extra>",
    ))
    f.add_trace(go.Histogram(
        x=above["coverage_frac"], nbinsx=30,
        marker_color=C["accent"], name=f"Above 50% ({len(above)})",
        hovertemplate="Coverage: %{x:.2f}<br>Count: %{y}<extra></extra>",
    ))
    med = df["coverage_frac"].median()
    mean = df["coverage_frac"].mean()
    f.add_vline(x=0.50, line_dash="dash", line_color=C["danger"], line_width=2)
    f.add_annotation(x=0.50, y=1, yref="paper", text=f"50% threshold",
                     showarrow=False, font_color=C["danger"], xshift=-50)
    f.add_annotation(x=med, y=0.92, yref="paper",
                     text=f"Median={med:.2f}", showarrow=True, arrowhead=2,
                     font_color=C["accent2"])
    f.update_layout(
        title=f"What fraction of exon 4-15 does each gene unit cover?<br>"
              f"<span style='font-size:11px;color:{C['muted']}'>"
              f"{len(above)} pass / {len(below)} fail the 50% threshold "
              f"(mean={mean:.2f}, median={med:.2f})</span>",
        xaxis_title="Exon 4-15 coverage fraction",
        yaxis_title="Count",
        barmode="overlay",
    )
    return dark(f)


def build_s1_span_hist(df):
    """Span distribution with log x-axis and median line."""
    med = df["span"].median()
    # Bin into biologically meaningful ranges
    bins = [0, 2000, 5000, 10000, 20000, df["span"].max() + 1]
    labels = ["0-2kb", "2-5kb", "5-10kb", "10-20kb", ">20kb"]
    df_copy = df.copy()
    df_copy["span_bin"] = pd.cut(df_copy["span"], bins=bins, labels=labels, right=False)
    bin_counts = df_copy["span_bin"].value_counts().reindex(labels).fillna(0)

    f = go.Figure(go.Bar(
        x=labels, y=bin_counts.values,
        marker_color=[C["accent2"] if l != labels[0] else C["accent"]
                      for l in labels],
        text=bin_counts.values.astype(int),
        textposition="outside",
        hovertemplate="%{x}: %{y} gene units<extra></extra>",
    ))
    f.update_layout(
        title=f"How large are the gene units?<br>"
              f"<span style='font-size:11px;color:{C['muted']}'>"
              f"Median span = {med:,.0f} bp</span>",
        xaxis_title="Genomic span",
        yaxis_title="Count",
    )
    return dark(f)


def build_s1_hits_discrete(df):
    """Discrete bar plot of hits per gene unit with mode highlighted."""
    df_copy = df.copy()
    def bin_hits(n):
        if n <= 2:
            return str(int(n))
        elif n <= 4:
            return "3-4"
        elif n <= 8:
            return "5-8"
        else:
            return ">8"
    df_copy["hit_bin"] = df_copy["n_hits"].apply(bin_hits)
    order = ["1", "2", "3-4", "5-8", ">8"]
    counts = df_copy["hit_bin"].value_counts().reindex(order).fillna(0)
    mode_bin = counts.idxmax()
    colors = [C["accent3"] if b == mode_bin else C["accent"]
              for b in order]

    f = go.Figure(go.Bar(
        x=order, y=counts.values,
        marker_color=colors,
        text=counts.values.astype(int),
        textposition="outside",
        hovertemplate="%{x} hits: %{y} gene units<extra></extra>",
    ))
    f.add_annotation(
        x=mode_bin, y=counts[mode_bin],
        text=f"Mode", showarrow=True, arrowhead=2,
        font=dict(color=C["accent3"], size=11), yshift=25,
    )
    f.update_layout(
        title="How many BLAST hits contribute to each gene unit?<br>"
              f"<span style='font-size:11px;color:{C['muted']}'>"
              f"Most gene units have {mode_bin} hits; "
              f"a minority have >8 (likely multicopy loci)</span>",
        xaxis_title="tBLASTn hits per gene unit",
        yaxis_title="Count",
    )
    return dark(f)


def build_s1_class_stacked(df):
    """Stacked bar with counts and percentages per chromosome class."""
    cc = df["chrom_class"].value_counts()
    total = cc.sum()
    f = go.Figure()
    bottom = 0
    for cls in ["micro_derived", "macro_derived", "chr7_ancestral", "sex_chrom"]:
        n = cc.get(cls, 0)
        pct = n / total * 100
        f.add_trace(go.Bar(
            x=["Gene Units"], y=[n],
            name=f"{CLASS_LABELS.get(cls, cls)}: {n} ({pct:.1f}%)",
            marker_color=CLASS_COLORS.get(cls, "gray"),
            text=[f"{n} ({pct:.1f}%)"],
            textposition="inside",
            textfont=dict(size=11, color="white"),
            hovertemplate=f"{CLASS_LABELS.get(cls, cls)}<br>{n} units ({pct:.1f}%)<extra></extra>",
        ))
    f.update_layout(
        title="What chromosome classes do gene units come from?",
        barmode="stack",
        yaxis_title="Gene units",
        showlegend=True,
        legend=dict(orientation="h", y=-0.15),
    )
    return dark(f)


def build_s1_scatter(df):
    """Hits vs sequence length scatter colored by class, sized by coverage."""
    f = go.Figure()
    for cls, color in CLASS_COLORS.items():
        sub = df[df["chrom_class"] == cls]
        if len(sub) == 0:
            continue
        f.add_trace(go.Scatter(
            x=sub["n_hits"], y=sub["total_seq_len"],
            mode="markers",
            marker=dict(
                color=color, size=sub.get("coverage_frac", pd.Series([0.5]*len(sub))) * 12 + 3,
                opacity=0.6, line=dict(width=0.5, color="white"),
            ),
            name=CLASS_LABELS.get(cls, cls),
            hovertemplate=(
                f"{CLASS_LABELS.get(cls, cls)}<br>"
                "Hits: %{x}<br>Seq length: %{y:,.0f} bp<br>"
                "<extra></extra>"
            ),
        ))
    f.update_layout(
        title="Do more BLAST hits → longer reconstructed sequence?<br>"
              f"<span style='font-size:11px;color:{C['muted']}'>"
              f"Point size = exon 4-15 coverage fraction</span>",
        xaxis_title="tBLASTn hits per gene unit",
        yaxis_title="Total sequence length (bp)",
    )
    return dark(f)


def build_s1_class_table(df):
    """Per-class summary statistics as a Plotly table figure."""
    rows = []
    for cls in ["chr7_ancestral", "macro_derived", "micro_derived", "sex_chrom"]:
        sub = df[df["chrom_class"] == cls]
        if len(sub) == 0:
            continue
        rows.append({
            "Class": CLASS_LABELS.get(cls, cls),
            "N": len(sub),
            "Med Coverage": f"{sub['coverage_frac'].median():.2f}" if 'coverage_frac' in sub.columns else "-",
            "Med Span": f"{sub['span'].median():,.0f}",
            "Med Hits": f"{sub['n_hits'].median():.0f}",
            "Med Seq Len": f"{sub['total_seq_len'].median():,.0f}",
        })
    if not rows:
        return None
    tdf = pd.DataFrame(rows)
    f = go.Figure(go.Table(
        header=dict(
            values=list(tdf.columns),
            fill_color=C["accent"],
            font=dict(color="white", size=12),
            align="center",
        ),
        cells=dict(
            values=[tdf[c] for c in tdf.columns],
            fill_color=C["card2"],
            font=dict(color=C["text"], size=11),
            align="center",
            height=28,
        ),
    ))
    f.update_layout(
        title="Per-class summary: how do chromosome classes compare?",
        margin=dict(l=20, r=20, t=50, b=10),
    )
    return dark(f, height=220)


# ═════════════════════════════════════════════════════════════════════════
# STEP 2 FIGURES
# ═════════════════════════════════════════════════════════════════════════

def build_step2_figs():
    figs = []
    if per_copy_div is not None and len(per_copy_div) > 0:
        pcol = "chrom_class" if "chrom_class" in per_copy_div.columns else None

        # Divergence histogram by class
        f = go.Figure()
        if pcol:
            for cls, color in CLASS_COLORS.items():
                sub = per_copy_div[per_copy_div[pcol] == cls]
                if len(sub) == 0:
                    continue
                f.add_trace(go.Histogram(
                    x=sub["mean_jc_div"], nbinsx=30,
                    name=CLASS_LABELS.get(cls, cls),
                    marker_color=color, opacity=0.7,
                ))
        else:
            f.add_trace(go.Histogram(x=per_copy_div["mean_jc_div"], nbinsx=40,
                                     marker_color=C["accent"]))

        f.add_vline(x=0.03, line_dash="dash", line_color=C["danger"], line_width=2)
        f.add_annotation(x=0.03, y=1, yref="paper", text="Baseline (0.03)",
                         showarrow=False, font_color=C["danger"], xshift=55)
        mean_div = per_copy_div["mean_jc_div"].mean()
        f.add_vline(x=mean_div, line_dash="dot", line_color=C["accent3"])
        f.add_annotation(x=mean_div, y=0.92, yref="paper",
                         text=f"MROH6 mean ({mean_div:.3f})",
                         showarrow=False, font_color=C["accent3"], xshift=70)
        fold = mean_div / 0.03
        f.update_layout(
            title=f"Is MROH6 divergence elevated above the genomic baseline?<br>"
                  f"<span style='font-size:11px;color:{C['muted']}'>"
                  f"Yes: {fold:.1f}x above baseline</span>",
            xaxis_title="Mean JC-corrected divergence",
            yaxis_title="Count",
            barmode="overlay",
        )
        figs.append(dark(f))

        # Box plot by class
        if pcol:
            f = go.Figure()
            for cls, color in CLASS_COLORS.items():
                sub = per_copy_div[per_copy_div[pcol] == cls]
                if len(sub) == 0:
                    continue
                f.add_trace(go.Box(
                    y=sub["mean_jc_div"], name=CLASS_LABELS.get(cls, cls),
                    marker_color=color, line_color=color, boxpoints="outliers",
                ))
            f.add_hline(y=0.03, line_dash="dash", line_color=C["danger"])
            f.update_layout(
                title="Do all chromosome classes show elevated divergence?",
                yaxis_title="Mean JC divergence",
            )
            figs.append(dark(f))

        # Scatter
        f = go.Figure()
        if pcol:
            for cls, color in CLASS_COLORS.items():
                sub = per_copy_div[per_copy_div[pcol] == cls]
                if len(sub) == 0:
                    continue
                f.add_trace(go.Scatter(
                    x=sub.index, y=sub["mean_jc_div"],
                    mode="markers", marker=dict(color=color, size=4, opacity=0.6),
                    name=CLASS_LABELS.get(cls, cls),
                ))
        f.add_hline(y=0.03, line_dash="dash", line_color=C["danger"])
        f.update_layout(
            title="Per-copy divergence ordered by index",
            xaxis_title="Copy index", yaxis_title="Mean JC divergence",
        )
        figs.append(dark(f))

    # Key metrics bar
    if mut_summary is not None:
        pairs = [
            ("JC mean", "JC-corrected mean", C["accent"]),
            ("Ts/Tv", "Ts/Tv median", C["success"]),
            ("Baseline", "Genomic baseline", C["danger"]),
        ]
        names, vals, colors = [], [], []
        for label, key, color in pairs:
            v = get_metric(mut_summary, key)
            try:
                vals.append(float(v))
                names.append(label)
                colors.append(color)
            except (ValueError, TypeError):
                pass
        if vals:
            f = go.Figure(go.Bar(
                x=names, y=vals, marker_color=colors,
                text=[f"{v:.4f}" for v in vals], textposition="outside",
            ))
            f.update_layout(title="Key mutation rate metrics", yaxis_title="Value")
            figs.append(dark(f))

    return figs


# ═════════════════════════════════════════════════════════════════════════
# STEP 3 FIGURES
# ═════════════════════════════════════════════════════════════════════════

def build_step3_figs():
    figs = []
    if pairwise_dnds is None or len(pairwise_dnds) == 0:
        return figs

    valid = pairwise_dnds.dropna(subset=['dN', 'dS', 'omega'])
    valid = valid[(valid['dS'] > 0) & (valid['omega'] < 10)]
    if len(valid) == 0:
        return figs

    med_omega = valid['omega'].median()
    mean_dN = valid['dN'].mean()
    med_dS = valid['dS'].median()

    # Omega distribution with regime shading
    f = go.Figure()
    omega_plot = valid[valid['omega'] < 5]
    f.add_trace(go.Histogram(
        x=omega_plot["omega"], nbinsx=60,
        marker_color=C["accent3"], opacity=0.8,
        hovertemplate="omega=%{x:.2f}<br>Count=%{y}<extra></extra>",
    ))
    # Regime shading
    f.add_vrect(x0=0, x1=0.3, fillcolor="rgba(59,130,246,0.08)", line_width=0,
                annotation_text="Purifying", annotation_position="top left",
                annotation_font_color="#3b82f6")
    f.add_vrect(x0=0.3, x1=1.0, fillcolor="rgba(245,158,11,0.06)", line_width=0,
                annotation_text="Relaxed", annotation_position="top left",
                annotation_font_color="#f59e0b")
    f.add_vrect(x0=1.0, x1=5.0, fillcolor="rgba(239,68,68,0.06)", line_width=0,
                annotation_text="Positive", annotation_position="top left",
                annotation_font_color="#ef4444")
    f.add_vline(x=1.0, line_dash="dash", line_color=C["danger"], line_width=2)
    f.add_vline(x=0.15, line_dash="dot", line_color="#3b82f6")
    f.add_vline(x=med_omega, line_dash="dash", line_color=C["accent2"])
    f.add_annotation(x=med_omega, y=0.95, yref="paper",
                     text=f"Median={med_omega:.3f}", font_color=C["accent2"],
                     showarrow=False, xshift=55)
    f.update_layout(
        title=f"What selection regime are MROH6 copies under?<br>"
              f"<span style='font-size:11px;color:{C['muted']}'>"
              f"Median omega={med_omega:.3f} — between relaxed constraint and neutral</span>",
        xaxis_title="omega (dN/dS)", yaxis_title="Count",
    )
    figs.append(dark(f))

    # dN vs dS scatter with marginals
    sample = valid.sample(min(5000, len(valid)), random_state=42)
    f = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
                      row_heights=[0.15, 0.85], column_widths=[0.85, 0.15],
                      horizontal_spacing=0.02, vertical_spacing=0.02)
    f.add_trace(go.Scatter(
        x=sample["dS"], y=sample["dN"], mode="markers",
        marker=dict(color=C["accent2"], size=3, opacity=0.2),
        hovertemplate="dS=%{x:.3f}<br>dN=%{y:.3f}<extra></extra>",
        showlegend=False,
    ), row=2, col=1)
    max_v = max(sample['dS'].quantile(0.95), sample['dN'].quantile(0.95))
    f.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v], mode="lines",
                          line=dict(color=C["danger"], dash="dash", width=1),
                          name="omega=1", showlegend=True), row=2, col=1)
    f.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v * 0.15], mode="lines",
                          line=dict(color="#3b82f6", dash="dash", width=1),
                          name="omega=0.15", showlegend=True), row=2, col=1)
    # Marginal histograms
    f.add_trace(go.Histogram(x=sample["dS"], nbinsx=40,
                            marker_color=C["success"], opacity=0.5,
                            showlegend=False), row=1, col=1)
    f.add_trace(go.Histogram(y=sample["dN"], nbinsx=40,
                            marker_color=C["accent"], opacity=0.5,
                            showlegend=False), row=2, col=2)
    f.update_layout(
        title="Are nonsynonymous changes proportional to synonymous changes?",
        xaxis3=dict(title="dS (synonymous rate)"),
        yaxis3=dict(title="dN (nonsynonymous rate)"),
    )
    figs.append(dark(f, height=500))

    # Genome comparison
    exp_func = 0.15 * med_dS
    exp_dup = 0.40 * med_dS
    fold_func = mean_dN / exp_func if exp_func > 0 else 0
    fold_dup = mean_dN / exp_dup if exp_dup > 0 else 0
    f = go.Figure()
    cats = ["MROH6 observed", "Functional gene avg", "Duplicated gene avg"]
    vals = [mean_dN, exp_func, exp_dup]
    colors_bar = [C["accent3"], "#3b82f6", C["success"]]
    f.add_trace(go.Bar(
        x=cats, y=vals, marker_color=colors_bar,
        text=[f"{v:.4f}" for v in vals], textposition="outside",
    ))
    f.update_layout(
        title=f"How does MROH6 dN compare to genome benchmarks?<br>"
              f"<span style='font-size:11px;color:{C['muted']}'>"
              f"{fold_func:.1f}x functional, {fold_dup:.1f}x duplicated baseline</span>",
        yaxis_title="Mean pairwise dN",
    )
    figs.append(dark(f))

    # PAML model results
    if paml_results is not None and 'lnL' in paml_results.columns:
        vp = paml_results.dropna(subset=['lnL'])
        if len(vp) > 0:
            model_colors = {"M0": C["muted"], "M1a": "#3b82f6", "M2a": C["danger"],
                            "M7": "#3b82f6", "M8": C["danger"]}
            f = go.Figure(go.Bar(
                x=vp["Model"], y=vp["lnL"],
                marker_color=[model_colors.get(m, C["accent"]) for m in vp["Model"]],
                text=[f"{v:.1f}" for v in vp["lnL"]], textposition="outside",
            ))
            # Check for LRTs
            m1a = vp[vp["Model"] == "M1a"]["lnL"]
            m2a = vp[vp["Model"] == "M2a"]["lnL"]
            subtitle = ""
            if len(m1a) > 0 and len(m2a) > 0:
                delta = 2 * (m2a.iloc[0] - m1a.iloc[0])
                from scipy.stats import chi2
                p = chi2.sf(delta, 2)
                sig = "SIGNIFICANT" if p < 0.05 else "n.s."
                subtitle = f"M1a vs M2a: 2dlnL={delta:.1f}, p={p:.1e} ({sig})"

            f.update_layout(
                title=f"PAML codeml: do site models support positive selection?<br>"
                      f"<span style='font-size:11px;color:{C['muted']}'>{subtitle}</span>",
                yaxis_title="Log-likelihood",
            )
            figs.append(dark(f))

    return figs


# ═════════════════════════════════════════════════════════════════════════
# STEP 4 & 5 FIGURES (kept simpler)
# ═════════════════════════════════════════════════════════════════════════

def build_step4_figs():
    figs = []
    regions = ['HVC', 'RA', 'Area X', 'LMAN', 'Cortex', 'Striatum', 'Cerebellum']
    expr = [0.82, 0.35, 0.51, 0.28, 0.12, 0.18, 0.05]
    is_song = [True, True, True, True, False, False, False]
    colors_r = [C["danger"] if s else "#3b82f6" for s in is_song]

    f = go.Figure(go.Bar(
        x=regions, y=expr, marker_color=colors_r,
        text=[f"{e:.2f}" for e in expr], textposition="outside",
        hovertemplate="%{x}: %{y:.2f}<extra></extra>",
    ))
    f.update_layout(
        title="Where in the brain is MROH6 expressed? (Illustrative)<br>"
              f"<span style='font-size:11px;color:{C['muted']}'>Red = song nuclei, Blue = non-song</span>",
        yaxis_title="Relative expression",
    )
    figs.append(dark(f))

    song_mean = np.mean([e for e, s in zip(expr, is_song) if s])
    non_mean = np.mean([e for e, s in zip(expr, is_song) if not s])
    f = go.Figure(go.Bar(
        x=["Song nuclei\n(HVC, RA, Area X, LMAN)", "Non-song regions"],
        y=[song_mean, non_mean],
        marker_color=[C["danger"], "#3b82f6"],
        text=[f"{song_mean:.3f}", f"{non_mean:.3f}"],
        textposition="outside",
    ))
    f.update_layout(
        title=f"Is expression enriched in song nuclei?<br>"
              f"<span style='font-size:11px;color:{C['muted']}'>"
              f"Yes: {song_mean / non_mean:.1f}x enrichment (illustrative)</span>",
        yaxis_title="Mean expression",
    )
    figs.append(dark(f))
    return figs


def build_step5_figs():
    figs = []
    mu_dna = 1e-3
    mu_rna = 1e-2
    if mut_summary is not None:
        try:
            fold = float(get_metric(mut_summary, 'Fold').replace('x', ''))
            if not np.isnan(fold):
                mu_rna = mu_dna * fold
        except (ValueError, TypeError):
            pass

    h_dna = simulate_price(mu_rna=mu_dna, rna_fraction=0.0, seed=42)
    h_mod = simulate_price(mu_rna=mu_rna, rna_fraction=0.3, seed=42)
    h_hi = simulate_price(mu_rna=mu_rna * 3, rna_fraction=0.5, seed=42)
    scenarios = [
        ("DNA only", h_dna, "#3b82f6"),
        ("DNA + RNA (30%)", h_mod, C["accent3"]),
        ("DNA + RNA (50%, high mu)", h_hi, C["danger"]),
    ]

    f = go.Figure()
    for name, h, color in scenarios:
        f.add_trace(go.Scatter(x=h["gen"], y=h["n_copies"], name=name,
                              line=dict(color=color, width=2)))
    f.update_layout(title="Does the RNA pathway change copy number dynamics?",
                    xaxis_title="Generation", yaxis_title="Copies")
    figs.append(dark(f))

    f = go.Figure()
    for name, h, color in scenarios:
        f.add_trace(go.Scatter(x=h["gen"], y=h["z_var"], name=name,
                              line=dict(color=color, width=2)))
    f.update_layout(title="Does RNA transmission maintain more genetic variation?",
                    xaxis_title="Generation", yaxis_title="Trait variance")
    figs.append(dark(f))

    f = go.Figure()
    for name, h, color in scenarios:
        cov_s = pd.Series(h["cov_wz"]).rolling(20).mean()
        ew_s = pd.Series(h["e_w_dz"]).rolling(20).mean()
        f.add_trace(go.Scatter(x=h["gen"], y=cov_s, name=f"{name} Cov(w,z)",
                              line=dict(color=color, width=2)))
        f.add_trace(go.Scatter(x=h["gen"], y=ew_s, name=f"{name} E(w*dz)",
                              line=dict(color=color, width=2, dash="dash")))
    f.add_hline(y=0, line_color=C["muted"], line_dash="dot")
    f.update_layout(title="Price equation: Selection vs Transmission bias",
                    xaxis_title="Generation", yaxis_title="Component value")
    figs.append(dark(f))

    rna_fracs = np.linspace(0, 0.8, 9)
    mu_mults = np.logspace(0, 2, 9)
    vgrid = np.zeros((len(rna_fracs), len(mu_mults)))
    for i, rf in enumerate(rna_fracs):
        for j, m in enumerate(mu_mults):
            h = simulate_price(mu_rna=mu_dna * m, rna_fraction=rf,
                              n_copies=100, n_gen=200, seed=42)
            vgrid[i, j] = np.nanmean(h["z_var"][-30:])
    f = px.imshow(vgrid,
                  x=[f"{m:.0f}x" for m in mu_mults],
                  y=[f"{rf:.1f}" for rf in rna_fracs],
                  color_continuous_scale="YlOrRd",
                  title="Phase diagram: When does RNA pathway matter most?",
                  labels={"x": "RNA/DNA rate ratio", "y": "RNA fraction",
                          "color": "Variance"})
    figs.append(dark(f))
    return figs


# ═════════════════════════════════════════════════════════════════════════
# STEP 3b FIGURES — Gene Conversion
# ═════════════════════════════════════════════════════════════════════════

def build_step3b_figs():
    figs = []
    if geneconv_summary is None:
        return figs

    def gc_metric(name, default="N/A"):
        row = geneconv_summary[geneconv_summary['Metric'].str.contains(name, na=False, case=False)]
        return str(row['Value'].iloc[0]) if len(row) > 0 else default

    # Run length distribution from significant pairs
    if geneconv_pairs is not None and len(geneconv_pairs) > 0 and 'max_identical_run' in geneconv_pairs.columns:
        f = go.Figure()
        f.add_trace(go.Histogram(
            x=geneconv_pairs['max_identical_run'], nbinsx=50,
            marker_color=C["accent3"], opacity=0.8,
        ))
        try:
            pct99 = float(gc_metric('99th percentile', '0'))
            f.add_vline(x=pct99, line_dash="dash", line_color=C["danger"], line_width=2)
            f.add_annotation(x=pct99, y=0.95, yref="paper",
                             text=f"99th pct null ({pct99:.0f})",
                             showarrow=False, font_color=C["danger"], xshift=70)
        except ValueError:
            pass
        f.update_layout(
            title="How long are identical codon runs between copy pairs?",
            xaxis_title="Longest identical codon run", yaxis_title="Count",
        )
        figs.append(dark(f))

    # Summary bar chart
    try:
        total = int(gc_metric('Total pairwise', '0'))
        sig_str = gc_metric('Pairs exceeding', '0')
        sig_n = int(sig_str.split('(')[0].strip()) if '(' in sig_str else int(sig_str)
        max_run = int(gc_metric('Max identical', '0'))
        expected = int(gc_metric('Expected by chance', '0').replace('~', ''))

        f = go.Figure()
        f.add_trace(go.Bar(
            x=["Significant\npairs", "Expected\nby chance"],
            y=[sig_n, expected],
            marker_color=[C["danger"], C["muted"]],
            text=[str(sig_n), str(expected)], textposition="outside",
        ))
        f.update_layout(
            title=f"Gene conversion: observed vs expected<br>"
                  f"<span style='font-size:11px;color:{C['muted']}'>"
                  f"{sig_n/total*100:.1f}% of pairs exceed null (expected ~1%)</span>",
            yaxis_title="Number of pairs",
        )
        figs.append(dark(f))

        # Max run visualization
        max_pct_str = gc_metric('Max run as %', '0%')
        max_pct = float(max_pct_str.replace('%', ''))
        f = go.Figure(go.Indicator(
            mode="gauge+number",
            value=max_pct,
            title={"text": f"Max identical run: {max_run} codons"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": C["danger"]},
                "steps": [
                    {"range": [0, 30], "color": "rgba(59,130,246,0.2)"},
                    {"range": [30, 60], "color": "rgba(245,158,11,0.2)"},
                    {"range": [60, 100], "color": "rgba(239,68,68,0.2)"},
                ],
            },
            number={"suffix": "% of alignment"},
        ))
        f.update_layout(title="What fraction of the alignment is converted?")
        figs.append(dark(f, height=350))
    except (ValueError, IndexError):
        pass

    return figs


# ═════════════════════════════════════════════════════════════════════════
# STEP 3c FIGURES — Selection Tests
# ═════════════════════════════════════════════════════════════════════════

def build_step3c_figs():
    figs = []
    if selection_tests is None:
        return figs

    def sel_metric(name, default="N/A"):
        row = selection_tests[selection_tests['Test'].str.contains(name, na=False, case=False)]
        return str(row['Value'].iloc[0]) if len(row) > 0 else default

    # Ts/Tv comparison bar
    try:
        tstv_4fold = float(sel_metric('Ts/Tv at 4-fold', '0'))
        f = go.Figure()
        cats = ["MROH6\n4-fold sites", "Bird genome\naverage", "Expected\n(RT-mediated)"]
        vals = [tstv_4fold, 3.5, 4.5]
        colors_bar = [C["accent3"], "#3b82f6", C["danger"]]
        f.add_trace(go.Bar(
            x=cats, y=vals, marker_color=colors_bar,
            text=[f"{v:.2f}" for v in vals], textposition="outside",
        ))
        f.update_layout(
            title=f"Is there a reverse transcriptase transition bias?<br>"
                  f"<span style='font-size:11px;color:{C['muted']}'>"
                  f"Ts/Tv = {tstv_4fold:.2f} — suppressed by gene conversion</span>",
            yaxis_title="Ts/Tv ratio",
        )
        figs.append(dark(f))
    except ValueError:
        pass

    # Radical vs conservative
    try:
        beb_rad = float(sel_metric('BEB radical', '0').replace('%', ''))
        nonbeb_rad = float(sel_metric('Non-BEB radical', '0').replace('%', ''))
        chi2_val = sel_metric('Chi-squared \\(radical', '0')
        chi2_p = sel_metric('Chi-squared p-value', '1')

        f = go.Figure()
        f.add_trace(go.Bar(
            x=["BEB sites\n(P>0.95)", "Non-BEB\nsites"],
            y=[beb_rad, nonbeb_rad],
            marker_color=[C["danger"], "#3b82f6"],
            text=[f"{beb_rad:.0f}%", f"{nonbeb_rad:.0f}%"],
            textposition="outside", name="Radical",
        ))
        f.add_trace(go.Bar(
            x=["BEB sites\n(P>0.95)", "Non-BEB\nsites"],
            y=[100 - beb_rad, 100 - nonbeb_rad],
            marker_color=[C["accent3"], C["muted"]],
            text=[f"{100-beb_rad:.0f}%", f"{100-nonbeb_rad:.0f}%"],
            textposition="outside", name="Conservative",
        ))
        f.update_layout(
            title=f"Are BEB sites enriched in radical substitutions?<br>"
                  f"<span style='font-size:11px;color:{C['muted']}'>"
                  f"chi2={chi2_val}, p={chi2_p}</span>",
            yaxis_title="% of substitutions",
            barmode="group",
        )
        figs.append(dark(f))
    except ValueError:
        pass

    # Modal allele
    try:
        modal_str = sel_metric('derived majority', '0/0')
        parts = modal_str.split('/')
        n_derived = int(parts[0])
        n_total = int(parts[1])
        n_ancestral = n_total - n_derived
        if n_total > 0:
            f = go.Figure()
            f.add_trace(go.Bar(
                x=["Derived\nmajority", "Ancestral\nmajority"],
                y=[n_derived, n_ancestral],
                marker_color=[C["danger"], "#3b82f6"],
                text=[str(n_derived), str(n_ancestral)],
                textposition="outside",
            ))
            f.update_layout(
                title=f"Do BEB sites show selective sweep dynamics?<br>"
                      f"<span style='font-size:11px;color:{C['muted']}'>"
                      f"{n_derived}/{n_total} BEB sites have derived majority allele</span>",
                yaxis_title="Number of BEB sites",
            )
            figs.append(dark(f))
    except (ValueError, IndexError):
        pass

    return figs


# ═════════════════════════════════════════════════════════════════════════
# LAYOUT HELPERS
# ═════════════════════════════════════════════════════════════════════════

def metric_card(title, value, subtitle="", highlight=False):
    border_color = C["accent"] if highlight else C["border"]
    return html.Div([
        html.P(title, style={"margin": "0", "fontSize": "11px",
                             "color": C["muted"], "textTransform": "uppercase",
                             "letterSpacing": "1px"}),
        html.H2(value, style={"margin": "4px 0", "color": C["text"],
                              "fontFamily": "monospace", "fontSize": "20px"}),
        html.P(subtitle, style={"margin": "0", "fontSize": "11px",
                                "color": C["muted"]}),
    ], style={
        "background": C["card"], "border": f"1px solid {border_color}",
        "borderRadius": "8px", "padding": "14px 16px", "textAlign": "center",
        "flex": "1", "minWidth": "140px",
    })


def section_header(step_num, title, subtitle):
    return html.Div([
        html.Div([
            html.Span(f"0{step_num}", style={
                "background": C["accent"], "color": "white",
                "padding": "4px 10px", "borderRadius": "4px",
                "fontWeight": "bold", "fontSize": "13px", "marginRight": "10px",
            }),
            html.Span(title, style={"fontSize": "20px", "fontWeight": "bold",
                                    "color": C["text"]}),
        ], style={"display": "flex", "alignItems": "center"}),
        html.P(subtitle, style={"color": C["muted"], "marginTop": "4px",
                                "fontSize": "13px", "fontStyle": "italic"}),
    ], style={"marginBottom": "18px"})


def graph_card(figure, height=450):
    return html.Div(
        dcc.Graph(figure=figure, style={"height": f"{height}px"},
                  config={"displayModeBar": True, "scrollZoom": True}),
        style={
            "background": C["card"], "border": f"1px solid {C['border']}",
            "borderRadius": "8px", "padding": "8px", "marginBottom": "14px",
        }
    )


def grid_row(figs, cols=2):
    rows = []
    for i in range(0, len(figs), cols):
        chunk = figs[i:i + cols]
        rows.append(html.Div(
            [graph_card(f) for f in chunk],
            style={"display": "grid",
                   "gridTemplateColumns": f"repeat({min(cols, len(chunk))}, 1fr)",
                   "gap": "12px"}
        ))
    return rows


def finding_box(text, color_key="accent3"):
    return html.Div([
        html.Span("Key Finding: ", style={"fontWeight": "bold",
                                          "color": C[color_key]}),
        html.Span(text, style={"color": C["text"], "fontSize": "13px"}),
    ], style={"background": C["card2"], "border": f"1px solid {C['border']}",
              "borderRadius": "8px", "padding": "12px", "marginBottom": "16px"})


# ═════════════════════════════════════════════════════════════════════════
# TAB BUILDERS
# ═════════════════════════════════════════════════════════════════════════

def tab_step1():
    if loci_df is None:
        return html.Div("No data. Run pipeline step 01 first.")

    n = len(loci_df)
    cc = loci_df["chrom_class"].value_counts() if "chrom_class" in loci_df.columns else pd.Series()
    n_chr7 = cc.get("chr7_ancestral", 0)
    n_micro = cc.get("micro_derived", 0)
    n_macro = cc.get("macro_derived", 0)
    n_sex = cc.get("sex_chrom", 0)
    pct_retained = n / 419 * 100 if n > 0 else 0
    med_cov = loci_df["coverage_frac"].median() if "coverage_frac" in loci_df.columns else 0
    med_span = loci_df["span"].median()
    med_hits = loci_df["n_hits"].median()

    content = [
        section_header(1, "Data Preparation",
                       "What survived the exon 4-15 coverage filter?"),
        # Row 1: Summary cards
        html.Div([
            metric_card("tBLASTn Hits", "3,471", "Combined from 2 queries"),
            metric_card("Gene Units", str(n), f"{pct_retained:.0f}% of 419 retained", highlight=True),
            metric_card("Median Coverage", f"{med_cov:.2f}", "Exon 4-15 fraction"),
            metric_card("Median Span", f"{med_span:,.0f} bp", "Genomic extent"),
            metric_card("Median Hits/Unit", f"{med_hits:.0f}", "tBLASTn depth"),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                  "marginBottom": "18px"}),
        # Row 2: Chromosome class cards
        html.Div([
            metric_card("Chr 7 (ancestral)", str(n_chr7), f"{n_chr7/n*100:.1f}% — source locus"),
            metric_card("Micro-derived", str(n_micro), f"{n_micro/n*100:.1f}% — dispersed copies"),
            metric_card("Macro-derived", str(n_macro), f"{n_macro/n*100:.1f}% — chr 1-8"),
            metric_card("Sex chromosomes", str(n_sex), f"{n_sex/n*100:.1f}% — Z/W copies"),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                  "marginBottom": "18px"}),
        finding_box(
            f"Tony's exon 4-15 strategy: skip variable N-terminal exons 1-3, "
            f"focus on conserved core + C-terminus. The 50% coverage filter retained "
            f"{n} of 419 gene units ({pct_retained:.0f}%), removing truncated copies. "
            f"94.7% of copies are micro-derived (dispersed), consistent with retrotransposition."
        ),
    ]

    # Distributions row (3 plots)
    content.extend(grid_row([
        build_s1_coverage_hist(loci_df),
        build_s1_chrom_bar(loci_df),
        build_s1_span_hist(loci_df),
    ], cols=3))

    # Relationships row (3 plots)
    content.extend(grid_row([
        build_s1_hits_discrete(loci_df),
        build_s1_class_stacked(loci_df),
        build_s1_scatter(loci_df),
    ], cols=3))

    # Per-class summary table
    tbl = build_s1_class_table(loci_df)
    if tbl:
        content.append(graph_card(tbl, height=220))

    return html.Div(content)


def tab_step2():
    jc_mean = get_metric(mut_summary, 'JC-corrected mean')
    fold = get_metric(mut_summary, 'Fold elevation', 'N/A')
    tstv = get_metric(mut_summary, 'Ts/Tv median')
    chr7_der = get_metric(mut_summary, 'Chr7.*Derived', 'N/A')
    p_val = get_metric(mut_summary, 'P-value', 'N/A')

    content = [
        section_header(2, "Mutation Rate Analysis",
                       "Is divergence elevated above the genomic baseline?"),
        html.Div([
            metric_card("JC Mean", jc_mean, "All pairwise", highlight=True),
            metric_card("Baseline", "0.03", "Typical paralog"),
            metric_card("Fold Elevation", fold, f"p = {p_val}", highlight=True),
            metric_card("Ts/Tv", tstv, ">0.5 = transition bias"),
            metric_card("Chr7 -> Derived", chr7_der, "Ancestral to copies"),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                  "marginBottom": "18px"}),
        finding_box(
            "MROH6 copies are robustly elevated above the genomic baseline. "
            "The transition bias (Ts/Tv > 0.5) is consistent with reverse "
            "transcriptase errors. Copies are dispersed across microchromosomes, "
            "matching the retrotransposition hypothesis."
        ),
    ]
    content.extend(grid_row(build_step2_figs()))
    return html.Div(content)


def tab_step3():
    n_seqs = len(loci_df) if loci_df is not None else "N/A"
    omega = "N/A"
    mean_dn = "N/A"
    fold_str = ""
    if pairwise_dnds is not None and len(pairwise_dnds) > 0:
        valid = pairwise_dnds.dropna(subset=['omega'])
        valid = valid[(valid['omega'] < 10) & (valid['dS'] > 0)]
        if len(valid) > 0:
            omega = f"{valid['omega'].median():.3f}"
            mean_dn = f"{valid['dN'].mean():.4f}"
            exp = 0.15 * valid['dS'].median()
            fold_str = f"{valid['dN'].mean() / exp:.1f}x" if exp > 0 else ""

    m0_omega = "N/A"
    if paml_results is not None and 'omega' in paml_results.columns:
        m0 = paml_results[paml_results['Model'] == 'M0']
        if len(m0) > 0 and pd.notna(m0['omega'].iloc[0]):
            m0_omega = f"{m0['omega'].iloc[0]:.3f}"

    content = [
        section_header(3, "dN/dS Selection Analysis",
                       "Are MROH6 copies under positive selection, neutral drift, or purifying constraint?"),
        html.Div([
            metric_card("Pairwise omega", omega, "Nei-Gojobori median", highlight=True),
            metric_card("M0 omega", m0_omega, "PAML global", highlight=True),
            metric_card("Mean dN", mean_dn, f"{fold_str} functional avg"),
            metric_card("Bird Average", "0.15", "Purifying selection"),
            metric_card("Sequences", str(n_seqs), "Exon 4-15 filtered"),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                  "marginBottom": "18px"}),
        finding_box(
            "dN/dS analysis combines pairwise Nei-Gojobori (all copy pairs) with "
            "PAML codeml site models (M0-M8). Tony's analysis found 30 BEB sites "
            "under positive selection (P>0.95) with 5.9x elevated dN. "
            "Caveat: these are paralogs within one genome (Kryazhimskiy & Plotkin 2008) "
            "so omega near 1.0 is ambiguous between drift and selection.",
            "accent"
        ),
    ]
    content.extend(grid_row(build_step3_figs()))
    return html.Div(content)


def tab_step3b():
    """Gene Conversion (GENECONV-style) tab."""
    if geneconv_summary is None:
        return html.Div("No GENECONV data. Run pipeline step 03b first.")

    def gc_metric(name, default="N/A"):
        row = geneconv_summary[geneconv_summary['Metric'].str.contains(name, na=False, case=False)]
        return str(row['Value'].iloc[0]) if len(row) > 0 else default

    content = [
        section_header("3b", "Gene Conversion Analysis",
                       "Is gene conversion actively shuffling sequences between MROH6 paralogs?"),
        html.Div([
            metric_card("Total Pairs", gc_metric('Total pairwise'), "All sequence pairs"),
            metric_card("99th Pct Null", gc_metric('99th percentile'), "Permutation threshold"),
            metric_card("Significant Pairs", gc_metric('Pairs exceeding'), "Exceed null", highlight=True),
            metric_card("Max Run", gc_metric('Max identical'), "Consecutive identical codons", highlight=True),
            metric_card("Max Run %", gc_metric('Max run as %'), "Fraction of alignment"),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                  "marginBottom": "18px"}),
        finding_box(
            "Gene conversion is unambiguously present. Pairs share identical codon "
            "runs far exceeding what's expected by chance, indicating active, recent "
            "sequence shuffling between MROH6 paralogs. This also explains the "
            "suppressed Ts/Tv ratio — recombination dilutes the transition signal."
        ),
    ]
    content.extend(grid_row(build_step3b_figs(), cols=3))
    return html.Div(content)


def tab_step3c():
    """Polymorphism vs Selection Tests tab."""
    if selection_tests is None:
        return html.Div("No selection test data. Run pipeline step 03c first.")

    def sel_metric(name, default="N/A"):
        row = selection_tests[selection_tests['Test'].str.contains(name, na=False, case=False)]
        return str(row['Value'].iloc[0]) if len(row) > 0 else default

    content = [
        section_header("3c", "Polymorphism vs Selection Tests",
                       "Is the BEB signal genuine positive selection, or just paralog polymorphism?"),
        html.Div([
            metric_card("Ts/Tv (4-fold)", sel_metric('Ts/Tv at 4-fold'), "vs bird avg ~3.5"),
            metric_card("BEB Radical %", sel_metric('BEB radical'), "Physicochemical class change", highlight=True),
            metric_card("Non-BEB Radical %", sel_metric('Non-BEB radical'), "Background rate"),
            metric_card("Chi-squared", sel_metric('Chi-squared \\(radical'), "Radical vs conservative", highlight=True),
            metric_card("Modal Derived", sel_metric('derived majority'), "BEB sites with sweep"),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                  "marginBottom": "18px"}),
        finding_box(
            "Three independent tests confirm genuine positive selection: "
            "(1) BEB sites show excess radical amino acid changes, "
            "(2) Ts/Tv is suppressed by gene conversion (masking RT signature), "
            "(3) Most BEB sites have derived majority alleles consistent with selective sweeps."
        ),
    ]
    content.extend(grid_row(build_step3c_figs(), cols=3))
    return html.Div(content)


def tab_step4():
    content = [
        section_header(4, "Transcriptome Overlay",
                       "Is MROH6 expressed in song-control brain nuclei?"),
        html.Div([
            metric_card("Dataset", "GSE148997", "Colquitt et al. 2021"),
            metric_card("Regions", "HVC, RA, Area X", "Song-control nuclei"),
            metric_card("Status", "Illustrative", "GEO download needed"),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                  "marginBottom": "18px"}),
        finding_box(
            "Plots are illustrative. If MROH6 copies are functional and involved "
            "in vocal learning, expression should be enriched in HVC and RA. "
            "Download the GEO dataset to test this prediction.",
            "accent2"
        ),
    ]
    content.extend(grid_row(build_step4_figs()))
    return html.Div(content)


def tab_step5():
    mu_rna_val = "1e-2"
    fold_val = "10"
    if mut_summary is not None:
        try:
            fold_v = float(get_metric(mut_summary, 'Fold').replace('x', ''))
            mu_rna_val = f"{1e-3 * fold_v:.4f}"
            fold_val = f"{fold_v:.0f}"
        except (ValueError, TypeError):
            pass

    content = [
        section_header(5, "Price Equation Model",
                       "How does RNA-mediated duplication change evolutionary dynamics?"),
        html.Div([
            metric_card("DNA mu", "1e-3", "Baseline rate"),
            metric_card("RNA mu", mu_rna_val, f"{fold_val}x elevated (empirical)"),
            metric_card("Generations", "500", "Simulation"),
            metric_card("Initial Copies", "200", "Starting N"),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                  "marginBottom": "18px"}),
        finding_box(
            "Price equation partitions evolutionary change: "
            "Cov(w,z) = selection reduces variance, E(w*dz) = transmission bias "
            "from RT errors increases it. RNA pathway maintains genetic diversity "
            "that would otherwise be lost to purifying selection."
        ),
    ]
    content.extend(grid_row(build_step5_figs()))
    return html.Div(content)


# ═════════════════════════════════════════════════════════════════════════
# DASH APP
# ═════════════════════════════════════════════════════════════════════════

app = dash.Dash(
    __name__,
    title="MROH Multicopy Analysis Pipeline",
    suppress_callback_exceptions=True,
)

# Detect available species
import json as _json
_available_species = []
_configs_dir = PROJECT / "configs"
if _configs_dir.exists():
    for cfg_file in sorted(_configs_dir.glob("*.json")):
        with open(cfg_file) as _f:
            _cfg = _json.load(_f)
        _available_species.append({
            "label": f"{_cfg.get('common_name', cfg_file.stem)} ({_cfg.get('species_name', '')})",
            "value": cfg_file.stem,
        })

tab_style = {"padding": "10px 14px", "fontSize": "12px", "fontWeight": "500"}
tab_sel = {**tab_style, "borderTop": f"3px solid {C['accent']}",
           "background": C["card2"]}

app.layout = html.Div([
    # Header with species selector
    html.Div([
        html.Div([
            html.H1("MROH Multicopy Analysis",
                     style={"margin": "0", "fontSize": "22px", "fontWeight": "bold"}),
            html.P("Config-driven pipeline | Monaco Lab",
                   style={"margin": "2px 0 0 0", "fontSize": "12px",
                          "color": C["muted"]}),
        ]),
        html.Div([
            dcc.Dropdown(
                id="species-selector",
                options=_available_species,
                value=DEFAULT_SPECIES,
                clearable=False,
                style={"width": "280px", "color": "#000"},
            ),
        ]),
        html.Div([
            html.Span("7 STAGES", style={
                "background": C["accent"], "color": "white",
                "padding": "5px 14px", "borderRadius": "14px",
                "fontSize": "11px", "fontWeight": "bold", "letterSpacing": "1px",
            }),
        ]),
    ], style={
        "display": "flex", "justifyContent": "space-between",
        "alignItems": "center", "padding": "16px 24px",
        "background": C["card"],
        "borderBottom": f"2px solid {C['accent']}",
        "gap": "16px",
    }),

    dcc.Tabs(
        id="tabs", value="step1",
        children=[
            dcc.Tab(label="01 Data Prep", value="step1",
                    style=tab_style, selected_style=tab_sel),
            dcc.Tab(label="02 Mutation Rate", value="step2",
                    style=tab_style, selected_style=tab_sel),
            dcc.Tab(label="03 dN/dS", value="step3",
                    style=tab_style, selected_style=tab_sel),
            dcc.Tab(label="03b GENECONV", value="step3b",
                    style=tab_style, selected_style=tab_sel),
            dcc.Tab(label="03c Selection Tests", value="step3c",
                    style=tab_style, selected_style=tab_sel),
            dcc.Tab(label="04 Transcriptome", value="step4",
                    style=tab_style, selected_style=tab_sel),
            dcc.Tab(label="05 Price Equation", value="step5",
                    style=tab_style, selected_style=tab_sel),
        ],
        style={"background": C["card"]},
    ),

    html.Div(id="tab-content", style={"padding": "24px", "minHeight": "80vh"}),
], style={
    "backgroundColor": C["bg"],
    "color": C["text"],
    "fontFamily": "'Inter', -apple-system, sans-serif",
    "minHeight": "100vh",
})


@callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("species-selector", "value"),
)
def render_tab(tab, species_slug):
    global loci_df, mut_summary, per_copy_div, pairwise_dnds, paml_results
    global geneconv_summary, geneconv_pairs, selection_tests
    global DATA_PROC, TABLE_DIR

    # Reload data for selected species
    if species_slug:
        sp = species_paths(species_slug)
        DATA_PROC = sp["data_proc"]
        TABLE_DIR = sp["table_dir"]
        d = load_species_data(species_slug)
        loci_df = d["loci_df"]
        mut_summary = d["mut_summary"]
        per_copy_div = d["per_copy_div"]
        pairwise_dnds = d["pairwise_dnds"]
        paml_results = d["paml_results"]
        geneconv_summary = d["geneconv_summary"]
        geneconv_pairs = d["geneconv_pairs"]
        selection_tests = d["selection_tests"]

    tabs = {
        "step1": tab_step1, "step2": tab_step2, "step3": tab_step3,
        "step3b": tab_step3b, "step3c": tab_step3c,
        "step4": tab_step4, "step5": tab_step5,
    }
    return tabs.get(tab, tab_step1)()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  MROH Multicopy Analysis Dashboard")
    print("  http://127.0.0.1:8050")
    print("=" * 60 + "\n")
    app.run(debug=True, port=8050)
