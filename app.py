#!/usr/bin/env python3
"""
MROH Multicopy Analysis Dashboard
====================================
Interactive Dash app with narrative-driven visualization, interactive
filtering, and per-class biology summaries.

Usage:
  python app.py
  -> Opens at http://127.0.0.1:8050
"""
import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path

# -- Paths ------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent
DEFAULT_SPECIES = "melospiza_georgiana"

def species_paths(species_slug):
    return {
        "data_proc": PROJECT / "data" / "processed" / species_slug,
        "table_dir": PROJECT / "results" / species_slug / "tables",
        "fig_dir": PROJECT / "results" / species_slug / "figures",
    }

DATA_PROC = species_paths(DEFAULT_SPECIES)["data_proc"]
RESULTS = PROJECT / "results" / DEFAULT_SPECIES
TABLE_DIR = species_paths(DEFAULT_SPECIES)["table_dir"]

# -- Chromosome class colors ------------------------------------------------
CLASS_COLORS = {
    "chr7_ancestral": "#e05c5c",
    "macro_derived":  "#5b8dd9",
    "micro_derived":  "#e8a838",
    "sex_chrom":      "#9b7dd4",
}
CLASS_LABELS = {
    "chr7_ancestral": "Ancestral",
    "macro_derived":  "Macro-derived",
    "micro_derived":  "Micro-derived",
    "sex_chrom":      "Sex chromosomes",
}

def _class_label_for_species(cls, cfg_data):
    """Generate species-aware class label."""
    if "ancestral" in cls and cfg_data:
        anc = cfg_data.get("ancestral_chromosome", "7")
        return f"Chr {anc} (ancestral)"
    return CLASS_LABELS.get(cls, cls)

# -- Design system ----------------------------------------------------------
#
# Palette: warm dark inspired by the logo's terracotta + plum tones.
# Typography:
#   - Headings: IBM Plex Mono  (engineer precision)
#   - Body:     IBM Plex Sans  (clarity at small sizes)
#   - Data:     JetBrains Mono (tabular numbers)
#
FONT_HEADING = "'IBM Plex Mono', 'SF Mono', 'Fira Code', monospace"
FONT_BODY    = "'IBM Plex Sans', 'Inter', -apple-system, system-ui, sans-serif"
FONT_DATA    = "'JetBrains Mono', 'IBM Plex Mono', 'SF Mono', monospace"

C = {
    "bg":       "#101216",
    "surface":  "#171a21",
    "card":     "#1c2029",
    "card2":    "#212636",
    "border":   "#2a2f3d",
    "border2":  "#353b4d",
    "text":     "#d8dae0",
    "text2":    "#b0b4c0",
    "muted":    "#7c8190",
    "dim":      "#555a68",
    "accent":   "#c87941",      # warm terracotta from logo
    "accent2":  "#5ba0d9",      # calm blue
    "accent3":  "#e8a838",      # amber
    "success":  "#5bbd8a",
    "danger":   "#d95b5b",
    "plum":     "#7b5ea7",      # from logo's dark purple
}

# Google Fonts link for the head
GOOGLE_FONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=IBM+Plex+Mono:wght@400;500;600;700"
    "&family=IBM+Plex+Sans:wght@300;400;500;600"
    "&family=JetBrains+Mono:wght@400;500;600"
    "&display=swap"
)


# ===================================================================
# DATA LOADING
# ===================================================================

def safe_load(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def load_species_data(species_slug):
    sp = species_paths(species_slug)
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

# Load species config JSON for display
import json as _json
def load_species_cfg(species_slug):
    cfg_path = PROJECT / "configs" / f"{species_slug}.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return _json.load(f)
    return {}

_data = load_species_data(DEFAULT_SPECIES)
_species_cfg = load_species_cfg(DEFAULT_SPECIES)
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


# ===================================================================
# PLOTLY FIGURE STYLING
# ===================================================================

def styled(fig, height=None):
    """Apply consistent theme to a Plotly figure."""
    layout = dict(
        template="plotly_dark",
        paper_bgcolor=C["card"],
        plot_bgcolor=C["card"],
        font=dict(family=FONT_BODY, color=C["text2"], size=13),
        title_font=dict(family=FONT_HEADING, size=16, color=C["text"]),
        margin=dict(l=60, r=28, t=72, b=52),
        legend=dict(
            font=dict(family=FONT_BODY, size=12, color=C["muted"]),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        xaxis=dict(
            gridcolor=C["border"], zerolinecolor=C["border"],
            title_font=dict(family=FONT_BODY, size=13, color=C["muted"]),
            tickfont=dict(family=FONT_DATA, size=12, color=C["dim"]),
        ),
        yaxis=dict(
            gridcolor=C["border"], zerolinecolor=C["border"],
            title_font=dict(family=FONT_BODY, size=13, color=C["muted"]),
            tickfont=dict(family=FONT_DATA, size=12, color=C["dim"]),
        ),
    )
    if height:
        layout["height"] = height
    fig.update_layout(**layout)
    return fig


# ===================================================================
# STEP 1 FIGURES
# ===================================================================

def build_s1_chrom_bar(df):
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
            name=_class_label_for_species(cls, _species_cfg),
            marker_color=color, marker_line_width=0,
            hovertemplate="Chr %{x}<br>%{y} gene units<extra></extra>",
        ))
    f.update_layout(
        title="Gene units per chromosome",
        xaxis_title="Chromosome",
        yaxis_title="Gene units",
        barmode="stack",
        xaxis=dict(categoryorder="total descending"),
    )
    return styled(f)


def build_s1_coverage_hist(df):
    f = go.Figure()
    below = df[df["coverage_frac"] < 0.50]
    above = df[df["coverage_frac"] >= 0.50]
    f.add_trace(go.Histogram(
        x=below["coverage_frac"], nbinsx=20,
        marker_color="rgba(124,129,144,0.35)", marker_line_width=0,
        name=f"Below 50% ({len(below)})",
    ))
    f.add_trace(go.Histogram(
        x=above["coverage_frac"], nbinsx=30,
        marker_color=C["accent"], marker_line_width=0,
        name=f"Above 50% ({len(above)})",
    ))
    med = df["coverage_frac"].median()
    f.add_vline(x=0.50, line_dash="dash", line_color=C["danger"], line_width=1.5,
                annotation_text="50% cutoff", annotation_font_color=C["danger"],
                annotation_font_size=10, annotation_position="top left")
    f.add_vline(x=med, line_dash="dot", line_color=C["accent2"], line_width=1,
                annotation_text=f"Median {med:.2f}", annotation_font_color=C["accent2"],
                annotation_font_size=10, annotation_position="top right")
    f.update_layout(
        title="Coverage filter distribution",
        xaxis_title="Coverage fraction",
        yaxis_title="Count",
        barmode="overlay",
    )
    return styled(f)


def build_s1_span_hist(df):
    med = df["span"].median()
    max_span = max(df["span"].max() + 1, 20001)
    bins = [0, 2000, 5000, 10000, 20000, max_span]
    labels = ["0-2kb", "2-5kb", "5-10kb", "10-20kb", ">20kb"]
    df_copy = df.copy()
    df_copy["span_bin"] = pd.cut(df_copy["span"], bins=bins, labels=labels, right=False)
    bin_counts = df_copy["span_bin"].value_counts().reindex(labels).fillna(0)
    f = go.Figure(go.Bar(
        x=labels, y=bin_counts.values,
        marker_color=[C["accent2"] if l != labels[0] else C["dim"] for l in labels],
        marker_line_width=0,
        text=bin_counts.values.astype(int), textposition="outside",
        textfont=dict(family=FONT_DATA, size=11, color=C["muted"]),
    ))
    f.update_layout(
        title=f"Genomic span distribution (median {med:,.0f} bp)",
        xaxis_title="Genomic span", yaxis_title="Count",
    )
    return styled(f)


def build_s1_hits_discrete(df):
    df_copy = df.copy()
    def bin_hits(n):
        if n <= 2: return str(int(n))
        elif n <= 4: return "3-4"
        elif n <= 8: return "5-8"
        else: return ">8"
    df_copy["hit_bin"] = df_copy["n_hits"].apply(bin_hits)
    order = ["1", "2", "3-4", "5-8", ">8"]
    counts = df_copy["hit_bin"].value_counts().reindex(order).fillna(0)
    mode_bin = counts.idxmax()
    colors = [C["accent3"] if b == mode_bin else C["dim"] for b in order]
    f = go.Figure(go.Bar(
        x=order, y=counts.values, marker_color=colors, marker_line_width=0,
        text=counts.values.astype(int), textposition="outside",
        textfont=dict(family=FONT_DATA, size=11, color=C["muted"]),
    ))
    f.update_layout(
        title="BLAST hits per gene unit",
        xaxis_title="tBLASTn hits", yaxis_title="Count",
    )
    return styled(f)


def build_s1_class_stacked(df):
    cc = df["chrom_class"].value_counts()
    total = cc.sum()
    f = go.Figure()
    for cls in ["micro_derived", "macro_derived", "chr7_ancestral", "sex_chrom"]:
        n = cc.get(cls, 0)
        pct = n / total * 100
        f.add_trace(go.Bar(
            x=["Gene Units"], y=[n],
            name=f"{_class_label_for_species(cls, _species_cfg)}: {n} ({pct:.1f}%)",
            marker_color=CLASS_COLORS.get(cls, "gray"), marker_line_width=0,
            text=[f"{n}"], textposition="inside",
            textfont=dict(family=FONT_DATA, size=12, color="white"),
        ))
    f.update_layout(
        title="Chromosome class breakdown",
        barmode="stack", yaxis_title="Gene units",
        showlegend=True, legend=dict(orientation="h", y=-0.2, font_size=10),
    )
    return styled(f)


def build_s1_scatter(df):
    f = go.Figure()
    for cls, color in CLASS_COLORS.items():
        sub = df[df["chrom_class"] == cls]
        if len(sub) == 0:
            continue
        f.add_trace(go.Scatter(
            x=sub["n_hits"], y=sub["total_seq_len"], mode="markers",
            marker=dict(color=color, size=5, opacity=0.5, line=dict(width=0)),
            name=_class_label_for_species(cls, _species_cfg),
        ))
    f.update_layout(
        title="Hits vs reconstructed sequence length",
        xaxis_title="tBLASTn hits", yaxis_title="Sequence length (bp)",
    )
    return styled(f)


def build_s1_class_table(df):
    rows = []
    for cls in ["chr7_ancestral", "macro_derived", "micro_derived", "sex_chrom"]:
        sub = df[df["chrom_class"] == cls]
        if len(sub) == 0:
            continue
        rows.append({
            "Class": _class_label_for_species(cls, _species_cfg),
            "N": len(sub),
            "Coverage": f"{sub['coverage_frac'].median():.2f}" if 'coverage_frac' in sub.columns else "-",
            "Span": f"{sub['span'].median():,.0f}",
            "Hits": f"{sub['n_hits'].median():.0f}",
            "Seq Len": f"{sub['total_seq_len'].median():,.0f}",
        })
    if not rows:
        return None
    tdf = pd.DataFrame(rows)
    f = go.Figure(go.Table(
        header=dict(
            values=list(tdf.columns),
            fill_color=C["card2"], line_color=C["border"],
            font=dict(family=FONT_HEADING, color=C["text2"], size=11),
            align="center", height=32,
        ),
        cells=dict(
            values=[tdf[c] for c in tdf.columns],
            fill_color=C["card"], line_color=C["border"],
            font=dict(family=FONT_DATA, color=C["text"], size=12),
            align="center", height=30,
        ),
    ))
    f.update_layout(margin=dict(l=16, r=16, t=16, b=8), height=280)
    return styled(f, height=280)


# ===================================================================
# STEP 2 FIGURES
# ===================================================================

def build_step2_figs():
    figs = []
    if per_copy_div is None or len(per_copy_div) == 0:
        return figs
    pcol = "chrom_class" if "chrom_class" in per_copy_div.columns else None
    baseline = float(_species_cfg.get("genomic_baseline", 0.03)) if _species_cfg else 0.03

    # Divergence histogram
    f = go.Figure()
    if pcol:
        for cls, color in CLASS_COLORS.items():
            sub = per_copy_div[per_copy_div[pcol] == cls]
            if len(sub) == 0:
                continue
            f.add_trace(go.Histogram(
                x=sub["mean_jc_div"], nbinsx=30,
                name=_class_label_for_species(cls, _species_cfg),
                marker_color=color, marker_line_width=0, opacity=0.7,
            ))
    else:
        f.add_trace(go.Histogram(x=per_copy_div["mean_jc_div"], nbinsx=40,
                                 marker_color=C["accent"], marker_line_width=0))

    mean_div = per_copy_div["mean_jc_div"].mean()
    fold = mean_div / baseline if baseline > 0 else 0
    f.add_vline(x=baseline, line_dash="dash", line_color=C["danger"], line_width=1.5)
    f.add_annotation(x=baseline, y=1, yref="paper", text=f"Baseline ({baseline})",
                     showarrow=False, font=dict(color=C["danger"], size=10), xshift=55)
    f.add_vline(x=mean_div, line_dash="dot", line_color=C["accent3"], line_width=1.5)
    f.add_annotation(x=mean_div, y=0.9, yref="paper",
                     text=f"Observed ({mean_div:.3f}, {fold:.1f}x)",
                     showarrow=False, font=dict(color=C["accent3"], size=10), xshift=75)
    f.update_layout(
        title=f"JC-corrected divergence distribution",
        xaxis_title="Mean JC divergence", yaxis_title="Count", barmode="overlay",
    )
    figs.append(styled(f))

    # Box plot by class
    if pcol:
        f = go.Figure()
        for cls, color in CLASS_COLORS.items():
            sub = per_copy_div[per_copy_div[pcol] == cls]
            if len(sub) == 0:
                continue
            f.add_trace(go.Box(
                y=sub["mean_jc_div"], name=_class_label_for_species(cls, _species_cfg),
                marker_color=color, line_color=color, boxpoints="outliers",
            ))
        f.add_hline(y=baseline, line_dash="dash", line_color=C["danger"], line_width=1)
        f.update_layout(title="Divergence by chromosome class", yaxis_title="Mean JC divergence")
        figs.append(styled(f))

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
                x=names, y=vals, marker_color=colors, marker_line_width=0,
                text=[f"{v:.4f}" for v in vals], textposition="outside",
                textfont=dict(family=FONT_DATA, size=12, color=C["text2"]),
            ))
            f.update_layout(title="Key mutation rate metrics", yaxis_title="Value")
            figs.append(styled(f))

    return figs


# ===================================================================
# STEP 3 FIGURES
# ===================================================================

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

    # Omega distribution
    f = go.Figure()
    omega_plot = valid[valid['omega'] < 5]
    f.add_trace(go.Histogram(
        x=omega_plot["omega"], nbinsx=60,
        marker_color=C["accent3"], marker_line_width=0, opacity=0.85,
    ))
    f.add_vrect(x0=0, x1=0.3, fillcolor="rgba(91,160,217,0.06)", line_width=0,
                annotation_text="Purifying", annotation_position="top left",
                annotation_font=dict(color=C["accent2"], size=10))
    f.add_vrect(x0=0.3, x1=1.0, fillcolor="rgba(232,168,56,0.04)", line_width=0,
                annotation_text="Relaxed", annotation_position="top left",
                annotation_font=dict(color=C["accent3"], size=10))
    f.add_vrect(x0=1.0, x1=5.0, fillcolor="rgba(217,91,91,0.04)", line_width=0,
                annotation_text="Positive", annotation_position="top left",
                annotation_font=dict(color=C["danger"], size=10))
    f.add_vline(x=1.0, line_dash="dash", line_color=C["danger"], line_width=1.5)
    f.add_vline(x=med_omega, line_dash="dash", line_color=C["accent2"], line_width=1.5)
    f.add_annotation(x=med_omega, y=0.95, yref="paper",
                     text=f"Median = {med_omega:.3f}",
                     font=dict(color=C["accent2"], size=10),
                     showarrow=False, xshift=55)
    f.update_layout(
        title=f"Selection regime (dN/dS distribution)",
        xaxis_title="omega (dN/dS)", yaxis_title="Count",
    )
    figs.append(styled(f))

    # dN vs dS
    sample = valid.sample(min(5000, len(valid)), random_state=42)
    f = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
                      row_heights=[0.15, 0.85], column_widths=[0.85, 0.15],
                      horizontal_spacing=0.02, vertical_spacing=0.02)
    f.add_trace(go.Scatter(
        x=sample["dS"], y=sample["dN"], mode="markers",
        marker=dict(color=C["accent2"], size=3, opacity=0.15),
        showlegend=False,
    ), row=2, col=1)
    max_v = max(sample['dS'].quantile(0.95), sample['dN'].quantile(0.95))
    f.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v], mode="lines",
                          line=dict(color=C["danger"], dash="dash", width=1),
                          name="omega = 1"), row=2, col=1)
    f.add_trace(go.Histogram(x=sample["dS"], nbinsx=40,
                            marker_color=C["success"], marker_line_width=0,
                            opacity=0.4, showlegend=False), row=1, col=1)
    f.add_trace(go.Histogram(y=sample["dN"], nbinsx=40,
                            marker_color=C["accent"], marker_line_width=0,
                            opacity=0.4, showlegend=False), row=2, col=2)
    f.update_layout(
        title="dN vs dS (synonymous vs nonsynonymous rates)",
        xaxis3=dict(title="dS"), yaxis3=dict(title="dN"),
    )
    figs.append(styled(f, height=480))

    # Genome comparison
    exp_func = 0.15 * med_dS
    exp_dup = 0.40 * med_dS
    fold_func = mean_dN / exp_func if exp_func > 0 else 0
    fold_dup = mean_dN / exp_dup if exp_dup > 0 else 0
    f = go.Figure(go.Bar(
        x=["Observed", "Functional avg", "Duplicated avg"],
        y=[mean_dN, exp_func, exp_dup],
        marker_color=[C["accent3"], C["accent2"], C["success"]],
        marker_line_width=0,
        text=[f"{v:.4f}" for v in [mean_dN, exp_func, exp_dup]],
        textposition="outside",
        textfont=dict(family=FONT_DATA, size=11, color=C["text2"]),
    ))
    f.update_layout(
        title=f"dN comparison ({fold_func:.1f}x functional, {fold_dup:.1f}x duplicated)",
        yaxis_title="Mean pairwise dN",
    )
    figs.append(styled(f))

    # PAML
    if paml_results is not None and 'lnL' in paml_results.columns:
        vp = paml_results.dropna(subset=['lnL'])
        if len(vp) > 0:
            model_colors = {"M0": C["dim"], "M1a": C["accent2"], "M2a": C["danger"],
                            "M7": C["accent2"], "M8": C["danger"]}
            f = go.Figure(go.Bar(
                x=vp["Model"], y=vp["lnL"],
                marker_color=[model_colors.get(m, C["accent"]) for m in vp["Model"]],
                marker_line_width=0,
                text=[f"{v:.1f}" for v in vp["lnL"]], textposition="outside",
                textfont=dict(family=FONT_DATA, size=11, color=C["text2"]),
            ))
            subtitle = ""
            m1a = vp[vp["Model"] == "M1a"]["lnL"]
            m2a = vp[vp["Model"] == "M2a"]["lnL"]
            if len(m1a) > 0 and len(m2a) > 0:
                delta = 2 * (m2a.iloc[0] - m1a.iloc[0])
                from scipy.stats import chi2
                p = chi2.sf(delta, 2)
                sig = "significant" if p < 0.05 else "n.s."
                subtitle = f" (M1a vs M2a: p={p:.1e}, {sig})"
            f.update_layout(
                title=f"PAML site model likelihoods{subtitle}",
                yaxis_title="Log-likelihood",
            )
            figs.append(styled(f))

    return figs


# ===================================================================
# STEP 3b, 3c, 4, 5 FIGURES
# ===================================================================

def build_step3b_figs():
    figs = []
    if geneconv_summary is None:
        return figs

    def gc_metric(name, default="N/A"):
        row = geneconv_summary[geneconv_summary['Metric'].str.contains(name, na=False, case=False)]
        return str(row['Value'].iloc[0]) if len(row) > 0 else default

    if geneconv_pairs is not None and len(geneconv_pairs) > 0 and 'max_identical_run' in geneconv_pairs.columns:
        f = go.Figure(go.Histogram(
            x=geneconv_pairs['max_identical_run'], nbinsx=50,
            marker_color=C["accent3"], marker_line_width=0, opacity=0.85,
        ))
        try:
            pct99 = float(gc_metric('99th percentile', '0'))
            f.add_vline(x=pct99, line_dash="dash", line_color=C["danger"], line_width=1.5)
            f.add_annotation(x=pct99, y=0.95, yref="paper",
                             text=f"99th pct null ({pct99:.0f})",
                             showarrow=False, font=dict(color=C["danger"], size=10), xshift=65)
        except ValueError:
            pass
        f.update_layout(
            title="Identical codon run lengths between pairs",
            xaxis_title="Longest identical codon run", yaxis_title="Count",
        )
        figs.append(styled(f))

    try:
        total = int(gc_metric('Total pairwise', '0'))
        sig_str = gc_metric('Pairs exceeding', '0')
        sig_n = int(sig_str.split('(')[0].strip()) if '(' in sig_str else int(sig_str)
        expected = int(gc_metric('Expected by chance', '0').replace('~', ''))
        max_run = int(gc_metric('Max identical', '0'))

        f = go.Figure(go.Bar(
            x=["Significant", "Expected by chance"],
            y=[sig_n, expected],
            marker_color=[C["danger"], C["dim"]], marker_line_width=0,
            text=[f"{sig_n:,}", f"{expected:,}"], textposition="outside",
            textfont=dict(family=FONT_DATA, size=12, color=C["text2"]),
        ))
        pct = sig_n / total * 100 if total > 0 else 0
        f.update_layout(
            title=f"Gene conversion pairs ({pct:.1f}% exceed null)",
            yaxis_title="Number of pairs",
        )
        figs.append(styled(f))

        max_pct_str = gc_metric('Max run as %', '0%')
        max_pct = float(max_pct_str.replace('%', ''))
        f = go.Figure(go.Indicator(
            mode="gauge+number", value=max_pct,
            title={"text": f"Max run: {max_run} codons",
                   "font": {"family": FONT_HEADING, "size": 13, "color": C["text2"]}},
            gauge=dict(
                axis=dict(range=[0, 100], tickfont=dict(size=10, color=C["dim"])),
                bar=dict(color=C["accent"]),
                bgcolor=C["card2"],
                steps=[
                    {"range": [0, 30], "color": "rgba(91,160,217,0.1)"},
                    {"range": [30, 60], "color": "rgba(232,168,56,0.1)"},
                    {"range": [60, 100], "color": "rgba(217,91,91,0.1)"},
                ],
            ),
            number=dict(suffix="% of alignment",
                       font=dict(family=FONT_DATA, size=20, color=C["text"])),
        ))
        figs.append(styled(f, height=320))
    except (ValueError, IndexError):
        pass

    return figs


def build_step3c_figs():
    figs = []
    if selection_tests is None:
        return figs

    def sel_metric(name, default="N/A"):
        row = selection_tests[selection_tests['Test'].str.contains(name, na=False, case=False)]
        return str(row['Value'].iloc[0]) if len(row) > 0 else default

    try:
        tstv_4fold = float(sel_metric('Ts/Tv at 4-fold', '0'))
        f = go.Figure(go.Bar(
            x=["Observed\n(4-fold sites)", "Bird genome\naverage", "Expected\n(RT-mediated)"],
            y=[tstv_4fold, 3.5, 4.5],
            marker_color=[C["accent3"], C["accent2"], C["danger"]], marker_line_width=0,
            text=[f"{v:.2f}" for v in [tstv_4fold, 3.5, 4.5]], textposition="outside",
            textfont=dict(family=FONT_DATA, size=12, color=C["text2"]),
        ))
        f.update_layout(title=f"Transition / transversion ratio", yaxis_title="Ts/Tv")
        figs.append(styled(f))
    except ValueError:
        pass

    try:
        beb_rad = float(sel_metric('BEB radical', '0').replace('%', ''))
        nonbeb_rad = float(sel_metric('Non-BEB radical', '0').replace('%', ''))
        f = go.Figure()
        f.add_trace(go.Bar(
            x=["BEB sites", "Non-BEB"], y=[beb_rad, nonbeb_rad],
            marker_color=[C["danger"], C["accent2"]], marker_line_width=0,
            text=[f"{beb_rad:.0f}%", f"{nonbeb_rad:.0f}%"], textposition="outside",
            textfont=dict(family=FONT_DATA, size=12), name="Radical",
        ))
        f.add_trace(go.Bar(
            x=["BEB sites", "Non-BEB"], y=[100-beb_rad, 100-nonbeb_rad],
            marker_color=[C["accent3"], C["dim"]], marker_line_width=0,
            text=[f"{100-beb_rad:.0f}%", f"{100-nonbeb_rad:.0f}%"], textposition="outside",
            textfont=dict(family=FONT_DATA, size=12), name="Conservative",
        ))
        f.update_layout(title="Radical vs conservative substitutions",
                       yaxis_title="% of substitutions", barmode="group")
        figs.append(styled(f))
    except ValueError:
        pass

    try:
        modal_str = sel_metric('derived majority', '0/0')
        parts = modal_str.split('/')
        n_derived = int(parts[0])
        n_total = int(parts[1])
        if n_total > 0:
            f = go.Figure(go.Bar(
                x=["Derived majority", "Ancestral majority"],
                y=[n_derived, n_total - n_derived],
                marker_color=[C["danger"], C["accent2"]], marker_line_width=0,
                text=[str(n_derived), str(n_total - n_derived)], textposition="outside",
                textfont=dict(family=FONT_DATA, size=12, color=C["text2"]),
            ))
            f.update_layout(title=f"Modal alleles at BEB sites ({n_derived}/{n_total} derived)",
                           yaxis_title="BEB sites")
            figs.append(styled(f))
    except (ValueError, IndexError):
        pass

    return figs


def build_step4_figs():
    figs = []
    regions = ['HVC', 'RA', 'Area X', 'LMAN', 'Cortex', 'Striatum', 'Cerebellum']
    expr = [0.82, 0.35, 0.51, 0.28, 0.12, 0.18, 0.05]
    is_song = [True, True, True, True, False, False, False]
    colors_r = [C["accent"] if s else C["accent2"] for s in is_song]

    f = go.Figure(go.Bar(
        x=regions, y=expr, marker_color=colors_r, marker_line_width=0,
        text=[f"{e:.2f}" for e in expr], textposition="outside",
        textfont=dict(family=FONT_DATA, size=11, color=C["text2"]),
    ))
    f.update_layout(
        title="Brain region expression (illustrative)",
        yaxis_title="Relative expression",
    )
    figs.append(styled(f))

    song_mean = np.mean([e for e, s in zip(expr, is_song) if s])
    non_mean = np.mean([e for e, s in zip(expr, is_song) if not s])
    f = go.Figure(go.Bar(
        x=["Song nuclei", "Non-song regions"],
        y=[song_mean, non_mean],
        marker_color=[C["accent"], C["accent2"]], marker_line_width=0,
        text=[f"{song_mean:.3f}", f"{non_mean:.3f}"], textposition="outside",
        textfont=dict(family=FONT_DATA, size=12, color=C["text2"]),
    ))
    enrichment = song_mean / non_mean if non_mean > 0 else 0
    f.update_layout(
        title=f"Song nuclei enrichment ({enrichment:.1f}x, illustrative)",
        yaxis_title="Mean expression",
    )
    figs.append(styled(f))
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
        ("DNA only", h_dna, C["accent2"]),
        ("DNA + RNA (30%)", h_mod, C["accent3"]),
        ("DNA + RNA (50%, high mu)", h_hi, C["danger"]),
    ]

    f = go.Figure()
    for name, h, color in scenarios:
        f.add_trace(go.Scatter(x=h["gen"], y=h["n_copies"], name=name,
                              line=dict(color=color, width=2)))
    f.update_layout(title="Copy number dynamics",
                    xaxis_title="Generation", yaxis_title="Copies")
    figs.append(styled(f))

    f = go.Figure()
    for name, h, color in scenarios:
        f.add_trace(go.Scatter(x=h["gen"], y=h["z_var"], name=name,
                              line=dict(color=color, width=2)))
    f.update_layout(title="Trait variance over time",
                    xaxis_title="Generation", yaxis_title="Trait variance")
    figs.append(styled(f))

    f = go.Figure()
    for name, h, color in scenarios:
        cov_s = pd.Series(h["cov_wz"]).rolling(20).mean()
        ew_s = pd.Series(h["e_w_dz"]).rolling(20).mean()
        f.add_trace(go.Scatter(x=h["gen"], y=cov_s, name=f"{name} — Cov(w,z)",
                              line=dict(color=color, width=2)))
        f.add_trace(go.Scatter(x=h["gen"], y=ew_s, name=f"{name} — E(w*dz)",
                              line=dict(color=color, width=2, dash="dash")))
    f.add_hline(y=0, line_color=C["dim"], line_dash="dot", line_width=1)
    f.update_layout(title="Price equation decomposition",
                    xaxis_title="Generation", yaxis_title="Component value")
    figs.append(styled(f))

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
                  labels={"x": "RNA/DNA rate ratio", "y": "RNA fraction",
                          "color": "Variance"})
    f.update_layout(title="Phase diagram: RNA pathway parameter space")
    figs.append(styled(f))
    return figs


# ===================================================================
# LAYOUT COMPONENTS
# ===================================================================

def metric_card(title, value, subtitle="", highlight=False):
    accent_bar = C["accent"] if highlight else "transparent"
    return html.Div([
        html.Div(style={
            "width": "4px", "height": "100%", "position": "absolute",
            "left": "0", "top": "0", "borderRadius": "10px 0 0 10px",
            "background": accent_bar,
        }),
        html.P(title, style={
            "margin": "0 0 10px 0", "fontSize": "11px", "fontWeight": "600",
            "fontFamily": FONT_HEADING, "color": C["muted"],
            "textTransform": "uppercase", "letterSpacing": "1.2px",
        }),
        html.P(value, style={
            "margin": "0 0 8px 0", "fontSize": "30px", "fontWeight": "600",
            "fontFamily": FONT_DATA, "color": C["text"],
            "lineHeight": "1.1",
        }),
        html.P(subtitle, style={
            "margin": "0", "fontSize": "12px", "fontFamily": FONT_BODY,
            "color": C["dim"], "lineHeight": "1.4",
        }),
    ], style={
        "position": "relative", "background": C["card"],
        "border": f"1px solid {C['border']}",
        "borderRadius": "10px", "padding": "22px 24px 20px 24px",
        "flex": "1", "minWidth": "160px",
    })


def section_header(step_num, title, subtitle):
    return html.Div([
        html.Div([
            html.Span(f"{step_num}", style={
                "display": "inline-flex", "alignItems": "center",
                "justifyContent": "center",
                "width": "40px", "height": "40px",
                "background": C["accent"], "color": "white",
                "borderRadius": "8px", "fontWeight": "700",
                "fontSize": "16px", "fontFamily": FONT_HEADING,
                "marginRight": "18px", "flexShrink": "0",
            }),
            html.Span(title, style={
                "fontSize": "28px", "fontWeight": "600",
                "fontFamily": FONT_HEADING, "color": C["text"],
                "letterSpacing": "-0.5px",
            }),
        ], style={"display": "flex", "alignItems": "center"}),
        html.P(subtitle, style={
            "color": C["muted"], "marginTop": "8px", "marginBottom": "0",
            "fontSize": "15px", "fontFamily": FONT_BODY,
            "fontStyle": "italic", "lineHeight": "1.5",
            "paddingLeft": "58px",
        }),
    ], style={"marginBottom": "32px"})


def graph_card(figure, height=520):
    return html.Div(
        dcc.Graph(figure=figure, style={"height": f"{height}px"},
                  config={"displayModeBar": "hover", "scrollZoom": True}),
        style={
            "background": C["card"], "border": f"1px solid {C['border']}",
            "borderRadius": "12px", "padding": "14px 16px 10px 16px",
            "marginBottom": "20px",
        }
    )


def grid_row(figs, cols=2):
    rows = []
    for i in range(0, len(figs), cols):
        chunk = figs[i:i + cols]
        rows.append(html.Div(
            [graph_card(f) for f in chunk],
            style={
                "display": "grid",
                "gridTemplateColumns": f"repeat({min(cols, len(chunk))}, 1fr)",
                "gap": "20px",
            }
        ))
    return rows


def finding_box(text, color_key="accent3"):
    return html.Div([
        html.Div(style={
            "width": "3px", "height": "100%", "position": "absolute",
            "left": "0", "top": "0", "borderRadius": "6px 0 0 6px",
            "background": C[color_key],
        }),
        html.Span("Key finding", style={
            "fontWeight": "600", "fontSize": "11px",
            "fontFamily": FONT_HEADING, "color": C[color_key],
            "textTransform": "uppercase", "letterSpacing": "1px",
            "display": "block", "marginBottom": "6px",
        }),
        html.Span(text, style={
            "color": C["text2"], "fontSize": "14px",
            "fontFamily": FONT_BODY, "lineHeight": "1.6",
        }),
    ], style={
        "position": "relative", "background": C["card2"],
        "border": f"1px solid {C['border']}",
        "borderRadius": "10px", "padding": "18px 24px 18px 24px",
        "marginBottom": "28px",
    })


def no_data_panel(step_name, step_num):
    return html.Div([
        html.Div(f"{step_num}", style={
            "fontSize": "48px", "fontWeight": "700", "fontFamily": FONT_HEADING,
            "color": C["border2"], "marginBottom": "8px",
        }),
        html.P(f"No data for {step_name}", style={
            "fontSize": "16px", "fontFamily": FONT_HEADING, "color": C["muted"],
            "marginBottom": "4px",
        }),
        html.P("Run the pipeline step first, then refresh.", style={
            "fontSize": "13px", "fontFamily": FONT_BODY, "color": C["dim"],
        }),
    ], style={
        "textAlign": "center", "padding": "80px 40px",
        "background": C["card"], "borderRadius": "12px",
        "border": f"1px solid {C['border']}",
    })


# ===================================================================
# TAB BUILDERS
# ===================================================================

def tab_step1():
    if loci_df is None:
        return no_data_panel("Data Preparation", "01")

    n = len(loci_df)
    cc = loci_df["chrom_class"].value_counts() if "chrom_class" in loci_df.columns else pd.Series()
    anc_chr = _species_cfg.get("ancestral_chromosome", "7") if _species_cfg else "7"
    anc_key = [k for k in cc.index if "ancestral" in k]
    n_anc = cc.get(anc_key[0], 0) if anc_key else 0
    n_micro = cc.get("micro_derived", 0)
    n_macro = cc.get("macro_derived", 0)
    n_sex = cc.get("sex_chrom", 0)
    med_cov = loci_df["coverage_frac"].median() if "coverage_frac" in loci_df.columns else 0
    med_span = loci_df["span"].median()
    med_hits = loci_df["n_hits"].median()

    content = [
        section_header("01", "Data Preparation",
                       "Locate every gene copy, extract sequences, and apply coverage filter"),
        html.Div([
            metric_card("Gene Units", str(n), "Passed coverage filter", highlight=True),
            metric_card("Median Coverage", f"{med_cov:.2f}", "Fraction of analysis region"),
            metric_card("Median Span", f"{med_span:,.0f} bp", "Genomic extent"),
            metric_card("Median Hits", f"{med_hits:.0f}", "tBLASTn depth per unit"),
        ], style={"display": "flex", "gap": "18px", "flexWrap": "wrap",
                  "marginBottom": "28px"}),
        html.Div([
            metric_card(f"Chr {anc_chr} (ancestral)", str(n_anc),
                       f"{n_anc/n*100:.1f}% of total" if n > 0 else ""),
            metric_card("Micro-derived", str(n_micro),
                       f"{n_micro/n*100:.1f}% dispersed" if n > 0 else ""),
            metric_card("Macro-derived", str(n_macro),
                       f"{n_macro/n*100:.1f}%" if n > 0 else ""),
            metric_card("Sex chromosomes", str(n_sex),
                       f"{n_sex/n*100:.1f}% Z/W" if n > 0 else ""),
        ], style={"display": "flex", "gap": "18px", "flexWrap": "wrap",
                  "marginBottom": "28px"}),
    ]

    if n_micro > 0 and n > 0:
        dispersal_pct = (n_micro + n_sex) / n * 100
        finding_text = (
            f"{dispersal_pct:.0f}% of gene units are on micro/sex chromosomes, "
            f"dispersed away from the ancestral locus on chr {anc_chr}. "
            f"This dispersal pattern is consistent with retrotransposition."
        )
        content.append(finding_box(finding_text))

    content.extend(grid_row([
        build_s1_chrom_bar(loci_df),
        build_s1_coverage_hist(loci_df),
    ]))
    content.extend(grid_row([
        build_s1_span_hist(loci_df),
        build_s1_hits_discrete(loci_df),
    ]))
    content.extend(grid_row([
        build_s1_class_stacked(loci_df),
        build_s1_scatter(loci_df),
    ]))

    tbl = build_s1_class_table(loci_df)
    if tbl:
        content.append(graph_card(tbl, height=280))

    return html.Div(content)


def tab_step2():
    if mut_summary is None and per_copy_div is None:
        return no_data_panel("Mutation Rate Analysis", "02")

    jc_mean = get_metric(mut_summary, 'JC-corrected mean')
    fold = get_metric(mut_summary, 'Fold elevation', 'N/A')
    tstv = get_metric(mut_summary, 'Ts/Tv median')
    p_val = get_metric(mut_summary, 'P-value', 'N/A')
    baseline = _species_cfg.get("genomic_baseline", 0.03) if _species_cfg else 0.03

    content = [
        section_header("02", "Mutation Rate Analysis",
                       "Is divergence elevated above the genomic baseline? Is there a transition bias?"),
        html.Div([
            metric_card("JC Mean", jc_mean, "All pairwise comparisons", highlight=True),
            metric_card("Baseline", str(baseline), "Expected for paralogs"),
            metric_card("Fold Elevation", fold, f"p = {p_val}", highlight=True),
            metric_card("Ts/Tv", tstv, ">0.5 indicates transition bias"),
        ], style={"display": "flex", "gap": "18px", "flexWrap": "wrap",
                  "marginBottom": "28px"}),
        finding_box(
            "Elevated divergence above the genomic baseline supports an RNA-mediated "
            "duplication mechanism. Reverse transcriptase has a much higher error rate "
            "than DNA polymerase and produces a transition bias (Ts/Tv > 0.5)."
        ),
    ]
    content.extend(grid_row(build_step2_figs()))
    return html.Div(content)


def tab_step3():
    if pairwise_dnds is None or len(pairwise_dnds) == 0:
        return no_data_panel("dN/dS Selection Analysis", "03")

    valid = pairwise_dnds.dropna(subset=['omega'])
    valid = valid[(valid['omega'] < 10) & (valid['dS'] > 0)]
    omega = f"{valid['omega'].median():.3f}" if len(valid) > 0 else "N/A"
    mean_dn = f"{valid['dN'].mean():.4f}" if len(valid) > 0 else "N/A"
    n_seqs = len(loci_df) if loci_df is not None else "N/A"

    m0_omega = "N/A"
    if paml_results is not None and 'omega' in paml_results.columns:
        m0 = paml_results[paml_results['Model'] == 'M0']
        if len(m0) > 0 and pd.notna(m0['omega'].iloc[0]):
            m0_omega = f"{m0['omega'].iloc[0]:.3f}"

    content = [
        section_header("03", "dN/dS Selection Analysis",
                       "Are copies under positive selection, neutral drift, or purifying constraint?"),
        html.Div([
            metric_card("Median omega", omega, "Nei-Gojobori pairwise", highlight=True),
            metric_card("M0 omega", m0_omega, "PAML global estimate"),
            metric_card("Mean dN", mean_dn, "Nonsynonymous rate"),
            metric_card("Sequences", str(n_seqs), "In alignment"),
        ], style={"display": "flex", "gap": "18px", "flexWrap": "wrap",
                  "marginBottom": "28px"}),
        finding_box(
            "dN/dS (omega) near 1.0 for paralogs within a single genome is "
            "ambiguous between neutral drift and positive selection "
            "(Kryazhimskiy & Plotkin 2008). PAML site models can identify "
            "specific codons under positive selection via BEB posterior probabilities.",
            "accent"
        ),
    ]
    content.extend(grid_row(build_step3_figs()))
    return html.Div(content)


def tab_step3b():
    if geneconv_summary is None:
        return no_data_panel("Gene Conversion Analysis", "3b")

    def gc_metric(name, default="N/A"):
        row = geneconv_summary[geneconv_summary['Metric'].str.contains(name, na=False, case=False)]
        return str(row['Value'].iloc[0]) if len(row) > 0 else default

    content = [
        section_header("3b", "Gene Conversion Analysis",
                       "Is recombination actively shuffling sequences between paralogs?"),
        html.Div([
            metric_card("Total Pairs", gc_metric('Total pairwise'), "All comparisons"),
            metric_card("99th Pct Null", gc_metric('99th percentile'), "Permutation threshold"),
            metric_card("Significant", gc_metric('Pairs exceeding'), "Exceed null", highlight=True),
            metric_card("Max Run", gc_metric('Max identical'), "Identical codons", highlight=True),
        ], style={"display": "flex", "gap": "18px", "flexWrap": "wrap",
                  "marginBottom": "28px"}),
        finding_box(
            "Long runs of identical codons between non-adjacent copies indicate "
            "active gene conversion. This recombination dilutes the transition bias "
            "from reverse transcriptase, explaining suppressed Ts/Tv ratios."
        ),
    ]
    content.extend(grid_row(build_step3b_figs()))
    return html.Div(content)


def tab_step3c():
    if selection_tests is None:
        return no_data_panel("Selection Tests", "3c")

    def sel_metric(name, default="N/A"):
        row = selection_tests[selection_tests['Test'].str.contains(name, na=False, case=False)]
        return str(row['Value'].iloc[0]) if len(row) > 0 else default

    content = [
        section_header("3c", "Polymorphism vs Selection Tests",
                       "Is the BEB signal genuine positive selection, or paralog polymorphism?"),
        html.Div([
            metric_card("Ts/Tv (4-fold)", sel_metric('Ts/Tv at 4-fold'), "vs bird avg ~3.5"),
            metric_card("BEB Radical %", sel_metric('BEB radical'), "Class change", highlight=True),
            metric_card("Chi-squared", sel_metric('Chi-squared \\(radical'), "Rad vs cons", highlight=True),
            metric_card("Modal Derived", sel_metric('derived majority'), "Sweep signal"),
        ], style={"display": "flex", "gap": "18px", "flexWrap": "wrap",
                  "marginBottom": "28px"}),
        finding_box(
            "Three independent tests can distinguish genuine positive selection from "
            "paralog polymorphism: radical substitution enrichment at BEB sites, "
            "synonymous diversity comparison, and modal allele analysis for sweep dynamics."
        ),
    ]
    content.extend(grid_row(build_step3c_figs()))
    return html.Div(content)


def tab_step4():
    content = [
        section_header("04", "Transcriptome Overlay",
                       "Are gene copies expressed in song-control brain nuclei?"),
        html.Div([
            metric_card("Status", "Illustrative", "Pending real data"),
            metric_card("Target Regions", "HVC, RA, Area X", "Song nuclei"),
            metric_card("Reference", "Colquitt et al. 2021", "Science 371(6530)"),
        ], style={"display": "flex", "gap": "18px", "flexWrap": "wrap",
                  "marginBottom": "28px"}),
        finding_box(
            "These plots are illustrative. If gene copies are functional and involved "
            "in vocal learning, expression should be enriched in song nuclei (HVC, RA). "
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
        section_header("05", "Price Equation Model",
                       "How does RNA-mediated duplication change evolutionary dynamics?"),
        html.Div([
            metric_card("DNA mu", "1e-3", "Baseline rate"),
            metric_card("RNA mu", mu_rna_val, f"{fold_val}x elevated"),
            metric_card("Generations", "500", "Simulation length"),
            metric_card("Initial N", "200", "Starting copies"),
        ], style={"display": "flex", "gap": "18px", "flexWrap": "wrap",
                  "marginBottom": "28px"}),
        finding_box(
            "The Price equation partitions evolutionary change into selection "
            "(Cov[w,z] reduces variance) and transmission bias "
            "(E[w*dz] increases variance via RT errors). The RNA pathway "
            "maintains genetic diversity that purifying selection would otherwise erode."
        ),
    ]
    content.extend(grid_row(build_step5_figs()))
    return html.Div(content)


# ===================================================================
# DASH APP
# ===================================================================

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

app = dash.Dash(
    __name__,
    title="MROH Multicopy Analysis",
    suppress_callback_exceptions=True,
)

# -- Tab styling ----
tab_base = {
    "padding": "14px 24px",
    "fontSize": "13px",
    "fontWeight": "500",
    "fontFamily": FONT_HEADING,
    "letterSpacing": "0.3px",
    "color": C["muted"],
    "background": C["surface"],
    "border": "none",
    "borderBottom": f"2px solid transparent",
}
tab_selected = {
    **tab_base,
    "color": C["text"],
    "borderBottom": f"2px solid {C['accent']}",
    "background": C["bg"],
}

app.layout = html.Div([
    # Google Fonts
    html.Link(rel="stylesheet", href=GOOGLE_FONTS_URL),

    # Header
    html.Div([
        html.Div([
            html.Div([
                html.H1("MROH Multicopy Analysis", style={
                    "margin": "0", "fontSize": "32px", "fontWeight": "700",
                    "fontFamily": FONT_HEADING, "color": C["text"],
                    "letterSpacing": "-0.8px", "lineHeight": "1.15",
                }),
                html.P("Retrotransposition hypothesis testing pipeline", style={
                    "margin": "6px 0 0 0", "fontSize": "14px",
                    "fontFamily": FONT_BODY, "color": C["muted"],
                    "fontWeight": "400", "letterSpacing": "0.2px",
                }),
            ]),
        ], style={"flex": "1"}),

        html.Div([
            html.Label("Species", style={
                "fontSize": "10px", "fontFamily": FONT_HEADING,
                "textTransform": "uppercase", "letterSpacing": "1.5px",
                "fontWeight": "600",
                "color": C["dim"], "marginBottom": "6px", "display": "block",
            }),
            dcc.Dropdown(
                id="species-selector",
                options=_available_species,
                value=DEFAULT_SPECIES,
                clearable=False,
                style={"width": "340px", "fontFamily": FONT_BODY, "fontSize": "14px"},
            ),
        ]),
    ], style={
        "display": "flex", "justifyContent": "space-between",
        "alignItems": "center", "padding": "28px 48px",
        "background": C["surface"],
        "borderBottom": f"1px solid {C['border']}",
    }),

    # Tabs
    dcc.Tabs(
        id="tabs", value="step1",
        children=[
            dcc.Tab(label="01  Data Prep", value="step1",
                    style=tab_base, selected_style=tab_selected),
            dcc.Tab(label="02  Mutation Rate", value="step2",
                    style=tab_base, selected_style=tab_selected),
            dcc.Tab(label="03  dN/dS", value="step3",
                    style=tab_base, selected_style=tab_selected),
            dcc.Tab(label="3b  Gene Conv.", value="step3b",
                    style=tab_base, selected_style=tab_selected),
            dcc.Tab(label="3c  Selection", value="step3c",
                    style=tab_base, selected_style=tab_selected),
            dcc.Tab(label="04  Transcriptome", value="step4",
                    style=tab_base, selected_style=tab_selected),
            dcc.Tab(label="05  Price Eq.", value="step5",
                    style=tab_base, selected_style=tab_selected),
        ],
        style={"borderBottom": f"1px solid {C['border']}"},
    ),

    # Content
    html.Div(id="tab-content", style={
        "padding": "40px 48px", "minHeight": "80vh",
        "maxWidth": "1600px", "margin": "0 auto",
    }),
], style={
    "backgroundColor": C["bg"],
    "color": C["text"],
    "fontFamily": FONT_BODY,
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
    global DATA_PROC, TABLE_DIR, _species_cfg

    if species_slug:
        sp = species_paths(species_slug)
        DATA_PROC = sp["data_proc"]
        TABLE_DIR = sp["table_dir"]
        _species_cfg = load_species_cfg(species_slug)
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
