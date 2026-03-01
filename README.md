# Research
Determining the mutation rate of MROH06 gene

# MROH6 Multicopy Analysis Pipeline

Computational pipeline investigating the evolutionary dynamics of MROH6 gene copies in the zebra finch (*Taeniopygia guttata*) genome. The central hypothesis is that the extraordinary copy number of MROH6 (~3,000 BLAST hits) arose through an RNA-intermediate mechanism — reverse transcription or RNA-dependent RNA polymerase activity — which would leave a distinctive molecular signature: elevated mutation rates (RT/RdRp lacks mismatch repair) and transition-biased substitutions.

---

## Background

MROH6 (Maestro Heat-Like Repeat Family Member 6) is a gene that, in most vertebrates, exists as a single-copy locus adjacent to the lanosterol synthase gene (LSS). In the zebra finch, however, tBLASTn of the MROH6 protein against the genome returns **3,039 alignment hits** distributed across nearly every chromosome. This dramatic amplification raises two questions:

1. **Primary question:** Do the MROH6 copies show a mutation rate that is robustly higher than the known genomic average for zebra finch? If yes, this constitutes indirect evidence for an RNA-intermediate duplication mechanism.

2. **Secondary question (conditional):** If mutation rate is elevated, is there evidence of positive selection (dN/dS > 1) acting on some copies, or are they evolving neutrally / under purifying selection?

## Data

- **Input:** `data/raw/MROH6_tBLASTn_Zebra_finch.txt` — 3,039 tBLASTn hits (FASTA format with genomic coordinates) from searching the MROH6 protein against the zebra finch genome assembly (bTaeGut7.mat).
- **Reference genome:** GCF_003957565.2 (*Taeniopygia guttata*, bTaeGut7.mat)
- **Transcriptome data:** Colquitt et al. 2021 (Science, doi:10.1126/science.abd9704), GEO accession GSE148997 — single-cell RNA-seq from zebra finch song-control nuclei (HVC, RA, Area X).
- **Key reference for dN/dS interpretation:** Kryazhimskiy & Plotkin 2008 (PMC2596312) — dN/dS has limited power intra-specifically; our multicopy-within-species situation is a hybrid case.

## Project Structure

```
Research/
├── README.md
├── environment.yml                          # Conda environment specification
├── data/
│   ├── raw/
│   │   └── MROH6_tBLASTn_Zebra_finch.txt    # 3,039 tBLASTn hits
│   ├── processed/
│   │   ├── mroh6_copies_filtered.fasta      # Merged, length-filtered loci
│   │   ├── mroh6_aligned.fasta              # MAFFT multiple alignment
│   │   ├── mroh6_aligned_trimmed.fasta      # Gap-trimmed alignment
│   │   ├── mroh6_loci_table.csv             # Locus metadata
│   │   └── paml_input/                      # PAML codeml input files
│   │       ├── mroh6_codon.phy              # Codon-aware PHYLIP alignment
│   │       ├── mroh6_nj.tree                # Neighbor-joining guide tree
│   │       ├── M0.ctl ... M8.ctl            # PAML control files
│   │       └── M0_out.txt ... M8_out.txt    # PAML results
│   └── transcriptome/
│       └── colquitt_2021/                   # scRNA-seq count matrices
├── notebooks/
│   ├── 01_data_prep.ipynb                   # Parse, filter, align
│   ├── 02_mutation_rate.ipynb               # Divergence & Ts/Tv analysis
│   ├── 03_dnds_analysis.ipynb               # PAML codeml dN/dS
│   ├── 04_transcriptome_overlay.ipynb       # Song nuclei expression
│   └── 05_price_equation.ipynb              # Evolutionary dynamics model
├── scripts/
│   └── utils.py                             # Shared utility functions
└── results/
    ├── figures/                             # Publication-ready plots
    └── tables/                              # Summary statistics (CSV)
```

## Pipeline Overview

### Step 1: Data Preparation (`01_data_prep.ipynb`)

Parses the raw tBLASTn output and converts 3,039 individual hits into biologically meaningful loci.

**Process:**

1. **Parse** — Extracts chromosome, start/end coordinates, strand, and nucleotide sequence from each FASTA header. Handles both forward (`NC_133032.1:28834524-28834700`) and complement (`NC_133032.1:c27748619-27748443`) orientations.
2. **Filter** — Removes fragments shorter than 150 bp.
3. **Merge** — Groups overlapping or nearby hits (within 500 bp on the same strand) into single-copy loci. This collapses fragmented BLAST alignments of the same genomic copy into one record.
4. **Classify** — Identifies the ancestral MROH6 copy (the locus adjacent to LSS on chromosome 7, ~28.8 Mb) and flags it as the reference for downstream divergence calculations. Classifies remaining loci as full-length (>= 1,000 bp) or partial (300-999 bp).
5. **Align** — Runs MAFFT (`--auto`) on all filtered loci, then trims alignment columns with > 50% gaps.

**Expected output:** ~200-500 analyzable loci from the initial 3,039 fragments. Chromosomal distribution is heavily skewed toward microchromosomes (chr 16, 25, 29, 30, 31, 33-37) with chr 7 carrying the ancestral locus.

### Step 2: Mutation Rate Analysis (`02_mutation_rate.ipynb`) — PRIMARY QUESTION

Tests whether MROH6 copies are more divergent from each other than expected under normal DNA replication.

**Analyses:**

| Analysis | Method | What it tells us |
|---|---|---|
| Pairwise divergence | Raw substitutions/site across all copy pairs | Overall mutation accumulation |
| JC-corrected distance | Jukes-Cantor model (corrects for multiple hits) | More accurate at higher divergence |
| Ancestral divergence | Each copy vs. the chr7 ancestral locus | Radial divergence from source |
| Ts/Tv ratio | Transition vs. transversion counts per pair | RT signature (transitions > transversions) |
| Genomic comparison | MROH6 mean div vs. 0.03 (typical recent paralog) | Fold elevation over baseline |

**Statistical framework:**

- One-sample t-test: H0 = MROH6 divergence equals the baseline for recent DNA-mediated paralogs (0.03 subs/site). One-sided alternative (greater).
- Bootstrap confidence interval: 10,000 resamples of pairwise divergence values, 95% percentile CI.
- Effect size: Cohen's d and fold-difference vs. baseline.

**Decision criteria:**

- Fold-difference > 3x and p < 0.05 → **robustly elevated** → supports RNA intermediate hypothesis → proceed to dN/dS.
- Fold-difference 1.5-3x and p < 0.05 → **moderately elevated** → proceed with caution.
- Fold-difference ~ 1x → **not elevated** → DNA replication alone can explain divergence.

**RT signature:** If the median Ts/Tv ratio is > 1.0 (above the 0.5 expected from random mutation), this independently supports reverse transcriptase involvement, since RT introduces transition-biased errors.

**Key reference values:**

- Zebra finch germline mutation rate: ~1.2-2.3 x 10^-9 subs/site/generation (Smeds et al. 2016)
- Generation time: ~1-2 years
- Expected divergence for paralogs duplicated < 1 Mya: < 0.003
- Typical recent tandem duplicates: 0.01-0.05
- Processed pseudogenes (RT-mediated): 0.05-0.20

### Step 3: dN/dS Analysis (`03_dnds_analysis.ipynb`) — SECONDARY QUESTION

Tests for signatures of natural selection across MROH6 copies using PAML codeml.

**Preparation:**

- Converts the nucleotide alignment to a codon-aware alignment (reading frame guided by the original protein query).
- Masks internal stop codons with NNN.
- Subsamples to 50 sequences if the dataset is large (PAML computational constraint), always retaining the ancestral copy.
- Builds a neighbor-joining tree from identity distances.

**PAML models run:**

| Model | Parameters | Tests for |
|---|---|---|
| M0 | Single global omega | Average selection pressure |
| M1a | Two omega classes (0 < omega < 1, omega = 1) | Nearly-neutral baseline |
| M2a | Three classes (adds omega > 1) | Positive selection (vs M1a) |
| M7 | Beta distribution of omega | Flexible neutral baseline |
| M8 | Beta + omega > 1 class | Positive selection (vs M7) |

**Likelihood ratio tests:**

- M1a vs. M2a: 2 degrees of freedom, chi-squared test
- M7 vs. M8: 2 degrees of freedom, chi-squared test

**Interpretation (with Kryazhimskiy & Plotkin 2008 caveats):**

These copies are paralogs within a single genome — a hybrid between inter-specific divergence (where dN/dS is well-calibrated) and intra-specific polymorphism (where dN/dS is problematic). Specific caveats:

- dN/dS ~ 1 is **ambiguous**: could mean neutrality OR strong selection + drift
- dN/dS < 1 is **ambiguous**: could mean purifying selection OR selective sweep
- dN/dS >> 1 at specific sites is the **strongest signal** for positive selection
- Global dN/dS is compared to the bird gene average (~0.15) to contextualize

### Step 4: Transcriptome Overlay (`04_transcriptome_overlay.ipynb`)

Asks whether MROH6 copies are transcriptionally active in song-control brain regions.

**Data source:** Colquitt et al. 2021, which profiled single-cell transcriptomes across zebra finch HVC, RA, and Area X — three nuclei critical for vocal learning and production.

**Process:**

1. Downloads processed count matrices from GEO (GSE148997).
2. Loads into scanpy AnnData; applies standard preprocessing if raw counts are detected (normalization, log-transform, PCA, UMAP, Leiden clustering).
3. Searches the feature list for MROH6 under multiple name variants (MROH6, LOC identifiers, maestro).
4. If found: generates UMAP expression overlays, violin plots by brain region and cell type (excitatory, inhibitory, glial), and dotplots comparing MROH6 to control genes (GAPDH, ACTB, FOXP2, BDNF) and the neighboring gene LSS.

**Possible outcomes:**

- MROH6 detected and expressed in song nuclei → functional relevance for vocal learning
- MROH6 not in feature list → copies may not be annotated as genes, or multi-mapping reads are filtered out during alignment (a known challenge for multicopy genes)

### Step 5: Price Equation Model (`05_price_equation.ipynb`)

Theoretical framework modeling how RNA-mediated copy number expansion interacts with selection.

**Model:**
The Price equation partitions evolutionary change into two components:

```
w_bar * delta_z_bar = Cov(w, z) + E(w * delta_z)
```

- **Cov(w, z):** Selection on existing DNA copies. Parameterized from empirical dN/dS (Step 3). Negative under stabilizing selection (reduces variance).
- **E(w * delta_z):** Transmission bias from the RNA pool. Parameterized from the empirical mutation rate fold-difference (Step 2). Positive when RNA copies introduce novel variation.

**Simulation:**

- N = 200 copies with trait values under stabilizing selection
- Each generation: DNA-level mutation, duplication (DNA or RNA pathway), fitness-weighted loss
- RNA pathway duplicates carry elevated mutation (mu_RNA >> mu_DNA)
- Three scenarios compared: DNA-only, DNA + moderate RNA (30%), DNA + high RNA (50%)

**Phase diagram:**
Sweeps two parameters:

- RNA fraction (0 to 0.8): what proportion of new copies come via retrotransposition
- RNA/DNA mutation rate ratio (1x to 100x): how much more error-prone is RT

Maps equilibrium trait variance (genetic diversity) and copy number across this parameter space.

**Predictions:**
RNA-mediated expansion is adaptive when:

1. The environment is variable (shifting fitness optimum)
2. Selection is not too strong (copies aren't purged faster than they're generated)
3. RNA mutation rate is 5-50x the DNA rate (consistent with known RT error rates of ~10^-4 to 10^-5 per site)

## Shared Utilities (`scripts/utils.py`)

| Function | Purpose |
|---|---|
| `parse_blast_fasta()` | Parses tBLASTn FASTA output into a DataFrame with genomic coordinates |
| `merge_overlapping_hits()` | Groups nearby hits on the same chrom/strand into loci (500 bp max gap) |
| `loci_to_fasta()` | Writes merged loci to FASTA format |
| `jukes_cantor_distance()` | JC69 correction for nucleotide divergence |
| `count_substitution_types()` | Counts transitions, transversions, identical sites between two sequences |
| `pairwise_divergence_matrix()` | Computes full NxN divergence, JC-distance, and Ts/Tv matrices |

## Getting Started

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate mroh6
```

This installs Python 3.11 with: biopython, scanpy, numpy, scipy, pandas, matplotlib, seaborn, ete3, mafft, blast+, paml, and jupyterlab.

### 2. Verify installation

```bash
python -c "import Bio; import scanpy; print('Python packages OK')"
mafft --version
echo "quit" | codeml    # should print PAML version
```

### 3. Run the pipeline

```bash
cd notebooks
jupyter lab
```

Execute notebooks in order: **01 → 02 → 03 → 04 → 05**. Each notebook depends on outputs from the previous step.

| Notebook | Runtime estimate | Dependencies |
|---|---|---|
| 01_data_prep | ~5-15 min (MAFFT alignment) | Raw BLAST file |
| 02_mutation_rate | ~2-10 min (pairwise matrix) | Trimmed alignment from 01 |
| 03_dnds_analysis | ~10 min to hours (PAML) | Trimmed alignment from 01 |
| 04_transcriptome_overlay | ~5-30 min (download + scanpy) | GEO data (downloaded in notebook) |
| 05_price_equation | ~2-5 min (simulations) | Optionally reads results from 02 |

### 4. Verification checklist

- [ ] Step 1: Filtered loci count is biologically reasonable (~200-500 from 3,039 fragments)
- [ ] Step 2: Divergence values fall in a plausible range (0.01-0.30 subs/site)
- [ ] Step 3: PAML converges for all models (check lnL values are finite)
- [ ] Step 4: GEO data downloads and loads; MROH6 searched in feature list
- [ ] Step 5: Simulations produce non-degenerate dynamics (copy number > 0)

## Outputs

### Figures (`results/figures/`)

| Figure | Description |
|---|---|
| `blast_hit_overview.png` | Chromosomal distribution and length distribution of raw BLAST hits |
| `alignment_gap_distribution.png` | Gap fraction across MAFFT alignment positions |
| `mutation_rate_analysis.png` | Four-panel: (A) pairwise divergence histogram with baseline, (B) divergence from ancestral copy, (C) Ts/Tv ratio distribution, (D) divergence heatmap |
| `mroh6_nj_tree.png` | Neighbor-joining gene tree of MROH6 copies |
| `mroh6_expression_umap.png` | UMAP colored by MROH6 expression and brain region |
| `mroh6_violin_region.png` | Expression violin plots by song nucleus (HVC, RA, Area X) |
| `mroh6_dotplot.png` | Dotplot comparing MROH6 to control genes across brain regions |
| `price_equation_simulations.png` | Four-panel: copy number, trait variance, Price equation components, mean trait over time |
| `price_phase_diagram.png` | Phase diagram: trait variance and copy number across RNA fraction x mutation rate space |

### Tables (`results/tables/`)

| Table | Contents |
|---|---|
| `mutation_rate_summary.csv` | Mean divergence, SD, JC-corrected divergence, Ts/Tv ratio, fold-difference, p-value, bootstrap CI |
| `paml_results.csv` | lnL, omega (dN/dS), kappa (Ts/Tv), number of parameters for each PAML model |

## Interpretive Framework

The logic connecting the analyses:

```
                    Elevated mutation rate?
                   (Step 2: divergence >> baseline)
                          /            \
                        YES             NO
                        /                \
              Ts/Tv > 1?            DNA replication
          (transition bias)         alone explains
                |                   copy divergence
                |
        Supports RT/RdRp
        mechanism
                |
                v
        Selection on copies?
        (Step 3: dN/dS via PAML)
              /    |     \
           <0.3   ~1    >1
            |      |      |
        Purifying  Ambiguous  Positive
        selection  (K&P 2008  selection
                   caveat)    on subset
                |
                v
        Expressed in song nuclei?
        (Step 4: Colquitt scRNA-seq)
              /          \
            YES           NO
            |              |
        Functional      Genomic
        relevance       passengers /
        for vocal       pseudogenes
        learning
                |
                v
        Price equation model
        (Step 5: parameterized from Steps 2-3)
        Predicts regimes where RNA-mediated
        expansion is adaptive
```

## Citation

If using this pipeline, please cite:

- Colquitt et al. 2021. Cellular transcriptomics reveals evolutionary identities of songbird vocal circuits. *Science* 371(6530).
- Kryazhimskiy & Plotkin 2008. The population genetics of dN/dS. *PLoS Genetics* 4(12).
- Smeds et al. 2016. Direct estimate of the rate of germline mutation in a bird. *Genome Research* 26(9).
 (mutation rate)
