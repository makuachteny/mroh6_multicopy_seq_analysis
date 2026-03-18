#!/usr/bin/env python3
"""
MROH Multicopy Analysis Pipeline — Master Runner (Generalized)
===============================================================
QUESTION: Did MROH gene family members (e.g., MROH6) expand via an
RNA-intermediate (retrotransposition) mechanism, and are the resulting
copies under natural selection?

REASONING:
  MROH (Maestro Heat-like Repeat-containing) genes are found in hundreds
  to thousands of copies across avian genomes. The origin of this massive
  expansion is unknown. Two hypotheses exist:

    (A) DNA-mediated tandem duplication — copies stay on the same chromosome,
        mutation rates match the genomic background, and copies are clustered.

    (B) RNA-mediated retrotransposition — an mRNA intermediate is reverse-
        transcribed and inserted elsewhere in the genome, producing dispersed
        copies with ELEVATED mutation rates (reverse transcriptase is error-
        prone) and potentially a transition bias (RT signature).

  This pipeline tests hypothesis (B) by measuring:
    Step 01: Where are the copies? (dispersed = retrotransposition)
    Step 02: Are mutation rates elevated? (>3x baseline = RNA signature)
    Step 03: Are copies under selection? (dN/dS analysis via PAML)
    Step 04: Are copies expressed in relevant tissues? (transcriptome)
    Step 05: Can an RNA pathway sustain copy number? (Price equation model)

PIPELINE ARCHITECTURE:
  Each step is a standalone Python script that reads from the previous step's
  output files. A JSON config file in configs/ defines all species-specific
  parameters (input files, chromosome classification, analysis thresholds).

  The pipeline supports a DECISION TREE in Step 01:
    1. First try the full gene (all exons) with a coverage filter
    2. If too few gene units pass, automatically fall back to a subset
       of exons (e.g., exons 4-15) that are more conserved

Usage:
  python run_pipeline.py --species melospiza_georgiana          # Run all steps
  python run_pipeline.py --species zebra_finch --step 1         # Run only step 1
  python run_pipeline.py --species melospiza_georgiana --no-dash # Skip dashboard
  python run_pipeline.py --list                                  # List available species

Input requirements per species (see configs/*.json):
  1. Exon 13 BLASTn file — 23-mer probe hits (FASTA or tabular format)
     anchors each gene copy in the genome
  2. tBLASTn alignment file(s) — protein query vs genome (FASTA format)
     provides the nucleotide sequences of each copy
  3. Reference protein FASTA — the query protein used for tBLASTn
"""
import subprocess
import sys
import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
NOTEBOOKS = PROJECT / 'notebooks'
CONFIGS = PROJECT / 'configs'

STEPS = [
    ('01',  NOTEBOOKS / '01_data_prep.py',            'Data Preparation'),
    ('02',  NOTEBOOKS / '02_mutation_rate.py',         'Mutation Rate Analysis'),
    ('03',  NOTEBOOKS / '03_dnds_analysis.py',         'dN/dS Selection Analysis'),
    ('03b', NOTEBOOKS / '03b_geneconv_analysis.py',    'Gene Conversion (GENECONV)'),
    ('03c', NOTEBOOKS / '03c_selection_tests.py',      'Polymorphism vs Selection Tests'),
    ('04',  NOTEBOOKS / '04_transcriptome_overlay.py', 'Transcriptome Overlay'),
    ('05',  NOTEBOOKS / '05_price_equation.py',        'Price Equation Model'),
]


def list_species():
    """List all available species configurations."""
    configs = sorted(CONFIGS.glob('*.json'))
    if not configs:
        print("  No species configurations found in configs/")
        return
    print("\n  Available species configurations:")
    print("  " + "-" * 50)
    for cfg_path in configs:
        with open(cfg_path) as f:
            cfg = json.load(f)
        slug = cfg_path.stem
        name = cfg.get('species_name', 'Unknown')
        common = cfg.get('common_name', '')
        gene = cfg.get('gene_name', 'MROH6')
        genome = cfg.get('genome_assembly', '?')
        print(f"  {slug:<30s} {name} ({common})")
        print(f"  {'':30s} Gene: {gene}, Genome: {genome}")
    print()


def run_step(step_num, script_path, description, species):
    print(f"\n{'#' * 70}")
    print(f"  RUNNING STEP {step_num}: {description}")
    print(f"  Species: {species}")
    print(f"  Script: {script_path.name}")
    print(f"{'#' * 70}\n")

    result = subprocess.run(
        [sys.executable, str(script_path), '--species', species],
        cwd=str(PROJECT)
    )
    if result.returncode != 0:
        print(f"\n  *** STEP {step_num} FAILED (exit code {result.returncode}) ***")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description='MROH Multicopy Analysis Pipeline Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --species melospiza_georgiana
  python run_pipeline.py --species zebra_finch --step 1
  python run_pipeline.py --list
        """
    )
    parser.add_argument('--species', help='Species config slug (e.g., melospiza_georgiana)')
    parser.add_argument('--step', type=int, help='Run only this step (1-5)')
    parser.add_argument('--no-dash', action='store_true', help='Skip dashboard')
    parser.add_argument('--dash-only', action='store_true', help='Only launch dashboard')
    parser.add_argument('--list', action='store_true', help='List available species')
    args = parser.parse_args()

    if args.list:
        list_species()
        return

    if not args.species:
        parser.error("--species is required (use --list to see available species)")

    # Validate species config exists
    cfg_path = CONFIGS / f"{args.species}.json"
    if not cfg_path.exists():
        print(f"  ERROR: Species config not found: {cfg_path}")
        print(f"  Available configs:")
        list_species()
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = json.load(f)

    gene = cfg.get('gene_name', 'MROH6')
    species_name = cfg.get('species_name', args.species)
    common_name = cfg.get('common_name', '')

    print("=" * 70)
    print(f"  {gene} MULTICOPY ANALYSIS PIPELINE")
    print(f"  {species_name} ({common_name})")
    print(f"  Genome: {cfg.get('genome_assembly', '?')} ({cfg.get('genome_accession', '?')})")
    print("=" * 70)

    if not args.dash_only:
        if args.step:
            step_idx = args.step - 1
            if 0 <= step_idx < len(STEPS):
                num, path, desc = STEPS[step_idx]
                success = run_step(num, path, desc, args.species)
                if not success:
                    sys.exit(1)
            else:
                print(f"Invalid step: {args.step}. Choose 1-{len(STEPS)}")
                sys.exit(1)
        else:
            for num, path, desc in STEPS:
                success = run_step(num, path, desc, args.species)
                if not success:
                    print(f"\nPipeline halted at step {num}.")
                    print("Fix the error and re-run, or skip with --step N")
                    sys.exit(1)

        species_slug = cfg_path.stem
        print("\n" + "=" * 70)
        print("  ALL PIPELINE STEPS COMPLETE")
        print(f"  Figures: {PROJECT / 'results' / species_slug / 'figures'}")
        print(f"  Tables:  {PROJECT / 'results' / species_slug / 'tables'}")
        print("=" * 70)

    if not args.no_dash:
        app_path = PROJECT / 'app.py'
        if app_path.exists():
            print("\n  Launching dashboard...")
            print("  Open http://127.0.0.1:8050 in your browser\n")
            subprocess.run([sys.executable, str(app_path)])
        else:
            print("  Dashboard (app.py) not found — skipping")


if __name__ == '__main__':
    main()
