"""
Shared utility functions for MROH6 multicopy analysis pipeline.
"""
import re
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"


def parse_blast_fasta(filepath):
    """Parse tBLASTn FASTA output into a DataFrame with genomic coordinates.

    Returns DataFrame with columns:
        accession, chrom, start, end, strand, seq_len, sequence, header
    """
    records = []
    for rec in SeqIO.parse(filepath, "fasta"):
        header = rec.description
        # Parse coordinate from header like NC_133032.1:28834524-28834700 or NC_133032.1:c27748619-27748443
        match = re.match(r'(\S+):(c?)(\d+)-(\d+)\s+(.*)', header)
        if not match:
            continue
        accession = match.group(1) #accession is the part before the colon
        is_complement = match.group(2) == 'c'
        coord1 = int(match.group(3))
        coord2 = int(match.group(4))
        desc = match.group(5)

        # Extract chromosome
        chrom_match = re.search(r'chromosome\s+(\S+)', desc)
        chrom = chrom_match.group(1).rstrip(',') if chrom_match else 'unknown'

        if is_complement:
            start, end = coord2, coord1
            strand = '-'
        else:
            start, end = coord1, coord2
            strand = '+'

        records.append({
            'accession': accession,
            'chrom': chrom,
            'start': start,
            'end': end,
            'strand': strand,
            'seq_len': len(rec.seq),
            'sequence': str(rec.seq),
            'header': header,
        })
    return pd.DataFrame(records)


def merge_overlapping_hits(df, max_gap=500):
    """Merge overlapping or nearby BLAST hits on the same chrom/strand into loci.

    Args:
        df: DataFrame from parse_blast_fasta
        max_gap: maximum gap (bp) to merge nearby hits

    Returns:
        DataFrame of merged loci with concatenated sequences
    """
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


def _make_locus(locus_id, chrom, strand, start, end, hit_rows):
    """Helper to create a locus record from merged hits."""
    # Sort hits by start position and concatenate sequences
    hit_rows_sorted = sorted(hit_rows, key=lambda r: r['start'])
    concat_seq = ''.join(r['sequence'] for r in hit_rows_sorted)
    return {
        'locus_id': locus_id,
        'chrom': chrom,
        'strand': strand,
        'start': start,
        'end': end,
        'span': end - start + 1,
        'n_hits': len(hit_rows),
        'total_seq_len': len(concat_seq),
        'sequence': concat_seq,
    }


def loci_to_fasta(loci_df, outpath):
    """Write loci DataFrame to FASTA file."""
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
    """Jukes-Cantor correction for nucleotide distance.

    Args:
        p: proportion of differing sites (raw divergence)
    Returns:
        JC-corrected distance, or np.nan if p >= 0.75 or p is NaN
    """
    if np.isnan(p) or p >= 0.75:
        return np.nan
    return -0.75 * np.log(1.0 - (4.0 / 3.0) * p)


def count_substitution_types(seq1, seq2):
    """Count transitions and transversions between two aligned sequences.

    Returns dict with keys: transitions, transversions, identical, gaps, total_compared
    """
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
        'transitions': ts,
        'transversions': tv,
        'identical': identical,
        'gaps': gaps,
        'total_compared': total,
        'raw_divergence': (ts + tv) / total if total > 0 else np.nan,
        'ts_tv_ratio': ts / tv if tv > 0 else np.inf,
    }


def pairwise_divergence_matrix(alignment_dict):
    """Compute pairwise raw divergence matrix from dict of {name: sequence}.

    Returns:
        names: list of sequence names
        raw_div: np.array of raw divergences
        jc_div: np.array of JC-corrected divergences
        ts_tv: np.array of Ts/Tv ratios
    """
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
