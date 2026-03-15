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
from bisect import bisect_left

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
        strand = row.get('strand', '.')
        rec = SeqRecord(
            Seq(row['sequence']),
            id=f"locus_{row['locus_id']}_chr{row['chrom']}_{row['start']}_{row['end']}_{strand}",
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


# ---------------------------------------------------------------------------
# Gene-unit pipeline: exon 13 anchors + combined tBLASTn
# ---------------------------------------------------------------------------

def parse_exon13_blastn(filepath):
    """Parse tabular BLASTn output for exon 13 anchor positions.

    Each hit defines one gene unit anchor. Returns DataFrame with columns:
        saccver, anchor_start, anchor_end, anchor_mid, strand, pident, evalue
    """
    cols = ['qaccver', 'saccver', 'pident', 'length', 'mismatch', 'gapopen',
            'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore']
    df = pd.read_csv(filepath, sep='\t', comment='#', names=cols,
                     skipinitialspace=True)
    df = df.dropna(subset=['saccver']).reset_index(drop=True)
    df['anchor_start'] = df[['sstart', 'send']].min(axis=1).astype(int)
    df['anchor_end'] = df[['sstart', 'send']].max(axis=1).astype(int)
    df['anchor_mid'] = ((df['anchor_start'] + df['anchor_end']) / 2).astype(int)
    df['strand'] = np.where(df['sstart'] <= df['send'], '+', '-')
    return df


def parse_combined_tblastn(filepath1, filepath2):
    """Parse and combine two tBLASTn FASTA files, deduplicating exact coordinate matches.

    Returns combined DataFrame from parse_blast_fasta with a 'source' column.
    """
    df1 = parse_blast_fasta(filepath1)
    df2 = parse_blast_fasta(filepath2)
    df1['source'] = Path(filepath1).stem
    df2['source'] = Path(filepath2).stem
    combined = pd.concat([df1, df2], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=['accession', 'start', 'end', 'strand'], keep='first'
    ).reset_index(drop=True)
    return combined


def build_accession_to_chrom(tblastn_df):
    """Build accession → chromosome mapping from tBLASTn DataFrame."""
    mapping = {}
    for _, row in tblastn_df.drop_duplicates('accession').iterrows():
        mapping[row['accession']] = row['chrom']
    return mapping


def define_gene_units(exon13_df, tblastn_df, accession_to_chrom,
                      max_dist=25000, merge_gap=500):
    """Define gene units using exon 13 anchors, then assign tBLASTn hits.

    For accessions WITH exon 13 anchors: each anchor = 1 gene unit.
    tBLASTn hits are assigned to the nearest anchor (within max_dist).

    For accessions WITHOUT exon 13 anchors (e.g. chr Z, chr 20):
    merge overlapping tBLASTn hits to form gene units.

    Returns DataFrame of gene units with concatenated, non-overlapping sequences.
    """
    gene_units = []
    gu_id = 0
    used_hit_indices = set()

    anchored_accs = set(exon13_df['saccver'].unique())
    all_blast_accs = set(tblastn_df['accession'].unique())

    # --- Anchored accessions ---
    for acc in sorted(anchored_accs):
        anchors = exon13_df[exon13_df['saccver'] == acc].sort_values('anchor_mid')
        anchor_mids = anchors['anchor_mid'].values
        hits = tblastn_df[tblastn_df['accession'] == acc].copy()
        chrom = accession_to_chrom.get(acc, 'unknown')

        if len(hits) == 0:
            # Gene units exist (exon 13 found them) but no tBLASTn coverage
            for _, anchor in anchors.iterrows():
                gene_units.append({
                    'gene_unit_id': gu_id, 'chrom': chrom,
                    'accession': acc, 'anchor_pos': int(anchor['anchor_mid']),
                    'start': int(anchor['anchor_start']),
                    'end': int(anchor['anchor_end']),
                    'span': int(anchor['anchor_end'] - anchor['anchor_start'] + 1),
                    'n_hits': 0, 'total_seq_len': 0, 'sequence': '',
                    'has_exon13': True,
                })
                gu_id += 1
            continue

        # For each hit, find nearest anchor
        hit_mids = ((hits['start'] + hits['end']) / 2).values
        # Distance matrix: hits × anchors
        dists = np.abs(hit_mids[:, None] - anchor_mids[None, :])
        nearest_anchor_idx = np.argmin(dists, axis=1)
        nearest_dist = np.min(dists, axis=1)

        hits = hits.copy()
        hits['_anchor_idx'] = nearest_anchor_idx
        hits['_anchor_dist'] = nearest_dist

        for local_idx, (_, anchor) in enumerate(anchors.iterrows()):
            # Hits assigned to this anchor and within range
            mask = (hits['_anchor_idx'] == local_idx) & (hits['_anchor_dist'] <= max_dist)
            anchor_hits = hits[mask]

            if len(anchor_hits) > 0:
                merged_seq, merged_start, merged_end, n_merged = _merge_hit_sequences(anchor_hits)
                used_hit_indices.update(anchor_hits.index)
            else:
                merged_seq = ''
                merged_start = int(anchor['anchor_start'])
                merged_end = int(anchor['anchor_end'])
                n_merged = 0

            gene_units.append({
                'gene_unit_id': gu_id, 'chrom': chrom,
                'accession': acc,
                'anchor_pos': int(anchor['anchor_mid']),
                'start': merged_start, 'end': merged_end,
                'span': merged_end - merged_start + 1,
                'n_hits': n_merged,
                'total_seq_len': len(merged_seq),
                'sequence': merged_seq, 'has_exon13': True,
            })
            gu_id += 1

    # --- Non-anchored accessions (chr Z, chr 20, etc.) ---
    non_anchored = all_blast_accs - anchored_accs
    for acc in sorted(non_anchored):
        hits = tblastn_df[tblastn_df['accession'] == acc]
        chrom = accession_to_chrom.get(acc, 'unknown')
        merged_loci = merge_overlapping_hits(hits, max_gap=merge_gap)
        for _, locus in merged_loci.iterrows():
            gene_units.append({
                'gene_unit_id': gu_id, 'chrom': chrom,
                'accession': acc,
                'anchor_pos': None,
                'start': int(locus['start']), 'end': int(locus['end']),
                'span': int(locus['span']),
                'n_hits': int(locus['n_hits']),
                'total_seq_len': int(locus['total_seq_len']),
                'sequence': locus['sequence'], 'has_exon13': False,
            })
            gu_id += 1

    return pd.DataFrame(gene_units)


def _merge_hit_sequences(hits_df):
    """Merge overlapping tBLASTn hit sequences within a gene unit.

    Sorts hits by start position, merges overlapping coordinate ranges,
    and concatenates only non-overlapping sequence portions.

    Returns (concatenated_sequence, start, end, n_hits).
    """
    hits_sorted = hits_df.sort_values('start').reset_index(drop=True)
    merged_seq_parts = []
    current_end = -1

    for _, row in hits_sorted.iterrows():
        if row['start'] > current_end:
            # No overlap — use full sequence
            merged_seq_parts.append(row['sequence'])
        else:
            # Overlap — trim the overlapping prefix from this hit's sequence
            overlap_bp = current_end - row['start'] + 1
            if overlap_bp < len(row['sequence']):
                merged_seq_parts.append(row['sequence'][overlap_bp:])
        current_end = max(current_end, row['end'])

    concat_seq = ''.join(merged_seq_parts)
    return (concat_seq,
            int(hits_sorted.iloc[0]['start']),
            int(hits_sorted['end'].max()),
            len(hits_sorted))


def gene_units_to_fasta(gu_df, outpath):
    """Write gene units DataFrame to FASTA file."""
    records = []
    for _, row in gu_df.iterrows():
        if row['total_seq_len'] == 0:
            continue
        rec = SeqRecord(
            Seq(row['sequence']),
            id=f"gu_{row['gene_unit_id']}_chr{row['chrom']}_{row['start']}_{row['end']}",
            description=f"n_hits={row['n_hits']} span={row['span']}bp exon13={row['has_exon13']}",
        )
        records.append(rec)
    SeqIO.write(records, outpath, "fasta")
    return len(records)
