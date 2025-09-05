#!/usr/bin/env python
# Copyright 2021 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from optparse import OptionParser
import gc
import json
import os
import re
import shutil

from intervaltree import IntervalTree
import numpy as np
import pandas as pd
import pybedtools
import pyranges as pr
from qnorm import quantile_normalize
from scipy.stats import pearsonr
from sklearn.metrics import explained_variance_score
import tensorflow as tf

import pygene
from baskerville import dataset
from baskerville import seqnn

"""
borzoi_test_genes_all_splits.py

Measure accuracy at gene-level for all splits: 'train', 'valid', and 'test',
including optional normalization by library size (matching by track identifier).

For each split, outputs:
  - <split>/gene_targets.tsv
  - <split>/gene_preds.tsv
under the provided out_dir.
"""

def untransform(array):
    return (array + 1)**2 - 1

def parse_group(desc: str) -> str:
    """Categorizes the track type based on description content."""
    desc_lower = desc.lower()
    if "pos_logfe" in desc_lower or "chip-exo" in desc_lower:
        return "ChIP-exo"
    elif "chip-mnase" in desc_lower:
        return "ChIP-MNase"
    elif "1000 strains rnaseq" in desc_lower:
        return "1000-RNA-seq"
    elif "rnaseq" in desc_lower or "rna_seq" in desc_lower:
        return "RNA-Seq"
    else:
        return "Other"

def make_genes_exon(genes_bed_file: str, genes_gtf_file: str, out_dir: str):
    """Make a BED file with each genes' exons, excluding exons overlapping
      across genes.

    Args:
      genes_bed_file (str): Output BED file of genes.
      genes_gtf_file (str): Input GTF file of genes.
      out_dir (str): Output directory for temporary files.
    """
    # read genes
    genes_gtf = pygene.GTF(genes_gtf_file)

    # write gene exons
    agenes_bed_file = "%s/genes_all.bed" % out_dir
    agenes_bed_out = open(agenes_bed_file, "w")
    for gene_id, gene in genes_gtf.genes.items():
        # collect exons
        gene_intervals = IntervalTree()
        for tx_id, tx in gene.transcripts.items():
            for exon in tx.exons:
                gene_intervals[exon.start - 1 : exon.end] = True

        # union
        gene_intervals.merge_overlaps()

        # write
        for interval in sorted(gene_intervals):
            cols = [
                gene.chrom,
                str(interval.begin),
                str(interval.end),
                gene_id,
                ".",
                gene.strand,
            ]
            print("\t".join(cols), file=agenes_bed_out)
    agenes_bed_out.close()

    # find overlapping exons
    genes1_bt = pybedtools.BedTool(agenes_bed_file)
    genes2_bt = pybedtools.BedTool(agenes_bed_file)
    overlapping_exons = set()
    for overlap in genes1_bt.intersect(genes2_bt, s=True, wo=True):
        gene1_id = overlap[3]
        gene1_start = int(overlap[1])
        gene1_end = int(overlap[2])
        overlapping_exons.add((gene1_id, gene1_start, gene1_end))

        gene2_id = overlap[9]
        gene2_start = int(overlap[7])
        gene2_end = int(overlap[8])
        overlapping_exons.add((gene2_id, gene2_start, gene2_end))

    # filter for nonoverlapping exons
    genes_bed_out = open(genes_bed_file, "w")
    for line in open(agenes_bed_file):
        a = line.split()
        start = int(a[1])
        end = int(a[2])
        gene_id = a[-1]
        if (gene_id, start, end) not in overlapping_exons:
            print(line, end="", file=genes_bed_out)
    genes_bed_out.close()


def make_genes_span(
    genes_bed_file: str, genes_gtf_file: str, out_dir: str, stranded: bool = True
):
    """Make a BED file with the span of each gene.

    Args:
      genes_bed_file (str): Output BED file of genes.
      genes_gtf_file (str): Input GTF file of genes.
      out_dir (str): Output directory for temporary files.
      stranded (bool): Perform stranded intersection.
    """
    # read genes
    genes_gtf = pygene.GTF(genes_gtf_file)

    # write all gene spans
    agenes_bed_file = "%s/genes_all.bed" % out_dir
    agenes_bed_out = open(agenes_bed_file, "w")
    for gene_id, gene in genes_gtf.genes.items():
        start, end = gene.span()
        cols = [gene.chrom, str(start - 1), str(end), gene_id, ".", gene.strand]
        print("\t".join(cols), file=agenes_bed_out)
    agenes_bed_out.close()

    # find overlapping genes
    genes1_bt = pybedtools.BedTool(agenes_bed_file)
    genes2_bt = pybedtools.BedTool(agenes_bed_file)
    overlapping_genes = set()
    for overlap in genes1_bt.intersect(genes2_bt, s=stranded, wo=True):
        gene1_id = overlap[3]
        gene2_id = overlap[7]
        if gene1_id != gene2_id:
            overlapping_genes.add(gene1_id)
            overlapping_genes.add(gene2_id)

    # filter for nonoverlapping genes
    genes_bed_out = open(genes_bed_file, "w")
    for line in open(agenes_bed_file):
        gene_id = line.split()[-1]
        if gene_id not in overlapping_genes:
            print(line, end="", file=genes_bed_out)
    genes_bed_out.close()


def compute_metrics(y_true, y_pred, eps=1e-8):
    """Compute Pearson and explained variance between y_true and y_pred."""
    if np.var(y_true) < eps or np.var(y_pred) < eps:
        return np.nan, np.nan
    try:
        pearson_val = pearsonr(y_true, y_pred)[0]
    except Exception:
        pearson_val = np.nan
    evs_val = explained_variance_score(y_true, y_pred)
    return pearson_val, evs_val

def evaluate_split(split_label,
                   params_model, params_train,
                   targets_df, targets_strand_df,
                   pool_width, gene_lengths, gene_strand,
                   libsize_arr, data_dir, model_file,
                   out_dir, statistics_path, tfr_pattern,
                   eval_dir, global_eval, genes_pr, options):
    """
    Evaluate model on one split (train/valid/test), write gene_preds.tsv and gene_targets.tsv
    into out_dir/<split_label>/. Uses libsize_arr aligned to targets_strand_df.
    """
    print(f"\n=== Evaluating split '{split_label}' ===")
    split_out = os.path.join(out_dir, split_label)
    os.makedirs(split_out, exist_ok=True)

    # Load statistics.json so we know pool_width etc.
    data_stats = json.load(open(statistics_path))
    pool_width = data_stats["pool_width"]
    num_species = data_stats.get("num_species", 1)
    if params_train["task"] == "fine-tune":
        num_species = 165

    # Reinitialize the model here (num_features already set in main)
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_file, options.head_i)
    seqnn_model.build_slice(targets_df.index)
    seqnn_model.build_ensemble(options.rc, options.shifts)

    # Read all sequences (no split-based filtering)
    seqs_df = pd.read_csv(os.path.join(data_dir, "sequences.bed"),
                          sep="\t", names=["Chromosome", "Start", "End", "Name"])
    seqs_df["Chromosome"] = seqs_df["Chromosome"].str.replace("chr", "", regex=False)
    seqs_pr = pr.PyRanges(seqs_df)

    # Intersect sequences with genes (uses prebuilt genes_pr)
    seqs_genes_pr = seqs_pr.join(genes_pr)

    # Prepare dictionaries to collect per-gene predictions and targets
    gene_preds_dict = {}
    gene_targets_dict = {}

    si = 0
    ds = dataset.SeqDataset(
        data_dir,
        split_label=split_label,
        batch_size=params_train["batch_size"],
        mode="eval",
        tfr_pattern=tfr_pattern,
        eval_dir=eval_dir,
        global_eval=global_eval
    )
    for x_batch, y_batch in ds.dataset:
        # Handle fine-tune feature dimension
        if params_train["task"] == "fine-tune":
            new_shape = x_batch.shape[:-1] + (num_species + 1,)
            x_new = tf.concat([x_batch, tf.zeros(new_shape)], axis=-1)
            x_new = tf.tensor_scatter_nd_update(
                x_new,
                indices=tf.constant([[i, j, 114]
                                     for i in range(x_new.shape[0])
                                     for j in range(x_new.shape[1])]),
                updates=tf.ones((x_new.shape[0] * x_new.shape[1],))
            )
            x_batch = x_new

        y_batch = y_batch.numpy()[..., targets_df.index]

        print(f"  Processing batch @ seq idx {si}...", end="")
        yh_batch = seqnn_model(x_batch)

        for b in range(x_batch.shape[0]):
            seq = seqs_df.iloc[si + b]
            chr_bins = seqs_genes_pr[seq.Chromosome].df
            if chr_bins.shape[0] == 0:
                continue
            seq_bins = chr_bins[chr_bins.Start == seq.Start]
            for _, sg in seq_bins.iterrows():
                gene_id = sg.Name_b
                gene_start = sg.Start_b
                gene_end = sg.End_b
                seq_start = sg.Start

                gene_seq_start = max(0, gene_start - seq_start)
                gene_seq_end = max(0, gene_end - seq_start)

                bin_start = int(np.round(gene_seq_start / pool_width))
                bin_end = int(np.round(gene_seq_end / pool_width))

                if options.untransform:
                    y_pred_vals = untransform(yh_batch[b, bin_start:bin_end].astype("float16"))
                    y_true_vals = untransform(y_batch[b, bin_start:bin_end].astype("float16"))
                else:
                    y_pred_vals = yh_batch[b, bin_start:bin_end].astype("float16")
                    y_true_vals = y_batch[b, bin_start:bin_end].astype("float16")

                if y_true_vals.shape[0] > 0:
                    gene_preds_dict.setdefault(gene_id, []).append(y_pred_vals)
                    gene_targets_dict.setdefault(gene_id, []).append(y_true_vals)

        si += x_batch.shape[0]
        print(" DONE")
        if si % 128 == 0:
            gc.collect()

    # Aggregate per-gene means
    gene_ids = sorted(gene_targets_dict.keys())
    gene_preds = []
    gene_targets = []

    for gene_id in gene_ids:
        pb = np.concatenate(gene_preds_dict[gene_id], axis=0).astype("float32")
        tb = np.concatenate(gene_targets_dict[gene_id], axis=0).astype("float32")

        # Slice out-of-strand columns
        if gene_strand[gene_id] == "+":
            mask = (targets_df.strand != "-").to_numpy()
        else:
            mask = (targets_df.strand != "+").to_numpy()

        pb = pb[:, mask]
        tb = tb[:, mask]

        # Normalize by library size if provided
        if libsize_arr is not None:
            pb = (pb / libsize_arr[mask]) * 1e6
            tb = (tb / libsize_arr[mask]) * 1e6

        # Mean over bins, then divide by pool_width
        pb_mean = pb.mean(axis=0) / float(pool_width)
        tb_mean = tb.mean(axis=0) / float(pool_width)

        # Scale by gene length
        pb_mean *= gene_lengths[gene_id]
        tb_mean *= gene_lengths[gene_id]

        gene_preds.append(pb_mean)
        gene_targets.append(tb_mean)

    gene_preds = np.array(gene_preds)
    gene_targets = np.array(gene_targets)
    
    gene_targets = np.log2(gene_targets + 1)
    gene_preds   = np.log2(gene_preds + 1)

    # Write outputs
    preds_df = pd.DataFrame(
        gene_preds, index=gene_ids, columns=targets_strand_df.identifier[mask]
    )
    targets_df_out = pd.DataFrame(
        gene_targets, index=gene_ids, columns=targets_strand_df.identifier[mask]
    )
    preds_file = os.path.join(split_out, "gene_preds.tsv")
    targets_file = os.path.join(split_out, "gene_targets.tsv")
    preds_df.to_csv(preds_file, sep="\t")
    targets_df_out.to_csv(targets_file, sep="\t")
    print(f"  Wrote: {preds_file}")
    print(f"  Wrote: {targets_file}")


if __name__ == "__main__":
    parser = OptionParser("usage: %prog [options] <params_file> <model_file> <data_dir> <genes_gtf>")
    parser.add_option("-f", "--file_type", dest="file_type", default="gtf", type="str",
                      help="Input file type: 'gtf' or 'bed' [Default: %default]")
    parser.add_option("--dataset_type", dest="dataset_type", default=None,
                      help="Filter targets by dataset_type (e.g. 'RNA-Seq'). [Default: none]")
    parser.add_option("--head", dest="head_i", default=0, type="int",
                      help="Model head index [Default: %default]")
    parser.add_option("-o", dest="out_dir", default="testg_out",
                      help="Output directory [Default: %default]")
    parser.add_option("--rc", dest="rc", default=False, action="store_true",
                      help="Average fwd and rc predictions [Default: %default]")
    parser.add_option("--shifts", dest="shifts", default="0",
                      help="Comma-separated ensemble shifts [Default: %default]")
    parser.add_option("--span", dest="span", default=False, action="store_true",
                      help="Aggregate entire gene span [Default: %default]")
    parser.add_option("-t", dest="targets_file", default=None, type="str",
                      help="Targets file (tab-delimited with description) [Default: data_dir/targets.txt]")
    parser.add_option("-s", dest="statistics", default=None, type="str",
                      help="Statistics JSON (Default: data_dir/statistics.json)")
    parser.add_option("--global_eval", dest="global_eval", default=False, action="store_true",
                      help="Use global eval dataset [Default: %default]")
    parser.add_option("--no_unclip", dest="no_unclip", default=False, action="store_true",
                      help="Do not unclip transform [Default: %default]")
    parser.add_option("--pseudo_qtl", dest="pseudo_qtl", default=None, type="float",
                      help="Quantile for pseudo-count addition [Default: %default]")
    parser.add_option("--tfr", dest="tfr_pattern", default=None,
                      help="TFRecord pattern override [Default: split_label]")
    parser.add_option("--eval_dir", dest="eval_dir", default=None,
                      help="Directory for global eval TFRecords [Default: %(default)s]")
    parser.add_option("--untransform", dest="untransform", default=False, action="store_true",
                      help="Apply untransform to preds/targets [Default: %(default}s]")
    parser.add_option("--libsize_file", dest="libsize_file", default=None, type="str",
                      help="CSV with columns 'file' and 'library_size' [Default: none]")
    (options, args) = parser.parse_args()

    if len(args) != 4:
        parser.error("Must provide params_file, model_file, data_dir, and genes_gtf")
    params_file, model_file, data_dir, genes_file = args
    os.makedirs(options.out_dir, exist_ok=True)

    # Parse shifts
    options.shifts = [int(x) for x in options.shifts.split(",")]

    # Read targets table
    if options.targets_file is None:
        options.targets_file = os.path.join(data_dir, "targets.txt")
    if options.statistics is None:
        options.statistics = os.path.join(data_dir, "statistics.json")

    targets_df = pd.read_csv(options.targets_file, index_col=0, sep="\t")
    targets_df["group"] = targets_df["description"].apply(parse_group)
    if options.dataset_type is not None:
        old = targets_df.shape[0]
        targets_df = targets_df[targets_df["group"] == options.dataset_type]
        new = targets_df.shape[0]
        if new == 0:
            print("No targets after filtering; exiting.")
            exit(1)

    # Load model params
    with open(params_file) as f:
        params = json.load(f)
    params_model = params["model"]
    params_train = params["train"]

    # Ensure num_features matches input shape
    data_stats = json.load(open(options.statistics))
    if params_train["task"] == "fine-tune":
        num_species = 165
        params_model["num_features"] = num_species + 5
    else:
        params_model["num_features"] = 4

    # Prep strand-specific targets
    targets_strand_df = dataset.targets_prep_strand(targets_df)

    # Load and process library sizes if requested
    libsize_arr = None
    if options.libsize_file:
        libsize_df = pd.read_csv(options.libsize_file)
        libsize_dict = {}
        for fp, size in zip(libsize_df["file"], libsize_df["library_size"]):
            base = os.path.basename(fp)
            name_no_ext = base.replace(".bw", "")
            # strip trailing "_coverage" or "_bamcov" if present
            track_id = re.sub(r"_(coverage|bamcov)$", "", name_no_ext)
            libsize_dict[track_id] = size
        libsize_arr = np.array([libsize_dict.get(tid, np.nan)
                                for tid in targets_strand_df.identifier])
    # Build genes.bed once, at top level
    genes_bed_file = os.path.join(options.out_dir, "genes.bed")
    if options.file_type.lower() == "gtf":
        if options.span:
            make_genes_span(genes_bed_file, genes_file, options.out_dir)
        else:
            make_genes_exon(genes_bed_file, genes_file, options.out_dir)
    else:
        shutil.copyfile(genes_file, genes_bed_file)
    genes_pr = pr.read_bed(genes_bed_file)

    # Compute gene lengths and strand
    gene_lengths = {}
    gene_strand = {}
    with open(genes_bed_file) as gf:
        for line in gf:
            cols = line.rstrip().split("\t")
            gid = cols[3]
            seg_len = int(cols[2]) - int(cols[1])
            gene_lengths[gid] = gene_lengths.get(gid, 0) + seg_len
            gene_strand[gid] = cols[5]

    # Loop over splits
    for split in ["train", "valid", "test"]:
        evaluate_split(
            split_label=split,
            params_model=params_model,
            params_train=params_train,
            targets_df=targets_df,
            targets_strand_df=targets_strand_df,
            pool_width=data_stats["pool_width"],
            gene_lengths=gene_lengths,
            gene_strand=gene_strand,
            libsize_arr=libsize_arr,
            data_dir=data_dir,
            model_file=model_file,
            out_dir=options.out_dir,
            statistics_path=options.statistics,
            tfr_pattern=options.tfr_pattern,
            eval_dir=options.eval_dir,
            global_eval=options.global_eval,
            genes_pr=genes_pr,
            options=options
        )
