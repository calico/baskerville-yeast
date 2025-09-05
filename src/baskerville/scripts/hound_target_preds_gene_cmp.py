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
import time
import re
import shutil  # <-- for copying the bed file directly

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
borzoi_test_genes.py

Measure accuracy at gene-level.
"""

def untransform(array):
    return (array + 1)**2 - 1

# ---- 1. Helper function to categorize based on description text ----
def parse_group(desc: str) -> str:
    """Categorizes the track type based on description content."""
    desc_lower = desc.lower()
    # print(desc_lower)
    if "pos_logfe" in desc_lower or "chip-exo" in desc_lower:
        return "ChIP-exo"
    elif "chip-mnase" in desc_lower:  # or "chip-mnase"
        return "ChIP-MNase"
    elif "1000 strains rnaseq" in desc_lower:
        return "1000-RNA-seq"
    elif "rnaseq" in desc_lower or "rna_seq" in desc_lower:
        return "RNA-Seq"
    else:
        return "Other"

#######################################################
# Improved: Accuracy stats per gene (across all tracks)

def compute_metrics(y_true, y_pred, eps=1e-8):
    """
    Compute Pearson correlation and explained variance score between
    y_true and y_pred. Returns (pearson, explained_variance).
    If either vector has near-zero variance, returns (np.nan, np.nan).
    """
    if np.var(y_true) < eps or np.var(y_pred) < eps:
        return np.nan, np.nan
    try:
        pearson_val = pearsonr(y_true, y_pred)[0]
    except Exception:
        pearson_val = np.nan
    evs_val = explained_variance_score(y_true, y_pred)
    return pearson_val, evs_val


################################################################################
# main
################################################################################
def main():
    usage = "usage: %prog [options] <params_file> <model_file> <data_dir> <genes_gtf>"
    parser = OptionParser(usage)
    parser.add_option(
        "-f",
        "--file_type",
        dest="file_type",
        default="gtf",
        type="str",
        help="Input file type: 'gtf' or 'bed' [Default: %default]",
    )
    parser.add_option(
        "--dataset_type",
        dest="dataset_type",
        default=None,
        help="Group of dataset to evaluate: 'Chip-exo', 'Chip-MNase', or 'RNAseq'. [Default: evaluate all]",
    )
    parser.add_option(
        "--head",
        dest="head_i",
        default=0,
        type="int",
        help="Parameters head [Default: %default]",
    )
    parser.add_option(
        "-o",
        dest="out_dir",
        default="testg_out",
        help="Output directory for predictions [Default: %default]",
    )
    parser.add_option(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Average the fwd and rc predictions [Default: %default]",
    )
    parser.add_option(
        "--shifts",
        dest="shifts",
        default="0",
        help="Ensemble prediction shifts [Default: %default]",
    )
    parser.add_option(
        "--span",
        dest="span",
        default=False,
        action="store_true",
        help="Aggregate entire gene span [Default: %default]",
    )
    parser.add_option(
        "-t",
        dest="targets_file",
        default=None,
        type="str",
        help="File specifying target indexes and labels in table format",
    )
    parser.add_option(
        "-s",
        dest="statistics",
        default=None,
        type="str",
        help="File specifying statistics",
    )
    parser.add_option(
        "--split",
        dest="split_label",
        default="test",
        help="Dataset split label for eg TFR pattern [Default: %default]",
    )
    parser.add_option(
        '--global_eval',
        dest='global_eval',
        default=False,
        action='store_true',
        help='Evaluating on the global evaluation dataset [Default: %default]',
    )
    parser.add_option(
        '--no_unclip',
        dest='no_unclip',
        default=False,
        action='store_true',
        help='Turn off unclip transform [Default: %default]',
    )
    parser.add_option(
        '--pseudo_qtl',
        dest='pseudo_qtl',
        default=None,
        type='float',
        help='Quantile of coverage to add as pseudo counts to genes [Default: %default]',
    )
    parser.add_option(
        "--tfr",
        dest="tfr_pattern",
        default=None,
        help="TFR pattern string appended to data_dir/tfrecords for subsetting [Default: %default]",
    )
    parser.add_option(
        "--eval_dir",
        default=None,
        help="The directory to the validation data_dir/tfrecords [Default: %(default)s]",
    )
    parser.add_option(
        "--untransform",
        default=False,
        action="store_true",
        help="Untransform the data [Default: %(default)s]",
    )

    (options, args) = parser.parse_args()

    if len(args) != 4:
        parser.error("Must provide parameters, model, data directory, and genes GTF")
    else:
        params_file = args[0]
        model_file = args[1]
        data_dir = args[2]
        genes_file = args[3]

    print("Options: ")
    print("  Params: %s" % params_file)
    print("  Model: %s" % model_file)
    print("  Data: %s" % data_dir)
    print("  Genes: %s" % genes_file)

    # After reading the `targets_file`, `model_file`, and others...
    # Read the library sizes from the CSV
    library_sizes_file = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/library_sizes.csv"
    library_sizes_df = pd.read_csv(library_sizes_file)
    print("Library sizes file:", library_sizes_file)
    print("library_sizes_df: ", library_sizes_df)
    library_sizes = dict(zip(library_sizes_df['identifier'], library_sizes_df['library_size']))

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    # parse shifts to integers
    options.shifts = [int(shift) for shift in options.shifts.split(",")]

    #######################################################
    # inputs
    # read targets
    if options.targets_file is None:
        options.targets_file = "%s/targets.txt" % data_dir
    if options.statistics is None:
        options.statistics = "%s/statistics.json" % data_dir
    print("Targets file:", options.targets_file)
    print("Statistics file:", options.statistics)
    targets_df = pd.read_csv(options.targets_file, index_col=0, sep="\t")
    # Add a 'group' column based on the description
    targets_df["group"] = targets_df["description"].apply(parse_group)
    print("Targets file read. All targets:", targets_df.shape[0], "entries")

    # Filter out the tracks that I want to evaluate.
    # If dataset_type is specified, filter to that group.
    if options.dataset_type is not None:
        old_count = targets_df.shape[0]
        targets_df = targets_df[targets_df["group"] == options.dataset_type]
        new_count = targets_df.shape[0]
        print(f"Filtering for {options.dataset_type}: {old_count} -> {new_count} entries")

    # If after filtering, targets_df is empty, we can check that and optionally quit:
    if targets_df.shape[0] == 0:
        print(f"No targets found matching dataset_type='{options.dataset_type}'. Exiting.")
        return
    
    # read model parameters
    with open(params_file) as params_open:
        params = json.load(params_open)
    params_model = params["model"]
    params_train = params["train"]

    # read data parameters
    with open(options.statistics) as data_open:
        data_stats = json.load(data_open)
        crop_bp = data_stats["crop_bp"]
        pool_width = data_stats["pool_width"]
    num_species = data_stats.get("num_species", 1)
    if params_train["task"] == "fine-tune":
        num_species = 165
    print("num_species: ", num_species)
    print("params_train: ", params_train)


    # prep strand
    targets_strand_df = dataset.targets_prep_strand(targets_df)
    num_targets = targets_df.shape[0]
    num_targets_strand = targets_strand_df.shape[0]

    print("Targets strand:", targets_strand_df.shape)   
    print(targets_strand_df)

    # construct eval data
    eval_data = dataset.SeqDataset(
        data_dir,
        split_label=options.split_label,
        batch_size=params_train["batch_size"],
        mode="eval",
        tfr_pattern=options.tfr_pattern,
        eval_dir=options.eval_dir, 
        global_eval = options.global_eval
    )

    params_model["num_features"] = 4
    if params_train["loss"] == 'mlm':
        params_model["num_features"] = num_species + 5

    if params_train["task"] == "fine-tune":
        params_model["num_features"] = num_species + 5
    
    # print("params_model[num_features]: ", params_model["num_features"])
    # print("load model_file: ", model_file)

    # initialize model
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_file, options.head_i)
    seqnn_model.build_slice(targets_df.index)
    seqnn_model.build_ensemble(options.rc, options.shifts)

    #######################################################
    # sequence intervals


    # read sequence positions
    seqs_df = pd.read_csv(
        "%s/sequences.bed" % data_dir,
        sep="\t",
        names=["Chromosome", "Start", "End", "Name"],
    )
    seqs_df = seqs_df[seqs_df.Name == options.split_label]

    # Remove "chr" from the Chromosome column
    seqs_df['Chromosome'] = seqs_df['Chromosome'].str.replace('chr', '', regex=False)
    seqs_pr = pr.PyRanges(seqs_df)


    #######################################################
    # make gene BED

    t0 = time.time()
    print("Making gene BED...", end="")
    genes_bed_file = "%s/genes.bed" % options.out_dir
    
    if options.file_type.lower() == "gtf":
        # Use the GTF to build a gene BED
        if options.span:
            make_genes_span(genes_bed_file, genes_file, options.out_dir)
        else:
            make_genes_exon(genes_bed_file, genes_file, options.out_dir)
    elif options.file_type.lower() == "bed":
        # The user has already provided a BED. Just copy it over.
        shutil.copyfile(genes_file, genes_bed_file)
    else:
        raise ValueError("Invalid file type. Must be 'gtf' or 'bed'.")
    genes_pr = pr.read_bed(genes_bed_file)
    print("genes_pr: ", genes_pr)
    print("DONE in %ds" % (time.time() - t0))


    # count gene normalization lengths
    gene_lengths = {}
    gene_strand = {}
    for line in open(genes_bed_file):
        a = line.rstrip().split("\t")
        gene_id = a[3]
        gene_seg_len = int(a[2]) - int(a[1])
        gene_lengths[gene_id] = gene_lengths.get(gene_id, 0) + gene_seg_len
        gene_strand[gene_id] = a[5]

    #######################################################
    # intersect genes w/ preds, targets

    # intersect seqs, genes
    t0 = time.time()
    print("Intersecting sequences w/ genes...", end="")
    seqs_genes_pr = seqs_pr.join(genes_pr)
    print("seqs_genes_pr: ", seqs_genes_pr)
    print("DONE in %ds" % (time.time() - t0), flush=True)

    # hash preds/targets by gene_id
    gene_preds_dict = {}
    gene_targets_dict = {}

    counter = 0
    si = 0
    for x, y in eval_data.dataset:
        # print("x.shape, y.shape: ", x.shape, y.shape)
        counter += 1

        if params_train["task"] == "fine-tune":
            # !!!Change the dimension of the X for fine-tuning
            # Create a new tensor filled with zeros of the desired shape
            new_shape = x.shape[:-1] + (num_species+1,)
            # Copy the original tensor into the first 4 positions
            x_new = tf.concat([
                x, 
                tf.zeros(new_shape)
            ], axis=-1)
            # Use TensorFlow indexing to set the desired column to 1
            x_new = tf.tensor_scatter_nd_update(
                x_new,
                indices=tf.constant([[i, j, 114] for i in range(x_new.shape[0]) for j in range(x_new.shape[1])]),
                updates=tf.ones((x_new.shape[0] * x_new.shape[1],))
            )
            x = x_new

        # print("new x.shape, y.shape: ", x.shape, y.shape)

        # predict only if gene overlaps
        yh = None
        y = y.numpy()[..., targets_df.index]

        t0 = time.time()
        print("Sequence %d..." % si, end="")
        for bsi in range(x.shape[0]):
            seq = seqs_df.iloc[si + bsi]

            cseqs_genes_df = seqs_genes_pr[seq.Chromosome].df
            if cseqs_genes_df.shape[0] == 0:
                # empty. no genes on this chromosome
                seq_genes_df = cseqs_genes_df
            else:
                seq_genes_df = cseqs_genes_df[cseqs_genes_df.Start == seq.Start]

            for _, seq_gene in seq_genes_df.iterrows():
                gene_id = seq_gene.Name_b
                gene_start = seq_gene.Start_b
                gene_end = seq_gene.End_b
                seq_start = seq_gene.Start

                # clip boundaries
                gene_seq_start = max(0, gene_start - seq_start)
                gene_seq_end = max(0, gene_end - seq_start)

                # requires >50% overlap
                bin_start = int(np.round(gene_seq_start / pool_width))
                bin_end = int(np.round(gene_seq_end / pool_width))

                # predict
                if yh is None:
                    yh = seqnn_model(x)
                    print("yh: ", yh.shape)

                if options.untransform:
                    yhb = untransform(yh[bsi, bin_start:bin_end].astype("float16"))
                    yb = untransform(y[bsi, bin_start:bin_end].astype("float16"))
                else:
                    # slice gene region
                    yhb = yh[bsi, bin_start:bin_end].astype("float16")
                    yb = y[bsi, bin_start:bin_end].astype("float16")

                if len(yb) > 0:
                    gene_preds_dict.setdefault(gene_id, []).append(yhb)
                    gene_targets_dict.setdefault(gene_id, []).append(yb)

        # advance sequence table index
        si += x.shape[0]
        print("DONE in %ds" % (time.time() - t0), flush=True)
        if si % 128 == 0:
            gc.collect()

        # if counter > 10:
        #     break

    gene_targets = []
    gene_preds = []

    # Define lists to hold normalized results
    all_gene_preds_norm = []
    all_gene_targets_norm = []

    gene_ids = sorted(gene_targets_dict.keys())

    gene_within = []     # unnormalized bin-level correlation across bins
    gene_wvar = []       # unnormalized bin-level variance

    # We'll store bin-level data for future normalization as well:
    all_bin_targets = []
    all_bin_preds = []
    all_bin_gene_idx = []  # which gene each bin belongs to

    num_genes = len(gene_ids)


    counter = 0
    for g, gene_id in enumerate(gene_ids):
        counter += 1

        print("Gene %s..." % gene_id)
        # Concatenate bin-level data for this gene, shape: [num_bins_in_gene, num_targets_strand]
        gene_preds_gi = np.concatenate(gene_preds_dict[gene_id], axis=0).astype("float32")
        gene_targets_gi = np.concatenate(gene_targets_dict[gene_id], axis=0).astype("float32")

        print("gene_preds_gi.shape, gene_targets_gi.shape: ", gene_preds_gi.shape, gene_targets_gi.shape)

        # slice strand
        if gene_strand[gene_id] == "+":
            gene_strand_mask = (targets_df.strand != "-").to_numpy()
        else:
            gene_strand_mask = (targets_df.strand != "+").to_numpy()
        gene_preds_gi = gene_preds_gi[:, gene_strand_mask]
        gene_targets_gi = gene_targets_gi[:, gene_strand_mask]

        if gene_targets_gi.shape[0] == 0:
            print(gene_id, gene_targets_gi.shape, gene_preds_gi.shape)

        num_bins_in_gene = gene_targets_gi.shape[0]
        num_targets_strand = gene_targets_gi.shape[1]  # might already be known above

        # -------------------------------------------------------------------------
        # 1A) Compute unnormalized within-gene correlation across bins for each track
        # -------------------------------------------------------------------------
        gene_corr_gi = np.zeros(num_targets_strand, dtype='float32')
        for ti in range(num_targets_strand):
            var_p = gene_preds_gi[:, ti].var()
            var_t = gene_targets_gi[:, ti].var()
            if var_p > 1e-6 and var_t > 1e-6:
                preds_log = np.log2(gene_preds_gi[:, ti] + 1)
                targets_log = np.log2(gene_targets_gi[:, ti] + 1)
                gene_corr_gi[ti] = pearsonr(preds_log, targets_log)[0]
            else:
                gene_corr_gi[ti] = np.nan

        # store
        gene_within.append(gene_corr_gi)
        gene_wvar.append(gene_targets_gi.var(axis=0))

        # -------------------------------------------------------------------------
        # 1B) For bin-level normalization (later), store this gene's bins in big arrays
        #     We'll push them AFTER log-transform below, so let's do that now
        # -------------------------------------------------------------------------
        # We do log2 transform first for the big arrays,
        # so that the normalization step can operate on log-scale data if desired.
        # (If you prefer to quantile-normalize on linear scale, then just skip log here.)
        # We'll do the same approach as you do for the unnormalized correlation above.
        # So let's store "log2(x+1)" directly into the big arrays.

        gene_targets_gi_log = np.log2(gene_targets_gi + 1)
        gene_preds_gi_log   = np.log2(gene_preds_gi + 1)

        all_bin_targets.append(gene_targets_gi_log)
        all_bin_preds.append(gene_preds_gi_log)
        all_bin_gene_idx += [g] * num_bins_in_gene

        # -------------------------------------------------------------------------
        # 1C) Now, *before* we proceed, let's do the gene-level pooling for unnormalized arrays
        #     so that we can do the rest of your logic. We do this AFTER the within-gene correlation
        #     has been computed from unnormalized data.
        # -------------------------------------------------------------------------
        # mean coverage across bins
        gene_preds_mean = gene_preds_gi.mean(axis=0) / float(pool_width)
        gene_targets_mean = gene_targets_gi.mean(axis=0) / float(pool_width)

        # scale by gene length
        gene_preds_mean *= gene_lengths[gene_id]
        gene_targets_mean *= gene_lengths[gene_id]

        # store these pooled values
        gene_preds.append(gene_preds_mean)
        gene_targets.append(gene_targets_mean)

        # -----------------------------------------------------------------------------
        # Normalize gene expression by library size (integrated into the loop)
        # -----------------------------------------------------------------------------
        gene_preds_norm_gi = np.zeros(num_targets_strand, dtype='float32')
        gene_targets_norm_gi = np.zeros(num_targets_strand, dtype='float32')
        
        # Normalize the targets and preds by library size
        for ti in range(num_targets_strand):
            track_id = targets_strand_df.iloc[ti]['identifier']  # Ensure correct positional indexing
            if track_id in library_sizes:
                library_size = library_sizes[track_id]
                # Normalize the values for both targets and predictions

                gene_targets_norm_gi[ti] = gene_targets_mean[ti] / library_size * 1e6
                gene_preds_norm_gi[ti] = gene_preds_mean[ti] / library_size * 1e6

                print("gene_targets_mean[ti]: ", gene_targets_mean[ti], "; library_size: ", library_size)
                print("gene_targets_norm_gi[ti]: ", gene_targets_norm_gi[ti] )
                print("gene_preds_mean[ti]: ", gene_preds_mean[ti], "; library_size: ", library_size)
                print("gene_preds_norm_gi[ti]: ", gene_preds_norm_gi[ti] )

        # Store the normalized results
        all_gene_preds_norm.append(gene_preds_norm_gi)
        all_gene_targets_norm.append(gene_targets_norm_gi)

        # if counter > 10:
        #     break


    # convert to np.array for unnormalized gene-level coverage
    gene_targets = np.array(gene_targets)  # shape: [num_genes, num_targets_strand]
    gene_preds = np.array(gene_preds)      # shape: [num_genes, num_targets_strand]
    gene_within = np.array(gene_within)    # shape: [num_genes, num_targets_strand]
    gene_wvar = np.array(gene_wvar)        # shape: [num_genes, num_targets_strand]

    # Convert big bin-level lists to arrays
    all_bin_targets = np.concatenate(all_bin_targets, axis=0)  # shape: [sum of all bins, num_targets_strand]
    all_bin_preds   = np.concatenate(all_bin_preds, axis=0)    # shape: [sum of all bins, num_targets_strand]
    all_bin_gene_idx = np.array(all_bin_gene_idx, dtype='int32')  # shape: [sum of all bins]

    # -----------------------------------------------------------------------------
    # 2) Save unnormalized results to disk (optional)
    # -----------------------------------------------------------------------------
    print("gene_targets.shape: ", gene_targets.shape)
    print("gene_preds.shape: ", gene_preds.shape)

    genes_targets_df = pd.DataFrame(
        gene_targets, index=gene_ids, columns=targets_strand_df.identifier
    )
    genes_targets_df.to_csv("%s/gene_targets.tsv" % options.out_dir, sep="\t")

    genes_preds_df = pd.DataFrame(
        gene_preds, index=gene_ids, columns=targets_strand_df.identifier
    )
    genes_preds_df.to_csv("%s/gene_preds.tsv" % options.out_dir, sep="\t")

    genes_within_df = pd.DataFrame(
        gene_within, index=gene_ids, columns=targets_strand_df.identifier
    )
    genes_within_df.to_csv("%s/gene_within.tsv" % options.out_dir, sep="\t")

    genes_var_df = pd.DataFrame(
        gene_wvar, index=gene_ids, columns=targets_strand_df.identifier
    )
    genes_var_df.to_csv("%s/gene_var.tsv" % options.out_dir, sep="\t")

    # -----------------------------------------------------------------------------
    # Save normalized results to a file (all at once)
    # -----------------------------------------------------------------------------
    genes_targets_norm_df = pd.DataFrame(
        all_gene_targets_norm, index=gene_ids, columns=targets_strand_df.identifier
    )
    genes_targets_norm_df.to_csv("%s/gene_targets_norm.tsv" % options.out_dir, sep="\t")

    genes_preds_norm_df = pd.DataFrame(
        all_gene_preds_norm, index=gene_ids, columns=targets_strand_df.identifier
    )
    genes_preds_norm_df.to_csv("%s/gene_preds_norm.tsv" % options.out_dir, sep="\t")
    



def genes_aggregate(genes_bed_file, values_bedgraph):
    """Aggregate values across genes.

    Args:
      genes_bed_file (str): BED file of genes.
      values_bedgraph (str): BedGraph file of values.

    Returns:
      gene_values (dict): Dictionary of gene values.
    """
    values_bt = pybedtools.BedTool(values_bedgraph)
    genes_bt = pybedtools.BedTool(genes_bed_file)

    gene_values = {}

    for overlap in genes_bt.intersect(values_bt, wo=True):
        gene_id = overlap[3]
        value = overlap[7]
        gene_values[gene_id] = gene_values.get(gene_id, 0) + value

    return gene_values


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


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()




    # # -----------------------------------------------------------------------------
    # # 3) Add pseudo-coverage at the *gene level* (like your original code).
    # #    We do not do pseudo coverage at the bin level unless you specifically want that.
    # # -----------------------------------------------------------------------------
    # if options.pseudo_qtl is not None:
    #     for ti in range(num_targets_strand):
    #         nonzero_index = np.nonzero(gene_targets[:, ti] != 0.)[0]
    #         if len(nonzero_index) > 0:
    #             pseudo_t = np.quantile(gene_targets[nonzero_index, ti], 
    #                                 q=options.pseudo_qtl)
    #             pseudo_p = np.quantile(gene_preds[nonzero_index,  ti], 
    #                                 q=options.pseudo_qtl)
    #             gene_targets[:, ti] += pseudo_t
    #             gene_preds[:, ti]   += pseudo_p

    # # -----------------------------------------------------------------------------
    # # 4) Log transform (gene-level) unnormalized + pseudo coverage
    # #    This is consistent with your original code. 
    # # -----------------------------------------------------------------------------
    # gene_targets = np.log2(gene_targets + 1)
    # gene_preds   = np.log2(gene_preds + 1)

    # # -----------------------------------------------------------------------------
    # # 5) Group targets by description for normalization
    # #    (We'll do a bin-level version for gene_within_norm.)
    # # -----------------------------------------------------------------------------
    # def get_group(description):
    #     if 'Chip-exo' in description:
    #         return 'ChIP-exo'
    #     elif 'Chip-MNase' in description:
    #         return 'ChIP-MNase'
    #     elif '1000 strains RNAseq' in description:
    #         return '1000-RNA-seq'
    #     elif 'RNAseq' in description:
    #         return 'RNA-seq'
    #     else:
    #         return 'Other'

    # targets_strand_df['group'] = targets_strand_df['description'].apply(get_group)
    # print("Targets strand:", targets_strand_df.shape)
    # print(targets_strand_df)

    # # -----------------------------------------------------------------------------
    # # 6) Gene-level normalization (as in your original script)
    # #    This yields: gene_targets_norm, gene_preds_norm
    # # -----------------------------------------------------------------------------
    # gene_targets_norm = np.zeros_like(gene_targets)
    # gene_preds_norm   = np.zeros_like(gene_preds)

    # for group in targets_strand_df['group'].unique():
    #     print(f"Normalizing group (gene-level): {group}")
    #     group_mask = (targets_strand_df['group'] == group)
    #     group_indices = np.where(group_mask)[0]

    #     # Quantile normalize across genes
    #     # shape is [num_genes, number_of_tracks_in_this_group]
    #     gene_targets_norm[:, group_indices] = quantile_normalize(
    #         gene_targets[:, group_indices], ncpus=2
    #     )
    #     gene_preds_norm[:, group_indices]   = quantile_normalize(
    #         gene_preds[:, group_indices], ncpus=2
    #     )

    #     # Mean center across tracks (axis=-1 => last dimension => across columns)
    #     gene_targets_norm[:, group_indices] -= gene_targets_norm[:, group_indices].mean(
    #         axis=-1, keepdims=True
    #     )
    #     gene_preds_norm[:, group_indices]   -= gene_preds_norm[:, group_indices].mean(
    #         axis=-1, keepdims=True
    #     )

    # # -----------------------------------------------------------------------------
    # # 7) BIN-level normalization for "within-gene correlation" in normalized space.
    # #    We already have `all_bin_targets` and `all_bin_preds` in log space.
    # # -----------------------------------------------------------------------------
    # all_bin_targets_norm = all_bin_targets.copy()
    # all_bin_preds_norm   = all_bin_preds.copy()

    # for group in targets_strand_df['group'].unique():
    #     group_mask = (targets_strand_df['group'] == group)
    #     group_indices = np.where(group_mask)[0]

    #     # If no tracks in this group, skip
    #     if len(group_indices) == 0:
    #         continue

    #     # all_bin_* arrays have shape (num_total_bins, num_targets_strand)
    #     # We'll quantile-normalize the columns in 'group_indices' across all bins
    #     all_bin_targets_norm[:, group_indices] = quantile_normalize(
    #         all_bin_targets_norm[:, group_indices], ncpus=2
    #     )
    #     all_bin_preds_norm[:, group_indices]   = quantile_normalize(
    #         all_bin_preds_norm[:, group_indices], ncpus=2
    #     )

    #     # Then mean-center across columns (tracks) for each bin
    #     all_bin_targets_norm[:, group_indices] -= all_bin_targets_norm[:, group_indices].mean(
    #         axis=-1, keepdims=True
    #     )
    #     all_bin_preds_norm[:, group_indices]   -= all_bin_preds_norm[:, group_indices].mean(
    #         axis=-1, keepdims=True
    #     )

    # # -----------------------------------------------------------------------------
    # # 7A) Now compute gene_within_norm (within-gene correlation in normalized space).
    # #     We'll split all_bin_*_norm back into genes, compute correlation across bins.
    # # -----------------------------------------------------------------------------
    # gene_within_norm = []
    # gene_wvar_norm   = []

    # for g, gene_id in enumerate(gene_ids):
    #     # extract the rows (bins) corresponding to this gene
    #     mask_gene_bins = (all_bin_gene_idx == g)
    #     # shape: [num_bins_in_gene, num_targets_strand]
    #     gb_targets = all_bin_targets_norm[mask_gene_bins, :]
    #     gb_preds   = all_bin_preds_norm[mask_gene_bins, :]

    #     num_bins_in_gene = gb_targets.shape[0]
    #     if num_bins_in_gene == 0:
    #         # shouldn't happen, but just in case
    #         gene_within_norm.append([np.nan]*num_targets_strand)
    #         gene_wvar_norm.append([0]*num_targets_strand)
    #         continue

    #     corr_array = np.zeros(num_targets_strand, dtype='float32')
    #     var_array  = np.zeros(num_targets_strand, dtype='float32')

    #     for ti in range(num_targets_strand):
    #         var_p = gb_preds[:, ti].var()
    #         var_t = gb_targets[:, ti].var()
    #         var_array[ti] = var_t
    #         if var_p > 1e-6 and var_t > 1e-6:
    #             # correlation across bins for track ti
    #             r = pearsonr(gb_preds[:, ti], gb_targets[:, ti])[0]
    #             corr_array[ti] = r
    #         else:
    #             corr_array[ti] = np.nan

    #     gene_within_norm.append(corr_array)
    #     gene_wvar_norm.append(var_array)

    # gene_within_norm = np.array(gene_within_norm)  # shape [num_genes, num_targets_strand]
    # gene_wvar_norm   = np.array(gene_wvar_norm)

    # # -----------------------------------------------------------------------------
    # # 7B) For the normalized gene-level coverage, we should also pool the bin-level
    # #     normalized predictions if you want consistent "gene_preds_norm" / "gene_targets_norm"
    # #     from the BIN-level approach. 
    # #     BUT you already have a GENE-level normalization (gene_targets_norm / gene_preds_norm)
    # #     from step (6). That is a separate approach.
    # #
    # #     If you truly want "gene_preds_norm" that is the result of BIN-level normalization
    # #     and then pooling, do so here. We'll call it gene_preds_norm_binlevel, etc.
    # # -----------------------------------------------------------------------------

    # gene_preds_norm_binlevel = []
    # gene_targets_norm_binlevel = []

    # for g, gene_id in enumerate(gene_ids):
    #     mask_gene_bins = (all_bin_gene_idx == g)
    #     # shape: [num_bins_in_gene, num_targets_strand]
    #     gb_targets = all_bin_targets_norm[mask_gene_bins, :]
    #     gb_preds   = all_bin_preds_norm[mask_gene_bins, :]

    #     if gb_targets.shape[0] == 0:
    #         # fallback
    #         gene_preds_norm_binlevel.append([np.nan]*num_targets_strand)
    #         gene_targets_norm_binlevel.append([np.nan]*num_targets_strand)
    #         continue

    #     # average across bins
    #     gb_targets_mean = gb_targets.mean(axis=0) / float(pool_width)
    #     gb_preds_mean   = gb_preds.mean(axis=0)   / float(pool_width)

    #     # scale by gene length
    #     gb_targets_mean *= gene_lengths[gene_id]
    #     gb_preds_mean   *= gene_lengths[gene_id]

    #     gene_preds_norm_binlevel.append(gb_preds_mean)
    #     gene_targets_norm_binlevel.append(gb_targets_mean)

    # gene_preds_norm_binlevel   = np.array(gene_preds_norm_binlevel)
    # gene_targets_norm_binlevel = np.array(gene_targets_norm_binlevel)

    # # NOTE: We have 3 sets of gene-level arrays now:
    # #   (a) gene_preds, gene_targets      : unnormalized coverage -> pseudo coverage -> log2
    # #   (b) gene_preds_norm, gene_targets_norm : gene-level quantile + mean-center
    # #   (c) gene_preds_norm_binlevel, gene_targets_norm_binlevel : bin-level QN + pooling

    # # If you prefer to remain consistent with your original approach (which QNs gene-level sums),
    # # you can skip using (c). If you want to do all correlation metrics from the bin-level
    # # normalization, then do log2 transform or not as you prefer. Right now, (c) is *already
    # # in log space* + mean-centering because that's how we handled it. Actually we quantile
    # # normalized log2(...) but did not necessarily add 1. It's up to you if you want to do
    # # an extra log transform or not. The code above is meant as a demonstration.

    # # -----------------------------------------------------------------------------
    # # 8) ACCURACY STATS (80th percentile threshold)
    # # -----------------------------------------------------------------------------
    # # wvar_t for unnormalized data
    # wvar_t = np.percentile(gene_wvar, 80, axis=0)
    # # wvar_t_norm for normalized bin-level data
    # wvar_t_norm = np.percentile(gene_wvar_norm, 80, axis=0)

    # acc_pearsonr   = []
    # acc_r2         = []
    # acc_npearsonr  = []
    # acc_nr2        = []
    # acc_wpearsonr  = []
    # acc_wpearsonr_norm = []

    # num_targets_strand = gene_targets.shape[1]  # or from earlier

    # for ti in range(num_targets_strand):
    #     # -- unnormalized gene-level (already log2 + pseudo coverage if any)
    #     r_ti = pearsonr(gene_targets[:, ti], gene_preds[:, ti])[0]
    #     acc_pearsonr.append(r_ti)
    #     r2_ti = explained_variance_score(gene_targets[:, ti], gene_preds[:, ti])
    #     acc_r2.append(r2_ti)

    #     # -- normalized gene-level (from step (6))
    #     nr_ti  = pearsonr(gene_targets_norm[:, ti], gene_preds_norm[:, ti])[0]
    #     nr2_ti = explained_variance_score(gene_targets_norm[:, ti], gene_preds_norm[:, ti])
    #     acc_npearsonr.append(nr_ti)
    #     acc_nr2.append(nr2_ti)

    #     # -- within-gene correlation (unnormalized), using 80th percentile
    #     var_mask = (gene_wvar[:, ti] > wvar_t[ti])
    #     wr_ti = np.nanmean(gene_within[var_mask, ti])
    #     acc_wpearsonr.append(wr_ti)

    #     # -- within-gene correlation (normalized), using 80th percentile
    #     var_mask_norm = (gene_wvar_norm[:, ti] > wvar_t_norm[ti])
    #     wr_ti_norm = np.nanmean(gene_within_norm[var_mask_norm, ti])
    #     acc_wpearsonr_norm.append(wr_ti_norm)

    # acc_df = pd.DataFrame({
    #     'identifier': targets_strand_df.identifier,
    #     'pearsonr': acc_pearsonr,
    #     'r2': acc_r2,
    #     'pearsonr_norm': acc_npearsonr,
    #     'r2_norm': acc_nr2,
    #     'pearsonr_gene': acc_wpearsonr,          # within-gene (unnorm)
    #     'pearsonr_gene_norm': acc_wpearsonr_norm,# within-gene (norm)
    #     'description': targets_strand_df.description,
    #     'group': targets_strand_df.group
    # })
    # acc_df.to_csv('%s/acc.txt' % options.out_dir, sep='\t', index=False)

    # # Print overall results
    # print('%d genes' % gene_targets.shape[0])
    # print('Overall results:')
    # print('  PearsonR (unnorm):           %.4f' % np.mean(acc_df.pearsonr))
    # print('  R2 (unnorm):                 %.4f' % np.mean(acc_df.r2))
    # print('  PearsonR (gene-level norm):  %.4f' % np.mean(acc_df.pearsonr_norm))
    # print('  R2 (gene-level norm):        %.4f' % np.mean(acc_df.r2_norm))
    # print('  Within-gene PearsonR (unnorm, 80th%%):    %.4f' % np.mean(acc_df.pearsonr_gene))
    # print('  Within-gene PearsonR (norm, 80th%%):      %.4f' % np.mean(acc_df.pearsonr_gene_norm))

    # for group in acc_df['group'].unique():
    #     group_df = acc_df[acc_df['group'] == group]
    #     print(f'\nResults for {group}:')
    #     print(f'  PearsonR:     {group_df.pearsonr.mean():.4f}')
    #     print(f'  R2:           {group_df.r2.mean():.4f}')
    #     print(f'  PearsonR_norm:  {group_df.pearsonr_norm.mean():.4f}')
    #     print(f'  R2_norm:        {group_df.r2_norm.mean():.4f}')
    #     print(f'  Within-gene PearsonR (unnorm): {group_df.pearsonr_gene.mean():.4f}')
    #     print(f'  Within-gene PearsonR (norm):   {group_df.pearsonr_gene_norm.mean():.4f}')
