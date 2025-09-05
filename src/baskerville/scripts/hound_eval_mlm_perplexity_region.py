#!/usr/bin/env python
# Copyright 2023 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
import argparse
import json
import os
import math

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import tensorflow as tf
from tqdm import tqdm

import pysam
import pyranges as pr

from baskerville import bed
from baskerville import dataset
from baskerville import seqnn
from baskerville import trainer

"""
hound_eval_mlm.py

Evaluate a trained masked language model on held-out sequences.
This script computes overall cross-entropy loss and perplexity, and—if a GTF file is provided—
computes region-specific metrics (for repeats, exonic, gene, and intergenic regions).
"""

def one_hot_encode(seq):
    """Convert a DNA sequence into one-hot encoding for A,C,G,T."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
    one_hot = np.zeros((len(seq), 4), dtype=np.float32)
    for i, nucleotide in enumerate(seq):
        if nucleotide in mapping:
            one_hot[i, mapping[nucleotide]] = 1.0
    return one_hot

def compute_complement(pr_obj, chrom_sizes):
    """
    Compute complement intervals (e.g. intergenic) from a PyRanges object.
    Return a DataFrame with columns [Chromosome, Start, End].
    """
    merged_pr = pr_obj.merge()
    merged_df = merged_pr.as_df()
    complement_list = []
    for chrom, size in chrom_sizes.items():
        chrom_df = merged_df[merged_df.Chromosome == chrom].sort_values("Start")
        current_start = 0
        if chrom_df.shape[0] == 0:
            complement_list.append({"Chromosome": chrom, "Start": 0, "End": size})
        else:
            for _, row in chrom_df.iterrows():
                if row["Start"] > current_start:
                    complement_list.append({"Chromosome": chrom,
                                            "Start": current_start,
                                            "End": row["Start"]})
                current_start = max(current_start, row["End"])
            if current_start < size:
                complement_list.append({"Chromosome": chrom,
                                        "Start": current_start,
                                        "End": size})
    return pd.DataFrame(complement_list)

def parse_gtf_annotations(gtf_file, chrom_sizes):
    """
    Parse a GTF file using pyranges and return a dictionary of annotations:
    { chrom: {"gene": [...], "exon": [...], "repeat": [...], "intergenic": [...]}, ... }
    """
    gtf_pr = pr.read_gtf(gtf_file)
    df = gtf_pr.as_df()

    # Extract gene and exon features
    gene_df = df[df.Feature == "gene"].copy()
    exon_df = df[df.Feature.isin(["exon", "CDS"])].copy()

    # **FIX** Example: If you want to treat lines with "repeat" in the "Feature" or "type" columns as repeats:
    repeat_df = df[df.Feature.str.contains("repeat", case=False, na=False)].copy()

    # Build a PyRanges for the gene intervals to get intergenic
    gene_pr = pr.PyRanges(gene_df)
    intergenic_df = compute_complement(gene_pr, chrom_sizes)
    intergenic_df["Feature"] = "intergenic"
    intergenic_df["gene_id"] = "intergenic"

    # Convert each chromosome’s annotation to a dictionary of intervals.
    annotations = {}
    for chrom in chrom_sizes.keys():
        gene_intervals = gene_df[gene_df.Chromosome == chrom][["Start", "End"]].to_records(index=False).tolist()
        exon_intervals = exon_df[exon_df.Chromosome == chrom][["Start", "End"]].to_records(index=False).tolist()
        # Intergenic intervals
        intergenic_intervals = intergenic_df[intergenic_df.Chromosome == chrom][["Start", "End"]].to_records(index=False).tolist()

        annotations[chrom] = {
            "gene": gene_intervals,
            "exon": exon_intervals,
            "intergenic": intergenic_intervals,
        }

    return annotations

def build_region_masks(genomic_start, seq_length, ann):
    """
    Given the genomic start coordinate and sequence length, and annotation intervals
    (dict with keys: "repeat", "exon", "gene", "intergenic"),
    return boolean masks for each region (length seq_length).
    """
    mask_repeat = np.zeros(seq_length, dtype=bool)
    mask_exon = np.zeros(seq_length, dtype=bool)
    mask_gene = np.zeros(seq_length, dtype=bool)
    mask_intergenic = np.zeros(seq_length, dtype=bool)

    for region in ["repeat", "exon", "gene", "intergenic"]:
        intervals = ann.get(region, [])
        for (start, end) in intervals:
            # Overlap between [genomic_start, genomic_start+seq_length) and [start, end)
            overlap_start = max(genomic_start, start)
            overlap_end = min(genomic_start + seq_length, end)
            if overlap_end > overlap_start:
                pos_start = overlap_start - genomic_start
                pos_end = overlap_end - genomic_start
                if region == "repeat":
                    mask_repeat[pos_start:pos_end] = True
                elif region == "exon":
                    mask_exon[pos_start:pos_end] = True
                elif region == "gene":
                    mask_gene[pos_start:pos_end] = True
                elif region == "intergenic":
                    mask_intergenic[pos_start:pos_end] = True

    return mask_repeat, mask_exon, mask_gene, mask_intergenic

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model with region-specific metrics.")
    parser.add_argument("-o", "--out_dir", default="eval_out",
                        help="Output directory [Default: %(default)s]")
    parser.add_argument("--rc", default=False, action="store_true",
                        help="Average fwd and reverse-complement predictions [Default: %(default)s]")
    parser.add_argument("--save", default=False, action="store_true",
                        help="Save targets and predictions arrays [Default: %(default)s]")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"],
                        help="Dataset split label [Default: %(default)s]")
    parser.add_argument("--tfr_pattern", default=None,
                        help="TFR pattern appended to data_dir/tfrecords [Default: %(default)s]")
    parser.add_argument("--tfr-root-dir", dest="tfr_root_dir", default="tfrecords",
                        help="Root directory for TFR files [Default: %(default)s]")
    parser.add_argument("--seq-bed", default=None,
                        help="BED file with sequences to evaluate [Default: %(default)s]")
    parser.add_argument("--eval_dir", default=None,
                        help="Directory of validation data_dir/tfrecords [Default: %(default)s]")
    parser.add_argument("--diff-species-encoding", default=False, action="store_true",
                        help="Use different species encoding [Default: %(default)s]")
    parser.add_argument("--gtf", default=None,
                        help="GTF file with genome annotations [Default: %(default)s]")

    parser.add_argument("params_file", help="JSON file with model parameters")
    parser.add_argument("model_file", help="Trained model HDF5 file")
    parser.add_argument("data_dir", help="Directory containing train/valid/test data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Read model parameters
    with open(args.params_file) as f:
        params = json.load(f)
    params_model = params["model"]
    params_train = params["train"]

    mask_rate = params_train["mask_rate"]
    seq_length = params_model["seq_length"]
    mask_size = int(mask_rate * seq_length)

    # Load data statistics
    data_stats_file = os.path.join(args.data_dir, "statistics.json")
    with open(data_stats_file) as f:
        data_stats = json.load(f)
    num_species = data_stats.get("num_species", 1)

    # **FIX** either read chrom_sizes from data_stats or define them. Example:
    chrom_sizes = data_stats.get("chrom_sizes", {})
    # If your data_stats does NOT have it, fallback to your test dictionary:
    if not chrom_sizes:
        chrom_sizes = {
            'I': 230218, 'II': 813184, 'III': 316620, 'IV': 1531933,
            'V': 576874, 'VI': 270161, 'VII': 1090940, 'VIII': 562643,
            'IX': 439888, 'X': 745751, 'XI': 666816, 'XII': 1078177,
            'XIII': 924431, 'XIV': 784333, 'XV': 1091291, 'XVI': 948066
        }

    # Adjust num_features
    params_model["num_features"] = 4
    if params_train["loss"] == "mlm":
        params_model["num_features"] = num_species + 5

    # Construct evaluation dataset
    eval_data = dataset.SeqDataset(
        args.data_dir,
        split_label=args.split,
        batch_size=1,
        mode="eval",
        tfr_pattern=args.tfr_pattern,
        tfr_root_dir=args.tfr_root_dir,
        has_targets=params_train.get("has_targets", True),
        has_label=params_train.get("has_label", False),
        has_mask=params_train.get("has_mask", False),
        has_repeat_mask=params_train.get("has_repeat_mask", False),
        eval_dir=args.eval_dir
    )

    # Initialize model
    seqnn_model = seqnn.SeqNN(params_model)
    print("Model summary:", seqnn_model)
    seqnn_model.restore(args.model_file, 0)

    # If GTF is provided, parse annotations
    if args.gtf is not None:
        annotations = parse_gtf_annotations(args.gtf, chrom_sizes)
    else:
        annotations = None

    # Prepare to store results
    x_trues = []
    x_preds = []
    labels = []
    weight_scale = []
    all_ce_losses = []  # cross-entropy per token, unweighted

    eval_dataset_list = list(eval_data.dataset)

    # Read BED for genomic coords
    bed_columns = ['chrom', 'start', 'end', 'name', 'species']
    df = pd.read_csv(args.seq_bed, sep='\t', names=bed_columns)

    if len(eval_dataset_list) != df.shape[0]:
        print("WARNING: number of dataset examples != number of BED entries.")

    for x_ix, x_tuple in enumerate(eval_dataset_list):
        if x_ix % 64 == 0:
            print("Evaluating sequence pattern =", x_ix, flush=True)

        # Unpack
        x, label_, exon_mask, repeat_mask = None, None, None, None
        if eval_data.has_mask and eval_data.has_repeat_mask:
            x, label_, exon_mask, repeat_mask = x_tuple
        elif eval_data.has_mask:
            x, label_, exon_mask = x_tuple
        elif eval_data.has_repeat_mask:
            x, label_, repeat_mask = x_tuple
        else:
            x, label_ = x_tuple

        x = x.numpy()              # shape [1, seq_length, 4]
        label_ = label_.numpy()    # shape [1, 1, #species?] or [1, #species?]

        # If using a special species encoding for your LM
        if args.diff_species_encoding:
            label_ = np.zeros_like(label_)

        if exon_mask is not None:
            exon_mask = exon_mask.numpy()   # shape [1, seq_length]
        if repeat_mask is not None:
            repeat_mask = repeat_mask.numpy()

        # Optionally do random reverse complement
        do_rc = tf.cast(
            tf.random.uniform([x.shape[0]], minval=0, maxval=2, dtype=tf.int32),
            dtype=tf.bool
        )
        x = tf.where(do_rc[:, None, None], tf.reverse(x, axis=[1, 2]), x)
        if exon_mask is not None:
            exon_mask = tf.where(do_rc[:, None], tf.reverse(exon_mask, axis=[1]), exon_mask)
        if repeat_mask is not None:
            repeat_mask = tf.where(do_rc[:, None], tf.reverse(repeat_mask, axis=[1]), repeat_mask)

        # Compute optional position-specific weighting
        sw = None
        exon_loss_scale = params.get("train", {}).get("exon_loss_scale", None)
        non_exon_loss_scale = params.get("train", {}).get("non_exon_loss_scale", None)
        repeat_loss_scale = params.get("train", {}).get("repeat_loss_scale", None)
        non_repeat_loss_scale = params.get("train", {}).get("non_repeat_loss_scale", None)

        if exon_mask is not None and exon_loss_scale is not None:
            sw = exon_mask * exon_loss_scale + (1 - exon_mask) * non_exon_loss_scale
        if repeat_mask is not None and repeat_loss_scale is not None:
            repeat_sw = repeat_mask * repeat_loss_scale + (1 - repeat_mask) * non_repeat_loss_scale
            if sw is None:
                sw = repeat_sw
            else:
                sw *= repeat_sw

        weight_scale.append(sw)  # shape [1, seq_length], appended for each example

        # Build model input
        x_inp = np.concatenate([
            x,                                  # [1, seq_length, 4]
            np.zeros((1, seq_length, 1)),       # mask channel
            np.tile(label_, (1, seq_length, 1)) # species label repeated along sequence
        ], axis=-1)                             # final shape [1, seq_length, 5 + #species?]

        # Mask positions in chunks
        inds = np.arange(seq_length, dtype='int32')
        np.random.shuffle(inds)
        if seq_length % mask_size > 0:
            missing_n = mask_size - (seq_length % mask_size)
            extra_inds = np.arange(seq_length, dtype='int32')
            np.random.shuffle(extra_inds)
            inds = np.concatenate([inds, extra_inds[:missing_n]], axis=0)

        x_pred = np.zeros_like(x, dtype='float16')  # [1, seq_length, 4]
        b_pred = np.zeros(seq_length, dtype=bool)

        while inds.shape[0] > 0:
            ind = inds[:mask_size]
            inds = inds[mask_size:]
            x_masked = np.copy(x_inp)
            for j in ind:
                # set the base channels to 0, mask channel=1
                x_masked[0, j, :4] = 0.
                x_masked[0, j, 4]  = 1.

            # Predict
            yp = seqnn_model.model.predict([x_masked], batch_size=1, verbose=False)[..., :4]
            yp = yp.astype('float16')

            # Optionally do RC prediction and average
            if args.rc:
                x_masked_rc = np.concatenate([
                    x_masked[0, ..., :4][::-1, ::-1],
                    x_masked[0, ..., 4:][::-1, :],
                ], axis=-1)[None, ...]
                yp_rc = seqnn_model.model.predict([x_masked_rc], batch_size=1, verbose=False)[..., :4]
                yp_rc = yp_rc.astype('float16')
                # average them
                yp = (yp + yp_rc[:, ::-1, ::-1]) / 2.

            # Fill in predictions
            for j in ind:
                if not b_pred[j]:
                    x_pred[0, j, :] = yp[0, j, :]
                    b_pred[j] = True

        # Store predictions
        x_trues.append(x)  # shape [1, seq_length, 4]
        x_preds.append(x_pred)  # shape [1, seq_length, 4]

        # For the species label, store the integer ID
        labels.append(np.argmax(label_, axis=-1))  # shape [1, 1] => store as integer

        # Compute unweighted cross-entropy per position:
        # sum over channel => shape [1, seq_length], then pick [0] => shape [seq_length]
        ce_loss = -np.sum(x * np.log(x_pred + 1e-8), axis=-1)[0]
        all_ce_losses.append(ce_loss)

    # Concatenate along batch dimension
    x_true = np.concatenate(x_trues, axis=0).astype('float16')  # [N, seq_length, 4]
    x_pred = np.concatenate(x_preds, axis=0).astype('float16')  # [N, seq_length, 4]
    label = np.concatenate(labels, axis=0).astype('int32')       # [N, 1] => [N]
    # If no weighting, sw might be None for some; fill with 1.0
    for i in range(len(weight_scale)):
        if weight_scale[i] is None:
            weight_scale[i] = np.ones((1, seq_length), dtype='float32')
    weight_scale = np.concatenate(weight_scale, axis=0).astype('float32')  # [N, seq_length]

    # Optionally save
    if args.save:
        np.savez_compressed(
            os.path.join(args.out_dir, f"preds_{args.split}.npz"),
            x_true=x_true,
            x_pred=x_pred,
            label=label,
            weight_scale=weight_scale
        )

    # Compute average loss & perplexity
    N = x_true.shape[0]
    eval_losses = np.zeros(N, dtype='float32')
    eval_loss_per_species = np.zeros(eval_data.num_species, dtype='float32')
    evals_per_species = np.zeros(eval_data.num_species, dtype='int32')

    for i in range(N):
        ce_loss = all_ce_losses[i]               # shape [seq_length]
        sw = weight_scale[i]                     # shape [seq_length]
        # Weighted average cross-entropy for this example
        numerator = np.sum(ce_loss * sw)
        denominator = np.sum(sw)
        example_loss = numerator / (denominator + 1e-8)
        eval_losses[i] = example_loss

        s_id = label[i]  # species
        eval_loss_per_species[s_id] += example_loss
        evals_per_species[s_id] += 1

    avg_ce_loss = np.mean(eval_losses)
    overall_perplexity = math.exp(avg_ce_loss)

    # Final loss per species
    for s in range(eval_data.num_species):
        if evals_per_species[s] > 0:
            eval_loss_per_species[s] /= float(evals_per_species[s])
        else:
            eval_loss_per_species[s] = 0.
    perplexity_per_species = np.exp(eval_loss_per_species)

    # Write per-species to file
    acc_df = pd.DataFrame({
        "species": np.arange(eval_loss_per_species.shape[0], dtype='int32'),
        "loss": eval_loss_per_species,
        "perplexity": perplexity_per_species,
        "n": evals_per_species,
    })
    acc_df.to_csv(
        os.path.join(args.out_dir, f"acc_{args.split}.txt"),
        sep="\t",
        index=False,
        float_format="%.5f"
    )

    print("Average Categorical Cross-Entropy loss =", round(avg_ce_loss, 5))
    print("Overall Perplexity =", round(overall_perplexity, 5))

    # If we have GTF annotations, compute region metrics
    if annotations is not None:
        region_total_loss = {"repeat": 0.0, "exon": 0.0, "gene": 0.0, "intergenic": 0.0}
        region_token_sum = {"repeat": 0.0, "exon": 0.0, "gene": 0.0, "intergenic": 0.0}

        for i in range(N):
            bed_row = df.iloc[i]
            # Safely remove "chr" prefix
            chrom = str(bed_row['chrom'])
            if chrom.startswith("chr"):
                chrom = chrom[3:]
            bed_start = int(bed_row['start'])

            ce_loss = all_ce_losses[i]     # shape [seq_length]
            sw = weight_scale[i]           # shape [seq_length]

            # Build masks for the region
            if chrom not in annotations:
                # skip if no annotation for this chrom
                continue
            ann = annotations[chrom]
            mask_repeat, mask_exon, mask_gene, mask_intergenic = build_region_masks(bed_start, seq_length, ann)

            # Weighted sum of cross-entropy for each region
            for region, mask in zip(["repeat", "exon", "gene", "intergenic"],
                                    [mask_repeat, mask_exon, mask_gene, mask_intergenic]):
                if np.any(mask):
                    region_total_loss[region] += np.sum(ce_loss[mask] * sw[mask])
                    region_token_sum[region] += np.sum(sw[mask])

        # Compute region-level averages and perplexities
        region_avg_loss = {}
        region_perplexity = {}
        for region in region_total_loss:
            if region_token_sum[region] > 0:
                avg_loss = region_total_loss[region] / region_token_sum[region]
                region_avg_loss[region] = avg_loss
                region_perplexity[region] = math.exp(avg_loss)
            else:
                region_avg_loss[region] = None
                region_perplexity[region] = None

        region_stats_df = pd.DataFrame({
            "region": list(region_avg_loss.keys()),
            "avg_loss": list(region_avg_loss.values()),
            "perplexity": list(region_perplexity.values()),
            "weighted_tokens": list(region_token_sum.values()),
        })
        region_stats_df.to_csv(
            os.path.join(args.out_dir, f"region_stats_{args.split}.txt"),
            sep="\t",
            index=False,
            float_format="%.5f"
        )
        print("Region-specific metrics:")
        print(region_stats_df)

if __name__ == "__main__":
    main()
