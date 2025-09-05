#!/usr/bin/env python
# Copyright 2017 Calico LLC
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

import json
import os

import h5py
import numpy as np
import pandas as pd

from baskerville import bed
from baskerville import dataset
from baskerville import dna
from baskerville import seqnn
from baskerville import snps

# Map nucleotides to indices
NUC_TO_INDEX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

"""
hound_ism_bed.py

Perform an in silico saturation mutagenesis of sequences in a BED file,
optionally using a SNP TSV file to mutate only specified alternate alleles.
"""

def main():
    usage = "usage: %prog [options] <params_file> <model_file> <bed_file>"
    parser = OptionParser(usage)
    parser.add_option(
        "-d",
        dest="mut_down",
        default=0,
        type="int",
        help="Nucleotides downstream of center sequence to mutate [Default: %default]",
    )
    parser.add_option(
        "-f",
        dest="genome_fasta",
        default=None,
        help="Genome FASTA for sequences [Default: %default]",
    )
    parser.add_option(
        "-l",
        dest="mut_len",
        default=0,
        type="int",
        help="Length of center sequence to mutate [Default: %default]",
    )
    parser.add_option(
        "-o",
        dest="out_dir",
        default="sat_mut",
        help="Output directory [Default: %default]",
    )
    parser.add_option(
        "-p",
        dest="processes",
        default=None,
        type="int",
        help="Number of processes, passed by multi script",
    )
    parser.add_option(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Ensemble forward and reverse complement predictions [Default: %default]",
    )
    parser.add_option(
        "--shifts",
        dest="shifts",
        default="0",
        help="Ensemble prediction shifts [Default: %default]",
    )
    parser.add_option(
        "--stats",
        dest="snp_stats",
        default="logSUM",
        help="Comma-separated list of stats to save. [Default: %default]",
    )
    parser.add_option(
        "-t",
        dest="targets_file",
        default=None,
        type="str",
        help="File specifying target indexes and labels in table format",
    )
    parser.add_option(
        "-u",
        dest="mut_up",
        default=0,
        type="int",
        help="Nucleotides upstream of center sequence to mutate [Default: %default]",
    )
    parser.add_option(
        "--untransform_old",
        dest="untransform_old",
        default=False,
        action="store_true",
        help="Untransform old models [Default: %default]",
    )
    parser.add_option(
        "-s",
        dest="snps_file",
        default=None,
        type="str",
        help="TSV file specifying SNPs with Reference and Alternate alleles",
    )
    (options, args) = parser.parse_args()

    if len(args) == 3:
        params_file = args[0]
        model_file = args[1]
        bed_file = args[2]
    else:
        parser.error("Must provide parameter and model files and BED file")

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    options.shifts = [int(shift) for shift in options.shifts.split(",")]
    options.snp_stats = [snp_stat for snp_stat in options.snp_stats.split(",")]

    if options.mut_up > 0 or options.mut_down > 0:
        options.mut_len = options.mut_up + options.mut_down
    else:
        assert options.mut_len > 0
        options.mut_up = options.mut_len // 2
        options.mut_down = options.mut_len - options.mut_up

    # read model parameters
    with open(params_file) as params_open:
        params = json.load(params_open)
    params_model = params["model"]
    params_train = params["train"]
    if params_train["task"] == "fine-tune":
        num_species = 165
        params_model["num_features"] = num_species + 5
        params_train['r64_idx'] = 109
    else:
        params_model["num_features"] = 4

    # read targets
    if options.targets_file is None:
        parser.error("Must provide targets file to clarify stranded datasets")
    targets_df = pd.read_csv(options.targets_file, sep="\t", index_col=0)

    # handle strand pairs
    if "strand_pair" in targets_df.columns:
        targets_strand_df = dataset.targets_prep_strand(targets_df)
        orig_new_index = dict(zip(targets_df.index, np.arange(targets_df.shape[0])))
        targets_strand_pair = np.array([
            orig_new_index[ti] for ti in targets_df.strand_pair
        ])
        params_model["strand_pair"] = [targets_strand_pair]
        strand_transform = dataset.make_strand_transform(targets_df, targets_strand_df)
    else:
        targets_strand_df = targets_df
        strand_transform = None
    num_targets = targets_strand_df.shape[0]

    # setup model
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_file)
    seqnn_model.build_slice(targets_df.index)
    seqnn_model.build_ensemble(options.rc)

    # read sequences from BED
    seqs_dna, seqs_coords = bed.make_bed_seqs(
        bed_file, options.genome_fasta, params_model["seq_length"], stranded=True
    )
    num_seqs = len(seqs_dna)

    # if SNP TSV provided, load and validate
    if options.snps_file:
        snps_df = pd.read_csv(options.snps_file, sep="\t")
        if snps_df.empty:
            parser.error("SNPs file is empty or not found")
        # load bed names to match SNP IDs
        bed_df = pd.read_csv(bed_file, sep="\t", header=None)
        if len(bed_df) != num_seqs:
            parser.error(
                f"Number of BED entries ({len(bed_df)}) does not match number of sequences ({num_seqs})"
            )

    # determine mutation region limits
    seq_mid = params_model["seq_length"] // 2
    mut_start = seq_mid - options.mut_up
    mut_end = mut_start + options.mut_len

    # setup output HDF5
    scores_h5_file = f"{options.out_dir}/scores.h5"
    if os.path.isfile(scores_h5_file):
        os.remove(scores_h5_file)
    scores_h5 = h5py.File(scores_h5_file, "w")
    scores_h5.create_dataset(
        "seqs",
        dtype="bool",
        shape=(num_seqs, options.mut_len, params_model["num_features"]),
    )
    for snp_stat in options.snp_stats:
        scores_h5.create_dataset(
            snp_stat,
            dtype="float16",
            shape=(num_seqs, options.mut_len, 4, num_targets),
        )

    # store coordinates
    scores_chr, scores_start, scores_end, scores_strand = [], [], [], []
    for seq_chr, seq_start, seq_end, seq_strand in seqs_coords:
        scores_chr.append(seq_chr)
        scores_strand.append(seq_strand)
        if seq_strand == "+":
            score_start = seq_start + mut_start
            score_end = score_start + options.mut_len
        else:
            score_end = seq_end - mut_start
            score_start = score_end - options.mut_len
        scores_start.append(score_start)
        scores_end.append(score_end)

    scores_h5.create_dataset("chr", data=np.array(scores_chr, dtype="S"))
    scores_h5.create_dataset("start", data=np.array(scores_start))
    scores_h5.create_dataset("end", data=np.array(scores_end))
    scores_h5.create_dataset("strand", data=np.array(scores_strand, dtype="S"))

    # iterate sequences
    for si, seq_dna in enumerate(seqs_dna):
        print(f"Predicting sequence {si}", flush=True)
        seq_chr, seq_start, seq_end, seq_strand = seqs_coords[si]
        print(
            f"Sequence {si} ({seq_chr}:{seq_start}-{seq_end}) strand {seq_strand}"
        )

        # one-hot encode reference
        if params_train["task"] == "fine-tune":
            ref_1hot = dna.dna_1hot_mask_species_encoding(
                seq_dna,
                num_species=num_species,
                species_index=params_train['r64_idx'],
            )
        else:
            ref_1hot = dna.dna_1hot(seq_dna)
        ref_1hot = np.expand_dims(ref_1hot, axis=0)

        # save sequence mask
        scores_h5["seqs"][si] = ref_1hot[0, mut_start:mut_end].astype("bool")

        # predict reference
        ref_preds = []
        for shift in options.shifts:
            ref_1hot_shift = dna.hot1_augment(ref_1hot, shift=shift)
            ref_preds.append(
                seqnn_model.predict_transform(
                    ref_1hot_shift,
                    targets_df,
                    strand_transform,
                    options.untransform_old,
                )
            )
        ref_preds = np.array(ref_preds)

        # perform SNP-specific or saturating mutagenesis
        if options.snps_file:
            snp_id = bed_df.iloc[si, 3]
            var_rows = snps_df[snps_df['SNP'] == snp_id]
            print("var_rows: ", var_rows)   
            if var_rows.shape[0] != 1:
                # raise ValueError(
                #     f"SNP ID mismatch for sequence {si}: '{snp_id}' found {var_rows.shape[0]} entries in SNPs file"
                # )
                var_rows = var_rows.iloc[0:1]
            var = var_rows.iloc[0]
            variant_pos = int(var['ChrPos'])
            ref_allele = var['Reference'].upper()
            alt_allele = var['Alternate'].upper()

            print("seq_start: ", seq_start, "; seq_end: ", seq_end)
            print("variant_pos: ", variant_pos)
            print("ref_allele: ", ref_allele, "; alt_allele: ", alt_allele)
            # determine local index of variant
            if seq_strand == "+":
                local_idx = variant_pos - seq_start - 1
            else:
                local_idx = seq_end - variant_pos - 1 + 1
            print("local_idx: ", local_idx)
            # validate reference base
            print("length of seq_dna: ", len(seq_dna))
            # seq_base = seq_dna[local_idx-3:local_idx+3]
            seq_base = seq_dna[local_idx]
            print("ref_allele: ", ref_allele, "; alt_allele: ", alt_allele)
            print(
                f"Reference base at seq {si} ({seq_chr}:{variant_pos}) is {seq_base}"
            )
            if seq_base.upper() != ref_allele:
                continue
                # raise ValueError(
                #     f"Reference allele mismatch at seq {si} ({seq_chr}:{variant_pos}): "
                #     f"expected {ref_allele}, found {seq_base}"
                # )

            # map alleles to indices
            ref_index = NUC_TO_INDEX[ref_allele]
            alt_index = NUC_TO_INDEX[alt_allele]

            # build alternate one-hot
            alt_1hot = np.copy(ref_1hot)
            alt_1hot[0, local_idx, :] = 0
            alt_1hot[0, local_idx, alt_index] = 1

            # predict alternate
            alt_preds = []
            for shift in options.shifts:
                alt_1hot_shift = dna.hot1_augment(alt_1hot, shift=shift)
                alt_preds.append(
                    seqnn_model.predict_transform(
                        alt_1hot_shift,
                        targets_df,
                        strand_transform,
                        options.untransform_old,
                    )
                )
            alt_preds = np.array(alt_preds)

            # compute and save SNP effect
            ism_scores = snps.compute_scores(
                ref_preds, alt_preds, options.snp_stats, None
            )
            var_ri = local_idx - mut_start
            for snp_stat in options.snp_stats:
                scores_h5[snp_stat][si, var_ri, alt_index] = ism_scores[snp_stat]
                print("ism_scores[snp_stat]: ", ism_scores[snp_stat])
        else:
            # original saturation mutagenesis
            for mi in range(mut_start, mut_end):
                for ni in range(4):
                    if ref_1hot[0, mi, ni] == 0:
                        alt_1hot = np.copy(ref_1hot)
                        alt_1hot[0, mi, :] = 0
                        alt_1hot[0, mi, ni] = 1

                        alt_preds = []
                        for shift in options.shifts:
                            alt_1hot_shift = dna.hot1_augment(alt_1hot, shift=shift)
                            alt_preds.append(
                                seqnn_model.predict_transform(
                                    alt_1hot_shift,
                                    targets_df,
                                    strand_transform,
                                    options.untransform_old,
                                )
                            )
                        alt_preds = np.array(alt_preds)

                        ism_scores = snps.compute_scores(
                            ref_preds, alt_preds, options.snp_stats, None
                        )
                        for snp_stat in options.snp_stats:
                            scores_h5[snp_stat][
                                si, mi - mut_start, ni
                            ] = ism_scores[snp_stat]

    # close HDF5
    scores_h5.close()


if __name__ == "__main__":
    main()
