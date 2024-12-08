#!/usr/bin/env python
# Copyright 2023 Calico LLC
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
import argparse
import json
import os
import pdb
import pickle

import h5py
import numpy as np
import pandas as pd
import pyBigWig
import tensorflow as tf

from baskerville import bed
from baskerville import dna
from baskerville import seqnn
from baskerville import stream

"""
hound_predbed.py

Predict sequences from a BED file.
"""


def main():
    parser = argparse.ArgumentParser(description="Predict sequences from a BED file.")
    parser.add_argument(
        "-b",
        "--bigwig_indexes",
        default=None,
        help="Comma-separated list of target indexes to write BigWigs",
    )
    parser.add_argument(
        "-e",
        "--embed_layer",
        default=None,
        type=int,
        help="Embed sequences using the specified layer index.",
    )
    parser.add_argument(
        "-f",
        "--genome_fasta",
        default=None,
        help="Genome FASTA for sequences [Default: %(default)s]",
    )
    parser.add_argument(
        "-g",
        "--genome_file",
        default=None,
        help="Chromosome length information [Default: %(default)s]",
    )
    parser.add_argument(
        "--head",
        default=0,
        type=int,
        help="Model head to evaluate [Default: %(default)s]",
    )
    parser.add_argument(
        "-l",
        "--site_length",
        default=None,
        type=int,
        help="Prediction site length. [Default: params.seq_length]",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="pred_out",
        help="Output directory [Default: %(default)s]",
    )
    parser.add_argument(
        "-p",
        "--processes",
        default=None,
        type=int,
        help="Number of processes, passed by multi script",
    )
    parser.add_argument(
        "--rc",
        default=False,
        action="store_true",
        help="Average the fwd and rc predictions [Default: %(default)s]",
    )
    parser.add_argument(
        "--save",
        default=False,
        action="store_true",
        help="Save predictions to file [Default: %(default)s]",
    )
    parser.add_argument(
        "-s",
        "--sum",
        default=False,
        action="store_true",
        help="Sum site predictions [Default: %(default)s]",
    )
    parser.add_argument(
        "--shifts",
        default="0",
        help="Ensemble prediction shifts [Default: %(default)s]",
    )
    parser.add_argument(
        "-t",
        "--targets_file",
        default=None,
        help="File specifying target indexes and labels in table format",
    )

    parser.add_argument("params_file", help="Parameters file")
    parser.add_argument("model_file", help="Model file")
    parser.add_argument("bed_file", help="BED file")
    args = parser.parse_args()

    if len(args) == 3:
        params_file = args[0]
        model_file = args[1]
        bed_file = args[2]

    elif len(args) == 5:
        # multi worker
        options_pkl_file = args[0]
        params_file = args[1]
        model_file = args[2]
        bed_file = args[3]
        worker_index = int(args[4])

        # load options
        options_pkl = open(options_pkl_file, "rb")
        options = pickle.load(options_pkl)
        options_pkl.close()

        # update output directory
        args.out_dir = "%s/job%d" % (args.out_dir, worker_index)
    else:
        parser.error("Must provide parameter and model files and BED file")

    os.makedirs(args.out_dir, exist_ok=True)

    args.shifts = [int(shift) for shift in args.shifts.split(",")]

    if args.bigwig_indexes is not None:
        args.bigwig_indexes = [int(bi) for bi in args.bigwig_indexes.split(",")]
    else:
        args.bigwig_indexes = []

    if len(args.bigwig_indexes) > 0:
        bigwig_dir = "%s/bigwig" % args.out_dir
        if not os.path.isdir(bigwig_dir):
            os.mkdir(bigwig_dir)

    #################################################################
    # read parameters and collet target information

    with open(params_file) as params_open:
        params = json.load(params_open)
    params_model = params["model"]

    if args.targets_file is None:
        target_slice = None
    else:
        targets_df = pd.read_table(args.targets_file, index_col=0)
        target_slice = targets_df.index

    #################################################################
    # setup model

    # initialize model
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_file, args.head)
    seqnn_model.build_slice(target_slice)
    seqnn_model.build_ensemble(args.rc, args.shifts)

    if args.embed_layer is not None:
        seqnn_model.build_embed(args.embed_layer)
    _, preds_length, preds_depth = seqnn_model.model.output.shape

    if type(preds_length) == tf.compat.v1.Dimension:
        preds_length = preds_length.value
        preds_depth = preds_depth.value

    preds_window = seqnn_model.model_strides[0]
    seq_crop = seqnn_model.target_crops[0] * preds_window

    #################################################################
    # sequence dataset

    if args.site_length is None:
        args.site_length = preds_window * preds_length
        print("site_length: %d" % args.site_length)

    # construct model sequences
    model_seqs_dna, model_seqs_coords = bed.make_bed_seqs(
        bed_file, args.genome_fasta, params_model["seq_length"], stranded=False
    )

    # construct site coordinates
    site_seqs_coords = bed.read_bed_coords(bed_file, args.site_length)

    # filter for worker SNPs
    if args.processes is not None:
        worker_bounds = np.linspace(
            0, len(model_seqs_dna), args.processes + 1, dtype="int"
        )
        model_seqs_dna = model_seqs_dna[
            worker_bounds[worker_index] : worker_bounds[worker_index + 1]
        ]
        model_seqs_coords = model_seqs_coords[
            worker_bounds[worker_index] : worker_bounds[worker_index + 1]
        ]
        site_seqs_coords = site_seqs_coords[
            worker_bounds[worker_index] : worker_bounds[worker_index + 1]
        ]

    num_seqs = len(model_seqs_dna)

    #################################################################
    # setup output

    if preds_length % 2 != 0:
        print("WARNING: preds_lengh is odd and therefore asymmetric.")
    preds_mid = preds_length // 2

    assert args.site_length % preds_window == 0
    site_preds_length = args.site_length // preds_window

    if site_preds_length % 2 != 0:
        print("WARNING: site_preds_length is odd and therefore asymmetric.")
    site_preds_start = preds_mid - site_preds_length // 2
    site_preds_end = site_preds_start + site_preds_length

    # initialize HDF5
    out_h5_file = "%s/predict.h5" % args.out_dir
    if os.path.isfile(out_h5_file):
        os.remove(out_h5_file)
    out_h5 = h5py.File(out_h5_file, "w")

    # create predictions
    if args.sum:
        out_h5.create_dataset("preds", dtype="float16", shape=(num_seqs, preds_depth))
    else:
        out_h5.create_dataset(
            "preds", dtype="float16", shape=(num_seqs, site_preds_length, preds_depth)
        )

    # store site coordinates
    site_seqs_chr, site_seqs_start, site_seqs_end = zip(*site_seqs_coords)
    site_seqs_chr = np.array(site_seqs_chr, dtype="S")
    site_seqs_start = np.array(site_seqs_start)
    site_seqs_end = np.array(site_seqs_end)
    out_h5.create_dataset("chrom", data=site_seqs_chr)
    out_h5.create_dataset("start", data=site_seqs_start)
    out_h5.create_dataset("end", data=site_seqs_end)

    #################################################################
    # predict scores, write output

    # define sequence generator
    def seqs_gen():
        for seq_dna in model_seqs_dna:
            yield dna.dna_1hot(seq_dna)

    # predict
    preds_stream = stream.PredStreamGen(
        seqnn_model, seqs_gen(), params["train"]["batch_size"]
    )

    for si in range(num_seqs):
        preds_seq = preds_stream[si]

        # slice site
        preds_site = preds_seq[site_preds_start:site_preds_end, :]

        # write
        if args.sum:
            out_h5["preds"][si] = preds_site.sum(axis=0)
        else:
            out_h5["preds"][si] = preds_site

        # write bigwig
        for ti in args.bigwig_indexes:
            bw_file = "%s/s%d_t%d.bw" % (bigwig_dir, si, ti)
            bigwig_write(
                preds_seq[:, ti],
                model_seqs_coords[si],
                bw_file,
                args.genome_file,
                seq_crop,
            )

    # close output HDF5
    out_h5.close()


def bigwig_open(bw_file, genome_file):
    """Open the bigwig file for writing and write the header."""

    bw_out = pyBigWig.open(bw_file, "w")

    chrom_sizes = []
    for line in open(genome_file):
        a = line.split()
        chrom_sizes.append((a[0], int(a[1])))

    bw_out.addHeader(chrom_sizes)

    return bw_out


def bigwig_write(signal, seq_coords, bw_file, genome_file, seq_crop=0):
    """Write a signal track to a BigWig file over the region
         specified by seqs_coords.

    Args
     signal:      Sequences x Length signal array
     seq_coords:  (chr,start,end)
     bw_file:     BigWig filename
     genome_file: Chromosome lengths file
     seq_crop:    Sequence length cropped from each side of the sequence.
    """
    target_length = len(signal)

    # open bigwig
    bw_out = bigwig_open(bw_file, genome_file)

    # initialize entry arrays
    entry_starts = []
    entry_ends = []

    # set entries
    chrm, start, end = seq_coords
    preds_pool = (end - start - 2 * seq_crop) // target_length

    bw_start = start + seq_crop
    for li in range(target_length):
        bw_end = bw_start + preds_pool
        entry_starts.append(bw_start)
        entry_ends.append(bw_end)
        bw_start = bw_end

    # add
    bw_out.addEntries(
        [chrm] * target_length,
        entry_starts,
        ends=entry_ends,
        values=[float(s) for s in signal],
    )

    bw_out.close()


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
