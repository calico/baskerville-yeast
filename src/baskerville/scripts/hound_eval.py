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

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import tensorflow as tf
from tqdm import tqdm

from sklearn.metrics import explained_variance_score

from baskerville import bed
from baskerville import dataset
from baskerville import seqnn
from baskerville import trainer

"""
hound_eval.py

Evaluate the accuracy of a trained model on held-out sequences.
"""


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument(
        "-b",
        "--bedgraph_indexes",
        help="Comma-separated list of target indexes to write predictions/targets as bedgraph [Default: %(default)s]",
    )
    parser.add_argument(
        "--head",
        dest="head_i",
        default=0,
        type=int,
        help="Parameters head to evaluate [Default: %(default)s]",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="eval_out",
        help="Output directory for evaluation statistics [Default: %(default)s]",
    )
    parser.add_argument(
        "--rank",
        default=False,
        action="store_true",
        help="Compute Spearman rank correlation [Default: %(default)s]",
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
        help="Save targets and predictions numpy arrays [Default: %(default)s]",
    )
    parser.add_argument(
        "--shifts",
        default="0",
        help="Ensemble prediction shifts [Default: %(default)s]",
    )
    parser.add_argument(
        "--step",
        default=1,
        type=int,
        help="Step across positions [Default: %(default)s]",
    )
    parser.add_argument(
        "-t",
        "--targets_file",
        default=None,
        help="File specifying target indexes and labels in table format",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "valid", "test"],
        help="Dataset split label for eg TFR pattern [Default: %(default)s]",
    )
    parser.add_argument(
        "--tfr_pattern",
        default=None,
        help="TFR pattern string appended to data_dir/tfrecords for subsetting [Default: %(default)s]",
    )

    parser.add_argument("params_file", help="JSON file with model parameters")
    parser.add_argument("model_file", help="Trained model HDF5.")
    parser.add_argument("data_dir", help="Train/valid/test data directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # parse shifts to integers
    args.shifts = [int(shift) for shift in args.shifts.split(",")]

    #######################################################
    # inputs

    # read targets
    if args.targets_file is None:
        args.targets_file = "%s/targets.txt" % args.data_dir
    targets_df = pd.read_csv(args.targets_file, index_col=0, sep="\t")

    # read model parameters
    with open(args.params_file) as params_open:
        params = json.load(params_open)
    params_model = params["model"]
    params_train = params["train"]

    num_species = 1
    # Get the number of species
    if params_train["task"] == "fine-tune":
        num_species = 165
    if params_train["task"] == "fine-tune":
        params_model["num_features"] = num_species + 5
        params_train['r64_idx'] = 109
    print("num_species: ", num_species)

    # set strand pairs
    if "strand_pair" in targets_df.columns:
        params_model["strand_pair"] = [np.array(targets_df.strand_pair)]

    # construct eval data
    eval_data = dataset.SeqDataset(
        args.data_dir,
        split_label=args.split,
        batch_size=params_train["batch_size"],
        mode="eval",
        tfr_pattern=args.tfr_pattern,
    )

    # initialize model
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(args.model_file, args.head_i)
    seqnn_model.build_ensemble(args.rc, args.shifts)

    #######################################################
    # evaluate
    loss_label = params_train.get("loss", "poisson").lower()
    spec_weight = params_train.get("spec_weight", 1)
    loss_fn = trainer.parse_loss(loss_label, spec_weight=spec_weight)

    # evaluate
    if params_train["task"] == "fine-tune":
        test_loss, test_metric1, test_metric2 = seqnn_model.evaluate_lm_fine_tuned(
            eval_data, params_train=params_train, num_species=num_species, loss_label=loss_label, loss_fn=loss_fn
        )
    else:
        test_loss, test_metric1, test_metric2 = seqnn_model.evaluate(
            eval_data, loss_label=loss_label, loss_fn=loss_fn
        )

    # print summary statistics
    print("\nTest Loss:         %7.5f" % test_loss)
    print("PearsonR:         %7.5f", test_metric1.mean())
    print("R2:         %7.5f", test_metric2.mean())

    # Save or merge into a DataFrame
    targets_acc_df = pd.DataFrame({
        "index": targets_df.index,
        "pearsonr": test_metric1,
        "r2": test_metric2,
        # "pearsonr_un": pearsonr_un,
        # "r2_un": r2_un,
        "identifier": targets_df.identifier,
        "description": targets_df.description
    })
    targets_acc_df.to_csv("%s/acc.txt" % args.out_dir, sep="\t", index=False, float_format="%.5f")

################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()