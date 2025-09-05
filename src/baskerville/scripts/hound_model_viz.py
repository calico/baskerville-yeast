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
from tensorflow.keras.utils import plot_model

"""
hound_model_viz.py

Visualize the model
"""


def main():
    usage = "usage: %prog [options] <params_file> <model_file>"
    parser = OptionParser(usage)
    parser.add_option(
        "-o",
        dest="out_dir",
        default="sat_mut",
        help="Output directory [Default: %default]",
    )
    parser.add_option(
        "-n",
        dest="exp_name",
        default="model_test",
        help="Expreriment name [Default: %default]",
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
    (options, args) = parser.parse_args()

    if len(args) == 2:
        # single worker
        params_file = args[0]
        model_file = args[1]
    else:
        parser.error("Must provide parameter and model files and BED file")

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    options.shifts = [int(shift) for shift in options.shifts.split(",")]

    #################################################################
    # read parameters and targets

    # read model parameters
    with open(params_file) as params_open:
        params = json.load(params_open)
    params_model = params["model"]
    params_train = params["train"]
    if params_train["task"] == "fine-tune":
        num_species = 165
        params_model["num_features"] = num_species + 5
        params_train['r64_idx'] = 109

    #################################################################
    # setup model

    seqnn_model = seqnn.SeqNN(params_model)
    # seqnn_model.restore(model_file)
    # seqnn_model.build_ensemble(options.rc)

    plot_model(seqnn_model.model, show_dtype=True, show_layer_names=True, show_shapes=True,  
               to_file=f'{options.out_dir}/{options.exp_name}_model.png')

################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
