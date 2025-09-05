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
import math

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import tensorflow as tf
from tqdm import tqdm

from baskerville import bed
from baskerville import dataset
from baskerville import seqnn
from baskerville import trainer

"""
hound_eval_mlm_perplexity.py

Evaluate the accuracy, cross-entropy loss and perplexity of a masked language model on held-out sequences.
"""


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument(
        "-o",
        "--out_dir",
        default="eval_out",
        help="Output directory for evaluation statistics [Default: %(default)s]",
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
    parser.add_argument(
        "--tfr-root-dir",
        dest="tfr_root_dir",
        default="tfrecords",
        help="Root directory for TFR files [Default: %default]",
    )
    parser.add_argument(
        "--seq-bed",
        default=None,
        help="BED file with sequences to evaluate [Default: %(default)s]",
    )
    parser.add_argument(
        "--eval_dir",
        default=None,
        help="The directory to the validation data_dir/tfrecords [Default: %(default)s]",
    )
    parser.add_argument(
        "--diff-species-encoding",
        default=False,
        action="store_true",
        help="Use different species encoding [Default: %(default)s]",
    )

    parser.add_argument("params_file", help="JSON file with model parameters")
    parser.add_argument("model_file", help="Trained model HDF5.")
    parser.add_argument("data_dir", help="Train/valid/test data directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    #######################################################
    # inputs

    # read model parameters
    with open(args.params_file) as params_open:
        params = json.load(params_open)
    params_model = params["model"]
    params_train = params["train"]
    
    # get masking parameters
    mask_rate = params_train["mask_rate"]
    seq_length = params_model["seq_length"]    
    mask_size = int(mask_rate * seq_length)

    # read data parameters
    data_stats_file = "%s/statistics.json" % args.data_dir
    with open(data_stats_file) as data_stats_open:
        data_stats = json.load(data_stats_open)
    num_species = data_stats.get("num_species", 1)

    # set number of input features
    params_model["num_features"] = 4
    if params_train["loss"] == 'mlm':
        params_model["num_features"] = num_species + 5

    # construct eval data
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
        has_repeat_mask= params_train.get("has_repeat_mask", False),
        eval_dir= args.eval_dir
    )

    # initialize model
    seqnn_model = seqnn.SeqNN(params_model)
    print("Model summary: ", seqnn_model)
    seqnn_model.restore(args.model_file, 0)

    #######################################################
    # evaluate
    
    x_trues = []
    x_preds = []
    labels = []
    weight_scale = []
    
    columns = ['chrom', 'start', 'end', 'name', 'species']  # typical BED columns
    df = pd.read_csv(args.seq_bed, sep='\t', names=columns)
    # compute predictions
    for x_ix, x_tuple in enumerate(eval_data.dataset) :
        if x_ix % 64 == 0 :
            print('Evaluating sequence pattern = ' + str(x_ix), flush=True)
        
        x, label, exon_mask, repeat_mask = None, None, None, None
        if eval_data.has_mask and eval_data.has_repeat_mask:
            x, label, exon_mask, repeat_mask = x_tuple
        elif eval_data.has_mask:
            x, label, exon_mask = x_tuple
        elif eval_data.has_repeat_mask:
            x, label, repeat_mask = x_tuple
        else:
            x, label = x_tuple

        # get as numpy arrays
        x = x.numpy()
        label = label.numpy()

        # Make labels all zero because all species are not used for LM training
        if args.diff_species_encoding:
            label = np.zeros_like(label)
        
        if eval_data.has_mask:
            exon_mask = exon_mask.numpy()
        if eval_data.has_repeat_mask:
            repeat_mask = repeat_mask.numpy()

        do_rc = tf.cast(tf.random.uniform([x.shape[0]], minval=0, maxval=2, dtype=tf.int32), dtype=tf.bool)
        x = tf.where(
            do_rc[:, None, None],
            tf.reverse(x, axis=[1, 2]),
            x,
        )
        if exon_mask is not None:
            exon_mask = tf.where(
                do_rc[:, None],
                tf.reverse(exon_mask, axis=[1]),
                exon_mask,
            )
        if repeat_mask is not None:
            repeat_mask = tf.where(
                do_rc[:, None],
                tf.reverse(repeat_mask, axis=[1]),
                repeat_mask,
            )
        
        # optionally set position-specific loss weight scales from binary mask
        sw = None
        exon_loss_scale = params.get("train", None).get("exon_loss_scale", None)
        non_exon_loss_scale = params.get("train", None).get("non_exon_loss_scale", None)
        repeat_loss_scale = params.get("train", None).get("repeat_loss_scale", None)
        non_repeat_loss_scale = params.get("train", None).get("non_repeat_loss_scale", None)

        # exon_mask scaling
        if exon_mask is not None and exon_loss_scale is not None:
            sw = exon_mask * exon_loss_scale + (1 - exon_mask) * non_exon_loss_scale

        # repeat_mask scaling
        if repeat_mask is not None and repeat_loss_scale is not None:
            repeat_sw = repeat_mask * repeat_loss_scale + (1 - repeat_mask) * non_repeat_loss_scale
            if sw is None:
                sw = repeat_sw
            else:
                sw *= repeat_sw
        weight_scale.append(sw)
        
        # construct input pattern
        x_inp = np.concatenate([
            x,
            np.zeros((1, seq_length, 1)),
            np.tile(label, (1, seq_length, 1)),
        ], axis=-1)
        
        inds = np.arange(seq_length, dtype='int32')
        np.random.shuffle(inds)
        
        # potentially pad indices
        if seq_length % mask_size > 0:
            missing_n = mask_size - seq_length % mask_size
            missing_inds = np.arange(seq_length, dtype='int32')
            np.random.shuffle(missing_inds)
            inds = np.concatenate([inds, missing_inds[:missing_n]], axis=0)
        
        # initialize predictions
        x_pred = np.zeros(x.shape, dtype='float16')
        b_pred = np.zeros(seq_length, dtype='bool')
        
        # loop over indices to predict
        while inds.shape[0] > 0:
            ind = inds[:mask_size]
            inds = inds[mask_size:]
            
            # mask input
            x_masked = np.copy(x_inp)
            for j in ind.tolist():
                x_masked[0, j, :4] = 0.
                x_masked[0, j, 4] = 1.
            
            # predict
            yp = seqnn_model.model.predict(x=[x_masked], batch_size=1, verbose=False)[..., :4].astype('float16')
            
            # optionally make reverse-complement predictions and average
            if args.rc:
                x_masked_rc = np.concatenate([
                    x_masked[0, ...][:, :4][::-1, ::-1],
                    x_masked[0, ...][:, 4:][::-1, :],
                ], axis=-1)[None, ...]
                yp_rc = seqnn_model.model.predict(x=[x_masked_rc], batch_size=1, verbose=False)[..., :4].astype('float16')
                yp = (yp + yp_rc[:, ::-1, ::-1]) / 2.
            yp = yp.astype('float16')
            
            # fill in predictions at masked positions
            for j in ind.tolist():
                if not b_pred[j]:
                    x_pred[0, j, :] = yp[0, j, :]
                    b_pred[j] = True
        
        # accumulate predictions, targets and species labels
        x_trues.append(x)
        x_preds.append(x_pred)
        labels.append(np.argmax(label, axis=-1))
    
    # concatenate data
    x_true = np.concatenate(x_trues, axis=0).astype('float16')
    x_pred = np.concatenate(x_preds, axis=0).astype('float16')
    label = np.concatenate(labels, axis=0).astype('int32')
    weight_scale = np.concatenate(weight_scale, axis=0).astype('float32')
    
    # optionally save predictions
    if args.save:
        np.savez_compressed(
            "%s/preds_%s" % (args.out_dir, args.split),
            x_true=x_true,
            x_pred=x_pred,
            label=label,
            weight_scale=weight_scale
        )
    
    # finally compute test loss (categorical cross-entropy) per example
    eval_losses = np.zeros(x_true.shape[0], dtype='float32')
    eval_loss_per_species = np.zeros(eval_data.num_species, dtype='float32')
    evals_per_species = np.zeros(eval_data.num_species, dtype='int32')
    
    # loop over eval examples
    for i in range(x_true.shape[0]):
        # Compute cross-entropy loss per token and average over sequence
        ce_loss = -np.sum(x_true[i] * np.log(x_pred[i] + 1e-8), axis=-1)  # add epsilon for numerical stability
        # Multiply by weight scale (if provided) and then average over tokens
        eval_loss = np.mean(ce_loss * weight_scale[i])
        eval_losses[i] = eval_loss
        
        # accumulate per species using label as species id
        species_id = label[i]
        eval_loss_per_species[species_id] += eval_loss
        evals_per_species[species_id] += 1
    
    # average loss (cross-entropy)
    avg_ce_loss = np.mean(eval_losses)
    # Calculate overall perplexity (exp of cross-entropy)
    perplexity = np.exp(avg_ce_loss)
    
    # average loss per species
    for s in range(eval_data.num_species):
        if evals_per_species[s] > 0:
            eval_loss_per_species[s] = eval_loss_per_species[s] / float(evals_per_species[s])
        else:
            eval_loss_per_species[s] = 0.
    
    # Compute per-species perplexity as well
    perplexity_per_species = np.exp(eval_loss_per_species)
    
    # write species-level statistics including perplexity
    acc_df = pd.DataFrame({
        "species": np.arange(eval_loss_per_species.shape[0], dtype='int32'),
        "loss": eval_loss_per_species,
        "perplexity": perplexity_per_species,
        "n": evals_per_species,
    })
    
    acc_df.to_csv(
        "%s/acc_%s.txt" % (args.out_dir, args.split),
        sep="\t",
        index=False,
        float_format="%.5f"
    )
    
    print("Average Categorical Cross-Entropy loss = " + str(round(avg_ce_loss, 5)))
    print("Overall Perplexity = " + str(round(perplexity, 5)))


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
