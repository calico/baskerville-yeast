# Copyright 2023 Calico LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
import glob
import json
import pdb
import sys, os

from natsort import natsorted
import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# TFRecord constants
TFR_INPUT = "sequence"
TFR_OUTPUT = "target"
TFR_LABEL = "species"
TFR_MASK = "mask"
TFR_REPEAT_MASK = "repeat_mask"

def file_to_records(filename: str):
    """Read TFRecord file into tf.data.Dataset."""
    return tf.data.TFRecordDataset(filename, compression_type="ZLIB")


class SeqDataset:
    """Labeled sequence dataset for Tensorflow.

    Args:
      data_dir (str): Dataset directory.
      split_label (str): Dataset split, e.g. train, valid, test.
      batch_size (int): Batch size.
      shuffle_buffer (int): Shuffle buffer size. Defaults to 128.
      seq_length_crop (int): Sequence length to crop from sides. Defaults to 0.
      mode (str): Dataset mode, e.g. train/eval. Defaults to 'eval'.
      tfr_pattern (str): TFRecord pattern to glob. Defaults to split_label.
      targets_slice_file (str): Targets table from which to slice a target subset.
    """

    def __init__(
        self,
        data_dir: str,
        split_label: str,
        batch_size: int,
        shuffle_buffer: int = 128,
        seq_length_crop: int = 0,
        mode: str = "eval",
        tfr_pattern: str = None,
        targets_slice_file: str = None,
        shuffle_records: bool = False,
        has_targets: bool = True,
        has_label: bool = False,
        has_mask: bool = False,
        has_repeat_mask: bool = False,
        eval_dir: str = "",
        global_eval: bool = False
    ):
        self.data_dir = data_dir
        self.split_label = split_label
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.seq_length_crop = seq_length_crop
        self.mode = mode
        self.tfr_pattern = tfr_pattern
        self.shuffle_records = shuffle_records
        self.has_targets = has_targets
        self.has_label = has_label
        self.has_mask = has_mask
        self.has_repeat_mask = has_repeat_mask
        self.eval_dir = eval_dir

        print("self.data_dir: ", self.data_dir)
        print("self.split_label: ", self.split_label)   
        print("self.batch_size: ", self.batch_size) 
        print("self.shuffle_buffer: ", self.shuffle_buffer) 
        print("self.seq_length_crop: ", self.seq_length_crop)
        print("self.mode: ", self.mode) 
        print("self.tfr_pattern: ", self.tfr_pattern)
        print("self.shuffle_records: ", self.shuffle_records)
        print("self.has_targets: ", self.has_targets)
        print("self.has_label: ", self.has_label)
        print("self.has_mask: ", self.has_mask)
        print("self.has_repeat_mask: ", self.has_repeat_mask)
        print("self.eval_dir: ", self.eval_dir)

        # self.eval_dir = "/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/"
        # read data parameters
        data_stats_file = "%s/statistics.json" % self.data_dir
        with open(data_stats_file) as data_stats_open:
            data_stats = json.load(data_stats_open)
        self.seq_length = data_stats["seq_length"]

        # set defaults
        self.seq_depth = data_stats.get("seq_depth", 4)
        self.seq_1hot = data_stats.get("seq_1hot", False)
        self.target_length = data_stats.get("target_length", 1)
        self.num_targets = data_stats.get("num_targets", 1)
        self.pool_width = data_stats.get("pool_width", 1)
        self.num_species = data_stats.get("num_species", 1)

        # slice targets
        if targets_slice_file is None:
            self.targets_slice = None
        else:
            targets_df = pd.read_csv(targets_slice_file, index_col=0, sep="\t")
            self.targets_slice = np.array(targets_df.index)

        if global_eval == True:
            if self.tfr_pattern is None:
                # self.tfr_path = "%s/tfrecords/%s-*.tfr" % (self.data_dir, self.split_label)
                self.tfr_path = "%s/%s_set/%s-*.tfr" % (self.eval_dir, self.split_label, self.split_label)
                self.num_seqs = data_stats["%s_seqs" % self.split_label]
            else:
                # self.tfr_path = "%s/tfrecords/%s-*.tfr" % (self.data_dir, self.split_label)
                self.tfr_path = "%s/%s_set/%s" % (self.eval_dir, self.split_label, self.split_label)
                self.compute_stats()
        else:
            # extract or compute sequence statistics
            if self.tfr_pattern is None:
                self.tfr_path = "%s/tfrecords/%s-*.tfr" % (self.data_dir, self.split_label)
                self.num_seqs = data_stats["%s_seqs" % self.split_label]
            else:
                self.tfr_path = "%s/tfrecords/%s" % (self.data_dir, self.tfr_pattern)
                self.compute_stats()
        # make tf.data.Dataset object
        self.make_dataset()

    def batches_per_epoch(self):
        """Compute number of batches per epoch."""
        return self.num_seqs // self.batch_size

    def distribute(self, strategy):
        """Wrap Dataset to distribute across devices."""
        self.dataset = strategy.experimental_distribute_dataset(self.dataset)

    def generate_parser(self, raw: bool = False):
        """Generate parser function for TFRecordDataset."""

        def parse_proto(example_protos):
            """Parse TFRecord protobuf."""

            # define features
            features = {
                TFR_INPUT: tf.io.FixedLenFeature([], tf.string),
            }
            if self.has_targets:
                features[TFR_OUTPUT] = tf.io.FixedLenFeature([], tf.string)
            if self.has_label:
                features[TFR_LABEL] = tf.io.FixedLenFeature([], tf.string)
            if self.has_mask:
                features[TFR_MASK] = tf.io.FixedLenFeature([], tf.string)
            if self.has_repeat_mask:
                features[TFR_REPEAT_MASK] = tf.io.FixedLenFeature([], tf.string)

            # parse example into features
            parsed_features = tf.io.parse_single_example(
                example_protos, features=features
            )

            # decode sequence
            sequence = tf.io.decode_raw(parsed_features[TFR_INPUT], tf.uint8)
            if not raw:
                if self.seq_1hot:
                    sequence = tf.reshape(sequence, [self.seq_length])
                    sequence = tf.one_hot(sequence, 1 + self.seq_depth, dtype=tf.uint8)
                    sequence = sequence[:, :-1]  # drop N
                else:
                    sequence = tf.reshape(sequence, [self.seq_length, self.seq_depth])
                if self.seq_length_crop > 0:
                    crop_len = (self.seq_length - self.seq_length_crop) // 2
                    sequence = sequence[crop_len:-crop_len, :]
                sequence = tf.cast(sequence, tf.float32)

            # decode targets
            if self.has_targets:
                targets = tf.io.decode_raw(parsed_features[TFR_OUTPUT], tf.float16)
                if not raw:
                    targets = tf.reshape(targets, [self.target_length, self.num_targets])
                    targets = tf.cast(targets, tf.float32)
                    if self.targets_slice is not None:
                        targets = targets[:, self.targets_slice]
            
            # decode binary label
            if self.has_label:
                label = tf.io.decode_raw(parsed_features[TFR_LABEL], tf.int32)
                if not raw:
                    label = tf.reshape(label, [1])
                    label = tf.one_hot(label, self.num_species, dtype=tf.int32)
                label = tf.cast(label, tf.float32)
            
            # decode binary mask
            if self.has_mask:
                mask = tf.io.decode_raw(parsed_features[TFR_MASK], tf.uint8)
                if not raw:
                    mask = tf.reshape(mask, [self.seq_length])
                mask = tf.cast(mask, tf.float32)

            # decode binary mask
            if self.has_repeat_mask:
                repeat_mask = tf.io.decode_raw(parsed_features[TFR_REPEAT_MASK], tf.uint8)
                if not raw:
                    repeat_mask = tf.reshape(repeat_mask, [self.seq_length])
                repeat_mask = tf.cast(repeat_mask, tf.float32)

            ret_tuple = [sequence]
            if self.has_targets:
                ret_tuple.append(targets)
            if self.has_label:
                ret_tuple.append(label)
            if self.has_mask:
                ret_tuple.append(mask)
            if self.has_repeat_mask:
                ret_tuple.append(repeat_mask)
            
            return ret_tuple

        return parse_proto

    def make_dataset(self, cycle_length=4):
        """Make tf.data.Dataset w/ transformations."""

        # initialize dataset from TFRecords glob
        tfr_files = natsorted(glob.glob(self.tfr_path))
    
        # optionally shuffle tfr record files
        if self.shuffle_records:
            tfr_shuffle_index = np.arange(len(tfr_files), dtype='int32')
      
            #rng = np.random.RandomState(42)
            #rng.shuffle(tfr_shuffle_index)
            np.random.shuffle(tfr_shuffle_index)
      
            tfr_files = [tfr_files[tfr_shuffle_index[i]] for i in range(len(tfr_files))]
        
        if tfr_files:
            dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
        else:
            print("Cannot order TFRecords %s" % self.tfr_path, file=sys.stderr)
            dataset = tf.data.Dataset.list_files(self.tfr_path)

        # train
        if self.mode == "train":
            # repeat
            dataset = dataset.repeat()

            # interleave files
            dataset = dataset.interleave(
                map_func=file_to_records,
                cycle_length=cycle_length,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )

            # shuffle
            dataset = dataset.shuffle(
                buffer_size=self.shuffle_buffer, reshuffle_each_iteration=True
            )

        # valid/test
        else:
            # flat mix files
            dataset = dataset.flat_map(file_to_records)

        # map parser across files
        dataset = dataset.map(self.generate_parser())

        # batch
        dataset = dataset.batch(self.batch_size)

        # prefetch
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        # hold on
        self.dataset = dataset

    def compute_stats(self):
        """Iterate over the TFRecords to count sequences, and infer
        seq_depth and num_targets."""
        with tf.name_scope("stats"):
            # read TF Records
            dataset = tf.data.Dataset.list_files(self.tfr_path)
            dataset = dataset.flat_map(file_to_records)
            dataset = dataset.map(self.generate_parser(raw=True))
            dataset = dataset.batch(1)

        self.num_seqs = 0
        if self.num_targets is not None:
            targets_nonzero = np.zeros(self.num_targets, dtype="bool")

        for raw_tuple in dataset:
            
            seq_raw = raw_tuple[0]
            
            if self.has_targets:
                targets_raw = raw_tuple[1]
            
            if self.has_label:
                label_raw = raw_tuple[2]
            
            # infer seq_depth
            seq_1hot = seq_raw.numpy().reshape((self.seq_length, -1))

            if self.has_targets:
                # infer num_targets
                targets1 = targets_raw.numpy().reshape(self.target_length, -1)
                if self.num_targets is None:
                    self.num_targets = targets1.shape[-1]
                    targets_nonzero = (targets1 != 0).sum(axis=0) > 0
                else:
                    assert self.num_targets == targets1.shape[-1]
                    targets_nonzero = np.logical_or(
                        targets_nonzero, (targets1 != 0).sum(axis=0) > 0
                    )
            elif self.num_targets is None:
                self.num_targets = 0
                targets_nonzero = 0

            # count sequences
            self.num_seqs += 1

        # warn user about nonzero targets
        if self.num_seqs > 0:
            self.num_targets_nonzero = (targets_nonzero > 0).sum()
            print(
                "%s has %d sequences with %d/%d targets"
                % (
                    self.tfr_path,
                    self.num_seqs,
                    self.num_targets_nonzero,
                    self.num_targets,
                ),
                flush=True,
            )
        else:
            self.num_targets_nonzero = None
            print(
                "%s has %d sequences with 0 targets" % (self.tfr_path, self.num_seqs),
                flush=True,
            )

    def numpy(
        self,
        return_inputs=True,
        return_outputs=True,
        return_labels=False,
        return_masks=False,
        return_repeat_masks=False,
        step=1,
        target_slice=None,
        dtype="float16",
    ):
        """Convert TFR inputs and/or outputs to numpy arrays."""
        with tf.name_scope("numpy"):
            # initialize dataset from TFRecords glob
            tfr_files = natsorted(glob.glob(self.tfr_path))
            if tfr_files:
                # dataset = tf.data.Dataset.list_files(tf.constant(tfr_files), shuffle=False)
                dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
            else:
                print("Cannot order TFRecords %s" % self.tfr_path, file=sys.stderr)
                dataset = tf.data.Dataset.list_files(self.tfr_path)

            # read TF Records
            dataset = dataset.flat_map(file_to_records)
            dataset = dataset.map(self.generate_parser(raw=True))
            dataset = dataset.batch(1)

        # initialize inputs, outputs and label
        seqs_1hot = []
        targets = []
        labels = []
        masks = []
        repeat_masks = []

        # collect inputs and outputs
        for raw_tuple in dataset:
            
            seq_raw = raw_tuple[0]
            
            targets_raw, label_raw, mask_raw, repeat_mask_raw = None, None, None, None
            
            if self.has_targets :
                targets_raw = raw_tuple[1]
            
                if self.has_label :
                    label_raw = raw_tuple[2]
            
                    if self.has_mask:
                        mask_raw = raw_tuple[3]
                        if self.has_repeat_mask:
                            repeat_mask_raw = raw_tuple[4]
                else :
                    if self.has_mask:
                        mask_raw = raw_tuple[2]
                        if self.has_repeat_mask:
                            repeat_mask_raw = raw_tuple[3]
            else :
                if self.has_label :
                    label_raw = raw_tuple[1]
            
                    if self.has_mask:
                        mask_raw = raw_tuple[2]
                        if self.has_repeat_mask:
                            repeat_mask_raw = raw_tuple[3]
                else :
                    if self.has_mask:
                        mask_raw = raw_tuple[1]
                        if self.has_repeat_mask:
                            repeat_mask_raw = raw_tuple[2]
            
            # sequence
            if return_inputs:
                seq_1hot = seq_raw.numpy().reshape((self.seq_length, -1))
                if self.seq_length_crop > 0:
                    crop_len = (self.seq_length - self.seq_length_crop) // 2
                    seq_1hot = seq_1hot[crop_len:-crop_len, :]
                seqs_1hot.append(seq_1hot)

            # targets
            if return_outputs:
                targets1 = targets_raw.numpy().astype(dtype)
                targets1 = np.reshape(targets1, (self.target_length, -1))
                if target_slice is not None:
                    targets1 = targets1[:, target_slice]
                if step > 1:
                    step_i = np.arange(0, self.target_length, step)
                    targets1 = targets1[step_i, :]
                targets.append(targets1)

            # labels
            if return_labels:
                label = label_raw.numpy().astype('int32')
                label = np.reshape(label, (1,))
                labels.append(label)
            
            # mask
            if return_masks:
                mask = mask_raw.numpy().reshape((self.seq_length,))
                if self.seq_length_crop > 0:
                    crop_len = (self.seq_length - self.seq_length_crop) // 2
                    mask = mask[crop_len:-crop_len]
                masks.append(mask)

            # mask
            if return_repeat_masks:
                repeat_mask = repeat_mask_raw.numpy().reshape((self.seq_length,))
                if self.seq_length_crop > 0:
                    crop_len = (self.seq_length - self.seq_length_crop) // 2
                    repeat_mask = repeat_mask[crop_len:-crop_len]
                repeat_masks.append(repeat_mask)

        # make arrays
        seqs_1hot = np.array(seqs_1hot)
        targets = np.array(targets, dtype=dtype)
        labels = np.array(labels, dtype='int32')
        masks = np.array(masks)
        repeat_masks = np.array(repeat_masks)

        # return bundle
        ret_tuple = []
        if return_inputs :
            ret_tuple.append(seqs_1hot)
        if return_outputs :
            ret_tuple.append(targets)
        if return_labels :
            ret_tuple.append(labels)
        if return_masks :
            ret_tuple.append(masks)
        if return_repeat_masks :
            ret_tuple.append(repeat_masks)
        
        return ret_tuple


def make_strand_transform(targets_df, targets_strand_df):
    """Make a sparse matrix to sum strand pairs.

    Args:
        targets_df (pd.DataFrame): Targets DataFrame.
        targets_strand_df (pd.DataFrame): Targets DataFrame, with strand pairs collapsed.

    Returns:
        scipy.sparse.dok_matrix: Sparse matrix to sum strand pairs.
    """

    # initialize sparse matrix
    strand_transform = dok_matrix((targets_df.shape[0], targets_strand_df.shape[0]))

    # fill in matrix
    ti = 0
    sti = 0
    for _, target in targets_df.iterrows():
        strand_transform[ti, sti] = True
        if target.strand_pair == target.name:
            sti += 1
        else:
            if target.identifier[-1] == "-":
                sti += 1
        ti += 1

    return strand_transform


def targets_prep_strand(targets_df):
    """Adjust targets table for merged stranded datasets.

    Args:
        targets_df: pandas DataFrame of targets

    Returns:
        targets_df: pandas DataFrame of targets, with stranded
            targets collapsed into a single row
    """
    # attach strand
    targets_strand = []
    for _, target in targets_df.iterrows():
        # print("target: ", target)
        # if target.strand_pair == target.name:
        #     targets_strand.append(".")
        # else:
        #     targets_strand.append(target.identifier[-1])

        targets_strand.append(".")
    targets_df["strand"] = targets_strand

    # collapse stranded
    strand_mask = targets_df.strand != "-"
    targets_strand_df = targets_df[strand_mask]

    return targets_strand_df


def untransform_preds(preds, targets_df, unscale=False, unclip=True):
    """Undo the squashing transformations performed for the tasks.

    Args:
      preds (np.array): Predictions LxT.
      targets_df (pd.DataFrame): Targets information table.

    Returns:
      preds (np.array): Untransformed predictions LxT.
    """
    # clip soft
    if unclip:
        cs = np.expand_dims(np.array(targets_df.clip_soft), axis=0)
        preds_unclip = cs - 1 + (preds - cs + 1) ** 2
        preds = np.where(preds > cs, preds_unclip, preds)

    # sqrt
    sqrt_mask = np.array([ss.find("_sqrt") != -1 for ss in targets_df.sum_stat])
    preds[:, sqrt_mask] = -1 + (preds[:, sqrt_mask] + 1) ** 2  # (4 / 3)

    # scale
    if unscale:
        scale = np.expand_dims(np.array(targets_df.scale), axis=0)
        preds = preds / scale

    return preds


def untransform_preds1(preds, targets_df, unscale=False, unclip=True):
    """Undo the squashing transformations performed for the tasks.

    Args:
      preds (np.array): Predictions LxT.
      targets_df (pd.DataFrame): Targets information table.

    Returns:
      preds (np.array): Untransformed predictions LxT.
    """
    # scale
    scale = np.expand_dims(np.array(targets_df.scale), axis=0)
    preds = preds / scale

    # clip soft
    if unclip:
        cs = np.expand_dims(np.array(targets_df.clip_soft), axis=0)
        preds_unclip = cs + (preds - cs) ** 2
        preds = np.where(preds > cs, preds_unclip, preds)

    # ** 0.75
    sqrt_mask = np.array([ss.find("_sqrt") != -1 for ss in targets_df.sum_stat])
    preds[:, sqrt_mask] = (preds[:, sqrt_mask]) ** (4 / 3)

    # unscale
    if not unscale:
        preds = preds * scale

    return preds
