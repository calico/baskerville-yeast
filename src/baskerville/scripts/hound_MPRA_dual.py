#!/usr/bin/env python
# Copyright 2022 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
#!/usr/bin/env python
# Copyright 2022 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from __future__ import print_function

from optparse import OptionParser
from collections import OrderedDict
import json
import pickle
import os
import sys
import time
from tqdm import tqdm

import h5py
import numpy as np
import pandas as pd
import pybedtools
import pysam
from scipy.special import rel_entr
import tensorflow as tf

from baskerville.gene import Transcriptome
from baskerville import dataset
from baskerville import seqnn
from baskerville import vcf as bvcf
from baskerville import dna

'''
borzoi_sed_replace.py

Compute Expression Difference (SED) scores for sequences in a tsv file,
relative to gene exons in a GTF file,
when substituting them in place of native genomic context specified in another tsv file.

Now also compute each inserted ALT or REF sequence against the original
genomic context separately, storing logSED_ALT_ORIG and logSED_REF_ORIG
if requested.
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <params_file> <model_file> <tsv_file>'
    parser = OptionParser(usage)
    parser.add_option(
        "-b",
        dest="bedgraph",
        default=False,
        action="store_true",
        help="Write ref/alt predictions as bedgraph [Default: %default]",
    )
    parser.add_option(
        "-f",
        dest="genome_fasta",
        default="%s/assembly/ucsc/hg38.fa" % os.environ.get('BORZOI_HG38', 'hg38'),
        help="Genome FASTA for sequences [Default: %default]",
    )
    parser.add_option(
        "-g",
        dest="genes_gtf",
        default="%s/genes/gencode41/gencode41_basic_nort.gtf" % os.environ.get('BORZOI_HG38', 'hg38'),
        help="GTF for gene definition [Default %default]",
    )
    parser.add_option(
        '--ctx',
        dest='ctx_tsv',
        default='/home/jlinder/seqnn/data/enhancers/unique_contexts_k562.tsv',
        help='TSV file containing genomic contexts to insert sequence into [Default %default]'
    )
    parser.add_option(
        "-o",
        dest="out_dir",
        default="sed",
        help="Output directory for tables and plots [Default: %default]",
    )
    parser.add_option(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Average forward and reverse complement predictions [Default: %default]",
    )
    parser.add_option(
        "--shifts",
        dest="shifts",
        default="0",
        type="str",
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
        "--stats",
        dest="sed_stats",
        default="SED",
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
        dest="untransform_old",
        default=False,
        action="store_true",
    )
    parser.add_option(
        "--no_untransform",
        dest="no_untransform",
        default=False,
        action="store_true",
    )
    parser.add_option(
        '--no_unclip',
        dest='no_unclip',
        default=False,
        action='store_true'
    )
    parser.add_option(
        '-d',
        dest='data_head',
        default=None,
        type='int',
        help='Index for dataset/head [Default: %default]'
    )
    (options, args) = parser.parse_args()

    if len(args) == 3:
        # single worker
        params_file = args[0]
        model_file = args[1]
        tsv_file = args[2]

    elif len(args) == 4:
        # multi separate
        options_pkl_file = args[0]
        params_file = args[1]
        model_file = args[2]
        tsv_file = args[3]

        # save out dir
        out_dir = options.out_dir

        # load options
        options_pkl = open(options_pkl_file, 'rb')
        options = pickle.load(options_pkl)
        options_pkl.close()

        # update output directory
        options.out_dir = out_dir

    elif len(args) == 5:
        # multi worker
        options_pkl_file = args[0]
        params_file = args[1]
        model_file = args[2]
        tsv_file = args[3]
        worker_index = int(args[4])

        # load options
        options_pkl = open(options_pkl_file, 'rb')
        options = pickle.load(options_pkl)
        options_pkl.close()

        # update output directory
        options.out_dir = '%s/job%d' % (options.out_dir, worker_index)

    else:
        parser.error('Must provide parameters/model, tsv, and genes GTF')

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    options.shifts = [int(shift) for shift in options.shifts.split(',')]
    options.sed_stats = options.sed_stats.split(',')

    #################################################################
    # read parameters and targets
    # read model parameters
    with open(params_file) as params_open:
        params = json.load(params_open)
    params_model = params["model"]
    params_train = params["train"]
    seq_len = params_model['seq_length']

    if params_train["task"] == "fine-tune":
        num_species = 165
        params_model["num_features"] = num_species + 5
        params_train['r64_idx'] = 109

    if options.targets_file is None:
        parser.error('Must provide targets table to properly handle strands.')
    else:
        targets_df = pd.read_csv(options.targets_file, sep='\t', index_col=0)

    # prep strand
    targets_strand_df = targets_prep_strand(targets_df)

    #################################################################
    # setup model
    seqnn_model = seqnn.SeqNN(params_model)
    if options.data_head is not None:
        print('data_head = ' + str(options.data_head), flush=True)
        seqnn_model.restore(model_file, options.data_head)
    else:
        seqnn_model.restore(model_file)
    seqnn_model.build_slice(targets_df.index)
    seqnn_model.build_ensemble(options.rc, options.shifts)

    model_stride = seqnn_model.model_strides[0]
    out_seq_len = seqnn_model.target_lengths[0] * model_stride

    #################################################################
    # read sequence / contexts / genes

    # read sequences (just alt and ref)
    seq_df = pd.read_csv(tsv_file, sep='\t')[['id', 'alt_sequence', 'ref_sequence']].copy().reset_index(drop=True)

    # read contexts
    ctx_df = pd.read_csv(options.ctx_tsv, sep='\t', encoding='utf-8')[
        ['chrom', 'start', 'end']
    ].copy().reset_index(drop=True)
    ctx_df['chrom'] = ctx_df['chrom'].str.strip()
    ctx_df['midp'] = (ctx_df['start'] + ctx_df['end']) // 2
    ctx_df['ctx_id'] = (
        ctx_df['chrom'].astype(str)
        + '_'
        + ctx_df['start'].astype(str)
        + '_'
        + ctx_df['end'].astype(str)
    )
    ctx_df['ctx_id'] = ctx_df['ctx_id'].str.strip()

    # read genes
    transcriptome = Transcriptome(options.genes_gtf)
    gene_strand = {}
    for gene_id, gene in transcriptome.genes.items():
        gene_strand[gene_id] = gene.strand

    # map sequence contexts to gene positions
    ctx_gene_slice = map_ctx_genes(ctx_df, out_seq_len, transcriptome, model_stride, options.span)

    # remove contexts w/o genes
    num_ctxs_pre = len(ctx_df)
    ctx_gene_mask = np.array([len(cgs) > 0 for cgs in ctx_gene_slice])
    ctx_df = ctx_df.loc[ctx_gene_mask].copy().reset_index(drop=True)
    ctx_gene_slice = [ctx_gene_slice[si] for si in range(num_ctxs_pre) if ctx_gene_mask[si]]
    num_ctxs = len(ctx_df)

    #################################################################
    # setup output
    sed_out = initialize_output_h5(options.out_dir, options.sed_stats, seq_df, ctx_df, ctx_gene_slice, targets_strand_df)

    #################################################################
    # predict SNP scores, write output

    # open genome
    genome_open = pysam.Fastafile(options.genome_fasta)

    # seq/context/gene index in the output
    xi = 0

    # for each sequence row (with alt_sequence, ref_sequence)
    for si, seq_row in tqdm(seq_df.iterrows(), total=len(seq_df)):
        insert_alt_seq = seq_row['alt_sequence']
        insert_ref_seq = seq_row['ref_sequence']
        insert_len = len(insert_alt_seq)  # should match len(ref_seq), but not guaranteed

        # for each context
        for ci, ctx_row in ctx_df.iterrows():
            ctx_chrom = ctx_row['chrom']
            ctx_midp = ctx_row['midp']

            # define left/right boundaries of the entire context
            ctx_start = ctx_midp - seq_len // 2
            ctx_end = ctx_start + seq_len

            # read the actual genomic context from disk
            ctx_ref_origin = make_seq(genome_open, ctx_chrom, ctx_start, ctx_end, seq_len)

            # figure out where to place the inserted sequence
            insert_start = ctx_midp - ctx_start - insert_len // 2
            insert_end = insert_start + insert_len

            # build ALT and REF versions of context
            ctx_alt = ctx_ref_origin[:insert_start] + insert_alt_seq + ctx_ref_origin[insert_end:]
            ctx_ref = ctx_ref_origin[:insert_start] + insert_ref_seq + ctx_ref_origin[insert_end:]

            # convert to 1-hot
            # real context
            real_1hot = dna.dna_1hot_mask_species_encoding(ctx_ref_origin, num_species=num_species, species_index=params_train['r64_idx'])
            alt_1hot = dna.dna_1hot_mask_species_encoding(ctx_alt, num_species=num_species, species_index=params_train['r64_idx'])
            userref_1hot = dna.dna_1hot_mask_species_encoding(ctx_ref, num_species=num_species, species_index=params_train['r64_idx'])

            # stack them up for a single model call if batch>1
            seqs_1hot = np.stack([real_1hot, alt_1hot, userref_1hot], axis=0)

            # get predictions
            # each of these is shape: (L, #targets)
            if params_train['batch_size'] == 1:
                real_preds = seqnn_model(seqs_1hot[:1])[0]
                alt_preds = seqnn_model(seqs_1hot[1:2])[0]
                userref_preds = seqnn_model(seqs_1hot[2:])[0]
            else:
                seq_preds = seqnn_model(seqs_1hot)
                real_preds, alt_preds, userref_preds = seq_preds[0], seq_preds[1], seq_preds[2]

            # untransform if requested
            if not options.no_untransform:
                if options.untransform_old:
                    real_preds = dataset.untransform_preds1(real_preds, targets_df, unclip=not options.no_unclip)
                    alt_preds = dataset.untransform_preds1(alt_preds, targets_df, unclip=not options.no_unclip)
                    userref_preds = dataset.untransform_preds1(userref_preds, targets_df, unclip=not options.no_unclip)
                else:
                    real_preds = dataset.untransform_preds(real_preds, targets_df, unclip=not options.no_unclip)
                    alt_preds = dataset.untransform_preds(alt_preds, targets_df, unclip=not options.no_unclip)
                    userref_preds = dataset.untransform_preds(userref_preds, targets_df, unclip=not options.no_unclip)

            # for each overlapping gene
            for gene_id, gene_slice in ctx_gene_slice[ci].items():
                # slice gene positions
                real_preds_gene = real_preds[gene_slice]
                alt_preds_gene = alt_preds[gene_slice]
                userref_preds_gene = userref_preds[gene_slice]

                # slice relevant strand targets
                if gene_strand[gene_id] == '+':
                    gene_strand_mask = (targets_df.strand != '-')
                else:
                    gene_strand_mask = (targets_df.strand != '+')
                real_preds_gene = real_preds_gene[..., gene_strand_mask]
                alt_preds_gene = alt_preds_gene[..., gene_strand_mask]
                userref_preds_gene = userref_preds_gene[..., gene_strand_mask]

                # compute 25th percentile across *real* predictions as a pseudocount
                pseudocounts = np.percentile(real_preds_gene, 25, axis=0)

                # 1) compute alt vs ref (old approach) if requested
                write_snp(
                    ref_preds= userref_preds_gene,    # user-provided "ref"
                    alt_preds= alt_preds_gene,        # alt
                    sed_out= sed_out,
                    xi= xi,
                    sed_stats= options.sed_stats,
                    pseudocounts= pseudocounts
                )

                # 2) compute each variant vs. the *original* context if user requests it
                #    i.e. logSED_ALT_ORIG = log2(ALT + 1) - log2(REAL + 1),
                #         logSED_REF_ORIG = log2(USERREF + 1) - log2(REAL + 1)
                #    We only do this if those stats are in `sed_stats`.
                if 'logSED_ALT_ORIG' in options.sed_stats or 'logSED_REF_ORIG' in options.sed_stats:
                    sum_real = np.sum(real_preds_gene, axis=0) + 1
                if 'logSED_ALT_ORIG' in options.sed_stats:
                    sum_alt = np.sum(alt_preds_gene, axis=0) + 1
                    logsed_alt = np.log2(sum_alt) - np.log2(sum_real)
                    sed_out['logSED_ALT_ORIG'][xi] = clip_float(logsed_alt).astype('float16')
                if 'logSED_REF_ORIG' in options.sed_stats:
                    sum_userref = np.sum(userref_preds_gene, axis=0) + 1
                    logsed_ref = np.log2(sum_userref) - np.log2(sum_real)
                    sed_out['logSED_REF_ORIG'][xi] = clip_float(logsed_ref).astype('float16')

                # increment row index in the HDF
                xi += 1

    # close genome
    genome_open.close()

    # done
    sed_out.close()


################################################################################
# Helper / subroutines
################################################################################

def clip_float(x, dtype=np.float16):
    """Clip array into float16 range safely to avoid overflows."""
    return np.clip(x, np.finfo(dtype).min, np.finfo(dtype).max)


def initialize_output_h5(out_dir: str, sed_stats, seq_df, ctx_df, ctx_gene_slice, targets_df):
    """Initialize an output HDF5 file for SED stats.

    We set up the dimension for the total (sequence x context x gene) combinations.
    Then create the datasets for whichever metrics are in `sed_stats`, plus
    some additional columns if needed.
    """
    sed_out = h5py.File('%s/sed.h5' % out_dir, 'w')

    # figure out how many (seq,context,gene) combos we have
    seq_indexes = []
    ctx_indexes = []
    gene_ids = []

    for seq_i, _ in seq_df.iterrows():
        for ctx_i, gene_slice in enumerate(ctx_gene_slice):
            gene_list = list(gene_slice.keys())
            gene_ids += gene_list
            seq_indexes += [seq_i] * len(gene_list)
            ctx_indexes += [ctx_i] * len(gene_list)

    num_scores = len(seq_indexes)

    # store the (si, ci, gene) arrays
    sed_out.create_dataset('si', data=np.array(seq_indexes))
    sed_out.create_dataset('ci', data=np.array(ctx_indexes))
    gene_ids = np.array(gene_ids, 'S')
    sed_out.create_dataset('gene', data=gene_ids)

    # store the unique seq_id and ctx_id as arrays for reference
    seq_ids = np.array(seq_df['id'].values.tolist(), 'S')
    sed_out.create_dataset('seq_id', data=seq_ids)
    ctx_ids = np.array(ctx_df['ctx_id'].values.tolist(), 'S')
    sed_out.create_dataset('ctx_id', data=ctx_ids)

    # store target IDs
    sed_out.create_dataset('target_ids', data=np.array(targets_df.identifier, 'S'))
    sed_out.create_dataset('target_labels', data=np.array(targets_df.description, 'S'))

    # now create relevant SED stats as float16 datasets
    num_targets = targets_df.shape[0]
    for sed_stat in sed_stats:
        # if standard metric name, create dataset
        # e.g. SED, logSED, D1, D2, etc.
        sed_out.create_dataset(
            sed_stat,
            shape=(num_scores, num_targets),
            dtype='float16'
        )
    return sed_out


def make_seq(genome_open, chrm, start, end, seq_len):
    """Fetch (potentially padded) reference sequence from the genome,
       ensuring length is always seq_len by padding with 'N' if needed.
    """
    if start < 0:
        seq_dna = "N" * (-start) + genome_open.fetch(chrm, 0, end)
    else:
        seq_dna = genome_open.fetch(chrm, start, end)
    # extend if shorter than seq_len
    if len(seq_dna) < seq_len:
        seq_dna += "N" * (seq_len - len(seq_dna))
    return seq_dna


def make_ctx_bedt(ctx_df, seq_len: int):
    """Create a BedTool object for all contexts, where seq_len
       is used to bound each region around its midpoint.
    """
    left_len = seq_len // 2
    right_len = seq_len // 2

    ctx_bed_lines = []
    for si, row in ctx_df.iterrows():
        ctx_start = max(0, row['midp'] - left_len)
        ctx_end = row['midp'] + right_len
        ctx_bed_lines.append('%s %d %d %d' % (row['chrom'], ctx_start, ctx_end, si))

    ctx_bedt = pybedtools.BedTool('\n'.join(ctx_bed_lines), from_string=True)
    return ctx_bedt


def map_ctx_genes(
    ctx_df,
    seq_len: int,
    transcriptome,
    model_stride: int,
    span: bool,
    majority_overlap: bool = True,
    intron1: bool = False
):
    """Intersect contexts with gene exons (or spans), constructing a list
       mapping each context to a dict {gene_id : [positions in the output array]}.
    """
    # either gene exons or entire span
    if span:
        genes_bedt = transcriptome.bedtool_span()
    else:
        genes_bedt = transcriptome.bedtool_exon()

    # build contexts
    ctx_bedt = make_ctx_bedt(ctx_df, seq_len)

    # initialize empty for each context
    ctx_gene_slice = []
    for _ in ctx_df.iterrows():
        ctx_gene_slice.append(OrderedDict())

    # overlap
    for overlap in genes_bedt.intersect(ctx_bedt, wo=True):
        gene_id = overlap[3]
        gene_start = int(overlap[1])
        gene_end = int(overlap[2])
        seq_start = int(overlap[7])
        seq_end = int(overlap[8])
        si = int(overlap[9])

        seq_len_chop = seq_end - seq_start
        seq_start -= (seq_len - seq_len_chop)

        # figure out bin range
        gene_seq_start = max(0, gene_start - seq_start)
        gene_seq_end = max(0, gene_end - seq_start)

        if majority_overlap:
            # requires >50% overlap
            bin_start = int(np.round(gene_seq_start / model_stride))
            bin_end = int(np.round(gene_seq_end / model_stride))
        else:
            bin_start = int(np.floor(gene_seq_start / model_stride))
            bin_end = int(np.ceil(gene_seq_end / model_stride))

        if intron1:
            bin_start -= 1
            bin_end += 1

        bin_max = int(seq_len / model_stride)
        bin_start = min(bin_start, bin_max)
        bin_end = min(bin_end, bin_max)
        bin_start = max(0, bin_start)
        bin_end = max(0, bin_end)

        if bin_end - bin_start > 0:
            ctx_gene_slice[si].setdefault(gene_id, []).extend(range(bin_start, bin_end))

    # remove duplicates
    for si in range(len(ctx_gene_slice)):
        for gene_id, gene_slice_vals in ctx_gene_slice[si].items():
            ctx_gene_slice[si][gene_id] = np.unique(gene_slice_vals)

    return ctx_gene_slice


def targets_prep_strand(targets_df):
    """Adjust the input targets table for merged stranded datasets,
       returning the new subset that excludes the redundant - strand if needed.
    """
    if "strand_pair" in targets_df.columns:
        # we have stranded pairs
        targets_strand = []
        for _, target in targets_df.iterrows():
            if target.strand_pair == target.name:
                targets_strand.append('.')
            else:
                # last character, e.g. + or -
                targets_strand.append(target.identifier[-1])
        targets_df['strand'] = targets_strand

        # keep only plus or dot
        strand_mask = (targets_df.strand != '-')
        targets_strand_df = targets_df[strand_mask]
    else:
        # no special pairing
        targets_strand_df = targets_df

    return targets_strand_df


def write_snp(ref_preds, alt_preds, sed_out, xi: int, sed_stats, pseudocounts):
    """
    Write alt vs ref difference metrics for a single gene overlap.

    ref_preds and alt_preds are shapes (L, T), i.e. length x #targets,
    after slicing to just that gene's bins and the relevant strand.

    We store results in sed_out[...] at row index xi.

    The original script writes e.g. SED, logSED, D1, D2, etc.
    """
    # sum across length
    ref_preds_sum = ref_preds.sum(axis=0)
    alt_preds_sum = alt_preds.sum(axis=0)

    # log sums
    ref_preds_sum_log = np.log2(ref_preds_sum + 1)
    alt_preds_sum_log = np.log2(alt_preds_sum + 1)

    # SED
    if 'SED' in sed_stats:
        sed = alt_preds_sum - ref_preds_sum
        sed_out['SED'][xi] = clip_float(sed).astype('float16')

    if 'logSED' in sed_stats:
        log_sed = alt_preds_sum_log - ref_preds_sum_log
        sed_out['logSED'][xi] = clip_float(log_sed).astype('float16')

    # D1
    if 'D1' in sed_stats:
        diff_abs = np.abs(ref_preds - alt_preds)
        diff_norm1 = diff_abs.sum(axis=0)
        sed_out['D1'][xi] = clip_float(diff_norm1).astype('float16')

    if 'logD1' in sed_stats:
        # take logs first
        ref_preds_log = np.log2(ref_preds + 1)
        alt_preds_log = np.log2(alt_preds + 1)
        diff1_log = np.abs(ref_preds_log - alt_preds_log)
        diff_log_norm1 = diff1_log.sum(axis=0)
        sed_out['logD1'][xi] = clip_float(diff_log_norm1).astype('float16')

    # D2
    if 'D2' in sed_stats:
        diff2 = np.power(ref_preds - alt_preds, 2)
        diff_norm2 = np.sqrt(diff2.sum(axis=0))
        sed_out['D2'][xi] = clip_float(diff_norm2).astype('float16')

    if 'logD2' in sed_stats:
        ref_preds_log = np.log2(ref_preds + 1)
        alt_preds_log = np.log2(alt_preds + 1)
        diff2_log = np.power(ref_preds_log - alt_preds_log, 2)
        diff_log_norm2 = np.sqrt(diff2_log.sum(axis=0))
        sed_out['logD2'][xi] = clip_float(diff_log_norm2).astype('float16')

    # normalized differences (pseudocount-based)
    ref_preds_norm = ref_preds + pseudocounts
    ref_preds_norm /= ref_preds_norm.sum(axis=0)
    alt_preds_norm = alt_preds + pseudocounts
    alt_preds_norm /= alt_preds_norm.sum(axis=0)

    if 'nD2' in sed_stats:
        ndiff2 = np.power(ref_preds_norm - alt_preds_norm, 2)
        ndiff_norm2 = np.sqrt(ndiff2.sum(axis=0))
        sed_out['nD2'][xi] = clip_float(ndiff_norm2).astype('float16')

    if 'nDi' in sed_stats:
        ndiff_abs = np.abs(ref_preds_norm - alt_preds_norm)
        ndiff_normi = ndiff_abs.max(axis=0)
        sed_out['nDi'][xi] = clip_float(ndiff_normi).astype('float16')

    if 'JS' in sed_stats:
        ref_alt_entr = rel_entr(ref_preds_norm, alt_preds_norm).sum(axis=0)
        alt_ref_entr = rel_entr(alt_preds_norm, ref_preds_norm).sum(axis=0)
        js_dist = (ref_alt_entr + alt_ref_entr) / 2
        sed_out['JS'][xi] = clip_float(js_dist).astype('float16')

    # raw predictions
    if 'REF' in sed_stats:
        sed_out['REF'][xi] = clip_float(ref_preds_sum).astype('float16')
    if 'ALT' in sed_stats:
        sed_out['ALT'][xi] = clip_float(alt_preds_sum).astype('float16')


if __name__ == '__main__':
    main()
