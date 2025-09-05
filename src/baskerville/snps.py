import concurrent
import json
import pdb
import sys

import h5py
import numpy as np
import pandas as pd
import pybedtools
import pysam
from scipy.special import rel_entr
from tqdm import tqdm

from baskerville import dna
from baskerville import dataset
from baskerville.gene import Transcriptome
from baskerville import seqnn
from baskerville import vcf as bvcf
from baskerville.helpers.trt_optimized_model import OptimizedModel

import matplotlib.pyplot as plt

def plot_coverage_track_pair_bins_w_ref_zoomed(
    y_wt, y_mut, chrom, start,
    search_gene, center_pos, gene_start, gene_end, strand,
    poses, track_indices, track_names, track_scales,
    track_transforms, clip_softs, snpweight,
    y_ground_truth=None, ref_track_indices=None,
    log_scale=False, sqrt_scale=False, plot_mut=True,
    plot_window=4096, normalize_window=4096,
    bin_size=32, pad=16, rescale_tracks=True,
    normalize_counts=False, save_figs=False,
    save_suffix="default", save_dir="./",
    gene_slice=None, anno_df=None, margin=100
):
    """
    Zoomed coverage plotting over [min(center_pos, gene_start) - margin,
    max(center_pos, gene_end) + margin], in binned coordinates.
    """
    # 1) define the zoom region in genomic coords
    region_start = min(center_pos, gene_start) - margin
    region_end   = max(center_pos, gene_end)   + margin

    # 2) convert to bin indices (once)
    plot_start_bin = (region_start - start) // bin_size - pad
    plot_end_bin   = (region_end   - start) // bin_size - pad
    center_bin     = (center_pos - start)    // bin_size - pad
    gene_start_bin = (gene_start - start)    // bin_size - pad
    gene_end_bin   = (gene_end   - start)    // bin_size - pad
    mut_bin        = (poses[0]   - start)    // bin_size - pad

    # 3) extract annotation bins
    anno_bins = []
    if anno_df is not None:
        df = anno_df.query(
            f"chrom=='{chrom}' and position_hg38>={region_start} and position_hg38<{region_end}"
        )
        anno_bins = [(pos - start)//bin_size - pad for pos in df['position_hg38'].values]

    # 4) prepare mean‐across‐axes vectors (will undo transforms inside loop)
    #    note: we keep full per‐track tensors for transform, but collapse axes now
    wt_base = np.mean(y_wt, axis=(0,1,3))
    mut_base = np.mean(y_mut, axis=(0,1,3))
    gt_base = None
    if y_ground_truth is not None:
        gt_base = np.mean(y_ground_truth, axis=(0,1,3))

    logSED = None

    # 5) per‐track plotting
    for name, idx, scale, transform, clip_soft in zip(
        track_names, track_indices, track_scales, track_transforms, clip_softs
    ):
        # copy & undo transforms
        wt = wt_base.copy().astype(np.float32)
        mut = mut_base.copy().astype(np.float32)
        gt  = gt_base.copy().astype(np.float32) if gt_base is not None else None

        if rescale_tracks:
            wt /= scale
            mut /= scale
            if gt is not None: gt /= scale

            if clip_soft is not None:
                # invert soft clipping
                for arr in (wt, mut, gt) if gt is not None else (wt, mut):
                    mask = arr > clip_soft
                    unclipped = (arr - clip_soft)**2 + clip_soft
                    arr[mask] = unclipped[mask]

            # invert power/sqrt
            wt **= (1.0/transform)
            mut **= (1.0/transform)
            if gt is not None: gt **= (1.0/transform)

        # slice to region & optional normalization
        wt = wt[plot_start_bin:plot_end_bin]
        mut = mut[plot_start_bin:plot_end_bin]
        if gt is not None:
            gt = gt[plot_start_bin:plot_end_bin]

        if normalize_counts:
            total_wt = wt.sum()
            total_mut = mut.sum()
            wt /= total_wt; mut /= total_mut; wt *= total_wt; mut *= total_wt
            if gt is not None:
                total_gt = gt.sum()
                gt /= total_gt; gt *= total_wt

        # compute logSED if desired
        sum_wt = wt.sum()
        sum_mut = mut.sum()
        logSED = np.log2(sum_mut + 1) - np.log2(sum_wt + 1)

        # optional log/sqrt scaling for plotting
        if log_scale:
            wt = np.log2(wt + 1)
            mut = np.log2(mut + 1)
            if gt is not None: gt = np.log2(gt + 1)
        elif sqrt_scale:
            wt = np.sqrt(wt + 1)
            mut = np.sqrt(mut + 1)
            if gt is not None: gt = np.sqrt(gt + 1)

        # determine y‐max for axis limits
        y_max = max(wt.max(), mut.max(), gt.max() if gt is not None else 0)

        # set up subplots
        if gt is not None:
            fig, (ax1, ax2) = plt.subplots(2,1, figsize=(16,4), sharex=True)
        else:
            # fig, ax1 = plt.subplots(1,1, figsize=(18,2))
            fig, ax1 = plt.subplots(1,1, figsize=(16,1.7))
            ax2 = None

        x = np.arange(plot_start_bin, plot_end_bin)
        # highlight SNP ± margin
        highlight_start_bin = ((center_pos - 40) - start)//bin_size - pad
        highlight_end_bin   = highlight_start_bin + 6        # Highlight SNP region ±margin
        # draw shaded region behind everything
        ax1.axvspan(
            highlight_start_bin,
            highlight_end_bin,
            color='lightgrey',
            alpha=0.5,
            zorder=0,
            label="±(40bp) ISM region"
        )
        # bars
        ax1.bar(x, wt, width=1, alpha=0.6, label="Ref")
        if plot_mut:
            ax1.bar(x, mut, width=1, alpha=0.6, label="Alt")
        if ax2 is not None:
            ax2.bar(x, gt, width=1, alpha=0.6, label="Ground Truth")

        # variant & gene lines
        ax1.scatter([mut_bin], [0.05*y_max], s=80, marker='*', color='k', label="SNP")
        ax1.axvline(center_bin, color='k', linestyle=':', label=f"Variant ({center_pos})")
        start_line, end_line = (gene_start_bin, gene_end_bin) if strand=='+' else (gene_end_bin, gene_start_bin)
        ax1.axvline(start_line, color='g', linestyle='--', label=f"{search_gene} start ({gene_start})")
        ax1.axvline(end_line,   color='r', linestyle='-.', label=f"{search_gene} end ({gene_end})")

        # annotation lines
        for b in anno_bins:
            ax1.axvline(b, color='cyan', linewidth=1, alpha=0.5, linestyle='-')

        # labels and legend
        ax1.set_xlim(plot_start_bin, plot_end_bin)
        ax1.set_ylabel("Coverage")
        ax1.set_title(f"{search_gene} {name} logSED={logSED:.3f}", fontsize=14)
        ax1.legend(loc="upper left", bbox_to_anchor=(1.02,1), fontsize=8)
        if ax2:
            ax2.set_xlim(plot_start_bin, plot_end_bin)
            ax2.set_ylabel("GT Signal")
            ax2.set_xlabel(f"{chrom}:{region_start}-{region_end}bp")
            ax2.legend(loc="upper left", bbox_to_anchor=(1.02,1), fontsize=8)
        else:
            ax1.set_xlabel(f"{chrom}:{region_start}-{region_end}bp")

        plt.tight_layout()

        # optional saving
        if save_figs:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"{search_gene}_{name}_{save_suffix}.png"), dpi=300)
            plt.close(fig)

    return logSED



def score_snps(params_file, model_file, vcf_file, worker_index, options):
    """
    Score SNPs in a VCF file with a SeqNN model.

    :param params_file: Model parameters
    :param model_file: Saved model weights
    :param vcf_file: VCF
    :param worker_index
    :param options: options from cmd args
    :return:
    """

    #################################################################
    # read parameters and targets

    # read model parameters
    with open(params_file) as params_open:
        params = json.load(params_open)
    params_model = params["model"]
    params_train = params["train"]    
    # read data parameters (data 0)
    # data_stats_file = "%s/statistics.json" % args.data_dirs[0]
    # with open(data_stats_file) as data_stats_open:
    #     data_stats = json.load(data_stats_open)
    # num_species = data_stats.get("num_species", 1)
    if params_train["task"] == "fine-tune":
        num_species = 165
        params_model["num_features"] = num_species + 5
        params_train['r64_idx'] = 109
    # read targets
    if options.targets_file is None:
        print("Must provide targets file to clarify stranded datasets", file=sys.stderr)
        exit(1)
    targets_df = pd.read_csv(options.targets_file, sep="\t", index_col=0)

    # handle strand pairs
    if "strand_pair" in targets_df.columns:
        # prep strand
        targets_strand_df = dataset.targets_prep_strand(targets_df)

        # set strand pairs (using new indexing)
        orig_new_index = dict(zip(targets_df.index, np.arange(targets_df.shape[0])))
        targets_strand_pair = np.array(
            [orig_new_index[ti] for ti in targets_df.strand_pair]
        )
        params_model["strand_pair"] = [targets_strand_pair]

        # construct strand sum transform
        strand_transform = dataset.make_strand_transform(targets_df, targets_strand_df)
    else:
        targets_strand_df = targets_df
        strand_transform = None

    #################################################################
    # setup model

    seqnn_model = seqnn.SeqNN(params_model)

    # load model
    sum_length = options.snp_stats == "SAD"
    if options.tensorrt:
        seqnn_model.model = OptimizedModel(model_file, seqnn_model.strand_pair)
        input_shape = tuple(seqnn_model.model.loaded_model_fn.inputs[0].shape.as_list())
    else:
        seqnn_model.restore(model_file)
        seqnn_model.build_slice(targets_df.index)
        if sum_length:
            seqnn_model.build_sad()
        seqnn_model.build_ensemble(options.rc)
        input_shape = seqnn_model.model.input_shape
        # # Update input shape to include species channels

    # make dummy predictions to warm up model
    dummy_input_shape = (1,) + input_shape[1:]
    dummy_input = np.random.random(dummy_input_shape).astype(np.float32)
    seqnn_model(dummy_input)

    # shift outside seqnn
    num_shifts = len(options.shifts)
    targets_length = seqnn_model.target_lengths[0]

    #################################################################
    # load SNPs

    # clustering SNPs requires sorted VCF and no reference flips
    snps_clustered = options.cluster_snps_pct > 0

    # filter for worker SNPs
    if options.processes is None:
        start_i = None
        end_i = None
    else:
        # determine boundaries
        num_snps = bvcf.vcf_count(vcf_file)
        worker_bounds = np.linspace(0, num_snps, options.processes + 1, dtype="int")
        start_i = worker_bounds[worker_index]
        end_i = worker_bounds[worker_index + 1]

    # read SNPs
    snps = bvcf.vcf_snps(
        vcf_file,
        require_sorted=snps_clustered,
        flip_ref=~snps_clustered,
        validate_ref_fasta=options.genome_fasta,
        start_i=start_i,
        end_i=end_i,
    )

    # cluster SNPs
    if snps_clustered:
        snp_clusters = cluster_snps(
            snps, params_model["seq_length"], options.cluster_snps_pct
        )
    else:
        snp_clusters = []
        for snp in snps:
            snp_clusters.append(SNPCluster())
            snp_clusters[-1].add_snp(snp)

    # delimit sequence boundaries
    [sc.delimit(params_model["seq_length"]) for sc in snp_clusters]

    # open genome FASTA
    genome_open = pysam.Fastafile(options.genome_fasta)

    #################################################################
    # predict SNP scores, write output

    # setup output
    scores_out = initialize_output_h5(
        options.out_dir,
        options.snp_stats,
        snps,
        targets_length,
        targets_strand_df,
        num_shifts,
    )

    # CPU computation
    def score_write(ref_preds, alt_preds, si):
        scores = compute_scores(
            ref_preds, alt_preds, options.snp_stats, strand_transform
        )
        for snp_stat in options.snp_stats:
            scores_out[snp_stat][si] = scores[snp_stat]

    if options.untransform_old:
        untransform = dataset.untransform_preds1
    else:
        untransform = dataset.untransform_preds

    # SNP index
    si = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # initialize 1 hot encoding
        sc0 = snp_clusters[0]
        s1l = executor.submit(sc0.get_1hots, genome_open, num_species=num_species, species_index=params_train['r64_idx'])

        for ci, sc in enumerate(tqdm(snp_clusters)):
            # pull latest 1 hot encoding
            snp_1hot_list = s1l.result()
            ref_1hot = np.expand_dims(snp_1hot_list[0], axis=0)

            # submit next 1 hot encoding
            if ci + 1 < len(snp_clusters):
                sc1 = snp_clusters[ci + 1]
                s1l = executor.submit(sc1.get_1hots, genome_open, num_species=num_species, species_index=params_train['r64_idx'])

            # predict reference
            ref_preds = []
            for shift in options.shifts:
                ref_1hot_shift = dna.hot1_augment(ref_1hot, shift=shift)
                ref_preds_shift = seqnn_model(ref_1hot_shift)[0]
                # untransform predictions
                if options.targets_file is None:
                    ref_preds.append(ref_preds_shift)
                else:
                    rpsf = executor.submit(untransform, ref_preds_shift, targets_df)
                    ref_preds.append(rpsf)

            for ai, alt_1hot in enumerate(snp_1hot_list[1:]):
                alt_1hot = np.expand_dims(alt_1hot, axis=0)

                # add compensation shifts for indels
                indel_size = sc.snps[ai].indel_size()
                if indel_size == 0:
                    alt_shifts = options.shifts
                else:
                    # repeat reference predictions, unless stitching
                    if not options.indel_stitch:
                        ref_preds = np.repeat(ref_preds, 2, axis=0)

                    # add compensation shifts
                    alt_shifts = []
                    for shift in options.shifts:
                        alt_shifts.append(shift)
                        alt_shifts.append(shift - indel_size)

                # predict alternate
                alt_preds = []
                for shift in alt_shifts:
                    alt_1hot_shift = dna.hot1_augment(alt_1hot, shift=shift)
                    alt_preds_shift = seqnn_model(alt_1hot_shift)[0]
                    # untransform predictions
                    if options.targets_file is None:
                        alt_preds.append(alt_preds_shift)
                    else:
                        apsf = executor.submit(untransform, alt_preds_shift, targets_df)
                        alt_preds.append(apsf)

                # result
                if options.targets_file is not None:
                    # get result, only if not already gotten
                    if isinstance(ref_preds[0], concurrent.futures.Future):
                        ref_preds = [rpsf.result() for rpsf in ref_preds]
                    alt_preds = [apsf.result() for apsf in alt_preds]

                # stitch indel compensation shifts
                if indel_size != 0 and options.indel_stitch:
                    alt_preds = stitch_preds(alt_preds, options.shifts)

                # flip reference and alternate
                if snps[si].flipped:
                    rp_snp = np.array(alt_preds)
                    ap_snp = np.array(ref_preds)
                else:
                    rp_snp = np.array(ref_preds)
                    ap_snp = np.array(alt_preds)

                # write SNP
                if sum_length:
                    write_snp(rp_snp, ap_snp, scores_out, si, options.snp_stats)
                else:
                    executor.submit(score_write, rp_snp, ap_snp, si)

                # update SNP index
                si += 1

    # close genome
    genome_open.close()

    # compute SAD distributions across variants
    write_pct(scores_out, options.snp_stats)
    scores_out.close()



def score_gene_snps(params_file, model_file, vcf_file, worker_index, options):
    """
    Score SNPs in a VCF file with a SeqNN model.

    :param params_file: Model parameters
    :param model_file: Saved model weights
    :param vcf_file: VCF
    :param worker_index
    :param options: options from cmd args
    :return:
    """

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

    # read targets
    if options.targets_file is None:
        print("Must provide targets file to clarify stranded datasets", file=sys.stderr)
        exit(1)
    targets_df = pd.read_csv(options.targets_file, sep="\t", index_col=0)

    # handle strand pairs
    if "strand_pair" in targets_df.columns:
        # prep strand
        targets_strand_df = dataset.targets_prep_strand(targets_df)

        # set strand pairs (using new indexing)
        orig_new_index = dict(zip(targets_df.index, np.arange(targets_df.shape[0])))
        targets_strand_pair = np.array(
            [orig_new_index[ti] for ti in targets_df.strand_pair]
        )
        params_model["strand_pair"] = [targets_strand_pair]
    else:
        targets_strand_df = targets_df

    # construct strand sum transform
    plus_mask = targets_df.strand != "-"
    minus_mask = targets_df.strand != "+"

    #################################################################
    # setup model
    seqnn_model = seqnn.SeqNN(params_model)

    # load model
    if options.tensorrt:
        seqnn_model.model = OptimizedModel(model_file, seqnn_model.strand_pair)
        input_shape = tuple(seqnn_model.model.loaded_model_fn.inputs[0].shape.as_list())
    else:
        seqnn_model.restore(model_file)
        seqnn_model.build_slice(targets_df.index)
        seqnn_model.build_ensemble(options.rc)
        input_shape = seqnn_model.model.input_shape

    # make dummy predictions to warm up model
    dummy_input_shape = (1,) + input_shape[1:]
    dummy_input = np.random.random(dummy_input_shape).astype(np.float32)
    seqnn_model(dummy_input)

    # shift outside seqnn
    num_shifts = len(options.shifts)
    targets_length = seqnn_model.target_lengths[0]
    model_stride = seqnn_model.model_strides[0]
    model_crop = seqnn_model.target_crops[0] * model_stride
    
    #################################################################
    # load SNPs

    # filter for worker SNPs
    if options.processes is None:
        start_i = None
        end_i = None
    else:
        # determine boundaries
        num_snps = bvcf.vcf_count(vcf_file)
        worker_bounds = np.linspace(0, num_snps, options.processes + 1, dtype="int")
        start_i = worker_bounds[worker_index]
        end_i = worker_bounds[worker_index + 1]

    # read SNPs
    snps = bvcf.vcf_snps(
        vcf_file,
        require_sorted=True,
        flip_ref=False,
        validate_ref_fasta=options.genome_fasta,
        start_i=start_i,
        end_i=end_i,
    )

    # read genes
    transcriptome = Transcriptome(options.genes_gtf)

    # cluster genes
    genesnp_clusters = cluster_genes(
        transcriptome, params_model["seq_length"], options.cluster_pct
    )

    # delimit sequence boundaries
    [gsc.delimit(params_model["seq_length"], model_crop) for gsc in genesnp_clusters]

    # assign SNPs to genes
    map_snps_genes(snps, genesnp_clusters)

    # remove genes w/o SNPs
    genesnp_clusters = [gsc for gsc in genesnp_clusters if len(gsc.snps) > 0]

    # open genome FASTA
    genome_open = pysam.Fastafile(options.genome_fasta)

    #################################################################
    # predict SNP scores, write output

    # setup output
    scores_out = initialize_output_h5(
        options.out_dir,
        options.snp_stats,
        snps,
        targets_length,
        targets_strand_df,
        num_shifts,
        genesnp_clusters,
    )

    # CPU computation
    def score_write(ref_preds, alt_preds, gene_id, snp_id):
        scores = compute_scores(ref_preds, alt_preds, options.snp_stats)

        # --- New: aggregate across ALL tracks, then compute logSED exactly as in script #1 ---
        num_shifts = ref_preds.shape[0]
        # 1) sum over shifts & bins, per target
        ref_per_target = ref_preds.sum(axis=(0,1)) / num_shifts
        alt_per_target = alt_preds.sum(axis=(0,1)) / num_shifts
        # 2) now sum across targets
        sum_ref_all = ref_per_target.sum()
        sum_alt_all = alt_per_target.sum()
        # 3) compute the “collapsed‐tracks” logSED
        logSED_agg = np.log2(sum_alt_all + 1) - np.log2(sum_ref_all + 1)
        # stash it in the same scores dict
        scores["logSED_agg"] = logSED_agg.astype("float16")  

        for snp_stat in options.snp_stats:
            stat_out = scores_out.require_group(snp_stat)
            snp_out = stat_out.require_group(snp_id)
            snp_out.create_dataset(gene_id, data=scores[snp_stat], dtype="float16")

        # --- New: store the aggregate scalar under its own group ---
        agg_out = scores_out.require_group("logSED_agg")
        agg_snp = agg_out.require_group(snp_id)
        agg_snp.create_dataset(gene_id, data=scores["logSED_agg"], dtype="float16")

    if options.untransform_old:
        untransform = dataset.untransform_preds1
    else:
        untransform = dataset.untransform_preds

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for gsc in tqdm(genesnp_clusters):
            snp_1hot_list = gsc.get_1hots(genome_open, num_species=num_species, species_index=params_train['r64_idx'])
            if snp_1hot_list is None:
                continue
            ref_1hot = np.expand_dims(snp_1hot_list[0], axis=0)
            # predict reference
            ref_preds = []
            for shift in options.shifts:
                ref_1hot_shift = dna.hot1_augment(ref_1hot, shift=shift)
                ref_preds_shift = seqnn_model(ref_1hot_shift)[0] 
                ref_preds.append(ref_preds_shift)

            for ai, alt_1hot in enumerate(snp_1hot_list[1:]):
                alt_1hot = np.expand_dims(alt_1hot, axis=0)

                # add compensation shifts for indels
                indel_size = gsc.snps[ai].indel_size()
                if indel_size == 0:
                    alt_shifts = options.shifts
                else:
                    # repeat reference predictions, unless stitching
                    if not options.indel_stitch:
                        ref_preds = np.repeat(ref_preds, 2, axis=0)

                    # add compensation shifts
                    alt_shifts = []
                    for shift in options.shifts:
                        alt_shifts.append(shift)
                        alt_shifts.append(shift - indel_size)

                # predict alternate
                alt_preds = []
                for shift in alt_shifts:
                    alt_1hot_shift = dna.hot1_augment(alt_1hot, shift=shift)
                    alt_preds_shift = seqnn_model(alt_1hot_shift)[0]
                    # untransform predictions
                    if options.targets_file is None:
                        alt_preds.append(alt_preds_shift)
                    else:
                        apsf = executor.submit(untransform, alt_preds_shift, targets_df)
                        alt_preds.append(apsf)

                # result
                if options.targets_file is not None:
                    # get result, only if not already gotten
                    if isinstance(ref_preds[0], concurrent.futures.Future):
                        ref_preds = [rpsf.result() for rpsf in ref_preds]
                    alt_preds = [apsf.result() for apsf in alt_preds]

                # flip reference and alternate
                if gsc.snps[ai].flipped:
                    rp_snp = np.array(alt_preds)
                    ap_snp = np.array(ref_preds)
                else:
                    rp_snp = np.array(ref_preds)
                    ap_snp = np.array(alt_preds)

                for gene in gsc.genes:
                    ### ADDED/CHANGED CODE BELOW
                    # Simple bounding check for entire gene start/end
                    gene_start, gene_end = gene.span()
                    if not (gene_start >= gsc.start and gene_end <= gsc.end):
                        continue
                    ### END OF ADDED/CHANGED CODE
                    # slice gene positions
                    gene_slice = gene.output_slice(
                        gsc.pstart, gsc.pend - gsc.pstart, model_stride
                    )

                    if len(gene_slice) == 0:
                        print(
                            f"WARNING: {gene.kv['gene_id']} exons fall outside prediction boundaries."
                        )
                    else:
                        rp_gene = rp_snp[:, gene_slice]
                        ap_gene = ap_snp[:, gene_slice]

                        # slice gene strand
                        if gene.strand == "+":
                            rp_gene = rp_gene[..., plus_mask]
                            ap_gene = ap_gene[..., plus_mask]
                        else:
                            rp_gene = rp_gene[..., minus_mask]
                            ap_gene = ap_gene[..., minus_mask]

                        # write SNP
                        executor.submit(
                            score_write,
                            rp_gene,
                            ap_gene,
                            gene.kv["gene_id"],
                            gsc.snps[ai].rsid,
                        )
    # close open files
    genome_open.close()
    scores_out.close()
    
from collections import defaultdict

def score_gene_snps_target(params_file, model_file, vcf_file, worker_index, options):
    """
    Score SNPs in a VCF file with a SeqNN model.

    :param params_file: Model parameters
    :param model_file: Saved model weights
    :param vcf_file: VCF
    :param worker_index
    :param options: options from cmd args
    :return:
    """

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

    # read targets
    if options.targets_file is None:
        print("Must provide targets file to clarify stranded datasets", file=sys.stderr)
        exit(1)
    targets_df = pd.read_csv(options.targets_file, sep="\t", index_col=0)

    # handle strand pairs
    if "strand_pair" in targets_df.columns:
        # prep strand
        targets_strand_df = dataset.targets_prep_strand(targets_df)

        # set strand pairs (using new indexing)
        orig_new_index = dict(zip(targets_df.index, np.arange(targets_df.shape[0])))
        targets_strand_pair = np.array(
            [orig_new_index[ti] for ti in targets_df.strand_pair]
        )
        params_model["strand_pair"] = [targets_strand_pair]
    else:
        targets_strand_df = targets_df

    # construct strand sum transform
    plus_mask = targets_df.strand != "-"
    minus_mask = targets_df.strand != "+"

    #################################################################
    # setup model
    seqnn_model = seqnn.SeqNN(params_model)

    # load model
    if options.tensorrt:
        seqnn_model.model = OptimizedModel(model_file, seqnn_model.strand_pair)
        input_shape = tuple(seqnn_model.model.loaded_model_fn.inputs[0].shape.as_list())
    else:
        seqnn_model.restore(model_file)
        seqnn_model.build_slice(targets_df.index)
        seqnn_model.build_ensemble(options.rc)
        input_shape = seqnn_model.model.input_shape

    # make dummy predictions to warm up model
    dummy_input_shape = (1,) + input_shape[1:]
    dummy_input = np.random.random(dummy_input_shape).astype(np.float32)
    seqnn_model(dummy_input)

    # shift outside seqnn
    num_shifts = len(options.shifts)
    targets_length = seqnn_model.target_lengths[0]
    model_stride = seqnn_model.model_strides[0]
    model_crop = seqnn_model.target_crops[0] * model_stride
    
    #################################################################
    # load SNPs
    # filter for worker SNPs
    if options.processes is None:
        start_i = None
        end_i = None
    else:
        # determine boundaries
        num_snps = bvcf.vcf_count(vcf_file)
        worker_bounds = np.linspace(0, num_snps, options.processes + 1, dtype="int")
        start_i = worker_bounds[worker_index]
        end_i = worker_bounds[worker_index + 1]

    # read SNPs
    snps = bvcf.vcf_snps(
        vcf_file,
        require_sorted=True,
        flip_ref=False,
        validate_ref_fasta=options.genome_fasta,
        start_i=start_i,
        end_i=end_i,
    )

    # read genes
    transcriptome = Transcriptome(options.genes_gtf)

    snps_by_gene = defaultdict(list)
    for snp in snps:
        # Each 'snp' here presumably has an info_dict or info attribute
        # that might look like: {"GENE": "YAL049C"} or so.
        # Make sure you know how your variant parser stores the INFO field.
        gene_id = None
        
        if snp.gene is not None:
            snps_by_gene[snp.gene].append(snp)

    # open genome FASTA
    genome_open = pysam.Fastafile(options.genome_fasta)

    genesnp_clusters = []
    for gene_id, snps in snps_by_gene.items():
        gene_keys = [gk for gk in transcriptome.genes.keys() if gene_id in gk]
        if not gene_keys:
            continue
        # We just grab the first key if there are multiple
        gene_obj = transcriptome.genes[gene_keys[0]]

        # Build a new cluster
        gsc = GeneSNPCluster()
        gsc.genes = [gene_obj]  # list with a single gene
        # If your downstream logic relies on gsc.delimit(...) to set pstart / pend:
        # (You likely have model_crop from earlier in your code.)
        gsc.delimit(params_model["seq_length"], model_crop)

        # If delimit came back invalid, skip
        if gsc.start < 0 or gsc.end < 0:
            continue
        
        for snp in snps:
            if snp.pos >= gsc.start and snp.pos < gsc.end:
                gsc.snps.append(snp)
        # Add to our final list
        genesnp_clusters.append(gsc)

    #################################################################
    # predict SNP scores, write output    
    # ensure our collapsed‐tracks stat is in the HDF5 group list
    all_stats = options.snp_stats + ["logSED_agg"]
    scores_out = initialize_output_h5(
        options.out_dir,
        all_stats,
        snps,
        targets_length,
        targets_strand_df,
        num_shifts,
        genesnp_clusters,
    )

    def score_write(ref_preds, alt_preds, gene_id, snp_id):
        # make sure we have a pure NumPy array
        ref_arr = np.asarray(ref_preds, dtype=np.float32)
        alt_arr = np.asarray(alt_preds, dtype=np.float32)

        # 1) all the per‐track stats
        scores = compute_scores(ref_arr, alt_arr, options.snp_stats)

        # 2) now the “collapsed‐tracks” logSED
        #    sum over shifts & bins → per‐target sums
        ref_per_target = ref_arr.sum(axis=(0,1)) / ref_arr.shape[0]
        alt_per_target = alt_arr.sum(axis=(0,1)) / alt_arr.shape[0]
        #    sum across targets → single scalar
        total_ref = float(ref_per_target.sum())
        total_alt = float(alt_per_target.sum())
        logSED_agg = np.float32(np.log2(total_alt + 1.) - np.log2(total_ref + 1.))

        # stash it just like the others
        scores["logSED_agg"] = logSED_agg

        # 3) write everything out
        for snp_stat in options.snp_stats:
            stat_out = scores_out.require_group(snp_stat)
            snp_out  = stat_out.require_group(snp_id)
            snp_out.create_dataset(gene_id, data=scores[snp_stat], dtype="float16")

        # the "logSED_agg" group was created above via all_stats
        stat_out = scores_out["logSED_agg"]
        snp_grp  = stat_out.require_group(snp_id)
        snp_grp.create_dataset(gene_id,
                               data=scores["logSED_agg"],
                               dtype="float32")

    if options.untransform_old:
        untransform = dataset.untransform_preds1
    else:
        untransform = dataset.untransform_preds

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for gsc in tqdm(genesnp_clusters):
            snp_1hot_list = gsc.get_1hots(genome_open, num_species=num_species, species_index=params_train['r64_idx'])
            if snp_1hot_list is None:
                continue
            ref_1hot = np.expand_dims(snp_1hot_list[0], axis=0)
            # predict reference
            ref_preds = []
            for shift in options.shifts:
                ref_1hot_shift = dna.hot1_augment(ref_1hot, shift=shift)
                ref_preds_shift = seqnn_model(ref_1hot_shift)[0]
                # untransform predictions
                if options.targets_file is None:
                    ref_preds.append(ref_preds_shift)
                else:
                    rpsf = executor.submit(untransform, ref_preds_shift, targets_df)
                    ref_preds.append(rpsf)

            for ai, alt_1hot in enumerate(snp_1hot_list[1:]):
                alt_1hot = np.expand_dims(alt_1hot, axis=0)

                # add compensation shifts for indels
                indel_size = gsc.snps[ai].indel_size()
                if indel_size == 0:
                    alt_shifts = options.shifts
                else:
                    # repeat reference predictions, unless stitching
                    if not options.indel_stitch:
                        ref_preds = np.repeat(ref_preds, 2, axis=0)
                    # add compensation shifts
                    alt_shifts = []
                    for shift in options.shifts:
                        alt_shifts.append(shift)
                        alt_shifts.append(shift - indel_size)

                # predict alternate
                alt_preds = []
                for shift in alt_shifts:
                    alt_1hot_shift = dna.hot1_augment(alt_1hot, shift=shift)
                    alt_preds_shift = seqnn_model(alt_1hot_shift)[0]
                    # untransform predictions
                    if options.targets_file is None:
                        alt_preds.append(alt_preds_shift)
                    else:
                        apsf = executor.submit(untransform, alt_preds_shift, targets_df)
                        alt_preds.append(apsf)

                # result
                if options.targets_file is not None:
                    # get result, only if not already gotten
                    if isinstance(ref_preds[0], concurrent.futures.Future):
                        ref_preds = [rpsf.result() for rpsf in ref_preds]
                    if isinstance(alt_preds[0], concurrent.futures.Future):
                        alt_preds = [apsf.result() for apsf in alt_preds]

                # flip reference and alternate
                if gsc.snps[ai].flipped:
                    rp_snp = np.array(alt_preds)
                    ap_snp = np.array(ref_preds)
                else:
                    rp_snp = np.array(ref_preds)
                    ap_snp = np.array(alt_preds)

                for gene in gsc.genes:
                    ### ADDED/CHANGED CODE BELOW
                    # Simple bounding check for entire gene start/end
                    gene_start, gene_end = gene.span()
                    if not (gene_start >= gsc.start and gene_end <= gsc.end):
                        print(
                            f"Skipping gene {gene.kv['gene_id']} at runtime because it is not "
                            f"fully contained in the window [{gsc.start}, {gsc.end}]. "
                            f"Gene range = [{gene.start}, {gene.end}]."
                        )
                        continue
                    ### END OF ADDED/CHANGED CODE
                    # slice gene positions
                    gene_slice = gene.output_slice(
                        gsc.pstart, gsc.pend - gsc.pstart, model_stride
                    )
                    if len(gene_slice) == 0:
                        print(
                            f"WARNING: {gene.kv['gene_id']} exons fall outside prediction boundaries."
                        )
                    else:
                        rp_gene = rp_snp[:, gene_slice]
                        ap_gene = ap_snp[:, gene_slice]

                        # slice gene strand
                        if gene.strand == "+":
                            rp_gene = rp_gene[..., plus_mask]
                            ap_gene = ap_gene[..., plus_mask]
                        else:
                            rp_gene = rp_gene[..., minus_mask]
                            ap_gene = ap_gene[..., minus_mask]

                        # write SNP
                        executor.submit(
                            score_write,
                            rp_gene,
                            ap_gene,
                            gene.kv["gene_id"],
                            gsc.snps[ai].rsid,
                        )
    # close open files
    genome_open.close()
    scores_out.close()


def cluster_genes(transcriptome, seq_length: int, center_pct: float):
    """Cluster genes into regions that will satisfy the required center_pct.

    Args:
        transcriptome (Transcriptome): Transcriptome object.
        seq_length (int): Sequence length.
        center_pct (float): Percent of sequence length to cluster genes.
    """
    valid_gene_distance = int(seq_length * center_pct)

    gene_clusters = []

    # re-sort genes by midpoint
    chromosomes = set([gene.chrom for gene in transcriptome.genes.values()])
    for chrom in chromosomes:
        gene_pos = []
        gene_objs = []
        for gene in transcriptome.genes.values():
            if gene.chrom == chrom:
                gene_pos.append(gene.midpoint())
                gene_objs.append(gene)

        cluster_pos0 = -valid_gene_distance
        for gi in np.argsort(gene_pos):
            gene = gene_objs[gi]
            if gene_pos[gi] < cluster_pos0 + valid_gene_distance:
                # append to latest cluster
                gene_clusters[-1].add_gene(gene)
            else:
                # initialize new cluster
                gene_clusters.append(GeneSNPCluster())
                gene_clusters[-1].add_gene(gene)
                cluster_pos0 = gene_pos[gi]

    return gene_clusters


def cluster_snps(snps, seq_len: int, center_pct: float):
    """Cluster a sorted list of SNPs into regions that will satisfy
       the required center_pct.

    Args:
        snps [SNP]: List of SNPs.
        seq_len (int): Sequence length.
        center_pct (float): Percent of sequence length to cluster SNPs.
    """
    valid_snp_distance = int(seq_len * center_pct)

    snp_clusters = []
    cluster_chr = None

    for snp in snps:
        if snp.chr == cluster_chr and snp.pos < cluster_pos0 + valid_snp_distance:
            # append to latest cluster
            snp_clusters[-1].add_snp(snp)
        else:
            # initialize new cluster
            snp_clusters.append(SNPCluster())
            snp_clusters[-1].add_snp(snp)
            cluster_chr = snp.chr
            cluster_pos0 = snp.pos

    return snp_clusters


def compute_scores(ref_preds, alt_preds, snp_stats, strand_transform=None):
    """Compute SNP scores from reference and alternative predictions.

    Args:
        ref_preds (np.array): Reference allele predictions.
        alt_preds (np.array): Alternative allele predictions.
        snp_stats [str]: List of SAD stats to compute.
        strand_transform (scipy.sparse): Strand transform matrix.
    """
    num_shifts, seq_length, num_targets = ref_preds.shape

    # log/sqrt
    ref_preds_log = np.log2(ref_preds + 1)
    alt_preds_log = np.log2(alt_preds + 1)
    ref_preds_sqrt = np.sqrt(ref_preds)
    alt_preds_sqrt = np.sqrt(alt_preds)

    # sum across length, mean across shifts
    ref_preds_sum = ref_preds.sum(axis=(0, 1)) / num_shifts
    alt_preds_sum = alt_preds.sum(axis=(0, 1)) / num_shifts

    # My implementation of logSED
    ref_preds_sum_log = np.log2(ref_preds_sum + 1)
    alt_preds_sum_log = np.log2(alt_preds_sum + 1)

    ref_preds_log_sum = ref_preds_log.sum(axis=(0, 1)) / num_shifts
    alt_preds_log_sum = alt_preds_log.sum(axis=(0, 1)) / num_shifts
    ref_preds_sqrt_sum = ref_preds_sqrt.sum(axis=(0, 1)) / num_shifts
    alt_preds_sqrt_sum = alt_preds_sqrt.sum(axis=(0, 1)) / num_shifts

    # difference
    altref_diff = alt_preds - ref_preds
    altref_log_diff = alt_preds_log - ref_preds_log
    altref_sqrt_diff = alt_preds_sqrt - ref_preds_sqrt

    # initialize scores dict
    scores = {}

    def strand_clip_save(key, score, d2=False):
        if strand_transform is not None:
            if d2:
                score = np.power(score, 2)
                score = score @ strand_transform
                score = np.sqrt(score)
            else:
                score = score @ strand_transform
        score = np.clip(score, np.finfo(np.float16).min, np.finfo(np.float16).max)
        scores[key] = score.astype("float16")

    # compare reference to alternative via sum subtraction
    if "SUM" in snp_stats:
        sad = alt_preds_sum - ref_preds_sum
        strand_clip_save("SUM", sad)
    if "logSUM" in snp_stats:
        log_sad = alt_preds_log_sum - ref_preds_log_sum
        strand_clip_save("logSUM", log_sad)

    # My implementation of logSED
    if "logSED" in snp_stats:
        log_sad = alt_preds_sum_log - ref_preds_sum_log
        strand_clip_save("logSED", log_sad)

    if "sqrtSUM" in snp_stats:
        sqrt_sad = alt_preds_sqrt_sum - ref_preds_sqrt_sum
        strand_clip_save("sqrtSUM", sqrt_sad)

    # compare reference to alternative via max subtraction
    if "SAX" in snp_stats:
        altref_adiff = np.abs(altref_diff)
        sax = []
        for s in range(num_shifts):
            max_i = np.argmax(altref_adiff[s], axis=0)
            sax.append(altref_diff[s, max_i, np.arange(num_targets)])
        sax = np.array(sax).mean(axis=0)
        strand_clip_save("SAX", sax)

    # L1 norm of difference vector
    if "D1" in snp_stats:
        sad_d1 = np.linalg.norm(altref_diff, ord=1, axis=1)
        strand_clip_save("D1", sad_d1.mean(axis=0))
    if "logD1" in snp_stats:
        log_d1 = np.linalg.norm(altref_log_diff, ord=1, axis=1)
        strand_clip_save("logD1", log_d1.mean(axis=0))
    if "sqrtD1" in snp_stats:
        sqrt_d1 = np.linalg.norm(altref_sqrt_diff, ord=1, axis=1)
        strand_clip_save("sqrtD1", sqrt_d1.mean(axis=0))

    # L2 norm of difference vector
    if "D2" in snp_stats:
        sad_d2 = np.linalg.norm(altref_diff, ord=2, axis=1)
        strand_clip_save("D2", sad_d2.mean(axis=0), d2=True)
    if "logD2" in snp_stats:
        log_d2 = np.linalg.norm(altref_log_diff, ord=2, axis=1)
        strand_clip_save("logD2", log_d2.mean(axis=0), d2=True)
    if "sqrtD2" in snp_stats:
        sqrt_d2 = np.linalg.norm(altref_sqrt_diff, ord=2, axis=1)
        strand_clip_save("sqrtD2", sqrt_d2.mean(axis=0), d2=True)

    if "JS" in snp_stats:
        # normalized scores
        pseudocounts = np.percentile(ref_preds, 25, axis=1)
        ref_preds_norm = ref_preds + pseudocounts
        ref_preds_norm /= ref_preds_norm.sum(axis=1)
        alt_preds_norm = alt_preds + pseudocounts
        alt_preds_norm /= alt_preds_norm.sum(axis=1)

        # compare normalized JS
        js_dist = []
        for s in range(num_shifts):
            ref_alt_entr = rel_entr(ref_preds_norm[s], alt_preds_norm[s]).sum(axis=0)
            alt_ref_entr = rel_entr(alt_preds_norm[s], ref_preds_norm[s]).sum(axis=0)
            js_dist.append((ref_alt_entr + alt_ref_entr) / 2)
        js_dist = np.mean(js_dist, axis=0)
        # handling strand this way is incorrect, but I'm punting for now
        strand_clip_save("JS", js_dist)

    if "logJS" in snp_stats:
        # normalized scores
        pseudocounts = np.percentile(ref_preds_log, 25, axis=0)
        ref_preds_log_norm = ref_preds_log + pseudocounts
        ref_preds_log_norm /= ref_preds_log_norm.sum(axis=0)
        alt_preds_log_norm = alt_preds_log + pseudocounts
        alt_preds_log_norm /= alt_preds_log_norm.sum(axis=0)

        # compare normalized JS
        log_js_dist = []
        for s in range(num_shifts):
            rps = ref_preds_log_norm[s]
            aps = alt_preds_log_norm[s]
            ref_alt_entr = rel_entr(rps, aps).sum(axis=0)
            alt_ref_entr = rel_entr(aps, rps).sum(axis=0)
            log_js_dist.append((ref_alt_entr + alt_ref_entr) / 2)
        log_js_dist = np.mean(log_js_dist, axis=0)
        # handling strand this way is incorrect, but I'm punting for now
        strand_clip_save("logJS", log_js_dist)

    # predictions
    if "REF" in snp_stats:
        ref_preds = np.clip(
            ref_preds, np.finfo(np.float16).min, np.finfo(np.float16).max
        )
        scores["REF"] = ref_preds.astype("float16")
    if "ALT" in snp_stats:
        alt_preds = np.clip(
            alt_preds, np.finfo(np.float16).min, np.finfo(np.float16).max
        )
        scores["ALT"] = alt_preds.astype("float16")

    return scores


def initialize_output_h5(
    out_dir,
    snp_stats,
    snps,
    targets_length,
    targets_df,
    num_shifts,
    geneseq_clusters=None,
):
    """Initialize an output HDF5 file for SAD stats.

    Args:
        out_dir (str): Output directory.
        snp_stats [str]: List of SAD stats to compute.
        snps [SNP]: List of SNPs.
        targets_length (int): Targets' sequence length
        targets_df (pd.DataFrame): Targets DataFrame.
        num_shifts (int): Number of shifts.
        geneseq_clusters [GeneSNPCluster]: Gene sequence clusters.
    """

    num_targets = targets_df.shape[0]
    num_snps = len(snps)

    scores_out = h5py.File("%s/scores.h5" % out_dir, "w")

    # write SNPs
    snp_ids = np.array([snp.rsid for snp in snps], "S")
    scores_out.create_dataset("snp", data=snp_ids)

    # write SNP chr
    snp_chr = np.array([snp.chr for snp in snps], "S")
    scores_out.create_dataset("chr", data=snp_chr)

    # write SNP pos
    snp_pos = np.array([snp.pos for snp in snps], dtype="uint32")
    scores_out.create_dataset("pos", data=snp_pos)

    # write SNP reference allele
    snp_refs = []
    snp_alts = []
    for snp in snps:
        if snp.flipped:
            snp_refs.append(snp.alt_alleles[0])
            snp_alts.append(snp.ref_allele)
        else:
            snp_refs.append(snp.ref_allele)
            snp_alts.append(snp.alt_alleles[0])
    snp_refs = np.array(snp_refs, "S")
    snp_alts = np.array(snp_alts, "S")
    scores_out.create_dataset("ref_allele", data=snp_refs)
    scores_out.create_dataset("alt_allele", data=snp_alts)

    # write targets
    scores_out.create_dataset("target_ids", data=np.array(targets_df.identifier, "S"))
    scores_out.create_dataset(
        "target_labels", data=np.array(targets_df.description, "S")
    )

    if geneseq_clusters is not None:
        # write genes
        gene_ids = []
        for gsc in geneseq_clusters:
            gene_ids.extend([gene.kv["gene_id"] for gene in gsc.genes])
        gene_ids = np.array(gene_ids, "S")
        scores_out.create_dataset("gene", data=gene_ids)
    else:
        # initialize stats
        for snp_stat in snp_stats:
            if snp_stat in ["REF", "ALT"]:
                scores_out.create_dataset(
                    snp_stat,
                    shape=(num_snps, num_shifts, targets_length, num_targets),
                    dtype="float16",
                )
            else:
                scores_out.create_dataset(
                    snp_stat, shape=(num_snps, num_targets), dtype="float16"
                )

    return scores_out


def make_alt_1hot(ref_1hot, snp_seq_pos, ref_allele, alt_allele):
    """Return alternative allele one hot coding.

    Args:
        ref_1hot (np.array): Reference allele one hot coding.
        snp_seq_pos (int): SNP position in sequence.
        ref_allele (str): Reference allele.
        alt_allele (str): Alternative allele.

    Returns:
        np.array: Alternative allele one hot coding.
    """
    ref_n = len(ref_allele)
    alt_n = len(alt_allele)

    # copy reference
    alt_1hot = np.copy(ref_1hot)

    if alt_n == ref_n:
        # SNP
        dna.hot1_set(alt_1hot, snp_seq_pos, alt_allele)

    elif ref_n > alt_n:
        # deletion
        delete_len = ref_n - alt_n
        if ref_allele[0] == alt_allele[0]:
            dna.hot1_delete(alt_1hot, snp_seq_pos + 1, delete_len)
        else:
            print(
                "WARNING: Deletion first nt does not match: %s %s"
                % (ref_allele, alt_allele),
                file=sys.stderr,
            )

    else:
        # insertion
        if ref_allele[0] == alt_allele[0]:
            dna.hot1_insert(alt_1hot, snp_seq_pos + 1, alt_allele[1:])
        else:
            print(
                "WARNING: Insertion first nt does not match: %s %s"
                % (ref_allele, alt_allele),
                file=sys.stderr,
            )

    return alt_1hot


def make_gene_bedt(genesnp_clusters):
    """Make a BedTool object for all gene sequences."""
    gene_bed_lines = []
    for gi, gsc in enumerate(genesnp_clusters):
        geneseq_start = max(0, gsc.start)
        gene_bed_lines.append("%s %d %d %d" % (gsc.chr, geneseq_start, gsc.end, gi))
    gene_bedt = pybedtools.BedTool("\n".join(gene_bed_lines), from_string=True)
    return gene_bedt


def make_snp_bedt(snps):
    """Make a BedTool object for all SNPs"""
    snp_bed_lines = []
    for si, snp in enumerate(snps):
        snp_bed_lines.append("%s %d %d %d" % (snp.chr, snp.pos - 1, snp.pos, si))
    snp_bedt = pybedtools.BedTool("\n".join(snp_bed_lines), from_string=True)
    return snp_bedt


def map_snps_genes(snps, genesnp_clusters):
    """Map SNPs to gene sequences."""
    geneseq_bedt = make_gene_bedt(genesnp_clusters)
    snp_bedt = make_snp_bedt(snps)

    for overlap in geneseq_bedt.intersect(snp_bedt, wa=True, wb=True):
        gchr, gstart, gend, gi, schr, spos, send, si = overlap
        gi, si = int(gi), int(si)
        genesnp_clusters[gi].add_snp(snps[si])


def stitch_preds(preds, shifts):
    """Stitch indel left and right compensation shifts.

    Args:
        preds [np.array]: List of predictions.
        shifts [int]: List of shifts.
    """
    cp = preds[0].shape[0] // 2
    preds_stitch = []
    for hi, shift in enumerate(shifts):
        hil = 2 * hi
        hir = hil + 1
        preds_stitch_i = np.concatenate((preds[hil][:cp], preds[hir][cp:]), axis=0)
        preds_stitch.append(preds_stitch_i)
    return preds_stitch


def write_pct(scores_out, snp_stats):
    """Compute percentile values for each target and write to HDF5.

    Args:
        scores_out (h5py.File): Output HDF5 file.
        snp_stats [str]: List of SAD stats to compute.
    """
    # define percentiles
    d_fine = 0.001
    d_coarse = 0.01
    percentiles_neg = np.arange(d_fine, 0.1, d_fine)
    percentiles_base = np.arange(0.1, 0.9, d_coarse)
    percentiles_pos = np.arange(0.9, 1, d_fine)

    percentiles = np.concatenate([percentiles_neg, percentiles_base, percentiles_pos])
    scores_out.create_dataset("percentiles", data=percentiles)

    for snp_stat in snp_stats:
        if snp_stat not in ["REF", "ALT"]:
            snp_stat_pct = "%s_pct" % snp_stat

            # compute
            sad_pct = np.percentile(scores_out[snp_stat], 100 * percentiles, axis=0).T
            sad_pct = sad_pct.astype("float16")

            # save
            scores_out.create_dataset(snp_stat_pct, data=sad_pct, dtype="float16")


def write_snp(ref_preds_sum, alt_preds_sum, scores_out, si, snp_stats):
    """Write SNP predictions to HDF, assuming the length dimension has
    been collapsed.

    Args:
        ref_preds_sum (np.array): Reference allele predictions.
        alt_preds_sum (np.array): Alternative allele predictions.
        scores_out (h5py.File): Output HDF5 file.
        si (int): SNP index.
        snp_stats [str]: List of SAD stats to compute.
    """

    # compare reference to alternative via mean subtraction
    if "SAD" in snp_stats:
        sad = alt_preds_sum - ref_preds_sum
        sad = sad.mean(axis=0)
        scores_out["SAD"][si] = sad.astype("float16")


class SNPCluster:
    def __init__(self):
        self.snps = []
        self.chr = None
        self.start = None
        self.end = None

    def add_snp(self, snp):
        """Add SNP to cluster."""
        self.snps.append(snp)

    def delimit(self, seq_len):
        """Delimit sequence boundaries."""
        positions = [snp.pos for snp in self.snps]
        pos_min = np.min(positions)
        pos_max = np.max(positions)
        pos_mid = (pos_min + pos_max) // 2
        self.chr = self.snps[0].chr
        self.start = pos_mid - seq_len // 2
        self.end = self.start + seq_len

    def get_1hots(self, genome_open, num_species=None, species_index=None):
        """Get list of one hot coded sequences."""
        seqs1_list = []

        # extract reference
        if self.start < 0:
            ref_seq = (
                "N" * (-self.start) + genome_open.fetch(self.chr, 0, self.end).upper()
            )
        else:
            ref_seq = genome_open.fetch(self.chr, self.start, self.end).upper()

        # extend to full length
        if len(ref_seq) < self.end - self.start:
            ref_seq += "N" * (self.end - self.start - len(ref_seq))
        # verify reference alleles
        for snp in self.snps:
            ref_n = len(snp.ref_allele)
            snp_pos = snp.pos - 1 - self.start
            ref_snp = ref_seq[snp_pos : snp_pos + ref_n]
            if snp.ref_allele != ref_snp:
                print(
                    "ERROR: %s does not match reference %s" % (snp, ref_snp),
                    file=sys.stderr,
                )
                # exit(1)
                return None

        # 1 hot code reference sequence
        ref_1hot = dna.dna_1hot_mask_species_encoding(ref_seq, num_species=num_species, species_index=species_index)
        seqs1_list = [ref_1hot]

        # make alternative 1 hot coded sequences
        # (assuming SNP is 1-based indexed)
        for snp in self.snps:
            snp_pos = snp.pos - 1 - self.start
            alt_1hot = make_alt_1hot(
                ref_1hot, snp_pos, snp.ref_allele, snp.alt_alleles[0]
            )
            seqs1_list.append(alt_1hot)

        return seqs1_list


class GeneSNPCluster(SNPCluster):
    def __init__(self):
        super().__init__()
        self.genes = []

    def add_gene(self, gene):
        """Add gene to cluster."""
        self.genes.append(gene)

    def delimit(self, seq_len, crop=0):
        """Delimit sequence boundaries."""
        self.chr = self.genes[0].chrom
        midp = int(np.mean([g.midpoint() for g in self.genes]))
        self.start = midp - seq_len // 2
        self.end = self.start + seq_len
        self.pstart = self.start + crop
        self.pend = self.end - crop