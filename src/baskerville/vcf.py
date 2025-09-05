# Copyright 2017 Calico LLC

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
import gzip
import os
import pdb
import subprocess
import sys
import tempfile

import numpy as np
import pysam

from baskerville import dna

"""
vcf.py

Methods and classes to support .vcf SNP analysis.
"""


def cap_allele(allele, cap=5):
    """Cap the length of an allele in the figures."""
    if len(allele) > cap:
        allele = allele[:cap] + "*"
    return allele


def intersect_seqs_snps(vcf_file, seqs, vision_p=1):
    """Intersect a VCF file with a list of sequence coordinates.

    In
     vcf_file:
     seqs: list of objects w/ chrom, start, end
     vision_p: proportion of sequences visible to center genes.

    Out
     seqs_snps: list of list mapping segment indexes to overlapping SNP indexes
    """

    # print segments to BED
    # hash segments to indexes
    seq_temp = tempfile.NamedTemporaryFile()
    seq_bed_file = seq_temp.name
    seq_bed_out = open(seq_bed_file, "w")
    seq_indexes = {}
    for si in range(len(seqs)):
        sstart = max(0, seqs[si].start)
        print("%s\t%d\t%d" % (seqs[si].chrom, sstart, seqs[si].end), file=seq_bed_out)
        seq_key = (seqs[si].chrom, sstart, seqs[si].end)
        seq_indexes[seq_key] = si
    seq_bed_out.close()

    # hash SNPs to indexes
    snp_indexes = {}
    si = 0

    vcf_in = open(vcf_file)
    line = vcf_in.readline()
    while line[0] == "#":
        line = vcf_in.readline()
    while line:
        a = line.split()
        snp_id = a[2]
        if snp_id in snp_indexes:
            raise Exception("Duplicate SNP id %s will break the script" % snp_id)
        snp_indexes[snp_id] = si
        si += 1
        line = vcf_in.readline()
    vcf_in.close()

    # initialize list of lists
    seqs_snps = []
    for _ in range(len(seqs)):
        seqs_snps.append([])

    # intersect
    p = subprocess.Popen(
        "bedtools intersect -wo -a %s -b %s" % (vcf_file, seq_bed_file),
        shell=True,
        stdout=subprocess.PIPE,
    )
    for line in p.stdout:
        line = line.decode("UTF-8")
        a = line.split()
        pos = int(a[1])
        snp_id = a[2]
        seq_chrom = a[-4]
        seq_start = int(a[-3])
        seq_end = int(a[-2])
        seq_key = (seq_chrom, seq_start, seq_end)

        vision_buffer = (seq_end - seq_start) * (1 - vision_p) // 2
        if seq_start + vision_buffer < pos < seq_end - vision_buffer:
            seqs_snps[seq_indexes[seq_key]].append(snp_indexes[snp_id])

    p.communicate()

    return seqs_snps


def intersect_snps_seqs(vcf_file, seq_coords, vision_p=1):
    """Intersect a VCF file with a list of sequence coordinates.

    In
     vcf_file:
     seq_coords: list of sequence coordinates
     vision_p: proportion of sequences visible to center genes.

    Out
     snp_segs: list of list mapping SNP indexes to overlapping sequence indexes
    """
    # print segments to BED
    # hash segments to indexes
    seg_temp = tempfile.NamedTemporaryFile()
    seg_bed_file = seg_temp.name
    seg_bed_out = open(seg_bed_file, "w")
    segment_indexes = {}

    for si in range(len(seq_coords)):
        segment_indexes[seq_coords[si]] = si
        print("%s\t%d\t%d" % seq_coords[si], file=seg_bed_out)

    seg_bed_out.close()

    # hash SNPs to indexes
    snp_indexes = {}
    si = 0

    vcf_in = open(vcf_file)
    line = vcf_in.readline()
    while line[0] == "#":
        line = vcf_in.readline()
    while line:
        a = line.split()
        snp_id = a[2]
        if snp_id in snp_indexes:
            raise Exception("Duplicate SNP id %s will break the script" % snp_id)
        snp_indexes[snp_id] = si
        si += 1
        line = vcf_in.readline()
    vcf_in.close()

    # initialize list of lists
    snp_segs = []
    for i in range(len(snp_indexes)):
        snp_segs.append([])

    # intersect
    p = subprocess.Popen(
        "bedtools intersect -wo -a %s -b %s" % (vcf_file, seg_bed_file),
        shell=True,
        stdout=subprocess.PIPE,
    )
    for line in p.stdout:
        line = line.decode("UTF-8")
        a = line.split()
        pos = int(a[1])
        snp_id = a[2]
        seg_chrom = a[-4]
        seg_start = int(a[-3])
        seg_end = int(a[-2])
        seg_key = (seg_chrom, seg_start, seg_end)

        vision_buffer = (seg_end - seg_start) * (1 - vision_p) // 2
        if seg_start + vision_buffer < pos < seg_end - vision_buffer:
            snp_segs[snp_indexes[snp_id]].append(segment_indexes[seg_key])

    p.communicate()

    return snp_segs


def snp_seq1(snp, seq_len, genome_open):
    """Produce one hot coded sequences for a SNP.

    Attrs:
        snp [SNP] :
        seq_len (int) : sequence length to code
        genome_open (File) : open genome FASTA file

    Return:
        seq_vecs_list [array] : list of one hot coded sequences surrounding the
        SNP
    """
    left_len = seq_len // 2 - 1
    right_len = seq_len // 2

    # initialize one hot coded vector list
    seq_vecs_list = []

    # specify positions in GFF-style 1-based
    seq_start = snp.pos - left_len
    seq_end = snp.pos + right_len + max(0, len(snp.ref_allele) - snp.longest_alt())

    # extract sequence as BED style
    if seq_start < 0:
        seq = "N" * (1 - seq_start) + genome_open.fetch(snp.chr, 0, seq_end).upper()
    else:
        seq = genome_open.fetch(snp.chr, seq_start - 1, seq_end).upper()

    # extend to full length
    if len(seq) < seq_len:
        seq += "N" * (seq_len - len(seq))

    # verify that ref allele matches ref sequence
    seq_ref = seq[left_len : left_len + len(snp.ref_allele)]
    ref_found = True
    if seq_ref != snp.ref_allele:
        # search for reference allele in alternatives
        ref_found = False

        # for each alternative allele
        for alt_al in snp.alt_alleles:
            # grab reference sequence matching alt length
            seq_ref_alt = seq[left_len : left_len + len(alt_al)]
            if seq_ref_alt == alt_al:
                # found it!
                ref_found = True

                # warn user
                print(
                    "WARNING: %s - alt (as opposed to ref) allele matches reference genome; changing reference genome to match."
                    % (snp.rsid),
                    file=sys.stderr,
                )

                # remove alt allele and include ref allele
                seq = seq[:left_len] + snp.ref_allele + seq[left_len + len(alt_al) :]
                break

    if not ref_found:
        print(
            "WARNING: %s - reference genome does not match any allele" % snp.rsid,
            file=sys.stderr,
        )

    else:
        # one hot code ref allele
        seq_vecs_ref, seq_ref = dna_length_1hot(seq, seq_len)
        seq_vecs_list.append(seq_vecs_ref)

        for alt_al in snp.alt_alleles:
            # remove ref allele and include alt allele
            seq_alt = seq[:left_len] + alt_al + seq[left_len + len(snp.ref_allele) :]

            # one hot code
            seq_vecs_alt, seq_alt = dna_length_1hot(seq_alt, seq_len)
            seq_vecs_list.append(seq_vecs_alt)

    return seq_vecs_list


def snps_seq1_species_encoding(
    snps,
    seq_len,
    genome_fasta,
    num_species=None,
    species_index=None,
    return_seqs=False,
):
    """Produce an array of one‐hot + species‐encoded sequences for a list of SNPs.

    Args:
        snps [SNP]: list of SNP objects with .chr, .pos, .ref_allele, .alt_alleles, .rsid, .longest_alt()
        seq_len (int): total length of sequence window
        genome_fasta (str): path to FASTA file for reference genome
        num_species (int, optional): total number of species channels
        species_index (int, optional): which species this sequence is from (0‐based)
        return_seqs (bool): if True, also return the raw sequence strings

    Returns:
        seq_vecs (np.ndarray): shape (n_seqs, seq_len, 4+1+num_species) or (n_seqs, seq_len, 4) if no species
        seq_headers [str]: list of FASTA‐style headers for each sequence
        seq_snps [SNP]: list of SNPs that passed filtering
        seqs [str] (optional): the actual DNA strings (only if return_seqs=True)
    """
    left_len = seq_len // 2 - 1
    right_len = seq_len // 2

    seq_vecs_list = []
    seq_snps = []
    seqs = []
    seq_headers = []

    genome_open = pysam.Fastafile(genome_fasta)

    for snp in snps:
        # compute window endpoints in 1-based coords
        seq_start = snp.pos - left_len
        seq_end = snp.pos + right_len + max(0, len(snp.ref_allele) - snp.longest_alt())

        # fetch sequence (pad with Ns if at chrom start)
        if seq_start < 1:
            left_pad = "N" * (1 - seq_start)
            seq = left_pad + genome_open.fetch(snp.chr, 0, seq_end).upper()
        else:
            # pysam uses 0‐based, half-open
            seq = genome_open.fetch(snp.chr, seq_start - 1, seq_end).upper()

        # pad on right if needed
        if len(seq) < seq_len:
            seq += "N" * (seq_len - len(seq))

        # check ref allele matches; if not, try to swap in alt and warn
        seq_ref = seq[left_len : left_len + len(snp.ref_allele)]
        if seq_ref != snp.ref_allele:
            matched = False
            for alt in snp.alt_alleles:
                if seq[left_len : left_len + len(alt)] == alt:
                    matched = True
                    print(f"WARNING: {snp.rsid} – alt allele matches; patching to ref.", file=sys.stderr)
                    seq = seq[:left_len] + snp.ref_allele + seq[left_len + len(alt) :]
                    break
            if not matched:
                print(f"WARNING: {snp.rsid} – neither ref nor alt matches genome; skipping.", file=sys.stderr)
                continue

        seq_snps.append(snp)

        # --- encode REF allele ---
        seq_vec_ref = dna.dna_1hot_mask_species_encoding(
            seq,
            seq_len=seq_len,
            n_uniform=False,
            n_sample=False,
            num_species=num_species,
            species_index=species_index,
        )
        seq_vecs_list.append(seq_vec_ref)
        seq_headers.append(f"{snp.rsid}_{cap_allele(snp.ref_allele)}")
        if return_seqs:
            seqs.append(seq)

        # --- encode each ALT allele ---
        for alt in snp.alt_alleles:
            seq_alt = seq[:left_len] + alt + seq[left_len + len(snp.ref_allele) :]
            seq_vec_alt = dna.dna_1hot_mask_species_encoding(
                seq_alt,
                seq_len=seq_len,
                n_uniform=False,
                n_sample=False,
                num_species=num_species,
                species_index=species_index,
            )
            seq_vecs_list.append(seq_vec_alt)
            seq_headers.append(f"{snp.rsid}_{cap_allele(alt)}")
            if return_seqs:
                seqs.append(seq_alt)

    # stack and return
    seq_vecs = np.stack(seq_vecs_list, axis=0)
    if return_seqs:
        return seq_vecs, seq_headers, seq_snps, seqs
    else:
        return seq_vecs, seq_headers, seq_snps


def snps2_seq1(snps, seq_len, genome1_fasta, genome2_fasta, return_seqs=False):
    """Produce an array of one hot coded sequences for a list of SNPs.

    Attrs:
        snps [SNP] : list of SNPs
        seq_len (int) : sequence length to code
        genome_fasta (str) : major allele genome FASTA file
        genome2_fasta (str) : minor allele genome FASTA file

    Return:
        seq_vecs (array) : one hot coded sequences surrounding the SNPs
        seq_headers [str] : headers for sequences
        seq_snps [SNP] : list of used SNPs
    """
    left_len = seq_len // 2 - 1
    right_len = seq_len // 2

    # open genome FASTA
    genome1 = pysam.Fastafile(genome1_fasta)
    genome2 = pysam.Fastafile(genome2_fasta)

    # initialize one hot coded vector list
    seq_vecs_list = []

    # save successful SNPs
    seq_snps = []

    # save sequence strings, too
    seqs = []

    # name sequences
    seq_headers = []

    for snp in snps:
        if len(snp.alt_alleles) > 1:
            raise Exception(
                "Major/minor genome mode requires only two alleles: %s" % snp.rsid
            )

        alt_al = snp.alt_alleles[0]

        # specify positions in GFF-style 1-based
        seq_start = snp.pos - left_len
        seq_end = snp.pos + right_len + len(snp.ref_allele)

        # extract sequence as BED style
        if seq_start < 0:
            seq_ref = "N" * (-seq_start) + genome1.fetch(snp.chr, 0, seq_end).upper()
        else:
            seq_ref = genome1.fetch(snp.chr, seq_start - 1, seq_end).upper()

        # extend to full length
        if len(seq_ref) < seq_end - seq_start:
            seq_ref += "N" * (seq_end - seq_start - len(seq_ref))

        # verify that ref allele matches ref sequence
        seq_ref_snp = seq_ref[left_len : left_len + len(snp.ref_allele)]
        if seq_ref_snp != snp.ref_allele:
            raise Exception(
                "WARNING: Major allele SNP %s doesnt match reference genome: %s vs %s"
                % (snp.rsid, snp.ref_allele, seq_ref_snp)
            )

        # specify positions in GFF-style 1-based
        seq_start = snp.pos2 - left_len
        seq_end = snp.pos2 + right_len + len(alt_al)

        # extract sequence as BED style
        if seq_start < 0:
            seq_alt = "N" * (-seq_start) + genome2.fetch(snp.chr, 0, seq_end).upper()
        else:
            seq_alt = genome2.fetch(snp.chr, seq_start - 1, seq_end).upper()

        # extend to full length
        if len(seq_alt) < seq_end - seq_start:
            seq_alt += "N" * (seq_end - seq_start - len(seq_alt))

        # verify that ref allele matches ref sequence
        seq_alt_snp = seq_alt[left_len : left_len + len(alt_al)]
        if seq_alt_snp != alt_al:
            raise Exception(
                "WARNING: Minor allele SNP %s doesnt match reference genome: %s vs %s"
                % (snp.rsid, snp.alt_alleles[0], seq_alt_snp)
            )

        seq_snps.append(snp)

        # one hot code ref allele
        seq_vecs_ref, seq_ref = dna_length_1hot(seq_ref, seq_len)
        seq_vecs_list.append(seq_vecs_ref)
        if return_seqs:
            seqs.append(seq_ref)

        # name ref allele
        seq_headers.append("%s_%s" % (snp.rsid, cap_allele(snp.ref_allele)))

        # one hot code alt allele
        seq_vecs_alt, seq_alt = dna_length_1hot(seq_alt, seq_len)
        seq_vecs_list.append(seq_vecs_alt)
        if return_seqs:
            seqs.append(seq_alt)

        # name
        seq_headers.append("%s_%s" % (snp.rsid, cap_allele(alt_al)))

    # convert to array
    seq_vecs = np.array(seq_vecs_list)

    if return_seqs:
        return seq_vecs, seq_headers, seq_snps, seqs
    else:
        return seq_vecs, seq_headers, seq_snps


def dna_length_1hot(seq, length):
    """Adjust the sequence length and compute
    a 1hot coding."""

    if length < len(seq):
        # trim the sequence
        seq_trim = (len(seq) - length) // 2
        seq = seq[seq_trim : seq_trim + length]

    elif length > len(seq):
        # extend with N's
        nfront = (length - len(seq)) // 2
        nback = length - len(seq) - nfront
        seq = "N" * nfront + seq + "N" * nback

    # n_uniform required to avoid different
    #   random nucleotides for each allele
    seq_1hot = dna.dna_1hot(seq, n_uniform=True)

    return seq_1hot, seq


def vcf_count(vcf_file):
    """Count SNPs in a VCF file"""
    if vcf_file[-3:] == ".gz":
        vcf_in = gzip.open(vcf_file, "rt")
    else:
        vcf_in = open(vcf_file)

    # read through header
    line = vcf_in.readline()
    while line[0] == "#":
        line = vcf_in.readline()

    # count SNPs
    num_snps = 0
    while line:
        num_snps += 1
        line = vcf_in.readline()

    vcf_in.close()

    return num_snps


def vcf_snps(
    vcf_file,
    require_sorted=False,
    validate_ref_fasta=None,
    flip_ref=False,
    pos2=False,
    start_i=None,
    end_i=None,
):
    """Load SNPs from a VCF file"""
    if vcf_file[-3:] == ".gz":
        vcf_in = gzip.open(vcf_file, "rt")
    else:
        vcf_in = open(vcf_file)

    # read through header
    line = vcf_in.readline()
    while line[0] == "#":
        line = vcf_in.readline()

    # to check sorted
    if require_sorted:
        seen_chrs = set()
        prev_chr = None
        prev_pos = -1

    # to check reference
    if validate_ref_fasta is not None:
        genome_open = pysam.Fastafile(validate_ref_fasta)

    # read in SNPs
    snps = []
    si = 0
    while line:
        if start_i is None or start_i <= si < end_i:
            snps.append(SNP(line, pos2))

            if require_sorted:
                if prev_chr is not None:
                    # same chromosome
                    if prev_chr == snps[-1].chr:
                        if snps[-1].pos < prev_pos:
                            print(
                                "Sorted VCF required. Mis-ordered position: %s"
                                % line.rstrip(),
                                file=sys.stderr,
                            )
                            exit(1)
                    elif snps[-1].chr in seen_chrs:
                        print(
                            "Sorted VCF required. Mis-ordered chromosome: %s"
                            % line.rstrip(),
                            file=sys.stderr,
                        )
                        exit(1)

                seen_chrs.add(snps[-1].chr)
                prev_chr = snps[-1].chr
                prev_pos = snps[-1].pos
            if validate_ref_fasta is not None:
                ref_n = len(snps[-1].ref_allele)
                snp_pos = snps[-1].pos - 1
                ref_snp = genome_open.fetch(
                    snps[-1].chr, snp_pos, snp_pos + ref_n
                ).upper()
                if snps[-1].ref_allele != ref_snp:
                    if not flip_ref:
                        # bail
                        print(
                            "ERROR: %s does not match reference %s"
                            % (snps[-1], ref_snp),
                            file=sys.stderr,
                        )
                        exit(1)

                    else:
                        alt_n = len(snps[-1].alt_alleles[0])
                        ref_snp = genome_open.fetch(
                            snps[-1].chr, snp_pos, snp_pos + alt_n
                        ).upper()

                        # if alt matches fasta reference
                        if snps[-1].alt_alleles[0] == ref_snp:
                            # flip alleles
                            snps[-1].flip_alleles()

                        else:
                            # bail
                            print(
                                "ERROR: %s does not match reference %s"
                                % (snps[-1], ref_snp),
                                file=sys.stderr,
                            )
                            exit(1)

        si += 1
        line = vcf_in.readline()

    vcf_in.close()

    return snps


def vcf_sort(vcf_file):
    # move
    os.rename(vcf_file, "%s.tmp" % vcf_file)

    # print header
    vcf_out = open(vcf_file, "w")
    print("##fileformat=VCFv4.0", file=vcf_out)
    vcf_out.close()

    # sort
    subprocess.call("bedtools sort -i %s.tmp >> %s" % (vcf_file, vcf_file), shell=True)

    # clean
    os.remove("%s.tmp" % vcf_file)


class SNP:
    """SNP

    Represent SNPs read in from a VCF file

    Attributes:
        vcf_line (str)
    """

    def __init__(self, vcf_line, pos2=False):
        # Split the line by tabs; standard VCF columns are:
        # [0]CHROM, [1]POS, [2]ID, [3]REF, [4]ALT, [5]QUAL, [6]FILTER, [7]INFO, ...
        columns = vcf_line.strip().split('\t')
        
        # Some VCFs might have fewer than 8 columns if they're malformed,
        # so be sure to check or handle errors accordingly.
        if len(columns) < 8:
            raise ValueError(f"Invalid VCF line (fewer than 8 columns): {vcf_line}")

        # 1) Chromosome
        chrom = columns[0]
        if not chrom.startswith("chr"):
            chrom = f"chr{chrom}"
        self.chr = chrom

        # 2) Position
        self.pos = int(columns[1])

        # 3) rsid
        self.rsid = columns[2]
        if self.rsid == ".":
            self.rsid = f"{self.chr}:{self.pos}"

        # 4) REF
        self.ref_allele = columns[3]

        # 5) ALT (could be multiple comma-separated)
        self.alt_alleles = columns[4].split(",")
        self.alt_allele = self.alt_alleles[0]  # We only handle the first alt explicitly
        self.flipped = False

        # (Optional) POS2
        self.pos2 = None
        if pos2:
            # some code references columns[5], but be careful if your VCF has standard fields in that column
            # You can adapt as needed
            self.pos2 = int(columns[5])

        self.gene = None
        if len(columns) == 8:
            # Parse INFO field to find 'GENE=' if present
            info_field = columns[7]
            info_parts = info_field.split(";")
            for kv in info_parts:
                if kv.startswith("GENE="):
                    self.gene = kv.split("=", 1)[1]
                    break

    def flip_alleles(self):
        """Flip reference and first alt allele."""
        assert len(self.alt_alleles) == 1
        self.ref_allele, self.alt_alleles[0] = self.alt_alleles[0], self.ref_allele
        self.alt_allele = self.alt_alleles[0]
        self.flipped = True

    def get_alleles(self):
        """Return a list of all alleles"""
        alleles = [self.ref_allele] + self.alt_alleles
        return alleles

    def indel_size(self):
        """Return the size of the indel."""
        return len(self.alt_allele) - len(self.ref_allele)

    def longest_alt(self):
        """Return the longest alt allele."""
        return max([len(al) for al in self.alt_alleles])

    def __str__(self):
        return "SNP(%s, %s:%d, %s/%s)" % (
            self.rsid,
            self.chr,
            self.pos,
            self.ref_allele,
            ",".join(self.alt_alleles),
        )
