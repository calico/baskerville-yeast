from optparse import OptionParser

def fasta_to_length_dict(filename):
    """
    Reads a FASTA file and returns a dictionary with chromosome names as keys
    and sequence lengths as values.
    """
    chrom_lengths = {}
    current_chrom = None
    current_seq = []

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            # Check if line is a header line
            if line.startswith(">"):
                # If there is a current chromosome, save its sequence length
                if current_chrom is not None:
                    chrom_lengths[current_chrom] = len(''.join(current_seq))
                # Get the chromosome name from header (using the first word after '>')
                current_chrom = line[1:].split()[0]
                current_seq = []  # reset for the next sequence
            else:
                # Accumulate the sequence lines
                current_seq.append(line)
        # Don't forget to add the last chromosome after file loop ends
        if current_chrom is not None:
            chrom_lengths[current_chrom] = len(''.join(current_seq))

    return chrom_lengths

# Example usage:
if __name__ == "__main__":
    usage = (
        "usage: %prog [options] -f <fasta_file>"
    )
    parser = OptionParser(usage)
    parser.add_option(
        "-f",
        dest="fasta_file",
        default=None,
        type="string",
        help="Input FASTA file [Default: %default]",
    )
    (options, args) = parser.parse_args()
    fasta_file = options.fasta_file
    lengths = fasta_to_length_dict(fasta_file)
    print(lengths)
