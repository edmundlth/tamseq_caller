import argparse
import os
from dataset import Dataset

DEFAULT_FILE_EXT = ".counts.csv"

def main():
    args = parse_args()
    args.func(args)
    return


def runcmd_pseudocount(args):
    extension = args.file_ext
    sample_name_to_filepaths = {
        os.path.basename(filepath).replace(extension, ''): filepath
        for filepath in args.control_datafiles
    }
    ds = Dataset(sample_name_to_filepaths, controls=sample_name_to_filepaths.keys(), sep=',')
    ds.get_df_raw()
    ds.get_df_base_count()
    ds.fit_control_dirichlet_model()
    ds.df_pseudocount.to_csv(args.output_file, sep='\t')
    return


def runcmd_variants(args):
    return


def runcmd_filter(args):
    return


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run variant calling tools."
    )

    subparsers = parser.add_subparsers()
    # Fitting dirichlet model on control data.
    parser_pseudocount = subparsers.add_parser(
        "pseudocount",
        help="Fit Dirichlet allele pseudocount on control data."
    )
    parser_pseudocount.add_argument(
        '--control_datafiles', metavar="FILE", type=str, required=True, nargs="*",
        help='Allele count outputs for control samples.'
    )
    parser_pseudocount.add_argument(
        '--output_file', metavar="FILE", type=str, required=True, nargs=1,
        help='Output file.'
    )
    parser_pseudocount.add_argument(
        '--max_num_iter', metavar="NUM", type=int, required=False,
        default=10000,
        help=''
    )
    parser_pseudocount.add_argument(
        "--eps", metavar="EPSILON", type=float, required=False,
        default=1e-6
    )
    parser_pseudocount.add_argument(
        "--file_ext", metavar="EXTENSION", type=str, required=False,
        default=DEFAULT_FILE_EXT,
        help="File extension of input files. "
             "Sample names are parsed by removing the file extension from file names."
             f"Default: {DEFAULT_FILE_EXT}"
    )
    parser_pseudocount.set_defaults(func=runcmd_pseudocount)

    # Variant calling
    parser_variant_calling = subparsers.add_parser(
        "variants"
    )
    parser_variant_calling.set_defaults(func=runcmd_variants)

    # Filter variants
    parser_filter_variant = subparsers.add_parser(
        "filter"
    )
    parser_filter_variant.set_defaults(func=runcmd_filter)

    return parser.parse_args()


if __name__ == "__main__":
    main()
