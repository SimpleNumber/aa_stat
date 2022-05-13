import argparse
import logging
import os
from . import AA_stat, utils, io


def main():
    pars = argparse.ArgumentParser(epilog='Instead of file lists, you can pass directory names. '
        'This will process all files in the directory.')
    pars.add_argument('--params', help='CFG file with parameters. If there is no file, AA_stat uses default one. '
        'An example can be found at https://github.com/SimpleNumber/aa_stat', required=False)

    pars.add_argument('--dir', help='Directory to store the results. Default value is current directory.', default='.')
    pars.add_argument('-v', '--verbosity', type=int, choices=range(4), default=1, help='Output verbosity.')

    input_spectra = pars.add_mutually_exclusive_group()
    input_spectra.add_argument('--mgf', nargs='+', help='MGF files to localize modifications.')
    input_spectra.add_argument('--mzml', nargs='+', help='mzML files to localize modifications.')

    input_file = pars.add_mutually_exclusive_group(required=True)
    input_file.add_argument('--pepxml', nargs='+', help='List of input files in pepXML format.')
    input_file.add_argument('--csv', nargs='+', help='List of input files in CSV format.')

    args = pars.parse_args()
    levels = [logging.WARNING, logging.INFO, logging.DEBUG, utils.INTERNAL]
    logging.basicConfig(format='{levelname:>8}: {asctime} {message}',
                        datefmt='[%H:%M:%S]', level=levels[args.verbosity], style='{')

    # Performance optimizations as per https://docs.python.org/3/howto/logging.html#optimization
    logging._srcfile = None
    logging.logThreads = 0
    logging.logProcesses = 0
    logger = logging.getLogger(__name__)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    logger.info('Starting...')

    io.resolve_filenames(args)
    total_inputs = len(args.csv or []) + len(args.pepxml or [])
    if not total_inputs:
        logger.error('No input files found.')
        return
    params_dict = io.get_params_dict(args)
    logger.debug(params_dict)

    os.makedirs(args.dir, exist_ok=True)
    AA_stat.AA_stat(params_dict, args)
    logger.info('Done.')
