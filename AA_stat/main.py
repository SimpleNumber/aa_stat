import argparse
import logging
import os
from collections import Counter, defaultdict
try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser
import pandas as pd
from pyteomics import mass

from . import AA_stat, locTools, utils, osPipe


def main():
    pars = argparse.ArgumentParser()
    subparsers = pars.add_subparsers()
#    def_pars = subparsers.add_parser('def')
    pipe_pars = subparsers.add_parser('pipe')

    pipe_pars.add_argument('MSFragger', help='Path to .jar file of MSFragger search engine.')
    pars.add_argument('--params', help='CFG file with parameters. If there is no file, AA_stat uses default one.'
        'An example can be found at https://github.com/SimpleNumber/aa_stat',
        required=False)
#    print(os.__file__)
    pars.add_argument('--dir', help='Directory to store the results. '
        'Default value is current directory.', default='.')
    pars.add_argument('-v', '--verbosity', type=int, choices=range(3), default=1, help='Output verbosity')

    input_spectra = pars.add_mutually_exclusive_group()
    input_spectra.add_argument('--mgf',  nargs='+', help='MGF files to localize modifications')
    input_spectra.add_argument('--mzML',  nargs='+', help='mzML files to localize modifications')

    input_file = pars.add_mutually_exclusive_group()
    input_file.add_argument('--pepxml', nargs='+', help='List of input files in pepXML format')
    input_file.add_argument('--csv', nargs='+', help='List of input files in CSV format')
    
    print(pipe_pars)
    args = pars.parse_args()
    

    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    logging.basicConfig(format='{levelname:>8}: {asctime} {message}',
                        datefmt='[%H:%M:%S]', level=levels[args.verbosity], style='{')
    logger = logging.getLogger(__name__)
    logger.info("Starting...")


    params = ConfigParser(delimiters=('=', ':'),
                          comment_prefixes=('#'),
                          inline_comment_prefixes=('#'))

    params.read(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'example.cfg'))
    if args.params:
        if not os.path.isfile(args.params):
            logger.error('PARAMETERS FILE NOT FOUND: %s', args.params)
        params.read(args.params)
    else:
        logger.info('Using default parameters')
#        params.read(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'example.cfg'))
    params_dict = utils.get_parameters(params)
    utils.set_additional_params(params_dict)
    params_dict['out_dir'] = args.dir
#    print(params_dict)

    AA_stat.AA_stat(params_dict, args)
    
#    AA_Stat
    
