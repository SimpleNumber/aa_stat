import subprocess
import os
import shutil
from . import AA_stat, utils
import argparse
import logging
import sys
#from collections import defaultdict
"""
Created on Sun Jan 26 15:41:40 2020

@author: julia
"""
OS_PARAMS_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'open_search.params')

FIX_MOD_ZERO_THRESH = 2 #in %
#FIX_MOD_THRESH = 90 # in %

logger = logging.getLogger(__name__)

dict_aa = {
    'add_G_glycine'       : 'G',
    'add_A_alanine'       : 'A',
    'add_S_serine'        : 'S',
    'add_P_proline'       : 'P',
    'add_V_valine'        : 'V',
    'add_T_threonine'     : 'T',
    'add_C_cysteine'      : 'C',
    'add_L_leucine'       : 'L',
    'add_I_isoleucine'    : 'I',
    'add_N_asparagine'    : 'N',
    'add_D_aspartic_acid' : 'D',
    'add_Q_glutamine'     : 'Q',
    'add_K_lysine'        : 'K',
    'add_E_glutamic_acid' : 'E',
    'add_M_methionine'    : 'M',
    'add_H_histidine'     : 'H',
    'add_F_phenylalanine' : 'F',
    'add_R_arginine'      : 'R',
    'add_Y_tyrosine'      : 'Y',
    'add_W_tryptophan'    : 'W',
}


def main():
    pars = argparse.ArgumentParser()
    pars.add_argument('--params', help='CFG file with parameters. If there is no file, AA_stat uses default one. '
        'An example can be found at https://github.com/SimpleNumber/aa_stat',
        required=False)
    pars.add_argument('--MSFragger', help ='Path to MSFragger .jar file. '
        'If not specified, MSFRAGGER environment variable is used.')
    pars.add_argument('--dir', help='Directory to store the results. Default value is current directory.', default='.')
    pars.add_argument('-v', '--verbosity', type=int, choices=range(3), default=1, help='Output verbosity.')

    input_spectra = pars.add_mutually_exclusive_group(required=True)
    input_spectra.add_argument('--mgf',  nargs='+', help='MGF files to search.', default=None)
    input_spectra.add_argument('--mzML',  nargs='+', help='mzML files to search.', default=None)

    pars.add_argument('--fasta', help='Fasta file with decoys for open search. Default decoy prefix is "DECOY_".'
                              'If it differs, do not forget to specify it in AA_stat params file.')
    pars.add_argument('--os-params', help='Custom open search parameters.')
    pars.add_argument('-x', '--optimize-fixed-mods',
        help='Run two searches, use the first one to determine which fixed modifications to apply.',
        action='store_true', default=False)
    pars.add_argument('-je', '--java-executable', default='java')
    pars.add_argument('-ja', '--java-args', default='')

    args = pars.parse_args()

    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    logging.basicConfig(format='{levelname:>8}: {asctime} {message}',
                        datefmt='[%H:%M:%S]', level=levels[args.verbosity], style='{')

    if not args.MSFragger:
        args.MSFragger = os.environ.get('MSFRAGGER')
    if not args.MSFragger:
        logger.critical('Please specify --MSFragger or set MSFRAGGER environment variable.')
        sys.exit(1)

    logger.info("Starting MSFragger and AA_stat pipeline.")
    spectra = args.mgf or args.mzML
    spectra = [os.path.abspath(i) for i in spectra]
    params = utils.read_config_file(args.params)

    params_dict = utils.get_parameters(params)
    utils.set_additional_params(params_dict)

    working_dir = args.dir

    if args.optimize_fixed_mods:
        logger.info('Starting two-step procedure.')
        logger.info('Starting preliminary open search.')
        preliminary_aastat, data_dict = run_step_os(spectra, 'preliminary_os', working_dir, args, params_dict, change_dict=None)
        logger.debug('Preliminary AA_stat results:\n%s', preliminary_aastat)
        aa_rel = preliminary_aastat[utils.mass_format(0)][2]
        logger.debug('aa_rel:\n%s', aa_rel)
        fix_mod_dict = {}
        candidates = aa_rel[aa_rel < FIX_MOD_ZERO_THRESH].index
        logger.debug('Fixed mod candidates: %s', candidates)

        for i in candidates:
            dist_aa = []
            for ms, v in data_dict.items():    
                v[1]['peptide'].apply(lambda x: x.count(i)).sum()
                dist_aa.append([v[0], v[1]['peptide'].apply(lambda x: x.count(i)).sum()])
            sorted_aa_dist = sorted(dist_aa, key=lambda tup: tup[1], reverse=True)
            fix_mod_dict[i] = utils.mass_format(sorted_aa_dist[0][0])
        
        if fix_mod_dict:
            logger.info('Starting second open search with fixed modifications %s', fix_mod_dict)
            run_step_os(spectra, 'second_os', working_dir, args, params_dict, change_dict=fix_mod_dict)
        else:
            logger.info('No fixed modifications found.')

    else:
        logger.info('Running one-shot search.')
        folder_name = ''
        run_step_os(spectra, folder_name, args.dir, args, params_dict, None)


def get_pepxml(input_file, d=None):
    initial = os.path.splitext(input_file)[0] + '.pepXML'
    if d is None:
        return initial
    sdir, f = os.path.split(initial)
    return os.path.join(d, f)


def run_os(java, jargs, spectra, msfragger, save_dir, parameters):
    command = [java] + jargs + ['-jar', msfragger, parameters, *spectra]
    logger.debug('Running command: %s', ' '.join(command))
    retval = subprocess.call(command)
    logger.debug('Subprocess returned %s', retval)
    if retval:
        logger.critical('MSFragger returned non-zero code %s. Exiting.', retval)
        sys.exit(retval)
    os.makedirs(save_dir, exist_ok=True)
    for s in spectra:
        pepxml = get_pepxml(s)
        if os.path.normpath(os.path.dirname(pepxml)) != os.path.normpath(save_dir):
            logger.debug('Moving %s to %s', pepxml, save_dir)
            shutil.move(pepxml, get_pepxml(s, save_dir))
        else:
            logger.debug('No need to move pepXML file.')


def create_os_params(output, original=None, mass_shifts=None, fastafile=None):
    original = original or OS_PARAMS_DEFAULT
    with open(output, 'w') as new_params, open(original) as default:
        for line in default:
            key = line.split('=')[0].strip()
            if key == 'database_name' and fastafile:
                new_params.write('database_name = {}\n'.format(fastafile))
            elif mass_shifts and dict_aa.get(key) in mass_shifts:
                aa = dict_aa[key]
                new_params.write(key + ' = ' + mass_shifts[aa] + '\n')
            else:
                new_params.write(line)


def run_step_os(spectra, folder_name, working_dir, args, params_dict, change_dict=None):
    dir = os.path.abspath(os.path.join(working_dir, folder_name))
    os.makedirs(dir, exist_ok=True)
    os_params_path = os.path.abspath(os.path.join(working_dir, folder_name, 'os.params'))
    create_os_params(os_params_path, args.os_params, change_dict, args.fasta)
    run_os(args.java_executable, args.java_args.split(), spectra, args.MSFragger, dir, os_params_path)
    args.pepxml = [get_pepxml(s, dir) for s in spectra]
    args.csv = None
    args.dir = dir
    return AA_stat.AA_stat(params_dict, args)
