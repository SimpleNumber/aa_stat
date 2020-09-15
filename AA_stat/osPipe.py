import subprocess
import os
import shutil
from . import AA_stat, utils
import argparse
import logging
import sys

"""
Created on Sun Jan 26 15:41:40 2020

@author: julia
"""
OS_PARAMS_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'open_search.params')

logger = logging.getLogger(__name__)

DICT_AA = {
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
    'add_Cterm_peptide'   : 'C-term',
    'add_Nterm_peptide'   : 'N-term',
}


def main():
    pars = argparse.ArgumentParser()
    pars.add_argument('--params', help='CFG file with parameters. If there is no file, AA_stat uses default one. '
                      'An example can be found at https://github.com/SimpleNumber/aa_stat', required=False)
    pars.add_argument('--MSFragger', help='Path to MSFragger .jar file. '
                      'If not specified, MSFRAGGER environment variable is used.')
    pars.add_argument('--dir', help='Directory to store the results. Default value is current directory.', default='.')
    pars.add_argument('-v', '--verbosity', type=int, choices=range(4), default=1, help='Output verbosity.')

    input_spectra = pars.add_mutually_exclusive_group(required=True)
    input_spectra.add_argument('--mgf', nargs='+', help='MGF files to search.', default=None)
    input_spectra.add_argument('--mzml', nargs='+', help='mzML files to search.', default=None)

    pars.add_argument('-db', '--fasta', help='Fasta file with decoys for open search. Default decoy prefix is "DECOY_".'
                      ' If it differs, do not forget to specify it in AA_stat params file.')
    pars.add_argument('--os-params', help='Custom open search parameters.')
    pars.add_argument('-x', '--optimize-fixed-mods',
                      help='Run multiple searches, automatically determine which fixed modifications to apply.',
                      action='store_true', default=False)
    pars.add_argument('-s', '--skip', help='Skip search if pepXML files exist already. If not specified, '
                      'no steps are skipped. If specified without value, first step may be skipped. '
                      'Value is number of steps to skip. Only works with "-x".',
                      nargs='?', default=0, const=1, type=int)
    pars.add_argument('-je', '--java-executable', default='java')
    pars.add_argument('-ja', '--java-args', default='')

    args = pars.parse_args()

    levels = [logging.WARNING, logging.INFO, logging.DEBUG, utils.INTERNAL]
    logging.basicConfig(format='{levelname:>8}: {asctime} {message}',
                        datefmt='[%H:%M:%S]', level=levels[args.verbosity], style='{')

    if not args.MSFragger:
        args.MSFragger = os.environ.get('MSFRAGGER')
    if not args.MSFragger:
        logger.critical('Please specify --MSFragger or set MSFRAGGER environment variable.')
        sys.exit(1)

    logger.info("Starting MSFragger and AA_stat pipeline.")
    spectra = args.mgf or args.mzml
    spectra = [os.path.abspath(i) for i in spectra]

    params_dict = utils.get_params_dict(args.params)

    working_dir = args.dir

    if args.optimize_fixed_mods:
        logger.debug('Skipping up to %d steps.', args.skip)
        step = 1
        fix_mod_dict = {}
        while True:
            logger.info('Starting step %d.', step)
            fig_data, aastat_table, locmod, data_dict, new_fix_mod_dict, var_mod_dict = run_step_os(
                spectra, 'os_step_{}'.format(step), working_dir, args, params_dict, change_dict=fix_mod_dict, step=step)

            if new_fix_mod_dict:
                for k, v in new_fix_mod_dict.items():
                    fix_mod_dict.setdefault(k, 0.)
                    fix_mod_dict[k] += data_dict[v][0]
                step += 1
            else:
                break
        try:
            if os.path.isfile(os.path.join(working_dir, 'report.html')):
                logger.debug('Removing existing report.html.')
                os.remove(os.path.join(working_dir, 'report.html'))
            os.symlink(os.path.join('os_step_1', 'report.html'), os.path.join(working_dir, 'report.html'))
        except Exception as e:
            logger.debug('Can\'t create symlink to report: %s', e)
        else:
            logger.debug('Symlink created successfully.')
        logger.info('Stopping after %d steps.', step)
    else:
        logger.info('Running one-shot search.')
        folder_name = ''
        run_step_os(spectra, folder_name, args.dir, args, params_dict)


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
            elif mass_shifts and DICT_AA.get(key) in mass_shifts:
                aa = DICT_AA[key]
                new_params.write(key + ' = ' + str(mass_shifts[aa]) + '\n')
            else:
                new_params.write(line)


def run_step_os(spectra, folder_name, working_dir, args, params_dict, change_dict=None, step=None):
    dir = os.path.abspath(os.path.join(working_dir, folder_name))
    os.makedirs(dir, exist_ok=True)
    os_params_path = os.path.abspath(os.path.join(working_dir, folder_name, 'os.params'))
    create_os_params(os_params_path, args.os_params, change_dict, args.fasta)
    pepxml_names = [get_pepxml(s, dir) for s in spectra]
    run = True
    if step is not None:
        if step <= args.skip:
            run = not all(os.path.isfile(f) for f in pepxml_names)
            logger.debug('On step %d, need to run search: %s', step, run)
        else:
            logger.debug('Can\'t skip step %d, running.', step)
    if run:
        run_os(args.java_executable, args.java_args.split(), spectra, args.MSFragger, dir, os_params_path)
    else:
        logger.info('Skipping search.')
    args.pepxml = pepxml_names
    args.csv = None
    args.dir = dir
    return AA_stat.AA_stat(params_dict, args, step=step)
