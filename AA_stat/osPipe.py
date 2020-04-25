import subprocess
import os
import shutil
from . import AA_stat, utils
import argparse
import logging
import sys
import numpy as np

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
    'add_Cterm_peptide'   : 'C-term',
    'add_Nterm_peptide'   : 'N-term',
}


def main():
    pars = argparse.ArgumentParser()
    pars.add_argument('--params', help='CFG file with parameters. If there is no file, AA_stat uses default one. '
        'An example can be found at https://github.com/SimpleNumber/aa_stat',
        required=False)
    pars.add_argument('--MSFragger', help ='Path to MSFragger .jar file. '
        'If not specified, MSFRAGGER environment variable is used.')
    pars.add_argument('--dir', help='Directory to store the results. Default value is current directory.', default='.')
    pars.add_argument('-v', '--verbosity', type=int, choices=range(4), default=1, help='Output verbosity.')

    input_spectra = pars.add_mutually_exclusive_group(required=True)
    input_spectra.add_argument('--mgf',  nargs='+', help='MGF files to search.', default=None)
    input_spectra.add_argument('--mzml',  nargs='+', help='mzML files to search.', default=None)

    pars.add_argument('-db', '--fasta', help='Fasta file with decoys for open search. Default decoy prefix is "DECOY_".'
                              'If it differs, do not forget to specify it in AA_stat params file.')
    pars.add_argument('--os-params', help='Custom open search parameters.')
    pars.add_argument('-x', '--optimize-fixed-mods',
        help='Run two searches, use the first one to determine which fixed modifications to apply.',
        action='store_true', default=False)
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
    params = utils.read_config_file(args.params)

    params_dict = utils.get_parameters(params)
    utils.set_additional_params(params_dict)

    working_dir = args.dir

    if args.optimize_fixed_mods:
        step = 1
        fix_mod_dict = {}
        while True:
            logger.info('Starting step %d.', step)
            logger.info('Starting preliminary open search.')
            fig_data, aastat_table, locmod, data_dict = run_step_os(
                spectra, 'os_step_{}'.format(step), working_dir, args, params_dict, change_dict=fix_mod_dict)

            new_fix_mod_dict = determine_fixed_mods(fig_data, aastat_table, locmod, data_dict, params_dict)

            if new_fix_mod_dict:
                for k, v in new_fix_mod_dict.items():
                    fix_mod_dict.setdefault(k, 0.)
                    fix_mod_dict[k] += v
                logger.info('Determined fixed modifications: %s', fix_mod_dict)
                step += 1
            else:
                logger.info('No fixed modifications found.')
                break
        logger.info('Stopping after %d steps.', step)
    else:
        logger.info('Running one-shot search.')
        folder_name = ''
        run_step_os(spectra, folder_name, args.dir, args, params_dict, None)


# def get_full_sum(ms_label, locmod_df, seen=set()):
#     sumof = locmod_df.at[ms_label, 'sum of mass shifts']
#     seen.add(ms_label)
#     if sumof is not np.nan:
#         logger.debug('Sum of shifts for %s: %s', ms_label, sumof)
#         for pair in sumof:
#             for shift in pair:
#                 if shift not in seen:
#                     seen.add(shift)
#                     seen.update(get_full_sum(shift, locmod_df, seen))
#     if locmod_df.at[ms_label, 'is isotope']:
#         seen.update(get_full_sum(locmod_df.at[ms_label, 'isotope_ind']))
#     return seen


def get_fixed_mod_raw(aa, data_dict, choices=None):
    dist_aa = []
    for ms, v in data_dict.items():
        if choices is None or ms in choices:
            dist_aa.append([v[0], v[1]['peptide'].apply(lambda x: x.count(aa)).sum()])
    utils.internal('Counts for %s: %s', aa, dist_aa)
    top_shift = max(dist_aa, key=lambda tup: tup[1])
    return utils.mass_format(top_shift[0])


def get_fix_mod_from_l10n(mslabel, locmod_df):
    l10n = locmod_df.at[mslabel, 'localization']
    logger.debug('Localizations for %s: %s', mslabel, l10n)
    if l10n:
        l10n.pop('non-localized', None)
        top_loc = max(l10n, key=l10n.get)
        logger.debug('Top localization label for %s: %s', mslabel, top_loc)
        return top_loc


def parse_l10n_site(site):
    aa, shift = site.split('_')
    return aa, shift

def determine_fixed_mods(aastat_result, aastat_df, locmod_df, data_dict, params_dict):
    reference = aastat_df.loc[aastat_df['is reference']].index[0]
    fix_mod_dict = {}
    if reference == utils.mass_format(0):
        logger.info('Reference bin is at zero shift.')

        aa_rel = aastat_result[reference][2]
        utils.internal('aa_rel:\n%s', aa_rel)
        candidates = aa_rel[aa_rel < FIX_MOD_ZERO_THRESH].index
        logger.debug('Fixed mod candidates: %s', candidates)
        for i in candidates:
            candidate_label = get_fixed_mod_raw(i, data_dict)
            if aastat_result[candidate_label][2][i] > FIX_MOD_ZERO_THRESH:
                fix_mod_dict[i] = data_dict[candidate_label][0]
            else:
                logger.info('Could not find %s anywhere. Can\'t fix.', i)
    else:
        logger.info('Reference bin is at %s. Looking for fixed modification to compensate.', reference)
        utils.internal('Localizations for %s: %s', reference, locmod_df.at[reference, 'localization'])
        loc = get_fix_mod_from_l10n(reference, locmod_df)
        label = reference
        while loc is None:
            del data_dict[label]
            label = max(data_dict, key=lambda k: data_dict[k][1].shape[0])
            loc = get_fix_mod_from_l10n(label, locmod_df)
            logger.debug('No luck. Trying %s. Got %s', label, loc)
            if not data_dict:
                break
        if loc:
            aa, shift = parse_l10n_site(loc)
            fix_mod_dict[aa] = data_dict[shift][0]
        else:
            logger.info('No localizations. Stopping.')

    return fix_mod_dict


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
                new_params.write(key + ' = ' + str(mass_shifts[aa]) + '\n')
            else:
                new_params.write(line)


def run_step_os(spectra, folder_name, working_dir, args, params_dict, change_dict=None):
    dir = os.path.abspath(os.path.join(working_dir, folder_name))
    os.makedirs(dir, exist_ok=True)
    os_params_path = os.path.abspath(os.path.join(working_dir, folder_name, 'os.params'))
    create_os_params(os_params_path, args.os_params, change_dict, args.fasta)
    # run_os(args.java_executable, args.java_args.split(), spectra, args.MSFragger, dir, os_params_path)
    args.pepxml = [get_pepxml(s, dir) for s in spectra]
    args.csv = None
    args.dir = dir
    return AA_stat.AA_stat(params_dict, args)
