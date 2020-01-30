#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import os
from . import AA_stat, utils
import argparse
import logging
try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser
from collections import defaultdict
"""
Created on Sun Jan 26 15:41:40 2020

@author: julia
"""
OS_PARAMS_DEF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'open_search.params')
FIX_MOD_ZERO_THRESH = 2 #in %
FIX_MOD_THRESH = 80 # in %


dict_aa = {
    'G' : 'add_G_glycine',
    'A' : 'add_A_alanine',
    'S' : 'add_S_serine',
    'P' : 'add_P_proline',
    'V' : 'add_V_valine',
    'T' : 'add_T_threonine',
    'C' : 'add_C_cysteine',
    'L' : 'add_L_leucine',
    'I' : 'add_I_isoleucine',
    'N' : 'add_N_asparagine',
    'D' : 'add_D_aspartic_acid',
    'Q' : 'add_Q_glutamine',
    'K' : 'add_K_lysine',
    'E' : 'add_E_glutamic_acid',
    'M' : 'add_M_methionine',
    'H' : 'add_H_histidine',
    'F' : 'add_F_phenylalanine',
    'R' : 'add_R_arginine',
    'Y' : 'add_Y_tyrosine',
    'W' : 'add_W_tryptophan'
    }

def main():
    pars = argparse.ArgumentParser()
    pars.add_argument('--params', help='CFG file with parameters. If there is no file, AA_stat uses default one.'
        'An example can be found at https://github.com/SimpleNumber/aa_stat',
        required=False)
    pars.add_argument('--MSFragger_path', help ='Path to MSFragger .jar file.', required=True)
    pars.add_argument('--dir', help='Directory to store the results. Default value is current directory.', default='.')
    pars.add_argument('-v', '--verbosity', type=int, choices=range(3), default=1, help='Output verbosity')

    input_spectra = pars.add_mutually_exclusive_group(required=True)
    input_spectra.add_argument('--mgf',  nargs='+', help='MGF files to localize modifications', default=None)
    input_spectra.add_argument('--mzML',  nargs='+', help='mzML files to localize modifications', default=None)
    
    input_os_par = pars.add_mutually_exclusive_group(required=True)
    input_os_par.add_argument('--fasta', help='Fasta file with decoys for open search. Default decoy prefix is "DECOY_".\
                              If it differs, do not forget to specify it in AA_stat params file')
    input_os_par.add_argument('--os_params', help='Custom open search parameters.')

    args = pars.parse_args()
    
    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    logging.basicConfig(format='{levelname:>8}: {asctime} {message}',
                        datefmt='[%H:%M:%S]', level=levels[args.verbosity], style='{')
    logger = logging.getLogger(__name__)
    logger.info("Starting MSFragger and AA_stat pipeline.")
    spectra = args.mgf or args.mzML
    spectra = [os.path.abspath(i) for i in spectra]
    params = ConfigParser(delimiters=('=', ':'),
                          comment_prefixes=('#'),
                          inline_comment_prefixes=('#'))

    params.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'example.cfg'))
    if args.params:
        if not os.path.isfile(args.params):
            logger.error('PARAMETERS FILE NOT FOUND: %s', args.params)
        params.read(pars.params)
    else:
        logger.info('Using default parameters for AA_stat.')
#    print(params)
    params_dict = utils.get_parameters(params)
    utils.set_additional_params(params_dict)
    working_dir = args.dir
    if args.os_params:
        logger.info('Custom paramenters for open search')
        folder_name = 'custom_os'
#        print(args)
        os.makedirs(os.path.abspath(os.path.join(args.dir, folder_name)), exist_ok=True)
        run_os(spectra, args.MSFragger_path, os.path.join(args.dir, folder_name), parameters=args.os_params)
        args.pepxml = [os.path.join(os.path.abspath(os.path.join(args.dir, folder_name)), x) \
                       for x in os.listdir(os.path.join(args.dir, folder_name)) if x.endswith('.pepXML')]
        args.dir = os.path.join(os.path.abspath(args.dir), folder_name, 'aa_stat_res')
        AA_stat.AA_stat(params_dict, args)
    else:
        logger.info('No parameters for open search. Start two spet optimization.')
        logger.info('Start preliminary open search.')
        preliminary_aastat = run_step_os(spectra, 'preliminary_os', working_dir, args, params_dict, change_dict=None )
        aa_rel = preliminary_aastat[utils.mass_format(0)][2] 
        final_cand = defaultdict(list)
        candidates = aa_rel[aa_rel < FIX_MOD_ZERO_THRESH].index
        for ms, data in preliminary_aastat.items():
            if ms != utils.mass_format(0):
                for i in candidates:
                    if data[2][i] > FIX_MOD_THRESH:
                        final_cand[i].append((ms, data[0]))
                    else:
                        print(ms, data[2][i])
#        print(final_cand)
        fix_mod_dict = {}
        for k, v in final_cand.items():
            sorted_v = sorted(v, key=lambda x: x[1], reverse=True)
            fix_mod_dict[k] = sorted_v[0][0]
#        print(fix_mod_dict)        
        logger.info('Start second open search with fixed modifications %s', fix_mod_dict)
        run_step_os(spectra, 'second_os', working_dir, args, params_dict, change_dict=fix_mod_dict )

def run_os(spectra, msf_path, save_dir, parameters=None):
    if not parameters:
        logging.error('No os parameters')
    else:
        subprocess.call(['java', '-jar', msf_path, parameters, *spectra])
        directory = os.path.dirname(spectra[0])
        os.makedirs(save_dir, exist_ok=True)
        for f in os.listdir(directory):
            if f.endswith('.pepXML'):
                os.rename(os.path.join(directory,f), os.path.join(save_dir, f))
def run_step_os(spectra, folder_name, working_dir, args, params_dict, change_dict=None):
    os.makedirs(os.path.abspath(os.path.join(working_dir, folder_name)), exist_ok=True)
    os_params_path = os.path.abspath(os.path.join(working_dir, folder_name, 'os.params'))
    with open(os_params_path, 'w') as new_params:
        with open(OS_PARAMS_DEF, 'r') as default:
            for l in default.readlines():
                print(l)
                if 'database_name' in l:
                        new_params.write(' '.join(['database_name = ', (args.fasta),'\n']))
                elif change_dict:
                    for aa, ms in change_dict.items():
                        if dict_aa[aa] in l:
                            new_params.write(' '.join([dict_aa[aa],'=', ms,'\n']))
                        else:
                            new_params.write(l)  
                else:
                    new_params.write(l)
#    print(folder_name, working_dir)
#    print(os.path.abspath(os.path.join(working_dir, folder_name)))
    subprocess.call(['java', '-jar', args.MSFragger_path, os_params_path, *spectra])
    directory = os.path.dirname(spectra[0])
    for f in os.listdir(directory):
        if f.endswith('.pepXML'):
            os.rename(os.path.join(directory,f), os.path.join(os.path.dirname(os_params_path), f))
    args.pepxml =[os.path.join(os.path.abspath(os.path.join(working_dir, folder_name)), x) \
                       for x in \
                       os.listdir(os.path.abspath(os.path.join(working_dir, folder_name)))\
                       if x.endswith('.pepXML')]
    os.makedirs(os.path.join(os.path.abspath(working_dir), folder_name, 'aa_stat_res'), exist_ok=True)
    args.dir = os.path.join(os.path.abspath(working_dir), folder_name, 'aa_stat_res')
#    
#    print(args)
    return AA_stat.AA_stat(params_dict, args)

