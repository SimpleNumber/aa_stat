import argparse
import logging
import os
import warnings
from collections import Counter
try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser
import pandas as pd
import numpy as np
from pyteomics import mass

from . import AA_stat, locTools



def main():
    pars = argparse.ArgumentParser()
    pars.add_argument('--params', help='CFG file with parameters.'
        'An example can be found at https://github.com/SimpleNumber/aa_stat',
        required=True)
    pars.add_argument('--dir', help='Directory to store the results. '
        'Default value is current directory.', default='.')
    pars.add_argument('-v', '--verbosity', action='count', default=1, help='Increase output verbosity')

    input_spectra = pars.add_mutually_exclusive_group()
    input_spectra.add_argument('--mgf',  nargs='+', help='MGF files to localize modifications')
    input_spectra.add_argument('--mzML',  nargs='+', help='mzML files to localize modifications')

    input_file = pars.add_mutually_exclusive_group(required=True)
    input_file.add_argument('--pepxml', nargs='+', help='List of input files in pepXML format')
    input_file.add_argument('--csv', nargs='+', help='List of input files in CSV format')
    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    args = pars.parse_args()
    save_directory = args.dir
    level = 2 if args.verbosity >= 2 else args.verbosity
    logging.basicConfig(format='%(levelname)5s: %(asctime)s %(message)s',
                        datefmt='[%H:%M:%S]', level=levels[level])
    logger = logging.getLogger(__name__)
    logger.info("Starting...")


    params = ConfigParser(delimiters=('=', ':'),
                          comment_prefixes=('#'),
                          inline_comment_prefixes=('#'))
    params.read(args.params)
    params_dict = AA_stat.get_parameters(params)
    params_dict = AA_stat.get_additional_params(params_dict) #params_dict 'window'

    data = AA_stat.read_input(args, params_dict)

    hist, popt_pvar = AA_stat.fit_peaks(data, args, params_dict)
#    print(popt_pvar)
#    print('=======================================')
    final_mass_shifts = AA_stat.filter_mass_shifts(popt_pvar)
#    print(final_mass_shifts)
    mass_shift_data_dict = AA_stat.group_specific_filtering(data, final_mass_shifts, params_dict)
#    print('======================',mass_shift_data_dict )
    zero_mass_shift = AA_stat.get_zero_mass_shift(list(mass_shift_data_dict.keys()))

    logger.info("Systematic mass shift equals to %s", AA_stat.mass_format(zero_mass_shift) )
    mass_shift_data_dict = AA_stat.systematic_mass_shift_correction(mass_shift_data_dict, zero_mass_shift)
    if len(mass_shift_data_dict) < 2:
        logger.info('Mass shifts were not found.')
        logger.info('Filtered mass shifts:')
        for i in mass_shift_data_dict.keys():
#            print(mass_shift_data_dict.keys())Da
            logger.info(AA_stat.MASS_FORMAT.format(i))
    else:
        distributions, number_of_PSMs, ms_labels = AA_stat.calculate_statistics(mass_shift_data_dict, 0, params_dict, args)

#    print(mass_shift_data_dict)
    table = AA_stat.save_table(distributions, number_of_PSMs, ms_labels)
#    print(table['mass shift'])
    table.to_csv(os.path.join(save_directory, 'aa_statistics_table.csv'), index=False)
#    print('=======================', table)

    AA_stat.summarizing_hist(table, save_directory)
    logger.info('Summarizing hist prepared')
    AA_stat.render_html_report(table, params_dict, save_directory)
    logger.info('AA_stat results saved to %s', os.path.abspath(args.dir))

    table.index = table['mass shift'].apply(AA_stat.mass_format)
    spectra_dict = AA_stat.read_spectra(args)
    if spectra_dict.keys():
        if args.mgf:
            params_dict['mzml_files'] = False
        else:
            params_dict['mzml_files'] = True
        logger.info('Starting Localization using MS/MS spectra...')
#        print(params_dict['mzml_files'])
        ms_labels = pd.Series(ms_labels)
        locmod_df = pd.DataFrame({'mass shift': ms_labels})
        locmod_df['# peptides in bin'] = table['# peptides in bin']
        locmod_df[['is isotope', 'isotop_ind']] =  locTools.find_isotopes(locmod_df['mass shift'], tolerance=AA_stat.ISOTOPE_TOLERANCE)
        locmod_df['sum of mass shifts'] = locTools.find_modifications(locmod_df.loc[~locmod_df['is isotope'], 'mass shift'])
        locmod_df['sum of mass shifts'].fillna(False, inplace=True)
        locmod_df['aa_stat candidates'] = locTools.get_candidates_from_aastat(table,
                 labels=params_dict['labels'], threshold=AA_stat.AA_STAT_CAND_THRESH)
        u = mass.Unimod().mods
        unimod_db = np.array(u)
        unimod_df = pd.DataFrame(u)
        locmod_df['unimod candidates'] = locmod_df['mass shift'].apply(lambda x: locTools.get_candidates_from_unimod(x, AA_stat.UNIIMOD_TOLERANCE, unimod_db, unimod_df))
        locmod_df['all candidates'] = locmod_df.apply(lambda x: set(x['unimod candidates'])|(set(x['aa_stat candidates'])), axis=1)
        locmod_df.to_csv(os.path.join(save_directory, 'test1.csv'))
        for i in locmod_df.loc[locmod_df['is isotope']].index:
#            print(i)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                locmod_df['all candidates'][i] = locmod_df['all candidates'][i].union(locmod_df['all candidates'][locmod_df['isotop_ind'][i]])
#            !print(locmod_df['all candidates'][i])
        localization_dict = {}
        for ms, df in mass_shift_data_dict.items():
            if locmod_df['sum of mass shifts'][AA_stat.mass_format(ms)] == False and ms != 0.0:
#                if abs(ms - 125) < 0.3:
#                    print('here')
#                    print([x for x in df[params_dict['peptides_column']] if 'C' in x])
                locations_ms = locmod_df.loc[AA_stat.mass_format(ms), 'all candidates']
                logger.info('For %s mass shift candidates %s', AA_stat.mass_format(ms), str(locations_ms))
                tmp = pd.DataFrame(df.apply(lambda x: AA_stat.localization_of_modification([ms], x, locations_ms,
                                                                                params_dict, spectra_dict), axis=1).to_list(),
                                 index=df.index, columns=['top_isoform', 'loc_counter'])
#                print(tmp['loc_counter'].sum())
                new_localizations = set(tmp['loc_counter'].sum().keys()).difference({'non-localized'})
#                print(new_localizations)
                df[['top_isoform', 'loc_counter' ]] = pd.DataFrame(df.apply(lambda x: AA_stat.localization_of_modification([ms], x, new_localizations,
                                                                                params_dict, spectra_dict), axis=1).to_list(),
                                 index=df.index, columns=['top_isoform', 'loc_counter'])
                localization_dict[AA_stat.mass_format(ms)] = df['loc_counter'].sum()
#                print(df['loc_counter'].sum())
        localization_dict[AA_stat.mass_format(0.000000)] = Counter()
        masses_to_calc = set(locmod_df.index).difference(set(localization_dict.keys()))
        if any(locmod_df['sum of mass shifts'] != False):
            cond = True
        else:
            cond = False
        locmod_df.to_csv(os.path.join(save_directory, 'localization_statistics.csv'), index=False)
        while cond:
            for ms in masses_to_calc:
                masses = locmod_df['sum of mass shifts'][ms]
                if masses != False:
                    if len(masses) == 1:
                        mass_1 = masses[0]
                        mass_2 = masses[0]
                    else:
                        mass_1, mass_2 = masses
                    if AA_stat.mass_format(mass_1) in localization_dict and AA_stat.mass_format(mass_2) in localization_dict:

                        df = mass_shift_data_dict[locmod_df['mass shift'][ms]]
                        locations_ms = locmod_df.loc[ms, 'all candidates']
                        locations_ms1 = set([x for x in localization_dict[AA_stat.mass_format(mass_1)].keys() if len(x) == 1])
                        locations_ms2 = set([x for x in localization_dict[AA_stat.mass_format(mass_2)].keys() if len(x) == 1])
                        tmp = pd.DataFrame(df.apply(lambda x: AA_stat.localization_of_modification([locmod_df['mass shift'][ms],mass_1, mass_2],
                                                                                        x, [locations_ms, locations_ms1,locations_ms2 ], params_dict,
                                                                                        spectra_dict, sum_mod=True), axis=1).to_list(),
                                 index=df.index, columns=['top_isoform', 'loc_counter'])
#                        print(tmp['loc_counter'].sum())
                        new_localizations = set(tmp['loc_counter'].sum().keys()).difference({'non-localized'})
                        locations_ms = []
                        locations_ms1 = []
                        locations_ms2 = []
                        for i in new_localizations:
                            if i.endswith('mod1'):
                                locations_ms1.append(i.split('_')[0])
                            elif i.endswith('mod2'):
                                locations_ms2.append(i.split('_')[0])
                            else:
                                locations_ms.append(i)
#                        print(new_localizations)
#                        print(locations_ms1)
#                        print(locations_ms2)
                        df[['top_isoform', 'loc_counter']] = pd.DataFrame(df.apply(lambda x: AA_stat.localization_of_modification([locmod_df['mass shift'][ms],mass_1, mass_2],
                                                                                        x, [locations_ms, locations_ms1,locations_ms2 ], params_dict,
                                                                                        spectra_dict, sum_mod=True), axis=1).to_list(),
                                 index=df.index, columns=['top_isoform', 'loc_counter'])
                        localization_dict[ms] = df['loc_counter'].sum()
#                        print(df['loc_counter'].sum())
                        masses_to_calc = masses_to_calc.difference(set([ms]))
            if len(masses_to_calc) == 0:
                cond = False
        locmod_df['localization'] = pd.Series(localization_dict)
        print(locmod_df)
        locmod_df.to_csv(os.path.join(save_directory, 'localization_statistics.csv'), index=False)
#        logger.info('Done')
    else:
        logger.info('No spectra files. MSMS spectrum localization is not performed.')