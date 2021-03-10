import pandas as pd
import numpy as np
import os

from collections import defaultdict
from scipy.stats import ttest_ind

import logging
import warnings
from pyteomics import parser
from . import utils, localization, html, io, stats, recommendations

logger = logging.getLogger(__name__)


def get_peptide_statistics(peptide_list):
    '''
    Calculates presence of amino acid in peptide sequences.

    Parameters
    ----------
    peptide_list : Iterable
        An iterable of peptides, that are already fully cleaved.

    Returns
    -------
    out : dict with amino acids as a key and its persentage of peptides with it as a value.
    '''
    sum_aa = 0
    pep_set = set(peptide_list)
    d = defaultdict(int)
    for seq in pep_set:
        for let in set(seq):
            d[let] += 1
        sum_aa += 1
    for i in d:
        d[i] = int(100 * d[i] / sum_aa)
    return d


def get_aa_distribution(peptide_list, rule):
    '''
    Calculates amino acid statistics for peptide list.
    In silico cleaves peptides to get fully cleaved set of peptides.

    Parameters
    ----------
    peptide_list : Iterable
        An iterable of peptides.
    rule : str or compiled regex.
        Cleavage rule in pyteomics format.

    Returns
    -------
    out : dict with amino acids as a key and its persentage as a value.
    '''
    sum_aa = 0
    pep_set = utils.make_0mc_peptides(peptide_list, rule)
    d = defaultdict(int)
    for seq in pep_set:
        for let in seq:
            d[let] += 1
            sum_aa += 1
    for i in d:
        d[i] /= sum_aa
    return d


def make_table(distributions, number_of_PSMs, mass_shifts, reference_label):
    '''
    Prepares amino acid statistis result table.

    Parameters
    ----------
    distributions : DataFrame
        Amino acids statistics, where indexes are amino acids, columns mass shifts (str)
    number_of_PSMs : Series
        Indexes are mass shifts (in str format) and values are numbers of filtered PSMs
    mass_shifts : dict
        Mass shift in str format (rounded) -> actual mass shift (float)

    Returns
    -------

    A table with mass shifts, psms, amino acid statistics columns.
    '''
    df = pd.DataFrame({'mass shift': [mass_shifts[k] for k in distributions.columns],
                       '# peptides in bin': number_of_PSMs},
                      index=distributions.columns)
    df['# peptides in bin'] = df['# peptides in bin'].astype(np.int64)
    out = pd.concat([df, distributions.T], axis=1)
    out['is reference'] = df.index == reference_label
    return out


def calculate_error_and_p_vals(pep_list, err_ref_df, reference, rule, aas):
    '''
    Calculates p-values and error standard deviation of amino acids statistics
    using bootstraping method.

    Parameters
    ----------
    pep_list : Iterable
        An iterable of peptides.
    err_ref_df : Series
        Indexes are amino acids and values are stds of a `reference` mass shift.
    reference : Series
        Indexes are amino acids and values are amino acids statistics of a reference mass shift.
    rule : str or compiled regex.
        Cleavage rule in pyteomics format.
    aas: Iterable
        An Iterable of amino acids to be considered.

    Returns
    -------

    Series of p-values, std of amino acid statistics for considered `pep_list`.
    '''
    d = pd.DataFrame(index=aas)
    for i in range(50):
        d[i] = pd.Series(get_aa_distribution(
            np.random.choice(np.array(pep_list),
            size=(len(pep_list) // 2), replace=False), rule)) / reference
    p_val = pd.Series()
    for i in aas:
        p_val[i] = ttest_ind(err_ref_df.loc[i, :], d.loc[i, :])[1]
    return p_val, d.std(axis=1)


def get_zero_mass_shift(mass_shift_data_dict, params_dict):
    """
    Shift of non-modified peak. Finds zero mass shift.

    Parameters
    ----------
    mass_shift_data_dict : dict
        dict of mass shifts.
    params_dict: dict

    Returns
    -------
    Mass shift label, Mass shift in float format.
    """
    values = [v[0] for v in mass_shift_data_dict.values()]
    keys = list(mass_shift_data_dict.keys())
    data = [v[1] for v in mass_shift_data_dict.values()]
    lref = np.argmin(np.abs(values))
    maxbin = max(df.shape[0] for df in data)
    logger.debug('Closest to zero: %s, with %d peptides. Top mass shift has %d peptides.',
                 keys[lref], data[lref].shape[0], maxbin)
    if abs(values[lref]) > params_dict['zero bin tolerance'] or data[lref].shape[0] / maxbin < params_dict['zero min intensity']:
        logger.warning('Too few unmodified peptides. Mass shift with most identifications will be the reference.')
        identifications = [df.shape[0] for df in data]
        lref = np.argmax(identifications)
    return keys[lref], values[lref]


def check_difference(shift1, shift2, tolerance=0.05):
    """
    Checks two mass shifts means to be closer than the sum of their std.

    Parameters
    ----------
    shift1 : List
        list that describes mass shift. On the first position have to be mean of mass shift,
        on  second position have to be std.
    shift2 : List
        list that describes mass shift. On the first position have to be mean of mass shift,
        on  second position have to be std.
    tolerance : float
        Matching tolerance in Da.

    Returns
    -------
    out : bool
    """
    mean_diff = (shift1[1] - shift2[1]) ** 2
    sigma_diff = (shift1[2] + shift2[2]) ** 2
    res = mean_diff > sigma_diff
    if abs(shift1[1] - shift2[1]) < tolerance:
        res = False
    return res


def filter_mass_shifts(results, tolerance=0.05):
    """
    Merges close mass shifts. If difference between means of two mass shifts less
    than sum of sigmas, they are merged.

    Parameters
    ----------
    results : numpy array
        Output of utils.fit_peaks function (poptperr matrix). An array of Gauss fitted mass shift
        parameters and their tolerances. [[A, mean, sigma, A_error, mean_error, sigma_error],...]
    tolerance : float
        Matching tolerance in Da.
    Returns
    -------
    Updated poptperr matrix.
    """
    logger.info('Discarding bad peaks...')
    temp = []
    out = []
    if not results.size:
        return []
    if results.size == 1:
        return [results[0]]

    temp = [results[0]]
    for mass_shift in results[1:]:
        if check_difference(temp[-1], mass_shift, tolerance=tolerance):
            if len(temp) > 1:
                logger.info('Joined mass shifts %s', ['{:0.4f}'.format(x[1]) for x in temp])
            out.append(max(temp, key=lambda x: x[0]))
            temp = [mass_shift]
        else:
            temp.append(mass_shift)
    out.append(max(temp, key=lambda x: x[0]))

    logger.info('Peaks for subsequent analysis: %s', len(out))
    return out


def calculate_statistics(mass_shifts_dict, reference_label, params_dict, args):
    """
    Calculates amino acid statistics, relative amino acids presence in peptides
    for all mass shifts.

    Paramenters
    -----------
    mass_shifts_dict : dict
        A dict with mass shifts (in str format) as key and values is a DF with filtered PSMs.
    zero_mass_shift : float
        Reference mass shift.
    params_dict : dict
        Dict with paramenters for parsing csv file.
        `labels`, `rule`, `peptides_column` and other params

    Returns
    -------

    DF with amino acid statistics, Series with number of PSMs and dict of data
    for mass shift figures.

    """
    logger.info('Calculating distributions...')
    labels = params_dict['labels']
    rule = params_dict['rule']
    expasy_rule = parser.expasy_rules.get(rule, rule)
    save_directory = args.dir
    peptides = params_dict['peptides_column']
    reference_bin = mass_shifts_dict[reference_label][1]

    number_of_PSMs = dict()  # pd.Series(index=list(mass_shifts_labels.keys()), dtype=int)
    reference = pd.Series(get_aa_distribution(reference_bin[peptides], expasy_rule))
    reference.fillna(0, inplace=True)

    # bootstraping for errors and p values calculation in reference (zero) mass shift
    err_reference_df = pd.DataFrame(index=labels)
    for i in range(50):
        err_reference_df[i] = pd.Series(get_aa_distribution(
            np.random.choice(np.array(reference_bin[peptides]), size=(len(reference_bin) // 2), replace=False),
            expasy_rule)) / reference

    logger.info('Mass shifts:')
    distributions = pd.DataFrame(index=labels)
    p_values = pd.DataFrame(index=labels)

    figure_args = {}

    for ms_label, (ms, ms_df) in mass_shifts_dict.items():
        aa_statistics = pd.Series(get_aa_distribution(ms_df[peptides], expasy_rule))
        peptide_stat = pd.Series(get_peptide_statistics(ms_df[peptides]), index=labels)
        number_of_PSMs[ms_label] = len(ms_df)
        aa_statistics.fillna(0, inplace=True)
        distributions[ms_label] = aa_statistics / reference
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_vals, errors = calculate_error_and_p_vals(ms_df[peptides], err_reference_df, reference, expasy_rule, labels)
        # errors.fillna(0, inplace=True)

        p_values[ms_label] = p_vals
        distributions.fillna(0, inplace=True)

        figure_args[ms_label] = (len(ms_df), [distributions[ms_label], errors], peptide_stat.fillna(0))
        logger.info('%s Da', ms_label)

    pout = p_values.T
    pout.fillna(0).to_csv(os.path.join(save_directory, 'p_values.csv'), index=False)
    return distributions, pd.Series(number_of_PSMs), figure_args


def systematic_mass_shift_correction(mass_shifts_dict, mass_correction):
    """

    Parameters
    ----------
    mass_shifts_dict : dict
        A dict with in the format: `mass_shift_label`: `(mass_shift_value, filtered_peptide_dataframe)`.
    mass_correction: float
        Mass of reference (zero) mass shift, that should be moved to 0.0

    Returns
    -------
    out : dict
        Updated `mass_shifts_dict`
    """
    out = {}
    for k, v in mass_shifts_dict.items():
        corr_mass = v[0] - mass_correction
        out[utils.mass_format(corr_mass)] = (corr_mass, v[1])
    return out


def AA_stat(params_dict, args, step=None):
    """
    Calculates all statistics, saves tables and pictures.
    """
    save_directory = args.dir

    logger.debug('Fixed modifications: %s', params_dict['fix_mod'])
    logger.debug('Variable modifications: %s', params_dict['var_mod'])
    logger.info('Using fixed modifications: %s.', utils.format_mod_dict(utils.masses_to_mods(params_dict['fix_mod'])))
    logger.info('Variable modifications in search results: %s.', utils.format_mod_dict(params_dict['var_mod']))
    data = io.read_input(args, params_dict)

    hist, popt_pvar = stats.fit_peaks(data, args, params_dict)
    # logger.debug('popt_pvar: %s', popt_pvar)
    final_mass_shifts = filter_mass_shifts(popt_pvar, tolerance=params_dict['shift_error'] * params_dict['bin_width'])
    # logger.debug('final_mass_shifts: %s', final_mass_shifts)
    mass_shift_data_dict = utils.group_specific_filtering(data, final_mass_shifts, params_dict)
    # logger.debug('mass_shift_data_dict: %s', mass_shift_data_dict)
    if not mass_shift_data_dict:
        html.render_html_report(None, mass_shift_data_dict, None, params_dict, {}, {}, {}, [], save_directory, [], step=step)
        return None, None, None, mass_shift_data_dict, {}

    reference_label, reference_mass_shift = get_zero_mass_shift(mass_shift_data_dict, params_dict)
    if abs(reference_mass_shift) < params_dict['zero bin tolerance']:
        logger.info('Systematic mass shift equals to %s', reference_label)
        if params_dict['calibration'] != 'off':
            mass_shift_data_dict = systematic_mass_shift_correction(mass_shift_data_dict, reference_mass_shift)
            reference_mass_shift = 0.0
            reference_label = utils.mass_format(0.0)
        else:
            logger.info('Leaving systematic shift in place (calibration disabled).')
    else:
        logger.info('Reference mass shift is %s', reference_label)
    ms_labels = {k: v[0] for k, v in mass_shift_data_dict.items()}
    logger.debug('Final shift labels: %s', ms_labels.keys())

    distributions, number_of_PSMs, figure_data = calculate_statistics(mass_shift_data_dict, reference_label, params_dict, args)

    table = make_table(distributions, number_of_PSMs, ms_labels, reference_label)

    stats.summarizing_hist(table, save_directory)
    logger.info('Summary histogram saved.')
    # table.index = table['mass shift'].apply(utils.mass_format)
    table[['is isotope', 'isotope index']] = utils.find_isotopes(
        table['mass shift'], table['# peptides in bin'], tolerance=params_dict['prec_acc'])
    table.at[reference_label, 'is isotope'] = False
    table.at[reference_label, 'isotope index'] = None
    logger.debug('Isotopes:\n%s', table.loc[table['is isotope']])
    u = utils.UNIMOD.mods
    unimod_df = pd.DataFrame(u)
    table['unimod candidates'], table['unimod accessions'] = zip(*table['mass shift'].apply(
        lambda x: utils.get_candidates_from_unimod(x, params_dict['prec_acc'], unimod_df)))

    table['sum of mass shifts'] = utils.find_sums(table.loc[~table['is isotope'], 'mass shift'],
            tolerance=params_dict['shift_error'] * params_dict['bin_width'])
    logger.debug('Sums of mass shifts:\n%s', table.loc[table['sum of mass shifts'].notna()])
    table.to_csv(os.path.join(save_directory, 'aa_statistics_table.csv'), index=False)

    spectra_dict = io.read_spectra(args)

    if spectra_dict:
        if args.mgf:
            params_dict['mzml_files'] = False
        else:
            params_dict['mzml_files'] = True
        logger.info('Starting Localization using MS/MS spectra...')
        ms_labels = pd.Series(ms_labels)
        locmod_df = table[['mass shift', '# peptides in bin', 'is isotope', 'isotope index', 'sum of mass shifts',
            'unimod candidates', 'unimod accessions']].copy()

        locmod_df['aa_stat candidates'] = localization.get_candidates_from_aastat(
            table, labels=params_dict['labels'], threshold=params_dict['candidate threshold'])

        if params_dict['use_all_loc']:
            logger.info('Localizaing all mass shifts on all amino acids. This may take some time.')
            locmod_df['all candidates'] = [set(parser.std_amino_acids)] * locmod_df.shape[0]
        else:
            locmod_df['all candidates'] = locmod_df.apply(
                lambda x: set(x['unimod candidates']) | set(x['aa_stat candidates']), axis=1)
            if params_dict['force_term_loc']:
                logger.debug('Adding terminal localizations for all mass shifts.')
                locmod_df['all candidates'] = locmod_df['all candidates'].apply(lambda x: x | {'N-term', 'C-term'})
        for i in locmod_df.loc[locmod_df['is isotope']].index:
            locmod_df.at[i, 'all candidates'] = locmod_df.at[i, 'all candidates'].union(
                locmod_df.at[locmod_df.at[i, 'isotope index'], 'all candidates'])
        for i in locmod_df.index:
            ac = locmod_df.at[i, 'all candidates']
            for term in ('N', 'C'):
                if 'Protein {}-term'.format(term) in ac and '{}-term'.format(term) in ac:
                    ac.remove('Protein {}-term'.format(term))
                    logger.debug('Removing protein %s-term localization for %s as redundant.', term, i)
        if reference_mass_shift == 0.0:
            locmod_df.at[reference_label, 'all candidates'] = set()
        locmod_df['candidates for loc'] = localization.get_full_set_of_candidates(locmod_df)

        logger.info('Reference mass shift %s', reference_label)
        localization_dict = {}

        for ms_label, (ms, df) in mass_shift_data_dict.items():
            localization_dict.update(localization.localization(
                df, ms, ms_label, locmod_df.at[ms_label, 'candidates for loc'],
                params_dict, spectra_dict, {k: v[0] for k, v in mass_shift_data_dict.items()}))

        locmod_df['localization'] = pd.Series(localization_dict).apply(dict)
        locmod_df.to_csv(os.path.join(save_directory, 'localization_statistics.csv'), index=False)

        if not locmod_df.at[reference_label, 'all candidates']:
            logger.debug('Explicitly writing out peptide table for reference mass shift.')
            df = mass_shift_data_dict[reference_label][1]
            io.save_df(reference_label, df, save_directory, params_dict)
        for reader in spectra_dict.values():
            reader.close()
    else:
        locmod_df = None
        io.save_peptides(mass_shift_data_dict, save_directory, params_dict)
        logger.info('No spectrum files. MS/MS localization is not performed.')
    logger.info('Plotting mass shift figures...')
    for ms_label, data in figure_data.items():
        if locmod_df is not None:
            localizations = locmod_df.at[ms_label, 'localization']
            sumof = locmod_df.at[ms_label, 'sum of mass shifts']
        else:
            localizations = None
            sumof = None
        stats.plot_figure(ms_label, *data, params_dict, save_directory, localizations, sumof)

    logger.info('AA_stat results saved to %s', os.path.abspath(args.dir))
    utils.internal('Data dict: \n%s', mass_shift_data_dict)
    recommended_fix_mods = recommendations.determine_fixed_mods(figure_data, table, locmod_df, mass_shift_data_dict, params_dict)
    logger.debug('Recommended fixed mods: %s', recommended_fix_mods)
    if recommended_fix_mods:
        logger.info('Recommended fixed modifications: %s.', utils.format_mod_dict_str(recommended_fix_mods))
    else:
        logger.info('Fixed modifications not recommended.')
    recommended_var_mods = recommendations.determine_var_mods(
        figure_data, table, locmod_df, mass_shift_data_dict, params_dict, recommended_fix_mods)
    logger.debug('Recommended variable mods: %s', recommended_var_mods)
    if recommended_var_mods:
        logger.info('Recommended variable modifications: %s.', utils.format_mod_list(recommended_var_mods))
    else:
        logger.info('Variable modifications not recommended.')
    combinations = utils.get_varmod_combinations(recommended_var_mods, ms_labels, params_dict['prec_acc'])
    logger.debug('Found combinations in recommended variable mods: %s', combinations)
    opposite = utils.get_opposite_mods(
        params_dict['fix_mod'], recommended_fix_mods, recommended_var_mods, ms_labels, params_dict['prec_acc'])
    logger.debug('Opposite modifications: %s', utils.format_mod_list([recommended_var_mods[i] for i in opposite]))
    html.render_html_report(table, mass_shift_data_dict, locmod_df, params_dict,
        recommended_fix_mods, recommended_var_mods, combinations, opposite, save_directory, ms_labels, step=step)
    return figure_data, table, locmod_df, mass_shift_data_dict, recommended_fix_mods, recommended_var_mods
