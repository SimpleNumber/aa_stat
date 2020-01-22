from __future__ import print_function, division
import pandas as pd
import numpy as  np
import os

from collections import defaultdict
from scipy.stats import ttest_ind

import logging
import warnings
from pyteomics import parser, pepxml
from . import utils
logger = logging.getLogger(__name__)


AA_STAT_CAND_THRESH = 1.5
ISOTOPE_TOLERANCE = 0.015
UNIIMOD_TOLERANCE = 0.01


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


def save_table(distributions, number_of_PSMs, mass_shifts):
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

    unimod = pd.Series({i: utils.get_unimod_url(float(i)) for i in number_of_PSMs.index})
    df = pd.DataFrame({'mass shift': [mass_shifts[k] for k in distributions.columns],
                       '# peptides in bin': number_of_PSMs},
                      index=distributions.columns)
    df['# peptides in bin'] = df['# peptides in bin'].astype(np.int64)
    out = pd.concat([df, distributions.T], axis=1)
    out['Unimod'] = unimod
    out.reset_index(inplace=True, drop=True)
    return out


def calculate_error_and_p_vals(pep_list, err_ref_df, reference, rule, l):
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
    l: Iterable
        An Iterable of amino acids to be considered.

    Returns
    -------

    Series of p-values, std of amino acid statistics for considered `pep_list`.
    '''
    d = pd.DataFrame(index=l)
    for i in range(50):
        d[i] = pd.Series(get_aa_distribution(
            np.random.choice(np.array(pep_list),
            size=(len(pep_list) // 2), replace=False), rule)) / reference
    p_val = pd.Series()
    for i in l:
        p_val[i] = ttest_ind(err_ref_df.loc[i, :], d.loc[i, :])[1]
    return p_val, d.std(axis=1)


def get_zero_mass_shift(mass_shifts):
    """
    Shift of non-modified peak. Finds zero mass shift.

    Parameters
    ----------
    mass_shifts : Series
        Series of mass shifts.

    Returns
    -------
    Mass shift in float format.
    """
    values = [v[0] for v in mass_shifts.values()]
    l = np.argmin(np.abs(values))
    return values[l]


def check_difference(shift1, shift2):
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

    Returns
    -------
    Boolean.
    True if distance between mass shifts more thn sum of stds.
    """
    mean_diff = (shift1[1] - shift2[1]) ** 2
    sigma_diff = (shift1[2] + shift2[2]) ** 2
    return mean_diff > sigma_diff


def filter_mass_shifts(results):
    """
    Merges close mass shifts. If difference between means of two mass shifts less
    than sum of sigmas, they are merged.

    Parameters
    ----------
    results : numpy array
        Output of utils.fit_peaks function (poptperr matrix). An array of Gauss fitted mass shift
        parameters and their tolerances. [[A, mean, sigma, A_error, mean_error, sigma_error],...]

    Returns
    -------
     Updated poptperr matrix.
    """
    logger.info('Discarding bad peaks...')
    out = []
    for ind, mass_shift in enumerate(results[:-1]):
        cond = check_difference(results[ind], results[ind+1])
        if cond:
            out.append(mass_shift)
        else:
            logger.info('Joined mass shifts %.4f and %.4f', results[ind][1], results[ind+1][1])
    out.append(results[-1])
    logger.info('Peaks for subsequent analysis: %s', len(out))
    return out


def group_specific_filtering(data, mass_shifts, params_dict):
    """
    Selects window around found mass shift and filters using TDA.
    Window is defined as mean +- 3*sigma.

    Parameters
    ----------
    data : DataFrame
        DF with all open search data.
    mass_shifts: numpy array
        Output of utils.fit_peaks function (poptperr matrix). An array of Gauss fitted mass shift
        parameters and their tolerances. [[A, mean, sigma, A_error, mean_error, sigma_error],...]
    params_dict : dict
        Dict with paramenters for parsing csv file.
        `mass_shifts_column`, `FDR`, `FDR_correction`, `peptides_column`

    Returns
    -------
    Dict with mass shifts (in str format) as key and values is a DF with filtered PSMs.
    """
    shifts = params_dict['mass_shifts_column']
    logger.info('Performing group-wise FDR filtering...')
    out_data = {} # dict corresponds list
    for mass_shift in mass_shifts:
        mask = np.abs(data[shifts] - mass_shift[1]) < 3 * mass_shift[2]
        data_slice = data.loc[mask].sort_values(by='expect').drop_duplicates(subset=params_dict['peptides_column'])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pepxml.filter_df(data_slice,
                fdr=params_dict['FDR'], correction=params_dict['FDR_correction'], is_decoy='is_decoy')
        if len(df) > 0:
            shift = np.mean(df[shifts]) ###!!!!!!!mean of from gauss fit!!!!
            out_data[utils.mass_format(shift)] = (shift, df)
    logger.info('# of filtered mass shifts = %s', len(out_data))
    return out_data


def calculate_statistics(mass_shifts_dict, zero_mass_shift, params_dict, args):
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
    zero_mass_shift_label = utils.mass_format(zero_mass_shift)
    zero_bin = mass_shifts_dict[zero_mass_shift_label][1]

    number_of_PSMs = dict()#pd.Series(index=list(mass_shifts_labels.keys()), dtype=int)
    reference = pd.Series(get_aa_distribution(zero_bin[peptides], expasy_rule))
    reference.fillna(0, inplace=True)

    #bootstraping for errors and p values calculation in reference (zero) mass shift
    err_reference_df = pd.DataFrame(index=labels)
    for i in range(50):
        err_reference_df[i] = pd.Series(get_aa_distribution(
            np.random.choice(np.array(zero_bin[peptides]), size=(len(zero_bin) // 2), replace=False),
        expasy_rule)) / reference

    logger.info('Mass shifts:')
    distributions = pd.DataFrame(index=labels)
    p_values = pd.DataFrame(index=labels)

    figure_args = {}

    for ms_label, (ms, ms_df) in mass_shifts_dict.items():
        aa_statistics = pd.Series(get_aa_distribution(ms_df[peptides], expasy_rule))
        peptide_stat = pd.Series(get_peptide_statistics(ms_df[peptides]))
        number_of_PSMs[ms_label] = len(ms_df)
        aa_statistics.fillna(0, inplace=True)
        distributions[ms_label] = aa_statistics / reference
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_vals, errors = calculate_error_and_p_vals(ms_df[peptides], err_reference_df, reference, expasy_rule, labels)
#        errors.fillna(0, inplace=True)

        p_values[ms_label] = p_vals
        distributions.fillna(0, inplace=True)

        labels_df = pd.DataFrame(index=labels)
        labels_df['pep_stat'] = pd.Series(peptide_stat)
        labels_df.fillna(0, inplace=True)
        figure_args[ms_label] = (len(ms_df), [distributions, errors], labels_df['pep_stat'])
        # plot_figure(ms_label, len(ms_df), [distributions, errors], labels_df['pep_stat'], params_dict, save_directory)
        logger.info('%s Da', ms_label)

    pout = p_values.T
    pout.fillna(0).to_csv(os.path.join(save_directory, 'p_values.csv'), index=False)
    return distributions, pd.Series(number_of_PSMs), figure_args


def systematic_mass_shift_correction(mass_shifts_dict, mass_correction):
    """

    Parameters
    ----------
    mass_shifts_dict : Dict
        A dict with mass shifts (in str format) as key and values is a DF with filtered PSMs.
    mass_correction: float
        Mass of reference (zero) mass shift, that should be moved to 0.0

    Returns
    -------
    Updated `mass_shifts_dict`
    """
    out = {}
    for k, v in mass_shifts_dict.items():
        corr_mass = v[0] - mass_correction
        out[utils.mass_format(corr_mass)] = (corr_mass, v[1])
    return out
