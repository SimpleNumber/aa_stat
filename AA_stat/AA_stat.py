import pandas as pd
import numpy as np
import os

from collections import defaultdict
from scipy.stats import ttest_ind

import logging
import warnings
from pyteomics import parser
from . import utils, locTools

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


def get_fix_mod_from_l10n(mslabel, locmod_df):
    l10n = locmod_df.at[mslabel, 'localization']
    logger.debug('Localizations for %s: %s', mslabel, l10n)
    if l10n:
        l10n.pop('non-localized', None)
        top_loc = max(l10n, key=l10n.get)
        logger.debug('Top localization label for %s: %s', mslabel, top_loc)
        return top_loc


def get_fixed_mod_raw(aa, data_dict, choices=None):
    dist_aa = []
    for ms, v in data_dict.items():
        if choices is None or ms in choices:
            dist_aa.append([v[0], v[1]['peptide'].apply(lambda x: x.count(aa)).sum()])
    utils.internal('Counts for %s: %s', aa, dist_aa)
    top_shift = max(dist_aa, key=lambda tup: tup[1])
    return utils.mass_format(top_shift[0])


def determine_fixed_mods_nonzero(reference, locmod_df, data_dict):
    """Determine fixed modifications in case the reference shift is not at zero.
    Needs localization.
    """
    utils.internal('Localizations for %s: %s', reference, locmod_df.at[reference, 'localization'])
    loc = get_fix_mod_from_l10n(reference, locmod_df)
    label = reference
    data_dict = data_dict.copy()
    while loc is None:
        del data_dict[label]
        label = max(data_dict, key=lambda k: data_dict[k][1].shape[0])
        loc = get_fix_mod_from_l10n(label, locmod_df)
        logger.debug('No luck. Trying %s. Got %s', label, loc)
        if not data_dict:
            break
    return loc


def determine_fixed_mods_zero(aastat_result, data_dict, params_dict):
    """Determine fixed modifications in case the reference shift is at zero.
    Does not need localization.
    """
    fix_mod_zero_thresh = params_dict['fix_mod_zero_thresh']
    min_fix_mod_pep_count_factor = params_dict['min_fix_mod_pep_count_factor']

    fix_mod_dict = {}
    reference = utils.mass_format(0)
    aa_rel = aastat_result[reference][2]
    utils.internal('aa_rel:\n%s', aa_rel)
    candidates = aa_rel[aa_rel < fix_mod_zero_thresh].index
    logger.debug('Fixed mod candidates: %s', candidates)
    for i in candidates:
        candidate_label = get_fixed_mod_raw(i, data_dict)
        if candidate_label != reference:
            # number of peptides with `i` at shift `candidate label` must be higher than ...
            count_cand = data_dict[candidate_label][1][params_dict['peptides_column']].str.contains(i).sum()
            # number of peptides with `i` at shift `reference` by a factor of `min_fix_mod_pep_count_factor`
            count_ref = data_dict[reference][1][params_dict['peptides_column']].str.contains(i).sum()
            # peptide count at candidate shift over # of peptides at reference
            est_ratio = count_cand / len(data_dict[reference][1])
            logger.debug('Peptides with %s: ~%d at %s, ~%d at %s. Estimated pct: %f',
                i, count_ref, reference, count_cand, candidate_label, est_ratio)
            if aastat_result[candidate_label][2][i] > fix_mod_zero_thresh and (
                    est_ratio * 100 > fix_mod_zero_thresh * min_fix_mod_pep_count_factor):
                fix_mod_dict[i] = candidate_label
            else:
                logger.debug('Could not find %s anywhere. Can\'t fix.', i)
        else:
            logger.debug('Reference shift is the best for %s.', i)
    return fix_mod_dict


def determine_fixed_mods(aastat_result, aastat_df, locmod_df, data_dict, params_dict):
    reference = aastat_df.loc[aastat_df['is reference']].index[0]
    if reference == utils.mass_format(0):
        logger.info('Reference bin is at zero shift.')
        fix_mod_dict = determine_fixed_mods_zero(aastat_result, data_dict, params_dict)
    else:
        if locmod_df is None:
            logger.warning('No localization data. '
                'Cannot determine fixed modifications when reference mass shift is non-zero.')
            return {}
        logger.info('Reference bin is at %s. Looking for fixed modification to compensate.', reference)
        loc = determine_fixed_mods_nonzero(reference, locmod_df, data_dict)
        if loc:
            aa, shift = utils.parse_l10n_site(loc)
            fix_mod_dict = {aa: shift}
        else:
            logger.info('No localizations. Stopping.')

    return fix_mod_dict


def recommend_isotope_error(aastat_df, locmod_df, params_dict):
    reference = aastat_df.loc[aastat_df['is reference']].index[0]
    ref_peptides = locmod_df.at[reference, '# peptides in bin']
    logger.debug('%d peptides at reference %s', ref_peptides, reference)
    ref_isotopes = []
    label = reference
    while label:
        label = utils.get_isotope_shift(label, locmod_df)
        ref_isotopes.append(label)
    ref_isotopes.pop()

    i = 0
    for i, label in enumerate(ref_isotopes, 1):
        peps = locmod_df.at[label, '# peptides in bin']
        logger.debug('%d peptides at %s.', peps, label)
        if peps * 100 / ref_peptides < params_dict['recommend isotope threshold']:
            return i - 1
    return i


def recalculate_counts(aa, ms, mods_and_counts, data_dict):
    mods_and_counts[aa].pop(ms, None)
    for i, row in data_dict[ms][1].iterrows():
        seq = row['top isoform'].split('.')[1]
        if row['top_terms'] is not None and ms in row['top_terms']:
            if aa == 'N-term' and seq[1] == '[':
                utils.internal('Reducing count of %s for %s (%s)', seq[0], seq, aa)
                if mods_and_counts[seq[0]].get(ms, 0) > 0:
                    mods_and_counts[seq[0]][ms] -= 1
            elif aa == 'C-term' and seq[-1] == ']':
                res = seq.split('[')[0][-1]
                utils.internal('Reducing count of %s for %s (%s)', res, seq, aa)
                if mods_and_counts[res].get(ms, 0) > 0:
                    mods_and_counts[res][ms] -= 1
            elif seq[:2] == aa + '[':
                utils.internal('Reducing count of N-term for %s', seq)
                if mods_and_counts['N-term'].get(ms, 0) > 0:
                    mods_and_counts['N-term'][ms] -= 1
            elif seq[-1] == ']' and seq.split('[')[0][-1] == aa:
                utils.internal('Reducing count of C-term for %s', seq)
                if mods_and_counts['C-term'].get(ms, 0) > 0:
                    mods_and_counts['C-term'][ms] -= 1


def recalculate_with_isotopes(aa, ms, isotope_rec, mods_and_counts, data_dict, locmod_df):
    logger.debug('Recalculating counts for %s @ %s', aa, ms)
    recalculate_counts(aa, ms, mods_and_counts, data_dict)
    i = 0
    while i < isotope_rec:
        label = utils.get_isotope_shift(ms, locmod_df)
        if label:
            logger.debug('Recalculating %s counts for isotope shift %s', aa, label)
            recalculate_counts(aa, label, mods_and_counts, data_dict)
            i += 1
        else:
            break


def determine_var_mods(aastat_result, aastat_df, locmod_df, data_dict, params_dict, recommended_fix_mods=None):
    if locmod_df is None:
        logger.info('Cannot recommend variable modifications without localization.')
        return {}
    var_mods = []
    recommended = set()
    multiple = params_dict['multiple_mods']
    if multiple:
        logger.info('Recommending multiple modifications on same residue.')
    else:
        logger.info('Recommending one modification per residue.')
    isotope_rec = recommend_isotope_error(aastat_df, locmod_df, params_dict)
    logger.info('Recommended isotope mass error: %d.', isotope_rec)
    if isotope_rec:
        var_mods.append(('isotope error', isotope_rec))
    reference = aastat_df.loc[aastat_df['is reference']].index[0]
    mods_and_counts = defaultdict(dict)  # dict of AA: shift label: count
    for shift in data_dict:
        if shift == reference:
            continue
        l10n = locmod_df.at[shift, 'localization']
        for k, count in l10n.items():
            if k == 'non-localized':
                continue
            aa, locshift = utils.parse_l10n_site(k)
            if locshift == shift:
                mods_and_counts[aa][shift] = count
    logger.debug('Without isotopes, localization counts are:')

    for k, d in mods_and_counts.items():
        logger.debug('%s: %s', k, d)
    if isotope_rec:
        for aa, dcounts in mods_and_counts.items():
            for shift, count in list(dcounts.items()):
                i = 0
                while i < isotope_rec:
                    label = utils.get_isotope_shift(shift, locmod_df)
                    if label:
                        dcounts[shift] = dcounts.get(shift, 0) + mods_and_counts[aa].get(label, 0)
                        # dcounts.pop(label, None)
                        i += 1
                    else:
                        break
        i = 0
        shift = reference
        while i < isotope_rec:
            label = utils.get_isotope_shift(shift, locmod_df)
            if label:
                logger.debug('Removing all counts for isotope shift %s', label)
                for aa, dcounts in mods_and_counts.items():
                    dcounts[label] = 0
                i += 1
            else:
                break
        logger.debug('With isotopes, localization counts are:')
        for k, d in mods_and_counts.items():
            logger.debug('%s: %s', k, d)

    if recommended_fix_mods:
        logger.debug('Subtracting counts for fixed mods.')
        for aa, shift in recommended_fix_mods.items():
            recalculate_with_isotopes(aa, shift, isotope_rec, mods_and_counts, data_dict, locmod_df)

    for i in range(params_dict['variable_mods']):
        logger.debug('Choosing variable modification %d. Counts are:', i + 1)
        for k, d in mods_and_counts.items():
            logger.debug('%s: %s', k, d)
            aa_shifts = {aa: max(dcounts, key=dcounts.get) for aa, dcounts in mods_and_counts.items() if dcounts}
        if mods_and_counts:
            aa_counts = {aa: mods_and_counts[aa][shift] for aa, shift in aa_shifts.items()}
            logger.debug('Best localization counts: %s', aa_shifts)
            logger.debug('Values: %s', aa_counts)
            if aa_shifts:
                top_aa = max(aa_shifts, key=aa_counts.get)
                top_shift = aa_shifts[top_aa]
                top_count = aa_counts[top_aa]
                if top_count < params_dict['min_loc_count']:
                    logger.debug('Localization count too small (%d), stopping.', top_count)
                    break
                recommended.add(top_aa)
                var_mods.append((top_aa, top_shift))
                logger.debug('Chose %s @ %s.', top_shift, top_aa)
                recalculate_with_isotopes(top_aa, top_shift, isotope_rec, mods_and_counts, data_dict, locmod_df)
                if not multiple:
                    logger.debug('Removing all counts for %s.', top_aa)
                    for sh in mods_and_counts[top_aa]:
                        mods_and_counts[top_aa][sh] = 0
    return var_mods


def AA_stat(params_dict, args, step=None):
    """
    Calculates all statistics, saves tables and pictures.
    """
    save_directory = args.dir

    logger.debug('Fixed modifications: %s', params_dict['fix_mod'])
    logger.info('Using fixed modifications: %s.', utils.format_mod_dict(utils.masses_to_mods(params_dict['fix_mod'])))
    data = utils.read_input(args, params_dict)

    hist, popt_pvar = utils.fit_peaks(data, args, params_dict)
    # logger.debug('popt_pvar: %s', popt_pvar)
    final_mass_shifts = filter_mass_shifts(popt_pvar, tolerance=params_dict['shift_error'] * params_dict['bin_width'])
    # logger.debug('final_mass_shifts: %s', final_mass_shifts)
    mass_shift_data_dict = utils.group_specific_filtering(data, final_mass_shifts, params_dict)
    # logger.debug('mass_shift_data_dict: %s', mass_shift_data_dict)
    if not mass_shift_data_dict:
        utils.render_html_report(None, mass_shift_data_dict, params_dict, {}, {}, {}, [], save_directory, [], step=step)
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

    utils.summarizing_hist(table, save_directory)
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

    spectra_dict = utils.read_spectra(args)

    if spectra_dict:
        if args.mgf:
            params_dict['mzml_files'] = False
        else:
            params_dict['mzml_files'] = True
        logger.info('Starting Localization using MS/MS spectra...')
        ms_labels = pd.Series(ms_labels)
        locmod_df = table[['mass shift', '# peptides in bin', 'is isotope', 'isotope index', 'sum of mass shifts',
            'unimod candidates', 'unimod accessions']].copy()

        locmod_df['aa_stat candidates'] = locTools.get_candidates_from_aastat(
            table, labels=params_dict['labels'], threshold=params_dict['candidate threshold'])

        locmod_df['all candidates'] = locmod_df.apply(
            lambda x: set(x['unimod candidates']) | (set(x['aa_stat candidates'])), axis=1)
        for i in locmod_df.loc[locmod_df['is isotope']].index:
            locmod_df.at[i, 'all candidates'] = locmod_df.at[i, 'all candidates'].union(
                locmod_df.at[locmod_df.at[i, 'isotope index'], 'all candidates'])
        locmod_df['candidates for loc'] = locTools.get_full_set_of_candicates(locmod_df)
        logger.info('Reference mass shift %s', reference_label)
        localization_dict = {}

        for ms_label, (ms, df) in mass_shift_data_dict.items():
            localization_dict.update(locTools.localization(
                df, ms, ms_label, locmod_df.at[ms_label, 'candidates for loc'],
                params_dict, spectra_dict, {k: v[0] for k, v in mass_shift_data_dict.items()}))

        locmod_df['localization'] = pd.Series(localization_dict).apply(dict)
        locmod_df.to_csv(os.path.join(save_directory, 'localization_statistics.csv'), index=False)

        if not locmod_df.at[reference_label, 'all candidates']:
            logger.debug('Explicitly writing out peptide table for reference mass shift.')
            df = mass_shift_data_dict[reference_label][1]
            utils.save_df(reference_label, df, save_directory, params_dict)
        for reader in spectra_dict.values():
            reader.close()
    else:
        locmod_df = None
        utils.save_peptides(mass_shift_data_dict, save_directory, params_dict)
        logger.info('No spectrum files. MS/MS localization is not performed.')
    logger.info('Plotting mass shift figures...')
    for ms_label, data in figure_data.items():
        if locmod_df is not None:
            localizations = locmod_df.at[ms_label, 'localization']
            sumof = locmod_df.at[ms_label, 'sum of mass shifts']
        else:
            localizations = None
            sumof = None
        utils.plot_figure(ms_label, *data, params_dict, save_directory, localizations, sumof)

    logger.info('AA_stat results saved to %s', os.path.abspath(args.dir))
    utils.internal('Data dict: \n%s', mass_shift_data_dict)
    recommended_fix_mods = determine_fixed_mods(figure_data, table, locmod_df, mass_shift_data_dict, params_dict)
    logger.debug('Recommended fixed mods: %s', recommended_fix_mods)
    if recommended_fix_mods:
        logger.info('Recommended fixed modifications: %s.', utils.format_mod_dict_str(recommended_fix_mods))
    else:
        logger.info('Fixed modifications not recommended.')
    recommended_var_mods = determine_var_mods(
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
    utils.render_html_report(table, mass_shift_data_dict, params_dict, recommended_fix_mods, recommended_var_mods, combinations, opposite,
        save_directory, ms_labels, step=step)
    return figure_data, table, locmod_df, mass_shift_data_dict, recommended_fix_mods, recommended_var_mods
