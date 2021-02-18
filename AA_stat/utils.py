import os
import operator
import logging

import pandas as pd
import numpy as np
import warnings
from collections import Counter
import re


import itertools as it
from pyteomics import parser, pepxml, mass

logger = logging.getLogger(__name__)

MASS_FORMAT = '{:+.4f}'
UNIMOD = mass.Unimod('file://' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'unimod.xml'))
INTERNAL = 5
DIFF_C13 = mass.calculate_mass(formula='C[13]') - mass.calculate_mass(formula='C')


def internal(*args, **kwargs):
    """Emit log message with level INTERNAL, which is lower than DEBUG."""
    logger.log(INTERNAL, *args, **kwargs)


def mass_format(mass):
    return MASS_FORMAT.format(mass)


def make_0mc_peptides(pep_list, rule):
    """b, y
    In silico cleaves all peptides with a given rule.

    Parameters
    ----------
    pep_list : Iterable
        An iterable of peptides
    rule : str or compiled regex.
        Cleavage rule in pyteomics format.

    Returns
    -------
    Set of fully cleaved peptides.

    """
    out_set = set()
    for i in pep_list:
        out_set.update(parser.cleave(i, rule))
    return out_set


def fdr_filter_mass_shift(mass_shift, data, params_dict):
    shifts = params_dict['mass_shifts_column']
    ms_shift = data.loc[np.abs(data[shifts] - mass_shift[1]) < mass_shift[2], shifts].mean()

    mask = np.abs(data[shifts] - mass_shift[1]) < 3 * mass_shift[2]
    internal('Mass shift %.3f +- 3 * %.3f', mass_shift[1], mass_shift[2])
    data_slice = data.loc[mask].sort_values(by=[params_dict['score_column'], 'spectrum'],
                                ascending=params_dict['score_ascending']).drop_duplicates(subset=params_dict['peptides_column'])
    internal('%d peptide rows selected for filtering', data_slice.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pepxml.filter_df(data_slice, key=params_dict['score_column'],
            fdr=params_dict['FDR'], reverse=not params_dict['score_ascending'], correction=params_dict['FDR_correction'], is_decoy='is_decoy')
    internal('Filtered data for %s: %d rows', mass_shift, df.shape[0])
    return ms_shift, df


def group_specific_filtering(data, mass_shifts, params_dict):
    """
    Selects window around found mass shift and filters using TDA.
    Window is defined as mean +- sigma.

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
    logger.info('Performing group-wise FDR filtering...')
    out_data = {}
    for ind, ms in enumerate(mass_shifts):
        if ind != len(mass_shifts) - 1:
            diff = abs(ms[1] - mass_shifts[ind + 1][1])
            width_sum = 3 * (ms[2] + mass_shifts[ind + 1][2])
            if diff < width_sum:
                coef = width_sum / diff
                ms[2] /= coef
                mass_shifts[ind + 1][2] /= coef
                logger.debug('Mass shifts %.3f and %.3f are too close, dividing their sigma by %.4f', ms[1], mass_shifts[ind + 1][1], coef)
        shift, df = fdr_filter_mass_shift(ms, data, params_dict)

        if len(df) > 0:
            #  shift = np.mean(df[shifts]) ###!!!!!!!mean of from  fit!!!!
            out_data[mass_format(shift)] = (shift, df)
    logger.info('# of filtered mass shifts = %s', len(out_data))
    return out_data



def check_composition(peptide, aa_labels):
    '''
    Checks composition of peptides.
    Parameters
    ----------
    peptide: str
        Peptide sequence
    aa_labels: list
        list of acceptable aa.
    Returns
    -------
    True if accebtable, False overwise.
    '''
    return set(peptide) < set(aa_labels)


def get_varmod_combinations(recommended_vmods, values, tolerance):
    logger.debug('Received recommended vmods: %s', recommended_vmods)
    counter = Counter(aa for aa, shift in recommended_vmods)
    eligible = {aa for aa, count in counter.items() if count >= 3}
    out = {}
    if eligible:
        for i, (aa, shift) in enumerate(recommended_vmods):
            if aa == 'isotope error' or aa not in eligible:
                continue
            candidates = [(aac, shiftc) for aac, shiftc in recommended_vmods if aac == aa and shiftc != shift]
            for c1, c2 in it.combinations(candidates, 2):
                if abs(values[c1[1]] + values[c2[1]] - values[shift]) <= tolerance:
                    out[i] = (c1[1], c2[1])
    return out


def get_opposite_mods(fmods, rec_fmods, rec_vmods, values, tolerance):
    fmods = masses_to_mods(fmods)
    for aa, mod in rec_fmods.items():
        if aa in fmods:
            fmods[aa] = fmods[aa] + values[mod]
        else:
            fmods[aa] = values[mod]
    logger.debug('Calculating opposites using effective fixed mod dict: %s', fmods)
    vmod_idx = []
    for aaf, fmod in fmods.items():
        for i, (aav, vmod) in enumerate(rec_vmods):
            if aaf == aav and abs(fmod + values[vmod]) < tolerance:
                vmod_idx.append(i)
    return vmod_idx


def find_isotopes(ms, peptides_in_bin, tolerance=0.01):
    """
    Find the isotopes between mass shifts using mass difference of C13 and C12, information of amino acids statistics as well.

    Paramenters
    -----------

    ms : Series
        Series with mass in str format as index and values float mass shift.
    peptides_in_bin : Series
        Series with # of peptides in each mass shift.
    tolerance : float
        Tolerance for isotop matching.

    Returns
    -------
    DataFrame with 'isotop'(boolean) and 'monoisotop_index' columns.
    """
    out = pd.DataFrame({'isotope': False, 'monoisotop_index': None}, index=ms.index)
    np_ms = ms.to_numpy()
    difference_matrix = np.abs(np_ms.reshape(-1, 1) - np_ms.reshape(1, -1) - DIFF_C13)
    isotop, monoisotop = np.where(difference_matrix < tolerance)
    logger.debug('Found %d potential isotopes.', isotop.sum())
    out.iloc[isotop, 0] = True
    out.iloc[isotop, 1] = out.iloc[monoisotop, :].index
    for i, row in out.iterrows():
        if row['isotope']:
            if peptides_in_bin[i] > peptides_in_bin[row['monoisotop_index']]:
                out.at[i, 'isotope'], out.at[i, 'monoisotop_index'] = False, None
    return out


def get_candidates_from_unimod(mass_shift, tolerance, unimod_df):
    """
    Find modifications for `mass_shift` in Unimod.org database with a given `tolerance`.


    Paramenters
    -----------
    mass_shift : float
        Modification mass in Da.
    tolerance : float
        Tolerance for the search in Unimod db.
    unimod_df : DataFrame
        DF with all unimod mo9difications.

    Returns
    -------
    List  of amino acids.

    """
    ind = abs(unimod_df['mono_mass'] - mass_shift) < tolerance
    sites_set = set()
    accessions = set()
    for i, row in unimod_df.loc[ind].iterrows():
        sites_set.update(s['site'] for s in row['specificity'])
        accessions.add(row['record_id'])
    return sites_set, accessions


def find_mod_sum(x, index, sum_matrix, tolerance):
    """
    Finds mass shift that are sum of given mass shift and other mass shift results in already existing mass shift.

    Parameters
    ----------
    x : float
        Mass shift that considered as a component of a modification.
    index : dict
        Map for mass shift indexes and their values.
    sum_matrix : numpy 2D array
        Matrix of sums for all mass shifts.
    tolerance: float
        Matching tolerance in Da.

    Returns
    -------
    List of tuples.
    """
    rows, cols = np.where(np.abs(sum_matrix - x) < tolerance)
    i = rows <= cols
    if rows.size:
        return list(zip(index[rows[i]], index[cols[i]]))
    return None


def find_sums(ms, tolerance=0.005):
    """
    Finds the sums of mass shifts in Series, if it exists.

    Parameters
    ----------
    ms : Series
        Series with mass in str format as index and values float mass shift.
    tolerance : float
        Matching tolerance in Da.

    Returns
    -------
    Series with pairs of mass shift for all mass shifts.

    """
    zero = mass_format(0.0)
    if zero in ms.index:
        col = ms.drop(zero)
    else:
        col = ms
        logger.info('Zero mass shift not found in candidates.')
    values = col.values
    sum_matrix = values.reshape(-1, 1) + values.reshape(1, -1)
    out = col.apply(find_mod_sum, args=(col.index, sum_matrix, tolerance))
    return out


def format_isoform(row, params_dict):
    ms = row['mod_dict']
    seq = row['top isoform']
    pc, nc = operator.itemgetter('prev_aa_column', 'next_aa_column')(params_dict)
    prev_aa, next_aa = operator.itemgetter(pc, nc)(row)
    sequence = re.sub(r'([a-z])([A-Z])', lambda m: '{}[{:+.0f}]'.format(m.group(2), float(ms[m.group(1)])), seq)
    return '{}.{}.{}'.format(prev_aa[0], sequence, next_aa[0])


def get_fix_modifications(pepxml_file):
    out = {}
    p = pepxml.PepXML(pepxml_file, use_index=False)
    mod_list = list(p.iterfind('aminoacid_modification'))
    logger.debug('mod_list: %s', mod_list)
    p.reset()
    term_mods = list(p.iterfind('terminal_modification'))
    logger.debug('term_mods: %s', term_mods)
    p.close()
    for m in mod_list:
        if m['variable'] == 'N':
            out[m['aminoacid']] = m['mass']
    for m in term_mods:
        if m['variable'] == 'N':
            if m['terminus'] == 'N':
                out['H-'] = m['mass']
            else:
                out['-OH'] = m['mass']
    return out


def get_specificity(pepxml_file):
    with pepxml.PepXML(pepxml_file, use_index=False) as p:
        s = next(p.iterfind('specificity'))
    logger.debug('Extracted enzyme specificity: %s', s)
    return s


def parse_l10n_site(site):
    aa, shift = site.split('_')
    return aa, shift


def mass_to_mod(label, value, aa_mass=mass.std_aa_mass):
    return value - aa_mass.get(label, 0)


def masses_to_mods(d):
    aa_mass = mass.std_aa_mass.copy()
    aa_mass['H-'] = 1.007825
    aa_mass['-OH'] = 17.00274
    d = {k: mass_to_mod(k, v, aa_mass) for k, v in d.items()}
    if 'H-' in d:
        d['N-term'] = d.pop('H-')
    if '-OH' in d:
        d['C-term'] = d.pop('-OH')
    return d


def format_mod_dict_str(d):
    if d:
        return ', '.join('{} @ {}'.format(v, k) for k, v in d.items())
    return 'none'


def format_mod_dict(d):
    if d:
        return ', '.join('{} @ {}'.format(mass_format(v), k) for k, v in d.items())
    return 'none'


def format_mod_list(items):
    if items:
        return ', '.join('{} @ {}'.format(v, k) for k, v in items)
    return 'none'


def get_isotope_shift(label, locmod_df):
    isotope = locmod_df[locmod_df['isotope index'] == label]
    if not isotope.shape[0]:
        return
    return isotope[isotope['# peptides in bin'] == isotope['# peptides in bin'].max()].index[0]


def format_localization_key(site, ms):
    if not isinstance(ms, str):
        ms = mass_format(ms)
    return site + '_' + ms
