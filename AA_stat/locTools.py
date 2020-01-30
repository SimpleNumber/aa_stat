# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:44:50 2019

@author: Julia
"""

from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as  np
from collections import defaultdict, Counter
import logging
from math import factorial
from pyteomics import mass, electrochem as ec
try:
    from pyteomics import cmass
except ImportError:
    cmass = mass
from . import utils
DIFF_C13 = mass.calculate_mass(formula='C[13]') - mass.calculate_mass(formula='C')
FRAG_ACC = 0.02
MIN_SPEC_MATCHED = 4
logger = logging.getLogger(__name__)


def get_theor_spectrum(peptide, acc_frag, types=('b', 'y'), maxcharge=None, **kwargs):
    """
    Calculates theoretical spectra in two ways: usual one and in integer format (mz / frag_acc).
    
    Parameters
    ----------
    
    peptide : str
        Peptide sequence.
    acc_frag : float
        Fragment mass accuracy in Da.
    types : tuple
        Fragment ion types. ('b', 'y')
        
    maxcharge: int
        Maximum charge of fragment ion.

    Returns
    -------
    Returns spectra in two ways (usual, integer). Usual is a dict with key [ ion type, charge] and m/z as a value.
    Integer is a dict, where key is ion type and value is a set of integers (m/z / fragment accuracy).
    """
#    print(peptide)
    peaks = {}
    theoretical_set = defaultdict(set)
    pl = len(peptide) - 1
    if not maxcharge:
        maxcharge = 1 + int(ec.charge(peptide, pH=2))
    for charge in range(1, maxcharge + 1):
        for ion_type in types:
            nterminal = ion_type[0] in 'abc'
            if nterminal:
                maxpart = peptide[:-1]
                maxmass = cmass.fast_mass(maxpart, ion_type=ion_type, charge=charge, **kwargs)
                marr = np.zeros((pl, ), dtype=float)
                marr[0] = maxmass
                for i in range(1, pl):
                    marr[i] = marr[i-1] - mass.fast_mass2([maxpart[-i]])/charge ### recalculate
            else:
                maxpart = peptide[1:]
                maxmass = cmass.fast_mass(maxpart, ion_type=ion_type, charge=charge, **kwargs)
                marr = np.zeros((pl, ), dtype=float)
                marr[pl-1] = maxmass
                for i in range(pl-2, -1, -1):
                    marr[i] = marr[i+1] - mass.fast_mass2([maxpart[-(i+2)]])/charge ### recalculate

            tmp = marr / acc_frag
            tmp = tmp.astype(int)
            theoretical_set[ion_type].update(tmp)
            marr.sort()
            peaks[ion_type, charge] = marr
    return peaks, theoretical_set


def RNHS_fast(spectrum_idict, theoretical_set, min_matched):
    """
    ---------
    Return score
=======
    spectrum_idict : list
        Experimental spectrum in integer format.  Output of preprocess_spectrum. 
    theoretical_set: dict 
        A dict where key is ion type and value is a set of integers (m/z / fragment accuracy).
        Output of get_theor_spec function.
    min_matched : int
        Minumum peaks to be matched.

    Returns
    -------
    
    Number of matched peaks, score.

    """
    isum = 0
    matched_approx_b, matched_approx_y = 0, 0
    for ion in theoretical_set['b']:
        if ion in spectrum_idict:
            matched_approx_b += 1
            isum += spectrum_idict[ion]

    for ion in theoretical_set['y']:
        if ion in spectrum_idict:
            matched_approx_y += 1
            isum += spectrum_idict[ion]

    matched_approx = matched_approx_b + matched_approx_y
    if matched_approx >= min_matched:
        return matched_approx, factorial(matched_approx_b) * factorial(matched_approx_y) * isum
    else:
        return 0, 0


_preprocessing_cache = {}
def preprocess_spectrum(reader, spec_id, kwargs):
    """
    Prepares experimental spectrum for matching, converts experimental spectrum to int format. Default settings for preprocessing : maximum peaks is 100, 
    dynamic range is 1000.
    
    Paramenters
    -----------
    
    reader : function
        Spectrum file reader.     
    spec_id : str
        Spectrum id.
    
    Returns
    -------
    List of experimental mass spectrum in integer format.
    """
    spectrum = _preprocessing_cache.setdefault((reader, spec_id), {})
    if spectrum:
        # logger.debug('Returning cached spectrum %s', spec_id)
        return spectrum
    # logger.debug('Preprocessing new spectrum %s', spec_id)
    original = reader[spec_id]
    maxpeaks = kwargs.get('maxpeaks', 100)
    dynrange = kwargs.get('dynrange', 1000)
    acc = kwargs.get('acc', FRAG_ACC)

    mz_array = original['m/z array']
    int_array = original['intensity array']
    int_array = int_array.astype(np.float32)

    if dynrange:
        i = int_array > int_array.max() / dynrange
        int_array = int_array[i]
        mz_array = mz_array[i]

    if maxpeaks and int_array.size > maxpeaks:
        i = np.argsort(int_array)[-maxpeaks:]
        j = np.argsort(mz_array[i])
        int_array = int_array[i][j]
        mz_array = mz_array[i][j]

    tmp = (mz_array / acc).astype(int)
    for idx, mt in enumerate(tmp):
        i = int_array[idx]
        spectrum[mt] = max(spectrum.get(mt, 0), i)
        spectrum[mt-1] = max(spectrum.get(mt-1, 0), i)
        spectrum[mt+1] = max(spectrum.get(mt+1, 0), i)
    return spectrum


def peptide_isoforms(sequence, localizations, sum_mod=False):
    """
    Generates isoforms for modification localization.
    
    Paramenters
    -----------
    
    sequence : str
        Sequence of peptide with no modifications.
    localizations: List
        List of lists if `sum_mod` is True. `localizations[0]` one-modification amino acid candidate list.
        `localizations[1]` and `localizations[2]`  is candidates lists for sum of modifications.
    sum_mod : Boolean
        True if sum of modifications have to be considered.
        
    Returns
    -------
    
    Set of peptide isoforms. Where 'm' for mono modification, 'n', 'k' for sum of modifications.
    """
    if sum_mod:
        loc_ = set(localizations[0])
        loc_1 = set(localizations[1])
        loc_2 = set(localizations[2])
        sum_seq_1 = []
        isoforms = []
        for i, j in enumerate(sequence):
            if j in loc_1:
                sum_seq_1.append(sequence[:i] + 'n' + sequence[i:])
        for s in sum_seq_1:
            new_s = '0' + s + '0'
            for i, j in enumerate(new_s[1:-1], 1):
                if j in loc_2 and new_s[i-1] != 'n':
                    isoforms.append(new_s[1:i] + 'k' + new_s[i:-1])
    else:
        loc_ = set(localizations)
        isoforms = []
    if 'N-term' in loc_:
        isoforms.append('m' + sequence)
    if 'C-term' in loc_:
        isoforms.append(sequence[:-1] + 'm' + sequence[-1])

    for i, j in enumerate(sequence):
        if j in loc_:
            isoforms.append(sequence[:i] + 'm' + sequence[i:])

    return set(isoforms)


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
    ind = abs(unimod_df['mono_mass']-mass_shift) < tolerance
    sites_set = set()
    for i, row in unimod_df.loc[ind].iterrows():
        sites_set.update(set(pd.DataFrame(row['specificity']).site))
    return list(sites_set)


def get_candidates_from_aastat(mass_shifts_table, labels, threshold = 1.5):
    """
    Get localization candidates from amono acid statistics.
    
    Paramenters
    -----------
    mass_shifts_table : DataFrame
        DF with amino acid statistics for all mass shifts.
    labels : list
        List of amino acids that should be considered.
    threshold : float
        Threshold to be considered as significantly changed.
    
    Results
    -------
    
    Series with mass shift as index and list of candidates as value.
    """
    
    df = mass_shifts_table.loc[:, labels]
    ms, aa = np.where(df > threshold)
    out = {ms: [] for ms in mass_shifts_table.index}
    for i, j in zip(ms, aa):
        out[df.index[i]].append(df.columns[j])
    return pd.Series(out)


def find_isotopes(ms, tolerance=0.01):
    """
    Find the isotopes between mass shifts using mass difference of C13 and C12, information of amino acids statistics as well.
    
    Paramenters
    -----------
    
    ms : Series
        Series with mass in str format as index and values float mass shift.
    tolerance : float
        Tolerance for isotop matching.
    
    Returns
    -------
    DataFrame with 'isotop'(boolean) and 'monoisotop_index' columns.
    """
    out = pd.DataFrame({'isotope': False, 'monoisotop_index': False}, index=ms.index)
    np_ms = ms.to_numpy()
    difference_matrix = np.abs(np_ms.reshape(-1, 1) - np_ms.reshape(1, -1) - DIFF_C13)
    isotop, monoisotop = np.where(difference_matrix < tolerance)
    out.iloc[isotop, 0] = True
    out.iloc[isotop, 1] = out.iloc[monoisotop, :].index
    return out


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
    return np.nan


def find_modifications(ms, tolerance=0.005):
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
    zero = utils.mass_format(0.0)
    if zero in ms.index:
        col = ms.drop(zero)
    else:
        col = ms
        logger.info('Zero mass shift not found in candidates.')
    values = col.values
    sum_matrix = values.reshape(-1, 1) + values.reshape(1, -1)
    out = col.apply(find_mod_sum, args=(col.index, sum_matrix, tolerance))
    return out


def localization_of_modification(mass_shift, row, loc_candidates, params_dict, spectra_dict, tolerance=FRAG_ACC, sum_mod=False):
    """
    Localizes modification for mass shift. If two peptides isoforms have the same max score, modification counts as 'non-modified'.
    
    Paramenters
    -----------
    mass_shift : float
        Considering mass shift.
    row : dict
        Data Frame row for filtered PSMs data.
    loc_candidates : list
        List or list of lists (in case of sum of modifications, `sum_mod`=True) with candidates for localization.
    params_dict : dict
        Dict with all paramenters.
    spectra_dict : dict
        Keys are filenames and values file with mass spectra.
    tolerance : float
        Tolerance for matching theoretical and experimental spectra.
    sum_mod : boolean
        True if sum of codifications should be considered.
    
    Returns
    -------
    Counter of localizations.    
    """
    mass_dict = mass.std_aa_mass
    peptide = params_dict['peptides_column']
    sequences = peptide_isoforms(row[peptide], loc_candidates, sum_mod=sum_mod)
    if not sequences:
        return Counter()
    if sum_mod:
        mass_dict.update({'m': mass_shift[0], 'n': mass_shift[1], 'k': mass_shift[2]})
        loc_cand, loc_cand_1, loc_cand_2  = loc_candidates
        if mass_shift[1] == mass_shift[2]:
            # logger.debug('Removing duplicate isoforms for %s', mass_shift)
            sequences = {s.replace('k', 'n') for s in sequences}
        labels = [utils.mass_format(ms) for ms in mass_shift]
    else:
        mass_dict.update({'m': mass_shift[0]})
        loc_cand = loc_candidates

    if params_dict['mzml_files']:
        scan = row[params_dict['spectrum_column']].split('.')[1]
        spectrum_id = 'controllerType=0 controllerNumber=1 scan=' + scan
    else:
        spectrum_id = row[params_dict['spectrum_column']]
    exp_dict = preprocess_spectrum(spectra_dict[row['file']], spectrum_id, {})
    loc_stat_dict = Counter()
    scores = [] # write for same scores return non-loc
    charge = row[params_dict['charge_column']]

    sequences = np.array(list(sequences))
    for seq in sequences:
        theor_spec = get_theor_spectrum(seq, tolerance, maxcharge=charge, aa_mass=mass_dict)
        scores.append(RNHS_fast(exp_dict, theor_spec[1], MIN_SPEC_MATCHED)[1])

    scores = np.array(scores)
    i = np.argsort(scores)[::-1]
    scores = scores[i]
    sequences = sequences[i]
    # if logger.level <= logging.DEBUG:
        # fname = os.path.join(params_dict['out_dir'], utils.mass_format(mass_shift[0])+'.txt')
        # logger.debug('Writing isoform scores for %s to %s', row[peptide], fname)
        # with open(fname, 'a') as dump:
        #     for seq, score in zip(sequences, scores):
        #         dump.write('{}\t{}\n'.format(seq, score))
        #     dump.write('\n')
    if len(scores) > 1:
        if scores[0] == scores[1]:
            loc_stat_dict['non-localized'] += 1
            return loc_stat_dict
        else:
            top_isoform = sequences[0]
    else:
        top_isoform = sequences[0]

    loc_index = top_isoform.find('m')
    if top_isoform[loc_index + 1] in loc_cand:
        loc_stat_dict[top_isoform[loc_index + 1]] += 1
    if 'N-term' in loc_cand and loc_index == 0:
        loc_stat_dict['N-term'] += 1
    if 'C-term' in loc_cand and loc_index == len(top_isoform) - 2:
        loc_stat_dict['C-term'] += 1
    loc_index = top_isoform.find('n')
    loc_index_2 = top_isoform.find('k')

    if loc_index > -1:
        if loc_index_2 == -1:
            loc_index_1 = top_isoform.rfind('n')
            # this should happen for duplicates where k was changed to n
            logger.debug('%s: %s, %s', top_isoform, loc_index, loc_index_2)
        if top_isoform[loc_index + 1] in loc_cand_1:
            loc_stat_dict[top_isoform[loc_index + 1] +'_' + labels[1]] += 1
            if loc_index_2 == -1:
                loc_stat_dict[top_isoform[loc_index_1 + 1] +'_' + labels[1]] += 1
            else:
                loc_stat_dict[top_isoform[loc_index_2 + 1] +'_' + labels[2]] += 1
        if 'N-term' in loc_cand_1 and loc_index == 0:
            loc_stat_dict['N-term_' + labels[1]] += 1
        if 'C-term' in loc_cand_1 and loc_index == len(top_isoform) - 2:
            loc_stat_dict['C-term_' + labels[1]] += 1
        if 'N-term' in loc_cand_2 and loc_index_2 == 0:
            loc_stat_dict['N-term_' + labels[2]] += 1
        if 'C-term' in loc_cand_2 and loc_index_2 == len(top_isoform) - 2:
            loc_stat_dict['C-term_' + labels[2]] += 1

    if not loc_stat_dict:
        return Counter()
    else:
        return loc_stat_dict


def two_step_localization(df, ms, locations_ms, params_dict, spectra_dict, sum_mod=False):
    """
    Localizes modification or sum of modifications for mass shift and repeat localization if there are redundant candidates. If two peptides isoforms have the same max score, modification counts as 'non-modified'.
    
    Paramenters
    -----------
    df : DataFrame
        DF with filtered peptides for considering mass shift.
    ms : float
        Considering mass shift.
    locations_ms : 
        List or list of lists (in case of sum of modifications, `sum_mod`=True) with candidates for localization.
    params_dict : dict
        Dict with all paramenters.
    spectra_dict : dict
        Keys are filenames and values file with mass spectra.
    sum_mod : boolean
        True if sum of codifications should be considered.
    
    Returns
    -------
    Counter of localizations.    
    """
    logger.debug('Localizing %s (sum_mod = %s) at %s', ms, sum_mod, locations_ms)
    tmp = df.apply(lambda x: localization_of_modification(
                    ms, x, locations_ms, params_dict, spectra_dict, sum_mod=sum_mod), axis=1)
    new_localizations = set(tmp.sum()).difference({'non-localized'})

    if sum_mod:
        locations_ms0 = set()
        locations_ms1 = set()
        locations_ms2 = set()
        for i in new_localizations:
            if i.endswith('_' + sum_mod[0]):
                locations_ms1.add(i.split('_')[0])
            elif i.endswith('_' + sum_mod[1]):
                locations_ms2.add(i.split('_')[0])
            else:
                locations_ms0.add(i)
        if ms[1] == ms[-1]:
            locations_ms2 = locations_ms1.copy()
        new_localizations = [locations_ms0, locations_ms1, locations_ms2]

    logger.debug('new localizations: %s', new_localizations)
    changed = new_localizations != locations_ms
    logger.debug('Localization did%schange.', [' not ', ' '][changed])
    if changed:
        return df.apply(lambda x: localization_of_modification(
            ms, x, new_localizations, params_dict, spectra_dict, sum_mod=sum_mod), axis=1).sum()
    else:
        return tmp.sum()
