# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:44:50 2019

@author: Julia
"""

from __future__ import print_function, division
import pandas as pd
import numpy as  np
from collections import defaultdict, Counter
import logging

from pyteomics import mass
try:
    from pyteomics import cmass
except ImportError:
    cmass = mass
import string
from . import utils
DIFF_C13 = mass.calculate_mass(formula='C[13]') - mass.calculate_mass(formula='C')
#FRAG_ACC = 0.02
MIN_SPEC_MATCHED = 4
logger = logging.getLogger(__name__)


def get_theor_spectrum(peptide, acc_frag, ion_types=('b', 'y'), maxcharge=1,\
                            aa_mass=mass.std_aa_mass, **kwargs):
    """
    Calculates theoretical spectra in two ways: usual one and in integer format (mz / frag_acc).

    Parameters
    ----------
    peptide : list
        Peptide sequence.
    acc_frag : float
        Fragment mass accuracy in Da.
    ion_types : tuple
        Fragment ion types. ('b', 'y')

    maxcharge: int
        Maximum charge of fragment ion.
    aa_mass: dict
        Amino acid masses

    Returns
    -------
    Returns spectrum in two ways (usual, integer). Usual is a dict with key [ion type, charge] and m/z as a value.
    Integer is a dict, where key is ion type and value is a set of integers (m/z / fragment accuracy).
    """
    if type(peptide) is not list:
        raise Exception('peptide is not a list')
        
    peaks = defaultdict(list)
    theor_set = defaultdict(list)
    for ind, pep in enumerate(peptide[:-1]):
        for ion_type in ion_types:
            nterminal = ion_type[0] in 'abc'
            for charge in range(1, maxcharge+1):
                if ind == 0:
                    if nterminal:
                        mz = cmass.fast_mass2(pep, ion_type=ion_type, charge=charge, aa_mass=aa_mass, **kwargs)
                    else:
                        mz = cmass.fast_mass2(''.join(peptide[1:]), ion_type=ion_type, charge=charge,\
                                             aa_mass=aa_mass, **kwargs)
                else:
                    if nterminal:
                        mz = peaks[ion_type, charge][-1] + aa_mass[pep]/charge
                    else:
                        mz = peaks[ion_type, charge][-1] - aa_mass[pep]/charge
                peaks[ion_type, charge].append(mz)
                theor_set[ion_type].append(int(mz / acc_frag))
#                 g = int(mz / acc_frag)
#                 theor_set[ion_type].extend([g, g+1])
    theor_set = {k:set(v) for k,v in theor_set.items()}
    return peaks, theor_set


def RNHS_fast(spectrum_idict, theoretical_set, min_matched, ion_types=('b', 'y')):
    """
    Matches experimental and theoretical spectra in int formats.

    Parameters
    ----------

    spectrum_idict : list
        Experimental spectrum in integer format.  Output of preprocess_spectrum.
    theoretical_set: dict
        A dict where key is ion type and value is a set of integers (m/z / fragment accuracy).
        Output of get_theor_spec function.
    min_matched : int
        Minumum peaks to be matched.
    ion_types : tuple
        Fragment ion types. ('b', 'y')

    Returns
    -------

    Number of matched peaks, score.

    """
    matched = []
    isum = 0
    for ion_type in ion_types:
        match = 0
        for ion in theoretical_set[ion_type]:
            if ion in spectrum_idict:
                match += 1
                isum += spectrum_idict[ion]
        matched.append(match)
    matched_approx = sum(matched)
    if matched_approx >= min_matched:
        return matched_approx, np.prod([np.math.factorial(m) for m in matched]) * isum
    else:
        return 0, 0


_preprocessing_cache = {}
def preprocess_spectrum(reader, spec_id, kwargs, acc=0.01):
    """
    Prepares experimental spectrum for matching, converts experimental spectrum to int format. Default settings for preprocessing : maximum peaks is 100,
    dynamic range is 1000.

    Paramenters
    -----------
    reader : file reader
        Spectrum file reader
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


def peptide_isoforms(peptide, m, sites):
    """
    Parameters
    ----------
    peptide : list
        Peptide sequence
    m: modification label to apply
    sites : set
        Amino acids eligible for modification

    Returns
    -------
    set of lists

    """
    isoforms = []
    if 'N-term' in sites and len(peptide[0]) == 1 and peptide[0] not in sites:
        isoforms.append((m + peptide[0],) + tuple(peptide[1:]))
    if 'C-term' in sites and len(peptide[-1]) == 1 and peptide[-1] not in sites:
        isoforms.append(tuple(peptide[:-1]) + (m + peptide[-1],))
    for ind, a in enumerate(peptide):
        if a in sites:
            isoforms.append(tuple(peptide[:ind]) + (m + a,) + tuple(peptide[ind+1:]))
    return isoforms


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


def get_full_set_of_candicates(locmod_df):
    """
    Build list of dicts from all_candidates column taking into account the sums of modification.

    Parameters
    ----------
    locmod_df : DataFrame
        DF with candidates for mass shifts.
    Returns
    -------
    Series
    """
    out = defaultdict(list)
    for ind in locmod_df.index:
        out[ind].append({ind: locmod_df.at[ind, 'all candidates']})
        if isinstance(locmod_df.at[ind, 'sum of mass shifts'], list):
            for pair in locmod_df.at[ind, 'sum of mass shifts']:
                tmp_dict = {}
                tmp_dict[pair[0]] = locmod_df.at[pair[0], 'all candidates']
                if len(pair) > 1:
                    tmp_dict[pair[1]] = locmod_df.at[pair[1], 'all candidates']
                out[ind].append(tmp_dict)
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


def localization_of_modification(ms_label, row, loc_candidates, params_dict, spectra_dict, mass_shift_data_dict):

    """
    Localizes modification for mass shift. If two peptides isoforms have the same max score, modification counts as 'non-localized'.

    Paramenters
    -----------
    ms_label : str
        Label for considered mass shift.
    row : dict
        Data Frame row for filtered PSMs data.
    loc_candidates : list
        List of dicts with candidates for localization. locmod_df['loc candidates']
    params_dict : dict
        Dict with all parameters.
    spectra_dict : dict
        Keys are filenames and values are Pyteomics readers.
    sum_mod : bool
        True if sum of codifications should be considered.

    Returns
    -------
    Counter of localizations, top isoform, score difference
    """
    mass_dict = mass.std_aa_mass.copy()
    mass_dict.update(params_dict['fix_mod'])
    peptide = params_dict['peptides_column']
    modif_labels = string.ascii_lowercase
    i = 0
    loc_stat_dict = Counter()
    isoforms = []
    for terms in loc_candidates:
        logger.debug('Generating isoforms for terms %s for shift %s', terms.keys(), ms_label)
        isoform_part = []
        new_isoform_part = []
        for ms in terms:
            mod_aa = {modif_labels[i] + aa: mass_shift_data_dict[ms][0] + mass_dict[aa] for aa in params_dict['labels']}
            mass_dict.update(mod_aa)
#            mass_dict('')
            mass_dict[modif_labels[i]] = mass_shift_data_dict[ms][0]
#            print(mass_dict)

            if not isoform_part: # first modification within this shift (or whole shift)
                logger.debug('Applying mod %s at shift %s...', ms, ms_label)
                isoform_part += peptide_isoforms(list(row[peptide]), modif_labels[i], terms[ms])
                if ms == ms_label:
                    # this is the whole-shift modification
                    isoforms += isoform_part
                elif len(terms) == 1:
                    # two equal mass shifts form this mass shift. Apply the second half
                    logger.debug('Repeating mod %s at shift %s...', ms, ms_label)
                    for p in isoform_part:
                        new_isoform_part += peptide_isoforms(p, modif_labels[i], terms[ms])
            else:
                # second mass shift
                logger.debug('Adding mod %s at shift %s...', ms, ms_label)
                for p in isoform_part:
                    new_isoform_part += peptide_isoforms(p, modif_labels[i], terms[ms])
            i += 1
        isoforms += new_isoform_part
    sequences = [list(x) for x in set(isoforms)]
    if len(sequences) < 1:
        return loc_stat_dict, None, None
    if params_dict['mzml_files']:
        scan = row[params_dict['spectrum_column']].split('.')[1]
        spectrum_id = 'controllerType=0 controllerNumber=1 scan=' + scan
    else:
        spectrum_id = row[params_dict['spectrum_column']]


    exp_dict = preprocess_spectrum(spectra_dict[row['file']], spectrum_id, {}, acc=params_dict['frag_acc'],)

    scores = []
    charge = row[params_dict['charge_column']]

    for seq in sequences:
        theor_spec = get_theor_spectrum(seq, params_dict['frag_acc'], maxcharge=charge, aa_mass=mass_dict, ion_types=params_dict['ion_types'])
        scores.append(RNHS_fast(exp_dict, theor_spec[1], MIN_SPEC_MATCHED, ion_types=params_dict['ion_types'])[1])
    scores = np.array(scores)
    i = np.argsort(scores)[::-1]
    scores = scores[i]
    sequences = np.array(sequences)[i]
    logger.debug('Sorted scores: %s', scores)
    logger.debug('Sorted isoforms: %s', sequences)
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
            return loc_stat_dict, None, None
        else:
            top_isoform = sequences[0]
    else:
        top_isoform = sequences[0]
#    print(top_isoform)
    for ind, a in enumerate(top_isoform):
        if len(a) > 1:
            if ind == 0:
                loc_stat_dict["_".join(['N-term', utils.mass_format(mass_dict[a[0]])])] += 1
            elif ind == len(top_isoform) - 1:
                loc_stat_dict["_".join(['C-term', utils.mass_format(mass_dict[a[0]])])] += 1
            loc_stat_dict["_".join([a[1], utils.mass_format(mass_dict[a[0]])])] += 1

    if not loc_stat_dict:
        return Counter(), None, None
    else:
        if len(scores) > 1:
            scorediff = (scores[0] - scores[1]) / scores[0]
        else:
            scorediff = 0
    logger.debug('Returning: %s %s %s', loc_stat_dict, ''.join(top_isoform), scorediff)
    return loc_stat_dict, ''.join(top_isoform), scorediff


def two_step_localization(df, ms, locations_ms, params_dict, spectra_dict, mass_shift_data_dict):
    """
    Localizes modification or sum of modifications for mass shift and repeat localization if there are redundant candidates.
    If two peptide isoforms have the same max score, modification counts as 'non-localized'.

    Paramenters
    -----------
    df : DataFrame
        DF with filtered peptides for considering mass shift.
    ms : str
        Considered mass shift label
    locations_ms :
       locmod_df['loc candidates']
    params_dict : dict
        Dict with all paramenters.
    spectra_dict : dict
        Keys are filenames and values file with mass spectra.
    sum_mod : bool
        True if sum of codifications should be considered.

    Returns
    -------
    Counter of localizations.
    """
    logger.info('Localizing %s at %s', ms, locations_ms)

    df['localization_count'], df['top isoform'], df['localization score'] = zip(*df.apply(lambda x: localization_of_modification(
                    ms, x, locations_ms, params_dict, spectra_dict, mass_shift_data_dict), axis=1))

    fname = utils.table_path(params_dict['out_dir'], ms)
    peptide = params_dict['peptides_column']
    labels_mod = {}
    mod_aa = string.ascii_lowercase
    i = 0
    for pair in locations_ms:
        for m in pair:
            labels_mod[mod_aa[i]] = m
            i += 1

    df['top isoform'] = df['top isoform'].fillna(df[peptide]).apply(utils.format_isoform, args=(labels_mod,))
    columns = ['top isoform', 'localization score', params_dict['spectrum_column']]
    df[columns].to_csv(fname, index=False, sep='\t')

    return {ms: df['localization_count'].sum()}
