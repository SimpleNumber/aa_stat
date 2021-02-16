"""
Created on Thu Oct 24 11:44:50 2019

@author: Julia
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import logging

from pyteomics import mass
try:
    from pyteomics import cmass
except ImportError:
    cmass = mass
import string
from . import utils, io

logger = logging.getLogger(__name__)


def get_theor_spectrum(peptide, acc_frag, ion_types=('b', 'y'), maxcharge=1,
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
    if not isinstance(peptide, list):
        raise Exception('peptide is not a list: {!r}'.format(peptide))

    peaks = defaultdict(list)
    theor_set = defaultdict(list)
    aa_mass = aa_mass.copy()
    H = mass.nist_mass['H'][0][0]
    nterm_mod = aa_mass.pop('H-', H)
    OH = H + mass.nist_mass['O'][0][0]
    cterm_mod = aa_mass.pop('-OH', OH)
    for ind, pep in enumerate(peptide[:-1]):
        for ion_type in ion_types:
            nterminal = ion_type[0] in 'abc'
            for charge in range(1, maxcharge + 1):
                if ind == 0:
                    if nterminal:
                        mz = cmass.fast_mass2(
                            pep, ion_type=ion_type, charge=charge, aa_mass=aa_mass, **kwargs) + (nterm_mod - H) / charge
                    else:
                        mz = cmass.fast_mass2(''.join(peptide[1:]), ion_type=ion_type, charge=charge,
                                             aa_mass=aa_mass, **kwargs) + (cterm_mod - OH) / charge
                else:
                    if nterminal:
                        mz = peaks[ion_type, charge][-1] + aa_mass[pep] / charge
                    else:
                        mz = peaks[ion_type, charge][-1] - aa_mass[pep] / charge
                peaks[ion_type, charge].append(mz)
                theor_set[ion_type].append(int(mz / acc_frag))
    theor_set = {k: set(v) for k, v in theor_set.items()}
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
    Prepares experimental spectrum for matching, converts experimental spectrum to int format.
    Default settings for preprocessing : maximum peaks is 100, dynamic range is 1000.

    Parameters
    ----------
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
        spectrum[mt - 1] = max(spectrum.get(mt - 1, 0), i)
        spectrum[mt + 1] = max(spectrum.get(mt + 1, 0), i)
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
            isoforms.append(tuple(peptide[:ind]) + (m + a,) + tuple(peptide[ind + 1:]))
    return isoforms


def get_candidates_from_aastat(mass_shifts_table, labels, threshold=1.5):
    """
    Get localization candidates from amono acid statistics.

    Parameters
    ----------
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


def get_full_set_of_candidates(locmod_df):
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


def localization_of_modification(ms, ms_label, row, loc_candidates, params_dict, spectra_dict, mass_shift_dict):
    """
    Localizes modification for mass shift. If two peptides isoforms have the same max score, modification counts as 'non-localized'.

    Parameters
    ----------
    ms: float
        mass shift
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
    mass_dict_0 = mass.std_aa_mass.copy()
    mass_dict_0.update(params_dict['fix_mod'])
    peptide = params_dict['peptides_column']
    modif_labels = string.ascii_lowercase

    loc_stat_dict = Counter()

    charge = row[params_dict['charge_column']]

    if params_dict['mzml_files']:
        scan = row[params_dict['spectrum_column']].split('.')[1].lstrip('0')
        spectrum_id = 'controllerType=0 controllerNumber=1 scan=' + scan
    else:
        spectrum_id = row[params_dict['spectrum_column']]
    exp_dict = preprocess_spectrum(spectra_dict[row['file']], spectrum_id, {}, acc=params_dict['frag_acc'],)

    top_score, second_score = 0, 0
    top_isoform = None
    top_terms = None
    for terms in loc_candidates:
        scores = []
        mass_dict = mass_dict_0.copy()
        isoform_part = []
        new_isoform_part = []
        i = 0
        isoforms = []
        sequences = []
        for _ms in terms:
            mod_aa = {modif_labels[i] + aa: mass_shift_dict[_ms] + mass_dict[aa] for aa in params_dict['labels']}
            mass_dict.update(mod_aa)
            mass_dict[modif_labels[i]] = mass_shift_dict[_ms]

            if not isoform_part:  # first modification within this shift (or whole shift)
                isoform_part += peptide_isoforms(list(row[peptide]), modif_labels[i], terms[_ms])
                if _ms == ms_label:
                    # this is the whole-shift modification
                    isoforms += isoform_part
                elif len(terms) == 1:
                    # two equal mass shifts form this mass shift. Apply the second half
                    for p in isoform_part:
                        new_isoform_part += peptide_isoforms(p, modif_labels[i], terms[_ms])
            else:
                # second mass shift
                for p in isoform_part:
                    new_isoform_part += peptide_isoforms(p, modif_labels[i], terms[_ms])
            i += 1
        isoforms += new_isoform_part
        sequences = [list(x) for x in isoforms]
        # utils.internal('Generated %d isoforms for terms %s at shift %s', len(sequences), terms.keys(), ms_label)
        for seq in sequences:
            # utils.internal('seq = %s', seq)
            theor_spec = get_theor_spectrum(seq,
                params_dict['frag_acc'], maxcharge=charge, aa_mass=mass_dict, ion_types=params_dict['ion_types'])
            scores.append(RNHS_fast(exp_dict, theor_spec[1], params_dict['min_spec_matched'], ion_types=params_dict['ion_types'])[1])
        scores = np.array(scores)
        i = np.argsort(scores)[::-1]
        scores = scores[i]
        sequences = np.array(sequences)[i]
        if scores.size:
            if scores[0] > top_score:
                second_score = top_score
                top_score = scores[0]
                top_isoform = sequences[0]
                top_terms = terms
            if scores.size > 1 and scores[1] > second_score:
                second_score = scores[1]

    if top_isoform is None:
        return loc_stat_dict, None, None, None

    if top_score == second_score:
        loc_stat_dict['non-localized'] += 1
        return loc_stat_dict, None, None, None

    mass_dict = mass_dict_0.copy()
    # utils.internal('Top isoform is %s for terms %s (shift %s)', top_isoform, top_terms, ms_label)
    i = 0
    for _ms in top_terms:
        mod_aa = {modif_labels[i] + aa: mass_shift_dict[_ms] + mass_dict[aa] for aa in params_dict['labels']}
        mass_dict.update(mod_aa)
        mass_dict[modif_labels[i]] = mass_shift_dict[_ms]
        i += 1
    for ind, a in enumerate(top_isoform):
        if len(a) > 1:
            if ind == 0:
                loc_stat_dict['N-term_' + utils.mass_format(mass_dict[a[0]])] += 1
            elif ind == len(top_isoform) - 1:
                loc_stat_dict['C-term_' + utils.mass_format(mass_dict[a[0]])] += 1
            loc_stat_dict["_".join([a[1], utils.mass_format(mass_dict[a[0]])])] += 1

    scorediff = (top_score - second_score) / top_score
    # utils.internal('Returning: %s %s %s', loc_stat_dict, ''.join(top_isoform), scorediff)
    return loc_stat_dict, ''.join(top_isoform), top_terms, scorediff


def localization(df, ms, ms_label, locations_ms, params_dict, spectra_dict, mass_shift_dict):
    """
    Localizes modification or sum of modifications for mass shift and repeat localization if there are redundant candidates.
    If two peptide isoforms have the same max score, modification counts as 'non-localized'.

    Parameters
    ----------
    df : DataFrame
        DF with filtered peptides for considered mass shift.
    ms: float
        mass shift
    ms_label : str
        Considered mass shift label
    locations_ms :
       locmod_df['loc candidates']
    params_dict : dict
        Dict with all paramenters.
    spectra_dict : dict
        Keys are filenames and values file with mass spectra.

    Returns
    -------
    Counter of localizations.
    """
    logger.info('Localizing %s...', ms_label)
    logger.debug('Localizations: %s', locations_ms)
    if len(locations_ms) < 2 and list(locations_ms[0].values())[0] == set():
        df['localization_count'], df['top isoform'], df['top_terms'], df['localization score'] = None, None, None, None
    else:
        df['localization_count'], df['top isoform'], df['top_terms'], df['localization score'] = zip(
            *df.apply(lambda x: localization_of_modification(
                    ms, ms_label, x, locations_ms, params_dict, spectra_dict, mass_shift_dict), axis=1))
    fname = io.table_path(params_dict['output directory'], ms_label)
    peptide = params_dict['peptides_column']

    mod_aa = string.ascii_lowercase

    mod_dicts = {}
    for pair in locations_ms:
        labels_mod = {}
        i = 0
        for m in pair:
            labels_mod[mod_aa[i]] = m
            i += 1
        mod_dicts[tuple(sorted(pair))] = labels_mod
    columns = ['top isoform', 'localization score', params_dict['spectrum_column']]
    df['top isoform'] = df['top isoform'].fillna(df[peptide])
    df.loc[df.top_terms.notna(), 'mod_dict'] = df.loc[df.top_terms.notna(), 'top_terms'].apply(lambda t: mod_dicts[tuple(sorted(t))])
    df['top isoform'] = df.apply(utils.format_isoform, axis=1, args=(params_dict,))
    df[columns].to_csv(fname, index=False, sep='\t')
    result = df['localization_count'].sum() or Counter()
    logger.debug('Localization result for %s: %s', ms_label, result)
    return {ms_label: result}
