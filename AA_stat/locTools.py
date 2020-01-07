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
import os
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
    Calculates theoretical spectra in two ways: usual one. and formatter in integer (mz / frag_acc).
    `peptide` -peptide sequence
    `acc_frag` - accuracy of matching.
    `types` - ion types.
    `maxcharge` - maximum charge.

    ----------
    Returns spectra in two ways (usual, integer)
    """
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
    Matches experimental and theoretical spectra.
    `spectrum_idict` - mass in int format (real mz / fragment accuracy)
    `theoretical_set` -output of get_theor_spec, dict where keys is ion type, values
    masses in int format.
    `min_matched` - minumum peaks matched.

    ---------
    Return score
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


def peptide_isoforms(sequence, localizations, sum_mod=False):
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
    Returns dict. {'modification name': [list of amino acids]}

    """
    ind = abs(unimod_df['mono_mass']-mass_shift) < tolerance
    sites_set = set()
    for i, row in unimod_df.loc[ind].iterrows():
        sites_set.update(set(pd.DataFrame(row['specificity']).site))
    return list(sites_set)


def get_candidates_from_aastat(mass_shifts_table, labels, threshold = 1.5):
    df = mass_shifts_table.loc[:, labels]
    ms, aa = np.where(df > threshold)
    out = {ms: [] for ms in mass_shifts_table.index}
    for i, j in zip(ms, aa):
        out[df.index[i]].append(df.columns[j])
    return pd.Series(out)


def find_isotopes(ms, tolerance=0.01):
    """
    Find the isotopes from the `mass_shift_list` using mass difference of C13 and C12, information of amino acids statistics as well.
    `locmod_ms` - Series there index in mass in str format, values actual mass shift.
    -----------
    Returns Series of boolean.
    """
    out = pd.DataFrame({'isotope': False, 'monoisotop_index': False}, index=ms.index)
    np_ms = ms.to_numpy()
    difference_matrix = np.abs(np_ms.reshape(-1, 1) - np_ms.reshape(1, -1) - DIFF_C13)
    isotop, monoisotop = np.where(difference_matrix < tolerance)
    out.iloc[isotop, 0] = True
    out.iloc[isotop, 1] = out.iloc[monoisotop, :].index
    return out


def find_mod_sum(x, df, sum_matrix, tolerance):
    out = df.loc[np.where(np.abs(sum_matrix - x['mass_shift']) < tolerance)[0],'mass_shift'].to_list()
    if len(out):
        return out
    else:
        return False


def find_modifications(ms, tolerance=0.005):
    """
    Finds the sums of mass shifts, if it exists.
    Returns Series, where index is the mass in str format, values is list of mass shifts that form the mass shift.
    """
    zero = utils.mass_format(0.0)
    if zero in ms.index:
        col = ms.drop(zero)
    else:
        col = ms
        logger.info('Zero mass shift not found in candidates.')
    df = pd.DataFrame({'mass_shift': col.values, 'index': col.index}, index=range(len(col)))
    sum_matrix = df['mass_shift'].to_numpy().reshape(-1, 1) + df['mass_shift'].to_numpy().reshape(1, -1)
    df['out'] = df.apply(lambda x: find_mod_sum(x, df, sum_matrix, tolerance), axis=1)
    df.index = df['index']
    return df.out


def localization_of_modification(mass_shift, row, loc_candidates, params_dict, spectra_dict, tolerance=FRAG_ACC, sum_mod=False):
    mass_dict = mass.std_aa_mass
    peptide = params_dict['peptides_column']
    sequences = peptide_isoforms(row[peptide], loc_candidates, sum_mod=sum_mod)
    if not sequences:
        return Counter()
    if sum_mod:
        mass_dict.update({'m': mass_shift[0], 'n': mass_shift[1], 'k': mass_shift[2]})
        loc_cand, loc_cand_1, loc_cand_2  = loc_candidates
        if mass_shift[1] == mass_shift[2]:
            logger.debug('Removing duplicate isoforms for %s', mass_shift)
            sequences = {s.replace('k', 'n') for s in sequences}
    else:
        mass_dict.update({'m': mass_shift[0]})
        loc_cand = loc_candidates

    if params_dict['mzml_files']:
        scan = row[params_dict['spectrum_column']].split('.')[1]
        spectrum_id = 'controllerType=0 controllerNumber=1 scan=' + scan
    else:
        spectrum_id = row[params_dict['spectrum_column']]
    exp_spec = spectra_dict[row['file']].get_by_id(spectrum_id)
    tmp = exp_spec['m/z array'] / tolerance
    tmp = tmp.astype(int)
    loc_stat_dict = Counter()
    exp_dict = dict(zip(tmp, exp_spec['intensity array']))
    scores = [] # write for same scores return non-loc
    charge = row[params_dict['charge_column']]

    sequences = np.array(list(sequences))
    for seq in sequences:
        theor_spec = get_theor_spectrum(seq, tolerance, maxcharge=charge, aa_data=mass_dict)
        scores.append(RNHS_fast(exp_dict, theor_spec[1], MIN_SPEC_MATCHED)[1])

    scores = np.array(scores)
    i = np.argsort(scores)[::-1]
    scores = scores[i]
    sequences = sequences[i]
    if logger.level <= logging.DEBUG:
        fname = os.path.join(params_dict['out_dir'], utils.mass_format(mass_shift[0])+'.txt')
        # logger.debug('Writing isoform scores for %s to %s', row[peptide], fname)
        with open(fname, 'a') as dump:
            for seq, score in zip(sequences, scores):
                dump.write('{}\t{}\n'.format(seq, score))
            dump.write('\n')
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
            loc_stat_dict[top_isoform[loc_index + 1] +'_mod1'] += 1
            if loc_index_2 == -1:
                loc_stat_dict[top_isoform[loc_index_1 + 1] +'_mod1'] += 1
            else:
                loc_stat_dict[top_isoform[loc_index_2 + 1] +'_mod2'] += 1
        if 'N-term' in loc_cand_1 and loc_index == 0:
            loc_stat_dict['N-term_mod1'] += 1
        if 'C-term' in loc_cand_1 and loc_index == len(top_isoform) - 2:
            loc_stat_dict['C-term_mod1'] += 1
        if 'N-term' in loc_cand_2 and loc_index_2 == 0:
            loc_stat_dict['N-term_mod2'] += 1
        if 'C-term' in loc_cand_2 and loc_index_2 == len(top_isoform) - 2:
            loc_stat_dict['C-term_mod2'] += 1

    if not loc_stat_dict:
        return Counter()
    else:
        return loc_stat_dict


def two_step_localization(df, ms, locations_ms, params_dict, spectra_dict, sum_mod=False):
    logger.debug('Localizing %s (sum_mod = %s)', ms, sum_mod)
    tmp = df.apply(lambda x: localization_of_modification(
                    ms, x, locations_ms, params_dict, spectra_dict, sum_mod=sum_mod), axis=1)
    new_localizations = set(tmp.sum().keys()).difference({'non-localized'})

    if sum_mod:
        locations_ms0 = []
        locations_ms1 = []
        locations_ms2 = []
        for i in new_localizations:
            if i.endswith('mod1'):
                locations_ms1.append(i.split('_')[0])
            elif i.endswith('mod2'):
                locations_ms2.append(i.split('_')[0])
            else:
                locations_ms0.append(i)
        if ms[1] == ms[-1]:
            locations_ms2 = locations_ms1[:]
        logger.debug('new locs: %s, %s, %s', locations_ms0, locations_ms1, locations_ms2)
        new_localizations = [locations_ms0, locations_ms1, locations_ms2]

    if new_localizations != locations_ms:
        logger.debug('new localizations: %s', new_localizations)
        df['loc_counter'] = df.apply(lambda x: localization_of_modification(
            ms, x, new_localizations, params_dict, spectra_dict, sum_mod=sum_mod), axis=1)
    else:
        df['loc_counter'] = tmp
