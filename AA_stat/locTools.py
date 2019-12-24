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
import pylab as plt
import os
import argparse
import ast
import seaborn as sb
from collections import defaultdict
from scipy.signal import argrelextrema, savgol_filter
from scipy.stats import ttest_ind
from scipy.optimize import curve_fit
import logging
import warnings
from pyteomics import parser, pepxml, mgf, mzml, mass
from pyteomics import electrochem as ec
from scipy.spatial import cKDTree
import numpy as np
from math import factorial
from copy import copy
try:
    from pyteomics import cmass
except ImportError:
    cmass = mass

#    import customparser as cparser
DIFF_C13 = mass.calculate_mass(formula='C[13]') - mass.calculate_mass(formula='C')
def get_spectra_from_mgf(file_dir, spectrum, suffix, tolerance=0.01):
    """
    Retrive MSMS `spectra` from `file_name` file.
    Returns  spectrum_idict {int_mass:intensity}
    """
    file_name = '.'.join([spectrum.split('.')[0], suffix])
    mgf_file = mgf.read(os.path.join(file_dir, file_name))
    mgf_file.get_spectrum(spectrum)
    experimantal_spectrum =  mgf_file.get_spectrum(spectrum)
    mz = np.array(experimantal_spectrum['m/z array'])/ tolerance 
    return dict(zip(mz.round(0).astype('int64'), experimantal_spectrum['intensity array']))

def get_theor_spectrum(peptide, acc_frag, types=('b', 'y'), maxcharge=None, reshape=False, **kwargs):
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
            if not reshape:
                marr.sort()
            else:
                n = marr.size
                marr = marr.reshape((n, 1))
            peaks[ion_type, charge] = marr
    return peaks, theoretical_set


def RNHS_fast(spectrum_idict, theoretical_set, min_matched):
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
    
    
def peptide_isoforms(sequence, variable_mods,):
    """
    Forms list of modified amino acid candidates.
    `variable_mods` - dict of name (key) and amino acids (values)
    Return list of lists. [[isoform1],[isoform2]] 
    
    """

    isoforms = []
    for j in  parser.isoforms(sequence, variable_mods=variable_mods, format='split'):
        isoforms.uppend([''.join(i) for i in j])
    return isoforms

def get_candidates_from_unimod(mass_shift, tolerance, unimod_db, unimod_df):
    """
    Find modifications for `mass_shift` in Unimod.org database with a given `tolerance`.
    Returns dict. {'modification name': [list of amino acids]}
    
    """
    ind = list(unimod_df[abs(unimod_df['mono_mass']-mass_shift) < tolerance].index)
    sites_set = set()
    for i in unimod_db[ind]:
        sites_set.update(set(pd.DataFrame(i['specificity']).site))
    return list(sites_set)

def get_candidates_from_aastat(mass_shifts_table, labels, threshold = 1.5,): 
    df = mass_shifts_table.loc[:,labels]
    ms, aa = np.where(df > threshold)
    out = {ms:[] for ms in mass_shifts_table.index}
    for i,j in zip(ms, aa):
        out[df.index[i]].append(df.columns[j])
    return pd.Series(out)
    
def find_isotopes(ms, tolerance=0.01):
    """
    Find the isotopes from the `mass_shift_list` using mass difference of C13 and C12, information of amino acids statistics as well.
    `locmod_ms` - Series there index in mass in str format, values actual mass shift.
    -----------
    Returns Series of boolean.
    """
    out = pd.Series(False, index=ms.index)
    np_ms = ms.to_numpy()
    difference_matrix = np.abs(np_ms.reshape(-1, 1) - np_ms.reshape(1, -1) - DIFF_C13)
    isotop, monoisotop = np.where(difference_matrix < tolerance)
    out.iloc[isotop] = True
    return out
def find_mod_sum(x, df, sum_matrix, tolerance):
    out = df.loc[np.where(np.abs(sum_matrix - x['mass_shift']) < tolerance)[0],'mass_shift'].to_list() 
    if len(out):
        return out
    else:
        return False
def find_modifications(ms, tolerance=0.005):
    """
    Finds the sums of mass shifts, if they exist.
    Returns Series, where index is the mass in str format, values is list of mass shifts that form the mass shift. 
    """
    col = ms.drop('0.0000') #drop zero mass shift
    df = pd.DataFrame({'mass_shift':col.values, 'index':col.index}, index=range(len(col)) )
    sum_matrix = df['mass_shift'].to_numpy().reshape(-1,1) + df['mass_shift'].to_numpy().reshape(1, -1)
    df['out'] = df.apply(lambda x: find_mod_sum(x, df, sum_matrix, tolerance), axis=1)
    df.index = df['index']
    return df.out

def localization_of_modification():
    pass
        

