import os
import operator
import logging
import pandas as pd
import numpy as np
import warnings
from collections import Counter
import re
import string
import pathlib
import itertools as it
from pyteomics import parser, pepxml, mass

logger = logging.getLogger(__name__)

MASS_FORMAT = '{:+.4f}'
UNIMOD = mass.Unimod(pathlib.Path(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'unimod.xml')).as_uri())
INTERNAL = 5
DIFF_C13 = mass.calculate_mass(formula='C[13]') - mass.calculate_mass(formula='C')
H = mass.nist_mass['H+'][0][0]


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


def get_ms_from_df(mass_shift, data, params_dict):
    shifts = params_dict['mass_shifts_column']
    ms_shift = data.loc[np.abs(data[shifts] - mass_shift[1]) < mass_shift[2], shifts].mean()

    mask = np.abs(data[shifts] - mass_shift[1]) < 3 * mass_shift[2]
    internal('Mass shift %.3f +- 3 * %.3f', mass_shift[1], mass_shift[2])
    data_slice = data.loc[mask].sort_values(by=[params_dict['score_column'],
        params_dict['spectrum_column']],
        ascending=params_dict['score_ascending']).drop_duplicates(subset=params_dict['peptides_column'])
    return ms_shift, data_slice


def fdr_filter_mass_shift(mass_shift, data, params_dict, preprocessing=False):
    if preprocessing:
        ms_shift, data_slice = get_ms_from_df(mass_shift, data, params_dict)
    else:
        ms_shift, data_slice = data.get_raw_data_by_ms(mass_shift)
    internal('%d peptide rows selected for filtering', data_slice.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pepxml.filter_df(data_slice, key=params_dict['score_column'], fdr=params_dict['FDR'],
            reverse=not params_dict['score_ascending'], correction=params_dict['FDR_correction'], is_decoy='is_decoy')
    internal('Filtered data for %s: %d rows', mass_shift, df.shape[0])
    return ms_shift, df


def group_specific_filtering(data, mass_shifts, params_dict):
    """
    Selects window around found mass shift and filters using TDA.
    Window is defined as mean +- sigma.

    Parameters
    ----------
    data : io.PsmDataHandler
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
        sites_set.update(s['site'] if s['position'][:3] == 'Any' else s['position'] for s in row['specificity'])
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


def apply_var_mods(seq, mods):
    parsed = parser.parse(seq)
    out = []
    for i, aa in enumerate(parsed):
        if i in mods:
            out.append('{{{:+.0f}}}'.format(mods[i]) + aa)
        else:
            out.append(aa)
    seqout = ''.join(out)
    internal('%s + %s = %s', seq, mods, seqout)
    return seqout


def get_column_with_mods(row, params_dict):
    peptide = params_dict['peptides_column']
    mods = get_var_mods(row, params_dict)
    return apply_var_mods(row[peptide], mods)


def format_isoform(row, params_dict):
    ms = row['mod_dict']
    seq = row['top isoform']

    pc, nc, mc = operator.itemgetter('prev_aa_column', 'next_aa_column', 'mods_column')(params_dict)
    prev_aa, next_aa = operator.itemgetter(pc, nc)(row)
    mods = get_var_mods(row, params_dict)
    seq = apply_var_mods(seq, mods)

    sequence = re.sub(r'([a-z])([A-Z])', lambda m: '{}[{:+.0f}]'.format(m.group(2), float(ms[m.group(1)])), seq)
    return '{}.{}.{}'.format(prev_aa[0], sequence, next_aa[0])


def get_fix_var_modifications(pepxml_file, labels):
    fout, vout = {}, []
    p = pepxml.PepXML(pepxml_file, use_index=False)
    mod_list = list(p.iterfind('aminoacid_modification'))
    logger.debug('mod_list: %s', mod_list)
    p.reset()
    term_mods = list(p.iterfind('terminal_modification'))
    logger.debug('term_mods: %s', term_mods)
    p.close()
    for m in mod_list:
        if m['aminoacid'] not in labels:
            continue
        if 'peptide_terminus' in m:
            key = '{}-term {}'.format(m['peptide_terminus'].upper(), m['aminoacid'])
        else:
            key = m['aminoacid']
        if m['variable'] == 'N':
            fout[key] = m['mass']
        else:
            vout.append((key, m['massdiff']))
    for m in term_mods:
        if m['variable'] == 'N':
            if m['terminus'] == 'N':
                fout['H-'] = m['mass']
            else:
                fout['-OH'] = m['mass']
        else:
            key = ('Protein ' if m.get('protein_terminus') == 'Y' else '') + m['terminus'] + '-term'
            vout.append((key, m['massdiff']))
    return fout, vout


def get_specificity(pepxml_file):
    with pepxml.PepXML(pepxml_file, use_index=False) as p:
        s = next(p.iterfind('specificity'))
    logger.debug('Extracted enzyme specificity: %s', s)
    return s


def parse_l10n_site(site):
    aa, shift = site.split('_')
    return aa, shift


def mass_to_mod(label, value, aa_mass=mass.std_aa_mass):
    words = label.split()
    if len(words) > 1:
        # terminal mod
        label = words[-1]
    return value - aa_mass.get(label, 0)


def masses_to_mods(d, fix_mod=None):
    aa_mass = mass.std_aa_mass.copy()
    aa_mass['H-'] = 1.007825
    aa_mass['-OH'] = 17.00274
    if fix_mod:
        aa_mass.update(fix_mod)
    d = {k: mass_to_mod(k, v, aa_mass) for k, v in d.items()}
    if 'H-' in d:
        d['N-term'] = d.pop('H-')
    if '-OH' in d:
        d['C-term'] = d.pop('-OH')
    return d


def get_var_mods(row, params_dict):
    # produce a dict for specific PSM: position (int) -> mass shift (float)
    modifications = row[params_dict['mods_column']]
    peptide = params_dict['peptides_column']
    mass_dict_0 = mass.std_aa_mass.copy()
    mass_dict_0['H-'] = 1.007825
    mass_dict_0['-OH'] = 17.00274
    mass_dict_0.update(params_dict['fix_mod'])
    mod_dict = {}
    if modifications:
        internal('Got modifications for peptide %s: %s', row[peptide], modifications)
    for m in modifications:
        # internal('Parsing modification: %s', m)
        mmass, pos = m.split('@')
        mmass = float(mmass)
        pos = int(pos)
        if pos == 0:
            key = 'H-'
        elif pos == len(row[peptide]) + 1:
            key = '-OH'
        else:
            key = row[peptide][pos-1]
        if abs(mmass - mass_dict_0[key]) > params_dict['frag_acc']:
            # utils.internal('%s modified in %s at position %s: %.3f -> %.3f', key, row[peptide], pos, mass_dict_0[key], mmass)
            mod_dict[pos] = mmass - mass_dict_0[key]
    if mod_dict:
        internal('Final mod dict: %s', mod_dict)
    return mod_dict


def format_grouped_keys(items, params_dict):
    out = []
    for k, td in items:
        if k[1:] == '-term':
            t = k[0]
            if isinstance(td, list):
                keys, values = zip(*td)
                diff = max(values) - min(values)
                label_condition = set(keys) >= set(params_dict['labels'])
                if diff < params_dict['prec_acc'] and label_condition:
                    out.append((k, values[0]))  # arbitrary amino acid, they all have the same modification
                    logger.debug('Collapsing %s-terminal mods.', t)
                else:
                    logger.debug('Not collapsing %s-term dict: diff in values is %.3f, set of labels condition is %ssatisfied',
                        t, diff, '' if label_condition else 'not ')
                    for aa, v in td:
                        out.append((k + ' ' + aa, v))
            else:
                out.append((k, td))
        else:
            out.append((k, td))
    logger.debug('Variable mods with grouped keys: %s', out)
    return out


def group_terminal(items):
    grouped = []
    tg = {}
    for k, v in items:
        prefix, protein, term, aa = re.match(r'((Protein)?(?: )?([NC]-term)?)(?: )?([A-Z])?', k).groups()
        if term is None or aa is None:
            grouped.append((k, v))
        else:
            tg.setdefault(prefix, []).append((aa, v))
    grouped.extend(tg.items())
    logger.debug('Variable mods after grouping: %s', grouped)
    return grouped


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


def measured_mz_series(df, params_dict):
    return (df[params_dict['measured_mass_column']] + df[params_dict['charge_column']] * H
        ) / df[params_dict['charge_column']]


def calculated_mz_series(df, params_dict):
    return (df[params_dict['calculated_mass_column']] + df[params_dict['charge_column']] * H
        ) / df[params_dict['charge_column']]


def format_list(lst, sep1=', ', sep2=' or '):
    lst = list(lst)
    if not lst:
        return ''
    if len(lst) == 1:
        return lst[0]
    *most, last = lst
    return sep1.join(most) + sep2 + last


def find_mass_shift(value, data, tolerance):
    data_dict = data.ms_stats()
    s = sorted(data_dict, key=lambda x: abs(value - data_dict[x][0]))
    if abs(data_dict[s[0]][0] - value) <= tolerance:
        return s[0]


def loc_positions(top_isoform):
    return [i for i, a in enumerate(top_isoform, 1) if len(a) > 1]


def choose_correct_massdiff(reported, calculated, params_dict):
    maxdiff = np.abs(reported - calculated).max()
    if maxdiff < params_dict['bin_width'] / 2:
        logger.debug('Maximum mass diff is within bounds: %.4f', maxdiff)
        return reported
    elif maxdiff < params_dict['prec_acc']:
        logger.warning('Reported mass shifts have a high calculation error (%.4f).'
        ' Using own calculations', maxdiff)
        return calculated
    else:
        logger.warning('Reported mass shifts differ from calculated values (up to %.4f).'
        ' Using the reported values. Consider reporting this to the developers.', maxdiff)
        return reported


def convert_tandem_cleave_rule_to_regexp(cleavage_rule, params_dict):

    def get_sense(c_term_rule, n_term_rule):
        if '{' in c_term_rule:
            return 'N'
        elif '{' in n_term_rule:
            return 'C'
        else:
            if len(c_term_rule) <= len(n_term_rule):
                return 'C'
            else:
                return 'N'

    def get_cut(cut, no_cut):
        aminoacids = set(params_dict['labels'])
        cut = ''.join(aminoacids & set(cut))
        if '{' in no_cut:
            no_cut = ''.join(aminoacids & set(no_cut))
            return cut, no_cut
        else:
            no_cut = ''.join(set(params_dict['labels']) - set(no_cut))
            return cut, no_cut

    protease = cleavage_rule.replace('X', ''.join(params_dict['labels']))
    c_term_rule, n_term_rule = protease.split('|')
    sense = get_sense(c_term_rule, n_term_rule)
    if sense == 'C':
        cut, no_cut = get_cut(c_term_rule, n_term_rule)
    else:
        cut, no_cut = get_cut(n_term_rule, c_term_rule)
    return {'sense': sense, 'cut': cut, 'no_cut': no_cut}


def parse_mod_list(s, kind):
    pairs = re.split(r'\s*[,;]\s*', s)
    if kind == 'fixed':
        out = {}
    elif kind == 'variable':
        out = []
    else:
        raise ValueError('`kind` must be "fixed" or "variable", not "{}".'.format(kind))

    for p in pairs:
        if p:
            m, aa = re.split(r'\s*@\s*', p)
            m = float(m)
            if kind == 'fixed':
                if aa == 'N-term':
                    out['H-'] = 1.007825 + m
                elif aa == 'C-term':
                    out['-OH'] = 17.00274 + m
                else:
                    out[aa] = mass.std_aa_mass[aa] + m
            else:
                out.append((aa, m))
    return out


def get_spectrum_id(row, params_dict):
    if params_dict['mzml_files']:
        scan = row[params_dict['spectrum_column']].split('.')[1].lstrip('0')
        return 'controllerType=0 controllerNumber=1 scan=' + scan
    return row[params_dict['spectrum_column']]


def get_loc_stats(top_isoform, mass_dict_0, top_terms, mass_shift_dict, params_dict):
    mass_dict = mass_dict_0.copy()
    modif_labels = string.ascii_lowercase
    # utils.internal('Top isoform is %s for terms %s (shift %s)', top_isoform, top_terms, ms_label)
    i = 0
    for _ms in top_terms:
        mod_aa = {modif_labels[i] + aa: mass_shift_dict[_ms] + mass_dict[aa] for aa in params_dict['labels']}
        mass_dict.update(mod_aa)
        mass_dict[modif_labels[i]] = mass_shift_dict[_ms]
        i += 1
    loc_stat_dict = Counter()
    for ind, a in enumerate(top_isoform):
        if len(a) > 1:
            if ind == 0:
                loc_stat_dict[format_localization_key('N-term', mass_dict[a[0]])] += 1
            elif ind == len(top_isoform) - 1:
                loc_stat_dict[format_localization_key('C-term', mass_dict[a[0]])] += 1
            loc_stat_dict[format_localization_key(a[1], mass_dict[a[0]])] += 1
    return loc_stat_dict
