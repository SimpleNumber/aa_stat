from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as  np
import pylab as plt
import os

import seaborn as sb
from collections import defaultdict
from scipy.stats import ttest_ind

import logging
import warnings
from pyteomics import parser, pepxml
from . import utils
logger = logging.getLogger(__name__)

cc = ["#FF6600",
      "#FFCC00",
      "#88AA00",
      "#006688",
      "#5FBCD3",
      "#7137C8",
      ]
sb.set_style('white')
colors = sb.color_palette(palette=cc)

AA_STAT_CAND_THRESH = 1.5
ISOTOPE_TOLERANCE = 0.015
UNIIMOD_TOLERANCE = 0.01


def get_peptide_statistics(peptide_list, rule):
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
    """
    Calculates amino acid statistics in a `peptide_list` and cleaves miscleaved peptides according to `rule`.
    Returns dict with amino acids as keys and their relative abundances as values.
    """
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

    A table with mass shifts, psms, aa statistics columns.
    '''

    unimod = pd.Series({i: utils.get_unimod_url(float(i)) for i in number_of_PSMs.index})
    df = pd.DataFrame({'mass shift': [mass_shifts[k] for k in distributions.columns],
                       '# peptides in bin': number_of_PSMs},
                      index=distributions.columns)
    df['# peptides in bin'] = df['# peptides in bin'].astype(np.int64)
    out = pd.concat([df, distributions.T], axis=1)
    out['Unimod'] = unimod
    out.index = range(len(out))
    i = ((out.drop(columns=['mass shift', 'Unimod', '# peptides in bin']).max(axis=1) - 1) * out['# peptides in bin']).argsort()
    return out.loc[i.values[::-1], :]


def calculate_error_and_p_vals(pep_list, err_ref_df, reference, rule, l):
    d = pd.DataFrame(index=l)
    for i in range(50):
        d[i] = pd.Series(get_aa_distribution(
            np.random.choice(np.array(pep_list),
            size=(len(pep_list) // 2), replace=False), rule)) / reference
    p_val = pd.Series()
    for i in l:
        p_val[i] = ttest_ind(err_ref_df.loc[i, :], d.loc[i, :])[1]
    return p_val, d.std(axis=1)


def summarizing_hist(table, save_directory):

    ax = table.sort_values('mass shift').plot(
        y='# peptides in bin', kind='bar', color=colors[2], figsize=(len(table), 5))
    ax.set_title("Peptides in mass shifts", fontsize=12) #PSMs
    ax.set_xlabel("Mass shift", fontsize=10)
    ax.set_ylabel('Number of peptides')
    ax.set_xticklabels(table.sort_values('mass shift')['mass shift'].apply(lambda x: round(x, 2)))

    total = sum(i.get_height() for i in ax.patches)
    max_height = 0
    for i in ax.patches:
        current_height = i.get_height()
        if current_height > max_height:
            max_height = current_height
        ax.text(i.get_x()-.03, current_height + 200,
            '{:.2%}'.format(i.get_height() / total), fontsize=10, color='dimgrey')

    plt.ylim(0, max_height * 1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, 'summary.png'), dpi=500)
    plt.savefig(os.path.join(save_directory, 'summary.svg'))


def get_zero_mass_shift(mass_shifts):
#    print(mass_shifts)
    """
    Shift of non-modified peak.
    Returns float.
    """
    l  = np.argmin(np.abs(mass_shifts))
    return mass_shifts[l]


def filter_mass_shifts(results):

    """
    Filter mass_shifts that close to each other.

    Return poptperr matrix.
    """
    logger.info('Discarding bad peaks...')
    out = []
    for ind, mass_shift in enumerate(results[:-1]):
        mean_diff = (results[ind][1] - results[ind+1][1]) ** 2
        sigma_diff = (results[ind][2] + results[ind+1][2]) ** 2
        if mean_diff > sigma_diff:
            out.append(mass_shift)
        else:
            logger.info('Joined mass shifts %.4f and %.4f', results[ind][1], results[ind+1][1])
    logger.info('Peaks for following analysis: %s', len(out))
    return out


def group_specific_filtering(data, mass_shifts, params_dict):
    """
    Selects window around found mass shift and filter using TDA. Window is defined as mu +- 3*sigma.
    Returns....
    """
    shifts = params_dict['mass_shifts_column']
    logger.info('Performing group-wise FDR filtering...')
    out_data = {} # dict corresponds list
    for mass_shift in mass_shifts:
        data_slice = data[np.abs(data[shifts] - mass_shift[1]) < 3 * mass_shift[2]].sort_values(by='expect') \
                         .drop_duplicates(subset=params_dict['peptides_column'])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pepxml.filter_df(data_slice,
                fdr=params_dict['FDR'], correction=params_dict['FDR_correction'], is_decoy='is_decoy')
        if len(df) > 0:
            out_data[np.mean(df[shifts])] = df   ###!!!!!!!mean of from gauss fit!!!!
    logger.info('# of filtered mass shifts = %s', len(out_data))
    return out_data


def plot_figure(ms_label, ms_counts, left, right, params_dict, save_directory):
    """
    'ms_label' mass shift in string format.
    'ms_counts' entries in a mass shift.
    'left

    """

    b = 0.2 # shift in bar plots
    width = 0.4 # for bar plots
    labels = params_dict['labels']
    distributions = left[0]
    errors = left[1]
    bar_plot, bar_left = plt.subplots()
    bar_plot.set_size_inches(params_dict['figsize'])
    bar_left.bar(np.arange(b, 2 * len(labels), 2), distributions.loc[labels,ms_label],
            yerr=errors.loc[labels], width=width, color=colors[2], linewidth=0,
            label=ms_label+' Da mass shift,\n'+str(ms_counts)+' peptides')
    bar_left.set_ylabel('Relative AA abundance', color=colors[2])
    bar_left.set_xticks(np.arange(2 * b , 2 * len(labels) + 2 * b, 2))#
    bar_left.set_xticklabels(labels)
    bar_left.hlines(1, -1, 2* len(labels), linestyles='dashed', color=colors[3])
    bar_right = bar_left.twinx()
    bar_right.bar(np.arange(4 * b, 2 * len(labels) + 4 * b, 2),right, width=width, linewidth=0, color=colors[0])
    bar_right.set_ylim(0,125)
    bar_right.set_yticks(np.arange(0,120, 20))
    bar_right.set_ylabel('Peptides with AA, %', color=colors[0])

    bar_left.spines['left'].set_color(colors[2])
    bar_right.spines['left'].set_color(colors[2])

    bar_left.spines['right'].set_color(colors[0])
    bar_right.spines['right'].set_color(colors[0])
    bar_left.tick_params('y', colors=colors[2])
    bar_right.tick_params('y', colors=colors[0])
    bar_right.annotate(ms_label + ' Da mass shift,'  + '\n' + str(ms_counts) +' peptides',
                      xy=(29,107), bbox=dict(boxstyle='round',fc='w', edgecolor='dimgrey'))
    bar_left.set_xlim(-3*b, 2*len(labels)-2 +9*b)
    bar_left.set_ylim(0,distributions.loc[labels, ms_label].max()*1.3)
    bar_plot.savefig(os.path.join(save_directory, ms_label + '.png'), dpi=500)
    bar_plot.savefig(os.path.join(save_directory, ms_label + '.svg'))
    plt.close()


def calculate_statistics(mass_shifts_dict, zero_mass_shift, params_dict, args):
    """
    Plot amino acid statistics
    'zero_mass_shift' is a systematic shift of zero masss shift, float.
    'mass_shifts_dict' is a dict there keys are mass shifts(float)
    and values are DataFrames of filtered windows(3 sigma) around this mass.
    'params_dict' is a dict of parameters from parsed cfg file.
    'args' files paths (need to take the saving directory)
    """
    logger.info('Plotting distributions...')
    labels = params_dict['labels']
    rule = params_dict['rule']
    expasy_rule = parser.expasy_rules.get(rule, rule)
    save_directory = args.dir
    mass_shifts_dict_formatted ={utils.mass_format(k): mass_shifts_dict[k] for k in mass_shifts_dict.keys()} # mass_shift_dict with printable labels
    mass_shifts_labels = {utils.mass_format(i): i for i in mass_shifts_dict.keys()}
    zero_mass_shift_label = utils.mass_format(zero_mass_shift)
    number_of_PSMs = dict()#pd.Series(index=list(mass_shifts_labels.keys()), dtype=int)
    reference = pd.Series(get_aa_distribution(mass_shifts_dict_formatted[zero_mass_shift_label][params_dict['peptides_column']], expasy_rule))
    reference.fillna( 0, inplace=True)

    #bootstraping for errors and p values calculation in reference(zero) mass shift
    err_reference_df = pd.DataFrame(index=labels)
    for i in range(50):
        err_reference_df[i] = pd.Series(get_aa_distribution(
        np.random.choice(np.array(mass_shifts_dict_formatted[zero_mass_shift_label][params_dict['peptides_column']]),
        size=(len(mass_shifts_dict_formatted[zero_mass_shift_label]) // 2), replace=False),
        expasy_rule)) / reference

    logger.info('Mass shifts:')
    distributions = pd.DataFrame(index=labels)
    p_values = pd.DataFrame(index=labels)
    for ms_label, ms_df in mass_shifts_dict_formatted.items():
        aa_statistics = pd.Series(get_aa_distribution(ms_df[params_dict['peptides_column']], expasy_rule))
        peptide_stat = pd.Series(get_peptide_statistics(ms_df[params_dict['peptides_column']], expasy_rule))
        number_of_PSMs[ms_label] = len(ms_df)
        aa_statistics.fillna(0, inplace=True)
        distributions[ms_label] = aa_statistics / reference
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_vals, errors = calculate_error_and_p_vals(
            ms_df[params_dict['peptides_column']], err_reference_df, reference, expasy_rule, labels)
#        errors.fillna(0, inplace=True)

        p_values[ms_label] = p_vals
        distributions.fillna(0, inplace=True)

        labels_df = pd.DataFrame(index=labels)
        labels_df['pep_stat'] = pd.Series(peptide_stat)
        labels_df.fillna(0, inplace=True)
        plot_figure(ms_label, len(ms_df), [distributions, errors], labels_df['pep_stat'], params_dict, save_directory )
        logger.info('%s Da', ms_label)

#    pout.insert(0, 'mass shift', [mass_shifts[i] for i in pout.index])
    pout = p_values.T
    pout.fillna(0).to_csv(os.path.join(save_directory, 'p_values.csv'), index=False)
    return distributions, pd.Series(number_of_PSMs), mass_shifts_labels


def render_html_report(table_, params_dict, save_directory):
    table = table_.copy()
    labels = params_dict['labels']
    report_template = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report.template')
    with open(report_template) as f:
        report = f.read()
    with pd.option_context('display.max_colwidth', -1):
        columns = list(table.columns)
        mslabel = '<a id="binh" href="#">mass shift</a>'
        columns[0] = mslabel
        table.columns = columns
        table_html = table.style.hide_index().applymap(
            lambda val: 'background-color: yellow' if val > 1.5 else '', subset=labels
            ).set_precision(3).set_table_styles([
            {'selector': 'tr:hover', 'props': [('background-color', 'lightyellow')]},
            {'selector': 'td, th', 'props': [('text-align', 'center')]},
            {'selector': 'td, th', 'props': [('border', '1px solid black')]}]
            ).format({'Unimod': '<a href="{}">search</a>'.format,
                mslabel: '<a href="#">{}</a>'.format(utils.MASS_FORMAT).format}
            ).bar(subset='# peptides in bin', color=cc[2]).render() #PSMs
    report = report.replace(r'%%%', table_html)
    with open(os.path.join(save_directory, 'report.html'), 'w') as f:
        f.write(report)


def systematic_mass_shift_correction(mass_shifts_dict, mass_correction):
   '''
   `mass_shifts_dict` - dict where keys are mass shifts (float) and values DataFrames that correspond to this mass shift.
   '''
   out = {}
   for k, v in mass_shifts_dict.items():
       out[k-mass_correction] = v
   return out
