from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as  np
import pylab as plt
import os

import ast
import seaborn as sb
from collections import defaultdict
from scipy.signal import argrelextrema, savgol_filter
from scipy.stats import ttest_ind
from scipy.optimize import curve_fit
import logging
import warnings
from pyteomics import parser, pepxml, mgf, mzml
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
MASS_FORMAT = '{:.4f}'
AA_STAT_CAND_THRESH = 1.5
ISOTOPE_TOLERANCE = 0.015
UNIIMOD_TOLERANCE = 0.01


def mass_format(mass):
    return MASS_FORMAT.format(mass)


def get_unimod_url(mass_shift):
    return ('http://www.unimod.org/modifications_list.php'
        '?a=search&value=1&SearchFor={:.0f}.&'
        'SearchOption=Starts+with+...&SearchField=mono_mass'.format(mass_shift))


def make_0mc_peptides(pep_list, rule):
    out_set = set()
    for i in pep_list:
        out_set.update(parser.cleave(i, rule))
    return out_set


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
    pep_set = make_0mc_peptides(peptide_list, rule)
    d = defaultdict(int)
    for seq in pep_set:
        for let in seq:
            d[let] += 1
            sum_aa += 1
    for i in d:
        d[i] /= sum_aa
    return d


def smooth(y, window_size=15, power=5):
    y_smooth = savgol_filter(y, window_size, power)
    return y_smooth


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

    unimod = pd.Series({i: get_unimod_url(float(i)) for i in number_of_PSMs.index})
    df = pd.DataFrame({'mass shift': [mass_shifts[k] for k in distributions.columns],
                       '# peptides in bin': number_of_PSMs},
                      index=distributions.columns)
    df['# peptides in bin'] = df['# peptides in bin'].astype(np.int64)
    out = pd.concat([df, distributions.T], axis=1)
    out['Unimod'] = unimod
    out.index = range(len(out))
    i = ((out.drop(columns=['mass shift', 'Unimod', '# peptides in bin']).max(axis=1) - 1) * out['# peptides in bin']).argsort()
    return out.loc[i.values[::-1], :]


def read_pepxml(fname, params_dict):
    return pepxml.DataFrame(fname, read_schema=False)


def read_csv(fname, params_dict):
    df = pd.read_csv(fname, sep=params_dict['csv_delimiter'])
    protein = params_dict['proteins_column']
    if (df[protein].str[0] == '[').all() and (df[protein].str[-1] == ']').all():
        df[protein] = df[protein].apply(ast.literal_eval)
    else:
        df[protein] = df[protein].str.split(params_dict['proteins_delimeter'])
    return df


def read_input(args, params_dict):
    """
    Reads open search output, assembles all data in one DataFrame.
    """
    dfs = []
    data = pd.DataFrame()
    window = 0.3
    zero_bin = 0
    logger.info('Reading input files...')
    readers = {
        'pepxml': read_pepxml,
        'csv': read_csv,
    }
    for ftype, reader in readers.items():
        filenames = getattr(args, ftype)
        if filenames:
            for filename in filenames:
                logger.info('Reading %s', filename)
                df = reader(filename, params_dict)
                hist_0 = np.histogram(df[abs(df[params_dict['mass_shifts_column']] - zero_bin) < window/2][params_dict['mass_shifts_column']], bins=10000)
                logger.debug('hist_0: %s', hist_0)
                hist_y = hist_0[0]
                hist_x = 1/2 * (hist_0[1][:-1] +hist_0[1][1:])
                popt, perr = gauss_fitting(max(hist_y), hist_x, hist_y)
                logger.info('Systematic shift for file is %.4f Da', popt[1])
                df[params_dict['mass_shifts_column']] -= popt[1]
                df['file'] = os.path.split(filename)[-1].split('.')[0]  # correct this
                dfs.append(df)
            break
    logger.info('Starting analysis...')
    data = pd.concat(dfs, axis=0)
    data.index = range(len(data))
    data['is_decoy'] = data[params_dict['proteins_column']].apply(
        lambda s: all(x.startswith(params_dict['decoy_prefix']) for x in s))

    data['bin'] = np.digitize(data[params_dict['mass_shifts_column']], params_dict['bins'])
    return data


def fit_peaks(data, args, params_dict):
    """
    Returns
    """
    logger.info('Performing Gaussian fit...')

    half_window = int(params_dict['window']/2) + 1
    hist = np.histogram(data[params_dict['mass_shifts_column']], bins=params_dict['bins'])
    hist_y = smooth(hist[0], window_size=params_dict['window'], power=5)
    hist_x = 1/2 * (hist[1][:-1] +hist[1][1:])
    loc_max_candidates_ind = argrelextrema(hist_y, np.greater_equal)[0]
    # smoothing and finding local maxima
    min_height = 2 * np.median([x for x in hist[0] if (x>1)])  # minimum bin height expected to be peak approximate noise level as median of all non-negative
    loc_max_candidates_ind = loc_max_candidates_ind[hist_y[loc_max_candidates_ind] >= min_height]

    poptpvar = []
    shape = int(np.sqrt(len(loc_max_candidates_ind))) + 1
    plt.figure(figsize=(shape * 3, shape * 4))
    plt.tight_layout()
    for index, center in enumerate(loc_max_candidates_ind, 1):

        x = hist_x[center - half_window: center + half_window + 1]
        y = hist[0][center - half_window: center + half_window + 1] #take non-smoothed data
#        y_= hist_y[center - half_window: center + half_window + 1]
        popt, perr = gauss_fitting(hist[0][center], x, y)
        plt.subplot(shape, shape, index)
        if popt is None:
            label = 'NO FIT'
        else:

            if x[0] <= popt[1] and popt[1] <= x[-1] and(perr[0]/popt[0] < params_dict['max_deviation_height']) \
            and (perr[2]/popt[2] < params_dict['max_deviation_sigma']):
                label = 'PASSED'
                poptpvar.append(np.concatenate([popt, perr]))
                plt.vlines(popt[1] - 3 * popt[2], 0, hist[0][center], label='3sigma interval' )
                plt.vlines(popt[1] + 3 * popt[2], 0, hist[0][center] )
            else:
                label='FAILED'
        plt.plot(x, y, 'b+:', label=label)
        if label != 'NO FIT':
            plt.scatter(x, gauss(x, *popt),
                        label='Gaussian fit\n $\sigma$ = ' + "{0:.4f}".format(popt[2]) )


        plt.legend()
        plt.title("{0:.3f}".format(hist[1][center]))
        plt.grid(True)
    plt.savefig(os.path.join(args.dir, 'gauss_fit.pdf'))
    plt.close()
    return hist, np.array(poptpvar)


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


def gauss(x,a,  x0, sigma):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return a/sigma/np.sqrt(2*np.pi) * np.exp(-(x - x0) * (x - x0) / (2 * sigma ** 2))


def gauss_fitting(center_y, x, y):
    """
    Fits with Gauss function
    `center_y` - starting point for `a` parameter of gauss
    `x` numpy array of mass shifts
    `y` numpy array of number of psms in this mass shifts

    """
    mean = sum(x*y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    a = center_y * sigma * np.sqrt(2*np.pi)
    try:
        popt, pcov = curve_fit(gauss, x, y, p0=(a, mean, sigma))
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except (RuntimeError, TypeError):
        return None, None


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


def group_specific_filtering(data, final_mass_shifts, params_dict):
    """
    Selects window around found mass shift and filter using TDA. Window is defined as mu +- 3*sigma.
    Returns....
    """
    logger.info('Performing group-wise FDR filtering...')
    out_data = {} # dict corresponds list
    for mass_shift in final_mass_shifts:
        data_slice = data[np.abs(data[params_dict['mass_shifts_column']] - mass_shift[1]) < 3 * mass_shift[2] ].sort_values(by='expect') \
                         .drop_duplicates(subset=params_dict['peptides_column'])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pepxml.filter_df(data_slice, fdr=params_dict['FDR'], correction=params_dict['FDR_correction'], is_decoy='is_decoy')
        if len(df) > 0:
            out_data[np.mean(df[params_dict['mass_shifts_column']])] = df   ###!!!!!!!mean of from gauss fit!!!!
    logger.info('# of filtered mass shifts = %s', len(out_data))
    return  out_data


def plot_figure(ms_label, ms_counts, left, right, params_dict, save_directory):
    """
    'ms_label' mass shift in string format.
    'ms_counts' entries in a mass shift.
    'left

    """

     #figure parameters
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
    mass_shifts_dict_formatted ={mass_format(k): mass_shifts_dict[k] for k in mass_shifts_dict.keys()} # mass_shift_dict with printable labels
    mass_shifts_labels = {mass_format(i): i for i in mass_shifts_dict.keys()}
    zero_mass_shift_label = mass_format(zero_mass_shift)
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
                mslabel: '<a href="#">{}</a>'.format(MASS_FORMAT).format}
            ).bar(subset='# peptides in bin', color=cc[2]).render() #PSMs
    report = report.replace(r'%%%', table_html)
    with open(os.path.join(save_directory, 'report.html'), 'w') as f:
        f.write(report)


def get_parameters(params):
    """
    Reads paramenters from cfg file to one dict.
    Returns dict.
    """
    parameters_dict = defaultdict()
    #data
    parameters_dict['decoy_prefix'] = params.get('data', 'decoy prefix')
    parameters_dict['FDR'] = params.getfloat('data', 'FDR')
    parameters_dict['labels'] = params.get('data', 'labels').strip().split()
    parameters_dict['rule'] = params.get('data', 'cleavage rule')
    # csv input
    parameters_dict['csv_delimiter'] = params.get('csv input', 'delimiter')
    parameters_dict['proteins_delimeter'] = params.get('csv input', 'proteins delimiter')
    parameters_dict['proteins_column'] = params.get('csv input', 'proteins column')
    parameters_dict['peptides_column'] = params.get('csv input', 'peptides column')
    parameters_dict['mass_shifts_column'] = params.get('csv input', 'mass shift column')
    #general
    parameters_dict['bin_width'] = params.getfloat('general', 'width of bin in histogram')
    parameters_dict['so_range'] = tuple(float(x) for x in params.get('general', 'open search range').split(','))
    parameters_dict['area_threshold'] = params.getint('general', 'threshold for bins') # area_thresh
    parameters_dict['walking_window'] = params.getfloat('general', 'shifting window') #shifting_window
    parameters_dict['FDR_correction'] = params.getboolean('general', 'FDR correction') #corrction

    parameters_dict['specific_mass_shift_flag'] = params.getboolean('general', 'use specific mass shift window') #spec_window_flag
    parameters_dict['specific_window'] = [float(x) for x in params.get('general', 'specific mass shift window').split(',')] #spec_window

    parameters_dict['figsize'] = tuple(float(x) for x in params.get('general', 'figure size in inches').split(','))
    #fit
#    parameters_dict['shift_error'] = params.getint('fit', 'shift error')
#    parameters_dict['max_deviation_x'] = params.getfloat('fit', 'standard deviation threshold for center of peak')
    parameters_dict['max_deviation_sigma'] = params.getfloat('fit', 'standard deviation threshold for sigma')
    parameters_dict['max_deviation_height'] = params.getfloat('fit', 'standard deviation threshold for height')
    #localization
    parameters_dict['spectrum_column'] =  params.get('localization', 'spectrum column')
    parameters_dict['charge_column'] = params.get('localization', 'charge column')
    return parameters_dict


def set_additional_params(params_dict):
    """
    Updates dict with new paramenters.
    Returns dict.
    """
    if params_dict['specific_mass_shift_flag']:
        logger.info('Custom bin: %s', params_dict['specific_window'])
        params_dict['so_range'] = params_dict['specific_window'][:]

    elif params_dict['so_range'][1] - params_dict['so_range'][0] > params_dict['walking_window']:
        window = params_dict['walking_window'] / params_dict['bin_width']

    else:
        window = (params_dict['so_range'][1] - params_dict['so_range']) / params_dict['bin_width']
    if int(window) % 2 == 0:
        params_dict['window'] = int(window) + 1
    else:
        params_dict['window'] = int(window)  #should be odd
    params_dict['bins'] = np.arange(params_dict['so_range'][0],
        params_dict['so_range'][1] + params_dict['bin_width'], params_dict['bin_width'])


def systematic_mass_shift_correction(mass_shifts_dict, mass_correction):
   '''
   `mass_shifts_dict` - dict where keys are mass shifts (float) and values DataFrames that correspond to this mass shift.
   '''
   out = {}
   for k, v in mass_shifts_dict.items():
       out[k-mass_correction] = v
   return out


def read_mgf(file_path):
    return mgf.IndexedMGF(file_path)


def read_mzml(file_path): # write this
    return mzml.PreIndexedMzML(file_path)


def read_spectra(args):
    """
    Reads spectra
    -----------
    Returns
    """
    readers = {
        'mgf': read_mgf,
        'mzML': read_mzml,
    }
    out_dict = {}
    for ftype, reader in readers.items():
        filenames = getattr(args, ftype)
        if filenames:
            for filename in filenames:
                name = os.path.split(filename)[-1].split('.')[0] #write it in a proper way
                out_dict[name] = reader(filename)
    return out_dict
