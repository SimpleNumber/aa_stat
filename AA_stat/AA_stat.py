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
try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser
from pyteomics import parser, pepxml

cc = ["#FF6600",
      "#FFCC00",
      "#88AA00",
      "#006688",
      "#5FBCD3",
      "#7137C8",
      ]
sb.set_style('white')
colors = sb.color_palette(palette = cc)

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
        d[i] = int(100*d[i] / sum_aa)
    return d
def get_aa_distribution(peptide_list, rule):
    sum_aa = 0
    pep_set = make_0mc_peptides(peptide_list, rule)
    d = defaultdict(int)
    for seq in pep_set:
        for let in seq:
            d[let] += 1
            sum_aa += 1
    for i in d:
        d[i] = d[i] / sum_aa
    return d

def save_table(distr, number_of_PSMs, mass_shifts):
    unimod = pd.Series({i: get_unimod_url(mass_shifts[i]) for i in number_of_PSMs.keys()})
    df = pd.DataFrame({'mass shift': [mass_shifts[k] for k in distr.columns],
                       '# peptides in bin': number_of_PSMs,
                       'Unimod': unimod},
                      index=distr.columns)
    df['# peptides in bin'] = df['# peptides in bin'].astype(np.int64)
    out = pd.concat([df, distr.T], axis=1)
    out.index = range(len(out))
    cols = list(out.columns)
    cols.remove('Unimod')
    cols = ['mass shift', '# peptides in bin'] + cols[2:] + ['Unimod']
    i = ((out.drop(columns=['mass shift', 'Unimod', '# peptides in bin']).max(axis=1) - 1) * out['# peptides in bin']).argsort()
    return out.loc[i.values[::-1], cols]

def read_pepxml(fname, params):
    return pepxml.DataFrame(fname, read_schema=False)

def read_csv(fname, params):
    df = pd.read_csv(fname, sep=params.get('csv input', 'delimiter'))
    proteins_column = params.get('csv input', 'proteins column')
    if df[proteins_column].str[0].all() == '[' and df[proteins_column].str[-1].all() == ']':
        df[proteins_column] = df[proteins_column].apply(ast.literal_eval)
    else:
        df[proteins_column] = df[proteins_column].str.split(
            params.get('csv input', 'proteins delimeter'))
    return df

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

def gauss(x, x0, sigma, a):
    return a * np.exp(- (x - x0) * (x - x0) / (2 * sigma ** 2))

def fitting(center, hist_y, w):
    try:
        popt, pcov = curve_fit(gauss, range(center - w, center + w + 1),
            hist_y[center - w: center + w + 1],
            np.array([center, w, hist_y[center]]))
    except (RuntimeError, TypeError):
        return

    if all(np.diag(pcov) > 0):
        return np.concatenate([popt, np.sqrt(np.diag(pcov))])


def smooth(y, window_size=15, power=5):
    y_smooth = savgol_filter(y, window_size, power)
    return y_smooth

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
    
def eliminate_systemstic_mass_error(data, mass_shifts_column):
    """
    Shifts all masses according non-modified peak.
    """
    data[abs(data[mass_shifts_column]) < 0.1)][mass_shifts_column]
def read_input(args, params, so_range):
    dfs = []
    data = pd.DataFrame()

    logging.info('Reading input files...')
    readers = {
        'pepxml': read_pepxml,
        'csv': read_csv,
    }
    for ftype, reader in readers.items():
        filenames = getattr(args, ftype)
        if filenames:
            for filename in filenames:
                logging.info('Reading %s', filename)
                df = reader(filename, params)
                dfs.append(df)
            break
    logging.info('Starting analysis...')
    data = pd.concat(dfs, axis=0)

    data.index = range(len(data))
    mass_shifts_column = params.get('csv input', 'mass shift column')
    bin_width = params.getfloat('general', 'width of bin in histogram')
    bins = np.arange(so_range[0], so_range[1] + bin_width, bin_width)
    data['bin'] = np.digitize(data[mass_shifts_column], bins)
    proteins_column = params.get('csv input', 'proteins column')
    decoy_pref = params.get('data', 'decoy prefix')
    data['is_decoy'] = data[proteins_column].apply(
        lambda s: all(x.startswith(decoy_pref) for x in s))
    return data

def fit_peaks(data, args, params, w, so_range):
    logging.info('Performing Gaussian fit...')
    save_directory = args.dir
    mass_shifts_column = params.get('csv input', 'mass shift column')
    bin_width = params.getfloat('general', 'width of bin in histogram')

    bins = np.arange(so_range[0], so_range[1] + bin_width, bin_width)
    hist = np.histogram(data[mass_shifts_column], bins=bins)
    hist_y = hist[0]
    indexes = argrelextrema(smooth(hist_y, window_size=w, power=5), np.greater_equal)[0]
    # smoothing and finding local maxima

    min_height = 7  # minimum bin height expected to be peak
    loc = indexes[hist_y[indexes] >= min_height]

    area_thresh = params.getint('general', 'threshold for bins')
    max_deviation_x = params.getfloat('fit', 'standard deviation threshold for center of peak')
    max_deviation_sigma = params.getfloat('fit', 'standard deviation threshold for sigma')
    max_deviation_height = params.getfloat('fit', 'standard deviation threshold for height')
    new_loc = []
    lenhist = len(hist_y)
    for i in loc:
        if i >= w and i+w <= lenhist and hist_y[i-w:i+w+1].sum() > area_thresh:
            new_loc.append(i)
    results = []
    counter = 1
    shape = int(np.sqrt(len(new_loc))) + 1
    plt.figure(figsize=(shape * 3, shape * 3.5))
    plt.tight_layout()
    for center in new_loc:
        x_ = range(center-w, center+w+1)
        y_ = hist_y[center-w:center+w+1]
        cur_fit = fitting(center, hist_y, w)
        plt.subplot(shape, shape, counter)
        if cur_fit is None:
            label = 'NO FIT'
        elif (cur_fit[3] < max_deviation_x) and (cur_fit[4] / cur_fit[1] < max_deviation_sigma
            ) and (cur_fit[5] / cur_fit[2] < max_deviation_height):
            results.append(cur_fit)
            label = 'PASSED'
        else:
            label = 'FAILED'
        plt.bar(x_, y_, label=label)
        if label != 'NO FIT':
            plt.scatter(x_, gauss(x_, *cur_fit[:3]), 
                        label='Gaussian fit')
        plt.legend()
        
        plt.title("{0:.3f}".format(hist[1][center]))
        plt.xticks(range(center - w, center + w + 1, 9),
            ["{0:.3f}".format(x) for x in hist[1][center - w : center + w + 1 : 9]])
        counter += 1
    plt.savefig(os.path.join(save_directory, 'gauss_fit.pdf'))
    plt.close()
    return hist, np.array(results)

def filter_errors(results, params):
    logging.info('Discarding bad peaks...')
    shift_error = params.getint('fit', 'shift error')
    resultsT = results.T
    shift_x = resultsT[0]
    error = resultsT[3]
    kick = []

    for i, x1 in enumerate(shift_x):
        for j, x2 in enumerate(shift_x):
            if abs(x1 - x2) < shift_error:
                if error[i] > error[j]:
                    kick.append(i)
                elif error[i] < error[j]:
                    kick.append(j)

    kick = set(kick)
    final = []
    for i, res in enumerate(results):
        if i not in kick:
            final.append(int(res[0]))
    return np.array(final)

def filter_bins(data, final, hist, params, w):
    logging.info('Performing group-wise FDR filtering...')
    mass_shifts_column = params.get('csv input', 'mass shift column')
    bin_width = params.getfloat('general', 'width of bin in histogram')
    spec_window_flag = params.getboolean('general', 'use specific mass shift window')
    correction = params.getboolean('general', 'FDR correction')
    peptides_column = params.get('csv input', 'peptides column')
    fdr = params.getfloat('data', 'FDR')
    out_data = {} # dict corresponds list 
    for dbin in final:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_slice = data.loc[(data.bin >= dbin - w) & (data.bin <= dbin + w), :].sort_values(by='expect').drop_duplicates(subset=peptides_column)
            df = pepxml.filter_df(data_slice, fdr=fdr, correction=correction, is_decoy='is_decoy')
            #out_data[dbin] = df
            if len(df) > 0:
                out_data[dbin] = df
    mass_shifts = {}
    for m in out_data:
        mass_shifts[m] = hist[1][m]
    if spec_window_flag:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out_data['zero_bin'] = pepxml.filter_df(
                data[(data[mass_shifts_column] >= -bin_width * w) & (
                      data[mass_shifts_column] <= bin_width * w)].sort_values(by='expect').drop_duplicates(subset=peptides_column),
                correction=correction, fdr=fdr, is_decoy='is_decoy')
            zero_bin = 'zero_bin'
    else:
        zero_bin = min(mass_shifts, key=lambda x: abs(mass_shifts[x]))
    return mass_shifts, out_data, zero_bin

def plot_results(data, out_data, zero_bin, mass_shifts, args, params, so_range):
    logging.info('Plotting distributions...')
    figsize = tuple(float(x) for x in params.get('general', 'figure size in inches').split(','))
    peptides_column = params.get('csv input', 'peptides column')
    mass_shifts_column = params.get('csv input', 'mass shift column')
    labels = params.get('data', 'labels').strip().split()
    rule = params.get('data', 'cleavage rule')
    expasy_rule = parser.expasy_rules.get(rule, rule)
    save_directory = args.dir
    b = 0.2 # shift in bar plots
    width = 0.4 # for bar plots

    number_of_PSMs = pd.Series(index=list(mass_shifts), dtype=int)
    reference = pd.Series(get_aa_distribution(out_data[zero_bin][peptides_column], expasy_rule))
    #reference.fillna( None, inplace=True)
    err_ref_df = pd.DataFrame(index=labels)
    for i in range(50):
        err_ref_df[i] = pd.Series(get_aa_distribution(
            np.random.choice(np.array(out_data[zero_bin][peptides_column]),
            size=(len(out_data[zero_bin]) // 2), replace=False),
            expasy_rule)) / reference
    if len(mass_shifts) < 1:
        fdr = params.getfloat('data', 'FDR')
        logging.warning('No peptides in specified bins after filtering')
        dat_slice = pepxml.filter_df(
            data[(data[mass_shifts_column] >= so_range[0]) & (
                  data[mass_shifts_column] <= so_range[1])].sort_values(by='expect').drop_duplicates(subset=peptides_column),
            dr=fdr, is_decoy='is_decoy')
        len_filtered = len(dat_slice)
        logging.warning('In window %s - %s only %s filtered indentifications'
            ' (without fdr correction), histogram for this window saved to %s',
            *so_range, len_filtered, os.path.abspath(args.dir))
        dloc = data.loc[(data[mass_shifts_column] >= so_range[0]) & (
            data[mass_shifts_column] <= so_range[1]), mass_shifts_column]
        plt.hist(dloc, bins=30)
        plt.title('Not passed {}'.format(len(dloc)))
        plt.savefig(os.path.join(args.dir, 'not_passed.png'))
    logging.info('Mass shifts:')
    distributions = pd.DataFrame(index=labels)
    p_values = pd.DataFrame(index=labels)

    for binn, mass_diff in mass_shifts.items():
        distr = pd.Series(get_aa_distribution(out_data[binn][peptides_column], expasy_rule))
        peptide_stat = pd.Series(get_peptide_statistics(out_data[binn][peptides_column], expasy_rule))
        formated_key = "{0:.3f}".format(mass_diff)
        number_of_PSMs[binn] = len(out_data[binn])
        distr.fillna(0, inplace=True)
        distributions[binn] = distr / reference
        bar_plot, bar1 = plt.subplots()
        bar_plot.set_size_inches(figsize)# = plt.figure(figsize=figsize)
        p_vals, errors = calculate_error_and_p_vals(
            out_data[binn][peptides_column], err_ref_df, reference, expasy_rule, labels)
        errors.fillna(0, inplace=True)
        p_values[binn] = p_vals
        distributions.fillna(0, inplace=True)
        #bar1 = bar_plot.add_subplot(111)
        bar1.bar(np.arange(b, 2*len(labels), 2), distributions.loc[labels, binn],
            yerr=errors.loc[labels], width=width, color=colors[2],linewidth=0,
            label= formated_key + ' Da mass shift,'  + '\n' + str(len(out_data[binn])) +' peptides')
        bar1.set_ylabel('Relative AA abundance', color=colors[2])
        labels_df = pd.DataFrame(index=labels)
        labels_df['label'] = labels_df.index
        labels_df['pep_stat'] =pd.Series(peptide_stat)
        labels_df.fillna(0, inplace=True)
        labels_df['out'] = labels_df['label'] #+ pd.Series(['\n']*len(labels), index=labels) +labels_df['pep_stat']
        #print(labels_df.loc[labels,'out'])
        bar1.set_xticks(np.arange(2*b, 2*len(labels)+2*b, 2))#
        bar1.set_xticklabels(labels_df.loc[labels,'out'])
        bar1.hlines(1, -1, 2* len(labels), linestyles='dashed', color=colors[3])
        bar2 = bar1.twinx()
        bar2.bar(np.arange(4 * b, 2 * len(labels) + 4 * b, 2),labels_df['pep_stat'],width=width, linewidth=0, color=colors[0])
        bar2.set_ylim(0,125)
        bar2.set_yticks(np.arange(0,120, 20))
        bar2.set_ylabel('Peptides with AA, %', color=colors[0])
        
        bar1.spines['left'].set_color(colors[2])
        bar2.spines['left'].set_color(colors[2])
        
        bar1.spines['right'].set_color(colors[0])
        bar2.spines['right'].set_color(colors[0])
        bar1.tick_params('y', colors=colors[2])
        bar2.tick_params('y', colors=colors[0])
        bar2.annotate(formated_key + ' Da mass shift,'  + '\n' + str(len(out_data[binn])) +' peptides',
                      xy=(29,107), bbox=dict(boxstyle='round',fc='w', edgecolor='dimgrey'))
        #plt.title('Mass shift = ' + formated_key + '; Peptides in bin = ' + str(len(out_data[binn]))) #PSMs
        #bar1.legend()
        bar1.set_xlim(-3*b, 2*len(labels)-2 +9*b)
        bar1.set_ylim(0,distributions.loc[labels, binn].max()*1.3)
        bar_plot.savefig(os.path.join(save_directory, formated_key + '.png'), dpi=500)
        bar_plot.savefig(os.path.join(save_directory, formated_key + '.svg'))
        plt.close()
        logging.info('%s Da', formated_key)
    pout = p_values.T
    pout.insert(0, 'mass shift', [mass_shifts[i] for i in pout.index])
    pout.to_csv(os.path.join(save_directory, 'p_values.csv'), index=False)
    return distributions, number_of_PSMs

def render_html_report(table, params, save_directory):
    labels = params.get('data', 'labels').strip().split()
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
                mslabel: '<a href="#">{:.3f}</a>'.format}
            ).bar(subset='# peptides in bin', color=cc[2]).render() #PSMs
    report = report.replace(r'%%%', table_html)
    with open(os.path.join(save_directory, 'report.html'), 'w') as f:
        f.write(report)

def main():
    pars = argparse.ArgumentParser()
    pars.add_argument('--params', help='CFG file with parameters.'
        'An example can be found at https://bitbucket.org/J_Bale/aa_stat/src/tip/example.cfg',
        required=True)
    pars.add_argument('--dir', help='Directory to store the results. '
        'Default value is current directory.', default='.')
    pars.add_argument('-v', '--verbosity', action='count', default=1,
                      help='Increase output verbosity')
   
    input_spectra = pars.add_mutually_exclusive_group()
    input_spectra.add_argument('--mgf',  nargs='+', help='MGF files to localize modifications')
    input_spectra.add_argument('--mzml',  nargs='+', help='mzML files to localize modifications')
    
    input_file = pars.add_mutually_exclusive_group(required=True)
    input_file.add_argument('--pepxml', nargs='+', help='List of input files in pepXML format')
    input_file.add_argument('--csv', nargs='+', help='List of input files in CSV format')
    levels = [logging.ERROR, logging.INFO, logging.DEBUG]
    args = pars.parse_args()
    save_directory = args.dir
    level = 2 if args.verbosity > 2 else args.verbosity
    logging.basicConfig(format='%(levelname)5s: %(asctime)s %(message)s',
                        datefmt='[%H:%M:%S]', level=levels[level])
    logging.info("Starting...")


    params = ConfigParser(delimiters=('=', ':'),
                          comment_prefixes=('#'),
                          inline_comment_prefixes=('#'))
    params.read(args.params)

    spec_window_flag = params.getboolean('general', 'use specific mass shift window')
    spec_window = [float(x) for x in params.get('general', 'specific mass shift window').split(',')]
    so_range = tuple(float(x) for x in params.get('general', 'open search range').split(','))
    if spec_window_flag:
        logging.info('Custom bin %s', spec_window)
        so_range = spec_window[:]
    bin_width = params.getfloat('general', 'width of bin in histogram')

    shift_window = params.getfloat('general', 'shifting window')

    if so_range[1] - so_range[0] > shift_window:
        window = shift_window / bin_width
    else:
        window = (so_range[1] - so_range[0]) / bin_width
    w = int(window / 2)

    data = read_input(args, params, so_range)
    hist, results = fit_peaks(data, args, params, w, so_range)
    final = filter_errors(results, params)
    mass_shifts, out_data, zero_bin = filter_bins(data, final, hist, params, w)
#    print(mass_shifts)
    distributions, number_of_PSMs = plot_results(
        data, out_data, zero_bin, mass_shifts, args, params, so_range)
    if len(args.mgf) > 0:
        logging.info('Localization using MS/MS spectra...')
        suffix = args.mgf[0].split('.')[-1]
        spectra_dir =  '/'.join(args.mgf[0].split('/')[:-1])
    elif len(args.mzml) > 0:
        logging.info('Localization using MS/MS spectra...')
        suffix = args.mgf[0].split('.')[-1]
        spectra_dir =  '/'.join(args.mzml[0].split('/')[:-1])
    else:
        logging.info('No spectra files. MSMS spectrum localization is not performed.')
        
    table = save_table(distributions, number_of_PSMs, mass_shifts)
    table.to_csv(os.path.join(save_directory, 'aa_statistics_table.csv'), index=False)

    logging.info('Summarizing hist prepared')
    summarizing_hist(table, save_directory)

    render_html_report(table, params, save_directory)
    logging.info('Results saved to %s', os.path.abspath(args.dir))

if __name__ == '__main__':
    main()
