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

def smooth(y, window_size=15, power=5):
    y_smooth = savgol_filter(y, window_size, power)
    return y_smooth

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

def read_pepxml(fname, params_dict):
    return pepxml.DataFrame(fname, read_schema=False)

def read_csv(fname, params_dict):
    df = pd.read_csv(fname, sep=params_dict['csv_delimiter'])
    if df[params_dict['proteins_column']].str[0].all() == '[' and df[params_dict['proteins_column']].str[-1].all() == ']':
        df[params_dict['proteins_column']] = df[params_dict['proteins_column']].apply(ast.literal_eval)
    else:
        df[params_dict['proteins_column']] = df[params_dict['proteins_column']].str.split(
            params_dict['proteins_delimeter'])
    return df

def read_input(args, params_dict):
    """
    Reads open search output, assemble all files in one DataFrame
    Returns DF
    """
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
                df = reader(filename, params_dict)
                dfs.append(df)
            break
    logging.info('Starting analysis...')
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
    logging.info('Performing Gaussian fit...')

    half_window = int(params_dict['window']/2) + 1
    hist = np.histogram(data[params_dict['mass_shifts_column']], bins=params_dict['bins'])
    hist_y = smooth(hist[0], window_size=params_dict['window'], power=5)
    hist_x = 1/2 * (hist[1][:-1] +hist[1][1:])
    loc_max_candidates_ind = argrelextrema(hist_y, np.greater_equal)[0]
    print(len(loc_max_candidates_ind))
#    logging.DEBUG('Local max number %i', str(len(loc_max_candidates_ind)))
    # smoothing and finding local maxima
    min_height = 2 * np.median([x for x in hist[0] if (x>1)])  # minimum bin height expected to be peak approximate noise level as median of all non-negative
    loc_max_candidates_ind = loc_max_candidates_ind[hist_y[loc_max_candidates_ind] >= min_height]
#    logging.DEBUG('Local max %s after noise filtering', str(len(loc_max_candidates_ind)))
    print(len(loc_max_candidates_ind))

    poptpvar = []
    shape = int(np.sqrt(len(loc_max_candidates_ind))) + 1
    plt.figure(figsize=(shape * 3, shape * 4))
    plt.tight_layout()
    for index, center in enumerate(loc_max_candidates_ind, 1):
        
        x = hist_x[center - half_window: center + half_window + 1]
        y = hist[0][center - half_window: center + half_window + 1] #take non-smoothed data
        popt, pcov = gauss_fitting(hist[0][center], x, y)
        plt.subplot(shape, shape, index)
        if popt is None:
            label = 'NO FIT'
        else:
            if x[0]<= popt[1] and popt[1]<=x[-1]:
                label = 'PASSED'
                poptpvar.append(np.concatenate([popt, np.diag(pcov)]))
            else:
                label='FAILED'
        plt.plot(x, y, 'b+:', label=label)
        if label != 'NO FIT':
            plt.scatter(x, gauss(x, *popt), 
                        label='Gaussian fit')
        plt.legend()
        
        plt.title("{0:.3f}".format(hist[1][center]))
#        print(hist[1][center])
        plt.xticks(x[::4], ["{0:.3f}".format(i) for i in x[::4] ], rotation=45)
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
    return a * np.exp(- (x - x0) * (x - x0) / (2 * sigma ** 2))

def gauss_fitting(center_y, x, y):
    """
    Fits with Gauss function
    `center_y` - starting point for `a` parameter of gauss
    `x` numpy array of mass shifts
    `y` numpy array of number of psms in this mass shifts
    
    """
    n = len(x)
    mean = sum(x *y) / sum(y)                  
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / n)
    try:
        popt, pcov = curve_fit(gauss, x, y, p0=(center_y, mean, sigma))
        return popt, pcov
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
    """
    Shifts all masses according non-modified peak.
    """
    return mass_shifts[np.argmin(abs(mass_shifts))]

def filter_mass_shifts(results):

    """
    Filter mass_shifts that close to each other.
    
    Return poptpcov matrix.
    """
    logging.info('Discarding bad peaks...')
    out = []
    for ind, mass_shift in enumerate(results[:-1]):
        if np.linalg.norm(results[ind] - results[ind + 1]) > 5: 
#            print(mass_shift)
            out.append(mass_shift)
    return out

def group_specific_filtering(data, final_mass_shifts, params_dict):
    """
    Selects window around found mass shift and filter using TDA. Window is defined as mu +- 3*sigma.
    Returns....
    """
    logging.info('Performing group-wise FDR filtering...')
    out_data = {} # dict corresponds list 
    for mass_shift in final_mass_shifts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
#        print(data[params_dict['mass_shifts_column']] - mass_shift[1] < 3 * mass_shift[4] )
#        print(mass_shift)
        data_slice = data[np.abs(data[params_dict['mass_shifts_column']] - mass_shift[1]) < 3 * mass_shift[2] ].sort_values(by='expect').drop_duplicates(subset=params_dict['peptides_column'])
        df = pepxml.filter_df(data_slice, fdr=params_dict['FDR'], correction=params_dict['FDR_correction'], is_decoy='is_decoy')
        if len(df) > 0:
            out_data[mass_shift[1]] = df
    return  out_data

def plot_results(mass_shifts_dict, zero_mass_shift, params_dict ,args):
    logging.info('Plotting distributions...')
    labels = params_dict['labels']
    rule = params_dict['rule']
    expasy_rule = parser.expasy_rules.get(rule, rule)
    save_directory = args.dir
    b = 0.2 # shift in bar plots
    width = 0.4 # for bar plots

    number_of_PSMs = pd.Series(index=list(mass_shifts), dtype=int)
    reference = pd.Series(get_aa_distribution(out_data[zero_bin][params_dict['peptides_column']], expasy_rule))
    #reference.fillna( None, inplace=True)
    err_ref_df = pd.DataFrame(index=labels)
    for i in range(50):
        err_ref_df[i] = pd.Series(get_aa_distribution(
            np.random.choice(np.array(out_data[zero_bin][params_dict['peptides_column']]),
            size=(len(out_data[zero_bin]) // 2), replace=False),
            expasy_rule)) / reference
    if len(mass_shifts) < 1:
        logging.warning('No peptides in specified bins after filtering')
        dat_slice = pepxml.filter_df(
            data[(data[params_dict['mass_shifts_column']] >= so_range[0]) & (
                  data[params_dict['mass_shifts_column']] <= so_range[1])].sort_values(by='expect').drop_duplicates(subset=params_dict['peptides_column']),
            dr=params_dict[''], is_decoy='is_decoy')
        len_filtered = len(dat_slice)
        logging.warning('In window %s - %s only %s filtered indentifications'
            ' (without fdr correction), histogram for this window saved to %s',
            *so_range, len_filtered, os.path.abspath(args.dir))
        dloc = data.loc[(data[params_dict['mass_shifts_column']] >= so_range[0]) & (
            data[params_dict['mass_shifts_column']] <= so_range[1]), params_dict['mass_shifts_column']]
        plt.hist(dloc, bins=30)
        plt.title('Not passed {}'.format(len(dloc)))
        plt.savefig(os.path.join(args.dir, 'not_passed.png'))
    logging.info('Mass shifts:')
    distributions = pd.DataFrame(index=labels)
    p_values = pd.DataFrame(index=labels)

    for binn, mass_diff in mass_shifts.items():
        distr = pd.Series(get_aa_distribution(out_data[binn][params_dict['peptides_column']], expasy_rule))
        peptide_stat = pd.Series(get_peptide_statistics(out_data[binn][params_dict['peptides_column']], expasy_rule))
        formated_key = "{0:.3f}".format(mass_diff)
        number_of_PSMs[binn] = len(out_data[binn])
        distr.fillna(0, inplace=True)
        distributions[binn] = distr / reference
        bar_plot, bar1 = plt.subplots()
        bar_plot.set_size_inches(params_dict['figsize'])# = plt.figure(figsize=figsize)
        p_vals, errors = calculate_error_and_p_vals(
            out_data[binn][params_dict['peptides_column']], err_ref_df, reference, expasy_rule, labels)
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

def render_html_report(table, params_dict, save_directory):
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
                mslabel: '<a href="#">{:.3f}</a>'.format}
            ).bar(subset='# peptides in bin', color=cc[2]).render() #PSMs
    report = report.replace(r'%%%', table_html)
    with open(os.path.join(save_directory, 'report.html'), 'w') as f:
        f.write(report)
def get_parameters(params):
    parameters_dict = {}
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
    parameters_dict['shift_error'] = params.getint('fit', 'shift error')
    parameters_dict['max_deviation_x'] = params.getfloat('fit', 'standard deviation threshold for center of peak')
    parameters_dict['max_deviation_sigma'] = params.getfloat('fit', 'standard deviation threshold for sigma')
    parameters_dict['max_deviation_height'] = params.getfloat('fit', 'standard deviation threshold for height')
    return parameters_dict

def get_additional_params(params_dict):
    if params_dict['specific_mass_shift_flag']:
        logging.info('Custom bin %s', params_dict['specific_window'])
        params_dict[ 'so_range'] = params_dict['specific_window'][:]

    elif params_dict[ 'so_range'][1] - params_dict[ 'so_range'][0] > params_dict['walking_window']:
        window = params_dict['walking_window'] /  params_dict['bin_width']
       
    else:
        window = ( params_dict[ 'so_range'][1] -  params_dict[ 'so_range']) / params_dict['bin_width']
    if int(window) % 2 == 0:
        params_dict['window']  = int(window) + 1
    else:
        params_dict['window']  = int(window)  #should be odd
#    print(params_dict['window'])
    params_dict['bins'] = np.arange(params_dict['so_range'][0], params_dict['so_range'][1] + params_dict['bin_width'], params_dict['bin_width'])
    return params_dict

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
    level = 2 if args.verbosity >= 2 else args.verbosity
    logging.basicConfig(format='%(levelname)5s: %(asctime)s %(message)s',
                        datefmt='[%H:%M:%S]', level=levels[level])
    logging.info("Starting...")


    params = ConfigParser(delimiters=('=', ':'),
                          comment_prefixes=('#'),
                          inline_comment_prefixes=('#'))
    params.read(args.params)
    params_dict = get_parameters(params)
    params_dict = get_additional_params(params_dict) #params_dict 'window'
    

    
    data = read_input(args, params_dict)
    
    hist, popt_pvar = fit_peaks(data, args, params_dict)
#    print(popt_pvar)
    print('=======================================')
    final_mass_shifts = filter_mass_shifts(popt_pvar)
#    print(final_mass_shifts)
    mass_shift_data_dict = group_specific_filtering(data, final_mass_shifts, params_dict)
    zero_mass_shift = get_zero_mass_shift(mass_shift_data_dict.keys())
    logging.info("Systematic mass shift equals to %s", "{0:.4f}".format(zero_mass_shift) )
    distributions, number_of_PSMs = plot_results(
        data, out_data, zero_bin, mass_shifts, args, params_dict, params_dict[ 'so_range'] )
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

    render_html_report(table, params_dict, save_directory)
    logging.info('Results saved to %s', os.path.abspath(args.dir))

if __name__ == '__main__':
    main()
