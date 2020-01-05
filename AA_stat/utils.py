import matplotlib
matplotlib.use('Agg')
import pylab as plt
import ast
import os
import logging
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema, savgol_filter
import pandas as pd
import numpy as  np
import warnings
from collections import defaultdict
from pyteomics import parser, pepxml, mgf, mzml

logger = logging.getLogger(__name__)
MASS_FORMAT = '{:.4f}'


def mass_format(mass):
    return MASS_FORMAT.format(mass)


def make_0mc_peptides(pep_list, rule):
    out_set = set()
    for i in pep_list:
        out_set.update(parser.cleave(i, rule))
    return out_set


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
    shifts = params_dict['mass_shifts_column']
    for ftype, reader in readers.items():
        filenames = getattr(args, ftype)
        if filenames:
            for filename in filenames:
                logger.info('Reading %s', filename)
                df = reader(filename, params_dict)
                hist_0 = np.histogram(df.loc[abs(df[shifts] - zero_bin) < window/2, shifts],
                    bins=10000)
                logger.debug('hist_0: %s', hist_0)
                hist_y = hist_0[0]
                hist_x = 0.5 * (hist_0[1][:-1] + hist_0[1][1:])
                popt, perr = gauss_fitting(max(hist_y), hist_x, hist_y)
                logger.info('Systematic shift for file is %.4f Da', popt[1])
                df[shifts] -= popt[1]
                df['file'] = os.path.split(filename)[-1].split('.')[0]  # correct this
                dfs.append(df)
            break
    logger.info('Starting analysis...')
    data = pd.concat(dfs, axis=0)
    data.index = range(len(data))
    data['is_decoy'] = data[params_dict['proteins_column']].apply(
        lambda s: all(x.startswith(params_dict['decoy_prefix']) for x in s))

    data['bin'] = np.digitize(data[shifts], params_dict['bins'])
    return data


def get_unimod_url(mass_shift):
    return ('http://www.unimod.org/modifications_list.php'
        '?a=search&value=1&SearchFor={:.0f}.&'
        'SearchOption=Starts+with+...&SearchField=mono_mass'.format(mass_shift))


def smooth(y, window_size=15, power=5):
    y_smooth = savgol_filter(y, window_size, power)
    return y_smooth


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


def fit_peaks(data, args, params_dict):
    logger.info('Performing Gaussian fit...')

    half_window = int(params_dict['window']/2) + 1
    hist = np.histogram(data[params_dict['mass_shifts_column']], bins=params_dict['bins'])
    hist_y = smooth(hist[0], window_size=params_dict['window'], power=5)
    hist_x = 0.5 * (hist[1][:-1] + hist[1][1:])
    loc_max_candidates_ind = argrelextrema(hist_y, np.greater_equal)[0]
    # smoothing and finding local maxima
    min_height = 2 * np.median([x for x in hist[0] if x > 1])
    # minimum bin height expected to be peak approximate noise level as median of all non-negative
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

            if (x[0] <= popt[1] and popt[1] <= x[-1] and perr[0]/popt[0] < params_dict['max_deviation_height']
                and perr[2]/popt[2] < params_dict['max_deviation_sigma']):
                label = 'PASSED'
                poptpvar.append(np.concatenate([popt, perr]))
                plt.vlines(popt[1] - 3 * popt[2], 0, hist[0][center], label='3sigma interval' )
                plt.vlines(popt[1] + 3 * popt[2], 0, hist[0][center])
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
