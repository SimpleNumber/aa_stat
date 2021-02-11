import matplotlib
matplotlib.use('Agg')

import pylab as plt
import ast
import os
import sys
import operator
import logging
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema, savgol_filter
import pandas as pd
import numpy as np
from sklearn import cluster
import warnings
from collections import defaultdict, Counter
import re
import seaborn as sb
try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser
import math
import multiprocessing as mp
import jinja2
import pkg_resources
from datetime import datetime
import itertools as it
import json
from pyteomics import parser, pepxml, mgf, mzml, mass

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.category').disabled = True

MASS_FORMAT = '{:+.4f}'
COMBINATION_TOLERANCE = 1e-4
UNIMOD = mass.Unimod('file://' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'unimod.xml'))
AA_STAT_PARAMS_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'example.cfg')
INTERNAL = 5
DIFF_C13 = mass.calculate_mass(formula='C[13]') - mass.calculate_mass(formula='C')

cc = ["#FF6600",
      "#FFCC00",
      "#88AA00",
      "#006688",
      "#5FBCD3",
      "#7137C8",
      ]
sb.set_style('white')
colors = sb.color_palette(palette=cc)


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


def _gauss_fit_slice(to_fit, unit, filename, suffix, params_dict):
    logger.debug('Fitting zero-shift peptides...')
    plt.figure()
    hist_0 = np.histogram(to_fit, bins=int(params_dict['zero_window'] / params_dict['bin_width']))
    hist_y = hist_0[0]
    hist_x = 0.5 * (hist_0[1][:-1] + hist_0[1][1:])
    plt.plot(hist_x, hist_y, 'b+')
    popt, perr = gauss_fitting(max(hist_y), hist_x, hist_y)
    plt.scatter(hist_x, gauss(hist_x, *popt), label='Gaussian fit')
    plt.xlabel('massdiff, ' + unit)
    plt.savefig(os.path.join(
        params_dict['output directory'], os.path.splitext(os.path.basename(filename))[0] + suffix + '_zerohist.png'))
    plt.close()
    logger.info('Systematic shift is %.4f %s for file %s [ %s ]', popt[1], unit, filename, suffix)
    return popt


def clusters(df, to_fit, unit, filename, params_dict):
    if to_fit.shape[0] < 500:
        logger.warning('Not enough data for cluster analysis. Need at least 500 peptides near zero, found %d.', to_fit.shape[0])
        return None
    X = np.empty((to_fit.shape[0], 2))
    X[:, 0] = to_fit
    X[:, 1] = df.loc[to_fit.index, params_dict['rt_column']]
    logger.debug('Clustering a %s array.', X.shape)
    logger.debug('Initial dimensions: %s to %s', X.min(axis=0), X.max(axis=0))
    logger.debug('Converting to square...')
    span_0 = X[:, 0].max() - X[:, 0].min()
    span_1 = X[:, 1].max() - X[:, 1].min()
    ratio = span_1 / span_0
    X[:, 0] *= ratio
    logger.debug('Transformed dimensions: %s to %s', X.min(axis=0), X.max(axis=0))

    eps = span_1 * params_dict['zero_window'] * params_dict['eps_adjust']
    logger.debug('Using eps=%f', eps)
    clustering = cluster.DBSCAN(eps=eps, min_samples=params_dict['min_samples']).fit(X)
    plt.figure()
    sc = plt.scatter(to_fit, X[:, 1], c=clustering.labels_)
    plt.legend(*sc.legend_elements(), title='Clusters')
    plt.xlabel(unit)
    plt.ylabel(params_dict['rt_column'])
    plt.savefig(os.path.join(
        params_dict['output directory'], os.path.splitext(os.path.basename(filename))[0] + '_clusters.png'))
    plt.close()
    plt.figure()
    for c in np.unique(clustering.labels_):
        plt.hist(X[clustering.labels_ == c, 1], label=c, alpha=0.5)
    plt.xlabel(params_dict['rt_column'])
    plt.legend()
    plt.savefig(os.path.join(
        params_dict['output directory'], os.path.splitext(os.path.basename(filename))[0] + '_cluster_hist.png'))
    plt.close()
    return clustering


def cluster_time_span(clustering, label, df, to_fit, params_dict):
    times = df.loc[to_fit.index].loc[clustering.labels_ == label, params_dict['rt_column']]
    return times.min(), times.max()

def span_percentage(span, df, to_fit, params_dict):
    start, end = span
    all_rt = df[params_dict['rt_column']]
    return (end - start) / (all_rt.max() - all_rt.min())


def cluster_time_percentage(clustering, label, df, to_fit, params_dict):
    span = cluster_time_span(clustering, label, df, to_fit, params_dict)
    return span_percentage(span, df, to_fit, params_dict)


def span_union(span_1, span_2):
    if span_1 is None:
        return span_2
    if span_2 is None:
        return span_1
    return min(span_1[0], span_2[0]), max(span_1[1], span_2[1])


def filter_clusters(clustering, df, to_fit, params_dict):
    nclusters = clustering.labels_.max() + 1
    logger.debug('Found %d clusters, %d labels assigned.', nclusters, clustering.labels_.size)
    if not nclusters:
        return []
    sizes = {}
    for i in np.unique(clustering.labels_):
        percentage = cluster_time_percentage(clustering, i, df, to_fit, params_dict)
        sizes[i] = percentage
        logger.debug('Cluster %d spans %.1f%% of the run.', i, percentage * 100)
    cum_pct_thresh = 0.9
    cum_pct = 0.0
    covered = None  # start with empty covered span
    out = []
    for i in sorted(sizes, key=sizes.get, reverse=True):
        if i == -1:
            continue
        npep = (clustering.labels_ == i).sum()
        if npep < params_dict['min_peptides_for_mass_calibration']:
            logger.debug('Cluster %s is too small for calibration (%d), discarding.', i, npep)
            continue
        union = span_union(covered, cluster_time_span(clustering, i, df, to_fit, params_dict))
        if union == covered:
            logger.debug('Cluster %s is already fully covered. Ignoring.', i)
            continue
        covered = union
        cum_pct = span_percentage(covered, df, to_fit, params_dict)
        out.append(i)
        logger.debug('Clusters %s cover %.1f%% of the run.', out, cum_pct * 100)
        if cum_pct > cum_pct_thresh:
            logger.debug('Threshold achieved at %d clusters.', len(out))
            break
    return out


def get_fittable_series(df, params_dict, mask=None):
    window = params_dict['zero_window']
    shifts = params_dict['mass_shifts_column']
    loc = df[shifts].abs() < window
    # logger.debug('loc size for zero shift: %s', loc.size)
    if params_dict['calibration'] == 'gauss':
        to_fit = df.loc[loc, shifts]
        unit = 'Da'
    elif params_dict['calibration'] == 'gauss_relative':
        to_fit = df.loc[loc, shifts] * 1e6 / df.loc[loc, params_dict['calculated_mass_column']]
        unit = 'ppm'
    elif params_dict['calibration'] == 'gauss_frequency':
        freq_measured = 1e6 / np.sqrt(df[params_dict['measured_mass_column']])
        freq_calculated = 1e6 / np.sqrt(df[params_dict['calculated_mass_column']])
        to_fit = (freq_measured - freq_calculated).loc[loc]
        unit = 'freq. units'
    if mask is not None:
        to_fit = to_fit.loc[mask]
    logger.debug('Returning a %s fittable series for a %s dataframe with a %s mask.', to_fit.shape, df.shape,
        mask.shape if mask is not None else None)
    return to_fit, unit


def get_cluster_masks(filtered_clusters, clustering, df, to_fit, params_dict):
    all_rt = df[params_dict['rt_column']]
    time_spans = {i: cluster_time_span(clustering, i, df, to_fit, params_dict) for i in filtered_clusters}
    sorted_clusters = sorted(filtered_clusters, key=time_spans.get)  # sorts by span start
    i = 0
    prev = all_rt.min()
    masks = {}
    while i < len(sorted_clusters):
        cur_end = time_spans[sorted_clusters[i]][1]
        if i == len(sorted_clusters) - 1:
            next_point = all_rt.max() + 1
        else:
            next_start = time_spans[sorted_clusters[i + 1]][0]
            next_point = (cur_end + next_start) / 2
        logger.debug('Time span %.1f - %.1f assigned to cluster %s', prev, next_point, sorted_clusters[i])
        masks[sorted_clusters[i]] = (all_rt >= prev) & (all_rt < next_point)
        i += 1
        prev = next_point

    assigned_masks = [masks[c] for c in filtered_clusters]
    return assigned_masks


def preprocess_df(df, filename, params_dict):
    '''
    Preprocesses DataFrame.

    Parameters
    ----------
    df: DataFrame
        Open search result df.
    filename: str
        Path of initial (pepxml or csv) file
    params_dict: dict
        Dict with all input parameters
    Returns
    -------
    DataFrame
    '''
    logger.debug('Preprocessing %s', filename)
    window = params_dict['zero_window']
    zero_bin = 0
    shifts = params_dict['mass_shifts_column']

    df['is_decoy'] = df[params_dict['proteins_column']].apply(
        lambda s: all(x.startswith(params_dict['decoy_prefix']) for x in s))
    ms, filtered = fdr_filter_mass_shift([None, zero_bin, window], df, params_dict)
    n = filtered.shape[0]
    logger.debug('%d filtered peptides near zero.', n)
    if params_dict['calibration'] == 'off':
        logger.info('Mass calibration is disabled. Skipping.')
    elif params_dict['calibration'] != 'simple':
        if n < params_dict['min_peptides_for_mass_calibration']:
            logger.warning('Skipping mass calibration: not enough peptides near zero mass shift.')
        else:
            to_fit, unit = get_fittable_series(filtered, params_dict)
            # save copies of mass shift column, for use in boolean indexing
            shift_copy = df[shifts].copy()
            old_shifts = filtered[shifts].copy()
            if params_dict['clustering']:
                clustering = clusters(filtered, to_fit, unit, filename, params_dict)
                filtered_clusters = filter_clusters(clustering, filtered, to_fit, params_dict)
                if len(filtered_clusters) == 1:
                    logger.info('One large cluster found in %s. Calibrating masses in the whole file.', filename)
                    filtered_clusters = None
                else:
                    logger.info('Splitting %s into %d pieces.', filename, len(filtered_clusters))
                    plt.figure()
                    for i in filtered_clusters:
                        plt.hist(filtered.loc[to_fit.index].loc[clustering.labels_ == i, shifts], label=i, alpha=0.2, bins=25, density=True)
                    plt.xlabel(shifts)
                    plt.legend()
                    plt.savefig(os.path.join(
                        params_dict['output directory'],
                        os.path.splitext(os.path.basename(filename))[0] + '_massdiff_hist.png'))
                    plt.close()
            else:
                filtered_clusters = None

            if not filtered_clusters:
                slices = [None]
                suffixes = ['']
                assigned_masks = [slice(None)]
                filtered_clusters = ['<all>']
            else:
                slices, suffixes = [], []
                for i in filtered_clusters:
                    slices.append(clustering.labels_ == i)
                    suffixes.append('_cluster_{}'.format(i))
                assigned_masks = get_cluster_masks(filtered_clusters, clustering, df, to_fit, params_dict)
            for c, slice_, suffix, mask in zip(filtered_clusters, slices, suffixes, assigned_masks):
                # logger.debug('Slice size for cluster %s is: %s', c, slice_.size if slice_ is not None else None)
                to_fit, unit = get_fittable_series(filtered, params_dict, slice_)
                popt = _gauss_fit_slice(to_fit, unit, filename, suffix, params_dict)

                if unit == 'Da':
                    shift_copy.loc[mask] -= popt[1]
                elif unit == 'ppm':
                    shift_copy.loc[mask] -= popt[1] * df[params_dict['calculated_mass_column']] / 1e6
                else:
                    np.testing.assert_allclose(
                        df[shifts],
                        df[params_dict['measured_mass_column']] - df[params_dict['calculated_mass_column']],
                        atol=1e-4)
                    freq_measured = 1e6 / np.sqrt(df.loc[mask, params_dict['measured_mass_column']]) - popt[1]
                    mass_corrected = (1e6 / freq_measured) ** 2
                    correction = mass_corrected - df.loc[mask, params_dict['measured_mass_column']]
                    logger.debug('Average systematic mass shift for cluster %s: %f', c, -correction.mean())
                    shift_copy.loc[mask] += correction

            # corrected mass shifts are written back here
            df[shifts] = shift_copy
            filtered[shifts] = df.loc[filtered.index, shifts]

            plt.figure()
            floc = filtered.loc[old_shifts.abs() < params_dict['zero_window']]
            sc = plt.scatter(floc[shifts], floc[params_dict['rt_column']],
                c=clustering.labels_ if params_dict['clustering'] else None)
            if params_dict['clustering']:
                plt.legend(*sc.legend_elements(), title='Clusters')
            plt.xlabel(shifts)
            plt.ylabel(params_dict['rt_column'])
            plt.savefig(os.path.join(
                params_dict['output directory'], os.path.splitext(os.path.basename(filename))[0] + '_massdiff_corrected.png'))
            plt.close()
            if filtered_clusters != ['<all>']:
                plt.figure()
                for i in filtered_clusters:
                    plt.hist(floc.loc[clustering.labels_ == i, shifts], label=i, alpha=0.2, bins=25, density=True)
                plt.xlabel(shifts)
                plt.legend()
                plt.savefig(os.path.join(
                    params_dict['output directory'],
                    os.path.splitext(os.path.basename(filename))[0] + '_massdiff_corrected_hist.png'))
                plt.close()
    df['file'] = os.path.splitext(os.path.basename(filename))[0]
    df['check_composition'] = df[params_dict['peptides_column']].apply(lambda x: check_composition(x, params_dict['labels']))
    return df.loc[df['check_composition']]


def fdr_filter_mass_shift(mass_shift, data, params_dict):
    shifts = params_dict['mass_shifts_column']
    ms_shift = data.loc[np.abs(data[shifts] - mass_shift[1]) < mass_shift[2], shifts].mean()

    mask = np.abs(data[shifts] - mass_shift[1]) < 3 * mass_shift[2]
    internal('Mass shift %.3f +- 3 * %.3f', mass_shift[1], mass_shift[2])
    data_slice = data.loc[mask].sort_values(by=[params_dict['score_column'], 'spectrum'],
                                ascending=params_dict['score_ascending']).drop_duplicates(subset=params_dict['peptides_column'])
    internal('%d peptide rows selected for filtering', data_slice.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pepxml.filter_df(data_slice, key=params_dict['score_column'],
            fdr=params_dict['FDR'], reverse=not params_dict['score_ascending'], correction=params_dict['FDR_correction'], is_decoy='is_decoy')
    internal('Filtered data for %s: %d rows', mass_shift, df.shape[0])
    return ms_shift, df


def group_specific_filtering(data, mass_shifts, params_dict):
    """
    Selects window around found mass shift and filters using TDA.
    Window is defined as mean +- sigma.

    Parameters
    ----------
    data : DataFrame
        DF with all open search data.
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
        if ind != len(mass_shifts) - 1:
            diff = abs(ms[1] - mass_shifts[ind + 1][1])
            if diff < 3 * ms[2]:
                ms[2] = diff / 6
                mass_shifts[ind + 1][2] = diff / 6
                logger.debug('Mass shifts %.3f and %.3f are too close, setting their sigma to %.4f', ms[1], mass_shifts[ind + 1][1], diff / 6)
        shift, df = fdr_filter_mass_shift(ms, data, params_dict)

        if len(df) > 0:
            #  shift = np.mean(df[shifts]) ###!!!!!!!mean of from  fit!!!!
            out_data[mass_format(shift)] = (shift, df)
    logger.info('# of filtered mass shifts = %s', len(out_data))
    return out_data


def read_pepxml(fname, params_dict):
    '''
    Reads pepxml file and preprocess it.
    Parameters
    ----------
    fname: str
        Path to pepxml file
    params_dict: dict
        Dict with all input parameters
    Returns
    -------
    DataFrame
    '''
    logger.debug('Reading %s', fname)
    df = pepxml.DataFrame(fname, read_schema=False, columns=operator.itemgetter(
        'peptides_column', 'proteins_column', 'spectrum_column', 'mass_shifts_column', 'charge_column')(params_dict))
    return preprocess_df(df, fname, params_dict)


def read_csv(fname, params_dict):
    """
    Reads csv file.

    Paramenters
    -----------
    fname : str
        Path to file name.
    params_dict : dict
        Dict with paramenters for parsing csv file.
            `csv_delimiter`, `proteins_column`, `proteins_delimiter`
    Returns
    -------
    A DataFrame of csv file.

    """
    # logger.info('Reading %s', fname)
    df = pd.read_csv(fname, sep=params_dict['csv_delimiter'])
    protein = params_dict['proteins_column']
    if (df[protein].str[0] == '[').all() and (df[protein].str[-1] == ']').all():
        df[protein] = df[protein].apply(ast.literal_eval)
    else:
        df[protein] = df[protein].str.split(params_dict['proteins_delimeter'])
    return preprocess_df(df, fname, params_dict)


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


def read_input(args, params_dict):
    """
    Reads open search output, assembles all data in one DataFrame.

    """
    dfs = []

    def update_dfs(result):
        dfs.append(result)

    data = pd.DataFrame()
    logger.info('Reading input files...')
    readers = {
        'pepxml': read_pepxml,
        'csv': read_csv,
    }
    shifts = params_dict['mass_shifts_column']
    nproc = params_dict['processes']
    if nproc == 1:
        logger.debug('Reading files in one process.')
        for ftype, reader in readers.items():
            filenames = getattr(args, ftype)
            logger.debug('Filenames [%s]: %s', ftype, filenames)
            if filenames:
                for filename in filenames:
                    # dfs.append(reader(filename, params_dict))
                    dfs.append(reader(filename, params_dict))
    else:
        nfiles = 0
        for ftype, reader in readers.items():
            filenames = getattr(args, ftype)
            if filenames:
                nfiles += len(filenames)
        if nproc > 0:
            nproc = min(nproc, nfiles)
        else:
            nproc = min(nfiles, mp.cpu_count())
        logger.debug('Reading files using %s processes.', nproc)
        pool = mp.Pool(nproc)
        for ftype, reader in readers.items():
            filenames = getattr(args, ftype)
            logger.debug('Filenames [%s]: %s', ftype, filenames)
            if filenames:
                for filename in filenames:
                    # dfs.append(reader(filename, params_dict))
                    pool.apply_async(reader, args=(filename, params_dict), callback=update_dfs)
        pool.close()
        pool.join()
    logger.info('Starting analysis...')
    logger.debug('%d dfs collected.', len(dfs))
    data = pd.concat(dfs, axis=0)
    data.index = range(len(data))

    data['bin'] = np.digitize(data[shifts], params_dict['bins'])
    return data


def smooth(y, window_size=15, power=5):
    """
    Smoothes function.
    Paramenters
    -----------
    y : array-like
        function to smooth.
    window_size : int
        Smothing window.
    power : int
        Power of smothing function.

    Returns
    -------
    Smoothed function

    """
    y_smooth = savgol_filter(y, window_size, power)
    return y_smooth


def gauss(x, a, x0, sigma):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return a / sigma / np.sqrt(2 * np.pi) * np.exp(-(x - x0) * (x - x0) / (2 * sigma ** 2))


def gauss_fitting(center_y, x, y):
    """
    Fits with Gauss function
    `center_y` - starting point for `a` parameter of gauss
    `x` numpy array of mass shifts
    `y` numpy array of number of psms in this mass shifts

    """
    mean = (x * y).sum() / y.sum()
    sigma = np.sqrt((y * (x - mean) ** 2).sum() / y.sum())
    a = center_y * sigma * np.sqrt(2 * np.pi)
    try:
        popt, pcov = curve_fit(gauss, x, y, p0=(a, mean, sigma))
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except (RuntimeError, TypeError):
        return None, None


def fit_worker(args):
    return fit_batch_worker(*args)


def fit_batch_worker(out_path, batch_size, xs, ys, half_window, height_error, sigma_error):
    shape = int(math.ceil(np.sqrt(batch_size)))
    figsize = (shape * 3, shape * 4)
    plt.figure(figsize=figsize)
    plt.tight_layout()
    logger.debug('Created a figure with size %s', figsize)
    poptpvar = []
    for i in range(batch_size):
        center = i * (2 * half_window + 1) + half_window
        x = xs[center - half_window: center + half_window + 1]
        y = ys[center - half_window: center + half_window + 1]
        popt, perr = gauss_fitting(ys[center], x, y)
        plt.subplot(shape, shape, i + 1)
        if popt is None:
            label = 'NO FIT'
        else:
            if (x[0] <= popt[1] and popt[1] <= x[-1] and perr[0] / popt[0] < height_error
                    and perr[2] / popt[2] < sigma_error):
                label = 'PASSED'
                poptpvar.append(np.concatenate([popt, perr]))
                plt.vlines(popt[1] - 3 * popt[2], 0, ys[center], label='3sigma interval')
                plt.vlines(popt[1] + 3 * popt[2], 0, ys[center])
            else:
                label = 'FAILED'
        plt.plot(x, y, 'b+:', label=label)
        if label != 'NO FIT':
            plt.scatter(x, gauss(x, *popt), label='Gaussian fit\n $\\sigma$ = {:.4f}'.format(popt[2]))

        plt.legend()
        plt.title("{0:.3f}".format(xs[center]))
        plt.grid(True)

    logger.debug('Fit done. Saving %s...', out_path)
    plt.savefig(out_path)
    plt.close()
    return poptpvar


def fit_peaks(data, args, params_dict):
    """
    Finds Gauss-like peaks in mass shift histogram.

    Parameters
    ----------
    data : DataFRame
        A DF with all (non-filtered) results of open search.
    args: argsparse
    params_dict : dict
        Parameters dict.
    """
    logger.info('Performing Gaussian fit...')
    fit_batch = params_dict['fit batch']
    half_window = int(params_dict['window'] / 2) + 1
    hist = np.histogram(data[data['is_decoy'] == False][params_dict['mass_shifts_column']], bins=params_dict['bins'])
    hist_y = smooth(hist[0], window_size=params_dict['window'], power=5)
    hist_x = 0.5 * (hist[1][:-1] + hist[1][1:])
    loc_max_candidates_ind = argrelextrema(hist_y, np.greater_equal)[0]
    # smoothing and finding local maxima
    min_height = 2 * np.median(hist[0][hist[0] > 1])
    # minimum bin height expected to be peak approximate noise level as median of all non-negative
    loc_max_candidates_ind = loc_max_candidates_ind[hist_y[loc_max_candidates_ind] >= min_height]
    if not loc_max_candidates_ind.size:
        logger.info('No peaks found for fit.')
        return hist, np.array([])
    height_error = params_dict['max_deviation_height']
    sigma_error = params_dict['max_deviation_sigma']
    logger.debug('Candidates for fit: %s', len(loc_max_candidates_ind))
    nproc = int(math.ceil(len(loc_max_candidates_ind) / fit_batch))
    maxproc = params_dict['processes']
    if maxproc > 0:
        nproc = min(nproc, maxproc)
    if nproc > 1:
        arguments = []
        logger.debug('Splitting the fit into %s batches...', nproc)
        n = min(nproc, mp.cpu_count())
        logger.debug('Creating a pool of %s processes.', n)
        pool = mp.Pool(n)
        for proc in range(nproc):
            xlist = [hist_x[center - half_window: center + half_window + 1]
                for center in loc_max_candidates_ind[proc * fit_batch: (proc + 1) * fit_batch]]
            xs = np.concatenate(xlist)
            ylist = [hist[0][center - half_window: center + half_window + 1]
                for center in loc_max_candidates_ind[proc * fit_batch: (proc + 1) * fit_batch]]
            ys = np.concatenate(ylist)
            out = os.path.join(args.dir, 'gauss_fit_{}.pdf'.format(proc + 1))
            arguments.append((out, len(xlist), xs, ys, half_window, height_error, sigma_error))
        res = pool.map_async(fit_worker, arguments)
        poptpvar_list = res.get()
        # logger.debug(poptpvar_list)
        pool.close()
        pool.join()
        logger.debug('Workers done.')
        poptpvar = [p for r in poptpvar_list for p in r]
    else:
        xs = np.concatenate([hist_x[center - half_window: center + half_window + 1]
                for center in loc_max_candidates_ind])
        ys = np.concatenate([hist[0][center - half_window: center + half_window + 1]
                for center in loc_max_candidates_ind])
        poptpvar = fit_batch_worker(os.path.join(args.dir, 'gauss_fit.pdf'),
            len(loc_max_candidates_ind), xs, ys, half_window, height_error, sigma_error)

    logger.debug('Returning from fit_peaks. Array size is %d.', len(poptpvar))
    return hist, np.array(poptpvar)


def read_mgf(file_path):
    return mgf.IndexedMGF(file_path)


def read_mzml(file_path):  # write this
    return mzml.PreIndexedMzML(file_path)


def read_spectra(args):

    readers = {
        'mgf': read_mgf,
        'mzml': read_mzml,
    }
    out_dict = {}
    for ftype, reader in readers.items():
        filenames = getattr(args, ftype)
        if filenames:
            for filename in filenames:
                name = os.path.split(filename)[-1].split('.')[0]  # write it in a proper way
                out_dict[name] = reader(filename)
    return out_dict


def read_config_file(fname):
    params = ConfigParser(delimiters=('=', ':'), comment_prefixes=('#'), inline_comment_prefixes=('#'))

    params.read(AA_STAT_PARAMS_DEFAULT)
    if fname:
        if not os.path.isfile(fname):
            logger.error('Configuration file not found: %s', fname)
        else:
            params.read(fname)
    else:
        logger.info('Using default parameters for AA_stat.')
    return params


def get_parameters(params):
    """
    Reads paramenters from cfg file to one dict.
    Returns dict.
    """
    parameters_dict = defaultdict()
    # data
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
    parameters_dict['score_column'] = params.get('csv input', 'score column')
    parameters_dict['measured_mass_column'] = params.get('csv input', 'measured mass column')
    parameters_dict['calculated_mass_column'] = params.get('csv input', 'calculated mass column')
    parameters_dict['rt_column'] = params.get('csv input', 'retention time column')
    parameters_dict['next_aa_column'] = params.get('csv input', 'next aa column')
    parameters_dict['prev_aa_column'] = params.get('csv input', 'previous aa column')
    parameters_dict['score_ascending'] = params.getboolean('csv input', 'score ascending')

    # general
    parameters_dict['bin_width'] = params.getfloat('general', 'width of bin in histogram')
    parameters_dict['so_range'] = tuple(float(x) for x in params.get('general', 'open search range').split(','))
    parameters_dict['walking_window'] = params.getfloat('general', 'shifting window')
    parameters_dict['FDR_correction'] = params.getboolean('general', 'FDR correction')
    parameters_dict['processes'] = params.getint('general', 'processes')
    parameters_dict['zero_window'] = params.getfloat('general', 'zero peak window')

    parameters_dict['zero bin tolerance'] = params.getfloat('general', 'zero shift mass tolerance')
    parameters_dict['zero min intensity'] = params.getfloat('general', 'zero shift minimum intensity')
    parameters_dict['min_peptides_for_mass_calibration'] = params.getint('general', 'minimum peptides for mass calibration')

    parameters_dict['specific_mass_shift_flag'] = params.getboolean('general', 'use specific mass shift window')
    parameters_dict['specific_window'] = [float(x) for x in params.get('general', 'specific mass shift window').split(',')]

    parameters_dict['figsize'] = tuple(float(x) for x in params.get('general', 'figure size in inches').split(','))
    parameters_dict['calibration'] = params.get('general', 'mass calibration')

    #clustering
    parameters_dict['clustering'] = params.getboolean('clustering', 'use clustering')
    parameters_dict['eps_adjust'] = params.getfloat('clustering', 'dbscan eps factor')
    parameters_dict['min_samples'] = params.getfloat('clustering', 'dbscan min_samples')

    # fit
    parameters_dict['shift_error'] = params.getint('fit', 'shift error')
    #    parameters_dict['max_deviation_x'] = params.getfloat('fit', 'standard deviation threshold for center of peak')
    parameters_dict['max_deviation_sigma'] = params.getfloat('fit', 'standard deviation threshold for sigma')
    parameters_dict['max_deviation_height'] = params.getfloat('fit', 'standard deviation threshold for height')
    parameters_dict['fit batch'] = params.getint('fit', 'batch')
    # localization
    parameters_dict['spectrum_column'] = params.get('localization', 'spectrum column')
    parameters_dict['charge_column'] = params.get('localization', 'charge column')
    parameters_dict['ion_types'] = tuple(params.get('localization', 'ion type').replace(' ', '').split(','))
    parameters_dict['frag_acc'] = params.getfloat('localization', 'fragmentation mass tolerance')
    parameters_dict['candidate threshold'] = params.getfloat('localization', 'frequency threshold')
    parameters_dict['isotope mass tolerance'] = params.getfloat('localization', 'isotope mass tolerance')
    parameters_dict['unimod mass tolerance'] = params.getfloat('localization', 'unimod mass tolerance')
    parameters_dict['min_spec_matched'] = params.getint('localization', 'minimum matched peaks')

    # modifications
    parameters_dict['variable_mods'] = params.getint('modifications', 'recommend variable modifications')
    parameters_dict['multiple_mods'] = params.getboolean('modifications', 'recommend multiple modifications on residue')
    parameters_dict['fix_mod_zero_thresh'] = params.getfloat('modifications', 'fixed modification intensity threshold')
    parameters_dict['min_fix_mod_pep_count_factor'] = params.getfloat('modifications', 'peptide count factor threshold')
    parameters_dict['recommend isotope threshold'] = params.getfloat('modifications', 'isotope error abundance threshold')
    parameters_dict['min_loc_count'] = params.getint('modifications', 'minimum localization count')
    return parameters_dict


def set_additional_params(params_dict):
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
        params_dict['window'] = int(window)  # should be odd
    params_dict['bins'] = np.arange(params_dict['so_range'][0],
        params_dict['so_range'][1] + params_dict['bin_width'], params_dict['bin_width'])


def get_params_dict(args):
    fname = args.params
    outdir = args.dir
    params = read_config_file(fname)
    params_dict = get_parameters(params)
    set_additional_params(params_dict)
    params_dict['output directory'] = outdir
    if args.pepxml:
        params_dict['fix_mod'] = get_fix_modifications(args.pepxml[0])
    return params_dict


_Mkstyle = matplotlib.markers.MarkerStyle
_marker_styles = [_Mkstyle('o', fillstyle='full'), (_Mkstyle('o', fillstyle='left'), _Mkstyle('o', fillstyle='right')),
    (_Mkstyle('o', fillstyle='top'), _Mkstyle('o', fillstyle='bottom')), (_Mkstyle(8), _Mkstyle(9)),
    (_Mkstyle('v'), _Mkstyle('^')), (_Mkstyle('|'), _Mkstyle('_')), (_Mkstyle('+'), _Mkstyle('x'))]


def _generate_pair_markers():
    '''Produce style & color pairs for localization markers (except the main one).'''
    for i in [3, 4, 5, 0, 1, 2]:
        for ms in _marker_styles[1:]:
            yield colors[i], ms


def _get_max(arr):
    values = [x for x in arr if x is not None]
    if values:
        return max(values)
    return 0


def plot_figure(ms_label, ms_counts, left, right, params_dict, save_directory, localizations=None, sumof=None):
    """
    Plots amino acid spatistics.

    Parameters
    ----------
    ms_label : str
        Mass shift in string format.
    ms_counts : int
        Number of peptides in a mass shift.
    left : list
        Amino acid statistics data [[values], [errors]]
    right : list
        Amino acid frequences in peptides
     params_dict : dict
        Parameters dict.
    save_directory: str
        Saving directory.
    localizations : Counter
         Localization counter using  ms/ms level.
    sumof : List
        List of str tuples for constituent mass shifts.

    """
    b = 0.1  # shift in bar plots
    width = 0.2  # for bar plots
    labels = params_dict['labels']
    labeltext = ms_label + ' Da mass shift,\n' + str(ms_counts) + ' peptides'
    x = np.arange(len(labels))
    distributions = left[0]
    errors = left[1]

    fig, ax_left = plt.subplots()
    fig.set_size_inches(params_dict['figsize'])

    ax_left.bar(x - b, distributions.loc[labels],
            yerr=errors.loc[labels], width=width, color=colors[2], linewidth=0)

    ax_left.set_ylabel('Relative AA abundance', color=colors[2])
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels)
    ax_left.hlines(1, -1, x[-1] + 1, linestyles='dashed', color=colors[3])
    ax_right = ax_left.twinx()

    ax_right.bar(x + b, right, width=width, linewidth=0, color=colors[0])

    ax_right.set_ylim(0, 125)
    ax_right.set_yticks(np.arange(0, 120, 20))
    ax_right.set_ylabel('Peptides with AA, %', color=colors[0])

    ax_left.spines['left'].set_color(colors[2])
    ax_right.spines['left'].set_color(colors[2])

    ax_left.spines['right'].set_color(colors[0])
    ax_right.spines['right'].set_color(colors[0])
    ax_left.tick_params('y', colors=colors[2])
    ax_right.tick_params('y', colors=colors[0])

    pright = matplotlib.lines.Line2D([], [], marker=None, label=labeltext, alpha=0)

    ax_left.set_xlim(-1, x[-1] + 1)
    ax_left.set_ylim(0, distributions.loc[labels].max() * 1.4)

    logger.debug('Localizations for %s figure: %s', ms_label, localizations)
    if localizations:
        ax3 = ax_left.twinx()
        ax3.spines['right'].set_position(('axes', 1.1))
        ax3.set_frame_on(True)
        ax3.patch.set_visible(False)
        ax3.set_ylabel('Localization count', color=colors[3])
        for sp in ax3.spines.values():
            sp.set_visible(False)
        ax3.spines['right'].set_visible(True)
        ax3.spines['right'].set_color(colors[3])
        ax3.tick_params('y', colors=colors[3])
        # plot simple modifications (not sum) with the first style,
        # then parts of sum
        values = [localizations.get(key + '_' + ms_label) for key in labels]
        maxcount = _get_max(values)
        label_prefix = 'Location of '
        ax3.scatter(x, values, marker=_marker_styles[0], color=colors[3], label=label_prefix + ms_label)
        if isinstance(sumof, list):
            for pair, (color, style) in zip(sumof, _generate_pair_markers()):
                values_1 = [localizations.get(key + '_' + pair[0]) for key in labels]
                maxcount = max(maxcount, _get_max(values_1))
                ax3.scatter(x, values_1, marker=style[0], color=color, label=label_prefix + pair[0])
                if pair[0] != pair[1]:
                    values_2 = [localizations.get(key + '_' + pair[1]) for key in labels]
                    if values_2:
                        maxcount = max(maxcount, _get_max(values_2))
                        ax3.scatter(x, values_2, marker=style[1], color=color, label=label_prefix + pair[1])
        terms = {key for key in localizations if key[1:6] == '-term'}
        # logger.debug('Found terminal localizations: %s', terms)
        for t in terms:
            label = '{} at {}: {}'.format(*reversed(t.split('_')), localizations[t])
            p = ax3.plot([], [], label=label)[0]
            p.set_visible(False)
        pright.set_label(pright.get_label() + '\nNot localized: {}'.format(localizations.get('non-localized', 0)))
        if maxcount:
            ax3.legend(loc='upper left', ncol=2)
        ax3.set_ylim(0, 1.4 * max(maxcount, 1))

    ax_right.legend(handles=[pright], loc='upper right', edgecolor='dimgrey', fancybox=True, handlelength=0)
    fig.tight_layout()
    fig.savefig(os.path.join(save_directory, ms_label + '.png'), dpi=500)
    fig.savefig(os.path.join(save_directory, ms_label + '.svg'))
    plt.close()


def summarizing_hist(table, save_directory):
    ax = table.sort_values('mass shift').plot(
        y='# peptides in bin', kind='bar', color=colors[2], figsize=(len(table), 5))
    ax.set_title('Peptides in mass shifts', fontsize=12)  # PSMs
    ax.set_xlabel('Mass shift', fontsize=10)
    ax.set_ylabel('Number of peptides')
    ax.set_xticklabels(table.sort_values('mass shift')['mass shift'].apply('{:.2f}'.format))

    total = sum(i.get_height() for i in ax.patches)
    max_height = 0
    for i in ax.patches:
        current_height = i.get_height()
        if current_height > max_height:
            max_height = current_height
        ax.text(i.get_x() - 0.03, current_height + 40,
            '{:>6.2%}'.format(i.get_height() / total), fontsize=10, color='dimgrey')

    plt.ylim(0, max_height * 1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, 'summary.png'))  # dpi=500
    plt.savefig(os.path.join(save_directory, 'summary.svg'))


def format_unimod_repr(record_id):
    record = UNIMOD[record_id]
    return '<a href="http://www.unimod.org/modifications_view.php?editid1={0[record_id]}">{0[title]}</a>'.format(record)


def matches(row, ms, sites, params_dict):
    ldict = row['localization_count']
    if 'non-localized' in ldict:
        return False
    for loc in ldict:
        site, shift = loc.split('_')
        if shift != ms:
            continue
        for possible_site, possible_position in sites:
            if site == possible_site:
                if possible_position[:3] == 'Any':  # Anywhere, Any C-term, Any N-term
                    return True
                if possible_position == 'Protein N-term' and row[params_dict['prev_aa_column']] == '-':
                    return True
                if possible_position == 'Protein C-term' and row[params_dict['next_aa_column']] == '-':
                    return True
    return False


def format_unimod_info(row, df, params_dict):
    out = []
    for record_id in row['unimod accessions']:
        logger.debug('Processing record %s', record_id)
        name = format_unimod_repr(record_id)
        if 'top isoform' in df:
            record = UNIMOD[record_id]
            sites = {(group['site'], group['position']) for group in record['specificity']}
            internal(sites)
            matching = df.apply(matches, args=(row.name, sites, params_dict), axis=1).sum()
            total = row['# peptides in bin']
            out.append('{} ({:.0%} match)'.format(name, matching / total))
        else:
            out.append(name)
    return out


def get_label(table, ms, second=False):
    row = table.loc[ms]
    if row['isotope index'] is None and row['sum of mass shifts'] is None:
        if len(row['unimod accessions']) == 1:
            return ('+' if second else '') + format_unimod_repr(next(iter(row['unimod accessions'])))
    return ms


def format_info(row, table, mass_shift_data_dict, params_dict):
    options = format_unimod_info(row, mass_shift_data_dict[row.name][1], params_dict)
    if row['isotope index']:
        options.append('isotope of {}'.format(get_label(table, row['isotope index'])))
    if isinstance(row['sum of mass shifts'], list):
        options.extend('{}{}'.format(get_label(table, s1), get_label(table, s2, True)) for s1, s2 in row['sum of mass shifts'])
    return ', '.join(options)


def get_varmod_combinations(recommended_vmods, values):
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
                if abs(values[c1[1]] + values[c2[1]] - values[shift]) <= COMBINATION_TOLERANCE:
                    out[i] = (c1[1], c2[1])
    return out


def get_opposite_mods(fmods, rec_fmods, rec_vmods, values):
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
            if aaf == aav and abs(fmod + values[vmod]) < COMBINATION_TOLERANCE:
                vmod_idx.append(i)
    return vmod_idx


def render_html_report(table_, mass_shift_data_dict, params_dict, recommended_fmods, recommended_vmods, vmod_combinations, opposite,
        save_directory, ms_labels, step=None):
    path = os.path.join(save_directory, 'report.html')
    if os.path.islink(path):
        logger.debug('Deleting link: %s.', path)
        os.remove(path)

    if table_ is None:
        with open(path, 'w') as f:
            f.write('No mass shifts found.')
        return
    table = table_.copy()
    labels = params_dict['labels']
    table['Possible interpretations'] = table.apply(format_info, axis=1, args=(table, mass_shift_data_dict, params_dict))

    with pd.option_context('display.max_colwidth', 250):
        columns = list(table.columns)
        mslabel = '<a id="binh" href="#">mass shift</a>'
        columns[0] = mslabel
        table.columns = columns
        to_hide = list({'is reference', 'sum of mass shifts', 'isotope index', 'unimod accessions',
            'is isotope', 'unimod candidates'}.intersection(columns))
        table_html = table.style.hide_index().hide_columns(to_hide).applymap(
            lambda val: 'background-color: yellow' if val > 1.5 else '', subset=labels
            ).set_precision(3).apply(
            lambda row: ['background-color: #cccccc' if row['is reference'] else '' for cell in row], axis=1).set_table_styles([
                {'selector': 'tr:hover', 'props': [('background-color', 'lightyellow')]},
                {'selector': 'td, th', 'props': [('text-align', 'center')]},
                {'selector': 'td, th', 'props': [('border', '1px solid black')]}]
            ).format({  #'Unimod': '<a href="{}">search</a>'.format,
                mslabel: '<a href="#">{}</a>'.format(MASS_FORMAT).format,
                '# peptides in bin': '<a href="#">{}</a>'.format}).bar(subset='# peptides in bin', color=cc[2]).render(
            uuid="aa_stat_table")

    peptide_tables = []
    for ms in table.index:
        fname = os.path.join(save_directory, ms + '.csv')
        if os.path.isfile(fname):
            df = pd.read_csv(fname, sep='\t')
            if 'localization score' in df:
                df.sort_values(['localization score'], ascending=False, inplace=True)
            peptide_tables.append(df.to_html(
                table_id='peptides_' + ms, classes=('peptide_table',), index=False, escape=False, na_rep='',
                formatters={
                    'top isoform': lambda form: re.sub(r'([A-Z]\[[+-]?[0-9]+\])', r'<span class="loc">\1</span>', form),
                    'localization score': '{:.2f}'.format}))
        else:
            logger.debug('File not found: %s', fname)

    if params_dict['fix_mod']:
        d = params_dict['fix_mod'].copy()
        d = masses_to_mods(d)
        fixmod = pd.DataFrame.from_dict(d, orient='index', columns=['value']).T.style.set_caption(
            'Configured, fixed').format(MASS_FORMAT).render(uuid="set_fix_mod_table")
    else:
        fixmod = "Set modifications: none."
    if recommended_fmods:
        recmod = pd.DataFrame.from_dict(recommended_fmods, orient='index', columns=['value']).T.style.set_caption(
            'Recommended, fixed').render(uuid="rec_fix_mod_table")
    else:
        recmod = "Recommended modifications: none."

    if recommended_vmods:
        vmod_comb_i = json.dumps(list(vmod_combinations))
        vmod_comb_val = json.dumps(['This modification is a combination of {} and {}.'.format(*v) for v in vmod_combinations.values()])
        opp_mod_i = json.dumps(opposite)
        opp_mod_v = json.dumps(['This modification negates a fixed modification.\n'
            'For closed search, it is equivalent to set {} @ {} as variable.'.format(
                mass_format(-ms_labels[recommended_vmods[i][1]]), recommended_vmods[i][0]) for i in opposite])
        table_styles = [{'selector': 'th.col_heading', 'props': [('display', 'none')]},
            {'selector': 'th.blank', 'props': [('display', 'none')]},
            {'selector': '.data.row0', 'props': [('font-weight', 'bold')]}]
        for i in vmod_combinations:
            table_styles.append({'selector': '.data.col{}'.format(i), 'props': [('background-color', 'lightyellow')]})
        for i in opposite:
            table_styles.append({'selector': '.data.col{}'.format(i), 'props': [('background-color', 'lightyellow')]})
        rec_var_mods = pd.DataFrame.from_records(recommended_vmods, columns=['', 'value']).T.style.set_caption(
            'Recommended, variable').format({'isotope error': '{:.0f}'}).set_table_styles(table_styles).render(uuid="rec_var_mod_table")
    else:
        rec_var_mods = "Recommended variable modifications: none."
        vmod_comb_i = vmod_comb_val = opp_mod_i = opp_mod_v = '[]'

    reference = table.loc[table['is reference']].index[0]

    if step is None:
        steps = ''
    else:
        if step != 1:
            prev_url = os.path.join(os.path.pardir, 'os_step_{}'.format(step - 1), 'report.html')
            prev_a = r'<a class="prev steplink" href="{}">Previous step</a>'.format(prev_url)
        else:
            prev_a = ''
        if recommended_fmods:
            next_url = os.path.join(os.path.pardir, 'os_step_{}'.format(step + 1), 'report.html')
            next_a = r'<a class="next steplink" href="{}">Next step</a>'.format(next_url)
        else:
            next_a = ''
        steps = prev_a + '\n' + next_a

    version = pkg_resources.get_distribution('AA_stat').version

    write_html(path, table_html=table_html, peptide_tables=peptide_tables, fixmod=fixmod, reference=reference,
        recmod=recmod, rec_var_mod=rec_var_mods, steps=steps, version=version, date=datetime.now(),
        vmod_comb_i=vmod_comb_i, vmod_comb_val=vmod_comb_val, opposite_i=opp_mod_i, opposite_v=opp_mod_v)


def write_html(path, **template_vars):
    with warnings.catch_warnings():
        if not sys.warnoptions:
            warnings.simplefilter('ignore')
        templateloader = jinja2.PackageLoader('AA_stat', '')
        templateenv = jinja2.Environment(loader=templateloader, autoescape=False)
        template_file = 'report.template'
        template = templateenv.get_template(template_file)

    with open(path, 'w') as output:
        output.write(template.render(template_vars))


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
        sites_set.update(s['site'] for s in row['specificity'])
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


def format_isoform(row):
    ms = row['mod_dict']
    seq = row['top isoform']
    return re.sub(r'([a-z])([A-Z])', lambda m: '{}[{:+.0f}]'.format(m.group(2), float(ms[m.group(1)])), seq)


def table_path(dir, ms):
    return os.path.join(dir, ms + '.csv')


def save_df(ms, df, save_directory, peptide, spectrum):
    with open(table_path(save_directory, ms), 'w') as out:
        df[[peptide, spectrum]].to_csv(out, index=False, sep='\t')


def save_peptides(data, save_directory, params_dict):
    peptide = params_dict['peptides_column']
    spectrum = params_dict['spectrum_column']
    for ms_label, (ms, df) in data.items():
        save_df(ms_label, df, save_directory, peptide, spectrum)


def get_fix_modifications(pepxml_file):
    out = {}
    p = pepxml.PepXML(pepxml_file, use_index=False)
    mod_list = list(p.iterfind('aminoacid_modification'))
    logger.debug('mod_list: %s', mod_list)
    p.reset()
    term_mods = list(p.iterfind('terminal_modification'))
    logger.debug('term_mods: %s', term_mods)
    p.close()
    for m in mod_list:
        if m['variable'] == 'N':
            out[m['aminoacid']] = m['mass']
    for m in term_mods:
        if m['variable'] == 'N':
            if m['terminus'] == 'N':
                out['H-'] = m['mass']
            else:
                out['-OH'] = m['mass']
    return out


def parse_l10n_site(site):
    aa, shift = site.split('_')
    return aa, shift


def mass_to_mod(label, value, aa_mass=mass.std_aa_mass):
    return value - aa_mass.get(label, 0)


def masses_to_mods(d):
    aa_mass = mass.std_aa_mass.copy()
    aa_mass['H-'] = 1.007825
    aa_mass['-OH'] = 17.00274
    d = {k: mass_to_mod(k, v, aa_mass) for k, v in d.items()}
    if 'H-' in d:
        d['N-term'] = d.pop('H-')
    if '-OH' in d:
        d['C-term'] = d.pop('-OH')
    return d


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
