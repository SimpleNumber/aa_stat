import matplotlib
matplotlib.use('Agg')
import pylab as plt

import math
import os
import logging
import warnings
import multiprocessing as mp

import numpy as np
from sklearn import cluster
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema, savgol_filter
import seaborn as sb

from . import utils


logger = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.category').disabled = True

cc = ["#FF6600",
      "#FFCC00",
      "#88AA00",
      "#006688",
      "#5FBCD3",
      "#7137C8",
      ]
sb.set_style('white')
colors = sb.color_palette(palette=cc)


def _gauss_fit_slice(to_fit, unit, filename, title, params_dict, mpl_back):
    logger.debug('Fitting zero-shift peptides...')
    f = plt.figure()
    hist_0 = np.histogram(to_fit, bins=int(params_dict['zero_window'] / params_dict['bin_width']))
    hist_y = hist_0[0]
    hist_x = 0.5 * (hist_0[1][:-1] + hist_0[1][1:])
    plt.plot(hist_x, hist_y, 'b+')
    popt, perr = gauss_fitting(max(hist_y), hist_x, hist_y)
    plt.scatter(hist_x, gauss(hist_x, *popt), label='Gaussian fit')
    plt.xlabel('massdiff, ' + unit)
    plt.title(title)
    mpl_back.savefig(f)
    plt.close()
    logger.info('Systematic shift is %.4f %s for file %s [ %s ]', popt[1], unit, filename, title)
    return popt


def clusters(df, to_fit, unit, filename, params_dict, mpl_back):
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
    f = plt.figure()
    sc = plt.scatter(to_fit, X[:, 1], c=clustering.labels_)
    plt.legend(*sc.legend_elements(), title='Clusters')
    plt.xlabel(unit)
    plt.ylabel(params_dict['rt_column'])
    mpl_back.savefig(f)
    plt.close()
    f = plt.figure()
    for c in np.unique(clustering.labels_):
        plt.hist(X[clustering.labels_ == c, 1], label=c, alpha=0.5)
    plt.xlabel(params_dict['rt_column'])
    plt.legend()
    mpl_back.savefig(f)
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


def filter_clusters(clustering, df, to_fit, params_dict):
    nclusters = clustering.labels_.max() + 1
    logger.debug('Found %d clusters, %d labels assigned.', nclusters, clustering.labels_.size)
    if not nclusters:
        return []

    out = []
    clustered_peps = 0
    for i in np.unique(clustering.labels_):
        if i == -1:
            continue
        npep = (clustering.labels_ == i).sum()
        if npep < params_dict['min_peptides_for_mass_calibration']:
            logger.debug('Cluster %s is too small for calibration (%d), discarding.', i, npep)
            continue
        span_pct = cluster_time_percentage(clustering, i, df, to_fit, params_dict)
        if span_pct < params_dict['cluster_span_min']:
            logger.debug('Cluster %s spans %.2f%% of the run (too small, thresh = %.2f%%). Discarding.',
                i, span_pct * 100, params_dict['cluster_span_min'] * 100)
            continue
        out.append(i)
        clustered_peps += npep

    logger.debug('Pre-selected clusters: %s', out)
    logger.debug('%.2f%% peptides in clusters, threshold is %.2f%%.',
        clustered_peps / df.shape[0] * 100, params_dict['clustered_pct_min'] * 100)
    if clustered_peps / df.shape[0] < params_dict['clustered_pct_min']:
        logger.debug('Too few peptides in clusters, discarding clusters altogether.')
        return []
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
        freq_measured = 1e6 / np.sqrt(utils.measured_mz_series(df, params_dict))
        freq_calculated = 1e6 / np.sqrt(utils.calculated_mz_series(df, params_dict))
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
    logger.debug('Fit batch worker called with: %s, %s, len(xs)=%s, len(ys)=%s, %s, %s, %s',
        out_path, batch_size, xs.size, ys.size, half_window, height_error, sigma_error)
    assert batch_size == int(xs.size / (2 * half_window + 1))
    poptpvar = []
    for i in range(batch_size):
        center = i * (2 * half_window + 1) + half_window
        x = xs[center - half_window: center + half_window + 1]
        y = ys[center - half_window: center + half_window + 1]
        utils.internal('len(x): %s, len(y): %s, center: %s', x.size, y.size, center)
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


def concat_slices(x, y, loc_max_candidates_ind, half_window):
    xs = np.concatenate([x[max(0, center - half_window): center + half_window + 1]
                for center in loc_max_candidates_ind])
    size_init = xs.size
    ys = np.concatenate([y[max(0, center - half_window): center + half_window + 1]
            for center in loc_max_candidates_ind])
    xstep = x[1] - x[0]
    if loc_max_candidates_ind[0] < half_window:
        n = half_window - loc_max_candidates_ind[0]
        add = np.arange(xs[0] - n * xstep, xs[0] - 0.5 * xstep, xstep)
        utils.internal('Adding padding on the left: index %d and half-window %d require %d extra positions: %s',
            loc_max_candidates_ind[0], half_window, n, add)
        xs = np.concatenate([add, xs])
        ys = np.concatenate([np.zeros(n), ys])
    if loc_max_candidates_ind[-1] + half_window + 1 > x.size:
        n = loc_max_candidates_ind[-1] + half_window + 1 - x.size
        add = np.arange(xs[-1] + xstep, xs[-1] + (n + 0.5) * xstep, xstep)
        utils.internal('Adding padding on the right: index %d and half-window %d require %d extra positions: %s',
            loc_max_candidates_ind[-1], half_window, n, add)
        xs = np.concatenate([xs, add])
        ys = np.concatenate([ys, np.zeros(n)])
    utils.internal('%d -> %d', size_init, xs.size)
    assert xs.size % len(loc_max_candidates_ind) == 0
    assert xs.size == ys.size
    return xs, ys


def fit_peaks(array, args, params_dict):
    """
    Finds Gauss-like peaks in mass shift histogram.

    Parameters
    ----------
    array : np.ndarray
        An array of observed mass shifts of target PSMs
    args: argparse
    params_dict : dict
        Parameters dict.
    """
    logger.info('Performing Gaussian fit...')
    fit_batch = params_dict['fit batch']
    half_window = int(params_dict['window'] / 2) + 1
    hist = np.histogram(array, bins=params_dict['bins'])
    hist_y = smooth(hist[0], window_size=params_dict['window'], power=5)
    hist_x = 0.5 * (hist[1][:-1] + hist[1][1:])
    logger.debug('Histogram sizes: hist: (%d, %d), hist_x: %d, hist_y: %d', hist[0].size, hist[1].size, hist_x.size, hist_y.size)
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
    utils.internal('Candidate locations: %s', loc_max_candidates_ind)
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
            batch = loc_max_candidates_ind[proc * fit_batch: (proc + 1) * fit_batch]
            xs, ys = concat_slices(hist_x, hist[0], batch, half_window)
            out = os.path.join(args.dir, 'gauss_fit_{}.pdf'.format(proc + 1))
            arguments.append((out, len(batch), xs, ys, half_window, height_error, sigma_error))
        res = pool.map_async(fit_worker, arguments)
        poptpvar_list = res.get()
        # logger.debug(poptpvar_list)
        pool.close()
        pool.join()
        logger.debug('Workers done.')
        poptpvar = [p for r in poptpvar_list for p in r]
    else:
        xs, ys = concat_slices(hist_x, hist[0], loc_max_candidates_ind, half_window)
        poptpvar = fit_batch_worker(os.path.join(args.dir, 'gauss_fit.pdf'),
            len(loc_max_candidates_ind), xs, ys, half_window, height_error, sigma_error)

    logger.debug('Returning from fit_peaks. Array size is %d.', len(poptpvar))
    return np.array(poptpvar)


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
    distributions = left[0].fillna(0)
    errors = left[1].fillna(0)
    logger.debug('Distributions for %s figure: %s', ms_label, distributions)
    logger.debug('Errors for %s figure: %s', ms_label, localizations)

    fig, ax_left = plt.subplots()
    fig.set_size_inches(params_dict['figsize'])

    ax_left.bar(x - b, distributions.loc[labels],
            yerr=errors.loc[labels], width=width, color=colors[2], linewidth=0)

    ax_left.set_ylabel('Relative AA abundance', color=colors[2])
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels)
    ax_left.hlines(1, -1, x[-1] + 1, linestyles='dashed', color=colors[2])
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
    fig.savefig(os.path.join(save_directory, ms_label + '.png'), dpi=params_dict['figure_dpi'])
    fig.savefig(os.path.join(save_directory, ms_label + '.svg'))
    plt.close()


def summarizing_hist(table, save_directory, dpi):
    width = 0.8
    fig, ax = plt.subplots(figsize=(len(table), 5))
    ax.bar(range(len(table)), table.sort_values('mass shift')['# peptides in bin'],
        color=colors[2], align='center', width=width)
    ax.set_title('Peptides in mass shifts', fontsize=12)
    ax.set_xlabel('Mass shift', fontsize=10)
    ax.set_ylabel('Number of peptides')
    ax.set_xlim((-1, len(table)))
    ax.set_xticks(range(len(table)))
    ax.set_xticklabels(table.sort_values('mass shift')['mass shift'].apply('{:.2f}'.format))

    total = table['# peptides in bin'].sum()
    vdist = table['# peptides in bin'].max() * 0.01
    max_height = 0
    for i, patch in enumerate(ax.patches):
        current_height = patch.get_height()
        if current_height > max_height:
            max_height = current_height
        ax.text(patch.get_x() + width / 2, current_height + vdist,
            '{:>6.2%}'.format(table.at[table.index[i], '# peptides in bin'] / total),
            fontsize=10, color='dimgrey', ha='center')

    plt.ylim(0, max_height * 1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, 'summary.png'), dpi=dpi)
    plt.savefig(os.path.join(save_directory, 'summary.svg'))
    plt.close()
