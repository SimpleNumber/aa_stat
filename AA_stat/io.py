import matplotlib
matplotlib.use('Agg')
import pylab as plt

import ast
import os
from configparser import ConfigParser
import multiprocessing as mp
from collections import defaultdict
import logging
import operator

import numpy as np
import pandas as pd
from pyteomics import pepxml, mgf, mzml
from . import utils, stats

AA_STAT_PARAMS_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default.cfg')
logger = logging.getLogger(__name__)


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
    ms, filtered = utils.fdr_filter_mass_shift([None, zero_bin, window], df, params_dict)
    n = filtered.shape[0]
    logger.debug('%d filtered peptides near zero.', n)
    if params_dict['calibration'] == 'off':
        logger.info('Mass calibration is disabled. Skipping.')
    elif params_dict['calibration'] != 'simple':
        if n < params_dict['min_peptides_for_mass_calibration']:
            logger.warning('Skipping mass calibration: not enough peptides near zero mass shift.')
        else:
            to_fit, unit = stats.get_fittable_series(filtered, params_dict)
            # save copies of mass shift column, for use in boolean indexing
            shift_copy = df[shifts].copy()
            old_shifts = filtered[shifts].copy()
            if params_dict['clustering']:
                clustering = stats.clusters(filtered, to_fit, unit, filename, params_dict)
                if clustering is None:
                    filtered_clusters = None
                else:
                    filtered_clusters = stats.filter_clusters(clustering, filtered, to_fit, params_dict)
                if not filtered_clusters:
                    logger.info('Clustering was unsuccesful for %s. Calibrating masses in the whole file.', filename)
                elif len(filtered_clusters) == 1:
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
                assigned_masks = stats.get_cluster_masks(filtered_clusters, clustering, df, to_fit, params_dict)
            for c, slice_, suffix, mask in zip(filtered_clusters, slices, suffixes, assigned_masks):
                # logger.debug('Slice size for cluster %s is: %s', c, slice_.size if slice_ is not None else None)
                to_fit, unit = stats.get_fittable_series(filtered, params_dict, slice_)
                popt = stats._gauss_fit_slice(to_fit, unit, filename, suffix, params_dict)

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
                c=clustering.labels_ if (params_dict['clustering'] and clustering) else None)
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
    df['check_composition'] = df[params_dict['peptides_column']].apply(lambda x: utils.check_composition(x, params_dict['labels']))
    return df.loc[df['check_composition']]


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
    params_dict = defaultdict()
    # data
    params_dict['decoy_prefix'] = params.get('data', 'decoy prefix')
    params_dict['FDR'] = params.getfloat('data', 'FDR')
    params_dict['labels'] = params.get('data', 'labels').strip().split()
    params_dict['rule'] = params.get('data', 'cleavage rule')
    # csv input
    params_dict['csv_delimiter'] = params.get('csv input', 'delimiter')
    params_dict['proteins_delimeter'] = params.get('csv input', 'proteins delimiter')
    params_dict['proteins_column'] = params.get('csv input', 'proteins column')
    params_dict['peptides_column'] = params.get('csv input', 'peptides column')
    params_dict['mass_shifts_column'] = params.get('csv input', 'mass shift column')
    params_dict['score_column'] = params.get('csv input', 'score column')
    params_dict['measured_mass_column'] = params.get('csv input', 'measured mass column')
    params_dict['calculated_mass_column'] = params.get('csv input', 'calculated mass column')
    params_dict['rt_column'] = params.get('csv input', 'retention time column')
    params_dict['next_aa_column'] = params.get('csv input', 'next aa column')
    params_dict['prev_aa_column'] = params.get('csv input', 'previous aa column')
    params_dict['spectrum_column'] = params.get('csv input', 'spectrum column')
    params_dict['charge_column'] = params.get('csv input', 'charge column')
    params_dict['score_ascending'] = params.getboolean('csv input', 'score ascending')

    # general
    params_dict['bin_width'] = params.getfloat('general', 'width of bin in histogram')
    params_dict['so_range'] = tuple(float(x) for x in params.get('general', 'open search range').split(','))
    params_dict['walking_window'] = params.getfloat('general', 'shifting window')
    params_dict['FDR_correction'] = params.getboolean('general', 'FDR correction')
    params_dict['processes'] = params.getint('general', 'processes')
    params_dict['zero_window'] = params.getfloat('general', 'zero peak window')
    params_dict['prec_acc'] = params.getfloat('general', 'mass shift tolerance')

    params_dict['zero bin tolerance'] = params.getfloat('general', 'zero shift mass tolerance')
    params_dict['zero min intensity'] = params.getfloat('general', 'zero shift minimum intensity')
    params_dict['min_peptides_for_mass_calibration'] = params.getint('general', 'minimum peptides for mass calibration')

    params_dict['specific_mass_shift_flag'] = params.getboolean('general', 'use specific mass shift window')
    params_dict['specific_window'] = [float(x) for x in params.get('general', 'specific mass shift window').split(',')]

    params_dict['figsize'] = tuple(float(x) for x in params.get('general', 'figure size in inches').split(','))
    params_dict['calibration'] = params.get('general', 'mass calibration')
    params_dict['artefact_thresh'] = params.getfloat('general', 'artefact detection threshold')

    #clustering
    params_dict['clustering'] = params.getboolean('clustering', 'use clustering')
    params_dict['eps_adjust'] = params.getfloat('clustering', 'dbscan eps factor')
    params_dict['min_samples'] = params.getfloat('clustering', 'dbscan min_samples')
    params_dict['clustered_pct_min'] = params.getfloat('clustering', 'total clustered peptide percentage minimum')
    params_dict['cluster_span_min'] = params.getfloat('clustering', 'cluster span percentage minimum')

    # fit
    params_dict['shift_error'] = params.getint('fit', 'shift error')
    params_dict['max_deviation_sigma'] = params.getfloat('fit', 'standard deviation threshold for sigma')
    params_dict['max_deviation_height'] = params.getfloat('fit', 'standard deviation threshold for height')
    params_dict['fit batch'] = params.getint('fit', 'batch')
    # localization
    params_dict['ion_types'] = tuple(params.get('localization', 'ion type').replace(' ', '').split(','))
    params_dict['frag_acc'] = params.getfloat('localization', 'fragment ion mass tolerance')
    params_dict['candidate threshold'] = params.getfloat('localization', 'frequency threshold')
    params_dict['min_spec_matched'] = params.getint('localization', 'minimum matched peaks')

    # modifications
    params_dict['variable_mods'] = params.getint('modifications', 'recommend variable modifications')
    params_dict['multiple_mods'] = params.getboolean('modifications', 'recommend multiple modifications on residue')
    params_dict['fix_mod_zero_thresh'] = params.getfloat('modifications', 'fixed modification intensity threshold')
    params_dict['min_fix_mod_pep_count_factor'] = params.getfloat('modifications', 'peptide count factor threshold')
    params_dict['recommend isotope threshold'] = params.getfloat('modifications', 'isotope error abundance threshold')
    params_dict['min_loc_count'] = params.getint('modifications', 'minimum localization count')
    return params_dict


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
        params_dict['fix_mod'] = utils.get_fix_modifications(args.pepxml[0])
        params_dict['enzyme'] = utils.get_specificity(args.pepxml[0])
    return params_dict


def table_path(dir, ms):
    return os.path.join(dir, ms + '.csv')


def save_df(ms, df, save_directory, params_dict):
    peptide = params_dict['peptides_column']
    spectrum = params_dict['spectrum_column']
    prev_aa = params_dict['prev_aa_column']
    next_aa = params_dict['next_aa_column']
    table = df[[peptide, spectrum]].copy()
    table[peptide] = df[prev_aa].str[0] + '.' + df[peptide] + '.' + df[next_aa].str[0]
    with open(table_path(save_directory, ms), 'w') as out:
        table.to_csv(out, index=False, sep='\t')


def save_peptides(data, save_directory, params_dict):
    for ms_label, (ms, df) in data.items():
        save_df(ms_label, df, save_directory, params_dict)