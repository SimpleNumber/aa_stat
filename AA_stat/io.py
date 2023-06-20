import matplotlib
matplotlib.use('Agg')
import pylab as plt
from matplotlib.backends.backend_pdf import PdfPages

import sys
import ast
import os
import glob
from configparser import ConfigParser
import multiprocessing as mp
from collections import defaultdict
import logging
import re
import numpy as np
import pandas as pd
from pyteomics import pepxml, mgf, mzml
from . import utils, stats

AA_STAT_PARAMS_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default.cfg')
logger = logging.getLogger(__name__)


def sanitize_df(df, params_dict):
    # drop unneeded columns
    column_keys = ['proteins_column', 'peptides_column', 'mass_shifts_column', 'score_column', 'measured_mass_column',
    'calculated_mass_column', 'rt_column', 'next_aa_column', 'prev_aa_column', 'spectrum_column', 'charge_column', 'mods_column']
    needed = {params_dict[k] for k in column_keys}
    to_drop = [c for c in df.columns if c not in needed]
    old_size = df.shape[1]
    df.drop(to_drop, axis=1, inplace=True)
    logger.debug('Kept %d and dropped %d out of %d initial columns.', df.shape[1], len(to_drop), old_size)

    # TODO: simplify and sanitize columns here
    return df


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
    pp = PdfPages(os.path.join(params_dict['output directory'], os.path.basename(filename) + '.clustering.pdf'))
    window = params_dict['zero_window']
    zero_bin = 0
    shifts = params_dict['mass_shifts_column']
    if not params_dict['decoy_prefix']:
        isdddict = {}
        for prefix in params_dict['decoy_prefix_list']:
            is_decoy = df[params_dict['proteins_column']].apply(
                lambda s: all(x.startswith(prefix) for x in s))
            isd = is_decoy.sum()
            logger.debug('Trying prefix %s for %s... Found %d decoys.', prefix, filename, isd)
            isdddict[prefix] = isd
        prefix = max(isdddict, key=isdddict.get)
        logger.debug('Selected prefix %s for file %s (%d decoys)', prefix, filename, isdddict[prefix])
    else:
        prefix = params_dict['decoy_prefix']

    df['is_decoy'] = df[params_dict['proteins_column']].apply(lambda s: all(x.startswith(prefix) for x in s))

    if not df['is_decoy'].sum():
        logger.error('No decoy IDs found in %s.', filename)
        if not params_dict['decoy_prefix']:
            logger.error('Configured decoy prefixes are: %s. Check you files or config.',
                ', '.join(params_dict['decoy_prefix_list']))
        else:
            logger.error('Configured decoy prefix is: %s. Check your files or config.', prefix)
        return
    ms, filtered = utils.fdr_filter_mass_shift([None, zero_bin, window], df, params_dict)
    n = filtered.shape[0]
    logger.debug('%d filtered peptides near zero.', n)
    df[shifts] = utils.choose_correct_massdiff(
        df[shifts],
        df[params_dict['measured_mass_column']] - df[params_dict['calculated_mass_column']], params_dict)
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
                clustering = stats.clusters(filtered, to_fit, unit, filename, params_dict, pp)
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
                    f = plt.figure()
                    for i in filtered_clusters:
                        plt.hist(filtered.loc[to_fit.index].loc[clustering.labels_ == i, shifts], label=i, alpha=0.2, bins=25, density=True)
                    plt.xlabel(shifts)
                    plt.title('Before correction')
                    plt.legend()
                    pp.savefig(f)
                    plt.close()
            else:
                filtered_clusters = None

            if not filtered_clusters:
                slices = [None]
                titles = ['Whole file']
                assigned_masks = [slice(None)]
                filtered_clusters = ['<all>']
            else:
                slices, titles = [], []
                for i in filtered_clusters:
                    slices.append(clustering.labels_ == i)
                    titles.append('Cluster {}'.format(i))
                assigned_masks = stats.get_cluster_masks(filtered_clusters, clustering, df, to_fit, params_dict)
            for c, slice_, title, mask in zip(filtered_clusters, slices, titles, assigned_masks):
                # logger.debug('Slice size for cluster %s is: %s', c, slice_.size if slice_ is not None else None)
                to_fit, unit = stats.get_fittable_series(filtered, params_dict, slice_)
                popt = stats._gauss_fit_slice(to_fit, unit, filename, title, params_dict, pp)

                if unit == 'Da':
                    shift_copy.loc[mask] -= popt[1]
                elif unit == 'ppm':
                    shift_copy.loc[mask] -= popt[1] * df[params_dict['calculated_mass_column']] / 1e6
                else:
                    freq_measured = 1e6 / np.sqrt(utils.measured_mz_series(df.loc[mask], params_dict)) - popt[1]
                    mass_corrected = (((1e6 / freq_measured) ** 2) * df.loc[mask, params_dict['charge_column']] -
                        utils.H * df.loc[mask, params_dict['charge_column']])
                    correction = mass_corrected - df.loc[mask, params_dict['measured_mass_column']]
                    logger.debug('Average systematic mass shift for cluster %s: %f', c, -correction.mean())
                    shift_copy.loc[mask] += correction

            # corrected mass shifts are written back here
            df[shifts] = shift_copy
            filtered[shifts] = df.loc[filtered.index, shifts]

            f = plt.figure()
            floc = filtered.loc[old_shifts.abs() < params_dict['zero_window']]
            sc = plt.scatter(floc[shifts], floc[params_dict['rt_column']],
                c=clustering.labels_ if (params_dict['clustering'] and clustering) else 'k')
            if params_dict['clustering'] and clustering:
                plt.legend(*sc.legend_elements(), title='Clusters')
            plt.xlabel(shifts)
            plt.ylabel(params_dict['rt_column'])
            plt.title('After correction')
            pp.savefig(f)
            plt.close()
            if filtered_clusters != ['<all>']:
                f = plt.figure()
                for i in filtered_clusters:
                    plt.hist(floc.loc[clustering.labels_ == i, shifts], label=i, alpha=0.2, bins=25, density=True)
                plt.xlabel(shifts)
                plt.legend()
                pp.savefig(f)
                plt.close()
    pp.close()
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
    df = pepxml.DataFrame(fname, read_schema=False)
    return preprocess_df(sanitize_df(df, params_dict), fname, params_dict)


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
    df[params_dict['mods_column']] = df[params_dict['mods_column']].apply(ast.literal_eval)
    protein = params_dict['proteins_column']
    prev = params_dict['prev_aa_column']
    next_ = params_dict['next_aa_column']
    for c in [protein, prev, next_]:
        if (df[c].str[0] == '[').all() and (df[c].str[-1] == ']').all():
            df[c] = df[c].apply(ast.literal_eval)
        else:
            df[c] = df[c].str.split(params_dict['proteins_delimeter'])
    return preprocess_df(sanitize_df(df, params_dict), fname, params_dict)


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
        spec_filenames = getattr(args, ftype)
        if spec_filenames:
            break
    else:
        return {}
    for inp in [args.pepxml, args.csv]:
        if inp:
            break
    if len(inp) != len(spec_filenames):
        logger.critical('Numbers of input files and spectrum files do not match (%d and %d).', len(inp), len(spec_filenames))
        sys.exit(1)

    for i, filename in zip(inp, spec_filenames):
        name = os.path.splitext(os.path.basename(i))[0]
        out_dict[name] = reader(filename)
    return out_dict


def read_input(args, params_dict):
    """
    Reads open search output, assembles all data in one DataFrame.

    """
    logger.info('Reading input files...')
    readers = {
        'pepxml': read_pepxml,
        'csv': read_csv,
    }
    shifts = params_dict['mass_shifts_column']
    nproc = params_dict['processes']
    if nproc == 1:
        dfs = []
        logger.debug('Reading files in one process.')
        for ftype, reader in readers.items():
            filenames = getattr(args, ftype)
            logger.debug('Filenames [%s]: %s', ftype, filenames)
            if filenames:
                for filename in filenames:
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
        results = []
        for ftype, reader in readers.items():
            filenames = getattr(args, ftype)
            logger.debug('Filenames [%s]: %s', ftype, filenames)
            if filenames:
                for filename in filenames:
                    results.append(pool.apply_async(reader, args=(filename, params_dict)))
        dfs = [r.get() for r in results]
        pool.close()
        pool.join()
    if any(x is None for x in dfs):
        logger.critical('There were errors when reading input.')
        return
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
    params_dict['decoy_prefix_list'] = re.split(r',\s*', params.get('data', 'decoy prefix list'))
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
    params_dict['mods_column'] = params.get('csv input', 'modifications column')
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
    params_dict['html_truncate'] = params.getint('general', 'html info truncation length')

    #clustering
    params_dict['clustering'] = params.getboolean('clustering', 'use clustering')
    params_dict['eps_adjust'] = params.getfloat('clustering', 'dbscan eps factor')
    params_dict['min_samples'] = params.getint('clustering', 'dbscan min_samples')
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
    params_dict['force_term_loc'] = params.getboolean('localization', 'always try terminal localization')
    params_dict['use_all_loc'] = params.getboolean('localization', 'try all localizations')

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
    logger.debug('Received args: %s', args)
    fname = args.params
    outdir = args.dir
    params = read_config_file(fname)
    params_dict = get_parameters(params)
    set_additional_params(params_dict)
    params_dict['output directory'] = outdir
    if args.pepxml:
        fmod, vmod = utils.get_fix_var_modifications(args.pepxml[0], params_dict['labels'])
        params_dict['fix_mod'] = fmod
        params_dict['var_mod'] = utils.format_grouped_keys(utils.group_terminal(vmod), params_dict)
        params_dict['enzyme'] = utils.get_specificity(args.pepxml[0])
    else:
        if args.fmods:
            params_dict['fix_mod'] = ast.literal_eval(args.fmods)
        else:
            params_dict['fix_mod'] = {}
            logger.info('No fixed modifications specified. Use --fmods to configure them.')
        if args.vmods:
            params_dict['var_mod'] = ast.literal_eval(args.vmods)
        else:
            params_dict['var_mod'] = {}
            logger.info('No variable modifications specified. Use --vmods to configure them.')
        if args.enzyme:
            params_dict['enzyme'] = ast.literal_eval(args.enzyme)
        else:
            logger.info('Enyzme not specified. Use --enzyme to configure.')
            params_dict['enzyme'] = None
    return params_dict


_format_globs = {
    'pepxml': ['*.pepXML', '*.pep.xml'],
    'csv': ['*.csv'],
    'mzml': ['*.mzML'],
    'mgf': ['*.mgf'],
}

def resolve_filenames(args):
    for fformat, gs in _format_globs.items():
        value = getattr(args, fformat)
        if value:
            logger.debug('Received %s list: %s', fformat, value)
            out = []
            for val in value:
                if os.path.isdir(val):
                    for g in gs:
                        files = glob.glob(os.path.join(val, g))
                        logger.debug('Found %d files for glob %s in %s', len(files), g, val)
                        out.extend(files)
                else:
                    out.append(val)
            logger.debug('Final %s list: %s', fformat, out)
            setattr(args, fformat, out)


def table_path(dir, ms):
    return os.path.join(dir, ms + '.csv')


def save_df(ms, df, save_directory, params_dict):
    peptide = params_dict['peptides_column']
    spectrum = params_dict['spectrum_column']
    prev_aa = params_dict['prev_aa_column']
    next_aa = params_dict['next_aa_column']
    table = df[[peptide, spectrum]].copy()
    peptide1 = df.apply(utils.get_column_with_mods, axis=1, args=(params_dict,))
    table[peptide] = df[prev_aa].str[0] + '.' + peptide1 + '.' + df[next_aa].str[0]
    with open(table_path(save_directory, ms), 'w') as out:
        table.to_csv(out, index=False, sep='\t')


def save_peptides(data, save_directory, params_dict):
    for ms_label, (ms, df) in data.items():
        save_df(ms_label, df, save_directory, params_dict)
