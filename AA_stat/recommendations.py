import logging
from collections import defaultdict
from . import utils

logger = logging.getLogger(__name__)


def get_fix_mod_from_l10n(mslabel, locmod_df):
    l10n = locmod_df.at[mslabel, 'localization']
    logger.debug('Localizations for %s: %s', mslabel, l10n)
    if l10n:
        l10n.pop('non-localized', None)
        top_loc = max(l10n, key=l10n.get)
        logger.debug('Top localization label for %s: %s', mslabel, top_loc)
        return top_loc


def get_fixed_mod_raw(aa, data_dict, choices=None):
    dist_aa = []
    for ms, v in data_dict.items():
        if choices is None or ms in choices:
            dist_aa.append([v[0], v[1]['peptide'].apply(lambda x: x.count(aa)).sum()])
    utils.internal('Counts for %s: %s', aa, dist_aa)
    top_shift = max(dist_aa, key=lambda tup: tup[1])
    return utils.mass_format(top_shift[0])


def determine_fixed_mods_nonzero(reference, locmod_df, data_dict):
    """Determine fixed modifications in case the reference shift is not at zero.
    Needs localization.
    """
    utils.internal('Localizations for %s: %s', reference, locmod_df.at[reference, 'localization'])
    loc = get_fix_mod_from_l10n(reference, locmod_df)
    label = reference
    data_dict = data_dict.copy()
    while loc is None:
        del data_dict[label]
        label = max(data_dict, key=lambda k: data_dict[k][1].shape[0])
        loc = get_fix_mod_from_l10n(label, locmod_df)
        logger.debug('No luck. Trying %s. Got %s', label, loc)
        if not data_dict:
            break
    return loc


def determine_fixed_mods_zero(aastat_result, data_dict, params_dict):
    """Determine fixed modifications in case the reference shift is at zero.
    Does not need localization.
    """
    fix_mod_zero_thresh = params_dict['fix_mod_zero_thresh']
    min_fix_mod_pep_count_factor = params_dict['min_fix_mod_pep_count_factor']

    fix_mod_dict = {}
    reference = utils.mass_format(0)
    aa_rel = aastat_result[reference][2]
    utils.internal('aa_rel:\n%s', aa_rel)
    candidates = aa_rel[aa_rel < fix_mod_zero_thresh].index
    logger.debug('Fixed mod candidates: %s', candidates)
    for i in candidates:
        candidate_label = get_fixed_mod_raw(i, data_dict)
        if candidate_label != reference:
            # number of peptides with `i` at shift `candidate label` must be higher than ...
            count_cand = data_dict[candidate_label][1][params_dict['peptides_column']].str.contains(i).sum()
            # number of peptides with `i` at shift `reference` by a factor of `min_fix_mod_pep_count_factor`
            count_ref = data_dict[reference][1][params_dict['peptides_column']].str.contains(i).sum()
            # peptide count at candidate shift over # of peptides at reference
            est_ratio = count_cand / len(data_dict[reference][1])
            logger.debug('Peptides with %s: ~%d at %s, ~%d at %s. Estimated pct: %f',
                i, count_ref, reference, count_cand, candidate_label, est_ratio)
            if aastat_result[candidate_label][2][i] > fix_mod_zero_thresh and (
                    est_ratio * 100 > fix_mod_zero_thresh * min_fix_mod_pep_count_factor):
                fix_mod_dict[i] = candidate_label
            else:
                logger.debug('Could not find %s anywhere. Can\'t fix.', i)
        else:
            logger.debug('Reference shift is the best for %s.', i)
    return fix_mod_dict


def determine_fixed_mods(aastat_result, aastat_df, locmod_df, data_dict, params_dict):
    reference = aastat_df.loc[aastat_df['is reference']].index[0]
    if reference == utils.mass_format(0):
        logger.info('Reference bin is at zero shift.')
        fix_mod_dict = determine_fixed_mods_zero(aastat_result, data_dict, params_dict)
    else:
        if locmod_df is None:
            logger.warning('No localization data. '
                'Cannot determine fixed modifications when reference mass shift is non-zero.')
            return {}
        logger.info('Reference bin is at %s. Looking for fixed modification to compensate.', reference)
        loc = determine_fixed_mods_nonzero(reference, locmod_df, data_dict)
        if loc:
            aa, shift = utils.parse_l10n_site(loc)
            fix_mod_dict = {aa: shift}
        else:
            logger.info('No localizations. Stopping.')

    return fix_mod_dict


def recommend_isotope_error(aastat_df, locmod_df, params_dict):
    reference = aastat_df.loc[aastat_df['is reference']].index[0]
    ref_peptides = locmod_df.at[reference, '# peptides in bin']
    logger.debug('%d peptides at reference %s', ref_peptides, reference)
    ref_isotopes = []
    label = reference
    while label:
        label = utils.get_isotope_shift(label, locmod_df)
        ref_isotopes.append(label)
    ref_isotopes.pop()

    i = 0
    for i, label in enumerate(ref_isotopes, 1):
        peps = locmod_df.at[label, '# peptides in bin']
        logger.debug('%d peptides at %s.', peps, label)
        if peps * 100 / ref_peptides < params_dict['recommend isotope threshold']:
            return i - 1
    return i


def recalculate_counts(aa, ms, mods_and_counts, data_dict):
    mods_and_counts[aa].pop(ms, None)
    for i, row in data_dict[ms][1].iterrows():
        seq = row['top isoform'].split('.')[1]
        if row['top_terms'] is not None and ms in row['top_terms']:
            if aa == 'N-term' and seq[1] == '[':
                utils.internal('Reducing count of %s for %s (%s)', seq[0], seq, aa)
                if mods_and_counts[seq[0]].get(ms, 0) > 0:
                    mods_and_counts[seq[0]][ms] -= 1
            elif aa == 'C-term' and seq[-1] == ']':
                res = seq.split('[')[0][-1]
                utils.internal('Reducing count of %s for %s (%s)', res, seq, aa)
                if mods_and_counts[res].get(ms, 0) > 0:
                    mods_and_counts[res][ms] -= 1
            elif seq[:2] == aa + '[':
                utils.internal('Reducing count of N-term for %s', seq)
                if mods_and_counts['N-term'].get(ms, 0) > 0:
                    mods_and_counts['N-term'][ms] -= 1
            elif seq[-1] == ']' and seq.split('[')[0][-1] == aa:
                utils.internal('Reducing count of C-term for %s', seq)
                if mods_and_counts['C-term'].get(ms, 0) > 0:
                    mods_and_counts['C-term'][ms] -= 1


def recalculate_with_isotopes(aa, ms, isotope_rec, mods_and_counts, data_dict, locmod_df):
    logger.debug('Recalculating counts for %s @ %s', aa, ms)
    recalculate_counts(aa, ms, mods_and_counts, data_dict)
    i = 0
    while i < isotope_rec:
        label = utils.get_isotope_shift(ms, locmod_df)
        if label:
            logger.debug('Recalculating %s counts for isotope shift %s', aa, label)
            recalculate_counts(aa, label, mods_and_counts, data_dict)
            i += 1
        else:
            break


def recalculate_varmods(data_dict, mods_and_counts, params_dict):
    for ms in data_dict:
        shift, df = data_dict[ms]
        for i, row in df.iterrows():
            if row['top_terms'] is not None and ms in row['top_terms']:
                peptide = row[params_dict['peptides_column']]
                if ']{' in row['top isoform']:  # localization and enabled variable modification on the same residue
                    # this should count towards sum of these shifts, not the localized one
                    pos = row['loc_position'][0]
                    mods = utils.get_var_mods(row, params_dict)
                    utils.internal('%s: extracting %d from %s', row['top isoform'], pos, mods)
                    if pos in mods:
                        vm = mods[pos]
                    elif pos == 1:
                        vm = mods[0]
                    elif pos == len(peptide):
                        vm = mods[pos + 1]
                    else:
                        raise KeyError()
                    aa = peptide[pos - 1]
                    if mods_and_counts[aa].get(ms, 0) > 0:
                        utils.internal('Reducing count of %s at %s', aa, ms)
                        mods_and_counts[aa][ms] -= 1
                    if pos == 1 and mods_and_counts['N-term'].get(ms, 0) > 0:
                        mods_and_counts['N-term'][ms] -= 1
                        utils.internal('Reducing count of N-term at %s', ms)
                    if pos == len(peptide) and mods_and_counts['C-term'].get(ms, 0) > 0:
                        utils.internal('Reducing count of C-term at %s', ms)
                        mods_and_counts['C-term'][ms] -= 1
                    sum_ms = utils.find_mass_shift(vm + shift, data_dict, params_dict['prec_acc'])
                    if sum_ms:
                        mods_and_counts[aa][sum_ms] = mods_and_counts[aa].get(sum_ms, 0) + 1
                        utils.internal('Increasing count of %s at %s', aa, sum_ms)
                        if pos == 1:
                            utils.internal('Increasing count of N-term at %s', sum_ms)
                            mods_and_counts['N-term'][sum_ms] = mods_and_counts['N-term'].get(sum_ms, 0) + 1
                        if pos == len(peptide):
                            utils.internal('Increasing count of C-term at %s', sum_ms)
                            mods_and_counts['C-term'][sum_ms] = mods_and_counts['C-term'].get(sum_ms, 0) + 1


def determine_var_mods(aastat_result, aastat_df, locmod_df, data_dict, params_dict, recommended_fix_mods=None):
    if locmod_df is None:
        logger.info('Cannot recommend variable modifications without localization.')
        return {}
    var_mods = []
    recommended = set()
    multiple = params_dict['multiple_mods']
    if multiple:
        logger.info('Recommending multiple modifications on same residue.')
    else:
        logger.info('Recommending one modification per residue.')
    isotope_rec = recommend_isotope_error(aastat_df, locmod_df, params_dict)
    logger.info('Recommended isotope mass error: %d.', isotope_rec)
    if isotope_rec:
        var_mods.append(('isotope error', isotope_rec))
    reference = aastat_df.loc[aastat_df['is reference']].index[0]
    mods_and_counts = defaultdict(dict)  # dict of AA: shift label: count
    for shift in data_dict:
        if shift == reference:
            continue
        l10n = locmod_df.at[shift, 'localization']
        for k, count in l10n.items():
            if k == 'non-localized':
                continue
            aa, locshift = utils.parse_l10n_site(k)
            if locshift == shift:
                mods_and_counts[aa][shift] = count
    logger.debug('Without isotopes, localization counts are:')

    for k, d in mods_and_counts.items():
        logger.debug('%s: %s', k, d)
    if isotope_rec:
        for aa, dcounts in mods_and_counts.items():
            for shift, count in list(dcounts.items()):
                i = 0
                while i < isotope_rec:
                    label = utils.get_isotope_shift(shift, locmod_df)
                    if label:
                        dcounts[shift] = dcounts.get(shift, 0) + mods_and_counts[aa].get(label, 0)
                        # dcounts.pop(label, None)
                        i += 1
                    else:
                        break
        i = 0
        shift = reference
        while i < isotope_rec:
            label = utils.get_isotope_shift(shift, locmod_df)
            if label:
                logger.debug('Removing all counts for isotope shift %s', label)
                for aa, dcounts in mods_and_counts.items():
                    dcounts[label] = 0
                i += 1
            else:
                break
        logger.debug('With isotopes, localization counts are:')
        for k, d in mods_and_counts.items():
            logger.debug('%s: %s', k, d)

    if recommended_fix_mods:
        logger.debug('Subtracting counts for fixed mods.')
        for aa, shift in recommended_fix_mods.items():
            recalculate_with_isotopes(aa, shift, isotope_rec, mods_and_counts, data_dict, locmod_df)

    if params_dict['var_mod']:
        if not multiple:
            logger.info('Multiple variable modifications are disabled, not recommending {} for variable modifications.'.format(
                utils.format_list(params_dict['var_mod'])))
            for aa, shift in params_dict['var_mod'].items():
                logger.debug('Removing all counts for %s.', aa)
                for sh in mods_and_counts[aa]:
                    mods_and_counts[aa][sh] = 0

        logger.debug('Subtracting counts for variable mods.')
        recalculate_varmods(data_dict, mods_and_counts, params_dict)

    for i in range(params_dict['variable_mods']):
        logger.debug('Choosing variable modification %d. Counts are:', i + 1)
        for k, d in mods_and_counts.items():
            logger.debug('%s: %s', k, d)
            aa_shifts = {aa: max(dcounts, key=dcounts.get) for aa, dcounts in mods_and_counts.items() if dcounts}
        if mods_and_counts:
            aa_counts = {aa: mods_and_counts[aa][shift] for aa, shift in aa_shifts.items()}
            logger.debug('Best localization counts: %s', aa_shifts)
            logger.debug('Values: %s', aa_counts)
            if aa_shifts:
                top_aa = max(aa_shifts, key=aa_counts.get)
                top_shift = aa_shifts[top_aa]
                top_count = aa_counts[top_aa]
                if top_count < params_dict['min_loc_count']:
                    logger.debug('Localization count too small (%d), stopping.', top_count)
                    break
                recommended.add(top_aa)
                var_mods.append((top_aa, top_shift))
                logger.debug('Chose %s @ %s.', top_shift, top_aa)
                recalculate_with_isotopes(top_aa, top_shift, isotope_rec, mods_and_counts, data_dict, locmod_df)
                if not multiple:
                    logger.debug('Removing all counts for %s.', top_aa)
                    for sh in mods_and_counts[top_aa]:
                        mods_and_counts[top_aa][sh] = 0
    return var_mods
