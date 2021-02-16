import jinja2
import logging
import os
import sys
import re
import warnings
import json
import pkg_resources
from datetime import datetime

import pandas as pd
from pyteomics import mass
from . import utils, stats

logger = logging.getLogger(__name__)

def format_unimod_repr(record_id):
    record = utils.UNIMOD[record_id]
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
        name = format_unimod_repr(record_id)
        if 'top isoform' in df:
            record = utils.UNIMOD[record_id]
            sites = {(group['site'], group['position']) for group in record['specificity']}
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


def _format_list(lst, sep1=', ', sep2=' or '):
    lst = list(lst)
    if not lst:
        return ''
    if len(lst) == 1:
        return lst[0]
    *most, last = lst
    return sep1.join(most) + sep2 + last


def get_artefact_interpretations(row, mass_shift_data_dict, params_dict):
    out = []
    aa_mass = mass.std_aa_mass.copy()
    aa_mass.update(params_dict['fix_mod'])
    enz = params_dict.get('enzyme')
    df = mass_shift_data_dict[row.name][1]
    peps = df[params_dict['peptides_column']]
    match_aa = set()

    for aa, m in aa_mass.items():
        if abs(abs(row['mass shift']) - m) < params_dict['frag_acc']:
            match_aa.add(aa)
    if not match_aa:
        return []

    if enz:
        cut = set(enz['cut']) & match_aa
        nocut = set(enz.get('nocut', []))
    else:
        cut, nocut = None, set()

    explained = False
    if row['mass shift'] < 0:
        # this can be a loss of any terminal amino acid, or ...
        # an artefact of open search, where the peptide is actually unmodified.
        # in the latter case the amino acid should be an enzyme cleavage site
        if cut:
            # possible artefact
            if enz['sense'] == 'C':
                pct = (
                    (peps.str[0].isin(cut) & ~peps.str[1].isin(nocut)) |  # extra amino acid at N-term
                    peps.str[-2].isin(cut)   # extra amino acid at C-term
                    ).sum() / df.shape[0]
            elif enz['sense'] == 'N':
                pct = (
                    peps.str[1].isin(cut) |
                    (peps.str[-1].isin(cut) & ~peps.str[-2].isin(nocut))
                    ).sum() / df.shape[0]
            else:
                logger.critical('Unknown value of sense in specificity: %s', enz)
                sys.exit(1)

            logger.debug('%.1f%% of peptides in %s %s with %s.',
                pct * 100, row.name, ('start', 'end')[enz['sense'] == 'N'], _format_list(cut))
            if pct > params_dict['artefact_thresh']:
                out.append('Search artefact: unmodified peptides with extra {} at {}-terminus ({:.0%} match)'.format(
                    _format_list(cut), 'CN'[enz['sense'] == 'C'], pct))
                explained = True
            else:
                logger.debug('Not enough peptide support search artefact interpretation.')
        if not explained:
            out.append('Loss of ' + _format_list(match_aa))
        if not enz:
            out[-1] += ' or an open search artefact'
    else:
        # this may be a missed cleavage
        if cut:
            keys = [params_dict['prev_aa_column'], params_dict['next_aa_column']]
            pct = df[keys].apply(
                lambda row: bool(cut.intersection(row[keys[0]] + row[keys[1]])), axis=1).sum() / df.shape[0]
            logger.debug('%.1f%% of peptides in %s have %s as neighbor amino acid.',
                pct * 100, row.name, _format_list(cut))
            if pct > params_dict['artefact_thresh']:
                out.append('Possible miscleavage (extra {} at terminus)'.format(_format_list(cut)))
            else:
                logger.debug('Not enough peptide support search artefact interpretation.')
    return out


def format_info(row, table, mass_shift_data_dict, params_dict):
    options = get_artefact_interpretations(row, mass_shift_data_dict, params_dict)
    options.extend(format_unimod_info(row, mass_shift_data_dict[row.name][1], params_dict))
    if row['isotope index']:
        options.append('isotope of {}'.format(get_label(table, row['isotope index'])))
    if isinstance(row['sum of mass shifts'], list):
        options.extend('{}{}'.format(get_label(table, s1), get_label(table, s2, True)) for s1, s2 in row['sum of mass shifts'])
    return ', '.join(options)

def html_format_isoform(isoform):
    out = re.sub(r'([A-Z]\[[+-]?[0-9]+\])', r'<span class="loc">\1</span>', isoform)
    out = re.sub(r'^([A-Z])\.', r'<span class="nterm"><span class="prev_aa">\1</span>.</span>', out)
    out = re.sub(r'\.([A-Z])$', r'<span class="cterm">.<span class="next_aa">\1</span></span>', out)
    return out


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
            ).format({
                mslabel: '<a href="#">{}</a>'.format(utils.MASS_FORMAT).format,
                '# peptides in bin': '<a href="#">{}</a>'.format}).bar(subset='# peptides in bin', color=stats.cc[2]).render(
            uuid="aa_stat_table")

    peptide_tables = []
    for ms in table.index:
        df = mass_shift_data_dict[ms][1]
        if 'localization score' in df and df['localization score'].notna().any():
            df = df.sort_values(['localization score'], ascending=False).loc[:,
                ['top isoform', 'localization score', params_dict['spectrum_column']]]
            df['localization score'] = df['localization score'].astype(float)
        else:
            dfc = df[[params_dict['peptides_column'], params_dict['spectrum_column']]].copy()
            dfc[params_dict['peptides_column']] = (
                df[params_dict['prev_aa_column']].str[0] + '.' + df[params_dict['peptides_column']] + '.' + df[params_dict['next_aa_column']].str[0])
            df = dfc
        peptide_tables.append(df.to_html(
            table_id='peptides_' + ms, classes=('peptide_table',), index=False, escape=False, na_rep='',
            formatters={
                'top isoform': html_format_isoform,
                params_dict['peptides_column']: html_format_isoform,
                'localization score': '{:.2f}'.format}))

    if params_dict['fix_mod']:
        d = params_dict['fix_mod'].copy()
        d = utils.masses_to_mods(d)
        fixmod = pd.DataFrame.from_dict(d, orient='index', columns=['value']).T.style.set_caption(
            'Configured, fixed').format(utils.MASS_FORMAT).render(uuid="set_fix_mod_table")
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
                utils.mass_format(-ms_labels[recommended_vmods[i][1]]), recommended_vmods[i][0]) for i in opposite])
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
