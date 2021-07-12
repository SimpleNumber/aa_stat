#!/usr/bin/env python

from setuptools import setup, find_packages
import os

# from https://packaging.python.org/guides/single-sourcing-package-version/

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name                 = 'AA_stat',
    version              = get_version('AA_stat/version.py'),
    description          = '''A utility for validation of peptide identification results in proteomics using amino acid counting.''',
    long_description     = (''.join(open('README.MD').readlines())),
    long_description_content_type="text/markdown",
    author               = 'Julia Bubis & Lev Levitsky',
    author_email         = 'julia.bubis@gmail.com',
    install_requires     = ['pyteomics>4.4.1', 'pandas', 'seaborn', 'scipy', 'numpy', 'lxml', 'jinja2', 'scikit-learn'],
    classifiers          = ['Intended Audience :: Science/Research',
                            'Programming Language :: Python :: 3',
                            'Topic :: Scientific/Engineering :: Bio-Informatics',
                            'Topic :: Scientific/Engineering :: Chemistry',
                            'Topic :: Scientific/Engineering :: Physics'],
    license              = 'License :: OSI Approved :: Apache Software License',
    packages             = find_packages(),
    package_data         = {'AA_stat': ['report.template', 'open_search.params', 'default.cfg', 'unimod.xml']},
    entry_points         = {'console_scripts': ['AA_stat=AA_stat.main:main', 'AA_search=AA_stat.aa_search:main',
                                                'AA_stat_GUI= AA_stat.gui.gui:main']}
    )
