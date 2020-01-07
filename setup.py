#!/usr/bin/env python

'''
setup.py file for aa_stats
'''

from setuptools import setup, find_packages

setup(
    name                 = 'AA_stat',
    version              = '1.0.6',
    description          = '''A utility for validation of peptide identification results in proteomics using amino acid counting.''',
    long_description     = (''.join(open('README.MD').readlines())),
    author               = 'Julia Bubis & Lev Levitsky',
    author_email         = 'julia.bubis@gmail.com',
    install_requires     = ['pyteomics', 'pandas', 'seaborn', 'scipy', 'numpy', 'lxml', 'jinja2'],
    classifiers          = ['Intended Audience :: Science/Research',
                            'Programming Language :: Python :: 2.7',
                            'Programming Language :: Python :: 3',
                            'Topic :: Education',
                            'Topic :: Scientific/Engineering :: Bio-Informatics',
                            'Topic :: Scientific/Engineering :: Chemistry',
                            'Topic :: Scientific/Engineering :: Physics',
                            'Topic :: Software Development :: Libraries'],
    license              = 'License :: OSI Approved :: Apache Software License',
    packages             = find_packages(),
    package_data         = {'AA_stat': ['report.template']},
    entry_points         = {'console_scripts': ['AA_stat=AA_stat.main:main']}
    )
