import unittest
import numpy as np
from .localization import get_theor_spectrum
from .AA_stat import AA_stat
from . import utils, io
import argparse
import logging
from pyteomics import mass
import os

class AAstatTheorSpectrumTest(unittest.TestCase):

    def setUp(self):
        self.spec_PEPTIDE = {
            ('b', 1): np.array([98.06003647, 227.10262647, 324.15538647, 425.20306647, 538.28712647, 653.31406647]),
            ('y', 1): np.array([148.06043115, 263.08737115, 376.17143115, 477.21911115, 574.27187115, 703.31446115]),
            ('b', 2): np.array([49.53365647, 114.05495147, 162.58133147, 213.10517147, 269.64720147, 327.16067147]),
            ('y', 2): np.array([74.53385381, 132.04732381, 188.58935381, 239.11319381, 287.63957381, 352.16086881])
        }
        self.spec_int_PEPTIDE = {
            'b': {4953, 9806, 11405, 16258, 21310, 22710, 26964, 32415, 32716, 42520, 53828, 65331},
            'y': {26308, 23911, 47721, 28763, 18858, 35216, 37617, 57427, 13204, 14806, 70331, 7453}
        }
        self.spec_mPEPTIDE = {
            ('b', 1): np.array([114.05495147, 243.09754147, 340.15030147, 441.19798147, 554.28204147, 669.30898147]),
            ('y', 1): np.array([148.06043115, 263.08737115, 376.17143115, 477.21911115, 574.27187115, 703.31446115]),
            ('b', 2): np.array([57.53111397, 122.05240897, 170.57878897, 221.10262897, 277.64465897, 335.15812897]),
            ('y', 2): np.array([74.53385381, 132.04732381, 188.58935381, 239.11319381, 287.63957381, 352.16086881])
        }
        self.spec_int_mPEPTIDE = {
            'b': {17057, 55428, 33515, 11405, 12205, 66930, 27764, 24309, 44119, 5753, 22110, 34015},
            'y': {7453, 13204, 14806, 18858, 23911, 26308, 28763, 35216, 37617, 47721, 57427, 70331}
        }
        self.spec_PEPTIDE_cz = {
            ('c', 1): np.array([115.08658557, 244.12917557, 341.18193557, 442.22961557, 555.31367557, 670.34061557]),
            ('z', 1): np.array([131.03388205, 246.06082205, 359.14488205, 460.19256205, 557.24532205, 686.28791205]),
            ('c', 2): np.array([58.04693102, 122.56822602, 171.09460602, 221.61844602, 278.16047602, 335.67394602]),
            ('z', 2): np.array([66.02057926, 123.53404926, 180.07607926, 230.59991926, 279.12629926, 343.64759426])
        }
        self.spec_int_PEPTIDE_cz = {
            'c': {12256, 34118, 27816, 55531, 5804, 22161, 11508, 17109, 67034, 24412, 44222, 33567},
            'z': {12353, 46019, 27912, 35914, 6602, 55724, 13103, 23059, 68628, 18007, 34364, 24606}
        }

    def _compare_spectra(self, spec, spec_int, spec_true, spec_int_true, eps=1e-6):
        spec = {k: sorted(v) for k, v in spec.items()}
        self.assertEqual(spec.keys(), spec_true.keys())
        for k in spec:
            spec[k].sort()
            self.assertTrue(np.allclose(spec[k], spec_true[k], atol=eps))

        self.assertEqual(spec_int, spec_int_true)

    def test_theor_spec_PEPTIDE(self):
        spec, spec_int = get_theor_spectrum(list('PEPTIDE'), 0.01, ion_types=('b', 'y'), maxcharge=2)
        self._compare_spectra(spec, spec_int, self.spec_PEPTIDE, self.spec_int_PEPTIDE)

    def test_theor_spec_mPEPTIDE(self):
        custom_mass = mass.std_aa_mass.copy()
        custom_mass['mP'] = mass.std_aa_mass['P'] + 15.994915
        spec, spec_int = get_theor_spectrum(['mP'] + list('EPTIDE'), 0.01, ion_types=('b', 'y'), maxcharge=2,
                                            aa_mass=custom_mass)
        self._compare_spectra(spec, spec_int, self.spec_mPEPTIDE, self.spec_int_mPEPTIDE)

    def test_theor_spec_PEPTIDE_cz(self):
        spec, spec_int = get_theor_spectrum(list('PEPTIDE'), 0.01, ion_types=('c', 'z'), maxcharge=2)
        self._compare_spectra(spec, spec_int, self.spec_PEPTIDE_cz, self.spec_int_PEPTIDE_cz)

    def test_theor_spec_termPEPTIDE(self):
        MOD = 42.12
        acc = 0.01
        custom_mass = mass.std_aa_mass.copy()
        custom_mass['H-'] = MOD + mass.nist_mass['H'][0][0]
        spec, spec_int = get_theor_spectrum(list('PEPTIDE'), acc, ion_types=('b', 'y'), maxcharge=2,
                                            aa_mass=custom_mass)
        spec_true = self.spec_PEPTIDE.copy()
        for k in spec_true:
            if k[0] == 'b':
                spec_true[k] += MOD / k[1]
        spec_int_true = self.spec_int_PEPTIDE.copy()
        spec_int_true['b'] = {int(x / acc) for x in np.concatenate((spec_true[('b', 1)], spec_true[('b', 2)]))}
        self._compare_spectra(spec, spec_int, spec_true, spec_int_true)

    def test_theor_spec_PEPTIDEterm(self):
        MOD = 42.12
        acc = 0.01
        custom_mass = mass.std_aa_mass.copy()
        custom_mass['-OH'] = MOD + mass.nist_mass['H'][0][0] + mass.nist_mass['O'][0][0]
        spec, spec_int = get_theor_spectrum(list('PEPTIDE'), acc, ion_types=('b', 'y'), maxcharge=2,
                                            aa_mass=custom_mass)
        spec_true = self.spec_PEPTIDE.copy()
        for k in spec_true:
            if k[0] == 'y':
                spec_true[k] += MOD / k[1]
        spec_int_true = self.spec_int_PEPTIDE.copy()
        spec_int_true['y'] = {int(x / acc) for x in np.concatenate((spec_true[('y', 1)], spec_true[('y', 2)]))}
        self._compare_spectra(spec, spec_int, spec_true, spec_int_true)


class AAstatResultTest(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_data')

        self.pepxml = [os.path.join(self.data_dir, 'SDS_01_0{}.pepXML'.format(num)) for num in [1, 2]]
        self.mzml = [os.path.join(self.data_dir, 'SDS_01_0{}.mzML'.format(num)) for num in [1, 2]]
        verbosity = int(os.environ.get('AASTAT_VERBOSITY', '1'))
        levels = [logging.WARNING, logging.INFO, logging.DEBUG, utils.INTERNAL]
        logging.basicConfig(format='{levelname:>8}: {asctime} {message}',
                        datefmt='[%H:%M:%S]', level=levels[verbosity], style='{')

    def test_aastat(self):
        if not os.path.isdir(self.data_dir):
            print('Test data not found, skipping integrative test.')
            return

        for f in self.pepxml + self.mzml:
            if not os.path.isfile(f):
                print(f, 'not found, skipping integrative test.')
                return

        args = argparse.Namespace(dir=self.data_dir, pepxml=self.pepxml, mzml=self.mzml,
            mgf=None, csv=None, params=None)
        params_dict = io.get_params_dict(args)
        self.figure_data, self.table, self.locmod_df, self.mass_shift_data_dict, self.fix_mods, self.var_mods = AA_stat(params_dict, args)

        self.assertEqual(self.table.index.tolist(),
            ['-246.1898', '-229.1630', '-203.1837', '-172.1413', '-171.1389', '-157.1415',
             '-147.1573', '-129.1468', '-116.0582', '-115.1201', '-114.1361', '-113.1335',
             '-100.1204', '-91.0091', '-72.1252', '-25.0315', '-18.0106', '-9.0368', '-2.0156', '-0.9374',
             '+0.0000', '+0.9842', '+1.0028', '+1.9874', '+2.0048', '+13.9786', '+15.0114',
             '+15.9949', '+16.9974', '+17.0268', '+17.9997', '+18.0283', '+30.9814', '+31.9894', '+32.9925',
             '+47.9847', '+52.9218', '+57.0217', '+58.0241', '+100.0160', '+229.1626', '+230.1649']
            )

        self.assertEqual(self.table['# peptides in bin'].tolist(),
            [57, 179, 172, 540, 97, 82, 102, 279, 57, 67, 283, 53, 102, 67, 125, 61, 139, 147, 72, 61,
             2854, 341, 558, 104, 79, 148, 72, 457, 170, 276, 74, 54, 108, 196, 62, 58, 342, 398, 110, 65, 158, 77]
            )

        self.assertEqual(self.fix_mods, {})

        self.assertEqual(self.var_mods, [
            ('isotope error', 1), ('N', '+0.9842'), ('K', '-114.1361'),
            ('M', '+15.9949'), ('K', '+57.0217'), ('C', '-9.0368'),
            ])

        # print(self.locmod_df['localization'].tolist())
        self.assertEqual(self.locmod_df['localization'].tolist(),
            [
                {'N-term_-246.1898': 24, 'C_-246.1898': 26, 'P_-246.1898': 8, 'H_-246.1898': 4, 'non-localized': 7, 'Y_-246.1898': 7},
                {'non-localized': 90, 'N-term_-246.1898': 3, 'C_-246.1898': 5, 'E_+17.0268': 13, 'P_-246.1898': 9, 'D_+17.0268': 13, 'H_-246.1898': 14, 'Y_-246.1898': 9, 'C-term_+17.0268': 5, 'K_+17.0268': 10, 'N-term_+17.0268': 7, 'R_+17.0268': 1},
                {'H_-203.1837': 32, 'Y_-203.1837': 86, 'N-term_-203.1837': 9, 'non-localized': 4},
                {},
                {},
                {'H_-157.1415': 17, 'N-term_-157.1415': 5},
                {'H_-147.1573': 21, 'non-localized': 5, 'N-term_-147.1573': 3, 'D_-18.0106': 1, 'H_-129.1468': 1},
                {'H_-129.1468': 44, 'N-term_-129.1468': 5},
                {'C_-116.0582': 24, 'non-localized': 28, 'N-term_-116.0582': 1},
                {'K_-115.1201': 62, 'C-term_-115.1201': 47},
                {'K_-114.1361': 173, 'Q_+0.9842': 4, 'C-term_-115.1201': 7, 'K_-115.1201': 7, 'H_-114.1361': 14, 'C-term_-114.1361': 151, 'N_+0.9842': 2, 'R_+0.9842': 1, 'non-localized': 6, 'N-term_-114.1361': 5, 'D_+15.0114': 1, 'H_-129.1468': 3, 'I_+15.0114': 1, 'N-term_+15.0114': 1, 'E_+15.0114': 1},
                {'K_-113.1335': 35, 'C-term_-113.1335': 27, 'N-term_-113.1335': 1, 'H_-113.1335': 9, 'non-localized': 2},
                {'C-term_-100.1204': 74, 'K_-100.1204': 97, 'H_-157.1415': 2, 'E_+57.0217': 1, 'N-term_-100.1204': 2, 'G_+57.0217': 1},
                {'C_-91.0091': 63, 'non-localized': 1},
                {'K_-72.1252': 124, 'C-term_-72.1252': 92, 'N-term_-72.1252': 4},
                {'C_-25.0315': 58, 'N-term_-25.0315': 3, 'Y_-25.0315': 1, 'non-localized': 1},
                {'Y_-18.0106': 2, 'non-localized': 18, 'T_-18.0106': 44, 'S_-18.0106': 22, 'E_-18.0106': 16, 'D_-18.0106': 35, 'C_-18.0106': 1, 'N-term_-18.0106': 3, 'C-term_-18.0106': 1},
                {'C_-9.0368': 127, 'non-localized': 10, 'C_-25.0315': 8, 'C_+15.9949': 1, 'G_+15.9949': 2, 'C-term_-9.0368': 1, 'R_-9.0368': 2, 'N-term_-9.0368': 16, 'E_+15.9949': 1, 'I_+15.9949': 1, 'M_+15.9949': 1, 'N-term_+15.9949': 1, 'A_+15.9949': 1, 'C-term_+15.9949': 1, 'K_+15.9949': 1},
                {'T_-18.0106': 3, 'D_+15.9949': 2, 'T_-2.0156': 1, 'C_-2.0156': 31, 'F_+15.9949': 1, 'Y_-18.0106': 1, 'L_+15.9949': 2, 'E_-18.0106': 3, 'non-localized': 10, 'C-term_-2.0156': 7, 'K_-2.0156': 7, 'S_-18.0106': 3, 'T_+15.9949': 1, 'S_-2.0156': 3, 'N-term_-2.0156': 2, 'V_-2.0156': 4, 'A_+15.9949': 3, 'C_-18.0106': 2, 'N-term_+15.9949': 1, 'Y_-2.0156': 2, 'K_+15.9949': 1, 'N-term_-18.0106': 1, 'D_-18.0106': 2, 'W_+15.9949': 1, 'M_+15.9949': 3},
                {},
                {},
                {'N_+0.9842': 232, 'Q_+0.9842': 73, 'C-term_+0.9842': 12, 'R_+0.9842': 15, 'non-localized': 4, 'N-term_+0.9842': 14},
                {'T_+1.0028': 34, 'D_+1.0028': 35, 'V_+1.0028': 42, 'Y_+1.0028': 33, 'non-localized': 61, 'S_+1.0028': 38, 'C_+1.0028': 36, 'P_+1.0028': 44, 'E_+1.0028': 65, 'G_+1.0028': 34, 'L_+1.0028': 45, 'A_+1.0028': 21, 'I_+1.0028': 27, 'M_+1.0028': 9, 'F_+1.0028': 34, 'N-term_+1.0028': 1},
                {'V_+1.9874': 9, 'C-term_+1.9874': 3, 'R_+1.9874': 3, 'Q_+1.9874': 15, 'N_+1.9874': 59, 'non-localized': 16, 'K_+1.9874': 1, 'W_+1.9874': 1},
                {'non-localized': 31, 'D_+2.0048': 1, 'Y_+2.0048': 5, 'L_+2.0048': 6, 'F_+2.0048': 2, 'E_+2.0048': 3, 'V_+2.0048': 6, 'S_+2.0048': 3, 'C_+2.0048': 3, 'G_+2.0048': 4, 'P_+2.0048': 7, 'I_+2.0048': 2, 'T_+2.0048': 4, 'A_+2.0048': 2},
                {'non-localized': 17, 'E_+13.9786': 9, 'T_-18.0106': 7, 'I_+31.9894': 1, 'P_+13.9786': 12, 'Q_+13.9786': 9, 'T_-2.0156': 3, 'W_+15.9949': 4, 'D_+15.9949': 2, 'Y_-2.0156': 3, 'V_+13.9786': 12, 'S_+13.9786': 5, 'I_+13.9786': 10, 'P_+31.9894': 2, 'N-term_+13.9786': 8, 'L_+13.9786': 19, 'W_+13.9786': 6, 'T_+13.9786': 6, 'C-term_+31.9894': 3, 'K_+31.9894': 2, 'C_+31.9894': 2, 'C-term_-2.0156': 6, 'K_-2.0156': 7, 'S_-18.0106': 3, 'W_+31.9894': 2, 'S_-2.0156': 3, 'V_+15.9949': 1, 'N-term_-2.0156': 1, 'Y_-18.0106': 1, 'R_+31.9894': 1, 'C-term_+13.9786': 1, 'R_+13.9786': 2, 'F_+15.9949': 1, 'C_+15.9949': 1, 'A_+13.9786': 9, 'E_-18.0106': 2, 'N-term_+15.9949': 2, 'L_+15.9949': 1, 'L_+31.9894': 1, 'Y_+31.9894': 2, 'G_+15.9949': 1, 'C_-2.0156': 1, 'A_+15.9949': 1, 'Y_+15.9949': 3, 'T_+15.9949': 1, 'M_+15.9949': 1, 'V_-2.0156': 1, 'D_-18.0106': 1, 'V_+31.9894': 1, 'Q_+15.9949': 1},
                {'non-localized': 12, 'D_+15.0114': 5, 'M_+15.0114': 33, 'E_+15.0114': 8, 'Y_+15.0114': 4, 'S_-2.0156': 1, 'C-term_+17.0268': 4, 'R_+17.0268': 1, 'N-term_+15.0114': 3, 'D_+17.0268': 2, 'V_-2.0156': 2, 'C_-2.0156': 2, 'T_-2.0156': 1, 'K_+17.0268': 3, 'I_+15.0114': 1, 'L_+15.0114': 3},
                {'A_+15.9949': 10, 'H_+15.9949': 1, 'non-localized': 62, 'M_+15.9949': 168, 'L_+15.9949': 15, 'V_+15.9949': 8, 'T_+15.9949': 10, 'W_+15.9949': 24, 'E_+15.9949': 11, 'S_+15.9949': 9, 'C_+15.9949': 42, 'Y_+15.9949': 35, 'Q_+0.9842': 4, 'M_+15.0114': 3, 'G_+15.9949': 4, 'P_+15.9949': 10, 'N-term_+15.9949': 20, 'N_+0.9842': 3, 'Q_+15.9949': 6, 'F_+15.9949': 11, 'I_+15.9949': 4, 'K_+15.9949': 8, 'C-term_+15.9949': 6, 'D_+15.9949': 4, 'I_+15.0114': 1, 'N_+15.9949': 7, 'N-term_+15.0114': 1, 'L_+15.0114': 3, 'E_+15.0114': 1, 'C-term_+0.9842': 1, 'R_+0.9842': 1},
                {'non-localized': 83, 'G_+16.9974': 1, 'A_+16.9974': 7, 'V_+16.9974': 9, 'F_+16.9974': 3, 'S_+16.9974': 3, 'L_+16.9974': 4, 'Y_+16.9974': 5, 'I_+16.9974': 3, 'P_+16.9974': 4, 'M_+16.9974': 24, 'N_+16.9974': 3, 'C_+16.9974': 8, 'W_+16.9974': 7, 'T_+16.9974': 1, 'C-term_+16.9974': 1, 'K_+16.9974': 1, 'E_+16.9974': 3, 'N-term_+16.9974': 1},
                {'C-term_+17.0268': 62, 'K_+17.0268': 46, 'R_+17.0268': 11, 'D_+17.0268': 95, 'E_+17.0268': 110, 'N-term_+17.0268': 5, 'non-localized': 9, 'P_+17.0268': 1, 'L_+17.0268': 1, 'V_+17.0268': 1, 'A_+17.0268': 1},
                {'C_+17.9997': 3, 'A_+17.9997': 1, 'non-localized': 51, 'S_+17.9997': 1, 'M_+17.9997': 2, 'L_+17.9997': 3, 'W_+17.9997': 1, 'Y_+17.9997': 2, 'E_+17.9997': 3, 'I_+17.9997': 2, 'N_+17.9997': 2, 'V_+17.9997': 1, 'F_+17.9997': 1, 'P_+17.9997': 1, 'N-term_+17.9997': 1},
                {'non-localized': 11, 'C_+18.0283': 14, 'D_+18.0283': 9, 'N-term_+18.0283': 1, 'Y_+18.0283': 1, 'E_+18.0283': 14, 'C-term_+18.0283': 5, 'R_+18.0283': 2, 'K_+18.0283': 2, 'P_+18.0283': 1},
                {'W_+30.9814': 16, 'N-term_+30.9814': 1},
                {'non-localized': 60, 'P_+31.9894': 8, 'W_+31.9894': 35, 'E_+31.9894': 11, 'Y_+31.9894': 19, 'I_+31.9894': 4, 'Y_-25.0315': 3, 'G_+57.0217': 1, 'C-term_+31.9894': 17, 'K_+31.9894': 11, 'C_+31.9894': 6, 'L_+31.9894': 12, 'F_+31.9894': 14, 'R_+31.9894': 6, 'C_-25.0315': 1, 'M_+57.0217': 1, 'V_+31.9894': 2, 'C_+57.0217': 1, 'H_+57.0217': 1, 'N-term_+31.9894': 11, 'M_+31.9894': 3},
                {'non-localized': 33, 'Y_+32.9925': 5, 'E_+32.9925': 4, 'P_+32.9925': 3, 'W_+32.9925': 8, 'L_+32.9925': 2, 'M_+32.9925': 2, 'F_+32.9925': 1, 'I_+32.9925': 2, 'V_+32.9925': 1, 'C_+32.9925': 1},
                {'I_+31.9894': 2, 'V_+15.9949': 3, 'non-localized': 18, 'E_+31.9894': 1, 'N_+15.9949': 1, 'W_+47.9847': 11, 'W_+31.9894': 15, 'C_+15.9949': 5, 'M_+15.9949': 2, 'F_+47.9847': 1, 'S_+15.9949': 1, 'E_+15.9949': 3, 'A_+15.9949': 1, 'Y_+47.9847': 8, 'C_+47.9847': 1, 'H_+57.0217': 1, 'C-term_-9.0368': 1, 'R_-9.0368': 1, 'N-term_+31.9894': 1, 'T_+15.9949': 1, 'W_+15.9949': 1},
                {},
                {'non-localized': 51, 'M_+57.0217': 39, 'S_+57.0217': 8, 'D_+57.0217': 3, 'T_+57.0217': 10, 'E_+57.0217': 12, 'C_+57.0217': 6, 'H_+57.0217': 37, 'G_+57.0217': 14, 'Y_+57.0217': 62, 'A_+57.0217': 14, 'C-term_+57.0217': 128, 'K_+57.0217': 141, 'N-term_+57.0217': 11},
                {'E_+58.0241': 4, 'M_+58.0241': 7, 'D_+58.0241': 3, 'Y_+58.0241': 14, 'non-localized': 43, 'T_+58.0241': 6, 'A_+58.0241': 3, 'H_+58.0241': 11, 'Q_+58.0241': 3, 'N-term_+58.0241': 3, 'K_+58.0241': 11, 'S_+58.0241': 2, 'G_+58.0241': 2, 'L_+58.0241': 1, 'C-term_+58.0241': 5},
                {'non-localized': 11, 'T_+229.1626': 2, 'H_-129.1468': 5, 'S_+100.0160': 47, 'N-term_+100.0160': 4, 'T_+100.0160': 2, 'S_+229.1626': 3, 'N-term_-129.1468': 1, 'N-term_+229.1626': 1},
                {'T_+229.1626': 38, 'S_+229.1626': 83, 'N-term_+229.1626': 12, 'C-term_+229.1626': 3, 'K_+229.1626': 3, 'non-localized': 14, 'H_+229.1626': 12, 'Q_+229.1626': 1, 'G_+229.1626': 2, 'A_+229.1626': 1, 'D_+229.1626': 1, 'E_+229.1626': 1, 'F_+229.1626': 1, 'V_+229.1626': 1},
                {'T_+230.1649': 15, 'N-term_+230.1649': 5, 'non-localized': 19, 'E_+230.1649': 2, 'S_+230.1649': 36, 'C-term_+230.1649': 1, 'K_+230.1649': 1, 'H_+230.1649': 4}
            ]
        )
