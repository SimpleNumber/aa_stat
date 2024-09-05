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
            # print(k)
            # print(spec[k])
            # print(spec_true[k])
            self.assertTrue(np.allclose(spec[k], spec_true[k], atol=eps))

        self.assertEqual(spec_int, spec_int_true)

    def test_theor_spec_PEPTIDE(self):
        spec, spec_int = get_theor_spectrum(list('PEPTIDE'), 0.01, ion_types=('b', 'y'), maxcharge=2)
        self._compare_spectra(spec, spec_int, self.spec_PEPTIDE, self.spec_int_PEPTIDE)

    def test_theor_spec_PEvPTIDE(self):
        MOD = 15.994915
        acc = 0.01
        pos = 3
        spec, spec_int = get_theor_spectrum(list('PEPTIDE'), acc, ion_types=('b', 'y'), maxcharge=2,
            modifications={pos: MOD})
        spec_true = self.spec_PEPTIDE.copy()
        for k in spec_true:
            if k[0] == 'b':
                spec_true[k][pos - 1:] += MOD / k[1]
            else:
                spec_true[k][7 - pos:] += MOD / k[1]
        spec_int_true = {}
        for t in ('b', 'y'):
            spec_int_true[t] = {int(x / acc) for x in np.concatenate((spec_true[(t, 1)], spec_true[(t, 2)]))}

        self._compare_spectra(spec, spec_int, spec_true, spec_int_true)

    def test_theor_spec_vPEPTIDE(self):
        MOD = 15.994915
        acc = 0.01
        pos = 1
        spec, spec_int = get_theor_spectrum(list('PEPTIDE'), acc, ion_types=('b', 'y'), maxcharge=2,
            modifications={pos: MOD})
        spec_true = self.spec_PEPTIDE.copy()
        for k in spec_true:
            if k[0] == 'b':
                spec_true[k][pos - 1:] += MOD / k[1]
            else:
                spec_true[k][7 - pos:] += MOD / k[1]
        spec_int_true = {}
        for t in ('b', 'y'):
            spec_int_true[t] = {int(x / acc) for x in np.concatenate((spec_true[(t, 1)], spec_true[(t, 2)]))}

        self._compare_spectra(spec, spec_int, spec_true, spec_int_true)

    def test_theor_spec_PEvaPTIDE(self):
        MOD = 15.994915
        acc = 0.01
        pos = 3
        aa_mass = mass.std_aa_mass.copy()
        aa_mass['aP'] = aa_mass['P'] + MOD
        peptide = list('PEPTIDE')
        peptide[pos - 1] = 'aP'
        spec, spec_int = get_theor_spectrum(peptide, acc, ion_types=('b', 'y'), maxcharge=2,
            modifications={pos: MOD}, aa_mass=aa_mass)
        spec_true = self.spec_PEPTIDE.copy()
        for k in spec_true:
            if k[0] == 'b':
                spec_true[k][pos - 1:] += 2 * MOD / k[1]
            else:
                spec_true[k][7 - pos:] += 2 * MOD / k[1]
        spec_int_true = {}
        for t in ('b', 'y'):
            spec_int_true[t] = {int(x / acc) for x in np.concatenate((spec_true[(t, 1)], spec_true[(t, 2)]))}

        self._compare_spectra(spec, spec_int, spec_true, spec_int_true)

    def test_theor_spec_mPEPTIDE(self):
        MOD = 15.994915
        acc = 0.01
        custom_mass = mass.std_aa_mass.copy()
        custom_mass['mP'] = mass.std_aa_mass['P'] + MOD
        spec, spec_int = get_theor_spectrum(['mP'] + list('EPTIDE'), acc, ion_types=('b', 'y'), maxcharge=2,
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
            mgf=None, csv=None, params=None, processes=int(os.environ.get('AASTAT_PROCESSES', '0')) or None)
        params_dict = io.get_params_dict(args)
        # params_dict['decoy_prefix'] = 'DECOY_'
        self.figure_data, self.table, self.locmod_df, self.mass_shift_data_dict, self.fix_mods, self.var_mods = AA_stat(params_dict, args)

        counts = [57, 179, 173, 540, 100, 82, 102, 279, 57, 67, 282, 52, 102, 66, 125, 60, 139, 145, 71, 2851, 341, 558, 103,
             79, 148, 128, 71, 460, 171, 277, 54, 51, 106, 197, 61, 57, 341, 397, 108, 67, 158, 78]
        # print(self.table['# peptides in bin'].tolist())
        print(self.table['# peptides in bin'].sum(), 'peptides found. The test has', sum(counts))
        shifts = ['-246.1898', '-229.1630', '-203.1838', '-172.1413', '-171.1388', '-157.1416', '-147.1573', '-129.1469',
             '-116.0580', '-115.1202', '-114.1359', '-113.1334', '-100.1205', '-91.0091', '-72.1252', '-25.0314',
             '-18.0105', '-9.0368', '-2.0156', '+0.0000', '+0.9842', '+1.0029', '+1.9874', '+2.0048', '+13.9786',
             '+14.9845', '+15.0114', '+15.9951', '+16.9976', '+17.0269', '+18.0103', '+18.0282', '+30.9811', '+31.9893',
             '+32.9926', '+47.9847', '+52.9219', '+57.0217', '+58.0243', '+100.0160', '+229.1628', '+230.1650']
        print(self.table.shape[0], 'mass shifts found. The test has', len(shifts))
        # print(self.table.index.tolist())

        self.assertEqual(self.table['# peptides in bin'].tolist(), counts)

        self.assertEqual(self.table.index.tolist(), shifts)

        self.assertEqual(self.fix_mods, {})

        self.assertEqual(self.var_mods, [
            ('isotope error', 1), ('N', '+0.9842'), ('M', '+15.9951'), ('K', '-114.1359'),
             ('K', '+57.0217'), ('S', '+229.1628'),
            ])

        # print(self.locmod_df['localization'].tolist())
        self.assertEqual(self.locmod_df['localization'].tolist(),
            [
                {'N-term_-246.1898': 25, 'C_-246.1898': 27, 'P_-246.1898': 7, 'H_-246.1898': 4, 'non-localized': 7, 'Y_-246.1898': 7},
                {'non-localized': 90, 'N-term_-246.1898': 3, 'C_-246.1898': 5, 'E_+17.0269': 13, 'P_-246.1898': 9, 'D_+17.0269': 13, 'H_-246.1898': 14, 'Y_-246.1898': 9, 'C-term_+17.0269': 5, 'K_+17.0269': 10, 'N-term_+17.0269': 7, 'R_+17.0269': 1},
                {'H_-203.1838': 32, 'Y_-203.1838': 87, 'N-term_-203.1838': 9, 'non-localized': 4},
                {},
                {},
                {'H_-157.1416': 17, 'N-term_-157.1416': 5},
                {'H_-147.1573': 21, 'non-localized': 5, 'N-term_-147.1573': 3, 'D_-18.0105': 1, 'H_-129.1469': 1},
                {'H_-129.1469': 43, 'N-term_-129.1469': 5},
                {'C_-116.0580': 20, 'non-localized': 33, 'N-term_-116.0580': 1},
                {'K_-115.1202': 63, 'C-term_-115.1202': 48},
                {'K_-114.1359': 174, 'Q_+0.9842': 5, 'C-term_-115.1202': 8, 'K_-115.1202': 8, 'H_-114.1359': 13, 'C-term_-114.1359': 152, 'N_+0.9842': 2, 'R_+0.9842': 1, 'non-localized': 5, 'N-term_-114.1359': 5, 'D_+15.0114': 1, 'H_-129.1469': 3, 'I_+15.0114': 1, 'N-term_+15.0114': 1, 'E_+15.0114': 1},
                {'K_-113.1334': 34, 'C-term_-113.1334': 27, 'N-term_-113.1334': 1, 'H_-113.1334': 9, 'non-localized': 2},
                {'C-term_-100.1205': 74, 'K_-100.1205': 97, 'H_-157.1416': 2, 'E_+57.0217': 1, 'N-term_-100.1205': 2, 'G_+57.0217': 1},
                {'non-localized': 1, 'C_-91.0091': 65},
                {'K_-72.1252': 124, 'C-term_-72.1252': 92, 'N-term_-72.1252': 4},
                {'C_-25.0314': 57, 'N-term_-25.0314': 3, 'Y_-25.0314': 1, 'non-localized': 1},
                {'T_-18.0105': 45, 'non-localized': 18, 'S_-18.0105': 22, 'D_-18.0105': 37, 'E_-18.0105': 15, 'N-term_-18.0105': 3, 'Y_-18.0105': 1, 'C-term_-18.0105': 1},
                {'C_-9.0368': 129, 'non-localized': 9, 'C-term_-9.0368': 1, 'R_-9.0368': 2, 'N-term_-9.0368': 16, 'G_+15.9951': 1, 'C_-25.0314': 5, 'I_+15.9951': 1, 'M_+15.9951': 1, 'N-term_+15.9951': 1, 'A_+15.9951': 1, 'C-term_+15.9951': 1, 'K_+15.9951': 1},
                {'T_-18.0105': 3, 'D_+15.9951': 2, 'T_-2.0156': 1, 'C_-2.0156': 36, 'F_+15.9951': 1, 'Y_-18.0105': 1, 'L_+15.9951': 2, 'E_-18.0105': 3, 'non-localized': 8, 'C-term_-2.0156': 7, 'K_-2.0156': 7, 'S_-18.0105': 1, 'T_+15.9951': 1, 'S_-2.0156': 3, 'N-term_-2.0156': 2, 'V_-2.0156': 4, 'N-term_+15.9951': 1, 'A_+15.9951': 1, 'Y_-2.0156': 2, 'N-term_-18.0105': 1, 'D_-18.0105': 2, 'W_+15.9951': 1, 'M_+15.9951': 2},
                {},
                {'N_+0.9842': 246, 'Q_+0.9842': 64, 'C-term_+0.9842': 10, 'R_+0.9842': 13, 'non-localized': 5, 'N-term_+0.9842': 15},
                {'T_+1.0029': 34, 'D_+1.0029': 33, 'V_+1.0029': 37, 'G_+1.0029': 34, 'Y_+1.0029': 35, 'S_+1.0029': 37, 'A_+1.0029': 22, 'C_+1.0029': 35, 'P_+1.0029': 46, 'E_+1.0029': 62, 'I_+1.0029': 30, 'non-localized': 61, 'L_+1.0029': 50, 'M_+1.0029': 9, 'F_+1.0029': 33, 'N-term_+1.0029': 1},
                {'non-localized': 17, 'N_+1.9874': 61, 'V_+1.9874': 7, 'Q_+1.9874': 14, 'C-term_+1.9874': 2, 'R_+1.9874': 2, 'K_+1.9874': 1, 'W_+1.9874': 1},
                {'non-localized': 32, 'A_+2.0048': 3, 'Y_+2.0048': 5, 'L_+2.0048': 6, 'E_+2.0048': 6, 'V_+2.0048': 6, 'S_+2.0048': 2, 'D_+2.0048': 1, 'G_+2.0048': 3, 'P_+2.0048': 7, 'I_+2.0048': 1, 'C_+2.0048': 2, 'T_+2.0048': 4, 'F_+2.0048': 1},
                {'non-localized': 17, 'E_+13.9786': 9, 'T_-18.0105': 8, 'I_+31.9893': 1, 'P_+13.9786': 12, 'Q_+13.9786': 9, 'T_-2.0156': 2, 'W_+15.9951': 4, 'D_+15.9951': 2, 'Y_-2.0156': 3, 'V_+13.9786': 12, 'S_+13.9786': 6, 'I_+13.9786': 10, 'P_+31.9893': 2, 'N-term_+13.9786': 8, 'L_+13.9786': 19, 'W_+13.9786': 6, 'T_+13.9786': 6, 'C-term_+31.9893': 3, 'K_+31.9893': 2, 'C_+31.9893': 2, 'C-term_-2.0156': 6, 'K_-2.0156': 7, 'S_-18.0105': 2, 'W_+31.9893': 2, 'S_-2.0156': 3, 'V_+15.9951': 1, 'N-term_-2.0156': 1, 'Y_-18.0105': 1, 'R_+31.9893': 1, 'C-term_+13.9786': 1, 'R_+13.9786': 2, 'F_+15.9951': 1, 'C_+15.9951': 1, 'A_+13.9786': 9, 'E_-18.0105': 2, 'N-term_+15.9951': 2, 'L_+15.9951': 1, 'L_+31.9893': 1, 'Y_+31.9893': 1, 'G_+15.9951': 1, 'C_-2.0156': 1, 'M_+31.9893': 1, 'Y_+15.9951': 3, 'T_+15.9951': 1, 'M_+15.9951': 1, 'V_-2.0156': 1, 'D_-18.0105': 1, 'V_+31.9893': 1, 'Q_+15.9951': 1},
                {'T_+14.9845': 9, 'W_+14.9845': 7, 'M_+14.9845': 14, 'non-localized': 31, 'S_+14.9845': 6, 'A_+14.9845': 5, 'P_+14.9845': 9, 'V_+14.9845': 9, 'E_+14.9845': 10, 'L_+14.9845': 17, 'Q_+14.9845': 3, 'N-term_+14.9845': 7, 'I_+14.9845': 6, 'C-term_+14.9845': 2, 'R_+14.9845': 2},
                {'non-localized': 12, 'M_+15.0114': 33, 'E_+15.0114': 8, 'Y_+15.0114': 4, 'S_-2.0156': 1, 'C-term_+17.0269': 4, 'R_+17.0269': 1, 'D_+15.0114': 4, 'N-term_+15.0114': 3, 'D_+17.0269': 2, 'V_-2.0156': 2, 'C_-2.0156': 2, 'T_-2.0156': 1, 'K_+17.0269': 3, 'I_+15.0114': 1, 'L_+15.0114': 3},
                {'M_+15.9951': 189, 'L_+15.9951': 14, 'W_+15.9951': 28, 'non-localized': 63, 'C_+15.9951': 42, 'P_+15.9951': 11, 'V_+15.9951': 9, 'E_+15.9951': 7, 'Y_+15.9951': 36, 'A_+15.9951': 5, 'Q_+0.9842': 4, 'M_+15.0114': 3, 'G_+15.9951': 3, 'N-term_+15.9951': 20, 'T_+15.9951': 6, 'S_+15.9951': 5, 'N_+0.9842': 3, 'Q_+15.9951': 4, 'F_+15.9951': 11, 'K_+15.9951': 6, 'C-term_+15.9951': 5, 'I_+15.0114': 1, 'N_+15.9951': 7, 'I_+15.9951': 3, 'D_+15.9951': 3, 'N-term_+15.0114': 1, 'L_+15.0114': 3, 'E_+15.0114': 1, 'C-term_+0.9842': 1, 'R_+0.9842': 1},
                {'non-localized': 89, 'A_+16.9976': 4, 'F_+16.9976': 3, 'V_+16.9976': 7, 'C_+16.9976': 9, 'S_+16.9976': 3, 'L_+16.9976': 4, 'Y_+16.9976': 5, 'I_+16.9976': 2, 'P_+16.9976': 4, 'N_+16.9976': 2, 'M_+16.9976': 27, 'W_+16.9976': 6, 'T_+16.9976': 1, 'C-term_+16.9976': 1, 'K_+16.9976': 1, 'E_+16.9976': 3, 'N-term_+16.9976': 1},
                {'C-term_+17.0269': 62, 'K_+17.0269': 46, 'R_+17.0269': 11, 'D_+17.0269': 95, 'E_+17.0269': 110, 'N-term_+17.0269': 5, 'non-localized': 10, 'P_+17.0269': 1, 'L_+17.0269': 1, 'V_+17.0269': 1, 'A_+17.0269': 1},
                {'F_+18.0103': 1, 'S_+18.0103': 1, 'non-localized': 42, 'L_+18.0103': 2, 'W_+18.0103': 1, 'E_+18.0103': 2, 'N_+18.0103': 2, 'C_+18.0103': 1, 'I_+18.0103': 1, 'P_+18.0103': 1},
                {'non-localized': 10, 'C_+18.0282': 14, 'D_+18.0282': 9, 'N-term_+18.0282': 1, 'Y_+18.0282': 1, 'E_+18.0282': 13, 'C-term_+18.0282': 4, 'R_+18.0282': 2, 'K_+18.0282': 1, 'P_+18.0282': 1},
                {'W_+30.9811': 16, 'N-term_+30.9811': 1},
                {'non-localized': 59, 'V_+31.9893': 3, 'P_+31.9893': 8, 'W_+31.9893': 35, 'E_+31.9893': 11, 'Y_+31.9893': 19, 'I_+31.9893': 5, 'Y_-25.0314': 3, 'G_+57.0217': 1, 'C-term_+31.9893': 17, 'K_+31.9893': 11, 'C_+31.9893': 6, 'L_+31.9893': 12, 'F_+31.9893': 14, 'R_+31.9893': 6, 'C_-25.0314': 1, 'M_+57.0217': 1, 'C_+57.0217': 1, 'H_+57.0217': 1, 'N-term_+31.9893': 11, 'M_+31.9893': 3},
                {'non-localized': 33, 'Y_+32.9926': 4, 'E_+32.9926': 4, 'P_+32.9926': 3, 'W_+32.9926': 8, 'L_+32.9926': 2, 'M_+32.9926': 2, 'F_+32.9926': 1, 'I_+32.9926': 2, 'V_+32.9926': 1, 'C_+32.9926': 1},
                {'I_+31.9893': 2, 'V_+15.9951': 2, 'non-localized': 18, 'E_+31.9893': 1, 'N_+15.9951': 1, 'W_+47.9847': 10, 'W_+31.9893': 15, 'C_+15.9951': 5, 'M_+15.9951': 3, 'F_+47.9847': 1, 'S_+15.9951': 1, 'E_+15.9951': 3, 'A_+15.9951': 1, 'Y_+47.9847': 8, 'C_+47.9847': 1, 'H_+57.0217': 1, 'C-term_-9.0368': 1, 'R_-9.0368': 1, 'N-term_+31.9893': 1, 'T_+15.9951': 1, 'W_+15.9951': 1},
                {},
                {'non-localized': 52, 'M_+57.0217': 39, 'H_+57.0217': 53, 'D_+57.0217': 1, 'Y_+57.0217': 65, 'N-term_+57.0217': 13, 'A_+57.0217': 8, 'C-term_+57.0217': 139, 'K_+57.0217': 158, 'T_+57.0217': 3, 'G_+57.0217': 9, 'S_+57.0217': 2, 'C_+57.0217': 2, 'E_+57.0217': 4},
                {'non-localized': 43, 'D_+58.0243': 3, 'Y_+58.0243': 14, 'M_+58.0243': 6, 'T_+58.0243': 6, 'H_+58.0243': 12, 'E_+58.0243': 3, 'Q_+58.0243': 3, 'K_+58.0243': 12, 'N-term_+58.0243': 3, 'S_+58.0243': 2, 'G_+58.0243': 2, 'A_+58.0243': 1, 'L_+58.0243': 1, 'C-term_+58.0243': 5},
                {'non-localized': 12, 'S_+100.0160': 51, 'H_-129.1469': 3, 'S_+229.1628': 2, 'N-term_-129.1469': 1, 'T_+229.1628': 1, 'N-term_+100.0160': 4, 'T_+100.0160': 1},
                {'S_+229.1628': 104, 'T_+229.1628': 24, 'non-localized': 16, 'C-term_+229.1628': 2, 'K_+229.1628': 2, 'H_+229.1628': 7, 'N-term_+229.1628': 10, 'A_+229.1628': 1, 'E_+229.1628': 1, 'F_+229.1628': 1, 'G_+229.1628': 1, 'V_+229.1628': 1},
                {'S_+230.1650': 40, 'T_+230.1650': 12, 'non-localized': 23, 'N-term_+230.1650': 3, 'E_+230.1650': 1, 'H_+230.1650': 2}
            ]
        )
