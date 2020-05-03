import unittest
import numpy as np
from .locTools import get_theor_spectrum
from pyteomics import mass


class AA_stat_Test(unittest.TestCase):

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
