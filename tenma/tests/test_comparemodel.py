# coding:utf-8
import unittest
import os
from tenma import comparemodel as cm
from pprint import pprint
import numpy as np


class TestComparemodel(unittest.TestCase):

    def inv_logit(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def test_normal(self):
        os.environ['env'] = "not test"
        expected = 1.0
        actual = cm.normal(1.0, 3.0)
        self.assertNotEqual(expected, actual)

        os.environ['env'] = "test"
        expected = 1.0
        actual = cm.normal(1.0, 3.0)
        self.assertEqual(expected, actual)

    def test_calc(self):
        params = {
            cm.A_MU : 1.0,
            cm.A_SIGMA : 1.0,
            cm.B_MU : 2.0,
            cm.B_SIGMA : 1.0
        }
        x = 3
        expected = self.inv_logit(1.0 * x + 2.0)
        actual = cm.calc(x, params)
        self.assertEqual(expected, actual)