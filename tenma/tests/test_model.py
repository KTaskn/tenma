# coding:utf-8
import unittest
import numpy as np
from tenma import model

class TestTenma(unittest.TestCase):
    """
    """

    def test_cutoff(self):
        actual = model.cutoff(0.1)
        expected = 0
        self.assertEqual(expected, actual)

        actual = model.cutoff(0.49)
        expected = 0
        self.assertEqual(expected, actual)

        actual = model.cutoff(0.5)
        expected = 1
        self.assertEqual(expected, actual)

        actual = model.cutoff(0.8)
        expected = 1
        self.assertEqual(expected, actual)

    def test_ability(self):
        X = [
            [0],
            [1],
            [2],
            [3]
        ]
        params = {
            model.COL_MU_TEMP % 1: [-3.0, 2.0, 1.0, -2.0],
            model.COL_SIGMA_TEMP % 1: [0.01, 0.01, 0.01, 0.01],
        }
        tmp = model.ability(X, 1, params)
        print(tmp)
        actual = list(map(model.cutoff, tmp))
        expected = [0, 1, 1, 0]
        self.assertEqual(expected, actual)


        X = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0]
        ]
        params = {
            model.COL_MU_TEMP % 1: [-3.0, 2.0, 1.0, -2.0],
            model.COL_SIGMA_TEMP % 1: [0.01, 0.01, 0.01, 0.01],
            model.COL_MU_TEMP % 2: [-3.0, 5.0, 1.0, -3.0],
            model.COL_SIGMA_TEMP % 2: [0.01, 0.01, 0.01, 0.01],
        }
        tmp = model.ability(X, 2, params)
        actual = list(map(model.cutoff, tmp))
        expected = [1, 1, 0, 0]
        self.assertEqual(expected, actual)



if __name__ == "__main__":
    unittest.main()