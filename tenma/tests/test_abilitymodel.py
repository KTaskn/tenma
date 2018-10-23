# coding:utf-8
import unittest
import numpy as np
from tenma import abilitymodel

class TestTenma(unittest.TestCase):
    """
    """

    def test_cutoff(self):
        actual = abilitymodel.cutoff(0.1)
        expected = 0
        self.assertEqual(expected, actual)

        actual = abilitymodel.cutoff(0.49)
        expected = 0
        self.assertEqual(expected, actual)

        actual = abilitymodel.cutoff(0.5)
        expected = 1
        self.assertEqual(expected, actual)

        actual = abilitymodel.cutoff(0.8)
        expected = 1
        self.assertEqual(expected, actual)

    def test_ability(self):
        X_R = [
            [0],
            [1],
            [2],
            [0]
        ]
        X_G = [
            [0],
            [1],
            [2],
            [1],
        ]
        X_RACE = [0, 1, 0, 1]
        params = {
            abilitymodel.COL_P_MU : [
                [-3.0, -3.0, -3.0],
                [2.0, 2.0, 2.0],
                [1.0, 1.0, 1.0],
                [-2.0, -2.0, -2.0]
            ],
            abilitymodel.COL_P_SIGMA : [
                [0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01],
            ],
            abilitymodel.COL_RACE_P_MU : [-5, 5],
            abilitymodel.COL_RACE_P_SIGMA : [0.01, 0.01]
        }
        tmp = abilitymodel.ability(X_R, X_G, X_RACE, params)
        print(tmp)
        actual = list(map(abilitymodel.cutoff, tmp))
        expected = [0, 1, 0, 1]
        self.assertEqual(expected, actual)


        X_R = [
            [0, 1],
            [1, 2],
            [2, 1],
            [2, 0]
        ]
        X_G = [
            [0, 1],
            [1, 2],
            [2, 1],
            [1, 0]
        ]
        X_RACE = [0, 1, 0, 0]
        params = {
            abilitymodel.COL_P_MU : [
                [-3.0, -3.0, -3.0],
                [2.0, 2.0, 2.0],
                [1.0, 1.0, 1.0],
                [-2.0, -2.0, -2.0]
            ],
            abilitymodel.COL_P_SIGMA : [
                [0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01],
            ],
            abilitymodel.COL_RACE_P_MU : [-5, 5],
            abilitymodel.COL_RACE_P_SIGMA : [0.01, 0.01]
        }
        tmp = abilitymodel.ability(X_R, X_G, X_RACE, params)
        actual = list(map(abilitymodel.cutoff, tmp))
        expected = [0, 1, 0, 0]
        self.assertEqual(expected, actual)



if __name__ == "__main__":
    unittest.main()