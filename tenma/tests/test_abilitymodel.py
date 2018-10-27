# coding:utf-8
import unittest
import os
from tenma import abilitymodel as am
from pprint import pprint


class TestAbilitymodel(unittest.TestCase):
    def test_normal(self):
        os.environ['env'] = "not test"
        expected = 1.0
        actual = am.normal(1.0, 3.0)
        self.assertNotEqual(expected, actual)

        os.environ['env'] = "test"
        expected = 1.0
        actual = am.normal(1.0, 3.0)
        self.assertEqual(expected, actual)

    def test_abilitymodel_calc_order(self):
        os.environ['env'] = "test"
        params = {
            am.P_MU : 1.0,
            am.P_SIGMA : 1.0
        }
        order = 18.0
        expected = 1.0 * (1.0 / order)
        actual = am.calc_order(order, params)
        self.assertEqual(expected, actual)


        params = {
            am.P_MU : 3.0,
            am.P_SIGMA : 1.0
        }
        order = 5.0
        expected = 3.0 * (1.0 / order)
        actual = am.calc_order(order, params)
        self.assertEqual(expected, actual)

    def test_abilitymodel_calc_grade(self):
        os.environ['env'] = "test"
        params = {
            am.G_A_MU : 1.0,
            am.G_A_SIGMA : 1.0,
            am.G_B_MU : 1.0,
            am.G_B_SIGMA : 1.0
        }

        now_grade = 3.0
        bf_grade = 1.0
        expected = am.inv_logit(1.0 * (now_grade - bf_grade) + 1.0)
        actual = am.calc_grade(bf_grade, now_grade, params)
        self.assertEqual(expected, actual)


        params = {
            am.G_A_MU : 2.0,
            am.G_A_SIGMA : 1.0,
            am.G_B_MU : 3.0,
            am.G_B_SIGMA : 1.0
        }
        now_grade = 4.0
        bf_grade = 2.0
        expected = am.inv_logit(2.0 * (now_grade - bf_grade) + 3.0)
        actual = am.calc_grade(bf_grade, now_grade, params)
        self.assertEqual(expected, actual)

    def test_abilitymodel_calc_bfrace(self):
        os.environ['env'] = "test"
        params = {
            am.P_MU : 1.0,
            am.P_SIGMA : 1.0,
            am.G_A_MU : 1.0,
            am.G_A_SIGMA : 1.0,
            am.G_B_MU : 1.0,
            am.G_B_SIGMA : 1.0
        }
        order = 1.0
        now_grade = 1.0
        bf_grade = 1.0
        expected = 1.0 * (1.0 / order) * am.inv_logit(1.0 * (now_grade - bf_grade) + 1.0)
        actual = am.calc_bfrace(order, bf_grade, now_grade, params)
        self.assertEqual(expected, actual)

        params = {
            am.P_MU : 2.0,
            am.P_SIGMA : 1.0,
            am.G_A_MU : 3.0,
            am.G_A_SIGMA : 1.0,
            am.G_B_MU : 4.0,
            am.G_B_SIGMA : 1.0
        }
        order = 5.0
        now_grade = 6.0
        bf_grade = 7.0
        expected = 2.0 * (1.0 / order) * am.inv_logit(3.0 * (now_grade - bf_grade) + 4.0)
        actual = am.calc_bfrace(order, bf_grade, now_grade, params)
        self.assertEqual(expected, actual)

    def test_abilitymodel_predict(self):
        os.environ['env'] = "test"
        params = {
            am.P_MU : [1.0, 2.0, 3.0, 4.0],
            am.P_SIGMA : [1.0, 1.0, 1.0, 1.0],
            am.BIAS_MU : 5.0,
            am.BIAS_SIGMA : 1.0,
            am.G_A_MU : 1.0,
            am.G_A_SIGMA : 1.0,
            am.G_B_MU : 1.0,
            am.G_B_SIGMA : 1.0
        }
        now_grade = 1.0
        l_order = [1.0, 1.0, 1.0, 1.0]
        l_bf_grade = [1.0, 1.0, 1.0, 1.0]
        expected = am.inv_logit(
            1.0 * (1.0 / l_order[0])
            * am.inv_logit(1.0 * (now_grade - l_bf_grade[0]) + 1.0)
        + 2.0 * (1.0 / l_order[1])
            * am.inv_logit(1.0 * (now_grade - l_bf_grade[1]) + 1.0)
        + 3.0 * (1.0 / l_order[2])
            * am.inv_logit(1.0 * (now_grade - l_bf_grade[2]) + 1.0)
        + 4.0 * (1.0 / l_order[3])
            * am.inv_logit(1.0 * (now_grade - l_bf_grade[3]) + 1.0)
        + 5.0)
        actual = am.predict(l_order, l_bf_grade, now_grade, params)
        self.assertEqual(expected, actual)

        
        


if __name__ == "__main__":
    unittest.main()