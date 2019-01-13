# coding:utf-8
import unittest
import os
import numpy as np
import pandas as pd
from tenma import naivebayes


class TestNaivebayes(unittest.TestCase):

    def setUp(self):
        self.nb = naivebayes.NaiveBayes(None, None, None, None, None, None)

    def test_P_A(self):
        col_A = 'col_1'
        df = pd.DataFrame({
            col_A: [1, 1, 1, 2, 2, 4]
        })
        
        expect = np.log10(0.5)
        A = 1
        self.assertEqual(expect, self.nb._NaiveBayes__P_A(df, col_A, A))

        expect = np.log10(1 / 12)
        A = 0
        self.assertEqual(expect, self.nb._NaiveBayes__P_A(df, col_A, A))

    def test_P_A_notcol(self):
        col_A = 'col_1'
        df = pd.DataFrame({
            "noname": [1, 1, 1, 2, 2, 4]
        })
        
        A = 0
        with self.assertRaises(KeyError):
            self.nb._NaiveBayes__P_A(df, col_A, A)


    def test_P_A_len0(self):
        col_A = 'col_1'
        df = pd.DataFrame({
            col_A: []
        })
        
        A = 0
        with self.assertRaises(RuntimeError):
            self.nb._NaiveBayes__P_A(df, col_A, A)


    def test_P_AB(self):
        col_A = 'col_1'
        col_B = 'col_2'
        df = pd.DataFrame({
            col_B: [1, 1, 2, 3, 3, 4],
            col_A: [1, 1, 1, 2, 5, 4]
        })
        
        expect = np.log10(1.0)
        B = 1
        A = 1
        self.assertEqual(expect, self.nb._NaiveBayes__P_AB(df, col_A, A, col_B, B))

        expect = np.log10(0.5)
        B = 3
        A = 2
        self.assertEqual(expect, self.nb._NaiveBayes__P_AB(df, col_A, A, col_B, B))

        expect = np.log10(1 / (6))
        B = 5
        A = 2
        self.assertEqual(expect, self.nb._NaiveBayes__P_AB(df, col_A, A, col_B, B))


    def test_P_AB_notcol(self):
        col_A = 'col_1'
        col_B = 'col_2'
        df = pd.DataFrame({
            "noname": [1, 1, 1, 2, 2, 4],
            col_B: [1, 2, 3, 4, 5, 6]
        })
        
        A = 0
        B = 1
        with self.assertRaises(KeyError):
            self.nb._NaiveBayes__P_AB(df, col_A, A, col_B, B)

        
        col_A = 'col_1'
        col_B = 'col_2'
        df = pd.DataFrame({
            col_A: [1, 1, 1, 2, 2, 4],
            "noname": [1, 2, 3, 4, 5, 6]
        })
        
        A = 0
        B = 1
        with self.assertRaises(KeyError):
            self.nb._NaiveBayes__P_AB(df, col_A, A, col_B, B)


    def test_P_AB_len0(self):
        col_A = 'col_1'
        col_B = 'col_2'
        df = pd.DataFrame({
            col_A: [],
            col_B: []
        })
        
        A = 0
        B = 1
        with self.assertRaises(RuntimeError):
            self.nb._NaiveBayes__P_AB(df, col_A, A, col_B, B)

    def test_P_BA(self):
        col_A = 'col_1'
        col_B = 'col_2'
        df = pd.DataFrame({
            col_B: [1, 1, 0, 0],
            col_A: [1, 0, 1, 1]
        })

        P_B = np.log10(1 / 2)

        P_A = 3 / 4
        P_AB = 1 / 2
        
        expect = np.log10(P_AB) + P_B - np.log10(P_A)
        B = 1
        A = 1
        self.assertEqual(expect, self.nb._NaiveBayes__P_BA(df, P_B, col_A, A, col_B, B))

    # def test_get_p_rentan(self):
    #     dic = {
    #         "01": {
    #             "a": 1,
    #             "b": 2,
    #             "c": 3
    #         },
    #         "02": {
    #             "a": 3,
    #             "b": 4,
    #             "c": 5
    #         },
    #         "03": {
    #             "a": 6,
    #             "b": 7,
    #             "c": 8
    #         }
    #     }

    #     tpl = ["a"]
    #     expect = 1
    #     self.assertEqual(expect, naivebayes.get_p_rentan(tpl, dic))
        
    #     tpl = ["a", "b"]
    #     expect = 5
    #     self.assertEqual(expect, naivebayes.get_p_rentan(tpl, dic))


    #     tpl = ["b", "c"]
    #     expect = 7
    #     self.assertEqual(expect, naivebayes.get_p_rentan(tpl, dic))