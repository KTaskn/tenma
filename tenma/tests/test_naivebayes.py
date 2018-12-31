# coding:utf-8
import unittest
import os
import numpy as np
import pandas as pd
from tenma import naivebayes as nb


class TestNaivebayes(unittest.TestCase):

    def test_P_A(self):
        col_A = 'col_1'
        df = pd.DataFrame({
            col_A: [1, 1, 1, 2, 2, 4]
        })
        
        expect = np.log(0.5)
        A = 1
        self.assertEqual(expect, nb.P_A(df, col_A, A))

        #expect = np.log(0.0)
        expect = -50.0
        A = 0
        self.assertEqual(expect, nb.P_A(df, col_A, A))

    def test_P_A_notcol(self):
        col_A = 'col_1'
        df = pd.DataFrame({
            "noname": [1, 1, 1, 2, 2, 4]
        })
        
        A = 0
        with self.assertRaises(KeyError):
            nb.P_A(df, col_A, A)


    def test_P_A_len0(self):
        col_A = 'col_1'
        df = pd.DataFrame({
            col_A: []
        })
        
        A = 0
        with self.assertRaises(RuntimeError):
            nb.P_A(df, col_A, A)


    def test_P_AB(self):
        col_A = 'col_1'
        col_B = 'col_2'
        df = pd.DataFrame({
            col_B: [1, 1, 2, 3, 3, 4],
            col_A: [1, 1, 1, 2, 5, 4]
        })
        
        expect = np.log(1.0)
        B = 1
        A = 1
        self.assertEqual(expect, nb.P_AB(df, col_A, A, col_B, B))

        expect = np.log(0.5)
        B = 3
        A = 2
        self.assertEqual(expect, nb.P_AB(df, col_A, A, col_B, B))

        #expect = np.log(0.0)
        expect = -50.0
        B = 5
        A = 2
        self.assertEqual(expect, nb.P_AB(df, col_A, A, col_B, B))


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
            nb.P_AB(df, col_A, A, col_B, B)

        
        col_A = 'col_1'
        col_B = 'col_2'
        df = pd.DataFrame({
            col_A: [1, 1, 1, 2, 2, 4],
            "noname": [1, 2, 3, 4, 5, 6]
        })
        
        A = 0
        B = 1
        with self.assertRaises(KeyError):
            nb.P_AB(df, col_A, A, col_B, B)


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
            nb.P_AB(df, col_A, A, col_B, B)

    def test_P_BA(self):
        col_A = 'col_1'
        col_B = 'col_2'
        df = pd.DataFrame({
            col_B: [1, 1, 0, 0],
            col_A: [1, 0, 1, 1]
        })

        P_B = np.log(1 / 2)

        P_A = 3 / 4
        P_AB = 1 / 2
        
        expect = np.log(P_AB) + P_B - np.log(P_A)
        B = 1
        A = 1
        self.assertEqual(expect, nb.P_BA(df, P_B, col_A, A, col_B, B))
