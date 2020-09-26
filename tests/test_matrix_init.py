import unittest
import numpy as np
from sc_jnmf.matrix_init import *


class TestMatrixInit(unittest.TestCase):
    def test_random_init(self):

        d1 = np.random.randint(4, size=(10, 10))
        d2 = np.random.randint(5, size=(10, 10))
        rank = 3

        w1, w2, h = random_init(d1, d2, rank)

        Is_non_negative = True
        Is_non_negative *= np.all(w1 >= 0)
        Is_non_negative *= np.all(w2 >= 0)
        Is_non_negative *= np.all(h >= 0)
        self.assertTrue(Is_non_negative)


if __name__ == '__main__':
    unittest.main()
