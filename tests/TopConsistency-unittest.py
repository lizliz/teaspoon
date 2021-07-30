"""
This is a script to run tests of the Topological Consistency module

- Liz Munch 7/21
"""

import unittest

from ripser import ripser
import teaspoon.TDA.TopologicalConsistency as TopCons



class TestOmerApprox(unittest.TestCase):

    def test_circle_function_single_level(self):
        """
        Checking that you can determine that a funtion that looks like an annulus
        has a single long bar
        """


        def f(x,y):
            r = np.sqrt(x**2 + y**2)
            # function value is distance from the circle

            return max( ( 1 - np.abs(1-r) ) **3, 0)

        L = .6
        eps = 0.05
        r = 0.5
        TopCons.OmerApprox(func = f, L = L, eps = eps,
                    r = r, domain = [-2,2,-2,2],
                    N = 300 )

        self.assertEqual(countLongBars, 1, "An annulus should have one long bar")


if __name__ == '__main__':
    unittest.main()
