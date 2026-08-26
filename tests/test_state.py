import numpy as np
import unittest

from sbayes.sampling.state import CacheNode, GroupedParameters


class TestArrayParameter(unittest.TestCase):

    N_GROUPS = 3
    N_ITEMS = 4

    def setUp(self) -> None:
        arr = np.arange(12).reshape((self.N_GROUPS, self.N_ITEMS))
        self.param = GroupedParameters(arr)
        self.calc = CacheNode(np.empty((self.N_GROUPS, self.N_ITEMS)))

    def test_initial_state(self):
        self.assertEqual(self.param.value[1, 2], 6)
        self.assertEqual(self.param.version, 0)

    def test_set_items(self):
        # Change parameter using set_items
        self.param.set_items((1, 2), 1000)

        # Validate changes in value and version number
        self.assertEqual(self.param.value[1, 2], 1000)
        self.assertEqual(self.param.version, 1)

    def test_edit(self):
        # Change parameter using the .edit() context manager
        with self.param.edit() as value:
            value[1, 2] = 1000

        # Validate changes in value and version number
        self.assertEqual(self.param.value[1, 2], 1000)
        self.assertEqual(self.param.version, 1)

    def test_set_value(self):
        # Change parameter using the .edit() context manager
        new_value = self.param.value.copy()
        new_value[1, 2] = 1000
        self.param.set_value(new_value)

        # Validate changes in value and version number
        self.assertEqual(self.param.value[1, 2], 1000)
        self.assertEqual(self.param.version, 1)


if __name__ == '__main__':
    unittest.main()
