#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import unittest

from sbayes.tools import extract_feature_states


class TestExctactFeatureStates(unittest.TestCase):

    """Simple validity test of ´extract_feature_states´ script."""

    def test_extract_feature_states(self):

        input_path = 'test/test_files/features.csv'
        output_path = 'test/test_files/feature_states.csv'
        expected_output_path = 'test/test_files/feature_states_expected.csv'

        extract_feature_states.main(['--input', input_path,
                                     '--output', output_path])

        with open(output_path, 'r') as output_file:
            output = output_file.read()
        with open(expected_output_path, 'r') as expected_output_file:
            expected_output = expected_output_file.read()

        assert output == expected_output

        os.remove(output_path)


if __name__ == '__main__':
    unittest.main()
