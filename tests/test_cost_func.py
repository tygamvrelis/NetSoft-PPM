#!/usr/bin/python
# Author: Tyler Gamvrelis
# Test cost functions

# Standard library imports
import os
import sys
import unittest

# Third party imports
from jsonschema import validate, ValidationError, SchemaError, Draft7Validator

# Local application imports
import init_tests
from cost_func import *

class TestCostFunc(unittest.TestCase):
    def test_cost_func_schema_is_valid(self):
        raised = False
        msg = ''
        try:
            Draft7Validator.check_schema(COST_FUNC_SCHEMA)
        except SchemaError as e:
            raised = True
            msg = e
        self.assertFalse(raised, msg)

class TestPowerFunc(unittest.TestCase):
    def test_simple_cost_func_is_schema_compliant(self):
        a_cpu = 0.223
        s_cpu = 3
        cf = PowerFunc(a_cpu, s_cpu)
        json = cf.to_json()
        raised = False
        msg = ''
        try:
            validate(json, COST_FUNC_SCHEMA)
        except ValidationError as e:
            raised = True
            msg = e
        self.assertFalse(raised, msg)

    def test_cost_func_can_be_constructed_from_its_json(self):
        # Setup
        a_cpu = 0.223
        s_cpu = 3
        cf = PowerFunc(a_cpu, s_cpu)
        json = cf.to_json()
        # Test
        new_cf = PowerFunc(json=json)
        self.assertTrue(cf == new_cf)


if __name__ == "__main__":
    unittest.main()
