#!/usr/bin/python
# Author: Tyler Gamvrelis
# Cost functions

# Standard library imports
from abc import ABC, abstractmethod
import json
import logging
import os

# Third party imports
from jsonschema import validate

# Local application imports
from utils import get_schema_dir

# Globals
logger = logging.getLogger(__name__)
SCHEMA_DIR = get_schema_dir()
with open(os.path.join(SCHEMA_DIR, 'cost_func.schema.json')) as f:
    COST_FUNC_SCHEMA = json.load(f)

class CostFunc(ABC):
    """Base class for all cost functions."""

    def __init__(self):
        pass

    @abstractmethod
    def get_min_marginal_cost(self):
        """Returns the derivative of the cost function, evaluated at 0."""
        pass
    
    @abstractmethod
    def get_max_marginal_cost(self):
        """Returns the derivative of the cost function, evaluated at 1."""
        pass

    @abstractmethod
    def to_json(self):
        """Returns a JSON representation of the cost function."""
        pass

    @abstractmethod
    def __call__(self, x):
        pass

    def __eq__(self, other):
        return str(self) == str(other)

    @abstractmethod
    def __str__(self):
        pass

class PowerFunc(CostFunc):
    """Power function, of the form a * (y ^ s), where y is the variable."""
    
    def __init__(self, a=-1, s=-1, a_digits=3, s_digits=3, json=None):
        super().__init__()
        if json == None:
            assert a > 0, "a must be greater than 0"
            assert s > 1, "s must be greater than 1"
            self.a = a
            self.s = s
        else:
            validate(json, COST_FUNC_SCHEMA)
            self.a = json['a']
            self.s = json['s']

        # The two constants below only come into play when the function's
        # string representation is requested
        self._A_DIGITS = a_digits
        self._S_DIGITS = s_digits

    def get_min_marginal_cost(self):
        return 0
    
    def get_max_marginal_cost(self):
        return self.s * self.a

    def inverse(self, y):
        """Returns the inverse of the function, evaluated at y."""
        return (y / (self.s * self.a)) ** (1 / (self.s - 1))
    
    def to_json(self):
        json = {}
        json['func'] = 'PowerFunc'
        json['s'] = self.s
        json['a'] = self.a
        validate(json, COST_FUNC_SCHEMA)
        return json

    def __call__(self, x):
        return self.a * (x ** self.s)

    def __str__(self):
        sb = ''
        sb += 'Power function '
        sb += 'y = '
        fstr = '{:.' + str(self._A_DIGITS) + 'e}'
        sb += fstr.format(self.a)
        sb += ' * x ** '
        fstr = '{:.' + str(self._S_DIGITS) + 'e}'
        sb += fstr.format(self.s)
        return sb

class CostFuncFactory:
    @classmethod
    def get_cost_func(self, json):
        """
        Creates a cost function based on the given JSON.

        Args:
            json : dict
                JSON description of a cost function
        
        Returns: CostFunc instance described by the JSON
        """
        validate(json, COST_FUNC_SCHEMA)
        if json['func'] == 'PowerFunc':
            return PowerFunc(json=json)
