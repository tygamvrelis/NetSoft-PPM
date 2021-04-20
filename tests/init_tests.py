#!/usr/bin/python
# Author: Tyler Gamvrelis
# Some nonsense to be able to run tests without installing as a package

import os
import sys
ppm_path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.insert(0, os.path.abspath(os.path.join(ppm_path, 'ppm')))
