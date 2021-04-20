#!/usr/bin/python
# Author: Tyler Gamvrelis

# Standard library imports
from enum import Enum

class ResourceType(Enum):
    CPU = 1
    MEM = 2
    GPU = 3
    DISK = 4
    LINK = 5
