#!/usr/bin/env python3
from enum import IntEnum


class DataAssociation(IntEnum):
    UNKNOWN = 0
    OBJECT = 1
    CELL = 2
    VERTEX = 3
    FACE = 4
