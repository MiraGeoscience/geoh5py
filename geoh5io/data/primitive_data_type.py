#!/usr/bin/env python3
from enum import IntEnum


class PrimitiveDataType(IntEnum):
    UNKNOWN = 0
    INTEGER = 1
    FLOAT = 2
    REFERENCED = 3
    TEXT = 4
    FILENAME = 5
    DATETIME = 6
    BLOB = 7
