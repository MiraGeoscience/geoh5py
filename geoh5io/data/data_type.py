#!/usr/bin/env python3
from shared.Type import Type


class DataType(Type):
    def __init__(self):
        self.colorMap = []
        self.valueMap = []
        self.primitiveType = None
        self.units = None
