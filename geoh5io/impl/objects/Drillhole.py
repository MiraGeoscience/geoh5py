#!/usr/bin/env python3
from objects.Object import Object


class Drillhole(Object):
    """DataType UUID : {7CAEBF0E-D16E-11E3-BC69-E4632694AA37}"""

    def __init__(self):
        self.vertices = []
        self.cells = []
        self.collar = None
        self.surveys = []
        self.trace = []
