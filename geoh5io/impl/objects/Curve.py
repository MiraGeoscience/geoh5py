#!/usr/bin/env python3
from objects.Object import Object


class Curve(Object):
    """DataType UUID :  {6A057FDC-B355-11E3-95BE-FD84A7FFCB88}"""

    def __init__(self):
        self.vertices = []
        self.cells = []
