#!/usr/bin/env python3
from objects.Object import Object


class Grid2D(Object):
    """DataType UUID : {48f5054a-1c5c-4ca4-9048-80f36dc60a06}"""

    def __init__(self):
        self.origin = None
        self.uSize = None
        self.vSize = None
        self.uCount = None
        self.vCount = None
        self.rotation = 0
        self.isVertical = 0
