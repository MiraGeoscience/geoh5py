#!/usr/bin/env python3
from objects.Object import Object


class BlockModel(Object):
    """DataType UUID : {B020A277-90E2-4CD7-84D6-612EE3F25051}"""

    def __init__(self):
        self.origin = None
        self.rotation = 0
        self.uCellDelimiters = []
        self.vCellDelimiters = []
        self.zCellDelimiters = []
