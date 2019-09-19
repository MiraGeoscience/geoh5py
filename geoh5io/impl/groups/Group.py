#!/usr/bin/env python3
from shared.Entity import Entity


class Group(Entity):
    def __init__(self):
        self.allowMove = 1
        self.clippingIDs = []
