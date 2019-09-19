#!/usr/bin/env python3
from shared.Type import Type


class GroupType(Type):
    """
    UUID for Container type: {61FBB4E8-A480-11E3-8D5A-2776BDF4F982}
    UUID for Drillhole group type: {825424FB-C2C6-4FEA-9F2B-6CD00023D393}
    """

    def __init__(self):
        self.classId = None
        self.allowMoveContent = 1
        self.allowDeleteContent = 1
