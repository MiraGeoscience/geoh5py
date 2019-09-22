import uuid
from typing import List

from . import GroupType
from geoh5io.shared import Entity


class Group(Entity):
    def __init__(self, type: GroupType):
        super().__init__()
        self._allow_move = True
        self._clipping_ids: List[uuid.UUID] = []
        self._type = type
