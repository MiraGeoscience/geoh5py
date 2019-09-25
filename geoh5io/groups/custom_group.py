from __future__ import annotations

import uuid
from typing import Optional

from .group import Group
from .group import GroupType


class CustomGroup(Group):
    """ A custom group, for an unlisted Group type.
    """
    def __init__(self, group_type: GroupType, name: str,
                 uid: uuid.UUID = None):
        super().__init__(group_type, name, uid)

    @classmethod
    def static_class_id(cls) -> Optional[uuid.UUID]:
        return None

    @classmethod
    def static_type_name(cls) -> Optional[str]:
        return None

    @classmethod
    def static_type_description(cls) -> Optional[str]:
        return None
