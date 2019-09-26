from __future__ import annotations

import uuid
from typing import Optional

from .group import Group, GroupType


class CustomGroup(Group):
    """ A custom group, for an unlisted Group type.
    """

    def __init__(self, group_type: GroupType, name: str, uid: uuid.UUID = None):
        super().__init__(group_type, name, uid)

    @classmethod
    def static_type_uid(cls) -> Optional[uuid.UUID]:
        raise RuntimeError(f"No predefined static type UUID for {cls}.")

    @classmethod
    def static_type_name(cls) -> Optional[str]:
        return None

    @classmethod
    def static_type_description(cls) -> Optional[str]:
        return None
