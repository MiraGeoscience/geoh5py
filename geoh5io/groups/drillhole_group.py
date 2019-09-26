from __future__ import annotations

import uuid

from .group import Group


class DrillholeGroup(Group):
    """ The type for the group containing drillholes."""

    __type_uid = uuid.UUID("{825424FB-C2C6-4FEA-9F2B-6CD00023D393}")

    @classmethod
    def static_type_uid(cls) -> uuid.UUID:
        return cls.__type_uid

    @classmethod
    def static_type_name(cls) -> str:
        return "Drillholes"
