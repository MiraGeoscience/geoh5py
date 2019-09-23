from __future__ import annotations

import uuid

from geoh5io.groups import Group


class DrillholeGroup(Group):
    """ The type for the group containing drillholes."""

    __class_id = uuid.UUID("{825424FB-C2C6-4FEA-9F2B-6CD00023D393}")
    __type_name = "Drillholes"

    @classmethod
    def static_class_id(cls) -> uuid.UUID:
        return cls.__class_id

    @classmethod
    def static_type_name(cls) -> str:
        return cls.__type_name

    @classmethod
    def static_type_description(cls) -> str:
        return cls.static_type_name()
