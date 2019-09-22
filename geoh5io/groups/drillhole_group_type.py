from __future__ import annotations

import uuid

from . import GroupType


class DrillholeGroupType(GroupType):
    """ The type for the group containing drillholes."""

    __class_id = uuid.UUID("{825424FB-C2C6-4FEA-9F2B-6CD00023D393}")

    def __init__(self, uid, name=None, description=None, class_id=None):
        super().__init__(uid, name, description, class_id)

    @classmethod
    def static_class_id(cls) -> uuid.UUID:
        return cls.__class_id

    @classmethod
    def create(cls) -> DrillholeGroupType:
        """ Creates a new instance of DrillholeGroupType with the UUID dedicated to this class.

        The same UUID is used for class_id. All created instances of DrillholeGroupType share the same UUID.
        It is actually expected to have a single instance of this class in a Workspace.
        """
        return DrillholeGroupType(
            cls.__class_id, "Drillholes", "Drillholes", cls.__class_id
        )
