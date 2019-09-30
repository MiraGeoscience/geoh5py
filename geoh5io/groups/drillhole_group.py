import uuid
from typing import Optional

from .group import Group


class DrillholeGroup(Group):
    """ The type for the group containing drillholes."""

    __TYPE_UID = uuid.UUID(
        fields=(0x825424FB, 0xC2C6, 0x4FEA, 0x9F, 0x2B, 0x6CD00023D393)
    )
    __CLASS_UID = uuid.UUID(
        fields=(0x177101E8, 0x24F7, 0x4DBA, 0x81, 0xBA, 0x2722E3652CFE)
    )

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    @classmethod
    def default_class_id(cls) -> Optional[uuid.UUID]:
        return cls.__CLASS_UID

    @classmethod
    def default_type_name(cls) -> str:
        return "Drillholes"
