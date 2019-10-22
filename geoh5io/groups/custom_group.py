import uuid
from typing import Optional

from .group import Group


class CustomGroup(Group):
    """ A custom group, for an unlisted Group type.
    """

    @classmethod
    def default_type_uid(cls) -> Optional[uuid.UUID]:
        raise RuntimeError(f"No predefined static type UUID for {cls}.")
        # return None

    @classmethod
    def default_type_name(cls) -> Optional[str]:
        return None

    @classmethod
    def default_type_description(cls) -> Optional[str]:
        return None
