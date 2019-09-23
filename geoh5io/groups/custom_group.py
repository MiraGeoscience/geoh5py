from __future__ import annotations

import uuid
from typing import Optional

from geoh5io.groups import Group


class CustomGroup(Group):
    """ A custom group, for an unlisted Group type.
    """

    @classmethod
    def static_class_id(cls) -> Optional[uuid.UUID]:
        return None

    @classmethod
    def static_type_name(cls) -> Optional[str]:
        return None

    @classmethod
    def static_type_description(cls) -> Optional[str]:
        return None
