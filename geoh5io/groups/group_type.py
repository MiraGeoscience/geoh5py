from __future__ import annotations

import uuid
from typing import Optional

from geoh5io.shared import Type


class GroupType(Type):
    def __init__(self, uid, name=None, description=None, class_id: uuid.UUID = None):
        super().__init__(uid, name, description)

        self._class_id = class_id
        self._allow_move_content = True
        self._allow_delete_content = True

    @property
    def allow_move_content(self) -> bool:
        return self._allow_move_content

    @allow_move_content.setter
    def allow_move_content(self, allow: bool):
        self._allow_move_content = bool(allow)

    @property
    def allow_delete_content(self) -> bool:
        return self._allow_delete_content

    @allow_delete_content.setter
    def allow_delete_content(self, allow: bool):
        self._allow_delete_content = bool(allow)

    @property
    def class_id(self) -> Optional[uuid.UUID]:
        return self._class_id
