#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of geoh5py.
#
#  geoh5py is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  geoh5py is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from ..shared import EntityType

if TYPE_CHECKING:
    from ..workspace import Workspace


class ObjectType(EntityType):
    """
    Object type class
    """

    @staticmethod
    def create_custom(workspace: Workspace) -> ObjectType:
        """Creates a new instance of ObjectType for an unlisted custom Object type with a
        new auto-generated UUID.

        :param workspace: An active Workspace class
        """
        return ObjectType(workspace)

    @classmethod
    def find_or_create(cls, workspace: Workspace, **kwargs) -> ObjectType:
        """
        Find or creates an EntityType with given uid that matches the given
        Group implementation class.

        It is expected to have a single instance of EntityType in the Workspace
        for each concrete Entity class.

        To find an object, the kwargs must contain an existing 'uid' keyword,
        or a 'entity_class' keyword, containing an object class.

        :param workspace: An active Workspace class

        :return: A new instance of GroupType.
        """
        if (
            getattr(kwargs.get("entity_class", None), "default_type_uid", None)
            is not None
        ):
            uid = kwargs["entity_class"].default_type_uid()
            kwargs["uid"] = uid
        else:
            uid = kwargs.get("uid", None)
            uid = kwargs.get("ID", uid)
            if isinstance(uid, str):
                uid = uuid.UUID(uid)
        if isinstance(uid, uuid.UUID):
            entity_type = cls.find(workspace, uid)
            if entity_type is not None:
                return entity_type

        if not isinstance(uid, (uuid.UUID, type(None))):
            raise TypeError(f"'uid' must be a valid UUID, find {uid}")

        return cls(workspace, **kwargs)
