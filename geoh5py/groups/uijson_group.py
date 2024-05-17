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

from .group import Group, GroupType


class UIJsonGroup(Group):
    """Group for SimPEG inversions."""

    __TYPE_UID = uuid.UUID("{BB50AC61-A657-4926-9C82-067658E246A0}")

    _name = "UIJson"
    _description = "UIJson"
    _options = None

    def __init__(self, group_type: GroupType, name="UIJson", **kwargs):
        assert group_type is not None
        super().__init__(group_type, name=name, **kwargs)

        if self.entity_type.name == "Entity":
            self.entity_type.name = "UIJson"

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    @property
    def options(self) -> dict | None:
        """
        Metadata attached to the entity.
        """
        if getattr(self, "_options", None) is None:
            self._options = self.workspace.fetch_metadata(self.uid, argument="options")

        if self._options is None:
            self._options = {}

        return self._options

    @options.setter
    def options(self, value: dict | None):
        if not isinstance(value, (dict, type(None))):
            raise ValueError(f"Input 'options' must be of type {dict} or None")

        self._options = value
        self.workspace.update_attribute(self, "options")
