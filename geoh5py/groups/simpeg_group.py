#  Copyright (c) 2023 Mira Geoscience Ltd Ltd.
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


class SimPEGGroup(Group):
    """Group for SimPEG inversions."""

    __TYPE_UID = uuid.UUID("{55ed3daf-c192-4d4b-a439-60fa987fe2b8}")

    _name = "SimPEG"
    _description = "SimPEG"
    _options = None

    def __init__(self, group_type: GroupType, **kwargs):
        assert group_type is not None
        super().__init__(group_type, **kwargs)

        if self.entity_type.name == "Entity":
            self.entity_type.name = "SimPEG"

        group_type.workspace._register_group(self)

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
        if value is not None:
            assert isinstance(
                value, dict
            ), f"Input 'options' must be of type {dict} or None"

        self._options = value
        self.workspace.update_attribute(self, "options")
