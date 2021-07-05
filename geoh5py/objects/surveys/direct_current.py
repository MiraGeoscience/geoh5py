#  Copyright (c) 2021 Mira Geoscience Ltd.
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
import uuid
from typing import Optional

from ..curve import Curve
from ..object_type import ObjectType


class CurrentElectrode(Curve):
    """
    Ground direct current electrode (transmitter).

    .. warning:: Partially implemented.

    """

    _attribute_map = {
        "Allow delete": "allow_delete",
        "Allow move": "allow_move",
        "Allow rename": "allow_rename",
        "ID": "uid",
        "Current line property ID": "current_line_id",
        "Last focus": "last_focus",
        "Name": "name",
        "Public": "public",
    }

    __TYPE_UID = uuid.UUID("{9b08bb5a-300c-48fe-9007-d206f971ea92}")

    def __init__(self, object_type: ObjectType, **kwargs):
        self._current_line_id: Optional[uuid.UUID] = None

        super().__init__(object_type, **kwargs)

    @property
    def current_line_id(self):

        if getattr(self, "_current_line_id", None) is None:
            self._current_line_id = uuid.uuid4()

        return self._current_line_id

    @current_line_id.setter
    def current_line_id(self, value: uuid.UUID):
        assert isinstance(value, uuid.UUID), (
            f"Input current_line_id value should be of type {uuid.UUID}."
            f" {type(value)} provided"
        )
        self._current_line_id = value
        self.modified_attributes = "attributes"

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID


class PotentialElectrode(CurrentElectrode):
    """
    Ground potential electrode (receiver).

    .. warning:: Partially implemented.

    """

    __TYPE_UID = uuid.UUID("{275ecee9-9c24-4378-bf94-65f3c5fbe163}")

    def __init__(self, object_type: ObjectType, **kwargs):
        super().__init__(object_type, **kwargs)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID
