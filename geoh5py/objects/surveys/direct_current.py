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
from typing import Dict, Optional, Union

from ..curve import Curve
from ..object_type import ObjectType


class PotentialElectrode(Curve):
    """
    Ground potential electrode (receiver).

    .. warning:: Partially implemented.

    """

    __TYPE_UID = uuid.UUID("{275ecee9-9c24-4378-bf94-65f3c5fbe163}")

    def __init__(self, object_type: ObjectType, **kwargs):
        self._metadata = None
        super().__init__(object_type, **kwargs)

    @property
    def metadata(self):
        return self._metadata

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID


class CurrentElectrode(Curve):
    """
    Ground direct current electrode (transmitter).

    .. warning:: Partially implemented.

    """

    __TYPE_UID = uuid.UUID("{9b08bb5a-300c-48fe-9007-d206f971ea92}")

    def __init__(self, object_type: ObjectType, **kwargs):
        self._current_line_id: Optional[uuid.UUID] = None
        self._metadata: Optional[Dict[uuid.UUID, uuid.UUID]] = None

        super().__init__(object_type, **kwargs)

    @property
    def potentials(self) -> Optional[PotentialElectrode]:
        """
        The associated potentials (receivers)
        """
        assert self.metadata is not None, "No Current-Receiver metadata set."
        potential = self.metadata[self.uid]

        try:
            return self.workspace.get_entity(potential)[0]
        except IndexError:
            print("Associate Potential entity not found in Workspace.")
            return None

    @potentials.setter
    def potentials(self, value: Union[PotentialElectrode, uuid.UUID]):
        assert isinstance(value, PotentialElectrode), (
            f"Provided potentials must be of type {PotentialElectrode}. "
            f"{value} provided."
        )

        if isinstance(value, PotentialElectrode):
            value = value.uid

        self.metadata = {self.uid, value}

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, values: Dict[uuid.UUID, uuid.UUID]):

        assert (
            len(values) == 2
        ), f"Metadata must have two key-value pairs. {values} provided."

        # try:
        #     assert value["Current Electrodes"]
        for key, value in values.items():
            assert (
                key in self.workspace.list_objects_name.keys()
            ), "Provided source uuid not in workspace."
            source = self.workspace.get_entity(key)[0]
            assert isinstance(source, CurrentElectrode), (
                f"Provided uuid for source must be of type {CurrentElectrode}. "
                f"Object uuid of type {type(source)} provided"
            )
            assert (
                value in self.workspace.list_objects_name.keys()
            ), "Provided receiver uuid not in workspace."
            receiver = self.workspace.get_entity(value)[0]
            assert isinstance(receiver, PotentialElectrode), (
                f"Provided uuid for receiver must be of type {PotentialElectrode}. "
                f"Object uuid of type {type(receiver)} provided"
            )

        self._metadata = values
        # self.receiver.metadata = dict

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID
