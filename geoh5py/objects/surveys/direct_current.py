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
from __future__ import annotations

import uuid

from ...data import Data
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
    def _ab_cell_id(self) -> Data:
        """
        Data object
        """
        ab_cell_id = self.get_data("A-B Cell ID")
        assert any(ab_cell_id), "No 'A-B Cell ID' found on the object."

        return ab_cell_id[0]

    @property
    def metadata(self) -> dict | None:
        """
        Metadata attached to the entity.
        """
        if getattr(self, "_metadata", None) is None:
            self._metadata = self.workspace.fetch_metadata(self.uid)

        return self._metadata

    @metadata.setter
    def metadata(self, values: dict[str, uuid.UUID]):

        assert (
            len(values) == 2
        ), f"Metadata must have two key-value pairs. {values} provided."

        default_keys = ["Current Electrodes", "Potential Electrodes"]
        assert (
            values.keys() == default_keys
        ), f"Input metadata must have for keys {default_keys}"

        if not self.workspace.get_entity(values["Current Electrodes"]):
            raise IndexError("Input Current Electrodes uuid not present in Workspace")

        if not self.workspace.get_entity(values["Potential Electrodes"]):
            raise IndexError("Input Current Electrodes uuid not present in Workspace")

        self._metadata = values

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID


class CurrentElectrode(PotentialElectrode):
    """
    Ground direct current electrode (transmitter).

    .. warning:: Partially implemented.

    """

    __TYPE_UID = uuid.UUID("{9b08bb5a-300c-48fe-9007-d206f971ea92}")

    def __init__(self, object_type: ObjectType, **kwargs):
        self._current_line_id: uuid.UUID | None
        self._metadata: dict[uuid.UUID, uuid.UUID] | None

        super().__init__(object_type, **kwargs)

    @property
    def potentials(self) -> PotentialElectrode | None:
        """
        The associated potentials (receivers)
        """
        assert self.metadata is not None, "No Current-Receiver metadata set."
        potential = self.metadata["Potential Electrodes"]

        try:
            return self.workspace.get_entity(potential)[0]
        except IndexError:
            print("Associate Potential entity not found in Workspace.")
            return None

    @potentials.setter
    def potentials(self, value: PotentialElectrode | uuid.UUID):
        if isinstance(value, uuid.UUID):
            value = self.workspace.get_entity(value)[0]

        assert isinstance(value, PotentialElectrode), (
            f"Provided potentials must be of type {PotentialElectrode}. "
            f"{value} provided."
        )

        metadata = {"Current Electrodes": self.uid, "Potential Electrodes": value}

        self.metadata = metadata
        value.metadata = metadata

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID
