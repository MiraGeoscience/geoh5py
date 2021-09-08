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

import numpy as np

from ...data import Data, ReferencedData
from ..curve import Curve
from ..object_type import ObjectType


class PotentialElectrode(Curve):
    """
    Ground potential electrode (receiver).
    """

    __TYPE_UID = uuid.UUID("{275ecee9-9c24-4378-bf94-65f3c5fbe163}")

    def __init__(self, object_type: ObjectType, **kwargs):
        self._metadata: dict | None = None
        self._ab_cell_id: ReferencedData | None = None

        super().__init__(object_type, **kwargs)

    @property
    def ab_cell_id(self) -> ReferencedData | None:
        """
        Reference data entity mapping cells to a unique current dipole.
        """
        if getattr(self, "_ab_cell_id", None) is None:
            child = self.get_data("A-B Cell ID")
            if any(child) and isinstance(child[0], ReferencedData):
                self.ab_cell_id = child[0]

        if getattr(self, "_ab_cell_id", None) is not None:
            return self._ab_cell_id

        return None

    @ab_cell_id.setter
    def ab_cell_id(self, data: Data | np.ndarray):
        if isinstance(data, Data):
            if not isinstance(data, ReferencedData):
                raise TypeError(f"ab_cell_id must be of type {ReferencedData}")
            self._ab_cell_id = data

        else:
            if not data.dtype == "int32":
                raise TypeError("ab_cell_id must be of type 'int32'")

            if any(self.get_data("A-B Cell ID")):
                child = self.get_data("A-B Cell ID")[0]
                if isinstance(child, ReferencedData):
                    child.values = data
            else:
                if (
                    getattr(self, "current_electrodes", None) is not None
                    and getattr(self.current_electrodes, "ab_cell_id", None) is not None
                ):
                    entity_type = self.current_electrodes.ab_cell_id.entity_type
                else:
                    entity_type = {"primitive_type": "REFERENCED"}

                data = self.add_data(
                    {
                        "A-B Cell ID": {
                            "values": data,
                            "association": "CELL",
                            "entity_type": entity_type,
                        }
                    }
                )

                if isinstance(data, ReferencedData):
                    self._ab_cell_id = data

        self.workspace.finalize()

    @property
    def ab_map(self) -> dict | None:
        """
        Get the ReferenceData.value_map of the ab_value_id
        """
        if isinstance(self.ab_cell_id, ReferencedData):
            return self.ab_cell_id.value_map
        return None

    @property
    def current_electrodes(self):
        """
        The associated current electrode object (sources).
        """
        if self.metadata is None:
            raise AttributeError("No Current-Receiver metadata set.")
        currents = self.metadata["Current Electrodes"]

        try:
            return self.workspace.get_entity(currents)[0]
        except IndexError:
            print("Associated CurrentElectrode entity not found in Workspace.")
            return None

    @current_electrodes.setter
    def current_electrodes(self, current_electrodes: CurrentElectrode):
        if not isinstance(current_electrodes, CurrentElectrode):
            raise TypeError(
                f"Provided current_electrodes must be of type {CurrentElectrode}. "
                f"{type(current_electrodes)} provided."
            )

        metadata = {
            "Current Electrodes": current_electrodes.uid,
            "Potential Electrodes": self.uid,
        }

        self.metadata = metadata
        current_electrodes.metadata = metadata

        if isinstance(current_electrodes.ab_cell_id, ReferencedData) and isinstance(
            self.ab_cell_id, ReferencedData
        ):
            self.ab_cell_id.entity_type = current_electrodes.ab_cell_id.entity_type
        self.workspace.finalize()

    @property
    def potential_electrodes(self):
        """
        The associated potential_electrodes (receivers)
        """
        return self

    @property
    def metadata(self) -> dict | None:
        """
        Metadata attached to the entity.
        """
        if getattr(self, "_metadata", None) is None:
            metadata = self.workspace.fetch_metadata(self.uid)
            for key, value in metadata.items():
                metadata[key] = uuid.UUID(value)

            self._metadata = metadata
        return self._metadata

    @metadata.setter
    def metadata(self, values: dict[str, uuid.UUID]):
        if not len(values) == 2:
            raise ValueError(
                f"Metadata must have two key-value pairs. {values} provided."
            )

        default_keys = ["Current Electrodes", "Potential Electrodes"]

        if list(values.keys()) != default_keys:
            raise ValueError(f"Input metadata must have for keys {default_keys}")

        if self.workspace.get_entity(values["Current Electrodes"])[0] is None:
            raise IndexError("Input Current Electrodes uuid not present in Workspace")

        if self.workspace.get_entity(values["Potential Electrodes"])[0] is None:
            raise IndexError("Input Current Electrodes uuid not present in Workspace")

        self._metadata = values
        self.modified_attributes = "metadata"

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID


class CurrentElectrode(PotentialElectrode):
    """
    Ground direct current electrode (transmitter).
    """

    __TYPE_UID = uuid.UUID("{9b08bb5a-300c-48fe-9007-d206f971ea92}")

    def __init__(self, object_type: ObjectType, **kwargs):
        self._current_line_id: uuid.UUID | None
        self._metadata: dict[uuid.UUID, uuid.UUID] | None

        super().__init__(object_type, **kwargs)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID

    @property
    def current_electrodes(self):
        """
        The associated current electrode object (sources).
        """
        return self

    @current_electrodes.setter
    def current_electrodes(self, _):
        ...

    @property
    def potential_electrodes(self) -> PotentialElectrode | None:
        """
        The associated potential_electrodes (receivers)
        """
        if self.metadata is None:
            raise AttributeError("No Current-Receiver metadata set.")

        potential = self.metadata["Potential Electrodes"]

        try:
            return self.workspace.get_entity(potential)[0]
        except IndexError:
            print("Associated PotentialElectrode entity not found in Workspace.")
            return None

    @potential_electrodes.setter
    def potential_electrodes(self, potential_electrodes: PotentialElectrode):
        if not isinstance(potential_electrodes, PotentialElectrode):
            raise TypeError(
                f"Provided potential_electrodes must be of type {PotentialElectrode}. "
                f"{type(potential_electrodes)} provided."
            )

        metadata = {
            "Current Electrodes": self.uid,
            "Potential Electrodes": potential_electrodes.uid,
        }

        self.metadata = metadata
        potential_electrodes.metadata = metadata

        if isinstance(potential_electrodes.ab_cell_id, ReferencedData) and isinstance(
            self.ab_cell_id, ReferencedData
        ):
            potential_electrodes.ab_cell_id.entity_type = self.ab_cell_id.entity_type

    def add_default_ab_cell_id(self):
        """
        Utility function to set ab_cell_id's based on curve cells.
        """
        if getattr(self, "cells", None) is None:
            raise AttributeError(
                "Cells must be set before assigning default ab_cell_id"
            )

        data = np.arange(self.n_cells) + 1
        value_map = {ii: str(ii) for ii in range(self.n_cells + 1)}
        value_map[0] = "Unknown"
        self._ab_cell_id = self.add_data(
            {
                "A-B Cell ID": {
                    "values": data,
                    "association": "CELL",
                    "entity_type": {"primitive_type": "REFERENCED"},
                    "value_map": value_map,
                }
            }
        )
        self._ab_cell_id.entity_type.name = "A-B"
