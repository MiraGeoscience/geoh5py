#  Copyright (c) 2023 Mira Geoscience Ltd.
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
from abc import ABC, abstractmethod

import numpy as np

from ...data import Data, ReferencedData
from ..curve import Curve
from ..object_type import ObjectType


class BaseElectrode(Curve, ABC):
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
            if data.dtype != np.int32:
                print("ab_cell_id values will be converted to type 'int32'")

            if any(self.get_data("A-B Cell ID")):
                child = self.get_data("A-B Cell ID")[0]
                if isinstance(child, ReferencedData):
                    child.values = data.astype(np.int32)
            else:
                if (
                    getattr(self, "current_electrodes", None) is not None
                    and getattr(self.current_electrodes, "ab_cell_id", None) is not None
                ):
                    entity_type = self.current_electrodes.ab_cell_id.entity_type
                else:
                    value_map = {ii: str(ii) for ii in range(data.max() + 1)}
                    value_map[0] = "Unknown"
                    entity_type = {
                        "primitive_type": "REFERENCED",
                        "value_map": value_map,
                    }

                data = self.add_data(
                    {
                        "A-B Cell ID": {
                            "values": data.astype(np.int32),
                            "association": "CELL",
                            "entity_type": entity_type,
                        }
                    }
                )

                if isinstance(data, ReferencedData):
                    self._ab_cell_id = data

    @property
    def ab_map(self) -> dict | None:
        """
        Get the ReferenceData.value_map of the ab_value_id
        """
        if isinstance(self.ab_cell_id, ReferencedData):
            return self.ab_cell_id.value_map
        return None

    def copy(
        self,
        parent=None,
        copy_children: bool = True,
        clear_cache: bool = False,
        extent: list[float] | np.ndarray | None = None,
        **kwargs,
    ):
        """
        Function to copy a survey to a different parent entity.

        :param parent: Target parent to copy the entity under. Copied to current
            :obj:`~geoh5py.shared.entity.Entity.parent` if None.
        :param copy_children: Create copies of all children entities along with it.
        :param clear_cache: Clear array attributes after copy.
        :param extent: Extent of the copied entity.

        :return entity: Registered Entity to the workspace.
        """
        if parent is None:
            parent = self.parent

        omit_list = [
            "_ab_cell_id",
            "_metadata",
            "_potential_electrodes",
            "_current_electrodes",
        ]
        new_entity = super().copy(
            parent=parent,
            clear_cache=clear_cache,
            copy_children=copy_children,
            extent=extent,
            **kwargs,
        )

        if isinstance(self, PotentialElectrode):
            complement = self.current_electrodes
        else:
            complement = self.potential_electrodes

        # Reset the extent of the complement
        if new_entity.ab_cell_id is not None and complement is not None:
            ab_cell_ids = np.unique(new_entity.ab_cell_id.values)
            indices = np.zeros(complement.n_vertices, dtype=bool)
            indices[complement.cells[ab_cell_ids, :]] = True

            new_complement = parent.workspace.copy_to_parent(
                complement,
                parent,
                omit_list=omit_list,
                clear_cache=clear_cache,
                mask=indices,
            )
            setattr(new_complement, "_ab_cell_id", None)
            if new_complement.ab_cell_id is None and complement.ab_cell_id is not None:
                parent.workspace.copy_to_parent(
                    complement.ab_cell_id,
                    new_complement,
                    mask=cell_indices,
                )

            if isinstance(self, PotentialElectrode):
                new_entity.current_electrodes = new_complement
            else:
                new_entity.potential_electrodes = new_complement

        return new_entity

    @property
    @abstractmethod
    def current_electrodes(self):
        """
        The associated current_electrodes (transmitters)
        """

    @property
    def metadata(self):
        """
        Metadata attached to the entity.
        """
        if getattr(self, "_metadata", None) is None:
            metadata = self.workspace.fetch_metadata(self.uid)
            self._metadata = metadata
        return self._metadata

    @metadata.setter
    def metadata(self, values):
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
            raise IndexError("Input Potential Electrodes uuid not present in Workspace")

        self._metadata = values
        self.workspace.update_attribute(self, "metadata")

    @property
    @abstractmethod
    def potential_electrodes(self):
        """
        The associated potential_electrodes (receivers)
        """


class PotentialElectrode(BaseElectrode):
    """
    Ground potential electrode (receiver).
    """

    __TYPE_UID = uuid.UUID("{275ecee9-9c24-4378-bf94-65f3c5fbe163}")

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

    @property
    def potential_electrodes(self):
        """
        The associated potential_electrodes (receivers)
        """
        return self

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID


class CurrentElectrode(BaseElectrode):
    """
    Ground direct current electrode (transmitter).
    """

    __TYPE_UID = uuid.UUID("{9b08bb5a-300c-48fe-9007-d206f971ea92}")

    def __init__(self, object_type: ObjectType, **kwargs):
        self._current_line_id: uuid.UUID | None = None

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
        potential_entity = self.workspace.get_entity(potential)[0]

        if isinstance(potential_entity, PotentialElectrode):
            return potential_entity

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
        if getattr(self, "cells", None) is None or self.n_cells is None:
            raise AttributeError(
                "Cells must be set before assigning default ab_cell_id"
            )

        data = np.arange(self.n_cells) + 1
        value_map = {ii: str(ii) for ii in range(self.n_cells + 1)}
        value_map[0] = "Unknown"
        ab_cell_id = self.add_data(
            {
                "A-B Cell ID": {
                    "values": data,
                    "association": "CELL",
                    "entity_type": {
                        "primitive_type": "REFERENCED",
                        "value_map": value_map,
                    },
                }
            }
        )
        if isinstance(ab_cell_id, ReferencedData):
            ab_cell_id.entity_type.name = "A-B"
            self._ab_cell_id = ab_cell_id
