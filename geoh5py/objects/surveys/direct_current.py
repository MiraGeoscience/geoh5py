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

# pylint: disable=too-many-ancestors

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from typing import cast

import numpy as np

from ...data import Data, ReferencedData
from ..curve import Curve
from ..object_type import ObjectType

logger = logging.getLogger(__name__)


class BaseElectrode(Curve, ABC):
    _potential_electrodes: PotentialElectrode | None = None
    _current_electrodes: CurrentElectrode | None = None

    def __init__(self, object_type: ObjectType, **kwargs):
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

            if data.parent.uid == self.uid:
                self._ab_cell_id = data
            else:
                self._ab_cell_id = cast(ReferencedData, data.copy(parent=self))
        else:
            if data.dtype != np.int32:
                logger.info("ab_cell_id values will be converted to type 'int32'")

            if any(self.get_data("A-B Cell ID")):
                child = self.get_data("A-B Cell ID")[0]
                if isinstance(child, ReferencedData):
                    child.values = data.astype(np.int32)
            else:
                complement: CurrentElectrode | PotentialElectrode = (
                    self.current_electrodes
                    if isinstance(self, PotentialElectrode)
                    else self.potential_electrodes
                )

                if complement is not None and complement.ab_cell_id is not None:
                    entity_type = complement.ab_cell_id.entity_type
                else:
                    value_map = {ii: str(ii) for ii in range(data.max() + 1)}
                    value_map[0] = "Unknown"
                    entity_type = {  # type: ignore
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
        mask: np.ndarray | None = None,
        cell_mask: np.ndarray | None = None,
        **kwargs,
    ):
        """
        Sub-class extension of :func:`~geoh5py.objects.cell_object.CellObject.copy`.
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
            mask=mask,
            cell_mask=cell_mask,
            omit_list=omit_list,
            **kwargs,
        )

        if self.cells is not None:
            if mask is not None:
                cell_mask = np.all(mask[self.cells], axis=1)
            else:
                cell_mask = np.ones(self.cells.shape[0], dtype=bool)

            if self.ab_cell_id is not None and self.ab_cell_id.values is not None:
                new_entity.ab_cell_id = self.ab_cell_id.values[cell_mask]

        complement: CurrentElectrode | PotentialElectrode = (
            self.current_electrodes
            if isinstance(self, PotentialElectrode)
            else self.potential_electrodes
        )

        # Set the mask of the complement
        if (
            new_entity.ab_cell_id is not None
            and complement is not None
            and complement.ab_cell_id is not None
            and complement.ab_cell_id.values is not None
            and complement.vertices is not None
            and complement.cells is not None
        ):
            intersect = np.intersect1d(
                new_entity.ab_cell_id.values,
                complement.ab_cell_id.values,
            )
            cell_mask = np.r_[
                [(val in intersect) for val in complement.ab_cell_id.values]
            ]

            # Convert cell indices to vertex indices
            mask = np.zeros(complement.vertices.shape[0], dtype=bool)
            mask[complement.cells[cell_mask, :]] = True

            new_complement = super(Curve, complement).copy(  # type: ignore
                parent=parent,
                omit_list=omit_list,
                copy_children=copy_children,
                clear_cache=clear_cache,
                mask=mask,
                cell_mask=cell_mask,
            )

            if isinstance(self, PotentialElectrode):
                new_entity.current_electrodes = new_complement
            else:
                new_entity.potential_electrodes = new_complement

            if new_complement.ab_cell_id is None and complement.ab_cell_id is not None:
                new_complement.ab_cell_id = complement.ab_cell_id.values[cell_mask]

            # Re-number the ab_cell_id
            value_map = {
                val: ind
                for ind, val in enumerate(
                    np.r_[0, np.unique(new_entity.current_electrodes.ab_cell_id.values)]
                )
            }
            new_map = {
                val: new_entity.current_electrodes.ab_cell_id.value_map.map[val]
                for val in value_map.values()
            }
            new_complement.ab_cell_id.values = np.asarray(
                [value_map[val] for val in new_complement.ab_cell_id.values]
            )
            new_entity.ab_cell_id.values = np.asarray(
                [value_map[val] for val in new_entity.ab_cell_id.values]
            )
            new_entity.ab_cell_id.value_map.map = new_map

        return new_entity

    @property
    @abstractmethod
    def current_electrodes(self):
        """
        The associated current_electrodes (transmitters)
        """

    @classmethod
    @abstractmethod
    def default_type_uid(cls) -> uuid.UUID:
        """Default unique identifier. Implemented on the child class."""

    @Curve.metadata.setter  # type: ignore
    def metadata(self, values: dict | None):
        if isinstance(values, dict):
            default_keys = ["Current Electrodes", "Potential Electrodes"]
            if sorted(list(values.keys())) != default_keys:
                raise ValueError(f"Input metadata must have for keys {default_keys}")

            for key in default_keys:
                if self.workspace.get_entity(values[key])[0] is None:
                    raise IndexError(f"Input {key} uuid not present in Workspace")

        super(Curve, Curve).metadata.fset(self, values)  # type: ignore

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
        if getattr(self, "_current_electrodes", None) is None:
            if self.metadata is not None and "Current Electrodes" in self.metadata:
                transmitter = self.metadata["Current Electrodes"]
                transmitter_entity = self.workspace.get_entity(transmitter)[0]

                if isinstance(transmitter_entity, CurrentElectrode):
                    self._current_electrodes = transmitter_entity

        return self._current_electrodes

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
        """"""

    @property
    def potential_electrodes(self) -> PotentialElectrode | None:
        """
        The associated potential_electrodes (receivers)
        """
        if getattr(self, "_potential_electrodes", None) is None:
            if self.metadata is not None and "Potential Electrodes" in self.metadata:
                potential = self.metadata["Potential Electrodes"]
                potential_entity = self.workspace.get_entity(potential)[0]

                if isinstance(potential_entity, PotentialElectrode):
                    self._potential_electrodes = potential_entity

        return self._potential_electrodes

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
