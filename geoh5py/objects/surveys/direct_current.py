# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2020-2026 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoh5py.                                               '
#                                                                              '
#  geoh5py is free software: you can redistribute it and/or modify             '
#  it under the terms of the GNU Lesser General Public License as published by '
#  the Free Software Foundation, either version 3 of the License, or           '
#  (at your option) any later version.                                         '
#                                                                              '
#  geoh5py is distributed in the hope that it will be useful,                  '
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              '
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               '
#  GNU Lesser General Public License for more details.                         '
#                                                                              '
#  You should have received a copy of the GNU Lesser General Public License    '
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.           '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


# pylint: disable=too-many-ancestors

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from typing import cast

import numpy as np

from ...data import Data, ReferencedData, ReferenceValueMap
from ...shared.utils import str_json_to_dict
from ..curve import Curve
from .base import BaseSurvey


logger = logging.getLogger(__name__)


class BaseElectrode(BaseSurvey, Curve, ABC):
    __TYPE = None
    __OMIT_LIST = (
        "_ab_cell_id",
        "_metadata",
        "_potential_electrodes",
        "_current_electrodes",
    )
    __TYPE_MAP = {
        "Transmitters": "current_electrodes",
        "Receivers": "potential_electrodes",
    }

    def __init__(self, **kwargs):
        self._ab_cell_id: ReferencedData | None = None
        self._current_electrodes: CurrentElectrode | None = None
        self._potential_electrodes: PotentialElectrode | None = None

        if "ab_cell_id" in kwargs:
            raise ValueError(
                "The 'ab_cell_id' must be set after instantiation of the class."
            )
        super().__init__(**kwargs)

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
                complement = getattr(self, "complement", None)
                attributes = {
                    "values": data.astype(np.int32),
                    "association": "CELL",
                }
                if complement is not None and complement.ab_cell_id is not None:
                    attributes["entity_type"] = complement.ab_cell_id.entity_type
                else:
                    value_map = {ii: str(ii) for ii in range(data.max() + 1)}
                    value_map[0] = "Unknown"
                    attributes.update(
                        {  # type: ignore
                            "primitive_type": "REFERENCED",
                            "value_map": value_map,
                        }
                    )

                data = self.add_data({"A-B Cell ID": attributes})

                if isinstance(data, ReferencedData):
                    self._ab_cell_id = data

    @property
    def ab_map(self) -> ReferenceValueMap | None:
        """
        Get the ReferenceData.value_map of the ab_value_id
        """
        if isinstance(self.ab_cell_id, ReferencedData):
            return self.ab_cell_id.value_map
        return None

    @property
    @abstractmethod
    def complement(self) -> BaseElectrode | None:
        """
        The complement object for the current object.
        """

    @property
    def complement_reference(self):
        """Reference data linking the geometry of complement entity."""
        return self.ab_cell_id

    @complement_reference.setter
    def complement_reference(self, value: ReferencedData):
        self.ab_cell_id = value

    @property
    @abstractmethod
    def current_electrodes(self):
        """
        The associated current_electrodes (transmitters)
        """

    @property
    def omit_list(self) -> tuple:
        """
        List of attributes to omit when copying.
        """
        return self.__OMIT_LIST

    @property
    @abstractmethod
    def potential_electrodes(self):
        """
        The associated potential_electrodes (receivers)
        """

    @property
    def type_map(self) -> dict[str, str]:
        """
        Mapping of the electrode types to the associated electrode.
        """
        return self.__TYPE_MAP

    def validate_metadata(self, value: dict | np.ndarray | bytes) -> dict:
        if isinstance(value, np.ndarray):
            value = value[0]

        if isinstance(value, bytes):
            value = str_json_to_dict(value)

        if isinstance(value, dict):
            default_keys = ["Current Electrodes", "Potential Electrodes"]

            # check if metadata has the required keys
            if not all(key in value for key in default_keys):
                raise ValueError(f"Input metadata must have for keys {default_keys}")

        return value

    def _get_complement_mask(
        self, mask: np.ndarray, complement: BaseElectrode
    ) -> np.ndarray:
        """
        Get the complement mask based on the input mask.

        :param mask: Mask on vertices or cells.
        :param complement: Complement entity.
        """
        if self.ab_cell_id is None or complement.ab_cell_id is None:
            raise ValueError(
                "Both the object and its complement have 'ab_cell_id' set."
            )

        intersect = np.intersect1d(
            self.ab_cell_id.values,
            complement.ab_cell_id.values,
        )
        cell_mask = np.r_[[(val in intersect) for val in complement.ab_cell_id.values]]

        return cell_mask

    def _renumber_reference_ids(self):
        """
        Renumber the ab_cell_ids based on unique values of currents.
        """
        if (
            self.complement is None
            or self.complement.ab_cell_id is None
            or self.ab_cell_id is None
        ):
            raise AttributeError(
                "The 'ab_cell_id' must be set on both the object and its complement."
            )

        # Re-number the ab_cell_id
        value_map = {
            val: ind
            for ind, val in enumerate(
                np.r_[0, np.unique(self.current_electrodes.ab_cell_id.values)]
            )
        }
        new_map = {
            val: dict(self.current_electrodes.ab_cell_id.value_map.map)[val]
            for val in value_map.values()
        }
        self.complement.ab_cell_id.values = np.asarray(
            [value_map[val] for val in self.complement.ab_cell_id.values]
        )
        self.ab_cell_id.values = np.asarray(
            [value_map[val] for val in self.ab_cell_id.values]
        )
        self.ab_cell_id.entity_type.value_map = new_map  # type: ignore


class PotentialElectrode(BaseElectrode):
    """
    Ground potential electrode (receiver).
    """

    _TYPE_UID = uuid.UUID("{275ecee9-9c24-4378-bf94-65f3c5fbe163}")
    _default_name = "PotentialElectrode"
    __TYPE = "Receivers"

    @property
    def complement(self) -> CurrentElectrode | None:
        return self.current_electrodes

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

    @property
    def type(self):
        """Survey element type"""
        return self.__TYPE


class CurrentElectrode(BaseElectrode):
    """
    Ground direct current electrode (transmitter).
    """

    _TYPE_UID = uuid.UUID("{9b08bb5a-300c-48fe-9007-d206f971ea92}")
    _default_name = "CurrentElectrode"
    __TYPE = "Transmitters"

    def __init__(self, **kwargs):
        self._current_line_id: uuid.UUID | None = None

        super().__init__(**kwargs)

    @property
    def complement(self) -> PotentialElectrode | None:
        return self.potential_electrodes

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
                    "primitive_type": "REFERENCED",
                    "value_map": value_map,
                }
            }
        )

        if not isinstance(ab_cell_id, ReferencedData):
            raise UserWarning("Could not create 'A-B Cell ID' data.")

        ab_cell_id.entity_type.name = "A-B"
        self._ab_cell_id = ab_cell_id

    @property
    def type(self):
        """Survey element type"""
        return self.__TYPE
