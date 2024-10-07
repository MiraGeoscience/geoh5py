# #  Copyright (c) 2024 Mira Geoscience Ltd.
# #
# #  This file is part of geoh5py.
# #
# #  geoh5py is free software: you can redistribute it and/or modify
# #  it under the terms of the GNU Lesser General Public License as published by
# #  the Free Software Foundation, either version 3 of the License, or
# #  (at your option) any later version.
# #
# #  geoh5py is distributed in the hope that it will be useful,
# #  but WITHOUT ANY WARRANTY; without even the implied warranty of
# #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# #  GNU Lesser General Public License for more details.
# #
# #  You should have received a copy of the GNU Lesser General Public License
# #  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.
#
from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import BaseModel, model_validator

from ..shared.utils import str2uuid
from .base import Group


if TYPE_CHECKING:  # pragma: no cover
    from ..objects import Curve, Slicer
    from ..shared.entity import Entity

ATTRIBUTE_MAP = {
    "Normal X": "normal_x",
    "Normal Y": "normal_y",
    "Normal Z": "normal_z",
    "Position X": "position_x",
    "Position Y": "position_y",
    "Position Z": "position_z",
}


class InterpretationSectionParams(BaseModel):
    """
    The class to store the interpretation section.

    :param normal_x: The normal x component.
    :param normal_y: The normal y component.
    :param normal_z: The normal z component.
    :param position_x: The position x component.
    :param position_y: The position y component.
    :param position_z: The position z component.
    """

    normal_x: float
    normal_y: float
    normal_z: float
    position_x: float
    position_y: float
    position_z: float

    def in_list(self, params_list: list[InterpretationSectionParams]) -> int | None:
        """
        Check if the parameters are in the list.

        :param params_list: The list of parameters.

        :return: The index of the parameters in the list else None.
        """
        for idx, param in enumerate(params_list):
            if param.map_back == self.map_back:
                return idx
        return None

    @model_validator(mode="before")
    @classmethod
    def map_attributes(cls, data: dict) -> dict:
        """
        Map the attributes to the right keys.

        :param data: The data to map.

        :return: The mapped data.
        """
        return {ATTRIBUTE_MAP.get(key, key): value for key, value in data.items()}

    @property
    def map_back(self) -> dict:
        """
        Map the attributes back to the original keys of attribute map.

        :return: The mapped data.
        """
        return {key: getattr(self, value) for key, value in ATTRIBUTE_MAP.items()}

    @staticmethod
    def verify(
        params: dict | InterpretationSectionParams,
    ) -> InterpretationSectionParams:
        """
        Verify if the parameters are correct.

        :param params: The parameters to verify.

        :return: An instance of the class.
        """
        if isinstance(params, dict):
            return InterpretationSectionParams(**params)
        if not isinstance(params, InterpretationSectionParams):
            raise TypeError(
                "Interpretation section must be an InterpretationSectionParams"
                f"or a dict, {type(params)} provided."
            )
        return params


class InterpretationSection(Group):
    """The type for the basic Container group."""

    _attribute_map = Group._attribute_map.copy()
    _attribute_map.update(
        {
            "Can add group": "can_add_group",
        }
    )

    _TYPE_UID = UUID(fields=(0x27EE59E1, 0xB1CE, 0x41EC, 0x8A, 0x86, 0x8BD3D229E198))
    _default_name = "Interpretation Section"

    def __init__(self, **kwargs):
        self._can_add_group: bool = False
        self._interpretation_curves: list[Curve] | None = None
        self._interpretation_sections: list[InterpretationSectionParams] | None = None
        self._section_object_id: Slicer | None = None

        super().__init__(**kwargs)

    def _load_attributes(self):
        """
        Load the data from the metadata only to please mypy.
        """
        _ = self.interpretation_curves
        _ = self.interpretation_sections
        _ = self.section_object_id

    def _load_from_metadata(self, key: str) -> Any:
        """
        Load the metadata from the key.

        :param key: the key to load the metadata from.

        :return: The value store in the metadata if exists.
        """
        metadata = self.metadata if self.metadata is not None else {}
        return metadata.get(key, None)

    def _update_to_metadata(self, key: str, values: list | None):
        """
        Update the metadata with the interpretation curves.

        :param key: The key to update.
        :param values: The values to update.
        """
        if values:
            self.update_metadata(
                {
                    key: [
                        value.map_back
                        if isinstance(value, InterpretationSectionParams)
                        else str(value.uid)
                        for value in values
                    ]
                }
            )
        else:
            setattr(self, key, None)
            self.update_metadata({key: None})

    def _verify_object(self, value, key: str):
        """
        Get the object and verify that it is a children of the group.

        :param value: The value to verify.
        :param key: The default name of the expected object type.

        :return: The verified object
        """
        if isinstance(value, str | UUID):
            value = str2uuid(value)
            value = self.workspace.get_entity(value)[0]

        if getattr(value, "_default_name", None) != key:
            raise TypeError(
                f"The {value} object must be a {key} object."
                f" {type(value)} provided."
            )

        return value

    def add_interpretation_curve(self, curves: Curve | list[Curve]):
        """
        Add an interpretation curve.

        :param curves: The interpretation curve to add.
        """
        if not isinstance(curves, list):
            curves = [curves]

        self._load_attributes()
        for curve in curves:
            curve = self._verify_object(curve, "Curve")
            self.add_children(curve)
            if self._interpretation_curves is None:
                self._interpretation_curves = [curve]
            elif curve not in self._interpretation_curves:
                self._interpretation_curves.append(curve)

        self._update_to_metadata("Interpretation curves", self._interpretation_curves)

    def add_interpretation_section(
        self, sections: InterpretationSectionParams | list[InterpretationSectionParams]
    ):
        """
        Add an interpretation section.

        :param sections: The interpretation section to add.
        """
        if not isinstance(sections, list):
            sections = [sections]

        self._load_attributes()
        for section in sections:
            section = InterpretationSectionParams.verify(section)
            if self._interpretation_sections is None:
                self._interpretation_sections = [section]
            elif section.in_list(self._interpretation_sections) is None:
                self._interpretation_sections.append(section)

        self._update_to_metadata(
            "Interpretation sections", self._interpretation_sections
        )

    @property
    def can_add_group(self) -> bool:
        """
        Get the can add group attribute.

        :return: The 'can add group' attribute.
        """
        return self._can_add_group

    @can_add_group.setter
    def can_add_group(self, value: bool):
        """
        Set the can add group attribute.

        :param value: The can add group attribute.
        """
        self._can_add_group = bool(value)

        if self.on_file:
            self.workspace.update_attribute(self, "can_add_group")

    @property
    def interpretation_curves(self) -> list[Curve] | None:
        """
        Get the interpretation curves.

        :return: The interpretation curves.
        """

        if self._interpretation_curves is None:
            self.interpretation_curves = self._load_from_metadata(
                "Interpretation curves"
            )

        return self._interpretation_curves

    @interpretation_curves.setter
    def interpretation_curves(self, curves: list[Curve] | None):
        """
        Set the interpretation curves.

        :param curves: The interpretation curves.
        """
        if curves is None:
            self._interpretation_curves = None
        else:
            if not isinstance(curves, list):
                raise TypeError("Interpretation curves must be a list of Curve.")

            self._interpretation_curves = [
                self._verify_object(curve, "Curve") for curve in curves
            ]

            self.add_children(self._interpretation_curves)  # type: ignore

        self._update_to_metadata("Interpretation curves", self._interpretation_curves)

    @property
    def interpretation_sections(self) -> list[InterpretationSectionParams] | None:
        """
        Get the interpretation sections.

        :return: The interpretation sections.
        """
        if self._interpretation_sections is None:
            self.interpretation_sections = self._load_from_metadata(
                "Interpretation sections"
            )

        return self._interpretation_sections

    @interpretation_sections.setter
    def interpretation_sections(
        self, sections: list[InterpretationSectionParams] | None
    ):
        """
        Set the interpretation sections.

        :param sections: The interpretation sections.
        """
        if sections is None:
            self._interpretation_sections = None
        else:
            if not isinstance(sections, list):
                raise TypeError(
                    "Interpretation sections must be a list of InterpretationSectionParams."
                )

            self._interpretation_sections = [
                InterpretationSectionParams.verify(section) for section in sections
            ]

        self._update_to_metadata(
            "Interpretation sections", self._interpretation_sections
        )

    def remove_interpretation_curve(self, curves: Curve | list[Curve]):
        """
        Remove an interpretation curve.

        :param curves: The interpretation curve to remove.
        """
        self._load_attributes()
        if self._interpretation_curves is None:
            return

        if not isinstance(curves, list):
            curves = [curves]

        for curve in curves:
            curve = self._verify_object(curve, "Curve")
            if curve in self._interpretation_curves:
                self._interpretation_curves.remove(curve)

        self._update_to_metadata("Interpretation curves", self._interpretation_curves)

    def remove_interpretation_section(
        self, sections: InterpretationSectionParams | list[InterpretationSectionParams]
    ):
        """
        Remove an interpretation section.

        :param sections: The interpretation section to remove.
        """
        self._load_attributes()
        if self._interpretation_sections is None:
            return

        if not isinstance(sections, list):
            sections = [sections]

        for section in sections:
            section = InterpretationSectionParams.verify(section)
            idx = section.in_list(self._interpretation_sections)
            if idx is not None:
                del self._interpretation_sections[idx]

        self._update_to_metadata(
            "Interpretation sections", self._interpretation_sections
        )

    @property
    def section_object_id(self) -> Slicer | None:
        """
        Get the section object ID.

        :return: The section object ID.
        """
        if self._section_object_id is None:
            self.section_object_id = self._load_from_metadata("Section object ID")

        return self._section_object_id

    @section_object_id.setter
    def section_object_id(self, slicer: Slicer | None):
        """
        Set the section object ID.

        :param slicer: The section object ID.
        """
        if slicer is None:
            self._section_object_id = None
            self.update_metadata({"Section object ID": None})
            return

        self._section_object_id = self._verify_object(slicer, "Slicer")

        self.add_children(slicer)

        self.update_metadata({"Section object ID": slicer.uid})
