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

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import BaseModel, Field

from ..shared.utils import str2uuid, to_tuple
from .base import Group


if TYPE_CHECKING:  # pragma: no cover
    from ..objects import Curve, ObjectBase, Slicer


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

    normal_x: float = Field(alias="Normal X")
    normal_y: float = Field(alias="Normal Y")
    normal_z: float = Field(alias="Normal Z")
    position_x: float = Field(alias="Position X")
    position_y: float = Field(alias="Position Y")
    position_z: float = Field(alias="Position Z")

    @classmethod
    def create_section(
        cls,
        params: dict | InterpretationSectionParams,
    ) -> InterpretationSectionParams:
        """
        Verify/Create the interpretation section from a mapper.

        :param params: The parameters to verify.

        :return: An instance of the class.
        """
        if isinstance(params, Mapping):
            return cls(**params)
        if isinstance(params, InterpretationSectionParams):
            return params
        raise TypeError(
            "Interpretation section must be an InterpretationSectionParams"
            f"or a dict, {type(params)} provided."
        )


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

    def __init__(self, can_add_group: bool = False, **kwargs):
        self._can_add_group: bool = self._validate_can_add_group(can_add_group)

        super().__init__(**kwargs)

        self._interpretation_curves: tuple[Curve] | tuple = ()
        self._interpretation_sections: tuple[InterpretationSectionParams] | tuple = ()
        self._section_object: Slicer | None = None

    def _load_from_metadata(self, key: str) -> Any:
        """
        Load the metadata from the key.
        :param key: the key to load the metadata from.
        :return: The value store in the metadata if exists.
        """
        return self.metadata.get(key, None) if self.metadata else None

    def _update_to_metadata(self, key: str, values: tuple):
        """
        Update the metadata with the interpretation curves.

        :param key: The key to update.
        :param values: The values to update.
        """
        metadata: None | list = None

        if values:
            if key == "Interpretation curves":
                metadata = [value.uid for value in values]
            elif key == "Interpretation sections":
                metadata = [value.model_dump(by_alias=True) for value in values]
            else:
                raise ValueError(f"Invalid key '{key}' provided.")

        self.update_metadata({key: metadata})

    @staticmethod
    def _validate_can_add_group(value: bool) -> bool:
        """
        Validate the can add group attribute.

        :param value: A boolean value.

        :return: the verified and converted value.
        """
        if value not in [True, False, 0, 1]:
            raise TypeError("'Can add group' must be a boolean value.")

        return bool(value)

    def _verify_object(self, value, key: str) -> ObjectBase:
        """
        Get the object and verify that it is a children of the group.

        :param value: The value to verify.
        :param key: The default name of the expected object type.

        :return: The verified object
        """
        if isinstance(value, str | UUID):
            value = self.workspace.get_entity(str2uuid(value))[0]
        if getattr(value, "_default_name", None) != key:
            raise TypeError(
                f"The {value} object must be a {key} object."
                f" {type(value)} provided."
            )

        return value

    def add_interpretation_curve(self, curves: Curve | tuple[Curve]):
        """
        Add an interpretation curve.

        :param curves: The interpretation curve to add.
        """
        if curves is None:
            return

        for curve in to_tuple(curves):
            curve = self._verify_object(curve, "Curve")
            self.add_children(curve)
            self._interpretation_curves += (curve,)

            if self.section_object is not None:
                curve.clipping_ids = [self.section_object.uid]

        self._update_to_metadata("Interpretation curves", self._interpretation_curves)

    def add_interpretation_section(
        self, sections: InterpretationSectionParams | list[InterpretationSectionParams]
    ):
        """
        Add an interpretation section.

        :param sections: The interpretation section to add.
        """
        if sections is None:
            return

        for section in to_tuple(sections):
            self._interpretation_sections += (
                InterpretationSectionParams.create_section(section),
            )

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
        self._can_add_group = self._validate_can_add_group(value)
        self.workspace.update_attribute(self, "attributes")

    @property
    def interpretation_curves(self) -> tuple[Curve]:
        """
        Get the interpretation curves.

        :return: The interpretation curves.
        """
        if not self._interpretation_curves:
            self.add_interpretation_curve(
                self._load_from_metadata("Interpretation curves")
            )
        return self._interpretation_curves

    @property
    def interpretation_sections(self) -> tuple[InterpretationSectionParams] | None:
        """
        Get the interpretation sections.

        :return: The interpretation sections.
        """
        if not self._interpretation_sections:
            self.add_interpretation_section(
                self._load_from_metadata("Interpretation sections")
            )

        return self._interpretation_sections

    def remove_interpretation_curve(self, curves: Curve | tuple[Curve]):
        """
        Remove an interpretation curve.

        :param curves: The interpretation curve to remove.
        """
        if not self.interpretation_curves:
            return

        verified = [self._verify_object(curve, "Curve") for curve in to_tuple(curves)]

        interpretation_curves = []
        for curve in self.interpretation_curves:
            if curve in verified:
                self.workspace.remove_entity(curve)
            else:
                interpretation_curves.append(curve)

        self._interpretation_curves = tuple(interpretation_curves)
        self._update_to_metadata("Interpretation curves", self._interpretation_curves)

    def remove_interpretation_section(
        self, sections: InterpretationSectionParams | tuple[InterpretationSectionParams]
    ):
        """
        Remove an interpretation section.

        :param sections: The interpretation section to remove.
        """
        if not self.interpretation_sections:
            return

        verified = [
            InterpretationSectionParams.create_section(section)
            for section in to_tuple(sections)
        ]
        self._interpretation_sections = tuple(
            section
            for section in self.interpretation_sections
            if section not in verified
        )
        self._update_to_metadata(
            "Interpretation sections", self._interpretation_sections
        )

    @property
    def section_object(self) -> Slicer | None:
        """
        Get the section object ID.

        :return: The section object ID.
        """
        if self._section_object is None:
            self.section_object = self._load_from_metadata("Section object ID")

        return self._section_object

    @section_object.setter
    def section_object(self, slicer: Slicer | None):
        """
        Set the section object ID.

        :param slicer: The section object ID.
        """
        original_slicer = self._section_object
        if slicer is None:
            self._section_object = None
            self.update_metadata({"Section object ID": None})
        else:
            # a lot of type ignore because of circular imports
            self._section_object = self._verify_object(slicer, "Slicer")  # type: ignore
            self.add_children(self._section_object)  # type: ignore
            self.clipping_ids = [self._section_object.uid]  # type: ignore
            self.update_metadata({"Section object ID": self._section_object.uid})  # type: ignore

        if original_slicer is not None:
            self.workspace.remove_entity(original_slicer)
