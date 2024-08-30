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

# pylint: disable=too-many-arguments, too-many-instance-attributes

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Literal, get_args
from uuid import UUID

import numpy as np

from ..shared import EntityType
from .color_map import ColorMap
from .primitive_type_enum import PrimitiveTypeEnum
from .reference_value_map import ReferenceValueMap

if TYPE_CHECKING:
    from ..workspace import Workspace
    from .data import Data  # noqa: F401


ColorMapping = Literal[
    "linear",
    "equal_area",
    "logarithmic",
    "cdf",
    "missing",
    "cumulative_distribution_function",
]


class DataType(EntityType):
    """
    DataType class.

    Controls all the attributes of the data type for displays in Geoscience ANALYST.

    :param workspace: An active Workspace.
    :param primitive_type: The primitive type of the data.
    :param color_map: The colormap used for plotting.
    :param duplicate_type_on_copy: Force a copy on copy of the data entity.
    :param hidden: If the data are hidden or not.
    :param mapping: The type of color stretching to plot the colormap.
    :param number_of_bins: The number of bins used by the histogram.
    :param transparent_no_data: If the no data values are displayed as transparent or not.
    :param units: The type of the units of the data.
    :param kwargs: Additional keyword arguments to set as attributes
        (see :obj:`...shared.entity_type.EntityType`).
    """

    _attribute_map = EntityType._attribute_map.copy()
    _attribute_map.update(
        {
            "Hidden": "hidden",
            "Mapping": "mapping",
            "Number of bins": "number_of_bins",
            "Primitive type": "primitive_type",
            "Transparent no data": "transparent_no_data",
        }
    )

    def __init__(
        self,
        workspace: Workspace,
        *,
        primitive_type: type[Data] | PrimitiveTypeEnum | str | None = None,
        color_map: ColorMap | None = None,
        duplicate_type_on_copy: bool = False,
        hidden: bool = False,
        mapping: ColorMapping = "equal_area",
        number_of_bins: int | None = None,
        transparent_no_data: bool = True,
        units: str | None = None,
        **kwargs,
    ):
        super().__init__(workspace, **kwargs)
        self.color_map = color_map
        self.duplicate_type_on_copy = duplicate_type_on_copy
        self.hidden = hidden
        self.mapping = mapping
        self.number_of_bins = number_of_bins
        self.primitive_type = primitive_type  # type: ignore
        self.transparent_no_data = transparent_no_data
        self.units = units

    @property
    def color_map(self) -> ColorMap | None:
        r"""
        The Colormap used for plotting

        The colormap can be set from a dictionary of sorted values with
        corresponding RGBA color.
        Or from a numpy array containing the RGBA values.

        .. code-block:: python

            color_map = {
                val_1: [r_1, g_1, b_1, a_1],
                ...,
                val_i: [r_i, g_i, b_i, a_i]
            }

        It can be set to None if non-existing.
        """
        return self._color_map

    @color_map.setter
    def color_map(self, color_map: ColorMap | dict | np.ndarray | None):
        if isinstance(color_map, dict):
            color_map = ColorMap(**color_map)

        elif isinstance(color_map, np.ndarray):
            color_map = ColorMap(values=color_map)

        if not isinstance(color_map, (ColorMap, type(None))):
            raise TypeError(
                f"Input value for 'color_map' must be of type {ColorMap},"
                f"numpy.ndarray or dict with 'values'."
            )

        if isinstance(color_map, ColorMap):
            color_map.parent = self

        self._color_map: ColorMap | None = color_map

        self.workspace.update_attribute(self, "color_map")

    @property
    def duplicate_type_on_copy(self) -> bool:
        """
        If the data type should be duplicated on copy.
        """
        return self._duplicate_type_on_copy

    @duplicate_type_on_copy.setter
    def duplicate_type_on_copy(self, value: bool):
        if not isinstance(value, bool) and value != 1 and value != 0:
            raise TypeError(f"transparent_no_data must be a bool, not {type(value)}")

        self._duplicate_type_on_copy = bool(value)
        self.workspace.update_attribute(self, "attributes")

    @classmethod
    def find_or_create_type(
        cls,
        workspace: Workspace,
        uid: UUID | None = None,
        dynamic_implementation_id: UUID | None = None,
        value_map: dict | tuple | None = None,
        **kwargs,
    ) -> DataType:
        """
        Get the data type for geometric data.

        :param workspace: An active Workspace class
        :param uid: The unique identifier of the entity type.
        :param dynamic_implementation_id: Optional dynamic implementation id.
        :param kwargs: The attributes of the entity type.

        :return: EntityType
        """
        if uid is not None:
            entity_type = DataType.find(workspace, uid)

            if entity_type is not None:
                return entity_type

        data_type = cls
        if dynamic_implementation_id is not None:
            data_type = GeometricDynamicData.find_type(uid, dynamic_implementation_id)
        elif value_map is not None:
            return ReferenceDataType(workspace, value_map, uid=uid, **kwargs)

        return data_type(workspace, uid=uid, **kwargs)

    @property
    def hidden(self) -> bool:
        """
        If the data are hidden or not.
        """
        return self._hidden

    @hidden.setter
    def hidden(self, value: bool):
        if not isinstance(value, bool) and value != 1 and value != 0:
            raise TypeError(f"hidden must be a bool, not {type(value)}")

        self._hidden: bool = bool(value)

        self.workspace.update_attribute(self, "attributes")

    @property
    def mapping(self) -> str:
        """
        The type of color stretching to plot the colormap.
        It chan be one of the following:
        'linear', 'equal_area', 'logarithmic', 'cdf', 'missing'
        """
        return self._mapping

    @mapping.setter
    def mapping(self, value: ColorMapping):
        if value not in get_args(ColorMapping):
            raise ValueError(
                f"Mapping {value} was provided but should be one of {get_args(ColorMapping)}"
            )
        self._mapping: str = value

        self.workspace.update_attribute(self, "attributes")

    @property
    def number_of_bins(self) -> int | None:
        """
        The number of bins used by the histogram.
        It can be None if no histogram is used.
        """
        return self._number_of_bins

    @number_of_bins.setter
    def number_of_bins(self, n_bins: int | None):
        if n_bins is None:
            pass
        elif not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
            raise ValueError(
                f"Number of bins should be an integer greater than 0 or None, not {n_bins}"
            )

        self._number_of_bins: int | None = n_bins

        self.workspace.update_attribute(self, "attributes")

    @property
    def primitive_type(self) -> PrimitiveTypeEnum | None:
        """
        The primitive type of the data.
        """
        return self._primitive_type

    @primitive_type.setter
    def primitive_type(self, value: str | type[Data] | PrimitiveTypeEnum | None):
        if isinstance(value, str):
            value = getattr(PrimitiveTypeEnum, value.replace("-", "_").upper())
        elif hasattr(value, "primitive_type"):
            value = getattr(value, "primitive_type")()
        if not isinstance(value, (PrimitiveTypeEnum, type(None))):
            raise ValueError(
                f"Primitive type value must be of type {PrimitiveTypeEnum}, find {type(value)}"
            )

        self._primitive_type = value

    @property
    def transparent_no_data(self) -> bool:
        """
        If the no data values are displayed as transparent or not.
        """
        return self._transparent_no_data

    @transparent_no_data.setter
    def transparent_no_data(self, value: bool):
        if not isinstance(value, bool) and value != 1 and value != 0:
            raise TypeError(f"transparent_no_data must be a bool, not {type(value)}")
        self._transparent_no_data = bool(value)

        self.workspace.update_attribute(self, "attributes")

    @property
    def units(self) -> str | None:
        """
        The type of the units of the data.
        """
        return self._units

    @units.setter
    def units(self, unit: str | None):
        if not isinstance(unit, (str, type(None))):
            raise TypeError(f"units must be a string, not {type(unit)}")
        self._units = unit

        self.workspace.update_attribute(self, "attributes")

    @classmethod
    def create(
        cls, workspace: Workspace, primitive_type: str, attribute_dict: dict
    ) -> DataType:
        """
        Get a dictionary of attributes and validate the type of data.

        :param workspace: An active Workspace.
        :param primitive_type: The primitive type of the data.
        :param attribute_dict: A dictionary of attributes of the new Datatype to create.

        :return: A new instance of DataType.
        """
        if not primitive_type.upper() in PrimitiveTypeEnum.__members__:
            raise ValueError(
                f"Data 'type' should be one of {PrimitiveTypeEnum.__members__}"
            )

        attribute_dict["primitive_type"] = primitive_type.upper()

        if attribute_dict["primitive_type"] in ["REFERENCED", "BOOLEAN"]:
            value_map = attribute_dict.pop("value_map", None)
            if value_map is None:
                if attribute_dict["primitive_type"] == "REFERENCED":
                    value_map = {
                        i: str(val)
                        for i, val in enumerate(set(attribute_dict["values"]))
                    }
                else:
                    value_map = {0: "False", 1: "True"}

            attribute_dict["value_map"] = value_map

        data_type = cls.find_or_create_type(workspace, **attribute_dict)

        return data_type

    @staticmethod
    def validate_primitive_type(values: np.ndarray | None) -> str:
        """
        Validate the primitive type of the data.
        """
        if values is None or (
            isinstance(values, np.ndarray) and np.issubdtype(values.dtype, np.floating)
        ):
            primitive_type = "FLOAT"

        elif isinstance(values, np.ndarray) and (
            np.issubdtype(values.dtype, np.integer)
        ):
            primitive_type = "INTEGER"
        elif isinstance(values, str) or (
            isinstance(values, np.ndarray) and values.dtype.kind in ["U", "S"]
        ):
            primitive_type = "TEXT"
        elif isinstance(values, np.ndarray) and (values.dtype == bool):
            primitive_type = "BOOLEAN"
        else:
            raise NotImplementedError(
                "Only add_data values of type FLOAT, INTEGER,"
                "BOOLEAN and TEXT have been implemented"
            )
        return primitive_type


class ReferenceDataType(DataType):
    """
    DataType class.

    Controls all the attributes of reference data.

    :param value_map: Reference value map for to map index with description.
    """

    def __init__(
        self,
        workspace: Workspace,
        value_map: dict[int, str] | tuple | ReferenceValueMap,
        **kwargs,
    ):
        super().__init__(workspace, **kwargs)
        self._value_map = self.validate_value_map(value_map)

    @staticmethod
    def validate_value_map(
        value_map: dict | tuple | ReferenceValueMap,
    ) -> ReferenceValueMap:
        """
        Validate the attribute of ReferencedDataType
        """
        if isinstance(value_map, dict):
            value_map = ReferenceValueMap(value_map)
        elif isinstance(value_map, tuple):
            value_map = ReferenceValueMap(*value_map)

        if not isinstance(value_map, ReferenceValueMap):
            raise TypeError(
                "Attribute 'value_map' must be provided as a dict, tuple[dict] "
                f"or {ReferenceValueMap}."
            )
        return value_map

    @property
    def value_map(self) -> ReferenceValueMap:
        r"""
        Reference value map for to map index with description.

        The value_map can be set from a dictionary of sorted values int
        values with text description.

        .. code-block:: python

            value_map = {
                val_1: str_1,
                ...,
                val_i: str_i
            }

        """
        return self._value_map

    @value_map.setter
    def value_map(self, value_map: dict | tuple | ReferenceValueMap):

        self._value_map = self.validate_value_map(value_map)

        if self.on_file:
            self.workspace.update_attribute(self, "value_map")


class GeometricDynamicData(DataType, ABC):
    """
    Data container for dynamic geometric data.
    """

    _attribute_map = DataType._attribute_map.copy()
    _attribute_map.update(
        {
            "Dynamic implementation ID": "dynamic_implementation_id",
        }
    )

    _TYPE_UID: UUID | None
    _DYNAMIC_IMPLEMENTATION_ID: UUID

    def __init__(
        self,
        workspace: Workspace,
        uid: UUID | None = None,
        **kwargs,
    ):
        if uid is None:
            uid = self._TYPE_UID

        super().__init__(workspace, uid=uid, **kwargs)

    @classmethod
    def default_type_uid(cls) -> UUID | None:
        """
        Default uuid for the entity type.
        """
        return cls._TYPE_UID

    @property
    def dynamic_implementation_id(self) -> UUID:
        """
        The dynamic implementation id.
        """
        return self._DYNAMIC_IMPLEMENTATION_ID

    @classmethod
    def find_type(
        cls, uid: UUID | None, dynamic_implementation_id: UUID
    ) -> type[DataType]:
        """
        Find the data type in the workspace.

        :param uid: The UUID of the data type.
        :param dynamic_implementation_id: The dynamic implementation id.
        """
        data_type = DYNAMIC_CLASS_IDS.get(dynamic_implementation_id, DataType)

        # Unknown geometric data type
        if (
            hasattr(data_type, "default_type_uid")
            and data_type.default_type_uid() is not None
            and uid is not None
            and data_type.default_type_uid() != uid
        ):
            return DataType

        return data_type


class GeometricDataValueMap(GeometricDynamicData):
    """
    Data container for value map
    """

    _DYNAMIC_IMPLEMENTATION_ID = UUID("{4b6ecb37-0623-4ea0-95f1-4873008890a8}")
    _TYPE_UID = None


class GeometricDataX(GeometricDynamicData):
    """
    Data container for X values
    """

    _DYNAMIC_IMPLEMENTATION_ID = UUID("{2dbf303e-05d6-44ba-9692-39474e88d516}")
    _TYPE_UID = UUID(fields=(0xE9E6B408, 0x4109, 0x4E42, 0xB6, 0xA8, 0x685C37A802EE))


class GeometricDataY(GeometricDynamicData):
    """
    Data container for Y values
    """

    _DYNAMIC_IMPLEMENTATION_ID = UUID("{d56406dc-5eeb-418d-add4-a1282a6ef668}")
    _TYPE_UID = UUID(fields=(0xF55B07BD, 0xD8A0, 0x4DFF, 0xBA, 0xE5, 0xC975D490D71C))


class GeometricDataZ(GeometricDynamicData):
    """
    Data container for X values
    """

    _DYNAMIC_IMPLEMENTATION_ID = UUID("{9dacdc3b-6878-408d-93ae-e9a95e640f0c}")
    _TYPE_UID = UUID(fields=(0xDBAFB885, 0x1531, 0x410C, 0xB1, 0x8E, 0x6AC9A40B4466))


DYNAMIC_CLASS_IDS = {
    cls._DYNAMIC_IMPLEMENTATION_ID: cls  # pylint: disable=protected-access
    for cls in GeometricDynamicData.__subclasses__()
}
