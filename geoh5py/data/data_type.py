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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, get_args
from uuid import UUID

import numpy as np

from ..shared import EntityType
from .color_map import ColorMap
from .primitive_type_enum import PrimitiveTypeEnum
from .reference_value_map import BOOLEAN_VALUE_MAP, ReferenceValueMap

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
        primitive_type: PrimitiveTypeEnum | str,
        uid: UUID | None = None,
        **kwargs,
    ) -> DataType:
        """
        Get the data type for geometric data.

        :param workspace: An active Workspace class
        :param primitive_type: The primitive type of the data.
        :param uid: The unique identifier of the entity type.
        :param kwargs: The attributes of the entity type.

        :return: EntityType
        """
        if uid is not None:
            entity_type = DataType.find(workspace, uid)

            if entity_type is not None:
                return entity_type

        primitive_type = cls.validate_primitive_type(primitive_type)

        if primitive_type in [
            PrimitiveTypeEnum.REFERENCED,
            PrimitiveTypeEnum.BOOLEAN,
            PrimitiveTypeEnum.GEOMETRIC,
        ]:
            return ReferenceDataType.create(
                workspace, primitive_type, uid=uid, **kwargs
            )

        return cls(workspace, primitive_type=primitive_type, uid=uid, **kwargs)

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
        if hasattr(value, "primitive_type"):
            value = getattr(value, "primitive_type")()

        elif value is not None:
            value = self.validate_primitive_type(value)

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

    @staticmethod
    def primitive_type_from_values(values: np.ndarray | None) -> PrimitiveTypeEnum:
        """
        Validate the primitive type of the data.
        """
        if values is None or (
            isinstance(values, np.ndarray) and np.issubdtype(values.dtype, np.floating)
        ):
            primitive_type = PrimitiveTypeEnum.FLOAT

        elif isinstance(values, np.ndarray) and (
            np.issubdtype(values.dtype, np.integer)
        ):
            primitive_type = PrimitiveTypeEnum.INTEGER
        elif isinstance(values, str) or (
            isinstance(values, np.ndarray) and values.dtype.kind in ["U", "S"]
        ):
            primitive_type = PrimitiveTypeEnum.TEXT
        elif isinstance(values, np.ndarray) and (values.dtype == bool):
            primitive_type = PrimitiveTypeEnum.BOOLEAN
        else:
            raise NotImplementedError(
                "Only add_data values of type FLOAT, INTEGER,"
                "BOOLEAN and TEXT have been implemented"
            )
        return primitive_type

    @staticmethod
    def validate_primitive_type(
        primitive_type: PrimitiveTypeEnum | str,
    ) -> PrimitiveTypeEnum:
        """
        Validate the primitive type of the data.
        """
        if isinstance(primitive_type, str):
            primitive_type = getattr(PrimitiveTypeEnum, primitive_type.upper())
        if not isinstance(primitive_type, PrimitiveTypeEnum):
            raise ValueError(
                f"Data 'type' should be one of {PrimitiveTypeEnum.__members__}"
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

    @classmethod
    def create(
        cls,
        workspace: Workspace,
        primitive_type: str | PrimitiveTypeEnum,
        dynamic_implementation_id: UUID | None = None,
        values: np.ndarray | None = None,
        value_map: (
            np.ndarray | dict[int, str] | tuple | ReferenceValueMap | None
        ) = None,
        **kwargs,
    ):
        """
        Create a new instance of ReferenceDataType
        """
        primitive_type = cls.validate_primitive_type(primitive_type)

        if (
            primitive_type == PrimitiveTypeEnum.GEOMETRIC
            and dynamic_implementation_id is not None
        ):
            data_type = DYNAMIC_CLASS_IDS.get(dynamic_implementation_id, DataType)

            return data_type(workspace, primitive_type=primitive_type, **kwargs)

        if value_map is None:
            if primitive_type == PrimitiveTypeEnum.REFERENCED:
                if values is None:
                    raise ValueError("Either 'values' or 'value_map' must be provided.")

                value_map = {i: str(val) for i, val in enumerate(set(values))}
            else:
                value_map = {0: "False", 1: "True"}

        if primitive_type is None:
            primitive_type = PrimitiveTypeEnum.REFERENCED

        if primitive_type == PrimitiveTypeEnum.BOOLEAN:
            return ReferencedBooleanType(
                workspace, value_map, primitive_type=primitive_type, **kwargs
            )

        return ReferencedValueMapType(
            workspace, value_map, primitive_type=primitive_type, **kwargs
        )

    @staticmethod
    @abstractmethod
    def validate_keys(value_map: ReferenceValueMap) -> ReferenceValueMap:
        """
        Validate the keys of the value map.
        """

    @classmethod
    def validate_value_map(
        cls,
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

        value_map = cls.validate_keys(value_map)

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


class ReferencedValueMapType(ReferenceDataType):
    """
    Data container for referenced value map.
    """

    _TYPE_UID = UUID(fields=(0x2D5D6C1E, 0x4D8C, 0x4F3A, 0x9B, 0x3F, 0x2E5A0D8E1C1F))

    def __init__(
        self,
        workspace: Workspace,
        value_map: dict[int, str] | tuple | ReferenceValueMap,
        data_maps: dict[str, ReferenceValueMap] | None = None,
        **kwargs,
    ):
        super().__init__(workspace, value_map, **kwargs)
        self.data_maps = data_maps

    @property
    def data_maps(self) -> dict[str, np.ndarray] | None:
        """
        A reference dictionary mapping properties to numpy arrays.
        """
        return self._data_maps

    @data_maps.setter
    def data_maps(self, value: dict[str, np.ndarray] | None):
        if value is not None:
            if not isinstance(value, dict):
                raise TypeError("Property maps must be a dictionary")
            for key, val in value.items():
                if not isinstance(val, GeometricDataValueMapType):
                    raise TypeError(
                        f"Property maps values for '{key}' must be a 'GeometricDataValueMapType'."
                    )

        self._data_maps = value

    @staticmethod
    def validate_keys(value_map: ReferenceValueMap) -> ReferenceValueMap:
        """
        Validate the keys of the value map.
        """
        if 0 not in value_map.map["Key"]:
            value_map.map.resize(len(value_map) + 1, refcheck=False)
            value_map.map[-1] = (0, "Unknown")

        if dict(value_map.map)[0] != "Unknown":
            raise ValueError("Value for key 0 must be 'Unknown'")

        return value_map


class ReferencedBooleanType(ReferenceDataType):
    """
    Data container for referenced boolean data.
    """

    @staticmethod
    def validate_keys(value_map: ReferenceValueMap) -> ReferenceValueMap:
        """
        Validate the keys of the value map.
        """
        if not np.all(value_map.map == BOOLEAN_VALUE_MAP):
            raise ValueError("Boolean value map must be (0: 'False', 1: 'True'")

        return value_map


class GeometricDynamicDataType(DataType, ABC):
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


class GeometricDataValueMapType(ReferenceDataType, GeometricDynamicDataType):
    """
    Data container for value map
    """

    _DYNAMIC_IMPLEMENTATION_ID = UUID("{4b6ecb37-0623-4ea0-95f1-4873008890a8}")
    _TYPE_UID = None

    def __init__(
        self,
        workspace: Workspace,
        parent: ReferenceDataType,
        value_map: ReferenceValueMap,
        uid: UUID | None = None,
        description: str = "Dynamic referenced data",
        primitive_type: PrimitiveTypeEnum | str = PrimitiveTypeEnum.GEOMETRIC,
        **kwargs,
    ):
        super().__init__(
            workspace,
            value_map,
            description=description,
            uid=uid,
            primitive_type=primitive_type,
            **kwargs,
        )

        if not isinstance(parent, ReferenceDataType):
            raise TypeError("Parent must be of type ReferenceDataType")

        self._parent = parent

    @property
    def parent(self) -> ReferenceDataType:
        """
        The parent data type.
        """
        return self._parent

    @staticmethod
    def validate_keys(value_map: ReferenceValueMap) -> ReferenceValueMap:
        """
        Validate the keys of the value map.
        """

        return value_map


class GeometricDataXType(GeometricDynamicDataType):
    """
    Data container for X values
    """

    _DYNAMIC_IMPLEMENTATION_ID = UUID("{2dbf303e-05d6-44ba-9692-39474e88d516}")
    _TYPE_UID = UUID(fields=(0xE9E6B408, 0x4109, 0x4E42, 0xB6, 0xA8, 0x685C37A802EE))


class GeometricDataYType(GeometricDynamicDataType):
    """
    Data container for Y values
    """

    _DYNAMIC_IMPLEMENTATION_ID = UUID("{d56406dc-5eeb-418d-add4-a1282a6ef668}")
    _TYPE_UID = UUID(fields=(0xF55B07BD, 0xD8A0, 0x4DFF, 0xBA, 0xE5, 0xC975D490D71C))


class GeometricDataZType(GeometricDynamicDataType):
    """
    Data container for X values
    """

    _DYNAMIC_IMPLEMENTATION_ID = UUID("{9dacdc3b-6878-408d-93ae-e9a95e640f0c}")
    _TYPE_UID = UUID(fields=(0xDBAFB885, 0x1531, 0x410C, 0xB1, 0x8E, 0x6AC9A40B4466))


DYNAMIC_CLASS_IDS = {
    cls._DYNAMIC_IMPLEMENTATION_ID: cls  # pylint: disable=protected-access
    for cls in GeometricDynamicDataType.__subclasses__()
}
