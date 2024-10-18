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

from ..shared import EntityType, utils
from .color_map import ColorMap
from .primitive_type_enum import PrimitiveTypeEnum
from .reference_value_map import BOOLEAN_VALUE_MAP, ReferenceValueMap


if TYPE_CHECKING:  # pragma: no cover
    from ..objects import ObjectBase
    from ..workspace import Workspace
    from .data import Data
    from .referenced_data import ReferencedData

ColorMapping = Literal[
    "linear",
    "equal_area",
    "logarithmic",
    "cdf",
    "cumulative_distribution_function",
    "missing",
]


class DataType(EntityType):
    """
    DataType class.

    Controls all the attributes of the data type for displays in Geoscience ANALYST.

    :param workspace: An active Workspace.
    :param primitive_type: The primitive type of the data.
    :param color_map: The colormap used for plotting.
    :param duplicate_on_copy: Force a copy on copy of the data entity.
    :param duplicate_type_on_copy: Force a copy on copy of the data entity.
    :param hidden: If the data are hidden or not.
    :param mapping: The type of color stretching to plot the colormap.
    :param number_of_bins: The number of bins used by the histogram.
    :param precision: The decimals precision of the data to display.
    :param scale: The type of scale of the data.
    :param scientific_notation: If the data should be displayed in scientific notation.
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
            "Precision": "precision",
            "Primitive type": "primitive_type",
            "Transparent no data": "transparent_no_data",
            "Scale": "scale",
            "Scientific notation": "scientific_notation",
        }
    )

    def __init__(
        self,
        workspace: Workspace,
        *,
        primitive_type: (
            type[Data] | PrimitiveTypeEnum | str
        ) = PrimitiveTypeEnum.INVALID,
        color_map: ColorMap | None = None,
        duplicate_on_copy: bool = False,
        duplicate_type_on_copy: bool = False,
        hidden: bool = False,
        mapping: ColorMapping = "equal_area",
        number_of_bins: int | None = None,
        precision: int = 2,
        scale: str | None = None,
        scientific_notation: bool = False,
        transparent_no_data: bool = True,
        units: str | None = None,
        **kwargs,
    ):
        super().__init__(workspace, **kwargs)

        self.color_map = color_map
        self.duplicate_on_copy = duplicate_on_copy
        self.duplicate_type_on_copy = duplicate_type_on_copy
        self.hidden = hidden
        self.mapping = mapping
        self.number_of_bins = number_of_bins
        self.precision = precision
        self.primitive_type = self.validate_primitive_type(primitive_type)
        self.scale = scale
        self.scientific_notation = scientific_notation
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
                f"Attribute 'color_map' must be of type {ColorMap},"
                f"numpy.ndarray or dict with 'values'."
            )

        if isinstance(color_map, ColorMap):
            color_map.parent = self

        self._color_map: ColorMap | None = color_map

        self.workspace.update_attribute(self, "color_map")

    @property
    def duplicate_on_copy(self) -> bool:
        """
        If the data type should be duplicated on copy.
        """
        return self._duplicate_on_copy

    @duplicate_on_copy.setter
    def duplicate_on_copy(self, value: bool):
        if not isinstance(value, bool) and value not in [1, 0]:
            raise TypeError(
                f"Attribute 'duplicate_on_copy' must be a bool, not {type(value)}"
            )

        self._duplicate_on_copy = bool(value)
        if self.on_file:
            self.workspace.update_attribute(self, "attributes")

    @property
    def duplicate_type_on_copy(self) -> bool:
        """
        If the data type should be duplicated on copy.
        """
        return self._duplicate_type_on_copy

    @duplicate_type_on_copy.setter
    def duplicate_type_on_copy(self, value: bool):
        if not isinstance(value, bool) and value != 1 and value != 0:
            raise TypeError(
                f"Attribute 'duplicate_type_on copy' must be a bool, not {type(value)}"
            )

        self._duplicate_type_on_copy = bool(value)
        self.workspace.update_attribute(self, "attributes")

    @classmethod
    def find_or_create_type(
        cls,
        workspace: Workspace,
        primitive_type: PrimitiveTypeEnum | str,
        dynamic_implementation_id: str | UUID | None = None,
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

        if primitive_type == PrimitiveTypeEnum.BOOLEAN:
            return ReferencedBooleanType(
                workspace, primitive_type=primitive_type, uid=uid, **kwargs
            )

        if (
            primitive_type == PrimitiveTypeEnum.GEOMETRIC
            and dynamic_implementation_id is not None
        ):
            data_type = DYNAMIC_CLASS_IDS.get(
                utils.str2uuid(dynamic_implementation_id), DataType
            )

            return data_type(
                workspace, primitive_type=primitive_type, uid=uid, **kwargs
            )

        if primitive_type == PrimitiveTypeEnum.REFERENCED:
            return ReferencedValueMapType(
                workspace, primitive_type=primitive_type, uid=uid, **kwargs
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
            raise TypeError(f"Attribute 'hidden' must be a bool, not {type(value)}")

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
                f"Attribute 'mapping' should be one of {get_args(ColorMapping)}. "
                f"Value '{value}' was provided."
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
                "Attribute 'number_of_bins' should be an integer greater than 0 "
                f"or None, not {n_bins}"
            )

        self._number_of_bins: int | None = n_bins

        self.workspace.update_attribute(self, "attributes")

    @property
    def precision(self) -> int:
        """
        The decimals precision of the data to display.
        """
        return self._precision

    @precision.setter
    def precision(self, value: int):
        if (
            not isinstance(value, (int, float, np.integer, np.floating))
            or (isinstance(value, (float, np.floating)) and not value.is_integer())
            or value < 0
        ):
            raise TypeError(
                f"Attribute 'precision' must be an integer greater than 0, not {value}"
            )

        self._precision = int(value)

        self.workspace.update_attribute(self, "attributes")

    @property
    def primitive_type(self) -> PrimitiveTypeEnum:
        """
        The primitive type of the data.
        """
        return self._primitive_type

    @primitive_type.setter
    def primitive_type(self, value: PrimitiveTypeEnum):
        if not isinstance(value, PrimitiveTypeEnum):
            raise ValueError(
                "Attribute 'primitive_type' value must be of type "
                f"{PrimitiveTypeEnum}, find {type(value)}"
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
            raise TypeError(
                f"Attribute 'transparent_no_data' must be a bool, not {type(value)}"
            )
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
            raise TypeError(f"Attribute 'units' must be a string, not {type(unit)}")
        self._units = unit

        self.workspace.update_attribute(self, "attributes")

    @staticmethod
    def primitive_type_from_values(values: np.ndarray | None) -> PrimitiveTypeEnum:
        """
        Validate the primitive type of the data.

        :param values: The values to validate.

        :return: The equivalent primitive type of the data.
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

    @property
    def scale(self) -> str | None:
        """
        The type of scale of the data.
        """
        return self._scale

    @scale.setter
    def scale(self, value: str | None):
        if value not in ["Linear", "Log", None]:
            raise ValueError(
                f"Attribute 'scale' must be one of 'Linear', 'Log', NoneType, not {value}"
            )
        self._scale = value
        self.workspace.update_attribute(self, "attributes")

    @property
    def scientific_notation(self) -> bool:
        """
        If the data should be displayed in scientific notation.
        """
        return self._scientific_notation

    @scientific_notation.setter
    def scientific_notation(self, value: bool):
        if value not in [True, False, 1, 0]:
            raise TypeError(
                f"Attribute 'scientific_notation' must be a bool, not {type(value)}"
            )

        self._scientific_notation = bool(value)
        self.workspace.update_attribute(self, "attributes")

    @staticmethod
    def validate_primitive_type(
        primitive_type: PrimitiveTypeEnum | str | type[Data],
    ) -> PrimitiveTypeEnum:
        """
        Validate the primitive type of the data.

        :param primitive_type: Some reference to the primitive type of the data.

        :return: A known primitive type.
        """
        if isinstance(primitive_type, str):
            primitive_type = getattr(
                PrimitiveTypeEnum, utils.INV_KEY_MAP.get(primitive_type, primitive_type)
            )

        if isinstance(primitive_type, type) and hasattr(
            primitive_type, "primitive_type"
        ):
            primitive_type = primitive_type.primitive_type()

        if not isinstance(primitive_type, PrimitiveTypeEnum):
            raise ValueError(
                f"Attribute 'primitive_type' should be one of {PrimitiveTypeEnum.__members__}"
            )
        return primitive_type


class ReferenceDataType(DataType):
    """
    DataType class.

    Controls all the attributes of reference data.

    :param value_map: Reference value to map index with description.
    """

    def __init__(
        self,
        workspace: Workspace,
        value_map: dict[int, str] | np.ndarray | tuple | ReferenceValueMap = (
            (0, "Unknown"),
        ),
        **kwargs,
    ):
        super().__init__(workspace, **kwargs)
        self.value_map = self.validate_value_map(value_map)

    @staticmethod
    @abstractmethod
    def validate_keys(value_map: ReferenceValueMap) -> ReferenceValueMap:
        """
        Validate the keys of the value map.
        """

    def validate_value_map(
        self,
        value_map: dict[int, str] | np.ndarray | tuple | ReferenceValueMap,
    ) -> ReferenceValueMap | None:
        """
        Validate the attribute of ReferencedDataType
        """

        if value_map is None:
            return None

        if isinstance(value_map, dict | np.ndarray | tuple):
            value_map = ReferenceValueMap(value_map)

        if not isinstance(value_map, ReferenceValueMap):
            raise TypeError(
                "Attribute 'value_map' must be provided as a dict, tuple[dict], "
                f"numpy.ndarray or {ReferenceValueMap}."
            )

        self.validate_keys(value_map)

        return value_map

    @property
    def value_map(self) -> ReferenceValueMap | None:
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
    def value_map(
        self, value_map: dict[int, str] | np.ndarray | tuple | ReferenceValueMap
    ):
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
        **kwargs,
    ):
        super().__init__(workspace, **kwargs)

    @staticmethod
    def validate_keys(value_map: ReferenceValueMap):
        """
        Validate the keys of the value map.
        """
        if 0 not in value_map.map["Key"]:
            value_map.map.resize(len(value_map) + 1, refcheck=False)
            value_map.map[-1] = (0, b"Unknown")

        if dict(value_map.map)[0] not in ["Unknown", b"Unknown"]:
            raise ValueError("Value for key 0 must be b'Unknown'")


class ReferencedBooleanType(ReferenceDataType):
    """
    Data container for referenced boolean data.
    """

    def __init__(
        self,
        workspace: Workspace,
        value_map: (
            dict[int, str] | np.ndarray | tuple | ReferenceValueMap
        ) = BOOLEAN_VALUE_MAP,
        **kwargs,
    ):
        super().__init__(workspace, value_map=value_map, **kwargs)

    @staticmethod
    def validate_keys(value_map: ReferenceValueMap):
        """
        Validate the keys of the value map.
        """
        if not np.all(value_map.map == BOOLEAN_VALUE_MAP):
            raise ValueError("Boolean value map must be (0: 'False', 1: 'True'")


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
        *,
        value_map: dict[int, str] | tuple | ReferenceValueMap | None = None,
        parent: ObjectBase | None = None,
        description: str = "Dynamic referenced data",
        primitive_type: PrimitiveTypeEnum | str = PrimitiveTypeEnum.GEOMETRIC,
        **kwargs,
    ):
        self._referenced_data = None

        super().__init__(
            workspace,
            value_map=value_map,
            description=description,
            primitive_type=primitive_type,
            **kwargs,
        )
        self._parent = parent

    def get_parent_reference(self, parent: ObjectBase):
        """
        Recover the parent ReferencedData by name.
        """
        ref_data_name = self.name.rsplit(":")[0]

        ref_data = []
        for child in parent.children:
            if (
                isinstance(child.entity_type, ReferencedValueMapType)
                and child.entity_type.name == ref_data_name
            ):
                ref_data.append(child)

        if len(ref_data) == 0:
            raise ValueError(f"Parent data '{ref_data_name}' not found.")

        return ref_data[0]

    @property
    def referenced_data(self) -> ReferencedData | None:
        """
        Reference data type holding the value map.
        """
        if self._referenced_data is None and self._parent is not None:
            self._referenced_data = self.get_parent_reference(self._parent)

        return self._referenced_data

    @staticmethod
    def validate_keys(value_map: ReferenceValueMap):
        """
        Validate the keys of the value map.
        """

    def validate_value_map(
        self,
        value_map: dict[int, str] | np.ndarray | tuple | ReferenceValueMap,
    ) -> ReferenceValueMap | None:
        """
        Validate the attribute of ReferencedDataType
        """

        if value_map is None:
            return None

        if isinstance(value_map, dict | np.ndarray | tuple):
            value_map = ReferenceValueMap(value_map, name=self.name.rsplit(": ")[1])

        if not isinstance(value_map, ReferenceValueMap):
            raise TypeError(
                "Attribute 'value_map' must be provided as a dict, tuple[dict] "
                f"or {ReferenceValueMap}."
            )

        return value_map

    @property
    def value_map(self) -> ReferenceValueMap | None:
        r"""
        Reference value to map index with description.

        The value_map can be set from a dictionary of sorted integer
        values with text description.

        .. code-block:: python

            value_map = {
                val_1: str_1,
                ...,
                val_i: str_i
            }

        """
        if self._value_map is None and self.referenced_data is not None:
            if (
                self.referenced_data.data_maps is None
                or self.referenced_data.metadata is None
            ):
                raise ValueError("Referenced data has no data maps.")

            value_map = None
            for count, name in enumerate(self.referenced_data.metadata):
                if name == self.name.rsplit(": ")[1]:
                    value_map = self.workspace.fetch_array_attribute(
                        self.referenced_data.entity_type, f"Value map {count + 1}"
                    )

            if value_map is not None:
                self._value_map = self.validate_value_map(
                    value_map.astype(ReferenceValueMap.MAP_DTYPE)
                )

        return self._value_map

    @value_map.setter
    def value_map(self, value_map: dict | tuple | ReferenceValueMap | None):
        self._value_map = self.validate_value_map(value_map)

        if self.on_file and self.referenced_data is not None:
            self.workspace.update_attribute(self.referenced_data, "data_map")


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
    Data container for Z values
    """

    _DYNAMIC_IMPLEMENTATION_ID = UUID("{9dacdc3b-6878-408d-93ae-e9a95e640f0c}")
    _TYPE_UID = UUID(fields=(0xDBAFB885, 0x1531, 0x410C, 0xB1, 0x8E, 0x6AC9A40B4466))


DYNAMIC_CLASS_IDS = {
    cls._DYNAMIC_IMPLEMENTATION_ID: cls  # pylint: disable=protected-access
    for cls in GeometricDynamicDataType.__subclasses__()
}
