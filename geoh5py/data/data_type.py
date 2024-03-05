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

import uuid
from typing import TYPE_CHECKING, Literal, cast, get_args

import numpy as np

from ..shared import EntityType
from .color_map import ColorMap
from .geometric_data_constants import GeometricDataConstants
from .primitive_type_enum import PrimitiveTypeEnum
from .reference_value_map import ReferenceValueMap

if TYPE_CHECKING:
    from ..workspace import Workspace
    from .data import Data  # noqa: F401


ColorMapping = Literal["linear", "equal_area", "logarithmic", "cdf", "missing"]


class DataType(EntityType):
    # pylint: disable=too-many-arguments
    """
    DataType class.

    Controls all the attributes of the data type for displays in Geoscience ANALYST.

    :param workspace: An active Workspace.
    :param primitive_type: The primitive type of the data.
    :param color_map: The colormap used for plotting.
    :param hidden: If the data are hidden or not.
    :param mapping: The type of color stretching to plot the colormap.
    :param number_of_bins: The number of bins used by the histogram.
    :param transparent_no_data: If the no data values are displayed as transparent or not.
    :param units: The type of the units of the data.
    :param value_map: Reference value map for to map index with description.
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
        primitive_type: type[Data] | PrimitiveTypeEnum | str | None = None,
        color_map: ColorMap | None = None,
        hidden: bool = False,
        mapping: ColorMapping = "equal_area",
        number_of_bins: int | None = None,
        transparent_no_data: bool = True,
        units: str | None = None,
        value_map: dict[int, str] | ReferenceValueMap | None = None,
        **kwargs,
    ):
        super().__init__(workspace, **kwargs)
        self.color_map = color_map
        self.hidden = hidden
        self.mapping = mapping
        self.number_of_bins = number_of_bins
        self.primitive_type = primitive_type  # type: ignore
        self.transparent_no_data = transparent_no_data
        self.units = units
        self.value_map = value_map  # type: ignore

    @classmethod
    def _for_geometric_data(cls, workspace: Workspace, uid: uuid.UUID) -> DataType:
        """
        Get the data type for geometric data.

        :param workspace: An active Workspace.
        :param uid: The uid of the existing data type to get.

        :return: A new instance of DataType.
        """
        geom_primitive_type = GeometricDataConstants.primitive_type()
        data_type = cast(DataType, workspace.find_type(uid, DataType))
        if data_type is not None:
            if not data_type.primitive_type == geom_primitive_type:
                raise ValueError(
                    f"Data type with uid {uid} is not of type {geom_primitive_type}"
                )
            return data_type
        return cls(workspace, uid=uid, primitive_type=geom_primitive_type)

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

    @classmethod
    def create(
        cls, workspace: Workspace, primitive_type: type[Data], **kwargs
    ) -> DataType:
        """
        Creates a new instance of :obj:`~geoh5py.data.data_type.DataType` with
        corresponding :obj:`~geoh5py.data.primitive_type_enum.PrimitiveTypeEnum`.

        :param workspace: An active Workspace.
        :param primitive_type: A :obj:`~geoh5py.data.data.Data` implementation class.
        :param kwargs: Keyword arguments to initialize the new DataType.

        :return: A new instance of :obj:`~geoh5py.data.data_type.DataType`.
        """
        # todo: this could be to deprecate.
        return cls(workspace, primitive_type=primitive_type, **kwargs)

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
        elif not isinstance(n_bins, int) or n_bins < 1:
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

    @staticmethod
    def validate_data_type(workspace: Workspace, attribute_dict: dict):
        """
        Get a dictionary of attributes and validate the type of data.

        :param workspace: An active Workspace.
        :param attribute_dict: A dictionary of attributes of the new Datatype to create.

        :return: A new instance of DataType.
        """

        entity_type = attribute_dict.get("entity_type")
        if entity_type is None:
            primitive_type = attribute_dict.get("type")
            if primitive_type is not None:
                assert (
                    primitive_type.upper() in PrimitiveTypeEnum.__members__
                ), f"Data 'type' should be one of {PrimitiveTypeEnum.__members__}"
                entity_type = {"primitive_type": primitive_type.upper()}
            else:
                values = attribute_dict.get("values")
                if values is None or (
                    isinstance(values, np.ndarray)
                    and (values.dtype in [np.float32, np.float64])
                ):
                    entity_type = {"primitive_type": "FLOAT"}
                elif isinstance(values, np.ndarray) and (
                    values.dtype in [np.uint32, np.int32]
                ):
                    entity_type = {"primitive_type": "INTEGER"}
                elif isinstance(values, str) or (
                    isinstance(values, np.ndarray) and values.dtype.kind in ["U", "S"]
                ):
                    entity_type = {"primitive_type": "TEXT"}
                elif isinstance(values, np.ndarray) and (values.dtype == bool):
                    entity_type = {"primitive_type": "BOOLEAN"}
                else:
                    raise NotImplementedError(
                        "Only add_data values of type FLOAT, INTEGER,"
                        "BOOLEAN and TEXT have been implemented"
                    )
        elif isinstance(entity_type, EntityType) and (
            (entity_type.uid not in getattr(workspace, "_types"))
            or (entity_type.workspace != workspace)
        ):
            return entity_type.copy(workspace=workspace)

        return entity_type

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
    def value_map(self, value_map: dict[int, str] | ReferenceValueMap | None):
        if isinstance(value_map, dict):
            value_map = ReferenceValueMap(value_map)
        if not isinstance(value_map, (ReferenceValueMap, type(None))):
            raise TypeError(
                f"'value_map' must be a {dict} or {ReferenceValueMap} or {type(None)}."
            )

        self._value_map: ReferenceValueMap | None = value_map

        self.workspace.update_attribute(self, "value_map")
