from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional, Type, cast

from geoh5io.shared import EntityType

from .color_map import ColorMap
from .data_unit import DataUnit
from .geometric_data_constants import GeometricDataConstants
from .primitive_type_enum import PrimitiveTypeEnum

if TYPE_CHECKING:
    from geoh5io import workspace
    from . import data  # noqa: F401


class DataType(EntityType):

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

    _primitive_type: Optional[PrimitiveTypeEnum] = None
    # TODO : add Property and Setters
    _color_map: Optional[ColorMap] = None
    _units = None
    _number_of_bins = 50
    _transparent_no_data = True
    _mapping = "equal_area"
    _hidden = False

    def __init__(self, workspace: "workspace.Workspace", **kwargs):
        assert workspace is not None
        super().__init__(workspace, **kwargs)

        workspace._register_type(self)

    @staticmethod
    def _is_abstract() -> bool:
        return False

    @property
    def units(self) -> Optional[DataUnit]:
        return self._units

    @units.setter
    def units(self, unit_name):
        assert isinstance(unit_name, str), f"Units must be of type {str}"
        self._units = DataUnit(unit_name)

    @property
    def primitive_type(self) -> Optional[PrimitiveTypeEnum]:
        return self._primitive_type

    @primitive_type.setter
    def primitive_type(self, value):
        if isinstance(value, str):
            self._primitive_type = getattr(PrimitiveTypeEnum, value.upper())
        else:
            assert isinstance(
                value, PrimitiveTypeEnum
            ), f"Primitive type value must be of type {PrimitiveTypeEnum}"
            self._primitive_type = value

    @classmethod
    def create(
        cls, workspace: "workspace.Workspace", data_class: Type["data.Data"]
    ) -> DataType:
        """ Creates a new instance of DataType with the primitive type from the given Data
        implementation class.

        :param data_class: A Data implementation class.
        :return: A new instance of DataType.
        """
        uid = uuid.uuid4()
        primitive_type = data_class.primitive_type()
        return cls(workspace, uid=uid, primitive_type=primitive_type)

    @classmethod
    def find_or_create(cls, workspace: "workspace.Workspace", **kwargs) -> DataType:
        """ Find or creates an EntityType with given UUID that matches the given
        Group implementation class.

        It is expected to have a single instance of EntityType in the Workspace
        for each concrete Entity class.

        :param workspace: An active Workspace class

        :return: A new instance of DataType.
        """
        uid = uuid.uuid4()

        for key, val in kwargs.items():
            if key.lower() in ["id", "uid"]:
                uid = uuid.UUID(val)

        entity_type = cls.find(workspace, uid)
        if entity_type is not None:
            return entity_type

        return cls(workspace, **kwargs)

    @classmethod
    def _for_geometric_data(
        cls, workspace: "workspace.Workspace", uid: uuid.UUID
    ) -> DataType:
        geom_primitive_type = GeometricDataConstants.primitive_type()
        data_type = cast(DataType, workspace.find_type(uid, DataType))
        if data_type is not None:
            assert data_type.primitive_type == geom_primitive_type
            return data_type
        return cls(workspace, uid=uid, primitive_type=geom_primitive_type)

    @classmethod
    def for_x_data(cls, workspace: "workspace.Workspace") -> DataType:
        return cls._for_geometric_data(
            workspace, GeometricDataConstants.x_datatype_uid()
        )

    @classmethod
    def for_y_data(cls, workspace: "workspace.Workspace") -> DataType:
        return cls._for_geometric_data(
            workspace, GeometricDataConstants.y_datatype_uid()
        )

    @classmethod
    def for_z_data(cls, workspace: "workspace.Workspace") -> DataType:
        return cls._for_geometric_data(
            workspace, GeometricDataConstants.z_datatype_uid()
        )
