from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional, cast

from geoh5io.shared import EntityType

from .color_map import ColorMap
from .data_unit import DataUnit
from .geometric_data_constants import GeometricDataConstants
from .primitive_type_enum import PrimitiveTypeEnum

if TYPE_CHECKING:
    from geoh5io import workspace
    from . import data  # noqa: F401


class DataType(EntityType):
    def __init__(
        self,
        workspace: "workspace.Workspace",
        uid: uuid.UUID,
        primitive_type: PrimitiveTypeEnum,
    ):
        super().__init__(workspace, uid)
        self.__primitive_type = primitive_type
        # TODO: define properties and setters
        self._color_map: Optional[ColorMap] = None
        self._units = None
        self._number_of_bins = 50
        self._transparent_no_data = True
        self._mapping = "equal_area"
        self._hidden = False

    @staticmethod
    def _is_abstract() -> bool:
        return False

    @property
    def units(self) -> Optional[DataUnit]:
        return self._units

    # @units.setter
    # def units(self, unit_name):
    #     if self._units is None:
    #
    #         assert isinstance(
    #             unit_name, str
    #         ), f"Units must be of type {str}"
    #         self._units = DataUnit(unit_name)

    @property
    def primitive_type(self) -> PrimitiveTypeEnum:
        return self.__primitive_type

    # @classmethod
    # def create(
    #     cls, workspace: "workspace.Workspace", data_class: Type["data.Data"]
    # ) -> DataType:
    #     """ Creates a new instance of DataType with the primitive type from the given Data
    #     implementation class.
    #
    #     :param data_class: A Data implementation class.
    #     :return: A new instance of DataType.
    #     """
    #     uid = uuid.uuid4()
    #     primitive_type = data_class.primitive_type()
    #     return cls(workspace, uid, primitive_type)

    @classmethod
    def find_or_create(
        cls, workspace: "workspace.Workspace", type_uid: uuid.UUID, primitive_type
    ) -> DataType:
        """ Find or creates the DataType with the pre-defined type UUID that matches the given
        Data implementation class.


        :param data_class: An Data implementation class.
        :return: A new instance of DataType.
        """
        object_type = cls.find(workspace, type_uid)
        if object_type is not None:
            return object_type

        uid = uuid.uuid4()
        # primitive_type = data_class.primitive_type()
        return cls(workspace, uid, primitive_type)

    @classmethod
    def _for_geometric_data(
        cls, workspace: "workspace.Workspace", uid: uuid.UUID
    ) -> DataType:
        geom_primitive_type = GeometricDataConstants.primitive_type()
        data_type = cast(DataType, workspace.find_type(uid, DataType))
        if data_type is not None:
            assert data_type.primitive_type == geom_primitive_type
            return data_type
        return cls(workspace, uid, geom_primitive_type)

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
