from __future__ import annotations

import uuid
from typing import Optional
from typing import Type

from geoh5io.data import ColorMap
from geoh5io.data import Data
from geoh5io.data import PrimitiveTypeEnum
from geoh5io.shared import Entity
from geoh5io.shared import EntityType


class DataType(EntityType):
    def __init__(self, uid: uuid.UUID, primitive_type: PrimitiveTypeEnum):
        super().__init__(uid)
        self._primitive_type = primitive_type
        # TODO: define properties and setters
        self._color_map: Optional[ColorMap] = None
        self._units = None

    @classmethod
    def _create(cls, entity_class: Type[Entity]) -> DataType:
        """ See method ``create()`` """
        assert issubclass(entity_class, Data)
        uid = uuid.uuid4()
        primitive_type = entity_class.primitive_type()
        return DataType(uid, primitive_type)

    @classmethod
    def create(cls, data_class: Type[Data]) -> DataType:
        """ Creates a new instance of GroupType with the primitive type from the given Data
        implementation class.

        :param data_class: A Data implementation class.
        :return: A new instance of DataType.
        """
        return cls._create(data_class)
