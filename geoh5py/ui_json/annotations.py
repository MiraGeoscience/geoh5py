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

import logging
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

from pydantic import BeforeValidator, Field, PlainSerializer

from geoh5py.data import DataAssociationEnum, DataTypeEnum
from geoh5py.groups import Group
from geoh5py.objects import ObjectBase
from geoh5py.shared.utils import (
    enum_name_to_str,
    none2str,
    str2none,
    stringify,
    workspace2path,
)
from geoh5py.shared.validators import (
    to_class,
    to_list,
    to_path,
    to_type_uid_or_class,
)
from geoh5py.ui_json.utils import optional_uuid_mapper


logger = logging.getLogger(__name__)


def deprecate(value, info):
    """Issue deprecation warning."""
    logger.warning("Skipping deprecated field: %s.", info.field_name)
    return value


Deprecated = Annotated[
    Any,
    Field(exclude=True),
    BeforeValidator(deprecate),
]

AssociationOptions = Annotated[
    DataAssociationEnum,
    PlainSerializer(enum_name_to_str, when_used="json"),
]

DataTypeOptions = Annotated[
    DataTypeEnum,
    PlainSerializer(enum_name_to_str, when_used="json"),
]

GroupTypes = Annotated[
    list[type[Group]],
    BeforeValidator(to_class),
    BeforeValidator(to_type_uid_or_class),
    BeforeValidator(to_list),
    PlainSerializer(stringify, when_used="json"),
]

MeshTypes = Annotated[
    list[type[ObjectBase]],
    BeforeValidator(to_class),
    BeforeValidator(to_type_uid_or_class),
    BeforeValidator(to_list),
    PlainSerializer(stringify, when_used="json"),
]

OptionalPath = Annotated[
    Path | None,
    BeforeValidator(str2none),
    BeforeValidator(workspace2path),
    PlainSerializer(none2str, when_used="json"),
]

OptionalString = Annotated[
    str | None,
    BeforeValidator(str2none),
    PlainSerializer(none2str, when_used="json"),
]

OptionalUUID = Annotated[
    UUID | None,
    BeforeValidator(optional_uuid_mapper),
    PlainSerializer(stringify, when_used="json"),
]

OptionalUUIDList = Annotated[
    list[UUID] | UUID | None,
    BeforeValidator(optional_uuid_mapper),
    PlainSerializer(stringify, when_used="json"),
]

OptionalValueList = Annotated[
    float | list[float] | None,
    BeforeValidator(optional_uuid_mapper),
]

PathList = Annotated[
    list[Path],
    BeforeValidator(to_path),
    BeforeValidator(to_list),
    PlainSerializer(stringify, when_used="json"),
]
