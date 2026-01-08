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
from typing import Annotated, Any
from uuid import UUID

from pydantic import BeforeValidator, Field, PlainSerializer

from geoh5py.ui_json.validations.form import empty_string_to_none, uuid_to_string


logger = logging.getLogger(__name__)

OptionalUUIDList = Annotated[
    list[UUID] | None,  # pylint: disable=unsupported-binary-operation
    BeforeValidator(empty_string_to_none),
    PlainSerializer(uuid_to_string),
]

OptionalValueList = Annotated[
    float | list[float] | None,
    BeforeValidator(empty_string_to_none),
]


def deprecate(value, info):
    """Issue deprecation warning."""
    logger.warning("Skipping deprecated field: %s.", info.field_name)
    return value


Deprecated = Annotated[
    Any,
    Field(exclude=True),
    BeforeValidator(deprecate),
]
