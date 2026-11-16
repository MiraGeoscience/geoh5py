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
# pylint: disable=arguments-differ

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from geoh5py import Workspace
    from geoh5py.groups import PropertyGroup
    from geoh5py.shared.entity import Entity


class Geoh5FileClosedError(ABC, Exception):
    """Error for closed geoh5 file."""


class BaseValidationError(ABC, Exception):
    """
    Base class for custom exceptions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(self.message(*args, **kwargs))

    @classmethod
    @abstractmethod
    def message(cls, *args, **kwargs) -> str:
        """
        Builds custom error message.

        Each subclass of BaseValidationError should implement the message class method.
        """


class ShapeValidationError(BaseValidationError):
    """Error on shape validation."""

    @classmethod
    def message(
        cls, name: str, value: tuple[int, ...], validation: tuple[int, ...] | str
    ) -> str:
        return (
            f"Parameter '{name}' with shape {value} was provided. "
            f"Expected {validation}."
        )


class MemoryValidationError(BaseValidationError):
    """Error on memory validation."""

    @classmethod
    def message(cls, name: str, value: Any, validation: float) -> str:
        return (
            f"Parameter '{name}' with value '{value}' "
            f"exceeds memory limit of {validation / 1e6} MB."
        )
